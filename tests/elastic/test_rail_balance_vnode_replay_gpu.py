"""C070: replay the owning snapshot produced by the C061 vnode round trip.

The replay call has no planner manifest or fingerprint array.  This test first
captures a real C061 2x4 run, clones the persisted protocol state into ordinary
CUDA tensors, poisons the planner-only inputs, and then checks the four-stage
replay against the independent CPU snapshot oracle.
"""

from __future__ import annotations

import argparse
import os
import struct
from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import align
import test_rail_balance_vnode_multidst as c061
import test_rail_balance_vnode_replay as replay_cpu


@dataclass(frozen=True)
class OwningReplayInputs:
    topk_idx: torch.Tensor
    quota: torch.Tensor
    rail_base_records: torch.Tensor
    rail_base_ready: torch.Tensor
    expert_records: torch.Tensor
    expert_ready: torch.Tensor
    expert_routes: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.topk_idx,
            self.quota,
            self.rail_base_records,
            self.rail_base_ready,
            self.expert_records,
            self.expert_ready,
            self.expert_routes,
        )


def _clone_replay_inputs(
    topk_idx: torch.Tensor,
    quota: torch.Tensor,
    roundtrip_outputs: Sequence[torch.Tensor],
) -> OwningReplayInputs:
    assert len(roundtrip_outputs) == 12
    rail_records = roundtrip_outputs[4]
    rail_ready = roundtrip_outputs[5]
    expert_records = roundtrip_outputs[7]
    expert_ready = roundtrip_outputs[8]
    expert_routes = roundtrip_outputs[9]
    result = OwningReplayInputs(
        topk_idx=topk_idx.clone(),
        quota=quota.clone(),
        rail_base_records=rail_records[:c061._BASE_CAPACITY].clone(),
        rail_base_ready=rail_ready[:c061._BASE_CAPACITY].clone(),
        expert_records=expert_records.clone(),
        expert_ready=expert_ready.clone(),
        expert_routes=expert_routes.clone(),
    )
    pointers = []
    for tensor in result.tensors():
        assert tensor.is_cuda and tensor.is_contiguous()
        assert tensor._base is None
        pointers.append(tensor.data_ptr())
    assert len(pointers) == len(set(pointers))
    return result


def _snapshot_input_bytes(
    inputs: OwningReplayInputs,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor.contiguous().view(torch.uint8).clone()
        for tensor in inputs.tensors()
    )


def _assert_input_bytes_unchanged(
    inputs: OwningReplayInputs,
    expected: Sequence[torch.Tensor],
) -> None:
    assert len(inputs.tensors()) == len(expected)
    for tensor, expected_bytes in zip(inputs.tensors(), expected):
        assert torch.equal(
            tensor.contiguous().view(torch.uint8), expected_bytes)


def _call_replay(
    buffer: deep_ep.ElasticBuffer,
    inputs: OwningReplayInputs,
    hidden: int,
    arena_offset: int,
) -> tuple[torch.Tensor, ...]:
    return buffer.runtime._rail_balance_vnode_replay(
        inputs.topk_idx,
        inputs.quota,
        inputs.rail_base_records,
        inputs.rail_base_ready,
        inputs.expert_records,
        inputs.expert_ready,
        inputs.expert_routes,
        hidden,
        arena_offset,
        c061._BASE_CAPACITY,
        c061._DESTINATION_CAPACITY,
        c061._NUM_DESTINATIONS,
        c061._GENERATION,
        c061._NUM_TOKENS,
        c061._NUM_SOURCE_RANKS,
        c061._EXPERT_BEGIN,
        c061._EXPERTS_PER_RANK,
    )


def _gather_cpu_snapshot(
    rank: int,
    control_group: dist.ProcessGroup,
    outputs: Sequence[torch.Tensor],
    replay_topk_idx: torch.Tensor,
    quota_cpu: torch.Tensor,
    hidden: int,
) -> replay_cpu.ReplaySnapshot:
    if rank < c061._NUM_SOURCE_RANKS:
        payload = {
            # The public routing tensor is topk_idx_t (currently int64),
            # while TokenLayout persists each expert index as protocol int32.
            "topk": replay_topk_idx.to(
                device="cpu", dtype=torch.int32).clone(),
            "rail": None,
        }
    else:
        payload = {
            "topk": None,
            "rail": (
                outputs[4].cpu().clone(),
                outputs[6].cpu().clone(),
                outputs[5].cpu().clone(),
            ),
        }
    gathered: list[object | None] = [None] * c061._WORLD_SIZE
    dist.all_gather_object(gathered, payload, group=control_group)

    source_topk = []
    for source_rank in range(c061._NUM_SOURCE_RANKS):
        item = gathered[source_rank]
        assert isinstance(item, dict)
        topk = item["topk"]
        assert isinstance(topk, torch.Tensor)
        source_topk.append(topk)

    destination_records = []
    destination_routes = []
    destination_ready = []
    for destination_rank in range(
            c061._NUM_SOURCE_RANKS, c061._WORLD_SIZE):
        item = gathered[destination_rank]
        assert isinstance(item, dict)
        rail = item["rail"]
        assert isinstance(rail, tuple) and len(rail) == 3
        records, routes, ready = rail
        assert isinstance(records, torch.Tensor)
        assert isinstance(routes, torch.Tensor)
        assert isinstance(ready, torch.Tensor)
        destination_records.append(records)
        destination_routes.append(routes)
        destination_ready.append(ready)

    layout = replay_cpu._record_layout(hidden)
    assert layout.record_bytes == outputs[4].shape[1]
    return replay_cpu.ReplaySnapshot(
        layout=layout,
        records=torch.stack(destination_records),
        routes=torch.stack(destination_routes),
        ready=torch.stack(destination_ready),
        quota=quota_cpu.clone(),
        topk_idx=torch.stack(source_topk),
    )


def _verify_replay_outputs(
    rank: int,
    outputs: Sequence[torch.Tensor],
    inputs: OwningReplayInputs,
    oracle: replay_cpu.ReplayResult,
    independent_partials: Sequence[torch.Tensor],
    independent_outputs: Sequence[torch.Tensor],
    layout: c061.Layout,
    hidden: int,
) -> None:
    assert len(outputs) == 12
    (combined_output, output_ready, owner_partials, owner_partial_ready,
     rail_records, rail_ready, rail_routes, expert_records, expert_ready,
     expert_routes, stage_status, arena_guard) = outputs
    for tensor in outputs:
        assert tensor.is_cuda and tensor.device.index == rank
        assert tensor.is_contiguous()
    assert tuple(combined_output.shape) == (c061._NUM_TOKENS, hidden)
    assert tuple(output_ready.shape) == (c061._NUM_TOKENS,)
    assert tuple(owner_partials.shape) == (
        c061._NUM_TOKENS, c061._NUM_TOPK, hidden)
    assert tuple(owner_partial_ready.shape) == (
        c061._NUM_TOKENS, c061._NUM_TOPK)
    assert tuple(rail_records.shape) == (
        layout.rail_capacity, layout.record_bytes)
    assert tuple(rail_ready.shape) == (layout.rail_capacity,)
    assert tuple(rail_routes.shape) == (layout.rail_capacity, 32)
    assert tuple(expert_records.shape) == (
        layout.expert_capacity, layout.record_bytes)
    assert tuple(expert_ready.shape) == (layout.expert_capacity,)
    assert tuple(expert_routes.shape) == (layout.expert_capacity, 32)
    assert tuple(stage_status.shape) == (4, 72)
    assert tuple(arena_guard.shape) == (c061._ARENA_GUARD_BYTES,)

    cpu = tuple(tensor.cpu() for tensor in outputs)
    (combined_output, output_ready, owner_partials, owner_partial_ready,
     rail_records, rail_ready, rail_routes, expert_records, expert_ready,
     expert_routes, stage_status, arena_guard) = cpu
    assert torch.count_nonzero(stage_status).item() == 0
    assert torch.equal(
        arena_guard,
        torch.full(
            (c061._ARENA_GUARD_BYTES,), 0xA5,
            dtype=torch.uint8, device="cpu"),
    )
    assert torch.equal(
        rail_records[:c061._BASE_CAPACITY],
        inputs.rail_base_records.cpu(),
    )
    assert torch.equal(
        rail_ready[:c061._BASE_CAPACITY],
        inputs.rail_base_ready.cpu(),
    )

    if rank < c061._NUM_SOURCE_RANKS:
        expected_ready = torch.zeros(
            c061._NUM_TOKENS, dtype=torch.int32, device="cpu")
        expected_ready[:c061._NUM_TOKENS - 1] = c061._GENERATION
        expected_partial_ready = torch.zeros(
            (c061._NUM_TOKENS, c061._NUM_TOPK),
            dtype=torch.int32, device="cpu")
        expected_partial_ready[:c061._NUM_TOKENS - 1] = c061._GENERATION
        assert torch.equal(output_ready, expected_ready)
        assert torch.equal(owner_partial_ready, expected_partial_ready)
        assert torch.equal(owner_partials, oracle.owner_partials[rank])
        assert torch.equal(combined_output, oracle.combined_output[rank])
        # The snapshot oracle above intentionally derives routing from replay
        # bytes. Also compare against C061's independent expert formula so a
        # consistently wrong GPU contribution cannot validate itself.
        assert torch.equal(owner_partials, independent_partials[rank])
        assert torch.equal(combined_output, independent_outputs[rank])
        assert torch.count_nonzero(expert_records).item() == 0
        assert torch.count_nonzero(expert_ready).item() == 0
        assert torch.count_nonzero(expert_routes).item() == 0
    else:
        assert torch.count_nonzero(combined_output).item() == 0
        assert torch.count_nonzero(output_ready).item() == 0
        assert torch.count_nonzero(owner_partials).item() == 0
        assert torch.count_nonzero(owner_partial_ready).item() == 0
        assert torch.equal(expert_records, inputs.expert_records.cpu())
        assert torch.equal(expert_ready, inputs.expert_ready.cpu())
        assert torch.equal(expert_routes, inputs.expert_routes.cpu())


def _assert_outputs_identical(
    first: Sequence[torch.Tensor],
    second: Sequence[torch.Tensor],
) -> None:
    assert len(first) == len(second) == 12
    for expected, observed in zip(first, second):
        assert expected.dtype == observed.dtype
        assert tuple(expected.shape) == tuple(observed.shape)
        assert torch.equal(
            expected.contiguous().view(torch.uint8),
            observed.contiguous().view(torch.uint8),
        )


def _clone_with_one_corrupt_route(
    rank: int,
    inputs: OwningReplayInputs,
) -> tuple[OwningReplayInputs, int]:
    corrupted = OwningReplayInputs(*(
        tensor.clone() for tensor in inputs.tensors()))
    target_slot = -1
    # Rank 2 has live expert slots in the frozen C061 fixture. Corrupt only the
    # persisted ingress slot: host preflight still accepts the safe rank/slot
    # bounds, then the expert kernel must report RouteMismatch before using a
    # peer pointer.
    if rank == c061._NUM_SOURCE_RANKS:
        live = torch.nonzero(
            corrupted.expert_ready.eq(c061._GENERATION), as_tuple=False)
        assert live.numel() > 0
        target_slot = int(live[0, 0])
        raw = bytearray(
            corrupted.expert_routes[target_slot].cpu().numpy().tobytes())
        fields = list(struct.unpack("<Q6i", raw))
        fields[3] = (fields[3] + 1) % c061._BASE_CAPACITY
        replacement = torch.tensor(
            list(struct.pack("<Q6i", *fields)),
            dtype=torch.uint8,
            device=corrupted.expert_routes.device,
        )
        corrupted.expert_routes[target_slot].copy_(replacement)
    return corrupted, target_slot


def _verify_corrupt_route_result(
    rank: int,
    target_slot: int,
    outputs: Sequence[torch.Tensor],
) -> None:
    assert len(outputs) == 12
    stage_status = outputs[10].cpu()
    arena_guard = outputs[11].cpu()
    assert tuple(stage_status.shape) == (4, 72)
    assert torch.equal(
        arena_guard,
        torch.full(
            (c061._ARENA_GUARD_BYTES,), 0xA5,
            dtype=torch.uint8,
            device="cpu",
        ),
    )
    if rank == c061._NUM_SOURCE_RANKS:
        assert target_slot >= 0
        assert int(stage_status[0, target_slot]) == 4  # RouteMismatch


@torch.inference_mode()
def _worker(
    local_rank: int,
    num_local_ranks: int,
    args: argparse.Namespace,
) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(
        local_rank, num_local_ranks, seed=c061._GENERATION)
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(seconds=args.timeout),
    )
    assert rank == local_rank and world_size == c061._WORLD_SIZE

    buffer = None
    clean_shutdown = False
    try:
        segmented, records = c061._checked_local_phase(
            "C061 destination-segmented oracle",
            control_group,
            c061._build_oracle,
        )
        quota_cpu = torch.tensor(
            segmented.records_per_egress_destination,
            dtype=torch.int32,
            device="cpu",
        )
        quota_cuda = quota_cpu.to(torch.device("cuda", rank))
        source_inputs = tuple(
            c061._source_inputs(owner, args.hidden)
            for owner in range(c061._NUM_SOURCE_RANKS)
        )
        expected_results = tuple(
            c061._expected_results(owner, args.hidden)
            for owner in range(c061._NUM_SOURCE_RANKS)
        )
        x, topk_idx, weights = c061._checked_local_phase(
            "deterministic local inputs",
            control_group,
            lambda: c061._make_local_inputs(rank, args.hidden),
        )
        manifest, fingerprints = c061._manifest_for_rank(records, rank)
        preflight = (
            manifest.cpu().tolist(),
            fingerprints.cpu().tolist(),
            quota_cuda.cpu().tolist(),
        )
        gathered_preflight = [None] * world_size
        dist.all_gather_object(
            gathered_preflight, preflight, group=control_group)
        c061._checked_local_phase(
            "C061 capture preflight",
            control_group,
            lambda: c061._verify_preflight(gathered_preflight, records),
        )

        layout = c061._checked_local_phase(
            "C070 layout ABI", control_group,
            lambda: c061._layout(args.hidden),
        )
        base_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
            ep_group,
            num_max_tokens_per_rank=c061._NUM_TOKENS,
            hidden=args.hidden,
            num_topk=c061._NUM_TOPK,
            use_fp8_dispatch=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=False,
        )
        arena_alignment = int(_C.get_elastic_buffer_alignment())
        arena_offset = align(base_bytes, arena_alignment)
        buffer = deep_ep.ElasticBuffer(
            ep_group,
            num_bytes=align(
                arena_offset + layout.arena_bytes
                + c061._ARENA_GUARD_BYTES,
                arena_alignment,
            ),
            num_max_tokens_per_rank=c061._NUM_TOKENS,
            hidden=args.hidden,
            num_topk=c061._NUM_TOPK,
            allow_hybrid_mode=False,
            allow_multiple_reduction=False,
            prefer_overlap_with_compute=False,
            explicitly_destroy=True,
            num_gpu_timeout_secs=args.timeout,
            num_cpu_timeout_secs=args.timeout,
        )
        assert buffer.get_logical_domain_size() == (1, c061._WORLD_SIZE)

        collective_config = (
            world_size, args.hidden, arena_offset, layout.arena_bytes,
            tuple(tuple(row) for row in quota_cpu.tolist()),
        )
        configs = [None] * world_size
        dist.all_gather_object(configs, collective_config, group=control_group)
        c061._checked_local_phase(
            "collective configuration",
            control_group,
            lambda: c061.assert_all_equal(configs),
        )
        dist.monitored_barrier(
            group=control_group,
            timeout=timedelta(seconds=args.timeout),
            wait_all_ranks=True,
        )

        roundtrip_outputs = buffer.runtime._rail_balance_vnode_roundtrip(
            x,
            topk_idx,
            weights,
            manifest,
            fingerprints,
            quota_cuda,
            arena_offset,
            c061._BASE_CAPACITY,
            c061._DESTINATION_CAPACITY,
            c061._NUM_DESTINATIONS,
            c061._GENERATION,
            c061._NUM_TOKENS,
            c061._NUM_SOURCE_RANKS,
            c061._EXPERT_BEGIN,
            c061._EXPERTS_PER_RANK,
        )
        torch.cuda.synchronize()
        c061._checked_local_phase(
            "C061 capture exact verification",
            control_group,
            lambda: c061._verify_local_outputs(
                rank,
                roundtrip_outputs,
                records,
                source_inputs,
                tuple(item[0] for item in expected_results),
                tuple(item[1] for item in expected_results),
                layout,
                args.hidden,
            ),
        )

        replay_inputs = _clone_replay_inputs(
            topk_idx, quota_cuda, roundtrip_outputs)
        input_bytes = _snapshot_input_bytes(replay_inputs)
        if manifest.numel():
            manifest.fill_(-777777)
        if fingerprints.numel():
            fingerprints.fill_(-777777777777)
        torch.cuda.synchronize()
        del manifest, fingerprints, roundtrip_outputs

        replay_outputs = _call_replay(
            buffer, replay_inputs, args.hidden, arena_offset)
        torch.cuda.synchronize()
        snapshot = _gather_cpu_snapshot(
            rank,
            control_group,
            replay_outputs,
            replay_inputs.topk_idx,
            quota_cpu,
            args.hidden,
        )
        oracle = c061._checked_local_phase(
            "manifest-free CPU replay oracle",
            control_group,
            lambda: replay_cpu.replay_c061_snapshot(snapshot),
        )
        assert len(oracle.base_descriptors) == 33
        assert len(oracle.contributions) == 72
        c061._checked_local_phase(
            "first replay exact verification",
            control_group,
            lambda: _verify_replay_outputs(
                rank, replay_outputs, replay_inputs,
                oracle,
                tuple(item[0] for item in expected_results),
                tuple(item[1] for item in expected_results),
                layout, args.hidden),
        )
        c061._checked_local_phase(
            "first replay input immutability",
            control_group,
            lambda: _assert_input_bytes_unchanged(
                replay_inputs, input_bytes),
        )
        first_outputs = tuple(tensor.clone() for tensor in replay_outputs)

        dist.monitored_barrier(
            group=control_group,
            timeout=timedelta(seconds=args.timeout),
            wait_all_ranks=True,
        )
        second_outputs = _call_replay(
            buffer, replay_inputs, args.hidden, arena_offset)
        torch.cuda.synchronize()
        c061._checked_local_phase(
            "same-snapshot replay reproducibility",
            control_group,
            lambda: _assert_outputs_identical(
                first_outputs, second_outputs),
        )
        c061._checked_local_phase(
            "second replay input immutability",
            control_group,
            lambda: _assert_input_bytes_unchanged(
                replay_inputs, input_bytes),
        )

        if args.check_route_corruption:
            corrupted_inputs, target_slot = _clone_with_one_corrupt_route(
                rank, replay_inputs)
            corrupted_bytes = _snapshot_input_bytes(corrupted_inputs)
            dist.monitored_barrier(
                group=control_group,
                timeout=timedelta(seconds=args.timeout),
                wait_all_ranks=True,
            )
            corrupted_outputs = _call_replay(
                buffer, corrupted_inputs, args.hidden, arena_offset)
            torch.cuda.synchronize()
            c061._checked_local_phase(
                "corrupt route reports bounded RouteMismatch",
                control_group,
                lambda: _verify_corrupt_route_result(
                    rank, target_slot, corrupted_outputs),
            )
            c061._checked_local_phase(
                "corrupt replay input immutability",
                control_group,
                lambda: _assert_input_bytes_unchanged(
                    corrupted_inputs, corrupted_bytes),
            )
            del corrupted_outputs, corrupted_inputs
        del replay_outputs, first_outputs, second_outputs, replay_inputs
        torch.cuda.synchronize()
        dist.monitored_barrier(
            group=control_group,
            timeout=timedelta(seconds=args.timeout),
            wait_all_ranks=True,
        )
        clean_shutdown = True
    finally:
        if clean_shutdown and buffer is not None:
            buffer.destroy()
            buffer = None
            dist.monitored_barrier(
                group=control_group,
                timeout=timedelta(seconds=args.timeout),
                wait_all_ranks=True,
            )
            if rank == 0:
                print(
                    "PASS C070 GPU replay: capture -> owning snapshot -> "
                    f"manifest-free exact replay, hidden={args.hidden}, "
                    "second replay byte-identical"
                    + (", corrupt route bounded" if
                       args.check_route_corruption else ""),
                    flush=True,
                )
            dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C070 deterministic C061 owning-snapshot GPU replay")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--num-processes", type=int, default=c061._WORLD_SIZE)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--master-port", type=int, default=29669)
    parser.add_argument("--check-route-corruption", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        parser.error("C070 correctness checks require Python assertions")
    if args.num_processes != c061._WORLD_SIZE:
        parser.error("C070 replay requires exactly 8 processes")
    if args.hidden <= 0 or args.hidden % 16:
        parser.error("--hidden must be a positive multiple of 16")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> None:
    args = _parse_args()
    c061.test_cpu_oracle()
    replay_cpu.test_cpu_replay_oracle()
    if args.cpu_only:
        print("PASS C070 GPU replay imports and both CPU oracles")
        return
    if torch.cuda.device_count() < c061._WORLD_SIZE:
        raise SystemExit("C070 replay requires at least 8 visible CUDA devices")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ.setdefault("EP_DISABLE_GIN", "1")
    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
        join=True,
    )


if __name__ == "__main__":
    main()
