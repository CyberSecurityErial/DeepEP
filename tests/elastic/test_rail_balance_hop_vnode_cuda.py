"""Minimal 4x2 H200 proof of hop-aware dispatch and inverse combine."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from datetime import timedelta
from fractions import Fraction
from pathlib import Path

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import align
from rail_balance_hop_reference import HopPlannerConfig
from rail_balance_hybrid_vnode_reference import (
    VnodeRoundTripCase,
    VnodeTopology,
    run_hop_vnode_roundtrip,
    run_vnode_roundtrip,
)
from test_rail_balance_hybrid_vnode import (
    _ARENA_ALIGNMENT,
    _ARENA_GUARD_BYTES,
    _checked_phase,
    _cpu_source_inputs,
    _expected_combined,
    _gather_objects,
    _layout_abis,
    _make_cuda_inputs,
    _monitored_barrier,
)


_WORLD_SIZE = 8
_G = 4
_D = 2
_HIDDEN = 256
_M = 16
_K = 1
_CHANNELS = 2
_PROXY_CAPACITY = 16
_GENERATION = 1201


def _case(adaptive: bool = False) -> VnodeRoundTripCase:
    topology = VnodeTopology(rails_per_node=_G, num_nodes=_D)
    targets = ((0,) * 16 if adaptive else
               (0,) * 4 + (1,) * 8 + (2,) * 2 + (3,) * 2)
    experts_per_rank = 2
    topk_idx = (
        tuple(((topology.physical(1, target) * experts_per_rank),)
              for target in targets),
    ) + ((),) * (_G - 1)
    topk_weights = (
        tuple(((Fraction(1),)) for _ in targets),
    ) + ((),) * (_G - 1)
    source_values = (
        tuple(
            tuple(Fraction(8 * ((token + column) & 7))
                  for column in range(3))
            for token in range(len(targets))
        ),
    ) + ((),) * (_G - 1)
    return VnodeRoundTripCase(
        name="hop_vnode_adaptive_4x2" if adaptive else
             "hop_vnode_one_hop_4x2",
        topology=topology,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        source_values=source_values,
        num_topk=_K,
        num_channels=_CHANNELS,
        num_max_tokens_per_rank=_M,
        num_experts=topology.world_size * experts_per_rank,
        experts_per_physical_rank=experts_per_rank,
        proxy_capacity_per_egress=_PROXY_CAPACITY,
        remainder_seed=0,
    )


def _channel_count(case: VnodeRoundTripCase) -> torch.Tensor:
    count = torch.zeros((_G, _CHANNELS, _D), dtype=torch.int32)
    experts_per_destination = case.num_experts // _D
    for owner, tokens in enumerate(case.topk_idx):
        for token, experts in enumerate(tokens):
            for destination in {
                    expert // experts_per_destination for expert in experts}:
                count[owner, token % _CHANNELS, destination] += 1
    return count


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int,
            args: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, world_group = init_dist(
        local_rank, num_local_ranks, seed=_GENERATION)
    control_group = dist.new_group(
        ranks=list(range(world_size)), backend="gloo",
        timeout=timedelta(seconds=args.timeout))
    source_group = dist.new_group(
        ranks=list(range(_G)), backend="nccl",
        timeout=timedelta(seconds=args.timeout),
        use_local_synchronization=False,
        device_id=torch.device(f"cuda:{local_rank}"),
        group_desc="hop-aware-vnode-source")

    case = _case(args.adaptive)
    baseline = run_vnode_roundtrip(case)
    hop_oracle = run_hop_vnode_roundtrip(
        case, HopPlannerConfig(
            num_rails=_G, num_destinations=_D,
            mode="adaptive" if args.adaptive else "one_hop",
            chunk_size=1, max_two_hop_ratio=0.5 if args.adaptive else 0))
    if args.adaptive:
        assert hop_oracle.plan.two_hop_units == 8
    else:
        assert {route.path_kind for route in hop_oracle.routes} >= {
            "direct", "dst_forward", "src_forward"}
        assert all(route.path_kind != "two_hop"
                   for route in hop_oracle.routes)

    source_buffer = None
    world_buffer = None
    source_invocation = _GENERATION * 10 + 1
    world_invocation = _GENERATION * 10 + 2
    clean = False
    try:
        source_inputs = _cpu_source_inputs(case, _HIDDEN)
        x, topk_idx, topk_weights = _make_cuda_inputs(
            rank, baseline, source_inputs, _HIDDEN)
        layout, hybrid_layout = _layout_abis(case, _HIDDEN)

        if rank < _G:
            source_base = deep_ep.ElasticBuffer.get_buffer_size_hint(
                source_group, num_max_tokens_per_rank=_M, hidden=_HIDDEN,
                num_topk=_K, use_fp8_dispatch=False,
                allow_hybrid_mode=False, allow_multiple_reduction=True)
            source_offset = align(source_base, _ARENA_ALIGNMENT)
            source_buffer = deep_ep.ElasticBuffer(
                source_group,
                num_bytes=align(source_offset + hybrid_layout[-1],
                                _ARENA_ALIGNMENT),
                num_max_tokens_per_rank=_M, hidden=_HIDDEN, num_topk=_K,
                allow_hybrid_mode=False, allow_multiple_reduction=True,
                prefer_overlap_with_compute=False, explicitly_destroy=True,
                num_gpu_timeout_secs=args.timeout,
                num_cpu_timeout_secs=args.timeout)
        else:
            source_offset = -1
        source_offsets = _gather_objects(source_offset, control_group)
        source_offset = int(source_offsets[0])
        assert len(set(source_offsets[:_G])) == 1

        world_base = deep_ep.ElasticBuffer.get_buffer_size_hint(
            world_group, num_max_tokens_per_rank=_M, hidden=_HIDDEN,
            num_topk=_K, use_fp8_dispatch=False, allow_hybrid_mode=False,
            allow_multiple_reduction=False)
        world_offset = align(world_base, _ARENA_ALIGNMENT)
        world_buffer = deep_ep.ElasticBuffer(
            world_group,
            num_bytes=align(
                world_offset + layout.vnode_arena_bytes + _ARENA_GUARD_BYTES,
                _ARENA_ALIGNMENT),
            num_max_tokens_per_rank=_M, hidden=_HIDDEN, num_topk=_K,
            allow_hybrid_mode=False, allow_multiple_reduction=False,
            prefer_overlap_with_compute=False, explicitly_destroy=True,
            num_gpu_timeout_secs=args.timeout,
            num_cpu_timeout_secs=args.timeout)
        _checked_phase(
            "warm source barrier", control_group,
            lambda: source_buffer.runtime.barrier(True, True, True)
            if source_buffer is not None else None)

        source_runtime = source_buffer.runtime if source_buffer else None

        def prepare_source() -> int | None:
            if source_runtime is None:
                return None
            return int(source_runtime._rail_balance_hybrid_plan_prepare(
                topk_idx, _HIDDEN, _CHANNELS, _M, case.num_experts, _D, 0,
                _PROXY_CAPACITY, source_offset, source_invocation,
                0, 0, 0, True, 0, 50 if args.adaptive else 0, 0))

        statuses = _gather_objects(_checked_phase(
            "hop source prepare", control_group, prepare_source), control_group)
        assert statuses[:_G] == [0] * _G and statuses[_G:] == [None] * _G

        def finish_source() -> tuple[torch.Tensor, ...] | None:
            if source_runtime is None:
                return None
            source_runtime._rail_balance_hybrid_plan_finish(source_invocation)
            return tuple(source_runtime._rail_balance_hop_plan_snapshot(
                source_invocation))

        source_plan = _checked_phase(
            "hop source finish", control_group, finish_source)
        if rank < _G:
            assert source_plan is not None
            assert source_plan[10].item() == 0
            path_units = tuple(int(value) for value in source_plan[8].cpu())
            if args.adaptive:
                assert path_units == (8, 0, 0, 8)
            else:
                assert path_units[0] > 0 and path_units[1] > 0
                assert path_units[2] > 0 and path_units[3] == 0

        _checked_phase(
            "hop source shuffle", control_group,
            lambda: source_runtime._rail_balance_hybrid_source_shuffle(
                x, topk_weights, source_invocation)
            if source_runtime is not None else None)
        _checked_phase(
            "hop source visibility", control_group,
            lambda: source_runtime.barrier(True, True, True)
            if source_runtime is not None else None)

        proxy_dispatch = (
            source_runtime._rail_balance_hybrid_proxy_dispatch_snapshot(
                source_invocation)
            if source_runtime is not None else torch.empty(
                (0, layout.dispatch_token_bytes), dtype=torch.uint8,
                device=torch.device("cuda", rank)))

        records_cpu = source_plan[0].cpu() if rank == 0 else None
        record_object = [records_cpu]
        dist.broadcast_object_list(record_object, src=0, group=control_group)
        assert isinstance(record_object[0], torch.Tensor)
        hop_records = record_object[0].to(
            torch.device("cuda", rank)).contiguous()
        channel_count = _channel_count(case).to(
            torch.device("cuda", rank)).contiguous()

        prepared = _checked_phase(
            "hop vnode prepare", control_group,
            lambda: tuple(world_buffer.runtime._rail_balance_hybrid_vnode_prepare(
                x, topk_idx, topk_weights, proxy_dispatch, channel_count,
                world_offset, _M, case.num_experts, _D, _G,
                _PROXY_CAPACITY, _GENERATION, world_invocation,
                0, 0, 0, hop_records,
                0, 50 if args.adaptive else 0, 0)))
        assert int(prepared[0]) == 0

        outputs = _checked_phase(
            "hop vnode dispatch/combine", control_group,
            lambda: tuple(world_buffer.runtime._rail_balance_hybrid_vnode_finish(
                world_invocation)))
        assert torch.count_nonzero(outputs[9]).item() == 0
        assert torch.all(outputs[10] == 0xA5).item()

        if source_runtime is not None:
            reduce_snapshot = source_runtime._rail_balance_hybrid_return_unshuffle_test(
                outputs[0], outputs[1], source_invocation)
            assert reduce_snapshot.is_cuda
            combined_x, combined_weights = \
                source_runtime._rail_balance_hybrid_combine_epilogue_test(
                    source_invocation)
            assert torch.equal(
                combined_x.cpu(), _expected_combined(rank, baseline, _HIDDEN))
            assert torch.equal(combined_weights.cpu(), source_inputs[rank][2])

        _monitored_barrier(control_group, args.timeout)
        clean = True
    finally:
        if world_buffer is not None:
            try:
                world_buffer.runtime._rail_balance_hybrid_vnode_abort(
                    world_invocation)
            except BaseException:
                pass
        if source_buffer is not None:
            try:
                source_buffer.runtime._rail_balance_hybrid_plan_abort(
                    source_invocation)
            except BaseException:
                pass
        if clean:
            _monitored_barrier(control_group, args.timeout)
            if source_buffer is not None:
                source_buffer.destroy()
            _monitored_barrier(control_group, args.timeout)
            assert world_buffer is not None
            world_buffer.destroy()
            _monitored_barrier(control_group, args.timeout)
        if clean and rank == 0:
            print(
                "PASS hop-aware vnode: " +
                ("bounded two-hop escape" if args.adaptive else
                 "direct + both one-hop paths") +
                ", dispatch/combine round trip", flush=True)
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--watchdog-seconds", type=int, default=900)
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()
    if args.worker:
        torch.multiprocessing.spawn(
            _worker, args=(_WORLD_SIZE, args), nprocs=_WORLD_SIZE, join=True)
        return

    command = [sys.executable, "-B", str(Path(__file__).resolve()),
               "--worker", "--timeout", str(args.timeout)]
    if args.adaptive:
        command.append("--adaptive")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=args.watchdog_seconds)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError("hop-aware vnode watchdog expired") from error
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
