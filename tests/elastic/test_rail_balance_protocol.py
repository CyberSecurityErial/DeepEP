"""C050: generation-aware one-shot proxy publication protocol.

This is an 8-GPU functional test. It proves forced out-of-order publication,
contiguous-tail safety, payload consumption during unfinished production, slow
consumer progress, progress-aware late publication, stale-generation rejection,
and bounded missing-slot failure. It does not claim ring reuse or performance
results.
"""

from __future__ import annotations

import argparse
import os
import struct
from datetime import timedelta

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.utils.envs import init_dist
import test_rail_balance_shuffle as c040


_WORLD_SIZE = c040._WORLD_SIZE
_NUM_TOKENS = c040._NUM_TOKENS
_HIDDEN = c040._HIDDEN
_NUM_TOPK = c040._NUM_TOPK
_PHYSICAL_CAPACITY = c040._PHYSICAL_CAPACITY
_STALE_GENERATION = 17
_GENERATION = 29
_LATE_PUBLISH_START = 24
_LATE_PUBLISH_AFTER_CONSUMED = 1
_SLOW_CONSUMER_CYCLES = 20_000_000
_PROGRESS_LATE_PUBLISH_AFTER_CONSUMED = 8
_PROGRESS_TIMEOUT_CAP_CYCLES = 600_000_000

_STATUS_COMPLETE = 2
_STATUS_FAILED = 3
_ERROR_NONE = 0
_ERROR_CONSUMER_TIMEOUT = 2


def _publish_key(generation: int, physical_slot: int) -> int:
    return (generation << 32) | (physical_slot + 1)


def _packed_tail(generation: int, tail: int) -> int:
    return (generation << 32) | tail


def _collective_protocol_preflight(
    control_group: dist.ProcessGroup,
    config: tuple,
) -> None:
    gathered = [None] * dist.get_world_size(control_group)
    dist.all_gather_object(gathered, config, group=control_group)
    assert all(item == gathered[0] for item in gathered)


def _decode_control(snapshot: torch.Tensor) -> dict[str, int]:
    assert snapshot.dtype == torch.uint8
    assert tuple(snapshot.shape) == (64,)
    values = struct.unpack("<QQ12i", bytes(snapshot.cpu().tolist()))
    names = (
        "published_tail",
        "consumed_tail",
        "generation",
        "expected_records",
        "status",
        "error_code",
        "error_slot",
        "first_hole_tail",
        "first_hole_ready_slot",
        "max_tail_gap",
        "hole_seen_generation",
        "trace_count",
        "consumer_start_count",
        "stale_observations",
    )
    return dict(zip(names, values, strict=True))


def _verify_layout() -> tuple[int, ...]:
    source = tuple(int(value) for value in
                   _C._get_rail_balance_source_shuffle_layout(
                       _HIDDEN, _NUM_TOPK, _PHYSICAL_CAPACITY))
    protocol = tuple(int(value) for value in
                     _C._get_rail_balance_protocol_layout(
                         _HIDDEN, _NUM_TOPK, _PHYSICAL_CAPACITY))
    (record_bytes, ready_offset, sequence_offset,
     control_offset, control_bytes, arena_bytes) = protocol
    assert record_bytes == source[4]
    assert ready_offset == source[5]
    assert sequence_offset >= ready_offset + _PHYSICAL_CAPACITY * 4
    assert sequence_offset % 32 == 0
    assert control_offset >= sequence_offset + _PHYSICAL_CAPACITY * 8
    assert control_offset % 64 == 0
    assert control_bytes == 64
    assert arena_bytes == source[6] == _C.get_elastic_buffer_alignment()
    return protocol


def _local_expected(expected_records, rank: int):
    local = sorted(
        (record for record in expected_records if record.egress == rank),
        key=lambda record: record.physical_slot,
    )
    assert [record.physical_slot for record in local] == list(
        range(_PHYSICAL_CAPACITY))
    return local


def _verify_success(
    rank: int,
    outputs,
    generation: int,
    expected_records,
    source_x: torch.Tensor,
    source_topk_idx: torch.Tensor,
    source_topk_weights: torch.Tensor,
    source_layout: tuple[int, ...],
    *,
    require_overlap: bool,
    require_slow_consumer_gap: bool,
) -> list[tuple[int, ...]]:
    (records, ready, sequences, consume_counts, ready_at_consume,
     trace, control_snapshot, producer_status) = outputs
    local_expected = _local_expected(expected_records, rank)
    assert tuple(records.shape) == (
        _PHYSICAL_CAPACITY, source_layout[4])
    assert records.dtype == torch.uint8 and records.is_contiguous()
    assert ready.dtype == consume_counts.dtype == torch.int32
    assert sequences.dtype == trace.dtype == torch.int64
    assert torch.equal(
        ready.cpu(),
        torch.full(
            (_PHYSICAL_CAPACITY,), generation,
            dtype=torch.int32, device="cpu"))
    assert sequences.cpu().tolist() == [
        _publish_key(generation, slot)
        for slot in range(_PHYSICAL_CAPACITY)
    ]
    assert consume_counts.cpu().tolist() == [1] * _PHYSICAL_CAPACITY
    assert producer_status.cpu().tolist() == [0] * producer_status.numel()

    control = _decode_control(control_snapshot)
    expected_tail = _packed_tail(generation, _PHYSICAL_CAPACITY)
    assert control["published_tail"] == expected_tail
    assert control["consumed_tail"] == expected_tail
    assert control["generation"] == generation
    assert control["expected_records"] == _PHYSICAL_CAPACITY
    assert control["status"] == _STATUS_COMPLETE
    assert control["error_code"] == _ERROR_NONE
    assert control["error_slot"] == -1
    assert control["first_hole_tail"] == 0
    assert (control["first_hole_ready_slot"] > 0 and
            control["first_hole_ready_slot"] % 2 == 1)
    assert control["hole_seen_generation"] == generation
    assert control["consumer_start_count"] == 1
    assert control["stale_observations"] >= 1

    trace_rows = trace.cpu()[:control["trace_count"]].tolist()
    assert trace_rows and trace_rows[0][0:3] == [0, 0, 0]
    published = 0
    for kind, old_tail, new_tail, _detail in trace_rows:
        if kind == 0:
            assert old_tail == new_tail == published
        else:
            assert kind == 1
            assert old_tail == published
            assert old_tail < new_tail <= _PHYSICAL_CAPACITY
            published = new_tail
    assert published == _PHYSICAL_CAPACITY

    ready_observations = ready_at_consume.cpu().tolist()
    assert all(1 <= value <= _PHYSICAL_CAPACITY
               for value in ready_observations)
    if require_overlap:
        # Late producers cannot publish until consumed_tail >= 1. Therefore
        # slot 0 is necessarily copied while at least one producer is pending.
        assert ready_observations[0] < _PHYSICAL_CAPACITY
    if require_slow_consumer_gap:
        assert control["max_tail_gap"] >= 2

    c040._GENERATION = generation
    records_cpu = records.cpu()
    return [
        c040._decode_and_verify_record(
            records_cpu[slot], expected,
            source_x, source_topk_idx, source_topk_weights, source_layout)
        for slot, expected in enumerate(local_expected)
    ]


def _verify_timeout(
    rank: int,
    outputs,
    generation: int,
    expected_records,
    source_x: torch.Tensor,
    source_topk_idx: torch.Tensor,
    source_topk_weights: torch.Tensor,
    source_layout: tuple[int, ...],
    drop_slot: int,
) -> int:
    (records, ready, sequences, consume_counts, _ready_at_consume,
     _trace, control_snapshot, producer_status) = outputs
    control = _decode_control(control_snapshot)
    assert control["status"] == _STATUS_FAILED
    assert control["error_code"] == _ERROR_CONSUMER_TIMEOUT
    assert control["error_slot"] == drop_slot
    assert control["published_tail"] == _packed_tail(generation, drop_slot)
    assert control["consumed_tail"] == _packed_tail(generation, drop_slot)
    assert consume_counts.cpu().tolist()[:drop_slot] == [1] * drop_slot
    assert consume_counts.cpu().tolist()[drop_slot:] == (
        [0] * (_PHYSICAL_CAPACITY - drop_slot))
    assert ready.cpu().tolist()[drop_slot] == generation - 1
    assert sequences.cpu().tolist()[drop_slot] == (
        _publish_key(generation - 1, drop_slot))
    assert ready.cpu().tolist()[drop_slot + 1] == generation
    assert sequences.cpu().tolist()[drop_slot + 1] == (
        _publish_key(generation, drop_slot + 1))
    statuses = producer_status.cpu().tolist()
    assert all(value in (0, -2) for value in statuses)

    c040._GENERATION = generation
    local_expected = _local_expected(expected_records, rank)
    records_cpu = records.cpu()
    for slot in range(drop_slot):
        c040._decode_and_verify_record(
            records_cpu[slot], local_expected[slot],
            source_x, source_topk_idx, source_topk_weights, source_layout)
    assert torch.equal(
        records_cpu[drop_slot:],
        torch.full(
            (_PHYSICAL_CAPACITY - drop_slot, source_layout[4]),
            0xCC,
            dtype=torch.uint8,
            device="cpu",
        ),
    )
    return statuses.count(-2)


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int,
            args: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(
        local_rank, num_local_ranks, seed=_GENERATION)
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(seconds=args.timeout),
    )
    assert rank == local_rank and world_size == _WORLD_SIZE

    buffer = None
    clean_shutdown = False
    try:
        x, topk_idx, topk_weights, local_destinations = (
            c040._checked_local_phase(
                "input generation", control_group,
                lambda: c040._make_local_inputs(rank),
            ))
        destinations_by_rank = [None] * world_size
        dist.all_gather_object(
            destinations_by_rank, local_destinations, group=control_group)
        oracle, expected_records = c040._checked_local_phase(
            "CPU physical oracle", control_group,
            lambda: c040._build_physical_oracle(destinations_by_rank),
        )
        assert oracle.incoming_copies_per_egress == (
            (_PHYSICAL_CAPACITY,) * _WORLD_SIZE)

        source_x = c040._all_gather_tensor(x, ep_group).cpu()
        source_topk_idx = c040._all_gather_tensor(topk_idx, ep_group).cpu()
        source_topk_weights = c040._all_gather_tensor(
            topk_weights, ep_group).cpu()
        protocol_layout = c040._checked_local_phase(
            "protocol layout ABI", control_group, _verify_layout)
        source_layout = tuple(int(value) for value in
                              _C._get_rail_balance_source_shuffle_layout(
                                  _HIDDEN, _NUM_TOPK,
                                  _PHYSICAL_CAPACITY))

        base_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
            ep_group,
            num_max_tokens_per_rank=_NUM_TOKENS,
            hidden=_HIDDEN,
            num_topk=_NUM_TOPK,
            use_fp8_dispatch=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=False,
        )
        arena_offset = base_bytes
        buffer = deep_ep.ElasticBuffer(
            ep_group,
            num_bytes=base_bytes + protocol_layout[-1],
            num_max_tokens_per_rank=_NUM_TOKENS,
            hidden=_HIDDEN,
            num_topk=_NUM_TOPK,
            allow_hybrid_mode=False,
            allow_multiple_reduction=False,
            prefer_overlap_with_compute=False,
            explicitly_destroy=True,
            num_gpu_timeout_secs=args.timeout,
            num_cpu_timeout_secs=args.timeout,
        )

        manifest, fingerprints = c040._manifest_for_owner(
            expected_records, rank)
        local_preflight = (
            (_WORLD_SIZE, _HIDDEN, _NUM_TOPK, _PHYSICAL_CAPACITY,
             arena_offset, protocol_layout[-1]),
            manifest.cpu().tolist(),
            fingerprints.cpu().tolist(),
        )
        gathered_preflight = [None] * world_size
        dist.all_gather_object(
            gathered_preflight, local_preflight, group=control_group)
        c040._checked_local_phase(
            "collective manifest preflight", control_group,
            lambda: c040._verify_collective_preflight(
                gathered_preflight, expected_records),
        )

        # Deterministic nonzero per-record jitter. It perturbs publication
        # order without making correctness depend on host scheduling.
        producer_delays = (
            (manifest[:, 6].to(torch.int64) * 104_729 +
             rank * 13_007 + manifest[:, 0].to(torch.int64) * 257)
            % 500_000
        ) + 1
        assert producer_delays.is_contiguous()
        assert int(producer_delays.min().item()) > 0
        consumer_delays = torch.zeros(
            _PHYSICAL_CAPACITY, dtype=torch.int64,
            device=torch.device("cuda", rank))

        _collective_protocol_preflight(control_group, (
            "overlap", _GENERATION, _STALE_GENERATION,
            _PHYSICAL_CAPACITY, _PHYSICAL_CAPACITY,
            False, True, -1,
            _LATE_PUBLISH_START, _LATE_PUBLISH_AFTER_CONSUMED,
            args.gpu_timeout_cycles,
        ))
        outputs = buffer.runtime._rail_balance_source_shuffle_protocol(
            x, topk_idx, topk_weights, manifest, fingerprints,
            producer_delays, consumer_delays,
            arena_offset, _PHYSICAL_CAPACITY,
            _GENERATION, _STALE_GENERATION,
            _NUM_TOKENS, _PHYSICAL_CAPACITY,
            False, True, -1,
            _LATE_PUBLISH_START, _LATE_PUBLISH_AFTER_CONSUMED,
            args.gpu_timeout_cycles,
        )
        torch.cuda.synchronize()
        local_coverage = c040._checked_local_phase(
            "forced-hole overlap verification", control_group,
            lambda: _verify_success(
                rank, outputs, _GENERATION, expected_records,
                source_x, source_topk_idx, source_topk_weights,
                source_layout,
                require_overlap=True,
                require_slow_consumer_gap=False,
            ),
        )
        coverage_by_rank = [None] * world_size
        dist.all_gather_object(
            coverage_by_rank, local_coverage, group=control_group)
        c040._checked_local_phase(
            "global overlap coverage", control_group,
            lambda: (
                lambda observed, expected: (
                    (_ for _ in ()).throw(AssertionError())
                    if observed != expected else None
                )
            )(
                sorted(item for items in coverage_by_rank for item in items),
                sorted(record.coverage_key for record in expected_records),
            ),
        )

        slow_consumer_delays = torch.zeros_like(consumer_delays)
        slow_consumer_delays[0] = _SLOW_CONSUMER_CYCLES
        _collective_protocol_preflight(control_group, (
            "slow_consumer", _GENERATION + 1, _GENERATION,
            _PHYSICAL_CAPACITY, _PHYSICAL_CAPACITY,
            True, True, -1, -1, 0, args.gpu_timeout_cycles,
        ))
        slow_outputs = buffer.runtime._rail_balance_source_shuffle_protocol(
            x, topk_idx, topk_weights, manifest, fingerprints,
            producer_delays, slow_consumer_delays,
            arena_offset, _PHYSICAL_CAPACITY,
            _GENERATION + 1, _GENERATION,
            _NUM_TOKENS, _PHYSICAL_CAPACITY,
            True, True, -1, -1, 0, args.gpu_timeout_cycles,
        )
        torch.cuda.synchronize()
        c040._checked_local_phase(
            "slow-consumer verification", control_group,
            lambda: _verify_success(
                rank, slow_outputs, _GENERATION + 1, expected_records,
                source_x, source_topk_idx, source_topk_weights,
                source_layout,
                require_overlap=False,
                require_slow_consumer_gap=True,
            ),
        )

        # The aggregate wait exceeds one watchdog interval, while each
        # individual consumption step remains below it. A fixed absolute
        # watchdog falsely times out; a progress-aware watchdog must pass.
        progress_timeout_cycles = min(
            args.gpu_timeout_cycles, _PROGRESS_TIMEOUT_CAP_CYCLES)
        progress_delay_cycles = progress_timeout_cycles * 2 // 5
        assert 0 < progress_delay_cycles < progress_timeout_cycles
        assert (progress_delay_cycles *
                _PROGRESS_LATE_PUBLISH_AFTER_CONSUMED >
                progress_timeout_cycles)
        progress_consumer_delays = torch.zeros_like(consumer_delays)
        progress_consumer_delays[
            :_PROGRESS_LATE_PUBLISH_AFTER_CONSUMED
        ] = progress_delay_cycles
        progress_generation = _GENERATION + 2
        _collective_protocol_preflight(control_group, (
            "late_slow_progress", progress_generation, _GENERATION + 1,
            _PHYSICAL_CAPACITY, _PHYSICAL_CAPACITY,
            True, True, -1,
            _LATE_PUBLISH_START,
            _PROGRESS_LATE_PUBLISH_AFTER_CONSUMED,
            progress_timeout_cycles,
        ))
        progress_outputs = (
            buffer.runtime._rail_balance_source_shuffle_protocol(
                x, topk_idx, topk_weights, manifest, fingerprints,
                producer_delays, progress_consumer_delays,
                arena_offset, _PHYSICAL_CAPACITY,
                progress_generation, _GENERATION + 1,
                _NUM_TOKENS, _PHYSICAL_CAPACITY,
                True, True, -1,
                _LATE_PUBLISH_START,
                _PROGRESS_LATE_PUBLISH_AFTER_CONSUMED,
                progress_timeout_cycles,
            ))
        torch.cuda.synchronize()
        c040._checked_local_phase(
            "late-gate slow-progress verification", control_group,
            lambda: _verify_success(
                rank, progress_outputs, progress_generation,
                expected_records, source_x, source_topk_idx,
                source_topk_weights, source_layout,
                require_overlap=True,
                require_slow_consumer_gap=True,
            ),
        )

        last_generation = progress_generation
        for repeat_idx in range(args.repeat_generations):
            repeat_generation = _GENERATION + 3 + repeat_idx
            _collective_protocol_preflight(control_group, (
                "repeat", repeat_idx, repeat_generation, last_generation,
                _PHYSICAL_CAPACITY, _PHYSICAL_CAPACITY,
                True, True, -1, -1, 0,
                args.gpu_timeout_cycles,
            ))
            repeat_outputs = (
                buffer.runtime._rail_balance_source_shuffle_protocol(
                    x, topk_idx, topk_weights, manifest, fingerprints,
                    producer_delays, consumer_delays,
                    arena_offset, _PHYSICAL_CAPACITY,
                    repeat_generation, last_generation,
                    _NUM_TOKENS, _PHYSICAL_CAPACITY,
                    True, True, -1, -1, 0,
                    args.gpu_timeout_cycles,
                ))
            torch.cuda.synchronize()
            c040._checked_local_phase(
                f"repeat generation {repeat_idx}", control_group,
                lambda generation=repeat_generation,
                       values=repeat_outputs: _verify_success(
                    rank, values, generation, expected_records,
                    source_x, source_topk_idx, source_topk_weights,
                    source_layout,
                    require_overlap=False,
                    require_slow_consumer_gap=False,
                ),
            )
            last_generation = repeat_generation

        drop_slot = _PHYSICAL_CAPACITY - 2
        fault_generation = last_generation + 1
        _collective_protocol_preflight(control_group, (
            "missing_slot", fault_generation, last_generation,
            _PHYSICAL_CAPACITY, _PHYSICAL_CAPACITY,
            True, True, drop_slot, -1, 0,
            args.gpu_timeout_cycles,
        ))
        timeout_outputs = (
            buffer.runtime._rail_balance_source_shuffle_protocol(
                x, topk_idx, topk_weights, manifest, fingerprints,
                producer_delays, consumer_delays,
                arena_offset, _PHYSICAL_CAPACITY,
                fault_generation, last_generation,
                _NUM_TOKENS, _PHYSICAL_CAPACITY,
                True, True, drop_slot, -1, 0,
                args.gpu_timeout_cycles,
            ))
        torch.cuda.synchronize()
        local_dropped = c040._checked_local_phase(
            "bounded missing-slot timeout", control_group,
            lambda: _verify_timeout(
                rank, timeout_outputs, fault_generation,
                expected_records, source_x, source_topk_idx,
                source_topk_weights, source_layout, drop_slot),
        )
        dropped_by_rank = [None] * world_size
        dist.all_gather_object(
            dropped_by_rank, local_dropped, group=control_group)
        assert sum(dropped_by_rank) == _WORLD_SIZE

        # Capacity is an atomic caller-side gate. A rejected physical plan
        # never enters the private method, so neither consumer nor producer is
        # launched on any rank.
        bypass_capacity = _PHYSICAL_CAPACITY - 1
        bypass = c040.materialize_physical_proxy_slots(
            oracle.logical_candidate,
            global_policy_budget=oracle.logical_candidate.moved_copies,
            physical_capacity_per_egress=bypass_capacity,
        )
        assert not bypass.enabled
        assert bypass.assignments == ()
        _collective_protocol_preflight(control_group, (
            "capacity_bypass", False, bypass_capacity,
            bypass.bypass_reason, len(bypass.assignments),
        ))

        assert torch.equal(x.cpu(), source_x[rank])
        assert torch.equal(topk_idx.cpu(), source_topk_idx[rank])
        assert torch.equal(topk_weights.cpu(), source_topk_weights[rank])
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
                    "PASS C050 one-shot proxy protocol: forced hole, "
                    "stale generation, live payload overlap, slow consumer, "
                    "progress-aware late gating, and bounded missing-slot "
                    "timeout verified",
                    flush=True,
                )
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C050 8-GPU one-shot proxy protocol functional test")
    if not __debug__:
        parser.error("C050 correctness checks require Python assertions")
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--master-port", type=int, default=29650)
    parser.add_argument(
        "--gpu-timeout-cycles", type=int, default=4_000_000_000)
    parser.add_argument(
        "--repeat-generations", type=int, default=16)
    args = parser.parse_args()
    if args.num_processes != _WORLD_SIZE:
        parser.error("C050 requires exactly 8 processes")
    if (args.timeout <= 0 or args.gpu_timeout_cycles < 16 or
            args.repeat_generations <= 0):
        parser.error("timeouts must be positive and GPU timeout >= 16 cycles")
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("C050 requires at least 8 visible CUDA devices")

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
