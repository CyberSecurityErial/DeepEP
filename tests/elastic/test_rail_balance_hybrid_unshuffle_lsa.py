"""C080-F: strict 8-GPU return-unshuffle LSA correctness test.

The production-shared unshuffle consumes only compact ``moved/group_prefix``
state and the preserved dispatch ``TokenLayout``.  This test supplies complete
combine ``TokenLayout`` records with unique raw-byte fingerprints, poisons the
legacy reduce arena, then verifies every target and every untouched byte.

CPU contract::

    PYTHONPATH=. python -B \
      tests/elastic/test_rail_balance_hybrid_unshuffle_lsa.py --oracle-only

Strict H200 path::

    PYTHONPATH=. EP_DISABLE_GIN=1 \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      python -B tests/elastic/test_rail_balance_hybrid_unshuffle_lsa.py

Named C100 functionality fixtures add either::

    --case-name c100_volume_h256
    --case-name c100_volume_h7168
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import align
from rail_balance_hybrid_reference import (
    HybridRailSchedule,
    ResolvedHybridDestinationCopy,
)
from test_rail_balance_hybrid_shuffle_lsa import (
    ShuffleCase,
    _C100_CASE_NAMES,
    _WORLD_SIZE,
    _abort,
    _assert_shuffle_oracle,
    _checked_phase,
    _finish,
    _gate_signature,
    _gather_objects,
    _local_source_inputs,
    _local_topk,
    _monitored_barrier,
    _moved_copies,
    _prepare_case,
    _schedule,
    _source_shuffle,
    _verify_outputs,
)
from test_rail_balance_hybrid_plan_lsa import PlanCase


_POISON = 0xA7
_STREAM_DELAY_CYCLES = 2_000_000
_OPERATION = "rail_balance_hybrid_return_unshuffle"


def _combine_token_bytes(hidden: int, num_topk: int) -> int:
    hidden_bytes = hidden * torch.bfloat16.itemsize
    metadata_bytes = num_topk * (4 + 4)
    return align(hidden_bytes, 32) + align(metadata_bytes, 32)


def _selected_cases(
    profile_case_name: str | None = None,
) -> tuple[ShuffleCase, ...]:
    by_name = {
        spec.plan.name: spec for spec in _assert_shuffle_oracle(
            include_c100_profiles=profile_case_name is not None)
    }
    base_cases = (
        by_name["c061_source_shuffle_seed62"],
        by_name["all_owner_c1024_d32_k4"],
        by_name["c061_h7168_moved_seed64"],
    )
    assert tuple(spec.hidden for spec in base_cases) == (256, 256, 7168)
    assert tuple(spec.expected_moved_copies for spec in base_cases) == (
        21, 24, 21)
    assert base_cases[1].plan.num_scaleout_ranks == 32
    assert base_cases[1].plan.num_topk == 4
    assert (
        base_cases[1].plan.num_scaleout_ranks
        > base_cases[1].plan.num_topk
    )

    wide_records = _moved_copies(
        base_cases[1].plan, _schedule(base_cases[1].plan))
    assert wide_records
    assert all(_legacy_reduce_row(base_cases[1].plan, record) == 3
               for record in wide_records)

    cases = base_cases
    if profile_case_name is not None:
        assert profile_case_name in _C100_CASE_NAMES
        profile_case = by_name[profile_case_name]
        assert profile_case.expected_moved_copies == 7_168
        assert profile_case.plan.proxy_capacity_per_egress == 896
        cases += (profile_case,)
    return cases


def _legacy_reduce_row(
    case: PlanCase,
    record: ResolvedHybridDestinationCopy,
) -> int:
    num_destinations = case.num_scaleout_ranks
    num_topk = case.num_topk
    destination = record.copy.destination
    if num_destinations <= num_topk:
        return destination

    num_experts = case.num_experts
    experts_per_destination = num_experts // num_destinations
    topk = case.topk_idx[record.copy.owner][record.copy.token]
    matching = [
        lane for lane, expert in enumerate(topk)
        if expert // experts_per_destination == destination
    ]
    assert matching
    # Legacy get_master_lane_idx/bfind selects the highest matching lane.
    return matching[-1]


def _fingerprint(iteration: int, egress: int, proxy_slot: int) -> int:
    value = (
        (0x5A << 56)
        | ((iteration & 0xFFFF) << 32)
        | ((egress & 0xFF) << 24)
        | (proxy_slot & 0xFFFFFF)
    )
    assert 0 <= value < 1 << 63
    return value


def _raw_return_row(
    token_bytes: int,
    iteration: int,
    egress: int,
    proxy_slot: int,
) -> torch.Tensor:
    byte = torch.arange(token_bytes, dtype=torch.int64, device="cpu")
    seed = iteration * 67 + egress * 29 + proxy_slot * 113
    row = ((byte * 37 + seed) & 0xFF).to(torch.uint8).contiguous()
    fingerprint = _fingerprint(iteration, egress, proxy_slot)
    row[:8] = torch.tensor(
        list(fingerprint.to_bytes(8, "little")),
        dtype=torch.uint8,
        device="cpu",
    )
    assert int.from_bytes(bytes(row[:8].tolist()), "little") == fingerprint
    return row


def _proxy_return_cpu(spec: ShuffleCase, egress: int) -> torch.Tensor:
    case = spec.plan
    token_bytes = _combine_token_bytes(spec.hidden, case.num_topk)
    rows = torch.empty(
        (case.proxy_capacity_per_egress, token_bytes),
        dtype=torch.uint8,
        device="cpu",
    )
    for proxy_slot in range(case.proxy_capacity_per_egress):
        rows[proxy_slot] = _raw_return_row(
            token_bytes, spec.iteration, egress, proxy_slot)
    fingerprints = {
        int.from_bytes(bytes(row[:8].tolist()), "little") for row in rows
    }
    assert len(fingerprints) == case.proxy_capacity_per_egress
    return rows.contiguous()


def _target_map(
    spec: ShuffleCase,
    schedule: HybridRailSchedule,
    records: Sequence[ResolvedHybridDestinationCopy],
) -> dict[tuple[int, int, int], ResolvedHybridDestinationCopy]:
    targets: dict[
        tuple[int, int, int], ResolvedHybridDestinationCopy
    ] = {}
    for record in records:
        row = _legacy_reduce_row(spec.plan, record)
        key = (record.copy.owner, row, record.copy.token)
        assert key not in targets, (key, targets.get(key), record)
        assert 0 <= row < (
            spec.plan.num_scaleout_ranks
            if spec.plan.num_scaleout_ranks <= spec.plan.num_topk
            else spec.plan.num_topk
        )
        targets[key] = record
    assert len(targets) == schedule.moved_copies == len(records)
    return targets


def _assert_cpu_oracle(
    profile_case_name: str | None = None,
) -> tuple[ShuffleCase, ...]:
    cases = _selected_cases(profile_case_name)
    global_fingerprints: set[int] = set()
    for spec in cases:
        schedule = _schedule(spec.plan)
        records = _moved_copies(spec.plan, schedule)
        targets = _target_map(spec, schedule, records)
        assert len(targets) == spec.expected_moved_copies
        for egress in range(_WORLD_SIZE):
            for proxy_slot in range(spec.plan.proxy_capacity_per_egress):
                fingerprint = _fingerprint(
                    spec.iteration, egress, proxy_slot)
                # Iterations differ across cases, so every full-suite record
                # identity is globally unique as well as egress-local.
                assert fingerprint not in global_fingerprints
                global_fingerprints.add(fingerprint)
            if spec.plan.proxy_capacity_per_egress > 0:
                probe_slot = spec.plan.proxy_capacity_per_egress - 1
                row = _raw_return_row(
                    _combine_token_bytes(spec.hidden, spec.plan.num_topk),
                    spec.iteration, egress, probe_slot)
                assert int.from_bytes(bytes(row[:8].tolist()), "little") == \
                    _fingerprint(spec.iteration, egress, probe_slot)
        if spec.plan.num_scaleout_ranks > spec.plan.num_topk:
            assert {key[1] for key in targets} == {3}
    return cases


def _return_unshuffle(
    runtime: object,
    proxy_return: torch.Tensor,
    reduce_seed: torch.Tensor,
    invocation_id: int,
    stream: torch.cuda.Stream | None,
) -> torch.Tensor:
    def call() -> torch.Tensor:
        value = runtime._rail_balance_hybrid_return_unshuffle_test(  # type: ignore[attr-defined]
            proxy_return, reduce_seed, invocation_id)
        assert isinstance(value, torch.Tensor)
        return value

    if stream is None:
        snapshot = call()
    else:
        with torch.cuda.stream(stream):
            snapshot = call()
    assert snapshot.is_cuda and snapshot.dtype == torch.uint8
    assert snapshot.is_contiguous() and snapshot.ndim == 2
    return snapshot.cpu().contiguous()


def _verify_local_reduce(
    *,
    spec: ShuffleCase,
    schedule: HybridRailSchedule,
    records: Sequence[ResolvedHybridDestinationCopy],
    rank: int,
    snapshot: torch.Tensor,
) -> tuple[tuple[int, int, int, int, int], ...]:
    case = spec.plan
    token_bytes = _combine_token_bytes(spec.hidden, case.num_topk)
    num_rows = (
        case.num_scaleout_ranks
        if case.num_scaleout_ranks <= case.num_topk
        else case.num_topk
    )
    assert snapshot.device.type == "cpu"
    assert snapshot.dtype == torch.uint8 and snapshot.is_contiguous()
    assert tuple(snapshot.shape) == (
        num_rows * case.num_max_tokens_per_rank,
        token_bytes,
    )

    expected = torch.full_like(snapshot, _POISON)
    targets = _target_map(spec, schedule, records)
    local_targets: set[int] = set()
    coverage = []
    for (owner, row, token), record in targets.items():
        if owner != rank:
            continue
        flat_slot = row * case.num_max_tokens_per_rank + token
        assert flat_slot not in local_targets
        local_targets.add(flat_slot)
        expected[flat_slot] = _raw_return_row(
            token_bytes,
            spec.iteration,
            record.resolution.egress,
            record.resolution.proxy_slot,
        )
        coverage.append((
            owner,
            row,
            token,
            record.resolution.egress,
            record.resolution.proxy_slot,
        ))

    assert torch.equal(snapshot, expected)
    untouched = torch.ones(
        snapshot.size(0), dtype=torch.bool, device="cpu")
    if local_targets:
        untouched[torch.tensor(
            sorted(local_targets), dtype=torch.long, device="cpu")] = False
    if bool(untouched.any()):
        assert torch.equal(
            snapshot[untouched],
            torch.full_like(snapshot[untouched], _POISON),
        )
    return tuple(sorted(coverage))


def _assert_second_call_rejected(
    runtime: object,
    proxy_return: torch.Tensor,
    reduce_seed: torch.Tensor,
    invocation_id: int,
    stream: torch.cuda.Stream | None,
    control_group: dist.ProcessGroup,
) -> None:
    error = None
    try:
        _return_unshuffle(
            runtime, proxy_return, reduce_seed, invocation_id, stream)
    except BaseException:
        error = traceback.format_exc()
    errors = _gather_objects(error, control_group)
    assert all(item is not None for item in errors), errors
    assert all("return_unshuffle_tested" in str(item) for item in errors)


def _run_transaction(
    *,
    runtime: object,
    rank: int,
    control_group: dist.ProcessGroup,
    timeout: int,
    spec: ShuffleCase,
    arena_offset: int,
    arena_bytes: int,
    invocation_id: int,
) -> None:
    case = spec.plan
    schedule = _schedule(case)
    records = _moved_copies(case, schedule)
    stream = torch.cuda.Stream(device=rank) if spec.nondefault_stream else None
    if stream is None:
        topk_idx = _local_topk(case, rank)
        x, topk_weights = _local_source_inputs(
            case, rank, spec.iteration, spec.hidden)
    else:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(_STREAM_DELAY_CYCLES)
            topk_idx = _local_topk(case, rank)
            x, topk_weights = _local_source_inputs(
                case, rank, spec.iteration, spec.hidden)

    try:
        status, prepare_error = _prepare_case(
            runtime,
            topk_idx,
            case,
            spec.hidden,
            arena_offset=arena_offset,
            invocation_id=invocation_id,
            stream=stream,
        )
        gate1 = _gather_objects((
            _OPERATION,
            1,
            invocation_id,
            status,
            _gate_signature(
                case, spec.hidden,
                arena_offset=arena_offset, arena_bytes=arena_bytes),
            int(topk_idx.shape[0]),
            prepare_error,
        ), control_group)
        assert tuple(int(item[3]) for item in gate1) == (0,) * _WORLD_SIZE
        assert len({item[4] for item in gate1}) == 1
        assert tuple(int(item[5]) for item in gate1) == \
            case.num_tokens_per_rank

        outputs, finish_status, finish_error = _finish(
            runtime, invocation_id, stream)
        gate2 = _gather_objects((
            _OPERATION,
            2,
            invocation_id,
            finish_status,
            finish_error,
        ), control_group)
        assert tuple(int(item[3]) for item in gate2) == (0,) * _WORLD_SIZE
        assert outputs is not None
        _checked_phase(
            "exact compact plan",
            control_group,
            lambda: _verify_outputs(outputs, schedule, rank),
        )
        _checked_phase(
            "source shuffle before return",
            control_group,
            lambda: _source_shuffle(
                runtime, x, topk_weights, invocation_id, stream),
        )

        token_bytes = _combine_token_bytes(spec.hidden, case.num_topk)
        num_rows = (
            case.num_scaleout_ranks
            if case.num_scaleout_ranks <= case.num_topk
            else case.num_topk
        )
        proxy_cpu = _proxy_return_cpu(spec, rank)
        seed_cpu = torch.full(
            (num_rows * case.num_max_tokens_per_rank, token_bytes),
            _POISON,
            dtype=torch.uint8,
            device="cpu",
        )
        device = torch.device("cuda", rank)
        if stream is None:
            proxy_return = proxy_cpu.to(device)
            reduce_seed = seed_cpu.to(device)
        else:
            with torch.cuda.stream(stream):
                torch.cuda._sleep(_STREAM_DELAY_CYCLES)
                proxy_return = proxy_cpu.to(device)
                reduce_seed = seed_cpu.to(device)

        snapshot = _checked_phase(
            "production-shared return unshuffle",
            control_group,
            lambda: _return_unshuffle(
                runtime, proxy_return, reduce_seed,
                invocation_id, stream),
        )
        local_coverage = _checked_phase(
            "byte-exact owner reduce snapshot",
            control_group,
            lambda: _verify_local_reduce(
                spec=spec,
                schedule=schedule,
                records=records,
                rank=rank,
                snapshot=snapshot,
            ),
        )
        coverage = _gather_objects(local_coverage, control_group)

        def verify_global_coverage() -> None:
            observed = sorted(
                item for rank_items in coverage for item in rank_items)
            expected = sorted((
                record.copy.owner,
                _legacy_reduce_row(case, record),
                record.copy.token,
                record.resolution.egress,
                record.resolution.proxy_slot,
            ) for record in records)
            assert observed == expected
            assert len(observed) == len(set(observed)) == \
                schedule.moved_copies

        _checked_phase(
            "global collision-free target coverage",
            control_group,
            verify_global_coverage,
        )
        _assert_second_call_rejected(
            runtime, proxy_return, reduce_seed,
            invocation_id, stream, control_group)
    finally:
        _abort(runtime, invocation_id, control_group, timeout)


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int,
            args: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(
        local_rank, num_local_ranks, seed=93)
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(seconds=args.timeout),
    )
    assert rank == local_rank and world_size == _WORLD_SIZE
    buffer = None
    clean_shutdown = False
    try:
        profile_case_name = (
            args.case_name if args.case_name in _C100_CASE_NAMES else None)
        cases = _assert_cpu_oracle(profile_case_name)
        if args.case_name is not None:
            cases = tuple(
                spec for spec in cases if spec.plan.name == args.case_name)
            assert len(cases) == 1

        layouts = []
        for spec in cases:
            layout = tuple(int(value) for value in
                           _C._get_rail_balance_hybrid_layout(
                               spec.hidden, spec.plan.num_topk,
                               spec.plan.proxy_capacity_per_egress))
            assert len(layout) == 10
            assert layout[7] == _combine_token_bytes(
                spec.hidden, spec.plan.num_topk)
            layouts.append(layout)
        arena_bytes = max(layout[-1] for layout in layouts)
        max_hidden = max(spec.hidden for spec in cases)
        max_tokens = max(
            spec.plan.num_max_tokens_per_rank for spec in cases)
        max_topk = max(spec.plan.num_topk for spec in cases)
        assert all(spec.plan.num_topk == max_topk for spec in cases)
        alignment = int(_C.get_elastic_buffer_alignment())
        base_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
            ep_group,
            num_max_tokens_per_rank=max_tokens,
            hidden=max_hidden,
            num_topk=max_topk,
            use_fp8_dispatch=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
        )
        arena_offset = align(base_bytes, alignment)
        assert arena_offset == base_bytes
        buffer = deep_ep.ElasticBuffer(
            ep_group,
            num_bytes=arena_offset + arena_bytes,
            num_max_tokens_per_rank=max_tokens,
            hidden=max_hidden,
            num_topk=max_topk,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
            prefer_overlap_with_compute=False,
            explicitly_destroy=True,
            num_gpu_timeout_secs=args.timeout,
            num_cpu_timeout_secs=args.timeout,
        )
        assert buffer.get_logical_domain_size() == (1, _WORLD_SIZE)
        assert buffer.scaleout_rank_idx == 0
        assert buffer.scaleup_rank_idx == rank

        _checked_phase(
            "warm LSA visibility barrier",
            control_group,
            lambda: buffer.runtime.barrier(  # type: ignore[attr-defined]
                True, True, True),
        )
        for index, spec in enumerate(cases):
            _run_transaction(
                runtime=buffer.runtime,
                rank=rank,
                control_group=control_group,
                timeout=args.timeout,
                spec=spec,
                arena_offset=arena_offset,
                arena_bytes=arena_bytes,
                invocation_id=951 + index,
            )

        _monitored_barrier(control_group, args.timeout)
        clean_shutdown = True
    finally:
        if clean_shutdown and buffer is not None:
            buffer.destroy()
            buffer = None
            _monitored_barrier(control_group, args.timeout)
            if rank == 0:
                if len(cases) == 1 and cases[0].plan.name in _C100_CASE_NAMES:
                    spec = cases[0]
                    print(
                        "PASS C100 controlled Hybrid return-unshuffle fixture: "
                        f"{spec.plan.name}, true 8-GPU LSA, "
                        f"moved={spec.expected_moved_copies}, "
                        "exact owner row/token bytes and poison-preserved "
                        "non-targets (functionality only)",
                        flush=True,
                    )
                else:
                    print(
                        "PASS C080-F Hybrid return unshuffle: true 8-GPU LSA, "
                        "C061 H256, all-owner C1024/D32>K4 highest-lane, "
                        "C061 H7168, unique complete combine TokenLayout raw "
                        "bytes, exact owner row/token targets, poison-preserved "
                        "non-targets, collision freedom, one-shot rejection, "
                        "and abort/recovery",
                        flush=True,
                    )
            dist.destroy_process_group()


def _run_watchdog(arguments: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker-suite",
        "--num-processes", str(arguments.num_processes),
        "--timeout", str(arguments.timeout),
        "--master-port", str(arguments.master_port),
        "--watchdog-seconds", str(arguments.watchdog_seconds),
    ]
    if arguments.case_name is not None:
        command.extend(("--case-name", arguments.case_name))
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=arguments.watchdog_seconds)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError(
            "C080-F subprocess watchdog expired after "
            f"{arguments.watchdog_seconds}s") from error
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C080-F strict 8-GPU Hybrid return-unshuffle test")
    if not __debug__:
        parser.error("C080-F correctness checks require Python assertions")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--worker-suite", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--master-port", type=int, default=29913)
    parser.add_argument("--watchdog-seconds", type=int, default=900)
    parser.add_argument(
        "--case-name",
        help="run one named GPU case while retaining the full CPU oracle",
    )
    arguments = parser.parse_args()
    if arguments.num_processes != _WORLD_SIZE:
        parser.error("C080-F requires exactly 8 processes")
    if arguments.timeout <= 0 or arguments.watchdog_seconds <= 0:
        parser.error("timeouts must be positive")

    profile_case_name = (
        arguments.case_name
        if arguments.case_name in _C100_CASE_NAMES else None)
    cases = _assert_cpu_oracle(profile_case_name)
    by_name = {spec.plan.name: spec for spec in cases}
    if arguments.case_name is not None and arguments.case_name not in by_name:
        parser.error(
            "unknown --case-name; expected one of "
            + ", ".join(sorted(set(by_name) | set(_C100_CASE_NAMES))))
    message = (
        "PASS C080-F CPU oracle: C061 H256/H7168 plus all-owner "
        "C1024/D32>K4 highest matching lane, unique return records, "
        "collision-free legacy targets, and poison-preserved non-targets"
    )
    if profile_case_name is not None:
        spec = by_name[profile_case_name]
        message += (
            f"; C100 {profile_case_name}, "
            f"moved={spec.expected_moved_copies}, "
            f"Pcap={spec.plan.proxy_capacity_per_egress} (named only)"
        )
    print(message, flush=True)
    if arguments.oracle_only:
        return

    required = (
        "_rail_balance_hybrid_plan_prepare",
        "_rail_balance_hybrid_plan_finish",
        "_rail_balance_hybrid_plan_abort",
        "_rail_balance_hybrid_source_shuffle",
        "_rail_balance_hybrid_return_unshuffle_test",
    )
    runtime_type = getattr(_C, "ElasticBuffer", None)
    missing = [
        name for name in required
        if runtime_type is None or not hasattr(runtime_type, name)
    ]
    if missing:
        parser.error(
            "rebuild the extension with the C080-F private API; missing "
            + ", ".join(missing))
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("C080-F requires at least 8 visible CUDA devices")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(arguments.master_port)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ.setdefault("EP_DISABLE_GIN", "1")
    if not arguments.worker_suite:
        _run_watchdog(arguments)
        return
    torch.multiprocessing.spawn(
        _worker,
        args=(arguments.num_processes, arguments),
        nprocs=arguments.num_processes,
        join=True,
    )


if __name__ == "__main__":
    main()
