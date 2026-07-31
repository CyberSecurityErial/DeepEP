"""Auditable C100 wall-clock benchmark for checked Hybrid LSA adapters.

This benchmark intentionally measures the current private, checked adapters,
not production Hybrid/Gin end-to-end dispatch.  It reuses the controlled C100
schedule and payload builders from the strict correctness harnesses.  Every
transaction owns a fresh invocation ID, while the ElasticBuffer and all input
tensors are reused.

Examples::

    PYTHONPATH=tests:tests/elastic:. EP_DISABLE_GIN=1 \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      python -B tests/elastic/bench_rail_balance_hybrid_lsa.py \
        --stage source --case-name c100_volume_h256 \
        --json-out .cache/rail_balance/c100/source-h256.json

    PYTHONPATH=tests:tests/elastic:. EP_DISABLE_GIN=1 \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      python -B tests/elastic/bench_rail_balance_hybrid_lsa.py \
        --stage return --case-name c100_volume_h7168 \
        --warmup-iters 0 --steady-iters 1

The second command is the supported minimal smoke shape.  It is still the
full controlled C100 tensor shape; only the iteration count is reduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, NamedTuple, Sequence, TypeVar

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import align
from rail_balance_hybrid_reference import HybridRailSchedule
from test_rail_balance_hybrid_plan_lsa import (
    _gather_objects,
    _local_topk,
    _monitored_barrier,
)
from test_rail_balance_hybrid_shuffle_lsa import (
    ShuffleCase,
    _C100_CASE_NAMES,
    _WORLD_SIZE,
    _assert_shuffle_oracle,
    _finish,
    _gate_signature,
    _local_source_inputs,
    _prepare_case,
    _schedule,
    _source_shuffle,
)
from test_rail_balance_hybrid_unshuffle_lsa import (
    _POISON,
    _combine_token_bytes,
    _proxy_return_cpu,
)


_LAYOUT_LABELS = (
    "control_offset",
    "control_bytes",
    "channel_count_offset",
    "channel_count_bytes",
    "proxy_dispatch_offset",
    "dispatch_token_bytes",
    "proxy_return_offset",
    "combine_token_bytes",
    "raw_bytes",
    "arena_bytes",
)
_ALL_PHASES = (
    "prepare",
    "finish",
    "prerequisite",
    "stage",
    "post_visibility",
    "abort",
    "transaction",
)
_ENV_EXACT = {
    "CUDA_VISIBLE_DEVICES",
    "EP_DISABLE_GIN",
    "EP_BUFFER_DEBUG",
    "OMP_NUM_THREADS",
    "PYTHONPATH",
}
_ENV_PREFIXES = ("CUDA_", "EP_", "NCCL_", "NVTE_", "TORCH_")
_CONTROL_TIMEOUT_SLACK_SECONDS = 60
_FORMAL_MIN_WARMUP_ITERATIONS = 10
_FORMAL_MIN_STEADY_ITERATIONS = 100
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CODE_IDENTITY_PATHS = (
    "csrc/elastic/buffer.hpp",
    "csrc/jit/compiler.hpp",
    "csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp",
    "csrc/kernels/elastic/rail_balance_hybrid_epilogue.hpp",
    "csrc/kernels/elastic/rail_balance_hybrid_shuffle.hpp",
    "csrc/kernels/elastic/rail_balance_hybrid_unshuffle.hpp",
    "deep_ep/include/deep_ep/common/rail_balance_hybrid_layout.cuh",
    "deep_ep/include/deep_ep/impls/rail_balance_hybrid_plan.cuh",
    "deep_ep/include/deep_ep/impls/rail_balance_hybrid_shuffle.cuh",
    "deep_ep/include/deep_ep/impls/rail_balance_hybrid_unshuffle.cuh",
    "deep_ep/buffers/elastic.py",
    "deep_ep/utils/envs.py",
    "deep_ep/utils/math.py",
    "tests/elastic/bench_rail_balance_hybrid_lsa.py",
    "tests/elastic/rail_balance_hybrid_reference.py",
    "tests/elastic/test_rail_balance_hybrid_plan_lsa.py",
    "tests/elastic/test_rail_balance_hybrid_shuffle_lsa.py",
    "tests/elastic/test_rail_balance_hybrid_unshuffle_lsa.py",
)
_COMMON_JIT_KERNEL_PREFIXES = (
    "kernel.combine_reduce_epilogue.",
    "kernel.rail_balance_hybrid_count_v1.",
    "kernel.rail_balance_hybrid_local_barrier_g8_",
    "kernel.rail_balance_hybrid_plan_v2.",
    "kernel.rail_balance_hybrid_prefix_v1.",
    "kernel.rail_balance_hybrid_return_unshuffle.",
)
_LEGACY_SOURCE_JIT_KERNEL_PREFIX = (
    "kernel.rail_balance_hybrid_source_shuffle.",
)
_HOP_ONE_HOP_JIT_KERNEL_PREFIXES = (
    "kernel.rail_balance_hop_record_v1.",
    "kernel.rail_balance_hop_precount_v1.",
    "kernel.rail_balance_hop_decision_v4.",
    "kernel.rail_balance_hop_endpoint_prefix_v1.",
    "kernel.rail_balance_hop_endpoint_assign_v1.",
    "kernel.rail_balance_hop_group_count_v1.",
    "kernel.rail_balance_hop_group_prefix_v1.",
    "kernel.rail_balance_hop_slot_finalize_v1.",
    "kernel.rail_balance_hop_source_shuffle.",
)
_HOP_ADAPTIVE_JIT_KERNEL_PREFIXES = (
    "kernel.rail_balance_hop_record_v1.",
    "kernel.rail_balance_hop_precount_v1.",
    "kernel.rail_balance_hop_plan_precounted_v3.",
    "kernel.rail_balance_hop_source_shuffle.",
)
_JIT_KERNEL_PREFIXES = (
    _COMMON_JIT_KERNEL_PREFIXES + _LEGACY_SOURCE_JIT_KERNEL_PREFIX +
    _HOP_ONE_HOP_JIT_KERNEL_PREFIXES + _HOP_ADAPTIVE_JIT_KERNEL_PREFIXES
)
_WATCHDOG_CHILD_ENV = "EP_RAIL_BALANCE_BENCH_WATCHDOG_CHILD"
_INTERFERENCE_MODES = ("none", "compute-only", "concurrent")
_HOP_MODES = ("legacy", "one_hop", "adaptive")
_INTERFERENCE_COMPUTE_SHAPE = (1024, 7168, 7168)
_TYPE = TypeVar("_TYPE")


class ComputeInterferenceState(NamedTuple):
    stream: torch.cuda.Stream
    lhs: torch.Tensor
    rhs: torch.Tensor
    output: torch.Tensor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "C100 8-rank checked-adapter Hybrid LSA wall benchmark"))
    parser.add_argument("--stage", required=True, choices=("source", "return"))
    parser.add_argument(
        "--case-name", required=True, choices=_C100_CASE_NAMES)
    parser.add_argument("--hop-mode", choices=_HOP_MODES, default="legacy")
    parser.add_argument("--two-hop-threshold-percent", type=int, default=0)
    parser.add_argument("--max-two-hop-percent", type=int, default=25)
    parser.add_argument("--hop-penalty-percent", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--steady-iters", type=int, default=100)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--master-port", type=int, default=30021)
    parser.add_argument("--watchdog-seconds", type=int, default=1800)
    parser.add_argument(
        "--nvtx", action="store_true",
        help=(
            "enable steady-only diagnostic NVTX ranges; reports are never "
            "eligible as profiler-free baselines"),
    )
    parser.add_argument(
        "--interference-mode",
        choices=_INTERFERENCE_MODES,
        default="none",
        help=(
            "default-off O080 diagnostic: none preserves the existing stage "
            "benchmark; compute-only times only a fixed BF16 GEMM; concurrent "
            "launches that GEMM on an independent stream before the stage API"),
    )
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE,
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-suite", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--launch-command-json", help=argparse.SUPPRESS)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    if not __debug__:
        parser.error("benchmark contract requires Python assertions")
    args = parser.parse_args(argv)
    if args.num_processes != _WORLD_SIZE:
        parser.error("benchmark requires exactly 8 processes")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be non-negative")
    if args.steady_iters <= 0:
        parser.error("--steady-iters must be positive")
    if not 0 <= args.two_hop_threshold_percent <= 10000:
        parser.error("--two-hop-threshold-percent must be in [0, 10000]")
    if not 0 <= args.max_two_hop_percent <= 100:
        parser.error("--max-two-hop-percent must be in [0, 100]")
    if not 0 <= args.hop_penalty_percent <= 10000:
        parser.error("--hop-penalty-percent must be in [0, 10000]")
    if args.timeout <= 0 or args.watchdog_seconds <= 0:
        parser.error("timeouts must be positive")
    if args.watchdog_seconds <= \
            args.timeout + _CONTROL_TIMEOUT_SLACK_SECONDS:
        parser.error(
            "--watchdog-seconds must exceed the GPU timeout plus the "
            f"{_CONTROL_TIMEOUT_SLACK_SECONDS}s control-plane slack")
    if not (1 <= args.master_port <= 65535):
        parser.error("--master-port must be in [1, 65535]")
    if args.launch_command_json is not None:
        try:
            command = json.loads(args.launch_command_json)
        except json.JSONDecodeError as error:
            parser.error(f"invalid internal launch command: {error}")
        if not isinstance(command, list) or not command or not all(
                isinstance(item, str) for item in command):
            parser.error("invalid internal launch command")
    if args.interference_mode != "none" and \
            not str(args.case_name).endswith("_h7168"):
        parser.error("--interference-mode requires an H7168 C100 case")
    return args


def _selected_case(case_name: str) -> ShuffleCase:
    matches = tuple(
        spec for spec in _assert_shuffle_oracle(include_c100_profiles=True)
        if spec.plan.name == case_name
    )
    assert len(matches) == 1
    spec = matches[0]
    schedule = _schedule(spec.plan)
    assert schedule.enabled and schedule.failure_reason is None
    assert schedule.moved_copies == spec.expected_moved_copies == 7_168
    assert sum(schedule.proxy_required) == schedule.moved_copies
    assert max(schedule.proxy_required) <= spec.plan.proxy_capacity_per_egress
    return spec


def _timed_window(
    function: Callable[[], _TYPE],
) -> tuple[_TYPE | None, int, str | None, int, int]:
    start = time.perf_counter_ns()
    value = None
    error = None
    try:
        value = function()
    except BaseException:
        error = traceback.format_exc()
    elapsed = time.perf_counter_ns() - start
    assert elapsed >= 0
    return value, elapsed, error, start, start + elapsed


def _timed(function: Callable[[], _TYPE]) -> tuple[_TYPE | None, int, str | None]:
    value, elapsed, error, _, _ = _timed_window(function)
    return value, elapsed, error


def _nvtx_wrapped(
    label: str,
    function: Callable[[], _TYPE],
) -> Callable[[], _TYPE]:
    def wrapped() -> _TYPE:
        torch.cuda.nvtx.range_push(label)
        try:
            return function()
        finally:
            torch.cuda.nvtx.range_pop()

    return wrapped


def _make_compute_interference_state(
    rank: int,
) -> ComputeInterferenceState:
    rows, inner, columns = _INTERFERENCE_COMPUTE_SHAPE
    device = torch.device("cuda", rank)
    stream = torch.cuda.Stream(device=device)
    lhs = torch.empty((rows, inner), dtype=torch.bfloat16, device=device)
    rhs = torch.empty((inner, columns), dtype=torch.bfloat16, device=device)
    output = torch.empty((rows, columns), dtype=torch.bfloat16, device=device)
    lhs.fill_(1)
    rhs.fill_(1)
    return ComputeInterferenceState(stream, lhs, rhs, output)


def _launch_compute_interference(
    state: ComputeInterferenceState,
    nvtx_label: str | None,
) -> None:
    def launch() -> None:
        with torch.cuda.stream(state.stream):
            torch.mm(state.lhs, state.rhs, out=state.output)

    if nvtx_label is None:
        launch()
    else:
        _nvtx_wrapped(nvtx_label, launch)()


def _compute_only_stage(
    state: ComputeInterferenceState,
    nvtx: bool,
) -> None:
    _launch_compute_interference(
        state,
        "c100/compute_interference/compute_only" if nvtx else None,
    )
    state.stream.synchronize()


def _concurrent_stage(
    stage_function: Callable[[], _TYPE],
    state: ComputeInterferenceState,
    nvtx: bool,
) -> _TYPE:
    _launch_compute_interference(
        state,
        "c100/compute_interference/concurrent_launch" if nvtx else None,
    )
    try:
        return stage_function()
    finally:
        state.stream.synchronize()


def _world_gate(
    label: str,
    payload: dict[str, object],
    control_group: dist.ProcessGroup,
) -> tuple[list[object], int]:
    start = time.perf_counter_ns()
    gathered = _gather_objects(payload, control_group)
    elapsed = time.perf_counter_ns() - start
    messages = []
    for rank, raw in enumerate(gathered):
        assert isinstance(raw, dict), (label, rank, raw)
        if raw.get("error") is not None:
            messages.append(f"rank {rank}:\n{raw['error']}")
        elif int(raw["status"]) != 0:
            messages.append(f"rank {rank}: status={raw['status']}")
    if messages:
        raise AssertionError(f"{label} failed:\n" + "\n".join(messages))
    return gathered, elapsed


def _finish_checked(
    runtime: object,
    invocation_id: int,
) -> tuple[torch.Tensor, ...]:
    outputs, status, error = _finish(runtime, invocation_id, None)
    if error is not None:
        raise RuntimeError(error)
    assert outputs is not None and status == 0
    return outputs


def _prepare_checked(
    runtime: object,
    topk_idx: torch.Tensor,
    spec: ShuffleCase,
    *,
    arena_offset: int,
    invocation_id: int,
    hop_mode: str,
    two_hop_threshold_percent: int,
    max_two_hop_percent: int,
    hop_penalty_percent: int,
) -> int:
    status, error = _prepare_case(
        runtime,
        topk_idx,
        spec.plan,
        spec.hidden,
        arena_offset=arena_offset,
        invocation_id=invocation_id,
        stream=None,
        hop_aware=hop_mode != "legacy",
        two_hop_threshold_percent=two_hop_threshold_percent,
        max_two_hop_percent=(
            max_two_hop_percent if hop_mode == "adaptive" else 0
        ),
        hop_penalty_percent=hop_penalty_percent,
    )
    if error is not None:
        raise RuntimeError(error)
    assert status == 0
    return status


def _return_stage(
    runtime: object,
    proxy_return: torch.Tensor,
    reduce_seed: torch.Tensor,
    invocation_id: int,
) -> torch.Tensor:
    # Call the runtime API directly.  The returned CUDA snapshot remains live
    # through abort, but is never copied to CPU by this measurement harness.
    snapshot = runtime._rail_balance_hybrid_return_unshuffle_test(  # type: ignore[attr-defined]
        proxy_return, reduce_seed, invocation_id)
    assert isinstance(snapshot, torch.Tensor)
    assert snapshot.is_cuda and snapshot.dtype == torch.uint8
    assert snapshot.is_contiguous() and snapshot.ndim == 2
    return snapshot


def _validate_prepare_gate(
    gathered: Sequence[object],
    spec: ShuffleCase,
) -> None:
    rows = tuple(item for item in gathered if isinstance(item, dict))
    assert len(rows) == _WORLD_SIZE
    assert len({tuple(row["signature"]) for row in rows}) == 1
    assert tuple(int(row["num_tokens"]) for row in rows) == \
        spec.plan.num_tokens_per_rank

def _run_iteration(
    *,
    runtime: object,
    rank: int,
    control_group: dist.ProcessGroup,
    control_timeout: int,
    stage: str,
    spec: ShuffleCase,
    topk_idx: torch.Tensor,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    proxy_return: torch.Tensor | None,
    reduce_seed: torch.Tensor | None,
    arena_offset: int,
    arena_bytes: int,
    invocation_id: int,
    category: str,
    category_index: int,
    logical_bytes_by_rank: Sequence[int],
    hop_mode: str,
    two_hop_threshold_percent: int,
    max_two_hop_percent: int,
    hop_penalty_percent: int,
    nvtx: bool,
    interference_mode: str,
    compute_state: ComputeInterferenceState | None,
) -> dict[str, object]:
    assert interference_mode in _INTERFERENCE_MODES
    assert (interference_mode == "none") == (compute_state is None)
    timings = {name: 0 for name in _ALL_PHASES}
    world_gate_ns: dict[str, int] = {}
    outputs: tuple[torch.Tensor, ...] | None = None
    stage_result: torch.Tensor | None = None
    phase_failure = None

    # Entry alignment is outside the transaction and target-stage timers.
    gate_start = time.perf_counter_ns()
    _monitored_barrier(control_group, control_timeout)
    world_gate_ns["entry"] = time.perf_counter_ns() - gate_start
    transaction_start = time.perf_counter_ns()

    try:
        prepare_function = lambda: _prepare_checked(
            runtime,
            topk_idx,
            spec,
            arena_offset=arena_offset,
            invocation_id=invocation_id,
            hop_mode=hop_mode,
            two_hop_threshold_percent=two_hop_threshold_percent,
            max_two_hop_percent=max_two_hop_percent,
            hop_penalty_percent=hop_penalty_percent,
        )
        if nvtx:
            prepare_function = _nvtx_wrapped(
                f"c100/{stage}/prepare", prepare_function)
        prepared, timings["prepare"], error = _timed(prepare_function)
        prepare_rows, world_gate_ns["prepare"] = _world_gate(
            "prepare WORLD gate",
            {
                "error": error,
                "status": -1 if prepared is None else int(prepared),
                "signature": list(_gate_signature(
                    spec.plan,
                    spec.hidden,
                    arena_offset=arena_offset,
                    arena_bytes=arena_bytes,
                )) + [
                    hop_mode,
                    two_hop_threshold_percent,
                    max_two_hop_percent if hop_mode == "adaptive" else 0,
                    hop_penalty_percent,
                ],
                "num_tokens": int(topk_idx.shape[0]),
            },
            control_group,
        )
        _validate_prepare_gate(prepare_rows, spec)

        finish_function = lambda: _finish_checked(runtime, invocation_id)
        if nvtx:
            finish_function = _nvtx_wrapped(
                f"c100/{stage}/finish", finish_function)
        outputs, timings["finish"], error = _timed(finish_function)
        _, world_gate_ns["finish"] = _world_gate(
            "finish WORLD gate",
            {
                "error": error,
                "status": -1 if outputs is None else 0,
            },
            control_group,
        )
        if stage == "return" and interference_mode != "compute-only":
            prerequisite_function = lambda: _source_shuffle(
                runtime, x, topk_weights, invocation_id, None)
            if nvtx:
                prerequisite_function = _nvtx_wrapped(
                    "c100/return/prerequisite_source",
                    prerequisite_function,
                )
            _, timings["prerequisite"], error = _timed(
                prerequisite_function)
            _, world_gate_ns["prerequisite"] = _world_gate(
                "return prerequisite WORLD gate",
                {"error": error, "status": 0 if error is None else -1},
                control_group,
            )
        # Keep the launch-start alignment separate from target stage time.
        gate_start = time.perf_counter_ns()
        _monitored_barrier(control_group, control_timeout)
        world_gate_ns["pre_stage"] = time.perf_counter_ns() - gate_start

        if stage == "source":
            adapter_function = lambda: _source_shuffle(
                runtime, x, topk_weights, invocation_id, None)
            if nvtx:
                adapter_function = _nvtx_wrapped(
                    f"c100/{stage}/stage/{category}/{category_index}",
                    adapter_function,
                )
                adapter_function = _nvtx_wrapped(
                    f"c100/{stage}/stage", adapter_function)
            if interference_mode == "compute-only":
                assert compute_state is not None
                stage_function = lambda: _compute_only_stage(
                    compute_state, nvtx)
            elif interference_mode == "concurrent":
                assert compute_state is not None
                stage_function = lambda: _concurrent_stage(
                    adapter_function, compute_state, nvtx)
            else:
                stage_function = adapter_function
            (_, timings["stage"], error,
             stage_start_ns, stage_end_ns) = _timed_window(
                stage_function)
        else:
            assert proxy_return is not None and reduce_seed is not None
            adapter_function = lambda: _return_stage(
                runtime, proxy_return, reduce_seed, invocation_id)
            if nvtx:
                adapter_function = _nvtx_wrapped(
                    f"c100/{stage}/stage/{category}/{category_index}",
                    adapter_function,
                )
                adapter_function = _nvtx_wrapped(
                    f"c100/{stage}/stage", adapter_function)
            if interference_mode == "compute-only":
                assert compute_state is not None
                stage_function = lambda: _compute_only_stage(
                    compute_state, nvtx)
            elif interference_mode == "concurrent":
                assert compute_state is not None
                stage_function = lambda: _concurrent_stage(
                    adapter_function, compute_state, nvtx)
            else:
                stage_function = adapter_function
            (stage_result, timings["stage"], error,
             stage_start_ns, stage_end_ns) = _timed_window(
                stage_function)
        _, world_gate_ns["stage"] = _world_gate(
            "stage completion WORLD gate",
            {"error": error, "status": 0 if error is None else -1},
            control_group,
        )
        if stage == "source" and interference_mode != "compute-only":
            visibility_function = lambda: runtime.barrier(  # type: ignore[attr-defined]
                True, True, True)
            if nvtx:
                visibility_function = _nvtx_wrapped(
                    "c100/source/post_visibility", visibility_function)
            _, timings["post_visibility"], error = _timed(
                visibility_function)
        else:
            # Return's measured API already includes its B4 LSA barrier and a
            # comm-stream synchronize.  A second device barrier would alter
            # the workload, so the separate phase is truthfully zero.
            error = None
            timings["post_visibility"] = 0
        _, world_gate_ns["post_visibility"] = _world_gate(
            "post-visibility WORLD gate",
            {"error": error, "status": 0 if error is None else -1},
            control_group,
        )
    except BaseException:
        phase_failure = traceback.format_exc()
    finally:
        abort_method = (
            runtime._rail_balance_hybrid_plan_abort  # type: ignore[attr-defined]
        )
        abort_function = lambda: abort_method(invocation_id)
        if nvtx:
            abort_function = _nvtx_wrapped(
                f"c100/{stage}/abort", abort_function)
        _, timings["abort"], abort_error = _timed(abort_function)
        try:
            _, world_gate_ns["abort"] = _world_gate(
                "abort WORLD gate",
                {
                    "error": abort_error,
                    "status": 0 if abort_error is None else -1,
                },
                control_group,
            )
            gate_start = time.perf_counter_ns()
            _monitored_barrier(control_group, control_timeout)
            world_gate_ns["exit"] = time.perf_counter_ns() - gate_start
        except BaseException:
            abort_trace = traceback.format_exc()
            phase_failure = (
                (phase_failure + "\n" if phase_failure is not None else "")
                + abort_trace
            )
        timings["transaction"] = time.perf_counter_ns() - transaction_start
        transaction_end_ns = transaction_start + timings["transaction"]

    if phase_failure is not None:
        raise AssertionError(
            f"transaction {invocation_id} failed after converged abort:\n"
            f"{phase_failure}")
    assert set(timings) == set(_ALL_PHASES)
    assert outputs is not None
    if stage == "return" and interference_mode != "compute-only":
        assert stage_result is not None and stage_result.is_cuda

    local = {
        "rank": rank,
        "interference_mode": interference_mode,
        "compute_stream_id": (
            None if compute_state is None else int(compute_state.stream.cuda_stream)
        ),
        "timings_ns": timings,
        "world_gate_ns": world_gate_ns,
        "stage_start_ns": stage_start_ns,
        "stage_end_ns": stage_end_ns,
        "transaction_start_ns": transaction_start,
        "transaction_end_ns": transaction_end_ns,
        "logical_bytes": int(logical_bytes_by_rank[rank]),
        "logical_bytes_per_second": (
            int(logical_bytes_by_rank[rank]) * 1e9 / timings["stage"]),
    }
    rank_rows = _gather_objects(local, control_group)
    rank_raw = sorted(
        (row for row in rank_rows if isinstance(row, dict)),
        key=lambda row: int(row["rank"]),
    )
    assert tuple(int(row["rank"]) for row in rank_raw) == \
        tuple(range(_WORLD_SIZE))
    rank_max_ns = {
        phase: max(int(row["timings_ns"][phase]) for row in rank_raw)
        for phase in _ALL_PHASES
    }
    rank_min_ns = {
        phase: min(int(row["timings_ns"][phase]) for row in rank_raw)
        for phase in _ALL_PHASES
    }
    rank_straggler_ratio = {
        phase: (
            1.0 if rank_max_ns[phase] == rank_min_ns[phase] == 0
            else None if rank_min_ns[phase] == 0
            else rank_max_ns[phase] / rank_min_ns[phase]
        )
        for phase in _ALL_PHASES
    }
    stage_global_span_ns = (
        max(int(row["stage_end_ns"]) for row in rank_raw) -
        min(int(row["stage_start_ns"]) for row in rank_raw)
    )
    transaction_global_span_ns = (
        max(int(row["transaction_end_ns"]) for row in rank_raw) -
        min(int(row["transaction_start_ns"]) for row in rank_raw)
    )
    assert stage_global_span_ns > 0 and transaction_global_span_ns > 0
    logical_bytes = sum(int(value) for value in logical_bytes_by_rank)
    return {
        "category": category,
        "category_index": category_index,
        "invocation_id": invocation_id,
        "rank_raw": rank_raw,
        "rank_min_ns": rank_min_ns,
        "rank_max_ns": rank_max_ns,
        "rank_straggler_ratio": rank_straggler_ratio,
        "stage_truth_ns": stage_global_span_ns,
        "step_truth_ns": transaction_global_span_ns,
        "stage_global_span_ns": stage_global_span_ns,
        "transaction_global_span_ns": transaction_global_span_ns,
        "stage_rank_max_duration_ns": rank_max_ns["stage"],
        "transaction_rank_max_duration_ns": rank_max_ns["transaction"],
        "logical_bytes": logical_bytes,
        "logical_bytes_per_second": (
            logical_bytes * 1e9 / stage_global_span_ns),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    assert values and 0.0 <= percentile <= 100.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    assert values
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
        "mean": mean,
        "std_population": std,
        "cv_population": 0.0 if mean == 0.0 else std / mean,
        "min": min(values),
        "max": max(values),
    }


def _movement_accounting(
    schedule: HybridRailSchedule,
    *,
    logical_token_bytes: int,
    stage: str,
) -> dict[str, object]:
    owner_to_egress = [
        [0 for _ in range(_WORLD_SIZE)] for _ in range(_WORLD_SIZE)
    ]
    for bucket in schedule.segments:
        for segment in bucket:
            owner_to_egress[int(segment.owner)][int(segment.egress)] += \
                int(segment.count)
    outgoing_counts = [sum(row) for row in owner_to_egress]
    incoming_counts = [
        sum(owner_to_egress[owner][egress] for owner in range(_WORLD_SIZE))
        for egress in range(_WORLD_SIZE)
    ]
    assert tuple(incoming_counts) == tuple(int(value)
                                           for value in schedule.proxy_required)
    assert sum(outgoing_counts) == sum(incoming_counts) == \
        int(schedule.moved_copies)
    selected_counts = outgoing_counts if stage == "source" else incoming_counts
    return {
        "owner_to_egress_moved_copies": owner_to_egress,
        "outgoing_moved_copies_per_rank": outgoing_counts,
        "incoming_moved_copies_per_egress": incoming_counts,
        "selected_moved_copies_per_rank": selected_counts,
        "outgoing_logical_bytes_per_rank": [
            value * logical_token_bytes for value in outgoing_counts
        ],
        "incoming_logical_bytes_per_egress": [
            value * logical_token_bytes for value in incoming_counts
        ],
        "selected_logical_bytes_per_rank": [
            value * logical_token_bytes for value in selected_counts
        ],
        "selected_logical_bytes_aggregate": (
            sum(selected_counts) * logical_token_bytes),
        "selected_rank_scope": (
            "source-owner outgoing moved records"
            if stage == "source"
            else "proxy-egress incoming moved records"),
    }


def _steady_summary(
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    assert records
    return {
        "step_truth_ns": _stats([
            float(record["step_truth_ns"]) for record in records]),
        "stage_truth_ns": _stats([
            float(record["stage_truth_ns"]) for record in records]),
        "logical_bytes_per_second": _stats([
            float(record["logical_bytes_per_second"])
            for record in records
        ]),
        "rank_max_phase_ns": {
            phase: _stats([
                float(record["rank_max_ns"][phase])
                for record in records
            ])
            for phase in _ALL_PHASES
        },
    }


def _git_manifest() -> dict[str, object]:
    def run(*args: str) -> str:
        result = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()

    try:
        commit = run("git", "rev-parse", "HEAD")
        status = run("git", "status", "--porcelain=v1",
                     "--untracked-files=all").splitlines()
        return {"commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as error:
        return {
            "commit": None,
            "dirty": None,
            "status": [],
            "error": repr(error),
        }


def _file_identity(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def _jit_cache_identity() -> dict[str, object]:
    root = Path(os.environ.get(
        "EP_JIT_CACHE_DIR", str(Path.home() / ".deep_ep"))).expanduser()
    cache = root.resolve() / "cache"
    artifacts = []
    if cache.is_dir():
        for directory in sorted(cache.iterdir()):
            if not directory.is_dir() or not directory.name.startswith(
                    _JIT_KERNEL_PREFIXES):
                continue
            for name in ("kernel.cu", "kernel.cubin", "kernel.ptx",
                         "kernel.sass"):
                path = directory / name
                if path.is_file():
                    identity = _file_identity(path)
                    identity["relative_path"] = str(path.relative_to(cache))
                    artifacts.append(identity)
    return {
        "root": str(root.resolve()),
        "cache_existed": cache.is_dir(),
        "artifacts": artifacts,
    }


def _artifact_identity() -> dict[str, object]:
    extension = Path(_C.__file__).resolve()
    sources = {
        relative: _file_identity(_REPOSITORY_ROOT / relative)
        for relative in _CODE_IDENTITY_PATHS
    }
    return {
        "git": _git_manifest(),
        "extension": _file_identity(extension),
        "loaded_libraries": _loaded_library_identities(),
        "sources": sources,
        "jit_cache": _jit_cache_identity(),
    }


def _loaded_library_identities() -> list[dict[str, object]]:
    paths = set()
    with Path("/proc/self/maps").open(encoding="utf-8") as handle:
        for line in handle:
            candidate = line.split()[-1]
            if candidate.startswith("/") and any(
                    marker in Path(candidate).name
                    for marker in ("libcudart.so", "libnccl.so")):
                paths.add(Path(candidate).resolve())
    return [_file_identity(path) for path in sorted(paths)]


def _capture_command(command: Sequence[str]) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        return {
            "argv": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "argv": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": repr(error),
        }


def _system_snapshot() -> dict[str, object]:
    cuda_home = deep_ep.find_cuda_home()
    nvcc_path = os.environ.get("EP_JIT_NVCC_COMPILER") or str(
        Path(cuda_home) / "bin" / "nvcc")
    process_table = _capture_command((
        "ps", "-eo", "pid=,ppid=,pgid=,cmd="))
    mps_rows = [
        line for line in str(process_table["stdout"]).splitlines()
        if ("nvidia-cuda-mps-control" in line or
            "nvidia-cuda-mps-server" in line)
    ]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_home": cuda_home,
        "nvcc_path": nvcc_path,
        "nvcc_version": _capture_command((nvcc_path, "--version")),
        "gpu_state": _capture_command((
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,pstate,"
            "temperature.gpu,power.draw,power.limit,clocks.current.sm,"
            "clocks.current.memory,clocks_throttle_reasons.active,"
            "mig.mode.current,compute_mode,"
            "clocks_throttle_reasons.gpu_idle,"
            "clocks_throttle_reasons.applications_clocks_setting,"
            "clocks_throttle_reasons.sw_power_cap,"
            "clocks_throttle_reasons.hw_slowdown,"
            "clocks_throttle_reasons.sw_thermal_slowdown,"
            "clocks_throttle_reasons.hw_thermal_slowdown,"
            "clocks_throttle_reasons.hw_power_brake_slowdown,"
            "clocks_throttle_reasons.sync_boost",
            "--format=csv,noheader,nounits",
        )),
        "compute_apps": _capture_command((
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )),
        "process_table": process_table,
        "mps_process_rows": mps_rows,
    }


def _compute_app_pids(snapshot: dict[str, object]) -> list[int]:
    command = snapshot["compute_apps"]
    assert isinstance(command, dict)
    if command.get("returncode") != 0:
        return []
    pids = []
    for line in str(command["stdout"]).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2:
            pids.append(int(fields[1]))
    return sorted(set(pids))


def _gpu_state_validation(
    pre_system: dict[str, object],
    post_system: dict[str, object],
) -> dict[str, object]:
    parsed = []
    reasons = []
    for label, snapshot in (("pre", pre_system), ("post", post_system)):
        command = snapshot["gpu_state"]
        assert isinstance(command, dict)
        rows = [
            [field.strip() for field in line.split(",")]
            for line in str(command["stdout"]).splitlines()
            if line.strip()
        ] if command.get("returncode") == 0 else []
        parsed.append(rows)
        if len(rows) != _WORLD_SIZE:
            reasons.append(f"{label}: expected 8 GPU rows, got {len(rows)}")
            continue
        if any(len(row) != 21 for row in rows):
            reasons.append(f"{label}: unexpected nvidia-smi column count")
            continue
        if {int(row[0]) for row in rows} != set(range(_WORLD_SIZE)):
            reasons.append(f"{label}: GPU indices are not 0..7")
        if len({row[1] for row in rows}) != _WORLD_SIZE:
            reasons.append(f"{label}: GPU UUIDs are not unique")
        if any(row[11] != "Disabled" for row in rows):
            reasons.append(f"{label}: MIG is not disabled on every GPU")
        if any(row[12] != "Default" for row in rows):
            reasons.append(f"{label}: compute mode is not Default")
        # GPU-idle and application-clock flags are descriptive. Power-cap,
        # hardware/thermal slowdown, power-brake, and sync-boost are treated
        # as non-benign for an automatically accepted baseline.
        if any(any(value != "Not Active" for value in row[15:21])
               for row in rows):
            reasons.append(f"{label}: non-benign clock throttle is active")
    if len(parsed) == 2 and all(len(rows) == _WORLD_SIZE for rows in parsed):
        if {row[1] for row in parsed[0]} != {row[1] for row in parsed[1]}:
            reasons.append("GPU UUID set changed during the run")
        if {row[3] for row in parsed[0]} != {row[3] for row in parsed[1]}:
            reasons.append("driver version changed during the run")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "pre_row_count": len(parsed[0]),
        "post_row_count": len(parsed[1]),
    }


def _environment() -> dict[str, str]:
    return {
        key: value for key, value in sorted(os.environ.items())
        if key in _ENV_EXACT or key.startswith(_ENV_PREFIXES)
    }


def _device_manifest(rank: int) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(rank)
    return {
        "rank": rank,
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
    }


def _atomic_json(path: Path, report: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(report, handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _build_report(
    *,
    args: argparse.Namespace,
    spec: ShuffleCase,
    layout: Sequence[int],
    arena_offset: int,
    base_bytes: int,
    logical_bytes: int,
    buffer_init_rows: Sequence[object],
    devices: Sequence[object],
    records: Sequence[dict[str, object]],
    pre_identity: dict[str, object],
    cold_jit_identity: dict[str, object],
    post_identity: dict[str, object],
    pre_system: dict[str, object],
    post_system: dict[str, object],
) -> dict[str, object]:
    cold = [record for record in records if record["category"] == "cold"]
    warm = [record for record in records if record["category"] == "warm"]
    steady = [
        record for record in records if record["category"] == "steady"]
    assert len(cold) == 1
    assert len(warm) == args.warmup_iters
    assert len(steady) == args.steady_iters
    schedule = _schedule(spec.plan)
    if args.launch_command_json is None:
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    else:
        command = json.loads(args.launch_command_json)
    post_visibility = (
        "separate runtime LSA barrier after the measured source adapter"
        if args.stage == "source"
        else "included in measured return API (B4); separate duration is 0"
    )
    prerequisite = (
        "none"
        if args.stage == "source"
        else "source adapter is timed separately before the return stage"
    )
    logical_token_bytes = int(
        layout[5] if args.stage == "source" else layout[7])
    movement = _movement_accounting(
        schedule,
        logical_token_bytes=logical_token_bytes,
        stage=args.stage,
    )
    created = datetime.now(timezone.utc)
    for key in ("git", "extension", "loaded_libraries", "sources"):
        assert pre_identity[key] == post_identity[key], key
    loaded_library_names = [
        Path(str(item["path"])).name
        for item in post_identity["loaded_libraries"]
    ]
    assert any("libnccl.so" in name for name in loaded_library_names)
    assert any("libcudart.so" in name for name in loaded_library_names)
    pre_jit = pre_identity["jit_cache"]
    post_jit = post_identity["jit_cache"]
    assert isinstance(pre_jit, dict) and isinstance(post_jit, dict)
    assert pre_jit["artifacts"] == []
    assert cold_jit_identity == post_jit
    cubin_directories = sorted({
        Path(str(artifact["relative_path"])).parts[0]
        for artifact in post_jit["artifacts"]
        if str(artifact["relative_path"]).endswith("/kernel.cubin")
    })
    if args.hop_mode == "legacy":
        mode_jit_prefixes = _LEGACY_SOURCE_JIT_KERNEL_PREFIX
    elif args.max_two_hop_percent == 0:
        mode_jit_prefixes = _HOP_ONE_HOP_JIT_KERNEL_PREFIXES
    else:
        mode_jit_prefixes = _HOP_ADAPTIVE_JIT_KERNEL_PREFIXES
    required_jit_prefixes = _COMMON_JIT_KERNEL_PREFIXES + mode_jit_prefixes
    for prefix in required_jit_prefixes:
        matches = [name for name in cubin_directories
                   if name.startswith(prefix)]
        assert len(matches) == 1, (prefix, matches)
    assert len(cubin_directories) == len(required_jit_prefixes)
    init_rows = sorted(
        (row for row in buffer_init_rows if isinstance(row, dict)),
        key=lambda row: int(row["rank"]),
    )
    assert tuple(int(row["rank"]) for row in init_rows) == \
        tuple(range(_WORLD_SIZE))
    init_values = [int(row["buffer_init_ns"]) for row in init_rows]
    init_min, init_max = min(init_values), max(init_values)
    shape = {
        "stage": args.stage,
        "case_name": spec.plan.name,
        "world_size": _WORLD_SIZE,
        "hidden": spec.hidden,
        "input_iteration": spec.iteration,
        "init_dist_seed": 100,
        "remainder_seed": spec.plan.remainder_seed,
        "tokens_per_rank": list(spec.plan.num_tokens_per_rank),
        "num_topk": spec.plan.num_topk,
        "num_channels": spec.plan.num_channels,
        "num_experts": spec.plan.num_experts,
        "num_destinations": spec.plan.num_scaleout_ranks,
        "local_destination": spec.plan.local_scaleout_rank,
        "proxy_capacity_per_egress":
            spec.plan.proxy_capacity_per_egress,
        "hop_mode": args.hop_mode,
        "two_hop_threshold_percent": args.two_hop_threshold_percent,
        "max_two_hop_percent": (
            args.max_two_hop_percent if args.hop_mode == "adaptive" else 0
        ),
        "hop_penalty_percent": args.hop_penalty_percent,
        "moved_copies_global": schedule.moved_copies,
        "proxy_required_per_egress": list(schedule.proxy_required),
        "owner_to_egress_moved_copies":
            movement["owner_to_egress_moved_copies"],
        "outgoing_moved_copies_per_rank":
            movement["outgoing_moved_copies_per_rank"],
        "incoming_moved_copies_per_egress":
            movement["incoming_moved_copies_per_egress"],
        "dtype": "bfloat16",
        "interference_mode": args.interference_mode,
        "interference_compute_shape": list(_INTERFERENCE_COMPUTE_SHAPE),
    }
    layout_manifest = {
        **dict(zip(_LAYOUT_LABELS, map(int, layout))),
        "legacy_base_bytes": base_bytes,
        "arena_offset": arena_offset,
        "total_buffer_bytes": arena_offset + int(layout[-1]),
    }
    full_environment = _environment()
    semantic_environment = {
        key: value for key, value in full_environment.items()
        if key not in {"EP_JIT_CACHE_DIR", _WATCHDOG_CHILD_ENV}
    }
    semantic_devices = [
        {key: value for key, value in row.items()
         if key not in {"pid", "pgid"}}
        for row in devices if isinstance(row, dict)
    ]
    assert len(semantic_devices) == _WORLD_SIZE
    semantic_config = {
        "shape": shape,
        "layout": layout_manifest,
        "diagnostic_nvtx": args.nvtx,
        "warmup_iterations": args.warmup_iters,
        "steady_iterations": args.steady_iters,
        "gpu_timeout_seconds": args.timeout,
        "control_timeout_seconds": (
            args.timeout + _CONTROL_TIMEOUT_SLACK_SECONDS),
        "environment": semantic_environment,
        "devices": semantic_devices,
    }
    semantic_config_hash = hashlib.sha256(json.dumps(
        semantic_config, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    execution_config = {
        "semantic_config_sha256": semantic_config_hash,
        "environment": full_environment,
        "command": command,
        "jit_cache_root": post_jit["root"],
    }
    execution_config_hash = hashlib.sha256(json.dumps(
        execution_config, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    benchmark_pids = sorted(int(row["pid"]) for row in devices)
    pre_app_pids = _compute_app_pids(pre_system)
    post_app_pids = _compute_app_pids(post_system)
    unexpected_app_pids = sorted(
        (set(pre_app_pids) | set(post_app_pids)) - set(benchmark_pids))
    gpu_state_validation = _gpu_state_validation(pre_system, post_system)
    system_commands_ok = all(
        snapshot[name].get("returncode") == 0
        for snapshot in (pre_system, post_system)
        for name in (
            "nvcc_version", "gpu_state", "compute_apps", "process_table")
    ) and bool(gpu_state_validation["valid"])
    no_mps = not pre_system["mps_process_rows"] and \
        not post_system["mps_process_rows"]
    git_clean = post_identity["git"].get("dirty") is False
    measurement_depth_ok = (
        args.warmup_iters >= _FORMAL_MIN_WARMUP_ITERATIONS and
        args.steady_iters >= _FORMAL_MIN_STEADY_ITERATIONS
    )
    persistent_report_requested = args.json_out is not None
    baseline_collection_eligible = (
        git_clean and system_commands_ok and no_mps and
        not unexpected_app_pids and measurement_depth_ok and
        persistent_report_requested and not args.nvtx and
        args.interference_mode == "none"
    )
    selected_logical_bytes_by_rank = list(
        movement["selected_logical_bytes_per_rank"])
    selected_logical_bytes_aggregate = int(
        movement["selected_logical_bytes_aggregate"])
    logical_numerator_available = (
        args.interference_mode != "compute-only" and args.hop_mode == "legacy"
    )
    if not logical_numerator_available:
        logical_bytes_by_rank = [0] * _WORLD_SIZE
        assert logical_bytes == 0
    else:
        logical_bytes_by_rank = selected_logical_bytes_by_rank
        assert logical_bytes == selected_logical_bytes_aggregate
    traffic_accounting: dict[str, object] = {
        "path_accounting_scope": (
            "measured legacy plan" if args.hop_mode == "legacy" else
            "legacy reference only; hop counters are not retained"
        ),
        "logical_numerator_scope": (
            "no moved-record numerator; target window is the fixed BF16 GEMM"
            if args.interference_mode == "compute-only" else
            "unavailable until measured hop-plan path counters are retained"
            if args.hop_mode != "legacy" else
            "moved TokenLayout record bytes counted once, aggregated across "
            "all 8 ranks; not physical link or memory-controller traffic"),
        "stage_rank_scope": movement["selected_rank_scope"],
        "logical_bytes_per_rank": logical_bytes_by_rank,
        "logical_bytes_aggregate": logical_bytes,
        "stage_logical_bytes_per_rank": logical_bytes_by_rank,
        "stage_logical_bytes_aggregate": logical_bytes,
        "legacy_reference_logical_bytes_per_rank":
            selected_logical_bytes_by_rank,
        "legacy_reference_logical_bytes_aggregate":
            selected_logical_bytes_aggregate,
        "owner_to_egress_moved_copies":
            movement["owner_to_egress_moved_copies"],
        "outgoing_moved_copies_per_rank":
            movement["outgoing_moved_copies_per_rank"],
        "incoming_moved_copies_per_egress":
            movement["incoming_moved_copies_per_egress"],
        "selected_moved_copies_per_rank":
            movement["selected_moved_copies_per_rank"],
        "outgoing_logical_bytes_per_rank":
            movement["outgoing_logical_bytes_per_rank"],
        "incoming_logical_bytes_per_egress":
            movement["incoming_logical_bytes_per_egress"],
    }
    if args.stage == "return":
        num_reduce_rows = min(
            spec.plan.num_scaleout_ranks, spec.plan.num_topk)
        reduce_copy_bytes = (
            num_reduce_rows * spec.plan.num_max_tokens_per_rank *
            logical_token_bytes
        )
        proxy_copy_bytes = (
            spec.plan.proxy_capacity_per_egress * logical_token_bytes)
        traffic_accounting["checked_wrapper_extra_d2d_bytes_per_rank"] = {
            "proxy_input_copy": proxy_copy_bytes,
            "reduce_seed_copy": reduce_copy_bytes,
            "reduce_snapshot_copy": reduce_copy_bytes,
        }
        traffic_accounting["return_numerator_excludes"] = (
            "full reduce-seed and snapshot copies, status D2H, barrier "
            "traffic, and any physical peer/HBM transaction amplification")
    return {
        "schema_version": 4,
        "run_id": (
            f"{created.strftime('%Y%m%dT%H%M%S.%fZ')}-"
            f"{args.stage}-{spec.plan.name}-p{os.getpid()}"),
        "claim_scope": "checked_adapter_only",
        "claim_exclusions": [
            "not public Hybrid dispatch/combine end-to-end",
            "not Gin/RDMA/NIC performance",
            "not a production speedup claim",
        ],
        "created_utc": created.isoformat(),
        "git": post_identity["git"],
        "identity": {
            "semantic_config": semantic_config,
            "semantic_config_sha256": semantic_config_hash,
            "execution_config_sha256": execution_config_hash,
            "code_identity_stable_during_run": True,
            "jit_cache_empty_before_run": True,
            "jit_identity_stable_after_cold": True,
            "required_jit_cubin_directories": cubin_directories,
            "reproduction_limit": (
                "the extension and loaded CUDA/NCCL libraries have exact "
                "binary hashes, but historical extension compiler flags "
                "cannot be reconstructed from an already-built binary"),
            "pre_measurement": pre_identity,
            "after_cold_transaction_jit": cold_jit_identity,
            "post_measurement": post_identity,
        },
        "command": {
            "argv": command,
            "shell": shlex.join(command),
            "cwd": str(Path.cwd().resolve()),
        },
        "environment": {
            "variables": full_environment,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nccl": list(torch.cuda.nccl.version()),
            "devices": list(devices),
            "runtime_state": {
                "pre_measurement": pre_system,
                "post_measurement": post_system,
                "benchmark_pids": benchmark_pids,
                "unexpected_compute_app_pids": unexpected_app_pids,
                "system_commands_ok": system_commands_ok,
                "mps_processes_absent": no_mps,
                "gpu_state_validation": gpu_state_validation,
                "sampling_limit": (
                    "pre/post snapshots can miss a transient mid-run process, "
                    "clock, power, or thermal event; accepted runs still "
                    "require stable raw timing distributions"),
            },
        },
        "shape": shape,
        "layout": layout_manifest,
        "traffic_accounting": traffic_accounting,
        "measurement": {
            "clock": (
                "time.perf_counter_ns common monotonic host clock; all ranks "
                "are processes on this single node"),
            "profiler_mode_declared": (
                "nvtx_diagnostic" if args.nvtx else "disabled"),
            "profiler_detection": (
                "explicit --nvtx diagnostic mode; never baseline eligible"
                if args.nvtx else
                "not auto-detected; only a direct unwrapped invocation may "
                "be accepted as a profiler-free baseline"),
            "cold_iterations": 1,
            "cold_definition": (
                "first transaction after CUDA context, ElasticBuffer, input "
                "allocation, input transfer, and one LSA barrier warmup; it "
                "retains first-use target adapter/JIT cost in timed prepare; "
                "the separately reported stage does not include that JIT"),
            "warmup_iterations": args.warmup_iters,
            "steady_iterations": args.steady_iters,
            "steady_summary_only": True,
            "warm_iterations_retained_unaggregated": True,
            "step_truth": (
                "max(transaction_end)-min(transaction_start) across 8 ranks"),
            "stage_truth": (
                "max(stage_end)-min(stage_start) across 8 ranks"),
            "percentile_method": "linear interpolation over sorted samples",
            "std_method": "population",
            "logical_bytes_per_iteration": logical_bytes,
            "logical_token_bytes": logical_token_bytes,
            "logical_bandwidth_denominator": (
                "same-node 8-rank global target-stage span"),
            "logical_bandwidth_scope": (
                "aggregate logical checked-adapter rate across 8 ranks; not "
                "raw-kernel throughput and not physical NVLink/HBM bandwidth"),
            "interference_mode": args.interference_mode,
            "interference_compute_shape": list(_INTERFERENCE_COMPUTE_SHAPE),
            "interference_stream_identity": (
                "per-rank compute_stream_id is stored in raw iteration records"
                if args.interference_mode != "none" else "not applicable"),
            "interference_scope": (
                "disabled; target window is the checked adapter only"
                if args.interference_mode == "none" else
                "diagnostic O080 mode; target window is compute-only or "
                "adapter plus independent-stream GEMM and is never baseline "
                "eligible without separate Nsys overlap evidence"),
            "gpu_timeout_seconds": args.timeout,
            "control_timeout_seconds": (
                args.timeout + _CONTROL_TIMEOUT_SLACK_SECONDS),
            "buffer_and_inputs_reused": True,
            "plan_outputs_allocated_by_runtime_per_transaction": True,
            "return_snapshot_allocated_by_runtime_per_iteration":
                args.stage == "return",
            "unique_invocation_per_iteration": True,
            "extra_synchronize_inside_stage": False,
            "nvtx": (
                "rank0 c100_nsys_window covers all steady iterations; each "
                "rank marks coarse phases and exact target-stage ordinals"
                if args.nvtx else
                "disabled; any later NVTX/Nsys capture is a separate "
                "diagnostic mode, never this profiler-free baseline"),
            "baseline_collection_eligible": baseline_collection_eligible,
            "collection_eligibility_scope": (
                "automatic precondition gate only; accepting a performance "
                "baseline additionally requires correctness evidence and "
                "post-run review of raw median/p95/p99/std/CV stability"),
            "measurement_depth_ok": measurement_depth_ok,
            "persistent_report_requested": persistent_report_requested,
            "baseline_collection_requirements": (
                "clean Git, isolated empty JIT cache, stable required cubins, "
                f"at least {_FORMAL_MIN_WARMUP_ITERATIONS} warmup and "
                f"{_FORMAL_MIN_STEADY_ITERATIONS} steady samples, persistent "
                "JSON, direct unwrapped execution, diagnostic NVTX disabled, "
                "valid 8-GPU pre/post state without non-benign throttle, no "
                "MPS, and no compute process outside the eight benchmark "
                "workers, and interference_mode=none"),
            "cold_jit_identity_barrier": (
                "two control-plane barriers and rank0 JIT hashing occur after "
                "the cold transaction and outside every recorded sample"),
        },
        "setup": {
            "buffer_init_scope": "ElasticBuffer constructor only",
            "buffer_init_rank_raw": init_rows,
            "buffer_init_rank_min_ns": init_min,
            "buffer_init_rank_max_ns": init_max,
            "buffer_init_straggler_ratio": (
                None if init_min == 0 else init_max / init_min),
        },
        "internal_synchronization": {
            "prepare": "device status D2H plus comm-stream synchronize",
            "finish": (
                "one node-local plan barrier, plan/prefix work, device status "
                "D2H, comm-stream synchronize, then Python Tensor.item() "
                "status materialization"),
            "prerequisite": prerequisite,
            "source_stage": (
                "compute-to-comm stream dependency, device status D2H, "
                "comm-stream synchronize"),
            "return_stage": (
                "proxy/seed D2D copies, B1 and B4 node-local barriers, "
                "return-unshuffle, CUDA snapshot D2D, device status D2H, "
                "comm-stream synchronize"),
            "post_visibility": post_visibility,
            "world_gates": (
                "Gloo all-gather after prepare/finish/prerequisite/stage/"
                "post-visibility/abort; monitored barriers align stage entry "
                "and transaction exit; all are outside target stage time"),
            "abort": "transaction-specific runtime abort and stream quiescence",
        },
        "cold": cold[0],
        "warm": warm,
        "steady": steady,
        "steady_summary": _steady_summary(steady),
    }


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int,
            args: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(
        local_rank, num_local_ranks, seed=100)
    control_timeout = args.timeout + _CONTROL_TIMEOUT_SLACK_SECONDS
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(seconds=control_timeout),
    )
    assert rank == local_rank and world_size == _WORLD_SIZE
    buffer = None
    clean_shutdown = False
    report = None
    nvtx_window_started = False
    pre_identity = _artifact_identity() if rank == 0 else None
    pre_system = _system_snapshot() if rank == 0 else None
    cold_jit_identity = None
    try:
        spec = _selected_case(args.case_name)
        if args.hop_mode != "legacy":
            hop_capacity = max(
                spec.plan.proxy_capacity_per_egress,
                spec.plan.num_max_tokens_per_rank * spec.plan.num_topk,
            )
            spec = replace(
                spec,
                plan=replace(
                    spec.plan, proxy_capacity_per_egress=hop_capacity
                ),
            )
        case = spec.plan
        layout = tuple(int(value) for value in
                       _C._get_rail_balance_hybrid_layout(
                           spec.hidden, case.num_topk,
                           case.proxy_capacity_per_egress))
        assert len(layout) == len(_LAYOUT_LABELS)
        alignment = int(_C.get_elastic_buffer_alignment())
        base_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
            ep_group,
            num_max_tokens_per_rank=case.num_max_tokens_per_rank,
            hidden=spec.hidden,
            num_topk=case.num_topk,
            use_fp8_dispatch=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
        )
        arena_offset = align(base_bytes, alignment)
        assert arena_offset == base_bytes
        buffer_init_start = time.perf_counter_ns()
        buffer = deep_ep.ElasticBuffer(
            ep_group,
            num_bytes=arena_offset + layout[-1],
            num_max_tokens_per_rank=case.num_max_tokens_per_rank,
            hidden=spec.hidden,
            num_topk=case.num_topk,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
            prefer_overlap_with_compute=False,
            explicitly_destroy=True,
            num_gpu_timeout_secs=args.timeout,
            num_cpu_timeout_secs=args.timeout,
        )
        buffer_init_ns = time.perf_counter_ns() - buffer_init_start
        assert buffer.get_logical_domain_size() == (1, _WORLD_SIZE)
        assert buffer.scaleout_rank_idx == 0
        assert buffer.scaleup_rank_idx == rank
        runtime = buffer.runtime

        # All workload tensors are allocated and populated once, before cold.
        topk_idx = _local_topk(case, rank)
        x, topk_weights = _local_source_inputs(
            case, rank, spec.iteration, spec.hidden)
        compute_state = (
            None if args.interference_mode == "none" else
            _make_compute_interference_state(rank)
        )
        proxy_return = None
        reduce_seed = None
        if args.stage == "return":
            proxy_return = _proxy_return_cpu(spec, rank).to(
                torch.device("cuda", rank))
            num_reduce_rows = (
                case.num_scaleout_ranks
                if case.num_scaleout_ranks <= case.num_topk
                else case.num_topk
            )
            reduce_seed = torch.full(
                (num_reduce_rows * case.num_max_tokens_per_rank,
                 _combine_token_bytes(spec.hidden, case.num_topk)),
                _POISON,
                dtype=torch.uint8,
                device=torch.device("cuda", rank),
            )
        torch.cuda.synchronize(rank)
        _monitored_barrier(control_group, control_timeout)
        runtime.barrier(True, True, True)  # type: ignore[attr-defined]
        _monitored_barrier(control_group, control_timeout)

        devices = _gather_objects(_device_manifest(rank), control_group)
        logical_token_bytes = layout[
            5 if args.stage == "source" else 7]
        schedule = _schedule(case)
        movement = _movement_accounting(
            schedule,
            logical_token_bytes=logical_token_bytes,
            stage=args.stage,
        )
        stage_logical_bytes_by_rank = tuple(
            int(value)
            for value in movement["selected_logical_bytes_per_rank"]
        )
        stage_logical_bytes = int(movement["selected_logical_bytes_aggregate"])
        assert stage_logical_bytes == \
            spec.expected_moved_copies * logical_token_bytes
        if args.interference_mode == "compute-only" or \
                args.hop_mode != "legacy":
            logical_bytes_by_rank = (0,) * _WORLD_SIZE
            logical_bytes = 0
        else:
            logical_bytes_by_rank = stage_logical_bytes_by_rank
            logical_bytes = stage_logical_bytes
        buffer_init_rows = _gather_objects({
            "rank": rank,
            "buffer_init_ns": buffer_init_ns,
        }, control_group)

        records = []
        total = 1 + args.warmup_iters + args.steady_iters
        for ordinal in range(total):
            if ordinal == 0:
                category, category_index = "cold", 0
            elif ordinal <= args.warmup_iters:
                category, category_index = "warm", ordinal - 1
            else:
                category, category_index = (
                    "steady", ordinal - 1 - args.warmup_iters)
            if args.nvtx and category == "steady" and category_index == 0:
                _monitored_barrier(control_group, control_timeout)
                nvtx_start_error = None
                if rank == 0:
                    try:
                        torch.cuda.nvtx.range_push("c100_nsys_window")
                        nvtx_window_started = True
                    except BaseException:
                        nvtx_start_error = traceback.format_exc()
                _world_gate(
                    "NVTX capture start WORLD gate",
                    {
                        "error": nvtx_start_error,
                        "status": 0 if nvtx_start_error is None else -1,
                    },
                    control_group,
                )
                nvtx_window_started = True
            records.append(_run_iteration(
                runtime=runtime,
                rank=rank,
                control_group=control_group,
                control_timeout=control_timeout,
                stage=args.stage,
                spec=spec,
                topk_idx=topk_idx,
                x=x,
                topk_weights=topk_weights,
                proxy_return=proxy_return,
                reduce_seed=reduce_seed,
                arena_offset=arena_offset,
                arena_bytes=layout[-1],
                invocation_id=100_000 + ordinal,
                category=category,
                category_index=category_index,
                logical_bytes_by_rank=logical_bytes_by_rank,
                hop_mode=args.hop_mode,
                two_hop_threshold_percent=args.two_hop_threshold_percent,
                max_two_hop_percent=args.max_two_hop_percent,
                hop_penalty_percent=args.hop_penalty_percent,
                nvtx=args.nvtx,
                interference_mode=args.interference_mode,
                compute_state=compute_state,
            ))
            if ordinal == 0:
                _monitored_barrier(control_group, control_timeout)
                cold_jit_error = None
                if rank == 0:
                    try:
                        cold_jit_identity = _jit_cache_identity()
                    except BaseException:
                        cold_jit_error = traceback.format_exc()
                cold_jit_errors = _gather_objects(
                    cold_jit_error, control_group)
                messages = [
                    f"rank {index}:\n{error}"
                    for index, error in enumerate(cold_jit_errors)
                    if error is not None
                ]
                if messages:
                    raise AssertionError(
                        "cold JIT identity failed:\n" + "\n".join(messages))

        if args.nvtx:
            assert nvtx_window_started
            _monitored_barrier(control_group, control_timeout)
            nvtx_stop_error = None
            if rank == 0:
                try:
                    torch.cuda.nvtx.range_pop()
                    nvtx_window_started = False
                except BaseException:
                    nvtx_stop_error = traceback.format_exc()
            _world_gate(
                "NVTX capture stop WORLD gate",
                {
                    "error": nvtx_stop_error,
                    "status": 0 if nvtx_stop_error is None else -1,
                },
                control_group,
            )
            nvtx_window_started = False

        clean_shutdown = True
    finally:
        if rank == 0 and nvtx_window_started:
            try:
                torch.cuda.nvtx.range_pop()
            except BaseException:
                pass
            nvtx_window_started = False
        if clean_shutdown and buffer is not None:
            buffer.destroy()
            buffer = None
            _monitored_barrier(control_group, control_timeout)

            write_error = None
            if rank == 0:
                try:
                    assert pre_identity is not None and pre_system is not None
                    assert cold_jit_identity is not None
                    post_identity = _artifact_identity()
                    post_system = _system_snapshot()
                    report = _build_report(
                        args=args,
                        spec=spec,
                        layout=layout,
                        arena_offset=arena_offset,
                        base_bytes=base_bytes,
                        logical_bytes=logical_bytes,
                        buffer_init_rows=buffer_init_rows,
                        devices=devices,
                        records=records,
                        pre_identity=pre_identity,
                        cold_jit_identity=cold_jit_identity,
                        post_identity=post_identity,
                        pre_system=pre_system,
                        post_system=post_system,
                    )
                    if args.json_out is not None:
                        _atomic_json(args.json_out, report)
                except BaseException:
                    write_error = traceback.format_exc()
            write_errors = _gather_objects(write_error, control_group)
            messages = [
                f"rank {index}:\n{error}"
                for index, error in enumerate(write_errors)
                if error is not None
            ]
            if messages:
                dist.destroy_process_group()
                raise AssertionError(
                    "report build/write failed:\n" + "\n".join(messages))
            _monitored_barrier(control_group, control_timeout)
            if rank == 0:
                assert report is not None
                summary = report["steady_summary"]
                stage_stats = summary["stage_truth_ns"]
                bandwidth = summary["logical_bytes_per_second"]
                bandwidth_text = (
                    "aggregate logical median=n/a, "
                    if args.interference_mode == "compute-only" or
                    args.hop_mode != "legacy" else
                    "aggregate logical median="
                    f"{bandwidth['median'] / 1e9:.3f} GB/s, "
                )
                print(
                    "PASS C100 checked-adapter benchmark: "
                    f"stage={args.stage}, case={args.case_name}, "
                    f"interference_mode={args.interference_mode}, "
                    f"steady={args.steady_iters}, "
                    f"stage median={stage_stats['median'] / 1e3:.3f} us, "
                    f"p95={stage_stats['p95'] / 1e3:.3f} us, "
                    f"{bandwidth_text}"
                    "not physical NVLink/HBM bandwidth, "
                    "claim_scope=checked_adapter_only, "
                    "baseline_collection_eligible="
                    f"{report['measurement']['baseline_collection_eligible']}",
                    flush=True,
                )
            dist.destroy_process_group()


def _runtime_preflight(parser: argparse.ArgumentParser) -> None:
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
            "rebuild the extension with the C100 private APIs; missing "
            + ", ".join(missing))
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("benchmark requires at least 8 visible CUDA devices")


def _run_watchdog(
    args: argparse.Namespace,
    launch_command: Sequence[str],
) -> None:
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker-suite",
        "--stage", args.stage,
        "--case-name", args.case_name,
        "--hop-mode", args.hop_mode,
        "--two-hop-threshold-percent", str(args.two_hop_threshold_percent),
        "--max-two-hop-percent", str(args.max_two_hop_percent),
        "--hop-penalty-percent", str(args.hop_penalty_percent),
        "--warmup-iters", str(args.warmup_iters),
        "--steady-iters", str(args.steady_iters),
        "--timeout", str(args.timeout),
        "--master-port", str(args.master_port),
        "--watchdog-seconds", str(args.watchdog_seconds),
        "--num-processes", str(args.num_processes),
        "--interference-mode", args.interference_mode,
        "--launch-command-json", json.dumps(list(launch_command)),
    ]
    if args.json_out is not None:
        command.extend(("--json-out", str(args.json_out)))
    if args.nvtx:
        command.append("--nvtx")
    child_environment = os.environ.copy()
    child_environment[_WATCHDOG_CHILD_ENV] = "1"
    process: subprocess.Popen[bytes] | None = None
    process_group_id: int | None = None
    cleanup_signals = (
        signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)

    def process_group_exists() -> bool:
        if process_group_id is None:
            return False
        try:
            os.killpg(process_group_id, 0)
            return True
        except ProcessLookupError:
            return False

    def stop_process_group() -> None:
        if process is None:
            return
        cleanup_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, cleanup_signals)
        cleanup_failure = None
        try:
            if process_group_exists():
                try:
                    assert process_group_id is not None
                    os.killpg(process_group_id, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 10
            while process_group_exists() and time.monotonic() < deadline:
                process.poll()
                time.sleep(0.1)
            if process_group_exists():
                try:
                    assert process_group_id is not None
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 10
                while (process_group_exists() and
                       time.monotonic() < deadline):
                    process.poll()
                    time.sleep(0.1)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                cleanup_failure = "watchdog group leader was not reaped"
            if process_group_exists():
                cleanup_failure = (
                    "watchdog process group survived SIGTERM and SIGKILL")
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, cleanup_mask)
        if cleanup_failure is not None:
            raise RuntimeError(cleanup_failure)

    def interrupt_parent(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received parent signal {signum}")

    spawn_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, cleanup_signals)
    signals_unblocked = False
    previous_handlers = {}
    try:
        for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGQUIT):
            previous_handlers[signum] = signal.signal(
                signum, interrupt_parent)
        try:
            process = subprocess.Popen(
                command, start_new_session=True, env=child_environment)
            process_group_id = process.pid
            signals_unblocked = True
            signal.pthread_sigmask(signal.SIG_SETMASK, spawn_mask)
            return_code = process.wait(timeout=args.watchdog_seconds)
        except subprocess.TimeoutExpired as error:
            stop_process_group()
            raise RuntimeError(
                "C100 benchmark watchdog expired after "
                f"{args.watchdog_seconds}s") from error
        except BaseException:
            stop_process_group()
            raise
        if return_code:
            stop_process_group()
            raise SystemExit(return_code)
        # A successful group leader should have reaped every worker. Probe the
        # PGID anyway so an unexpected orphan cannot silently retain a GPU.
        stop_process_group()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if not signals_unblocked:
            signal.pthread_sigmask(signal.SIG_SETMASK, spawn_mask)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    parser = _parser()
    _runtime_preflight(parser)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ.setdefault("EP_DISABLE_GIN", "1")
    if not args.worker_suite:
        if "EP_JIT_CACHE_DIR" not in os.environ:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            os.environ["EP_JIT_CACHE_DIR"] = str((
                _REPOSITORY_ROOT / ".cache" / "rail_balance" / "c100" /
                "jit" / f"{stamp}-p{os.getpid()}"
            ).resolve())
        requested = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            *(sys.argv[1:] if argv is None else argv),
        ]
        _run_watchdog(args, requested)
        return
    if os.environ.get(_WATCHDOG_CHILD_ENV) != "1":
        parser.error("--worker-suite is an internal watchdog child mode")
    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
        join=True,
    )


if __name__ == "__main__":
    main()
