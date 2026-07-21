"""C080-D: bounded Hybrid source-shuffle fault and recovery checks."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import align
from test_rail_balance_hybrid_plan_lsa import (
    _abort,
    _finish,
    _gather_objects,
    _local_topk,
    _monitored_barrier,
    _prepare,
    _schedule,
    _skew_case,
)
from rail_balance_hybrid_reference import (
    enumerate_resolved_copies,
    map_topk_experts_to_destinations,
)


_WORLD_SIZE = 8
_HIDDEN = 256
_MAX_TOKENS = 10
_NUM_TOPK = 4
_PROXY_CAPACITY = 8
_OPERATION = "rail_balance_hybrid_source_shuffle_faults"


def _capture(function) -> str | None:
    try:
        function()
    except BaseException:
        return traceback.format_exc()
    return None


def _local_source_inputs(case, rank: int,
                         iteration: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = len(case.topk_idx[rank])
    x = (torch.arange(num_tokens * _HIDDEN, dtype=torch.int32)
         .reshape(num_tokens, _HIDDEN) % 29 + rank * 31 + iteration)
    token = torch.arange(num_tokens, dtype=torch.float32)[:, None]
    lane = torch.arange(case.num_topk, dtype=torch.float32)[None, :]
    weights = iteration * 10000.0 + rank * 1000.0 + token * 32.0 + lane
    device = torch.device("cuda", rank)
    return x.to(torch.bfloat16).contiguous().to(device), \
        weights.contiguous().to(device)


def _moved_owners(case) -> set[int]:
    destinations = map_topk_experts_to_destinations(
        case.topk_idx,
        num_topk=case.num_topk,
        num_experts=case.num_experts,
        num_scaleout_ranks=case.num_scaleout_ranks,
        local_scaleout_rank=case.local_scaleout_rank,
    )
    return {
        record.copy.owner
        for record in enumerate_resolved_copies(destinations, _schedule(case))
        if record.resolution.moved
    }


def _prepare_finish(
    runtime: object,
    rank: int,
    control_group: dist.ProcessGroup,
    case,
    arena_offset: int,
    invocation_id: int,
    expected_finish_status: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    topk_idx = _local_topk(case, rank)
    status, error = _prepare(
        runtime, topk_idx, case,
        arena_offset=arena_offset,
        invocation_id=invocation_id,
        stream=None,
    )
    gate1 = _gather_objects(
        (_OPERATION, 1, invocation_id, status, error), control_group)
    assert tuple(int(item[3]) for item in gate1) == (0,) * _WORLD_SIZE
    assert all(item[4] is None for item in gate1)

    outputs, finish_status, finish_error = _finish(
        runtime, invocation_id, None)
    gate2 = _gather_objects(
        (_OPERATION, 2, invocation_id, finish_status, finish_error),
        control_group,
    )
    assert tuple(int(item[3]) for item in gate2) == \
        (expected_finish_status,) * _WORLD_SIZE
    assert all(item[4] is None for item in gate2)
    assert outputs is not None
    return topk_idx, outputs


def _run_corrupt_plan(
    *,
    runtime: object,
    rank: int,
    control_group: dist.ProcessGroup,
    timeout: int,
    case,
    arena_offset: int,
    invocation_id: int,
    corruption: str,
) -> None:
    _, outputs = _prepare_finish(
        runtime, rank, control_group, case, arena_offset,
        invocation_id, expected_finish_status=0)
    if corruption == "num_segments":
        outputs[5].fill_(_WORLD_SIZE)
    elif corruption == "moved_prefix_origin":
        outputs[9][..., 0].fill_(1)
    elif corruption == "moved_prefix_terminal":
        outputs[9][..., -1].zero_()
    elif corruption == "group_prefix":
        outputs[10].fill_(case.proxy_capacity_per_egress)
    elif corruption == "proxy_required":
        outputs[11].zero_()
    else:
        raise AssertionError(corruption)

    x, weights = _local_source_inputs(case, rank, invocation_id)
    first_error = _capture(lambda: runtime._rail_balance_hybrid_source_shuffle(  # type: ignore[attr-defined]
        x, weights, invocation_id))
    local_status = int(outputs[-1].item())
    first = _gather_objects((first_error, local_status), control_group)
    moved_owners = _moved_owners(case)
    for owner, (error, status) in enumerate(first):
        if owner in moved_owners:
            assert error is not None and int(status) == 4, (corruption, first)
        else:
            assert error is None and int(status) == 0, (corruption, first)

    # Error ranks are sticky status-4; zero-token ranks completed their
    # one-shot call. Both classes must reject a second invocation locally.
    second_error = _capture(
        lambda: runtime._rail_balance_hybrid_source_shuffle(  # type: ignore[attr-defined]
            x, weights, invocation_id))
    second = _gather_objects(second_error, control_group)
    assert all(error is not None for error in second), (corruption, second)
    _abort(runtime, invocation_id, control_group, timeout)


def _run_capacity_failure(
    *, runtime: object, rank: int, control_group: dist.ProcessGroup,
    timeout: int, case, arena_offset: int, invocation_id: int,
) -> None:
    _, outputs = _prepare_finish(
        runtime, rank, control_group, case, arena_offset,
        invocation_id, expected_finish_status=1)
    x, weights = _local_source_inputs(case, rank, invocation_id)
    error = _capture(lambda: runtime._rail_balance_hybrid_source_shuffle(  # type: ignore[attr-defined]
        x, weights, invocation_id))
    observed = _gather_objects((error, int(outputs[-1].item())), control_group)
    assert all(item[0] is not None and int(item[1]) == 1 for item in observed)
    _abort(runtime, invocation_id, control_group, timeout)


def _run_duplicate_prepare(
    *, runtime: object, rank: int, control_group: dist.ProcessGroup,
    timeout: int, case, arena_offset: int, invocation_id: int,
) -> None:
    topk_idx = _local_topk(case, rank)
    if rank == 0:
        assert topk_idx.shape[0] > 0 and topk_idx.shape[1] > 1
        topk_idx = topk_idx.clone()
        topk_idx[0, 1] = topk_idx[0, 0]
    status, error = _prepare(
        runtime, topk_idx, case,
        arena_offset=arena_offset,
        invocation_id=invocation_id,
        stream=None,
    )
    observed = _gather_objects((status, error), control_group)
    assert tuple(int(item[0]) for item in observed) == \
        (3,) + (0,) * (_WORLD_SIZE - 1)
    assert all(item[1] is None for item in observed)
    _abort(runtime, invocation_id, control_group, timeout)


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int,
            arguments: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(
        local_rank, num_local_ranks, seed=82)
    control_group = dist.new_group(
        ranks=list(range(world_size)), backend="gloo",
        timeout=timedelta(seconds=arguments.timeout))
    assert rank == local_rank and world_size == _WORLD_SIZE

    case = _skew_case(
        capacity=_PROXY_CAPACITY, seed=62, name="fault_base")
    capacity_case = _skew_case(
        capacity=4, seed=62, name="fault_capacity")
    assert _schedule(case).enabled
    assert not _schedule(capacity_case).enabled

    layout = tuple(int(value) for value in
                   _C._get_rail_balance_hybrid_layout(
                       _HIDDEN, _NUM_TOPK, _PROXY_CAPACITY))
    arena_bytes = layout[-1]
    alignment = int(_C.get_elastic_buffer_alignment())
    base_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
        ep_group,
        num_max_tokens_per_rank=_MAX_TOKENS,
        hidden=_HIDDEN,
        num_topk=_NUM_TOPK,
        use_fp8_dispatch=False,
        allow_hybrid_mode=False,
        allow_multiple_reduction=True,
    )
    arena_offset = align(base_bytes, alignment)
    assert arena_offset == base_bytes
    buffer = deep_ep.ElasticBuffer(
        ep_group,
        num_bytes=arena_offset + arena_bytes,
        num_max_tokens_per_rank=_MAX_TOKENS,
        hidden=_HIDDEN,
        num_topk=_NUM_TOPK,
        allow_hybrid_mode=False,
        allow_multiple_reduction=True,
        prefer_overlap_with_compute=False,
        explicitly_destroy=True,
        num_gpu_timeout_secs=arguments.timeout,
        num_cpu_timeout_secs=arguments.timeout,
    )
    clean_shutdown = False
    try:
        for offset, corruption in enumerate((
            "num_segments", "moved_prefix_origin",
            "moved_prefix_terminal", "group_prefix", "proxy_required")):
            _run_corrupt_plan(
                runtime=buffer.runtime, rank=rank,
                control_group=control_group, timeout=arguments.timeout,
                case=case, arena_offset=arena_offset,
                invocation_id=910 + offset, corruption=corruption)

        _run_capacity_failure(
            runtime=buffer.runtime, rank=rank,
            control_group=control_group, timeout=arguments.timeout,
            case=capacity_case, arena_offset=arena_offset,
            invocation_id=920)
        _run_duplicate_prepare(
            runtime=buffer.runtime, rank=rank,
            control_group=control_group, timeout=arguments.timeout,
            case=case, arena_offset=arena_offset,
            invocation_id=930)

        # A complete source-shuffle transaction after every abort proves that
        # no failed private transaction stranded a barrier phase or owner.
        _, recovery_outputs = _prepare_finish(
            buffer.runtime, rank, control_group, case,
            arena_offset, 940, expected_finish_status=0)
        x, weights = _local_source_inputs(case, rank, 940)
        recovery_error = _capture(
            lambda: buffer.runtime._rail_balance_hybrid_source_shuffle(  # type: ignore[attr-defined]
                x, weights, 940))
        recovery = _gather_objects(
            (recovery_error, int(recovery_outputs[-1].item())),
            control_group)
        assert all(item[0] is None and int(item[1]) == 0
                   for item in recovery), recovery
        buffer.runtime.barrier(True, True, True)  # type: ignore[attr-defined]
        _abort(buffer.runtime, 940, control_group, arguments.timeout)
        _monitored_barrier(control_group, arguments.timeout)
        clean_shutdown = True
    finally:
        if clean_shutdown:
            buffer.destroy()
            _monitored_barrier(control_group, arguments.timeout)
            if rank == 0:
                print(
                    "PASS C080-D Hybrid source-shuffle faults: status4 "
                    "sticky x3, second-call rejection, capacity status1, "
                    "duplicate status3, abort/recovery",
                    flush=True,
                )
            dist.destroy_process_group()


def _run_watchdog(arguments: argparse.Namespace) -> None:
    command = [
        sys.executable, "-B", str(Path(__file__).resolve()),
        "--worker-suite", "--num-processes", str(arguments.num_processes),
        "--timeout", str(arguments.timeout),
        "--master-port", str(arguments.master_port),
        "--watchdog-seconds", str(arguments.watchdog_seconds),
    ]
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
            f"C080-D fault watchdog expired after "
            f"{arguments.watchdog_seconds}s") from error
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C080-D Hybrid source-shuffle fault test")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--worker-suite", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--master-port", type=int, default=29884)
    parser.add_argument("--watchdog-seconds", type=int, default=900)
    arguments = parser.parse_args()
    if arguments.num_processes != _WORLD_SIZE:
        parser.error("C080-D faults require exactly 8 processes")
    if arguments.timeout <= 0 or arguments.watchdog_seconds <= 0:
        parser.error("timeouts must be positive")

    case = _skew_case(
        capacity=_PROXY_CAPACITY, seed=62, name="fault_oracle")
    assert _schedule(case).enabled
    assert _moved_owners(case) == {0, 1}
    if arguments.oracle_only:
        print("PASS C080-D fault oracle", flush=True)
        return

    required = (
        "_rail_balance_hybrid_plan_prepare",
        "_rail_balance_hybrid_plan_finish",
        "_rail_balance_hybrid_plan_abort",
        "_rail_balance_hybrid_source_shuffle",
    )
    missing = [name for name in required
               if not hasattr(_C.ElasticBuffer, name)]
    if missing:
        parser.error("missing private API: " + ", ".join(missing))
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("C080-D faults require 8 visible CUDA devices")

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
