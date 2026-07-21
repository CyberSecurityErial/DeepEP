"""C080-H3b fixed-WORLD-gate coverage around the real Hybrid planner.

This is a private eight-GPU liveness/correctness test.  It does not enable the
public force path and it deliberately does not publish source payloads.

Run:

    EP_DISABLE_GIN=1 PYTHONPATH=.:tests/elastic \
      /home/chen/.cache/deepep-sjlgpt/bin/python -B \
      tests/elastic/test_rail_balance_hybrid_plan_world_gate.py
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import traceback
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.buffers import elastic as elastic_module
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import align
from test_rail_balance_hybrid_plan_lsa import (
    _ARENA_PROXY_CAPACITY,
    _HIDDEN,
    _MAX_TOKENS,
    _MAX_TOPK,
    _WORLD_SIZE,
    PlanCase,
    _assert_oracle_contract,
    _local_topk,
    _schedule,
    _verify_outputs,
)


_GATE_WORDS = 128
_MANIFEST_MAGIC = int.from_bytes(b"RBH3", byteorder="little")
_MANIFEST_VERSION = 1
_MANIFEST_OPERATION_PLAN = 1
_MANIFEST_FLAGS_VIRTUAL_DESTINATION_HARNESS = 1
_MANIFEST_WIDTH = 19

_PREPARE_EXCEPTION_PRIORITY = 10
_PREPARE_STATUS_PRIORITY = 20
_FINISH_EXCEPTION_PRIORITY = 40
_FINISH_STATUS_PRIORITY = 50
_INJECTED_GATE2_PRIORITY = 70


def _manifest(
    case: PlanCase,
    *,
    actual_num_topk: int,
    phase: int,
    invocation_id: int,
    num_rails: int,
    arena_offset: int,
    arena_bytes: int,
) -> tuple[int, ...]:
    # N and local rail/destination identities are intentionally absent.  They
    # legitimately differ between ranks or source servers.  The width is part
    # of the fixed contract because pair encoding alone cannot distinguish a
    # missing trailing zero field.
    fields = (
        _MANIFEST_MAGIC,
        _MANIFEST_VERSION,
        _MANIFEST_OPERATION_PLAN,
        phase,
        _MANIFEST_WIDTH,
        invocation_id,
        _WORLD_SIZE,
        case.num_scaleout_ranks,
        num_rails,
        _HIDDEN,
        actual_num_topk,
        case.num_channels,
        case.num_max_tokens_per_rank,
        case.num_experts,
        case.proxy_capacity_per_egress,
        arena_offset,
        arena_bytes,
        case.remainder_seed,
        _MANIFEST_FLAGS_VIRTUAL_DESTINATION_HARNESS,
    )
    assert len(fields) == _MANIFEST_WIDTH
    assert all(type(value) is int and value >= 0 for value in fields)
    return fields


def _assert_cpu_contract() -> tuple[PlanCase, PlanCase, PlanCase]:
    success, capacity_failure, boundary = _assert_oracle_contract()
    manifest = _manifest(
        success,
        actual_num_topk=success.num_topk,
        phase=1,
        invocation_id=1,
        num_rails=_WORLD_SIZE,
        arena_offset=0,
        arena_bytes=0,
    )
    assert len(manifest) == _MANIFEST_WIDTH
    assert success.num_tokens_per_rank == (9, 9, 0, 0, 0, 0, 0, 0)
    assert _schedule(success).enabled
    assert not _schedule(capacity_failure).enabled

    # Dynamic tests prove phase reuse.  This source guard freezes the stronger
    # statement that one first finish contains exactly one local barrier and
    # cannot publish source payloads.
    buffer_header = (
        Path(__file__).resolve().parents[2] / "csrc" / "elastic" /
        "buffer.hpp"
    ).read_text(encoding="utf-8")
    begin = buffer_header.index(
        "RailBalanceHybridPlanTensors rail_balance_hybrid_plan_finish(")
    end = buffer_header.index(
        "void rail_balance_hybrid_source_shuffle(", begin)
    finish_body = buffer_header[begin:end]
    # H4b freezes the barrier LaunchArgs before Gate1; finish may only submit
    # that prepared launch once and must not reconstruct arguments from live
    # tensors after the cross-rank transaction begins.
    assert finish_body.count(
        "submit_prepared_rail_balance_hybrid_local_barrier(") == 1
    assert "submit_prepared_rail_balance_hybrid_source_shuffle(" not in \
        finish_body
    assert "launch_prepared_rail_balance_hybrid_source_shuffle(" not in \
        finish_body

    words = torch.empty(_GATE_WORDS, dtype=torch.int64, device="cpu")
    elastic_module._encode_rail_balance_world_gate(words, 0, manifest)
    before = words.clone()
    error = _make_error(_INJECTED_GATE2_PRIORITY, 5)
    elastic_module._patch_rail_balance_world_gate_prevalidated(
        words, error, 3, 2)
    assert words[0].item() == error
    assert words[8:10].tolist() == [2, -2]
    before[0] = error
    before[8] = 2
    before[9] = -2
    assert torch.equal(words, before)
    return success, capacity_failure, boundary


def _make_error(priority: int, rank: int) -> int:
    return elastic_module._make_rail_balance_world_gate_error_key(
        priority, rank)


def _status_error(
    status: int,
    exception: str | None,
    *,
    rank: int,
    exception_priority: int,
    status_priority: int,
) -> int:
    if exception is not None:
        return _make_error(exception_priority, rank)
    if status != 0:
        return _make_error(status_priority + status, rank)
    return 0


def _prepare(
    runtime: object,
    topk_idx: torch.Tensor,
    case: PlanCase,
    *,
    arena_offset: int,
    invocation_id: int,
) -> tuple[int, str | None]:
    try:
        return int(runtime._rail_balance_hybrid_plan_prepare(  # type: ignore[attr-defined]
            topk_idx,
            _HIDDEN,
            case.num_channels,
            case.num_max_tokens_per_rank,
            case.num_experts,
            case.num_scaleout_ranks,
            case.local_scaleout_rank,
            case.proxy_capacity_per_egress,
            arena_offset,
            invocation_id,
            case.remainder_seed,
        )), None
    except BaseException:
        return -1, traceback.format_exc()


def _finish(
    runtime: object,
    invocation_id: int,
) -> tuple[tuple[torch.Tensor, ...] | None, int, str | None]:
    try:
        values = runtime._rail_balance_hybrid_plan_finish(  # type: ignore[attr-defined]
            invocation_id)
        outputs = tuple(values)
        if len(outputs) != 14 or not all(
                isinstance(value, torch.Tensor) for value in outputs):
            raise AssertionError("planner finish did not return fourteen tensors")
        return outputs, int(outputs[-1].item()), None
    except BaseException:
        return None, -1, traceback.format_exc()


def _abort(runtime: object, invocation_id: int) -> str | None:
    try:
        runtime._rail_balance_hybrid_plan_abort(invocation_id)  # type: ignore[attr-defined]
        runtime._rail_balance_hybrid_plan_abort(invocation_id)  # type: ignore[attr-defined]
        return None
    except BaseException:
        return traceback.format_exc()


def _report_local_result(
    label: str,
    local_error: str | None,
    control_group: dist.ProcessGroup,
    timeout: int,
) -> None:
    gathered: list[str | None] = [None] * dist.get_world_size(control_group)
    dist.all_gather_object(gathered, local_error, group=control_group)
    messages = [
        f"rank {rank}:\n{message}"
        for rank, message in enumerate(gathered)
        if message is not None
    ]
    if messages:
        raise AssertionError(f"{label} failed:\n" + "\n".join(messages))
    dist.monitored_barrier(
        group=control_group,
        timeout=timedelta(seconds=timeout),
        wait_all_ranks=True,
    )


def _gate(
    *,
    device_words: torch.Tensor,
    host_words: torch.Tensor,
    ep_group: dist.ProcessGroup,
) -> tuple[int, int, int, int]:
    return elastic_module._run_rail_balance_world_gate(
        device_words, host_words, ep_group)


def _verify_and_abort(
    *,
    runtime: object,
    invocation_id: int,
    outputs: tuple[torch.Tensor, ...] | None,
    case: PlanCase,
    rank: int,
    label: str,
    local_checks: Sequence[tuple[bool, str]],
    control_group: dist.ProcessGroup,
    timeout: int,
) -> None:
    abort_error = _abort(runtime, invocation_id)
    local_error = abort_error
    if local_error is None:
        try:
            for condition, message in local_checks:
                assert condition, message
            assert outputs is not None
            _verify_outputs(outputs, _schedule(case), rank)
        except BaseException:
            local_error = traceback.format_exc()
    _report_local_result(label, local_error, control_group, timeout)


def _run_success_or_gate2_failure(
    *,
    runtime: object,
    rank: int,
    ep_group: dist.ProcessGroup,
    control_group: dist.ProcessGroup,
    timeout: int,
    device_words: torch.Tensor,
    host_words: torch.Tensor,
    gate_calls: list[int],
    case: PlanCase,
    arena_offset: int,
    arena_bytes: int,
    invocation_id: int,
    label: str,
    expect_capacity_failure: bool = False,
    injected_gate2_rank: int | None = None,
    nondefault_stream: bool = False,
    stale_abort_invocation_id: int | None = None,
) -> None:
    schedule_enabled = _schedule(case).enabled
    assert schedule_enabled == (not expect_capacity_failure), (
        label, schedule_enabled, expect_capacity_failure)
    topk_idx = _local_topk(case, rank)
    stream = torch.cuda.Stream(device=rank) if nondefault_stream else None
    manifest1 = _manifest(
        case, actual_num_topk=int(topk_idx.shape[1]),
        phase=1, invocation_id=invocation_id,
        num_rails=_WORLD_SIZE, arena_offset=arena_offset,
        arena_bytes=arena_bytes)
    manifest2 = _manifest(
        case, actual_num_topk=int(topk_idx.shape[1]),
        phase=2, invocation_id=invocation_id,
        num_rails=_WORLD_SIZE, arena_offset=arena_offset,
        arena_bytes=arena_bytes)
    assert manifest1[:3] == manifest2[:3]
    assert manifest1[3] == 1 and manifest2[3] == 2
    assert manifest1[4:] == manifest2[4:]
    # All checked encoding happens before prepare owns transaction state.
    elastic_module._encode_rail_balance_world_gate(
        host_words, 0, manifest1)
    calls_before = gate_calls[0]

    def execute_transaction():
        prepare_status, prepare_exception = _prepare(
            runtime, topk_idx, case, arena_offset=arena_offset,
            invocation_id=invocation_id)
        if prepare_exception is None and stale_abort_invocation_id is not None:
            try:
                runtime._rail_balance_hybrid_plan_abort(  # type: ignore[attr-defined]
                    stale_abort_invocation_id)
            except BaseException:
                prepare_exception = traceback.format_exc()
        gate1_error = _status_error(
            prepare_status, prepare_exception, rank=rank,
            exception_priority=_PREPARE_EXCEPTION_PRIORITY,
            status_priority=_PREPARE_STATUS_PRIORITY)
        elastic_module._patch_rail_balance_world_gate_prevalidated(
            host_words, gate1_error)
        gate1 = _gate(
            device_words=device_words, host_words=host_words,
            ep_group=ep_group)

        # Every rank observes the same decoded gate.  On success, enter finish
        # immediately: no Gloo control operation, allocation, stream-context
        # transition, or local oracle sits between Gate1 and the LSA barrier.
        if gate1 != (0, -1, 0, 0):
            return (prepare_status, prepare_exception, gate1,
                    None, -1, None, None)
        outputs, finish_status, finish_exception = _finish(
            runtime, invocation_id)

        local_gate2_error = _status_error(
            finish_status, finish_exception, rank=rank,
            exception_priority=_FINISH_EXCEPTION_PRIORITY,
            status_priority=_FINISH_STATUS_PRIORITY)
        if injected_gate2_rank == rank and local_gate2_error == 0:
            local_gate2_error = _make_error(_INJECTED_GATE2_PRIORITY, rank)
        elastic_module._patch_rail_balance_world_gate_prevalidated(
            host_words, local_gate2_error, 3, 2)
        gate2 = _gate(
            device_words=device_words, host_words=host_words,
            ep_group=ep_group)
        return (prepare_status, prepare_exception, gate1,
                outputs, finish_status, finish_exception, gate2)

    if stream is None:
        transaction = execute_transaction()
    else:
        # Context entry happens before prepare and exit happens after Gate2.
        # There is no rank-local stream transition in the committed interval.
        with torch.cuda.stream(stream):
            transaction = execute_transaction()
    (prepare_status, prepare_exception, gate1, outputs,
     finish_status, finish_exception, gate2) = transaction

    if gate1 != (0, -1, 0, 0):
        abort_error = _abort(runtime, invocation_id)
        error = abort_error or (
            f"unexpected Gate1 rejection for {label}: {gate1}")
        _report_local_result(label, error, control_group, timeout)
        return
    expected_finish_status = 1 if expect_capacity_failure else 0
    if expect_capacity_failure:
        expected_gate2 = (_make_error(
            _FINISH_STATUS_PRIORITY + 1, 0), -1, 0, 0)
    elif injected_gate2_rank is not None:
        expected_gate2 = (_make_error(
            _INJECTED_GATE2_PRIORITY, injected_gate2_rank), -1, 0, 0)
    else:
        expected_gate2 = (0, -1, 0, 0)

    _verify_and_abort(
        runtime=runtime, invocation_id=invocation_id,
        outputs=outputs, case=case, rank=rank, label=label,
        local_checks=(
            (prepare_status == 0 and prepare_exception is None,
             f"prepare status/error: {prepare_status}, {prepare_exception}"),
            (finish_status == expected_finish_status,
             f"finish status {finish_status}, expected {expected_finish_status}"),
            (finish_exception is None, f"finish exception: {finish_exception}"),
            (gate2 == expected_gate2,
             f"Gate2 {gate2}, expected {expected_gate2}"),
            (gate_calls[0] - calls_before == 2,
             "transaction did not use exactly two WORLD MAX gates"),
        ),
        control_group=control_group, timeout=timeout)


def _run_gate1_failure(
    *,
    runtime: object,
    rank: int,
    ep_group: dist.ProcessGroup,
    control_group: dist.ProcessGroup,
    timeout: int,
    device_words: torch.Tensor,
    host_words: torch.Tensor,
    gate_calls: list[int],
    case: PlanCase,
    arena_offset: int,
    arena_bytes: int,
    invocation_id: int,
    label: str,
    invalid_route_rank: int | None = None,
    topk_width_mismatch_rank: int | None = None,
) -> None:
    local_case = case
    topk_idx = _local_topk(local_case, rank)
    if topk_width_mismatch_rank == rank:
        assert topk_idx.shape[0] == 0
        topk_idx = torch.empty(
            (0, local_case.num_topk + 1),
            dtype=_C.topk_idx_t,
            device=torch.device("cuda", rank),
        )
    if invalid_route_rank == rank:
        assert topk_idx.numel() > 0
        topk_idx = topk_idx.clone()
        topk_idx[0, 0] = local_case.num_experts
    manifest = _manifest(
        local_case, actual_num_topk=int(topk_idx.shape[1]),
        phase=1, invocation_id=invocation_id,
        num_rails=_WORLD_SIZE, arena_offset=arena_offset,
        arena_bytes=arena_bytes)
    elastic_module._encode_rail_balance_world_gate(
        host_words, 0, manifest)
    calls_before = gate_calls[0]
    prepare_status, prepare_exception = _prepare(
        runtime, topk_idx, local_case, arena_offset=arena_offset,
        invocation_id=invocation_id)
    gate1_error = _status_error(
        prepare_status, prepare_exception, rank=rank,
        exception_priority=_PREPARE_EXCEPTION_PRIORITY,
        status_priority=_PREPARE_STATUS_PRIORITY)
    elastic_module._patch_rail_balance_world_gate_prevalidated(
        host_words, gate1_error)
    gate1 = _gate(
        device_words=device_words, host_words=host_words,
        ep_group=ep_group)
    abort_error = _abort(runtime, invocation_id)

    local_error = abort_error
    if local_error is None:
        try:
            assert gate_calls[0] - calls_before == 1
            if topk_width_mismatch_rank is not None:
                assert invalid_route_rank is None
                assert prepare_status == 0 and prepare_exception is None
                assert gate1 == (
                    0, 10, case.num_topk, case.num_topk + 1)
            else:
                assert invalid_route_rank is not None
                expected = _make_error(
                    _PREPARE_STATUS_PRIORITY + 2, invalid_route_rank)
                assert gate1 == (expected, -1, 0, 0)
                if rank == invalid_route_rank:
                    assert prepare_status == 2 and prepare_exception is None
                else:
                    assert prepare_status == 0 and prepare_exception is None
        except BaseException:
            local_error = traceback.format_exc()
    _report_local_result(label, local_error, control_group, timeout)


@torch.inference_mode()
def _worker(
    local_rank: int,
    num_processes: int,
    arguments: argparse.Namespace,
) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(
        local_rank, num_processes, seed=803)
    control_group = dist.new_group(
        ranks=list(range(world_size)), backend="gloo",
        timeout=timedelta(seconds=arguments.timeout))
    assert rank == local_rank and world_size == _WORLD_SIZE

    device = torch.device("cuda", rank)
    device_words = torch.empty(
        _GATE_WORDS, dtype=torch.int64, device=device)
    host_words = torch.empty(
        _GATE_WORDS, dtype=torch.int64, device="cpu", pin_memory=True)
    elastic_module._validate_rail_balance_world_gate_storage(
        device_words, host_words)
    device_ptr = device_words.data_ptr()
    host_ptr = host_words.data_ptr()

    original_all_reduce = dist.all_reduce
    gate_calls = [0]

    def checked_all_reduce(
        tensor: torch.Tensor,
        op: dist.ReduceOp | None = None,
        group: dist.ProcessGroup | None = None,
        async_op: bool = False,
    ):
        gate_calls[0] += 1
        assert tensor.data_ptr() == device_ptr
        assert tensor.dtype == torch.int64
        assert tuple(tensor.shape) == (_GATE_WORDS,)
        assert op == dist.ReduceOp.MAX
        assert group is ep_group
        assert async_op is False
        return original_all_reduce(
            tensor, op=op, group=group, async_op=async_op)

    dist.all_reduce = checked_all_reduce
    buffer = None
    clean_shutdown = False
    try:
        success, capacity_failure, boundary = _assert_cpu_contract()
        arena_bytes = int(_C._get_rail_balance_hybrid_layout(
            _HIDDEN, _MAX_TOPK, _ARENA_PROXY_CAPACITY)[-1])
        alignment = int(_C.get_elastic_buffer_alignment())
        base_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
            ep_group,
            num_max_tokens_per_rank=_MAX_TOKENS,
            hidden=_HIDDEN,
            num_topk=_MAX_TOPK,
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
            num_topk=_MAX_TOPK,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
            prefer_overlap_with_compute=False,
            explicitly_destroy=True,
            num_gpu_timeout_secs=arguments.timeout,
            num_cpu_timeout_secs=arguments.timeout,
        )
        runtime = buffer.runtime
        assert buffer.get_logical_domain_size() == (1, _WORLD_SIZE)

        _run_success_or_gate2_failure(
            runtime=runtime, rank=rank, ep_group=ep_group,
            control_group=control_group, timeout=arguments.timeout,
            device_words=device_words, host_words=host_words,
            gate_calls=gate_calls, case=success,
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=8301, label="variable-zero-N success",
            stale_abort_invocation_id=8300)

        # Rank five owns N=0, so changing only its physical top-k width is a
        # valid local prepare.  Gate1 must fingerprint the actual tensor K,
        # not the nominal PlanCase K, and reject before finish.
        _run_gate1_failure(
            runtime=runtime, rank=rank, ep_group=ep_group,
            control_group=control_group, timeout=arguments.timeout,
            device_words=device_words, host_words=host_words,
            gate_calls=gate_calls,
            case=replace(success, remainder_seed=65),
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=8302, label="Gate1 zero-N actual-K mismatch",
            topk_width_mismatch_rank=5)
        _run_success_or_gate2_failure(
            runtime=runtime, rank=rank, ep_group=ep_group,
            control_group=control_group, timeout=arguments.timeout,
            device_words=device_words, host_words=host_words,
            gate_calls=gate_calls,
            case=replace(success, remainder_seed=65),
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=8303, label="retry after actual-K abort")

        _run_gate1_failure(
            runtime=runtime, rank=rank, ep_group=ep_group,
            control_group=control_group, timeout=arguments.timeout,
            device_words=device_words, host_words=host_words,
            gate_calls=gate_calls,
            case=replace(success, remainder_seed=72),
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=8304, label="Gate1 invalid route",
            invalid_route_rank=1)
        _run_success_or_gate2_failure(
            runtime=runtime, rank=rank, ep_group=ep_group,
            control_group=control_group, timeout=arguments.timeout,
            device_words=device_words, host_words=host_words,
            gate_calls=gate_calls,
            case=replace(success, remainder_seed=72),
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=8305, label="retry after route abort")

        _run_success_or_gate2_failure(
            runtime=runtime, rank=rank, ep_group=ep_group,
            control_group=control_group, timeout=arguments.timeout,
            device_words=device_words, host_words=host_words,
            gate_calls=gate_calls, case=capacity_failure,
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=8306, label="Gate2 capacity fail-close",
            expect_capacity_failure=True)
        _run_success_or_gate2_failure(
            runtime=runtime, rank=rank, ep_group=ep_group,
            control_group=control_group, timeout=arguments.timeout,
            device_words=device_words, host_words=host_words,
            gate_calls=gate_calls,
            case=replace(success, remainder_seed=73),
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=8307, label="retry after capacity abort")

        _run_success_or_gate2_failure(
            runtime=runtime, rank=rank, ep_group=ep_group,
            control_group=control_group, timeout=arguments.timeout,
            device_words=device_words, host_words=host_words,
            gate_calls=gate_calls,
            case=replace(success, remainder_seed=74),
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=8308, label="asymmetric Gate2 error",
            injected_gate2_rank=5)
        _run_success_or_gate2_failure(
            runtime=runtime, rank=rank, ep_group=ep_group,
            control_group=control_group, timeout=arguments.timeout,
            device_words=device_words, host_words=host_words,
            gate_calls=gate_calls,
            case=replace(success, remainder_seed=74),
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=8309, label="retry after asymmetric Gate2 abort")

        _run_success_or_gate2_failure(
            runtime=runtime, rank=rank, ep_group=ep_group,
            control_group=control_group, timeout=arguments.timeout,
            device_words=device_words, host_words=host_words,
            gate_calls=gate_calls, case=boundary,
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=8310, label="C1024-D32 nondefault stream",
            nondefault_stream=True)

        local_error = None
        try:
            assert gate_calls[0] == 18, gate_calls[0]
            assert device_words.data_ptr() == device_ptr
            assert host_words.data_ptr() == host_ptr
        except BaseException:
            local_error = traceback.format_exc()
        _report_local_result(
            "fixed storage/final gate count", local_error,
            control_group, arguments.timeout)
        clean_shutdown = True
    finally:
        dist.all_reduce = original_all_reduce
        if clean_shutdown and buffer is not None:
            buffer.destroy()
            buffer = None
            dist.monitored_barrier(
                group=control_group,
                timeout=timedelta(seconds=arguments.timeout),
                wait_all_ranks=True,
            )
            if rank == 0:
                print(
                    "PASS C080-H3b real planner WORLD gates: variable/zero "
                    "N, exact 14 outputs, Gate1 mismatch/route recovery, "
                    "Gate2 capacity/asymmetric pre-publication recovery, "
                    "stable storage, non-default-stream compatibility smoke, "
                    "18 fixed MAX gates",
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
            "C080-H3b subprocess watchdog expired after "
            f"{arguments.watchdog_seconds}s") from error
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C080-H3b real planner fixed WORLD-gate test")
    if not __debug__:
        parser.error("C080-H3b correctness checks require Python assertions")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--worker-suite", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--master-port", type=int, default=29953)
    parser.add_argument("--watchdog-seconds", type=int, default=900)
    arguments = parser.parse_args()
    if arguments.num_processes != _WORLD_SIZE:
        parser.error("C080-H3b requires exactly 8 processes")
    if arguments.timeout <= 0 or arguments.watchdog_seconds <= 0:
        parser.error("timeouts must be positive")

    success, capacity_failure, boundary = _assert_cpu_contract()
    print(
        "PASS C080-H3b CPU contract: fixed manifest width, source guard "
        "one direct finish barrier/no direct payload launch, variable/zero N, "
        f"Pcap={success.proxy_capacity_per_egress}/"
        f"{capacity_failure.proxy_capacity_per_egress}, "
        f"C={boundary.num_channels}/D={boundary.num_scaleout_ranks}",
        flush=True,
    )
    if arguments.cpu_only:
        return

    required = (
        "_rail_balance_hybrid_plan_prepare",
        "_rail_balance_hybrid_plan_finish",
        "_rail_balance_hybrid_plan_abort",
    )
    runtime_type = getattr(_C, "ElasticBuffer", None)
    missing = [
        name for name in required
        if runtime_type is None or not hasattr(runtime_type, name)
    ]
    if missing:
        parser.error("rebuild the C080 private API; missing " + ", ".join(missing))
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("C080-H3b requires at least eight visible CUDA devices")

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
