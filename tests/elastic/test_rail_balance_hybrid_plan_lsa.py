"""C080-B2: strict 8-GPU LSA snapshot/Hybrid plan transaction test.

This is a pure single-node functional test.  Every process owns one physical
GPU in the same LSA team, while ``num_scaleout_ranks`` remains a virtual
destination-server namespace.  Gloo is used only as a short-timeout test gate;
it is not the production force consensus protocol.

The private transaction under test is::

    status = runtime._rail_balance_hybrid_plan_prepare(
        topk_idx, hidden, C, M, E, D, local_d, Pcap,
        arena_offset, invocation_id, seed)
    outputs = runtime._rail_balance_hybrid_plan_finish(invocation_id)
    runtime._rail_balance_hybrid_plan_abort(invocation_id)

``finish`` returns the same fourteen int32 tensors as C080-B1.  The test uses
different local token counts (including zero), compares every tensor with the
CPU Hybrid oracle, and compares a complete snapshot digest across all ranks.

Run the CPU contract while the private API is being built::

    PYTHONPATH=. python -B tests/elastic/test_rail_balance_hybrid_plan_lsa.py \
      --oracle-only

Run the strict H200 path after rebuilding the extension::

    PYTHONPATH=. EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      python -B tests/elastic/test_rail_balance_hybrid_plan_lsa.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import traceback
from dataclasses import dataclass, replace
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
    build_hybrid_rail_schedule,
)


_WORLD_SIZE = 8
_HIDDEN = 256
_MAX_TOKENS = 10
_MAX_TOPK = 32
_ARENA_PROXY_CAPACITY = 16
_LOCAL_DESTINATION = 0
_OPERATION = "rail_balance_hybrid_plan"
_HOST_PREPARE_FAILURE = 100
_HOST_FINISH_FAILURE = 101

_OUTPUT_LABELS = (
    "channel_count",
    "count",
    "quota",
    "keep_count",
    "segments",
    "num_segments",
    "owner_channel_prefix",
    "retained",
    "moved",
    "moved_channel_prefix",
    "group_prefix",
    "proxy_required",
    "moved_copies",
    "status",
)

# C061's three virtual remote destinations are shifted by one because real
# destination zero is the local server in the strict Hybrid contract.
_C061_VIRTUAL_ROUTES = (
    (
        (0, 1, 2, 0),
        (1, 2, 1, 0),
        (2, 2, 0, 0),
        (2, 0, 0, 2),
        (0, 0, 2, 2),
        (0, 2, 2, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    ),
    (
        (1, 2, 1, 0),
        (2, 2, 0, 1),
        (2, 1, 1, 2),
        (1, 1, 2, 2),
        (1, 2, 2, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
    ),
)


@dataclass(frozen=True)
class PlanCase:
    name: str
    topk_idx: tuple[tuple[tuple[int, ...], ...], ...]
    num_topk: int
    num_channels: int
    num_max_tokens_per_rank: int
    num_experts: int
    num_scaleout_ranks: int
    local_scaleout_rank: int
    proxy_capacity_per_egress: int
    remainder_seed: int

    @property
    def num_tokens_per_rank(self) -> tuple[int, ...]:
        return tuple(len(rows) for rows in self.topk_idx)


def _encode_destination_rows(
    rows: Sequence[Sequence[int]],
    *,
    num_destinations: int,
    experts_per_destination: int,
) -> tuple[tuple[int, ...], ...]:
    """Encode destination lanes as distinct real experts within each token."""

    encoded = []
    for destinations in rows:
        occurrence = [0] * num_destinations
        experts = []
        for destination in destinations:
            assert 0 <= destination < num_destinations
            lane = occurrence[destination]
            occurrence[destination] += 1
            assert lane < experts_per_destination
            experts.append(destination * experts_per_destination + lane)
        assert len(experts) == len(set(experts))
        encoded.append(tuple(experts))
    return tuple(encoded)


def _skew_case(*, capacity: int, seed: int, name: str) -> PlanCase:
    destinations = tuple(
        tuple(tuple(value + 1 for value in token) for token in owner)
        for owner in _C061_VIRTUAL_ROUTES
    ) + ((),) * (_WORLD_SIZE - len(_C061_VIRTUAL_ROUTES))
    experts_per_destination = 8
    topk_idx = tuple(
        _encode_destination_rows(
            owner,
            num_destinations=4,
            experts_per_destination=experts_per_destination,
        )
        for owner in destinations
    )
    return PlanCase(
        name=name,
        topk_idx=topk_idx,
        num_topk=4,
        num_channels=2,
        num_max_tokens_per_rank=_MAX_TOKENS,
        num_experts=4 * experts_per_destination,
        num_scaleout_ranks=4,
        local_scaleout_rank=_LOCAL_DESTINATION,
        proxy_capacity_per_egress=capacity,
        remainder_seed=seed,
    )


def _boundary_case() -> PlanCase:
    token_counts = (2, 1, 1, 0, 0, 0, 0, 0)
    rows = []
    for rank, count in enumerate(token_counts):
        token = tuple(destination * _WORLD_SIZE + rank
                      for destination in range(32))
        rows.append(tuple(token for _ in range(count)))
    return PlanCase(
        name="c1024_d32",
        topk_idx=tuple(rows),
        num_topk=32,
        num_channels=1024,
        num_max_tokens_per_rank=_MAX_TOKENS,
        num_experts=32 * _WORLD_SIZE,
        num_scaleout_ranks=32,
        local_scaleout_rank=_LOCAL_DESTINATION,
        proxy_capacity_per_egress=16,
        remainder_seed=(1 << 40) + 7,
    )


def _schedule(case: PlanCase) -> HybridRailSchedule:
    return build_hybrid_rail_schedule(
        case.topk_idx,
        num_topk=case.num_topk,
        num_experts=case.num_experts,
        num_scaleout_ranks=case.num_scaleout_ranks,
        local_scaleout_rank=case.local_scaleout_rank,
        num_channels=case.num_channels,
        num_max_tokens_per_rank=case.num_max_tokens_per_rank,
        proxy_capacity_per_egress=case.proxy_capacity_per_egress,
        remainder_seed=case.remainder_seed,
    )


def _int32_tensor(value: object) -> torch.Tensor:
    # init_dist installs the rank-local CUDA device as PyTorch's default.
    # Oracle tensors must remain CPU tensors regardless of that process state.
    return torch.tensor(
        value, dtype=torch.int32, device="cpu").contiguous()


def _expected_outputs(schedule: HybridRailSchedule) -> tuple[torch.Tensor, ...]:
    segments = torch.full(
        (schedule.num_destinations, schedule.num_rails - 1, 5),
        -1,
        dtype=torch.int32,
        device="cpu",
    )
    for destination, bucket in enumerate(schedule.segments):
        for index, segment in enumerate(bucket):
            segments[destination, index] = _int32_tensor((
                segment.owner,
                segment.egress,
                segment.owner_begin,
                segment.count,
                segment.egress_begin,
            ))
    values = (
        schedule.channel_count,
        schedule.count,
        schedule.quota,
        schedule.keep_count,
        None,
        schedule.num_segments,
        schedule.owner_channel_prefix,
        schedule.retained,
        schedule.moved,
        schedule.moved_channel_prefix,
        schedule.group_prefix,
        schedule.proxy_required,
        (schedule.moved_copies,),
        (0 if schedule.enabled else 1,),
    )
    outputs = tuple(
        segments if value is None else _int32_tensor(value)
        for value in values
    )
    assert len(outputs) == len(_OUTPUT_LABELS) == 14
    return outputs


def _assert_oracle_contract() -> tuple[PlanCase, PlanCase, PlanCase]:
    success = _skew_case(capacity=5, seed=62, name="c061_skew_success")
    failure = replace(
        success,
        name="c061_skew_capacity_failure",
        proxy_capacity_per_egress=4,
    )
    boundary = _boundary_case()
    success_schedule = _schedule(success)
    failure_schedule = _schedule(failure)
    boundary_schedule = _schedule(boundary)

    assert success.num_tokens_per_rank == (9, 9, 0, 0, 0, 0, 0, 0)
    assert success_schedule.count == (
        (0, 9, 2, 6),
        (0, 2, 9, 5),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    assert success_schedule.proxy_required == (0, 0, 5, 3, 3, 3, 3, 4)
    assert success_schedule.moved_copies == 21
    assert success_schedule.enabled and success_schedule.failure_reason is None
    assert not failure_schedule.enabled
    assert failure_schedule.failure_reason == (
        "proxy_capacity_per_egress_exceeded")
    success_outputs = _expected_outputs(success_schedule)
    failure_outputs = _expected_outputs(failure_schedule)
    assert all(torch.equal(left, right) for left, right in zip(
        success_outputs[:-1], failure_outputs[:-1]))
    assert success_outputs[-1].tolist() == [0]
    assert failure_outputs[-1].tolist() == [1]

    assert boundary.num_tokens_per_rank == (2, 1, 1, 0, 0, 0, 0, 0)
    assert boundary_schedule.enabled
    assert boundary_schedule.proxy_required == (0, 0, 0, 16, 4, 4, 4, 3)
    assert boundary_schedule.moved_copies == 31
    boundary_outputs = _expected_outputs(boundary_schedule)
    assert tuple(boundary_outputs[0].shape) == (_WORLD_SIZE, 1024, 32)
    assert tuple(boundary_outputs[4].shape) == (32, _WORLD_SIZE - 1, 5)
    assert tuple(boundary_outputs[9].shape) == (_WORLD_SIZE, 32, 1025)
    return success, failure, boundary


def _local_topk(case: PlanCase, rank: int) -> torch.Tensor:
    rows = case.topk_idx[rank]
    device = torch.device("cuda", rank)
    if not rows:
        return torch.empty(
            (0, case.num_topk), dtype=_C.topk_idx_t, device=device)
    return torch.tensor(rows, dtype=_C.topk_idx_t, device=device).contiguous()


def _gate_signature(
    case: PlanCase,
    *,
    arena_offset: int,
    arena_bytes: int,
) -> tuple[object, ...]:
    # N is intentionally absent: ranks may own different token counts.
    return (
        _OPERATION,
        _WORLD_SIZE,
        _HIDDEN,
        case.num_topk,
        case.num_channels,
        case.num_max_tokens_per_rank,
        case.num_experts,
        case.num_scaleout_ranks,
        case.local_scaleout_rank,
        case.proxy_capacity_per_egress,
        arena_offset,
        arena_bytes,
        case.remainder_seed,
        "force-v1-local-lsa",
    )


def _gather_objects(
    value: object,
    control_group: dist.ProcessGroup,
) -> list[object]:
    gathered: list[object | None] = [None] * dist.get_world_size(control_group)
    dist.all_gather_object(gathered, value, group=control_group)
    # ``None`` is itself a valid payload for the synchronized no-error path.
    # Gloo completes the list in place, so there is no separate empty sentinel
    # to assert here.
    return gathered  # type: ignore[return-value]


def _monitored_barrier(
    control_group: dist.ProcessGroup,
    timeout: int,
) -> None:
    dist.monitored_barrier(
        group=control_group,
        timeout=timedelta(seconds=timeout),
        wait_all_ranks=True,
    )


def _abort(
    runtime: object,
    invocation_id: int,
    control_group: dist.ProcessGroup,
    timeout: int,
) -> None:
    failure = None
    try:
        runtime._rail_balance_hybrid_plan_abort(invocation_id)  # type: ignore[attr-defined]
        # The private recovery primitive is explicitly idempotent.
        runtime._rail_balance_hybrid_plan_abort(invocation_id)  # type: ignore[attr-defined]
    except BaseException:
        failure = traceback.format_exc()
    failures = _gather_objects(failure, control_group)
    messages = [
        f"rank {rank}:\n{message}"
        for rank, message in enumerate(failures)
        if message is not None
    ]
    if messages:
        raise AssertionError("abort failed:\n" + "\n".join(messages))
    _monitored_barrier(control_group, timeout)


def _prepare(
    runtime: object,
    topk_idx: torch.Tensor,
    case: PlanCase,
    *,
    arena_offset: int,
    invocation_id: int,
    stream: torch.cuda.Stream | None,
) -> tuple[int, str | None]:
    try:
        def call() -> int:
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
            ))

        if stream is None:
            status = call()
        else:
            with torch.cuda.stream(stream):
                status = call()
        return status, None
    except BaseException:
        return _HOST_PREPARE_FAILURE, traceback.format_exc()


def _finish(
    runtime: object,
    invocation_id: int,
    stream: torch.cuda.Stream | None,
) -> tuple[tuple[torch.Tensor, ...] | None, int, str | None]:
    try:
        if stream is None:
            values = runtime._rail_balance_hybrid_plan_finish(  # type: ignore[attr-defined]
                invocation_id)
        else:
            with torch.cuda.stream(stream):
                values = runtime._rail_balance_hybrid_plan_finish(  # type: ignore[attr-defined]
                    invocation_id)
        outputs = tuple(values)
        assert len(outputs) == 14
        assert all(isinstance(value, torch.Tensor) for value in outputs)
        status = int(outputs[-1].item())
        return outputs, status, None
    except BaseException:
        return None, _HOST_FINISH_FAILURE, traceback.format_exc()


def _snapshot_digest(outputs: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for label, tensor in zip(_OUTPUT_LABELS, outputs):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(label.encode("ascii"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _verify_outputs(
    outputs: Sequence[torch.Tensor],
    schedule: HybridRailSchedule,
    rank: int,
) -> str:
    expected = _expected_outputs(schedule)
    assert len(outputs) == len(expected) == 14
    for label, actual, wanted in zip(_OUTPUT_LABELS, outputs, expected):
        assert actual.is_cuda and actual.device.index == rank, label
        assert actual.dtype == torch.int32 and actual.is_contiguous(), label
        actual_cpu = actual.cpu()
        assert tuple(actual_cpu.shape) == tuple(wanted.shape), (
            label, actual_cpu.shape, wanted.shape)
        assert torch.equal(actual_cpu, wanted), label
    return _snapshot_digest(outputs)


def _run_gate1_failure(
    *,
    runtime: object,
    rank: int,
    control_group: dist.ProcessGroup,
    timeout: int,
    case: PlanCase,
    arena_offset: int,
    arena_bytes: int,
    invocation_id: int,
    expected_statuses: Sequence[int],
    mismatched_case: PlanCase | None = None,
    invalid_route_rank: int | None = None,
) -> None:
    local_case = mismatched_case if mismatched_case is not None and rank == 5 \
        else case
    topk_idx = _local_topk(local_case, rank)
    if invalid_route_rank == rank:
        assert topk_idx.numel() > 0
        topk_idx = topk_idx.clone()
        topk_idx[0, 0] = -1
    status, error = _prepare(
        runtime,
        topk_idx,
        local_case,
        arena_offset=arena_offset,
        invocation_id=invocation_id,
        stream=None,
    )
    payload = (
        _OPERATION,
        1,
        invocation_id,
        status,
        _gate_signature(
            local_case, arena_offset=arena_offset, arena_bytes=arena_bytes),
        int(topk_idx.shape[0]),
        error,
    )
    gathered = _gather_objects(payload, control_group)
    statuses = tuple(int(item[3]) for item in gathered)  # type: ignore[index]
    signatures = tuple(item[4] for item in gathered)  # type: ignore[index]
    epochs = tuple(item[2] for item in gathered)  # type: ignore[index]
    assert statuses == tuple(expected_statuses)
    assert epochs == (invocation_id,) * _WORLD_SIZE
    if mismatched_case is None:
        assert len(set(signatures)) == 1
        assert invalid_route_rank is not None
    else:
        assert len(set(signatures)) == 2
        assert invalid_route_rank is None
    accepted = all(value == 0 for value in statuses) and len(set(signatures)) == 1
    assert not accepted
    _abort(runtime, invocation_id, control_group, timeout)


def _run_transaction(
    *,
    runtime: object,
    rank: int,
    control_group: dist.ProcessGroup,
    timeout: int,
    case: PlanCase,
    arena_offset: int,
    arena_bytes: int,
    invocation_id: int,
    nondefault_stream: bool = False,
) -> None:
    topk_idx = _local_topk(case, rank)
    stream = torch.cuda.Stream(device=rank) if nondefault_stream else None
    status, prepare_error = _prepare(
        runtime,
        topk_idx,
        case,
        arena_offset=arena_offset,
        invocation_id=invocation_id,
        stream=stream,
    )
    gate1_payload = (
        _OPERATION,
        1,
        invocation_id,
        status,
        _gate_signature(case, arena_offset=arena_offset,
                        arena_bytes=arena_bytes),
        int(topk_idx.shape[0]),
        prepare_error,
    )
    gate1 = _gather_objects(gate1_payload, control_group)
    gate1_statuses = tuple(int(item[3]) for item in gate1)  # type: ignore[index]
    gate1_signatures = tuple(item[4] for item in gate1)  # type: ignore[index]
    gate1_tokens = tuple(int(item[5]) for item in gate1)  # type: ignore[index]
    assert gate1_statuses == (0,) * _WORLD_SIZE, gate1
    assert len(set(gate1_signatures)) == 1, gate1
    assert gate1_tokens == case.num_tokens_per_rank

    # Gate #1 commits exactly one local-only LSA barrier in finish, including
    # the capacity-failure path.  No rank may skip this call after consensus.
    outputs, finish_status, finish_error = _finish(
        runtime, invocation_id, stream)
    gate2_payload = (
        _OPERATION,
        2,
        invocation_id,
        finish_status,
        finish_error,
    )
    gate2 = _gather_objects(gate2_payload, control_group)
    gate2_epochs = tuple(int(item[2]) for item in gate2)  # type: ignore[index]
    gate2_statuses = tuple(int(item[3]) for item in gate2)  # type: ignore[index]
    expected_status = 0 if _schedule(case).enabled else 1
    assert gate2_epochs == (invocation_id,) * _WORLD_SIZE
    assert gate2_statuses == (expected_status,) * _WORLD_SIZE, gate2
    assert outputs is not None

    # Release pending transaction state before any local comparison can fail.
    _abort(runtime, invocation_id, control_group, timeout)

    verification_error = None
    digest = None
    try:
        digest = _verify_outputs(outputs, _schedule(case), rank)
    except BaseException:
        verification_error = traceback.format_exc()
    verification = _gather_objects(
        (verification_error, digest), control_group)
    messages = [
        f"rank {index}:\n{item[0]}"
        for index, item in enumerate(verification)  # type: ignore[index]
        if item[0] is not None  # type: ignore[index]
    ]
    if messages:
        raise AssertionError(
            f"{case.name} exact comparison failed:\n" + "\n".join(messages))
    digests = tuple(item[1] for item in verification)  # type: ignore[index]
    assert len(set(digests)) == 1
    _monitored_barrier(control_group, timeout)


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int,
            args: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(
        local_rank, num_local_ranks, seed=80)
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(seconds=args.timeout),
    )
    assert rank == local_rank and world_size == _WORLD_SIZE
    buffer = None
    clean_shutdown = False
    try:
        success, capacity_failure, boundary = _assert_oracle_contract()
        layout = tuple(int(value) for value in
                       _C._get_rail_balance_hybrid_layout(
                           _HIDDEN, _MAX_TOPK, _ARENA_PROXY_CAPACITY))
        arena_bytes = layout[-1]
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
            num_gpu_timeout_secs=args.timeout,
            num_cpu_timeout_secs=args.timeout,
        )
        assert buffer.get_logical_domain_size() == (1, _WORLD_SIZE)
        assert buffer.scaleout_rank_idx == 0
        assert buffer.scaleup_rank_idx == rank
        runtime = buffer.runtime

        # Three accepted finishes plus a capacity-rejected finish prove that
        # the shared legacy barrier phase is monotonic and reusable.
        _run_transaction(
            runtime=runtime, rank=rank, control_group=control_group,
            timeout=args.timeout, case=success, arena_offset=arena_offset,
            arena_bytes=arena_bytes, invocation_id=801)
        _run_transaction(
            runtime=runtime, rank=rank, control_group=control_group,
            timeout=args.timeout,
            case=replace(success, name="phase_reuse_seed63",
                         remainder_seed=63),
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=802)
        _run_transaction(
            runtime=runtime, rank=rank, control_group=control_group,
            timeout=args.timeout, case=capacity_failure,
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=803)
        _run_transaction(
            runtime=runtime, rank=rank, control_group=control_group,
            timeout=args.timeout,
            case=replace(success, name="after_capacity_failure",
                         remainder_seed=64),
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=804)

        # One route-invalid rank must stop at Gate #1 and consume no LSA phase.
        _run_gate1_failure(
            runtime=runtime, rank=rank, control_group=control_group,
            timeout=args.timeout,
            case=replace(success, name="invalid_route", remainder_seed=65),
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=805,
            expected_statuses=(0, 2, 0, 0, 0, 0, 0, 0),
            invalid_route_rank=1)
        _run_transaction(
            runtime=runtime, rank=rank, control_group=control_group,
            timeout=args.timeout,
            case=replace(success, name="after_route_abort", remainder_seed=65),
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=806)

        # Rank five has N=0 and a different C.  Local prepare is valid, but
        # fixed-signature consensus rejects the call before the LSA barrier.
        mismatch_base = replace(
            success, name="config_mismatch", remainder_seed=66)
        mismatch = replace(mismatch_base, num_channels=3)
        _run_gate1_failure(
            runtime=runtime, rank=rank, control_group=control_group,
            timeout=args.timeout, case=mismatch_base,
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=807,
            expected_statuses=(0,) * _WORLD_SIZE,
            mismatched_case=mismatch)
        _run_transaction(
            runtime=runtime, rank=rank, control_group=control_group,
            timeout=args.timeout,
            case=replace(success, name="after_config_abort", remainder_seed=66),
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=808, nondefault_stream=True)

        # This is the force-v1 dimensional boundary and also exercises a
        # nonnegative int64 seed on a non-default caller stream.
        _run_transaction(
            runtime=runtime, rank=rank, control_group=control_group,
            timeout=args.timeout, case=boundary,
            arena_offset=arena_offset, arena_bytes=arena_bytes,
            invocation_id=809, nondefault_stream=True)

        _monitored_barrier(control_group, args.timeout)
        clean_shutdown = True
    finally:
        # ElasticBuffer.destroy is collective.  Enter only after the final
        # Gloo checkpoint; unexpected device failures are left to mp.spawn and
        # the outer subprocess watchdog rather than risking a second deadlock.
        if clean_shutdown and buffer is not None:
            buffer.destroy()
            buffer = None
            _monitored_barrier(control_group, args.timeout)
            if rank == 0:
                print(
                    "PASS C080-B2 local LSA plan transaction: variable/zero "
                    "N, exact 14-tensor snapshots, reusable barrier phase, "
                    "Gate1 route/config abort recovery, Gate2 capacity "
                    "recovery, C1024/D32, non-default stream",
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
            f"C080-B2 subprocess watchdog expired after "
            f"{arguments.watchdog_seconds}s") from error
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C080-B2 strict 8-GPU local-LSA Hybrid plan test")
    if not __debug__:
        parser.error("C080-B2 correctness checks require Python assertions")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--worker-suite", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--master-port", type=int, default=29882)
    parser.add_argument("--watchdog-seconds", type=int, default=900)
    arguments = parser.parse_args()
    if arguments.num_processes != _WORLD_SIZE:
        parser.error("C080-B2 requires exactly 8 processes")
    if arguments.timeout <= 0 or arguments.watchdog_seconds <= 0:
        parser.error("timeouts must be positive")

    success, failure, boundary = _assert_oracle_contract()
    print(
        "PASS C080-B2 CPU oracle: C061-like 8-rail skew "
        f"{success.num_tokens_per_rank}, moved=21, Pcap=5/4, "
        f"boundary C={boundary.num_channels}/D={boundary.num_scaleout_ranks}",
        flush=True,
    )
    assert not _schedule(failure).enabled
    if arguments.oracle_only:
        return

    required = (
        "_rail_balance_hybrid_plan_prepare",
        "_rail_balance_hybrid_plan_finish",
        "_rail_balance_hybrid_plan_abort",
    )
    runtime_type = getattr(_C, "ElasticBuffer", None)
    missing = [name for name in required
               if runtime_type is None or not hasattr(runtime_type, name)]
    if missing:
        parser.error(
            "rebuild the extension with the C080-B2 private API; missing "
            + ", ".join(missing))
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("C080-B2 requires at least 8 visible CUDA devices")

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
