"""C080-D: strict 8-GPU Hybrid source-shuffle correctness test.

This test deliberately exercises only the new node-local data path.  Eight
physical GPUs form one source node; destination servers remain a logical
namespace.  A compact B2 plan is built from real expert IDs, then every moved
``(token, destination)`` copy is written directly into its target egress
GPU's legacy Hybrid proxy-dispatch slot through LSA.

There is no manifest, descriptor, ready flag, or ring in this contract.  The
test parses the raw legacy BF16 ``TokenLayout`` bytes and compares them with
``enumerate_resolved_copies``.  A snapshot taken before the shuffle also
proves that slots outside the dense active prefix remain byte-for-byte
unchanged without imposing a production memset.

CPU contract::

    PYTHONPATH=. python -B \
      tests/elastic/test_rail_balance_hybrid_shuffle_lsa.py --oracle-only

Strict H200 path::

    PYTHONPATH=. EP_DISABLE_GIN=1 \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      python -B tests/elastic/test_rail_balance_hybrid_shuffle_lsa.py
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import align
from rail_balance_hybrid_reference import (
    HybridRailSchedule,
    ResolvedHybridDestinationCopy,
    enumerate_resolved_copies,
    map_topk_experts_to_destinations,
)
from test_rail_balance_hybrid_plan_lsa import (
    PlanCase,
    _abort,
    _encode_destination_rows,
    _finish,
    _gather_objects,
    _local_topk,
    _monitored_barrier,
    _schedule,
    _skew_case,
    _verify_outputs,
)


_WORLD_SIZE = 8
_SMALL_HIDDEN = 256
_LARGE_HIDDEN = 7168
_NUM_TOPK = 4
_MAX_TOKENS = 10
_PROXY_CAPACITY = 6
_C100_NUM_TOKENS = 1024
_C100_NUM_CHANNELS = 256
_C100_NUM_DESTINATIONS = _WORLD_SIZE + 1
_C100_PROXY_CAPACITY = _C100_NUM_TOKENS * (_WORLD_SIZE - 1) // _WORLD_SIZE
_C100_CASE_NAMES = ("c100_volume_h256", "c100_volume_h7168")
_TMA_ALIGNMENT = 32
_STREAM_DELAY_CYCLES = 2_000_000
_OPERATION = "rail_balance_hybrid_source_shuffle"
_TYPE = TypeVar("_TYPE")


@dataclass(frozen=True)
class ShuffleCase:
    plan: PlanCase
    hidden: int
    iteration: int
    nondefault_stream: bool
    expected_moved_copies: int


def _align(value: int, alignment: int = _TMA_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _mapped_destinations(
    case: PlanCase,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return map_topk_experts_to_destinations(
        case.topk_idx,
        num_topk=case.num_topk,
        num_experts=case.num_experts,
        num_scaleout_ranks=case.num_scaleout_ranks,
        local_scaleout_rank=case.local_scaleout_rank,
    )


def _moved_copies(
    case: PlanCase,
    schedule: HybridRailSchedule,
) -> tuple[ResolvedHybridDestinationCopy, ...]:
    return tuple(
        record
        for record in enumerate_resolved_copies(
            _mapped_destinations(case), schedule)
        if record.resolution.moved
    )


def _wide_destination_case() -> PlanCase:
    destinations = (1, 5, 9, 13, 17, 21, 25, 31)
    experts_per_destination = _WORLD_SIZE
    owners = []
    for destination in destinations:
        rows = ((destination,) * _NUM_TOPK,) * 4
        owners.append(_encode_destination_rows(
            rows,
            num_destinations=32,
            experts_per_destination=experts_per_destination,
        ))
    return PlanCase(
        name="all_owner_c1024_d32_k4",
        topk_idx=tuple(owners),
        num_topk=_NUM_TOPK,
        num_channels=1024,
        num_max_tokens_per_rank=_MAX_TOKENS,
        num_experts=32 * experts_per_destination,
        num_scaleout_ranks=32,
        local_scaleout_rank=0,
        proxy_capacity_per_egress=_PROXY_CAPACITY,
        remainder_seed=17,
    )


def _balanced_case() -> PlanCase:
    # Every owner contributes exactly two copies to each remote destination,
    # so quota equals count and the source shuffle must publish no bytes.
    owners = []
    for owner in range(_WORLD_SIZE):
        token = (owner, 8 + owner, 16 + owner, 24 + owner)
        owners.append((token, token))
    return PlanCase(
        name="balanced_zero_move",
        topk_idx=tuple(owners),
        num_topk=_NUM_TOPK,
        num_channels=2,
        num_max_tokens_per_rank=_MAX_TOKENS,
        num_experts=32,
        num_scaleout_ranks=4,
        local_scaleout_rank=0,
        proxy_capacity_per_egress=_PROXY_CAPACITY,
        remainder_seed=67,
    )


def _single_token_case() -> PlanCase:
    # The count-aware remainder policy keeps each of this one token's remote
    # copies on its owner.  Seven ranks exercise the zero-N launch path.
    return PlanCase(
        name="single_global_token_zero_move",
        topk_idx=(((0, 8, 16, 24),),) + ((),) * (_WORLD_SIZE - 1),
        num_topk=_NUM_TOPK,
        num_channels=2,
        num_max_tokens_per_rank=_MAX_TOKENS,
        num_experts=32,
        num_scaleout_ranks=4,
        local_scaleout_rank=0,
        proxy_capacity_per_egress=_PROXY_CAPACITY,
        remainder_seed=68,
    )


def _c100_volume_case(name: str) -> PlanCase:
    """Build one controlled all-rank payload-scaling fixture.

    Owner ``g`` sends every token to remote destination ``g + 1``.  With 1024
    tokens and eight rails, each destination keeps 128 records on its owner
    and moves 128 records to each of the other seven egresses.  Thus every
    producer and consumer executes the same 896-record data path.
    """
    assert _C100_NUM_TOKENS % _WORLD_SIZE == 0
    experts_per_destination = _WORLD_SIZE
    owners = []
    for owner in range(_WORLD_SIZE):
        destination = owner + 1
        rows = ((destination,) * _NUM_TOPK,) * _C100_NUM_TOKENS
        owners.append(_encode_destination_rows(
            rows,
            num_destinations=_C100_NUM_DESTINATIONS,
            experts_per_destination=experts_per_destination,
        ))
    return PlanCase(
        name=name,
        topk_idx=tuple(owners),
        num_topk=_NUM_TOPK,
        num_channels=_C100_NUM_CHANNELS,
        num_max_tokens_per_rank=_C100_NUM_TOKENS,
        num_experts=_C100_NUM_DESTINATIONS * experts_per_destination,
        num_scaleout_ranks=_C100_NUM_DESTINATIONS,
        local_scaleout_rank=0,
        proxy_capacity_per_egress=_C100_PROXY_CAPACITY,
        remainder_seed=100,
    )


def _c100_profile_cases() -> tuple[ShuffleCase, ShuffleCase]:
    small_plan = _c100_volume_case("c100_volume_h256")
    large_plan = replace(small_plan, name="c100_volume_h7168")
    small = ShuffleCase(
        small_plan, _SMALL_HIDDEN, 100, False,
        _C100_PROXY_CAPACITY * _WORLD_SIZE,
    )
    large = ShuffleCase(
        large_plan, _LARGE_HIDDEN, 100, False,
        _C100_PROXY_CAPACITY * _WORLD_SIZE,
    )
    small_schedule = _schedule(small.plan)
    large_schedule = _schedule(large.plan)
    assert small_schedule == large_schedule
    assert small.plan.topk_idx == large.plan.topk_idx
    assert small.plan.remainder_seed == large.plan.remainder_seed
    assert small.plan.num_tokens_per_rank == \
        (_C100_NUM_TOKENS,) * _WORLD_SIZE
    assert small_schedule.enabled and small_schedule.failure_reason is None
    assert small_schedule.moved_copies == 7168
    assert small_schedule.proxy_required == \
        (_C100_PROXY_CAPACITY,) * _WORLD_SIZE
    assert small_schedule.num_segments == \
        (0,) + (_WORLD_SIZE - 1,) * _WORLD_SIZE
    records = _moved_copies(small.plan, small_schedule)
    assert Counter(record.copy.owner for record in records) == \
        Counter({_owner: _C100_PROXY_CAPACITY
                 for _owner in range(_WORLD_SIZE)})
    assert Counter(record.resolution.egress for record in records) == \
        Counter({_egress: _C100_PROXY_CAPACITY
                 for _egress in range(_WORLD_SIZE)})
    for owner in range(_WORLD_SIZE):
        destination = owner + 1
        owner_records = [
            record for record in records if record.copy.owner == owner
        ]
        assert len({record.resolution.source_channel
                    for record in owner_records}) == \
            _C100_NUM_CHANNELS * (_WORLD_SIZE - 1) // _WORLD_SIZE
        assert small_schedule.count[owner][destination] == \
            _C100_NUM_TOKENS
        assert small_schedule.keep_count[owner][destination] == \
            _C100_NUM_TOKENS // _WORLD_SIZE
    return small, large


def _copies_by_egress(
    records: Sequence[ResolvedHybridDestinationCopy],
) -> tuple[dict[int, ResolvedHybridDestinationCopy], ...]:
    result: list[dict[int, ResolvedHybridDestinationCopy]] = [
        {} for _ in range(_WORLD_SIZE)
    ]
    for record in records:
        resolution = record.resolution
        slots = result[resolution.egress]
        assert resolution.proxy_slot not in slots
        slots[resolution.proxy_slot] = record
    return tuple(result)


def _assert_shuffle_oracle(
    *,
    include_c100_profiles: bool = False,
) -> tuple[ShuffleCase, ...]:
    first = _skew_case(
        capacity=_PROXY_CAPACITY,
        seed=62,
        name="c061_source_shuffle_seed62",
    )
    second = replace(
        first,
        name="c061_source_shuffle_reuse_seed63",
        remainder_seed=63,
    )
    wide = _wide_destination_case()
    balanced = _balanced_case()
    single = _single_token_case()
    large = replace(
        first,
        name="c061_h7168_moved_seed64",
        remainder_seed=64,
    )
    assert first.num_topk == second.num_topk == _NUM_TOPK
    assert first.num_max_tokens_per_rank == _MAX_TOKENS
    assert first.num_tokens_per_rank == (9, 9, 0, 0, 0, 0, 0, 0)

    cases = (
        ShuffleCase(first, _SMALL_HIDDEN, 1, False, 21),
        ShuffleCase(second, _SMALL_HIDDEN, 2, True, 21),
        ShuffleCase(wide, _SMALL_HIDDEN, 3, True, 24),
        ShuffleCase(balanced, _SMALL_HIDDEN, 4, False, 0),
        ShuffleCase(single, _SMALL_HIDDEN, 5, True, 0),
        ShuffleCase(large, _LARGE_HIDDEN, 6, True, 21),
        # Reinterpret the same tail arena back to the small layout after the
        # large transaction; no stale large-layout payload may stay live.
        ShuffleCase(first, _SMALL_HIDDEN, 7, True, 21),
    )
    if include_c100_profiles:
        cases += _c100_profile_cases()
    for spec in cases:
        case = spec.plan
        schedule = _schedule(case)
        records = _moved_copies(case, schedule)
        by_egress = _copies_by_egress(records)
        assert schedule.enabled and schedule.moved_copies == len(records)
        assert schedule.moved_copies == spec.expected_moved_copies
        assert max(schedule.proxy_required) <= case.proxy_capacity_per_egress
        for egress, slots in enumerate(by_egress):
            required = schedule.proxy_required[egress]
            assert set(slots) == set(range(required))
            for proxy_slot, record in slots.items():
                resolution = record.resolution
                assert resolution.egress == egress
                assert resolution.proxy_slot == proxy_slot
                assert proxy_slot == (
                    schedule.group_prefix[egress][resolution.channel][
                        record.copy.destination]
                    + resolution.incoming_ordinal
                    - schedule.moved_channel_prefix[egress][
                        record.copy.destination][resolution.channel]
                )

        if case.name.startswith("c061"):
            # The C061 route is not merely one-hot: one physical owner token
            # can contribute multiple independently balanced destinations.
            multiplicity = Counter(
                (record.copy.owner, record.copy.token) for record in records)
            assert max(multiplicity.values()) >= 2
            multi = [
                key for key, count in multiplicity.items() if count >= 2
            ]
            assert any(
                len({
                    record.copy.destination
                    for record in records
                    if (record.copy.owner, record.copy.token) == key
                }) >= 2
                for key in multi
            )

    wide_schedule = _schedule(wide)
    assert wide.num_scaleout_ranks == 32 > wide.num_topk == 4
    assert wide.num_channels == 1024
    assert wide.num_tokens_per_rank == (4,) * _WORLD_SIZE
    assert wide_schedule.moved_copies == 24
    assert wide_schedule.proxy_required == (4, 1, 4, 4, 3, 2, 3, 3)
    assert max(_schedule(second).proxy_required) == _PROXY_CAPACITY
    assert _schedule(balanced).proxy_required == (0,) * _WORLD_SIZE
    assert balanced.num_tokens_per_rank == (2,) * _WORLD_SIZE
    assert _schedule(single).proxy_required == (0,) * _WORLD_SIZE
    assert single.num_tokens_per_rank == (1, 0, 0, 0, 0, 0, 0, 0)
    assert any(
        spec.hidden == _LARGE_HIDDEN and spec.expected_moved_copies > 0
        for spec in cases
    )
    return cases


def _cpu_source_inputs(
    case: PlanCase,
    owner: int,
    iteration: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = len(case.topk_idx[owner])
    linear = torch.arange(
        num_tokens * hidden, dtype=torch.int32, device="cpu")
    x = (
        linear.reshape(num_tokens, hidden) % 29
        + owner * 31
        + iteration * 3
    ).to(torch.bfloat16).contiguous()
    token = torch.arange(
        num_tokens, dtype=torch.float32, device="cpu")[:, None]
    lane = torch.arange(
        case.num_topk, dtype=torch.float32, device="cpu")[None, :]
    weights = (
        iteration * 10000.0 + owner * 1000.0 + token * 32.0 + lane
    ).contiguous()
    assert x.shape == (num_tokens, hidden)
    assert weights.shape == (num_tokens, case.num_topk)
    return x, weights


def _local_source_inputs(
    case: PlanCase,
    rank: int,
    iteration: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, weights = _cpu_source_inputs(case, rank, iteration, hidden)
    device = torch.device("cuda", rank)
    return x.to(device), weights.to(device)


def _token_layout(
    case: PlanCase,
    hidden: int,
) -> tuple[int, int, int, int, int]:
    hidden_bytes = hidden * torch.bfloat16.itemsize
    metadata_offset = _align(hidden_bytes)
    topk_offset = metadata_offset
    weights_offset = topk_offset + case.num_topk * 4
    src_global_offset = weights_offset + case.num_topk * 4
    linked_offset = src_global_offset + 4
    metadata_bytes = case.num_topk * (4 + 4) + (1 + case.num_topk) * 4
    token_bytes = _align(hidden_bytes) + _align(metadata_bytes)
    assert linked_offset + case.num_topk * 4 <= token_bytes
    return (
        topk_offset,
        weights_offset,
        src_global_offset,
        linked_offset,
        token_bytes,
    )


def _reinterpret(
    row: torch.Tensor,
    offset: int,
    count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_bytes = count * torch.empty((), dtype=dtype).element_size()
    return row.narrow(0, offset, num_bytes).contiguous().view(dtype)


def _verify_local_snapshot(
    *,
    case: PlanCase,
    schedule: HybridRailSchedule,
    records: Sequence[ResolvedHybridDestinationCopy],
    rank: int,
    baseline: torch.Tensor,
    snapshot: torch.Tensor,
    iteration: int,
    hidden: int,
) -> tuple[tuple[int, ...], ...]:
    assert baseline.device.type == snapshot.device.type == "cpu"
    assert baseline.dtype == snapshot.dtype == torch.uint8
    assert baseline.is_contiguous() and snapshot.is_contiguous()
    topk_offset, weights_offset, src_offset, linked_offset, token_bytes = \
        _token_layout(case, hidden)
    assert tuple(baseline.shape) == tuple(snapshot.shape) == (
        case.proxy_capacity_per_egress, token_bytes)

    by_slot = _copies_by_egress(records)[rank]
    required = schedule.proxy_required[rank]
    assert set(by_slot) == set(range(required))
    assert required <= case.proxy_capacity_per_egress
    assert torch.equal(snapshot[required:], baseline[required:])

    coverage = []
    expected_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for proxy_slot in range(required):
        record = by_slot[proxy_slot]
        copy = record.copy
        resolution = record.resolution
        assert resolution.egress == rank
        assert resolution.proxy_slot == proxy_slot
        assert proxy_slot < schedule.proxy_required[rank]
        assert proxy_slot == (
            schedule.group_prefix[rank][resolution.channel][copy.destination]
            + resolution.incoming_ordinal
            - schedule.moved_channel_prefix[rank][copy.destination][
                resolution.channel]
        )

        row = snapshot[proxy_slot]
        if copy.owner not in expected_sources:
            expected_sources[copy.owner] = _cpu_source_inputs(
                case, copy.owner, iteration, hidden)
        expected_x, expected_weights = expected_sources[copy.owner]
        hidden_values = _reinterpret(row, 0, hidden, torch.bfloat16)
        topk_idx = _reinterpret(
            row, topk_offset, case.num_topk, torch.int32)
        topk_weights = _reinterpret(
            row, weights_offset, case.num_topk, torch.float32)
        src_global = int(_reinterpret(
            row, src_offset, 1, torch.int32)[0].item())
        linked = _reinterpret(
            row, linked_offset, case.num_topk, torch.int32)

        assert torch.equal(hidden_values, expected_x[copy.token])
        assert topk_idx.tolist() == list(case.topk_idx[copy.owner][copy.token])
        assert torch.equal(topk_weights, expected_weights[copy.token])
        assert src_global == copy.owner * case.num_max_tokens_per_rank + copy.token
        assert linked.tolist() == [proxy_slot] + [-1] * (case.num_topk - 1)
        metadata_end = linked_offset + case.num_topk * 4
        assert torch.count_nonzero(row[metadata_end:token_bytes]).item() == 0
        coverage.append((
            copy.owner,
            copy.token,
            copy.destination,
            resolution.source_channel,
            resolution.owner_ordinal,
            rank,
            resolution.channel,
            resolution.remote_slot,
            proxy_slot,
        ))
    return tuple(coverage)


def _checked_phase(
    label: str,
    control_group: dist.ProcessGroup,
    function: Callable[[], _TYPE],
) -> _TYPE:
    value = None
    error = None
    try:
        value = function()
    except BaseException:
        error = traceback.format_exc()
    errors = _gather_objects(error, control_group)
    messages = [
        f"rank {rank}:\n{message}"
        for rank, message in enumerate(errors)
        if message is not None
    ]
    if messages:
        raise AssertionError(f"{label} failed:\n" + "\n".join(messages))
    return value  # type: ignore[return-value]


def _snapshot(
    runtime: object,
    invocation_id: int,
    stream: torch.cuda.Stream | None,
) -> torch.Tensor:
    def call() -> torch.Tensor:
        value = runtime._rail_balance_hybrid_proxy_dispatch_snapshot(  # type: ignore[attr-defined]
            invocation_id)
        assert isinstance(value, torch.Tensor)
        return value

    if stream is None:
        value = call()
    else:
        with torch.cuda.stream(stream):
            value = call()
        # The snapshot copy is queued on ``stream``.  Do not consume the CUDA
        # tensor from the caller's default stream without an explicit edge.
        stream.synchronize()
    assert value.is_cuda and value.dtype == torch.uint8
    assert value.is_contiguous() and value.ndim == 2
    return value.cpu().contiguous()


def _source_shuffle(
    runtime: object,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    invocation_id: int,
    stream: torch.cuda.Stream | None,
) -> None:
    def call() -> None:
        result = runtime._rail_balance_hybrid_source_shuffle(  # type: ignore[attr-defined]
            x, topk_weights, invocation_id)
        assert result is None

    if stream is None:
        call()
    else:
        with torch.cuda.stream(stream):
            call()


def _prepare_case(
    runtime: object,
    topk_idx: torch.Tensor,
    case: PlanCase,
    hidden: int,
    *,
    arena_offset: int,
    invocation_id: int,
    stream: torch.cuda.Stream | None,
) -> tuple[int, str | None]:
    try:
        def call() -> int:
            return int(runtime._rail_balance_hybrid_plan_prepare(  # type: ignore[attr-defined]
                topk_idx,
                hidden,
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
        return 100, traceback.format_exc()


def _gate_signature(
    case: PlanCase,
    hidden: int,
    *,
    arena_offset: int,
    arena_bytes: int,
) -> tuple[object, ...]:
    # Local N intentionally differs across ranks and is gated separately.
    return (
        _OPERATION,
        _WORLD_SIZE,
        hidden,
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


def _run_shuffle_transaction(
    *,
    runtime: object,
    rank: int,
    control_group: dist.ProcessGroup,
    timeout: int,
    case: PlanCase,
    hidden: int,
    arena_offset: int,
    arena_bytes: int,
    invocation_id: int,
    iteration: int,
    nondefault_stream: bool,
) -> None:
    schedule = _schedule(case)
    records = _moved_copies(case, schedule)
    stream = torch.cuda.Stream(device=rank) if nondefault_stream else None
    if stream is None:
        topk_idx = _local_topk(case, rank)
        x, topk_weights = _local_source_inputs(
            case, rank, iteration, hidden)
    else:
        # Produce all inputs on the non-default caller stream.  Both prepare
        # and shuffle must explicitly order their private comm stream behind
        # it; no default-stream completion is available to hide a missing edge.
        with torch.cuda.stream(stream):
            torch.cuda._sleep(_STREAM_DELAY_CYCLES)
            topk_idx = _local_topk(case, rank)
            x, topk_weights = _local_source_inputs(
                case, rank, iteration, hidden)
    rows = case.topk_idx[rank]
    source_topk = torch.tensor(
        rows, dtype=_C.topk_idx_t, device="cpu").reshape(
            len(rows), case.num_topk)
    source_x, source_weights = _cpu_source_inputs(
        case, rank, iteration, hidden)

    try:
        status, prepare_error = _prepare_case(
            runtime,
            topk_idx,
            case,
            hidden,
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
                case, hidden,
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

        baseline = _checked_phase(
            "pre-shuffle proxy snapshot",
            control_group,
            lambda: _snapshot(runtime, invocation_id, stream),
        )
        _checked_phase(
            "direct-final LSA source shuffle",
            control_group,
            lambda: _source_shuffle(
                runtime, x, topk_weights, invocation_id, stream),
        )

        # The preceding checked phase is the host WORLD agreement that every
        # producer returned.  Enter the existing device barrier on every rank
        # in the same order to make peer LSA publication visible before any
        # egress snapshots its proxy slots.  This is test-only synchronization;
        # the production shuffle core deliberately contains no extra barrier.
        _checked_phase(
            "post-shuffle LSA visibility barrier",
            control_group,
            lambda: runtime.barrier(True, True, True),  # type: ignore[attr-defined]
        )
        snapshot = _checked_phase(
            "post-shuffle proxy snapshot",
            control_group,
            lambda: _snapshot(runtime, invocation_id, stream),
        )
        local_coverage = _checked_phase(
            "raw legacy TokenLayout comparison",
            control_group,
            lambda: _verify_local_snapshot(
                case=case,
                schedule=schedule,
                records=records,
                rank=rank,
                baseline=baseline,
                snapshot=snapshot,
                iteration=iteration,
                hidden=hidden,
            ),
        )
        coverage = _gather_objects(local_coverage, control_group)

        def verify_global_coverage() -> None:
            observed = sorted(
                item for rank_items in coverage for item in rank_items)
            expected = sorted((
                record.copy.owner,
                record.copy.token,
                record.copy.destination,
                record.resolution.source_channel,
                record.resolution.owner_ordinal,
                record.resolution.egress,
                record.resolution.channel,
                record.resolution.remote_slot,
                record.resolution.proxy_slot,
            ) for record in records)
            assert observed == expected
            assert len(observed) == len(set(observed)) == schedule.moved_copies

        _checked_phase(
            "global moved-copy coverage", control_group,
            verify_global_coverage)

        def verify_sources_unchanged() -> None:
            assert torch.equal(topk_idx.cpu(), source_topk)
            assert torch.equal(x.cpu(), source_x)
            assert torch.equal(topk_weights.cpu(), source_weights)

        _checked_phase(
            "source tensors remain immutable",
            control_group,
            verify_sources_unchanged,
        )
    finally:
        _abort(runtime, invocation_id, control_group, timeout)


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int,
            args: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(
        local_rank, num_local_ranks, seed=81)
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(seconds=args.timeout),
    )
    assert rank == local_rank and world_size == _WORLD_SIZE
    buffer = None
    clean_shutdown = False
    try:
        cases = _assert_shuffle_oracle()
        if args.case_name is not None:
            cases = _assert_shuffle_oracle(include_c100_profiles=True)
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
            assert layout[5] == _token_layout(spec.plan, spec.hidden)[-1]
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

        # Warm the same barrier before any proxy publication.  Its shared
        # workspace phase is monotonic, so every later transaction must enter
        # it exactly once and in the same order on all eight ranks.
        _checked_phase(
            "warm LSA visibility barrier",
            control_group,
            lambda: buffer.runtime.barrier(  # type: ignore[attr-defined]
                True, True, True),
        )

        for index, spec in enumerate(cases):
            _run_shuffle_transaction(
                runtime=buffer.runtime,
                rank=rank,
                control_group=control_group,
                timeout=args.timeout,
                case=spec.plan,
                hidden=spec.hidden,
                arena_offset=arena_offset,
                arena_bytes=arena_bytes,
                invocation_id=881 + index,
                iteration=spec.iteration,
                nondefault_stream=spec.nondefault_stream,
            )

        _monitored_barrier(control_group, args.timeout)
        clean_shutdown = True
    finally:
        if clean_shutdown and buffer is not None:
            buffer.destroy()
            buffer = None
            _monitored_barrier(control_group, args.timeout)
            if rank == 0:
                if len(cases) == 1 and cases[0].plan.name.startswith("c100_"):
                    spec = cases[0]
                    print(
                        "PASS C100 controlled Hybrid source fixture: "
                        f"{spec.plan.name}, true 8-GPU LSA, "
                        f"moved={spec.expected_moved_copies}, "
                        f"per-owner/egress={_C100_PROXY_CAPACITY}, "
                        "exact legacy TokenLayout bytes and immutable inputs "
                        "(functionality only)",
                        flush=True,
                    )
                else:
                    print(
                        "PASS C080-D Hybrid source shuffle: true 8-GPU LSA, "
                        "C061 skew/zero-N/multi-destination/multi-copy plus "
                        "all-owner C1024/D32/K4, H7168 moved, balanced/zero-"
                        "move and single-token cases, exact-capacity, exact "
                        "legacy TokenLayout bytes, unused unchanged, reusable "
                        "transaction, H256/H7168/H256 arena reuse, and explicit "
                        "non-default stream ordering",
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
            "C080-D subprocess watchdog expired after "
            f"{arguments.watchdog_seconds}s") from error
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C080-D strict 8-GPU Hybrid source-shuffle test")
    if not __debug__:
        parser.error("C080-D correctness checks require Python assertions")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--worker-suite", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--master-port", type=int, default=29883)
    parser.add_argument("--watchdog-seconds", type=int, default=900)
    parser.add_argument(
        "--case-name",
        help="run one named GPU case while retaining the full CPU oracle",
    )
    arguments = parser.parse_args()
    if arguments.num_processes != _WORLD_SIZE:
        parser.error("C080-D requires exactly 8 processes")
    if arguments.timeout <= 0 or arguments.watchdog_seconds <= 0:
        parser.error("timeouts must be positive")

    include_c100_profiles = arguments.oracle_only or \
        arguments.case_name in _C100_CASE_NAMES
    cases = _assert_shuffle_oracle(
        include_c100_profiles=include_c100_profiles)
    by_name = {spec.plan.name: spec for spec in cases}
    if arguments.case_name is not None and arguments.case_name not in by_name:
        parser.error(
            "unknown --case-name; expected one of "
            + ", ".join(sorted(set(by_name) | set(_C100_CASE_NAMES))))
    first = by_name["c061_source_shuffle_seed62"].plan
    second = by_name["c061_source_shuffle_reuse_seed63"].plan
    wide = by_name["all_owner_c1024_d32_k4"].plan
    message = (
        "PASS C080-D CPU oracle: C061-like 8-rail source shuffle, "
        "zero-N owners, multi-destination/multi-copy, moved=21, "
        f"Pcap={_PROXY_CAPACITY}, seeds="
        f"{first.remainder_seed}/{second.remainder_seed}; all-owner "
        f"C={wide.num_channels}/D={wide.num_scaleout_ranks}/K={wide.num_topk}; "
        "H256/H7168/H256 reuse, balanced zero-move, global single-token"
    )
    if include_c100_profiles:
        c100_small = by_name["c100_volume_h256"]
        c100_large = by_name["c100_volume_h7168"]
        assert _schedule(c100_small.plan) == _schedule(c100_large.plan)
        message += (
            f"; C100 controlled H256/H7168 "
            f"moved={c100_small.expected_moved_copies}, "
            f"Pcap={c100_small.plan.proxy_capacity_per_egress} (named only)"
        )
    print(message, flush=True)
    if arguments.oracle_only:
        return

    required = (
        "_rail_balance_hybrid_plan_prepare",
        "_rail_balance_hybrid_plan_finish",
        "_rail_balance_hybrid_plan_abort",
        "_rail_balance_hybrid_source_shuffle",
        "_rail_balance_hybrid_proxy_dispatch_snapshot",
    )
    runtime_type = getattr(_C, "ElasticBuffer", None)
    missing = [
        name for name in required
        if runtime_type is None or not hasattr(runtime_type, name)
    ]
    if missing:
        parser.error(
            "rebuild the extension with the C080-D private API; missing "
            + ", ".join(missing))
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("C080-D requires at least 8 visible CUDA devices")

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
