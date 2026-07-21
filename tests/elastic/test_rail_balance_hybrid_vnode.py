"""C080-D: strict Hybrid source-shuffle -> vnode -> combine proof.

Eight H200s are split into a prefix source node and one or more virtual
destination nodes.  The source ranks use their own NCCL communicator and
``ElasticBuffer`` to build the production compact plan and publish moved
copies.  A separate eight-rank communicator owns the vnode arena.  The only
objects crossing that boundary are ordinary owning CUDA tensors.

The test deliberately keeps both symmetric windows serial:

``source plan/shuffle/snapshot -> world B0..B5 -> source unshuffle/epilogue``.

CPU contract::

    PYTHONPATH=. python -B \
      tests/elastic/test_rail_balance_hybrid_vnode.py --oracle-only

Strict H200 path::

    PYTHONPATH=. EP_DISABLE_GIN=1 \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      python -B tests/elastic/test_rail_balance_hybrid_vnode.py

For Nsight Systems add ``--nvtx`` and select one case with ``--case-name``.
The private bridge remains a single-node correctness vehicle; it is not a
replacement for a real multi-node GIN/RDMA measurement.
"""

from __future__ import annotations

import argparse
import os
import signal
import struct
import subprocess
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator, Sequence, TypeVar

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import align
from rail_balance_hybrid_vnode_reference import (
    VnodeCopyRoute,
    VnodeRoundTripCase,
    VnodeRoundTripResult,
    VnodeTopology,
    build_2x4_case,
    build_4x2_case,
    build_rounding_sensitive_case,
    run_vnode_roundtrip,
)
from rail_balance_hybrid_reference import build_hybrid_rail_schedule
from test_rail_balance_hybrid_plan_lsa import (
    _expected_outputs as _expected_plan_outputs,
    _snapshot_digest as _plan_digest,
)


_WORLD_SIZE = 8
_SMALL_HIDDEN = 256
_LARGE_HIDDEN = 7168
_ARENA_ALIGNMENT = 2_097_152
_TMA_ALIGNMENT = 32
_ARENA_GUARD_BYTES = 4096
_HEAD_CANARY = 0xDEADBEEF
_TAIL_CANARY = 0xC001D00D
_PLAN_FAULT_CASE = "hybrid_vnode_plan_faults_h256"
_TYPE = TypeVar("_TYPE")


@dataclass(frozen=True)
class GpuCase:
    name: str
    fixture: str
    hidden: int
    generation: int
    non_default_stream: bool = False
    repetitions: int = 1


@dataclass(frozen=True)
class RecordLayout:
    hidden_bytes: int
    metadata_offset: int
    topk_offset: int
    weights_offset: int
    src_global_offset: int
    linked_offset: int
    dispatch_token_bytes: int
    combine_token_bytes: int
    descriptor_offset: int
    tail_offset: int
    record_bytes: int
    physical_capacity: int
    rail_capacity: int
    expert_capacity: int
    vnode_arena_bytes: int


def _specs() -> tuple[GpuCase, ...]:
    return (
        GpuCase("hybrid_vnode_4x2_h256", "4x2", 256, 801),
        GpuCase("hybrid_vnode_4x2_h7168", "4x2", 7168, 802),
        GpuCase("hybrid_vnode_2x4_h256", "2x4", 256, 803),
        GpuCase("hybrid_vnode_2x4_h7168", "2x4", 7168, 804),
        GpuCase("hybrid_vnode_rounding_h256", "rounding", 256, 805),
        GpuCase(
            "hybrid_vnode_4x2_empty_egress_h256",
            "4x2_empty_egress",
            256,
            806,
            non_default_stream=True,
            repetitions=2,
        ),
        GpuCase(
            "hybrid_vnode_all_zero_h256",
            "all_zero",
            256,
            808,
        ),
        GpuCase(_PLAN_FAULT_CASE, "4x2", 256, 809),
    )


def _build_4x2_empty_egress_case() -> VnodeRoundTripCase:
    base = build_4x2_case()
    return replace(
        base,
        name="hybrid_vnode_4x2_empty_egress",
        topk_idx=(base.topk_idx[0], (), (), ()),
        topk_weights=(base.topk_weights[0], (), (), ()),
        source_values=(base.source_values[0], (), (), ()),
    )


def _build_all_zero_case() -> VnodeRoundTripCase:
    return VnodeRoundTripCase(
        name="hybrid_vnode_all_zero",
        topology=VnodeTopology(rails_per_node=2, num_nodes=4),
        topk_idx=((), ()),
        topk_weights=((), ()),
        source_values=((), ()),
        num_topk=2,
        num_channels=1,
        num_max_tokens_per_rank=1,
        num_experts=16,
        experts_per_physical_rank=2,
        proxy_capacity_per_egress=1,
        remainder_seed=0,
    )


def _base_case(name: str) -> VnodeRoundTripCase:
    builders = {
        "4x2": build_4x2_case,
        "4x2_empty_egress": _build_4x2_empty_egress_case,
        "all_zero": _build_all_zero_case,
        "2x4": build_2x4_case,
        "rounding": build_rounding_sensitive_case,
    }
    return builders[name]()


def _expanded_case(spec: GpuCase) -> VnodeRoundTripCase:
    base = _base_case(spec.fixture)
    if spec.fixture == "rounding":
        oracle_width = 3
        source_values = tuple(
            tuple(
                (Fraction(0),) * oracle_width
                for _ in owner_tokens
            )
            for owner_tokens in base.topk_idx
        )
    else:
        # Routing and expert transforms are column-independent.  Eight columns
        # cover the complete source pattern; GPU expectations tile these bytes
        # to H256/H7168 instead of materializing millions of Fraction objects
        # in every spawned worker.
        oracle_width = 8
        source_values = tuple(
            tuple(
                tuple(Fraction(
                    8 * ((5 * owner + 3 * token + (column & 7)) & 7)
                ) for column in range(oracle_width))
                for token in range(len(base.topk_idx[owner]))
            )
            for owner in range(base.topology.rails_per_node)
        )
    return replace(base, name=spec.name, source_values=source_values)


def _oracle(spec: GpuCase) -> VnodeRoundTripResult:
    expanded = _expanded_case(spec)
    if spec.fixture == "all_zero":
        schedule = build_hybrid_rail_schedule(
            expanded.topk_idx,
            num_topk=expanded.num_topk,
            num_experts=expanded.num_experts,
            num_scaleout_ranks=expanded.topology.num_nodes,
            local_scaleout_rank=expanded.topology.source_node,
            num_channels=expanded.num_channels,
            num_max_tokens_per_rank=expanded.num_max_tokens_per_rank,
            proxy_capacity_per_egress=
                expanded.proxy_capacity_per_egress,
            remainder_seed=expanded.remainder_seed,
        )
        result = VnodeRoundTripResult(
            case=expanded,
            schedule=schedule,
            routes=(),
            contributions=(),
            reduce_rows=((), ()),
            combined=((), ()),
            direct=((), ()),
        )
    else:
        result = run_vnode_roundtrip(expanded)
    case = result.case
    g = case.topology.rails_per_node
    d = case.topology.num_nodes
    assert case.topology.world_size == _WORLD_SIZE
    assert len(result.routes) == len(result.contributions)
    assert result.schedule.enabled
    assert result.schedule.num_tokens_per_owner == \
        case.num_tokens_per_owner
    assert all(route.destination > 0 for route in result.routes)
    assert all(route.final_owner_physical == route.owner
               for route in result.routes)
    if spec.fixture == "4x2":
        assert (g, d, case.num_topk) == (4, 2, 4)
        assert d <= case.num_topk
        assert case.num_tokens_per_owner == (4, 2, 1, 1)
        assert result.schedule.moved_copies == 2
    elif spec.fixture == "4x2_empty_egress":
        assert (g, d, case.num_topk) == (4, 2, 4)
        assert case.num_tokens_per_owner == (4, 0, 0, 0)
        assert result.schedule.quota == (
            (0, 1), (0, 1), (0, 1), (0, 1))
        assert result.schedule.proxy_required == (0, 1, 1, 1)
        assert result.schedule.moved_copies == 3
        assert {
            route.egress for route in result.routes if route.moved
        } == {1, 2, 3}
        assert all(
            route.owner == 0
            for route in result.routes if route.moved
        )
    elif spec.fixture == "2x4":
        assert (g, d, case.num_topk) == (2, 4, 2)
        assert d > case.num_topk
        assert case.num_tokens_per_owner == (6, 6)
        assert result.schedule.moved_copies == 6
        assert any(route.target_channel == 1 for route in result.routes)
        assert any(route.proxy_slot == 3 for route in result.routes)
    elif spec.fixture == "all_zero":
        assert (g, d, case.num_topk) == (2, 4, 2)
        assert case.num_tokens_per_owner == (0, 0)
        assert result.schedule.count == ((0, 0, 0, 0),) * 2
        assert result.schedule.quota == ((0, 0, 0, 0),) * 2
        assert result.schedule.proxy_required == (0, 0)
        assert result.schedule.moved_copies == 0
        assert not result.routes and not result.contributions
        assert result.combined == result.direct == ((), ())
    else:
        assert (g, d, case.num_topk) == (2, 4, 3)
        assert case.num_tokens_per_owner == (1, 0)
        assert result.schedule.moved_copies == 0
        assert result.combined != result.direct
        assert result.combined[0][0][0] == Fraction(1)
        assert result.direct[0][0][0] == Fraction(129, 128)
    return result


def _assert_cpu_oracle(specs: Sequence[GpuCase]) -> None:
    for spec in specs:
        result = _oracle(spec)
        case = result.case
        assert len(result.combined) == case.topology.rails_per_node
        assert all(len(owner) == count for owner, count in zip(
            result.combined, case.num_tokens_per_owner))
        expected_width = 3 if spec.fixture == "rounding" else 8
        assert all(len(vector) == expected_width
                   for owner in result.combined for vector in owner)


def _align32(value: int) -> int:
    return align(value, _TMA_ALIGNMENT)


def _record_layout(case: VnodeRoundTripCase, hidden: int) -> RecordLayout:
    g = case.topology.rails_per_node
    d = case.topology.num_nodes
    k = case.num_topk
    m = case.num_max_tokens_per_rank
    hidden_bytes = hidden * torch.bfloat16.itemsize
    metadata_offset = _align32(hidden_bytes)
    topk_offset = metadata_offset
    weights_offset = topk_offset + k * 4
    src_global_offset = weights_offset + k * 4
    linked_offset = src_global_offset + 4
    dispatch_token_bytes = (
        _align32(hidden_bytes) + _align32(k * 12 + 4))
    combine_token_bytes = _align32(hidden_bytes) + _align32(k * 8)
    descriptor_offset = 32 + dispatch_token_bytes
    tail_offset = descriptor_offset + 64
    record_bytes = _align32(tail_offset + 32)
    physical_capacity = (d - 1) * m
    rail_capacity = physical_capacity * (k + 1)
    expert_capacity = g * m * k

    def stage_bytes(capacity: int) -> int:
        ready_offset = record_bytes * capacity
        route_offset = _align32(ready_offset + _align32(capacity * 4))
        return align(route_offset + capacity * 32, _ARENA_ALIGNMENT)

    rail_arena = stage_bytes(rail_capacity)
    expert_arena = stage_bytes(expert_capacity)
    owner_values = hidden_bytes * m * k
    owner_ready = _align32(owner_values)
    owner_arena = align(
        owner_ready + _align32(m * k * 4), _ARENA_ALIGNMENT)
    vnode_arena_bytes = rail_arena + max(expert_arena, owner_arena)
    return RecordLayout(
        hidden_bytes=hidden_bytes,
        metadata_offset=metadata_offset,
        topk_offset=topk_offset,
        weights_offset=weights_offset,
        src_global_offset=src_global_offset,
        linked_offset=linked_offset,
        dispatch_token_bytes=dispatch_token_bytes,
        combine_token_bytes=combine_token_bytes,
        descriptor_offset=descriptor_offset,
        tail_offset=tail_offset,
        record_bytes=record_bytes,
        physical_capacity=physical_capacity,
        rail_capacity=rail_capacity,
        expert_capacity=expert_capacity,
        vnode_arena_bytes=vnode_arena_bytes,
    )


def _cpu_source_inputs(
    case: VnodeRoundTripCase,
    hidden: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    owners = []
    for owner in range(case.topology.rails_per_node):
        values = case.source_values[owner]
        if values:
            pattern = torch.tensor(
                [[float(value) for value in row] for row in values],
                dtype=torch.bfloat16,
                device="cpu",
            ).contiguous()
            repeats = (hidden + pattern.size(1) - 1) // pattern.size(1)
            x = pattern.repeat(1, repeats)[:, :hidden].contiguous()
            topk_idx = torch.tensor(
                case.topk_idx[owner], dtype=_C.topk_idx_t, device="cpu",
            ).contiguous()
            weights = torch.tensor(
                [[float(value) for value in row]
                 for row in case.topk_weights[owner]],
                dtype=torch.float32,
                device="cpu",
            ).contiguous()
        else:
            x = torch.empty(
                (0, hidden), dtype=torch.bfloat16, device="cpu")
            topk_idx = torch.empty(
                (0, case.num_topk), dtype=_C.topk_idx_t, device="cpu")
            weights = torch.empty(
                (0, case.num_topk), dtype=torch.float32, device="cpu")
        owners.append((x, topk_idx, weights))
    return tuple(owners)


def _bytes_tensor(data: bytes) -> torch.Tensor:
    return torch.tensor(list(data), dtype=torch.uint8, device="cpu")


def _copy_typed_bytes(
    row: torch.Tensor,
    offset: int,
    tensor: torch.Tensor,
) -> None:
    raw = tensor.contiguous().view(torch.uint8).reshape(-1)
    row[offset:offset + raw.numel()].copy_(raw)


def _dispatch_token(
    *,
    case: VnodeRoundTripCase,
    layout: RecordLayout,
    source_inputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    route: VnodeCopyRoute,
    hidden: torch.Tensor,
) -> torch.Tensor:
    row = torch.zeros(
        layout.dispatch_token_bytes, dtype=torch.uint8, device="cpu")
    _copy_typed_bytes(row, 0, hidden)
    _copy_typed_bytes(
        row, layout.topk_offset,
        source_inputs[route.owner][1][route.token].to(torch.int32))
    _copy_typed_bytes(
        row, layout.weights_offset,
        source_inputs[route.owner][2][route.token])
    _copy_typed_bytes(
        row, layout.src_global_offset,
        torch.tensor(
            [route.owner * case.num_max_tokens_per_rank + route.token],
            dtype=torch.int32, device="cpu"))
    linked = [-1] * case.num_topk
    if route.moved:
        linked[0] = route.proxy_slot
    _copy_typed_bytes(
        row, layout.linked_offset,
        torch.tensor(linked, dtype=torch.int32, device="cpu"))
    return row


def _fingerprint(
    case: VnodeRoundTripCase,
    route: VnodeCopyRoute,
    generation: int,
) -> int:
    identity = (
        (route.owner * case.num_max_tokens_per_rank + route.token)
        * case.topology.num_nodes + route.destination)
    assert 0 <= identity < 1 << 32
    return ((generation & 0xFFFFFFFF) << 32) | identity


def _record(
    *,
    case: VnodeRoundTripCase,
    layout: RecordLayout,
    source_inputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    route: VnodeCopyRoute,
    hidden: torch.Tensor,
    generation: int,
) -> torch.Tensor:
    row = torch.zeros(layout.record_bytes, dtype=torch.uint8, device="cpu")
    row[:32].copy_(_bytes_tensor(struct.pack(
        "<8I", *([_HEAD_CANARY] * 8))))
    row[32:32 + layout.dispatch_token_bytes].copy_(_dispatch_token(
        case=case,
        layout=layout,
        source_inputs=source_inputs,
        route=route,
        hidden=hidden,
    ))
    descriptor = struct.pack(
        "<Q14i",
        _fingerprint(case, route, generation),
        route.owner,
        route.token,
        route.destination - 1,
        route.owner_ordinal,
        route.egress,
        route.target_channel,
        route.remote_slot,
        route.vnode_slot,
        generation,
        route.owner * case.num_max_tokens_per_rank + route.token,
        0, 0, 0, 0,
    )
    row[layout.descriptor_offset:layout.descriptor_offset + 64].copy_(
        _bytes_tensor(descriptor))
    row[layout.tail_offset:layout.tail_offset + 32].copy_(
        _bytes_tensor(struct.pack("<8I", *([_TAIL_CANARY] * 8))))
    return row


def _route_sidecar(
    case: VnodeRoundTripCase,
    route: VnodeCopyRoute,
    lane: int,
    generation: int,
) -> torch.Tensor:
    expert = case.topk_idx[route.owner][route.token][lane]
    expert_rank = expert // case.experts_per_physical_rank
    expert_slot = (
        (route.egress * case.num_max_tokens_per_rank + route.dense_slot)
        * case.num_topk + lane)
    return _bytes_tensor(struct.pack(
        "<Q6i",
        _fingerprint(case, route, generation),
        generation,
        route.ingress_physical,
        route.vnode_slot,
        lane,
        expert_rank,
        expert_slot,
    ))


def _expert_partial(
    case: VnodeRoundTripCase,
    source_inputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    route: VnodeCopyRoute,
    lane: int,
) -> torch.Tensor:
    source = source_inputs[route.owner][0][route.token]
    expert = int(case.topk_idx[route.owner][route.token][lane])
    expert_begin = case.num_experts // case.topology.num_nodes
    bias = float(8 * (expert - expert_begin + 1))
    weight = source_inputs[route.owner][2][route.token, lane]
    return ((source.float() + bias) * weight).to(torch.bfloat16)


def _combine_token(
    *,
    case: VnodeRoundTripCase,
    layout: RecordLayout,
    source_inputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    route: VnodeCopyRoute,
    hidden: torch.Tensor,
) -> torch.Tensor:
    row = torch.zeros(
        layout.combine_token_bytes, dtype=torch.uint8, device="cpu")
    _copy_typed_bytes(row, 0, hidden)
    _copy_typed_bytes(
        row, layout.metadata_offset,
        source_inputs[route.owner][1][route.token].to(torch.int32))
    _copy_typed_bytes(
        row, layout.metadata_offset + case.num_topk * 4,
        source_inputs[route.owner][2][route.token])
    return row


def _fraction_vector(
    values: Sequence[Fraction],
    hidden: int,
) -> torch.Tensor:
    pattern = torch.tensor(
        [float(value) for value in values],
        dtype=torch.bfloat16,
        device="cpu",
    )
    repeats = (hidden + pattern.numel() - 1) // pattern.numel()
    return pattern.repeat(repeats)[:hidden].contiguous()


def _expected_plan_digest(result: VnodeRoundTripResult) -> str:
    return _plan_digest(_expected_plan_outputs(result.schedule))


def _verify_plan(
    tensors: Sequence[torch.Tensor],
    result: VnodeRoundTripResult,
    device_index: int,
) -> str:
    expected = _expected_plan_outputs(result.schedule)
    assert len(tensors) == len(expected) == 14, \
        "world plan tensor count mismatch"
    for index, (actual, wanted) in enumerate(zip(tensors, expected)):
        assert actual.is_cuda and actual.device.index == device_index, \
            f"world plan tensor {index} device mismatch"
        assert actual.dtype == torch.int32 and actual.is_contiguous(), \
            f"world plan tensor {index} layout mismatch"
        assert tuple(actual.shape) == tuple(wanted.shape), \
            f"world plan tensor {index} shape mismatch"
        assert torch.equal(actual.cpu(), wanted), \
            f"world plan tensor {index} value mismatch"
    return _plan_digest(tensors)


def _verify_proxy_dispatch(
    *,
    baseline: torch.Tensor,
    snapshot: torch.Tensor,
    rank: int,
    result: VnodeRoundTripResult,
    source_inputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    layout: RecordLayout,
) -> None:
    case = result.case
    assert baseline.is_cuda and baseline.device.index == rank
    assert baseline.dtype == torch.uint8 and baseline.is_contiguous()
    assert snapshot.is_cuda and snapshot.device.index == rank
    assert snapshot.dtype == torch.uint8 and snapshot.is_contiguous()
    assert tuple(snapshot.shape) == (
        case.proxy_capacity_per_egress, layout.dispatch_token_bytes)
    assert tuple(baseline.shape) == tuple(snapshot.shape)
    baseline_cpu = baseline.cpu()
    cpu = snapshot.cpu()
    routes = {
        route.proxy_slot: route for route in result.routes
        if route.moved and route.egress == rank
    }
    assert set(routes) == set(range(result.schedule.proxy_required[rank]))
    required = result.schedule.proxy_required[rank]
    assert torch.equal(cpu[required:], baseline_cpu[required:])
    for proxy_slot, route in sorted(routes.items()):
        expected = _dispatch_token(
            case=case,
            layout=layout,
            source_inputs=source_inputs,
            route=route,
            hidden=source_inputs[route.owner][0][route.token],
        )
        assert torch.equal(cpu[proxy_slot], expected)


def _verify_world_outputs(
    *,
    rank: int,
    outputs: Sequence[object],
    result: VnodeRoundTripResult,
    source_inputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    layout: RecordLayout,
    generation: int,
) -> None:
    case = result.case
    g = case.topology.rails_per_node
    d = case.topology.num_nodes
    k = case.num_topk
    m = case.num_max_tokens_per_rank
    assert len(outputs) == 12
    assert isinstance(outputs[-1], (tuple, list))
    plan = tuple(outputs[-1])
    tensors = tuple(outputs[:-1])
    assert len(tensors) == 11
    assert all(isinstance(value, torch.Tensor) for value in tensors)
    for tensor in tensors:
        assert tensor.is_cuda and tensor.device.index == rank
        assert tensor.is_contiguous()
    (proxy_return, reduce_seed, compact_quota,
     rail_records, rail_ready, rail_routes,
     expert_records, expert_ready, expert_routes,
     stage_status, arena_guard) = tensors
    assert proxy_return.dtype == reduce_seed.dtype == torch.uint8
    assert compact_quota.dtype == torch.int32
    assert rail_records.dtype == rail_routes.dtype == torch.uint8
    assert expert_records.dtype == expert_routes.dtype == torch.uint8
    assert rail_ready.dtype == expert_ready.dtype == torch.int32
    assert stage_status.dtype == torch.int32
    assert arena_guard.dtype == torch.uint8
    assert tuple(proxy_return.shape) == (
        case.proxy_capacity_per_egress, layout.combine_token_bytes)
    assert tuple(reduce_seed.shape) == (
        min(d, k) * m, layout.combine_token_bytes)
    assert tuple(compact_quota.shape) == (g, d - 1)
    assert tuple(rail_records.shape) == (
        layout.rail_capacity, layout.record_bytes)
    assert tuple(rail_ready.shape) == (layout.rail_capacity,)
    assert tuple(rail_routes.shape) == (layout.rail_capacity, 32)
    assert tuple(expert_records.shape) == (
        layout.expert_capacity, layout.record_bytes)
    assert tuple(expert_ready.shape) == (layout.expert_capacity,)
    assert tuple(expert_routes.shape) == (layout.expert_capacity, 32)
    status_stride = max(
        case.num_channels,
        layout.physical_capacity * k,
        layout.expert_capacity,
    )
    assert tuple(stage_status.shape) == (6, status_stride)
    assert tuple(arena_guard.shape) == (_ARENA_GUARD_BYTES,)

    cpu = tuple(tensor.cpu() for tensor in tensors)
    (proxy_return, reduce_seed, compact_quota,
     rail_records, rail_ready, rail_routes,
     expert_records, expert_ready, expert_routes,
     stage_status, arena_guard) = cpu
    assert torch.count_nonzero(stage_status).item() == 0
    assert torch.equal(
        arena_guard,
        torch.full_like(arena_guard, 0xA5))
    expected_compact = torch.tensor(
        [row[1:] for row in result.schedule.quota],
        dtype=torch.int32,
        device="cpu",
    )
    assert torch.equal(compact_quota, expected_compact)
    assert _verify_plan(plan, result, rank) == _expected_plan_digest(result)

    contribution_values = {
        (item.route.owner, item.route.token, item.route.destination):
            _fraction_vector(item.value, layout.hidden_bytes // 2)
        for item in result.contributions
    }
    expected_proxy = torch.zeros_like(proxy_return)
    expected_seed = torch.zeros_like(reduce_seed)
    proxy_slots: set[int] = set()
    seed_slots: set[int] = set()
    if rank < g:
        for route in result.routes:
            value = contribution_values[
                (route.owner, route.token, route.destination)]
            token = _combine_token(
                case=case,
                layout=layout,
                source_inputs=source_inputs,
                route=route,
                hidden=value,
            )
            if route.moved and route.egress == rank:
                assert route.proxy_slot not in proxy_slots
                proxy_slots.add(route.proxy_slot)
                expected_proxy[route.proxy_slot] = token
            elif not route.moved and route.owner == rank:
                slot = route.reduce_row * m + route.token
                assert slot not in seed_slots
                seed_slots.add(slot)
                expected_seed[slot] = token
    assert torch.equal(proxy_return, expected_proxy)
    assert torch.equal(reduce_seed, expected_seed)

    expected_rail_records = torch.zeros_like(rail_records)
    expected_rail_ready = torch.zeros_like(rail_ready)
    expected_rail_routes = torch.zeros_like(rail_routes)
    expected_expert_records = torch.zeros_like(expert_records)
    expected_expert_ready = torch.zeros_like(expert_ready)
    expected_expert_routes = torch.zeros_like(expert_routes)
    rail_record_slots: set[int] = set()
    rail_route_slots: set[int] = set()
    expert_slots: set[int] = set()
    for route in result.routes:
        source_hidden = source_inputs[route.owner][0][route.token]
        base = _record(
            case=case,
            layout=layout,
            source_inputs=source_inputs,
            route=route,
            hidden=source_hidden,
            generation=generation,
        )
        if rank in (route.egress_physical, route.ingress_physical):
            assert route.vnode_slot not in rail_record_slots
            rail_record_slots.add(route.vnode_slot)
            expected_rail_records[route.vnode_slot] = base
            expected_rail_ready[route.vnode_slot] = generation

        for lane, expert_rank in zip(
                route.topk_lanes, route.expert_physicals):
            partial = _expert_partial(case, source_inputs, route, lane)
            contribution_slot = (
                layout.physical_capacity + route.vnode_slot * k + lane)
            sidecar = _route_sidecar(case, route, lane, generation)
            contribution_record = _record(
                case=case,
                layout=layout,
                source_inputs=source_inputs,
                route=route,
                hidden=partial,
                generation=generation,
            )
            if rank in (route.egress_physical, route.ingress_physical):
                assert contribution_slot not in rail_record_slots
                assert contribution_slot not in rail_route_slots
                rail_record_slots.add(contribution_slot)
                rail_route_slots.add(contribution_slot)
                expected_rail_records[contribution_slot] = \
                    contribution_record
                expected_rail_ready[contribution_slot] = generation
                expected_rail_routes[contribution_slot] = sidecar

            expert_slot = (
                (route.egress * m + route.dense_slot) * k + lane)
            if rank == expert_rank:
                assert expert_slot not in expert_slots
                expert_slots.add(expert_slot)
                expected_expert_records[expert_slot] = base
                expected_expert_ready[expert_slot] = generation
                expected_expert_routes[expert_slot] = sidecar

    assert torch.equal(rail_records, expected_rail_records)
    assert torch.equal(rail_ready, expected_rail_ready)
    assert torch.equal(rail_routes, expected_rail_routes)
    assert torch.equal(expert_records, expected_expert_records)
    assert torch.equal(expert_ready, expected_expert_ready)
    assert torch.equal(expert_routes, expected_expert_routes)


def _expected_reduce(
    *,
    owner: int,
    result: VnodeRoundTripResult,
    source_inputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    layout: RecordLayout,
) -> torch.Tensor:
    case = result.case
    rows = min(case.topology.num_nodes, case.num_topk)
    expected = torch.zeros(
        (rows * case.num_max_tokens_per_rank, layout.combine_token_bytes),
        dtype=torch.uint8,
        device="cpu",
    )
    occupied: set[int] = set()
    for contribution in result.contributions:
        route = contribution.route
        if route.owner != owner:
            continue
        slot = route.reduce_row * case.num_max_tokens_per_rank + route.token
        assert slot not in occupied
        occupied.add(slot)
        expected[slot] = _combine_token(
            case=case,
            layout=layout,
            source_inputs=source_inputs,
            route=route,
            hidden=_fraction_vector(
                contribution.value, layout.hidden_bytes // 2),
        )
    return expected


def _expected_combined(
    owner: int,
    result: VnodeRoundTripResult,
    hidden: int,
) -> torch.Tensor:
    values = result.combined[owner]
    if not values:
        return torch.empty(
            (0, hidden), dtype=torch.bfloat16, device="cpu")
    pattern = torch.tensor(
        [[float(value) for value in row] for row in values],
        dtype=torch.bfloat16,
        device="cpu",
    ).contiguous()
    repeats = (hidden + pattern.size(1) - 1) // pattern.size(1)
    return pattern.repeat(1, repeats)[:, :hidden].contiguous()


def _gather_objects(
    value: object,
    control_group: dist.ProcessGroup,
) -> list[object]:
    values: list[object | None] = [None] * dist.get_world_size(control_group)
    dist.all_gather_object(values, value, group=control_group)
    return values  # type: ignore[return-value]


def _monitored_barrier(
    control_group: dist.ProcessGroup,
    timeout: int,
) -> None:
    dist.monitored_barrier(
        group=control_group,
        timeout=timedelta(seconds=timeout),
        wait_all_ranks=True,
    )


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
        raise AssertionError(
            f"{label} failed on one or more ranks:\n" + "\n".join(messages))
    return value  # type: ignore[return-value]


@contextmanager
def _nvtx(enabled: bool, label: str) -> Iterator[None]:
    if enabled:
        torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


@contextmanager
def _caller_stream(
    stream: torch.cuda.Stream | None,
) -> Iterator[None]:
    if stream is None:
        yield
        return
    with torch.cuda.stream(stream):
        yield


def _layout_abis(
    case: VnodeRoundTripCase,
    hidden: int,
) -> tuple[RecordLayout, tuple[int, ...]]:
    layout = _record_layout(case, hidden)
    hybrid = tuple(int(value) for value in
                   _C._get_rail_balance_hybrid_layout(
                       hidden, case.num_topk,
                       case.proxy_capacity_per_egress))
    assert len(hybrid) == 10
    assert hybrid[5] == layout.dispatch_token_bytes
    assert hybrid[7] == layout.combine_token_bytes
    vnode = tuple(int(value) for value in
                  _C._get_rail_balance_vnode_multidst_layout(
                      hidden,
                      case.num_topk,
                      layout.physical_capacity,
                      case.num_max_tokens_per_rank,
                      case.topology.num_nodes - 1,
                      case.topology.rails_per_node,
                      case.num_max_tokens_per_rank,
                  ))
    assert len(vnode) == 13
    assert vnode[0] == layout.record_bytes
    assert vnode[1] == 32
    assert vnode[2] == layout.rail_capacity
    assert vnode[6] == layout.expert_capacity
    assert vnode[-1] == layout.vnode_arena_bytes
    return layout, hybrid


def _make_cuda_inputs(
    rank: int,
    result: VnodeRoundTripResult,
    source_inputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    case = result.case
    device = torch.device("cuda", rank)
    if rank < case.topology.rails_per_node:
        cpu = source_inputs[rank]
    else:
        cpu = (
            torch.empty(
                (0, hidden), dtype=torch.bfloat16, device="cpu"),
            torch.empty(
                (0, case.num_topk), dtype=_C.topk_idx_t, device="cpu"),
            torch.empty(
                (0, case.num_topk), dtype=torch.float32, device="cpu"),
        )
    return tuple(value.to(device) for value in cpu)  # type: ignore[return-value]


def _assert_equal(values: Sequence[object]) -> None:
    assert values
    assert all(value == values[0] for value in values)


def _abort_transactions(
    *,
    rank: int,
    num_source_ranks: int,
    source_runtime: object | None,
    world_runtime: object,
    source_invocation: int,
    world_invocation: int,
    control_group: dist.ProcessGroup,
    timeout: int,
) -> None:
    def abort_world() -> None:
        world_runtime._rail_balance_hybrid_vnode_abort(  # type: ignore[attr-defined]
            world_invocation)
        world_runtime._rail_balance_hybrid_vnode_abort(  # type: ignore[attr-defined]
            world_invocation)

    _checked_phase("idempotent world abort", control_group, abort_world)

    def abort_source() -> None:
        if rank < num_source_ranks:
            assert source_runtime is not None
            source_runtime._rail_balance_hybrid_plan_abort(  # type: ignore[attr-defined]
                source_invocation)
            source_runtime._rail_balance_hybrid_plan_abort(  # type: ignore[attr-defined]
                source_invocation)

    _checked_phase("idempotent source abort", control_group, abort_source)
    _monitored_barrier(control_group, timeout)


def _run_transaction(
    *,
    rank: int,
    args: argparse.Namespace,
    control_group: dist.ProcessGroup,
    source_buffer: deep_ep.ElasticBuffer | None,
    world_buffer: deep_ep.ElasticBuffer,
    result: VnodeRoundTripResult,
    source_inputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    layout: RecordLayout,
    source_arena_offset: int,
    world_arena_offset: int,
    generation: int,
    plan_fault: str | None = None,
) -> None:
    case = result.case
    g = case.topology.rails_per_node
    d = case.topology.num_nodes
    source_invocation = generation * 10 + 1
    world_invocation = generation * 10 + 2
    source_runtime = source_buffer.runtime if source_buffer is not None else None
    world_runtime = world_buffer.runtime
    source_plan: tuple[torch.Tensor, ...] | None = None
    proxy_baseline: torch.Tensor | None = None
    proxy_dispatch: torch.Tensor | None = None

    try:
        def source_prepare() -> int | None:
            if rank >= g:
                return None
            assert source_runtime is not None
            with _nvtx(args.nvtx, "c080_source_plan_prepare"):
                return int(source_runtime._rail_balance_hybrid_plan_prepare(  # type: ignore[attr-defined]
                    topk_idx,
                    args.hidden,
                    case.num_channels,
                    case.num_max_tokens_per_rank,
                    case.num_experts,
                    d,
                    0,
                    case.proxy_capacity_per_egress,
                    source_arena_offset,
                    source_invocation,
                    case.remainder_seed,
                ))

        source_status = _checked_phase(
            "source Gate 1 prepare", control_group, source_prepare)
        statuses = _gather_objects(source_status, control_group)
        assert tuple(statuses[:g]) == (0,) * g
        assert tuple(statuses[g:]) == (None,) * (_WORLD_SIZE - g)

        def source_finish() -> tuple[torch.Tensor, ...] | None:
            if rank >= g:
                return None
            assert source_runtime is not None
            with _nvtx(args.nvtx, "c080_source_plan_finish"):
                values = tuple(
                    source_runtime._rail_balance_hybrid_plan_finish(  # type: ignore[attr-defined]
                        source_invocation))
            assert len(values) == 14
            return values

        source_plan = _checked_phase(
            "source Gate 2 finish", control_group, source_finish)

        def verify_source_plan() -> str | None:
            if rank >= g:
                return None
            assert source_plan is not None
            return _verify_plan(source_plan, result, rank)

        source_digest = _checked_phase(
            "source exact compact plan", control_group, verify_source_plan)
        source_digests = _gather_objects(source_digest, control_group)
        assert len(set(source_digests[:g])) == 1
        assert source_digests[0] == _expected_plan_digest(result)
        assert tuple(source_digests[g:]) == (None,) * (_WORLD_SIZE - g)

        def source_baseline_snapshot() -> torch.Tensor | None:
            if rank >= g:
                return None
            assert source_runtime is not None
            value = source_runtime._rail_balance_hybrid_proxy_dispatch_snapshot(  # type: ignore[attr-defined]
                source_invocation)
            assert isinstance(value, torch.Tensor)
            torch.cuda.synchronize()
            return value

        proxy_baseline = _checked_phase(
            "pre-shuffle owning proxy snapshot",
            control_group,
            source_baseline_snapshot,
        )

        def source_shuffle() -> None:
            if rank >= g:
                return
            assert source_runtime is not None
            with _nvtx(args.nvtx, "c080_source_shuffle"):
                source_runtime._rail_balance_hybrid_source_shuffle(  # type: ignore[attr-defined]
                    x, topk_weights, source_invocation)

        _checked_phase(
            "source direct-final shuffle", control_group, source_shuffle)

        def source_visibility_barrier() -> None:
            if rank >= g:
                return
            assert source_runtime is not None
            with _nvtx(args.nvtx, "c080_source_visibility"):
                source_runtime.barrier(True, True, True)  # type: ignore[attr-defined]

        _checked_phase(
            "source LSA visibility", control_group,
            source_visibility_barrier)

        def source_snapshot() -> torch.Tensor | None:
            if rank >= g:
                return None
            assert source_runtime is not None
            with _nvtx(args.nvtx, "c080_source_snapshot"):
                value = source_runtime._rail_balance_hybrid_proxy_dispatch_snapshot(  # type: ignore[attr-defined]
                    source_invocation)
                assert isinstance(value, torch.Tensor)
                torch.cuda.synchronize()
                return value

        proxy_dispatch = _checked_phase(
            "owning source proxy snapshot", control_group, source_snapshot)

        def verify_proxy() -> None:
            if rank >= g:
                return
            assert proxy_baseline is not None and proxy_dispatch is not None
            _verify_proxy_dispatch(
                baseline=proxy_baseline,
                snapshot=proxy_dispatch,
                rank=rank,
                result=result,
                source_inputs=source_inputs,
                layout=layout,
            )

        _checked_phase(
            "exact production proxy TokenLayout",
            control_group,
            verify_proxy,
        )

        def source_count_snapshot() -> torch.Tensor | None:
            if rank != 0:
                return None
            assert source_plan is not None
            return source_plan[0].cpu().contiguous()

        rank_zero_count = _checked_phase(
            "rank-zero owning channel-count snapshot",
            control_group,
            source_count_snapshot,
        )

        def broadcast_count() -> torch.Tensor:
            channel_count_object = [rank_zero_count]
            dist.broadcast_object_list(
                channel_count_object, src=0, group=control_group)
            value = channel_count_object[0]
            assert isinstance(value, torch.Tensor)
            return value

        channel_count_cpu = _checked_phase(
            "Gloo channel-count broadcast",
            control_group,
            broadcast_count,
        )

        def materialize_local_count() -> torch.Tensor:
            assert channel_count_cpu.device.type == "cpu"
            assert channel_count_cpu.dtype == torch.int32
            expected_channel_count = _expected_plan_outputs(
                result.schedule)[0]
            assert torch.equal(channel_count_cpu, expected_channel_count)
            return channel_count_cpu.to(
                torch.device("cuda", rank)).contiguous()

        channel_count = _checked_phase(
            "owning per-rank channel-count materialization",
            control_group,
            materialize_local_count,
        )

        def materialize_world_proxy() -> torch.Tensor:
            if rank < g:
                assert proxy_dispatch is not None
                return proxy_dispatch
            return torch.empty(
                (0, layout.dispatch_token_bytes),
                dtype=torch.uint8,
                device=torch.device("cuda", rank),
            )

        proxy_dispatch = _checked_phase(
            "owning world proxy materialization",
            control_group,
            materialize_world_proxy,
        )

        def world_prepare() -> tuple[object, ...]:
            with _nvtx(args.nvtx, "c080_world_prepare"):
                values = tuple(
                    world_runtime._rail_balance_hybrid_vnode_prepare(  # type: ignore[attr-defined]
                        x,
                        topk_idx,
                        topk_weights,
                        proxy_dispatch,
                        channel_count,
                        world_arena_offset,
                        case.num_max_tokens_per_rank,
                        case.num_experts,
                        d,
                        g,
                        case.proxy_capacity_per_egress,
                        generation,
                        world_invocation,
                        case.remainder_seed,
                    ))
            assert len(values) == 3
            assert int(values[0]) == 0
            assert isinstance(values[1], (tuple, list))
            assert len(values[1]) == 14
            assert all(isinstance(value, torch.Tensor)
                       for value in values[1])
            assert isinstance(values[2], torch.Tensor)
            return values

        prepared = _checked_phase(
            "world pre-commit prepare", control_group, world_prepare)
        _prepare_status, world_plan_value, compact_quota = prepared
        world_plan = tuple(world_plan_value)

        if plan_fault == "corrupt" and rank == 0:
            assert world_plan[0].numel() > 0
            world_plan[0].view(-1)[0].add_(1)
        elif plan_fault == "missing" and rank == 1:
            world_plan = world_plan[:-1]
        elif plan_fault not in (None, "corrupt", "missing"):
            raise AssertionError(f"unknown world-plan fault: {plan_fault}")

        def verify_world_preflight() -> str:
            plan_digest = _verify_plan(world_plan, result, rank)
            wanted = torch.tensor(
                [row[1:] for row in result.schedule.quota],
                dtype=torch.int32,
                device="cpu",
            )
            assert compact_quota.is_cuda
            assert compact_quota.device.index == rank
            assert compact_quota.dtype == torch.int32
            assert compact_quota.is_contiguous()
            assert torch.equal(compact_quota.cpu(), wanted)
            return plan_digest

        if plan_fault is not None:
            preflight_error = None
            try:
                verify_world_preflight()
            except BaseException:
                preflight_error = traceback.format_exc()
            preflight_errors = _gather_objects(
                preflight_error, control_group)
            expected_rank = 0 if plan_fault == "corrupt" else 1
            assert all(
                (error is not None) == (index == expected_rank)
                for index, error in enumerate(preflight_errors)
            ), preflight_errors
            expected_error = (
                "world plan tensor 0 value mismatch"
                if plan_fault == "corrupt"
                else "world plan tensor count mismatch"
            )
            assert expected_error in preflight_errors[expected_rank]
            _monitored_barrier(control_group, args.timeout)
            return

        world_preflight = _checked_phase(
            "CPU/source/world pre-B0 identity",
            control_group,
            verify_world_preflight,
        )
        gathered_preflight = _gather_objects(
            (source_digest, world_preflight), control_group)
        expected_digest = _expected_plan_digest(result)
        assert all(item[1] == expected_digest
                   for item in gathered_preflight)
        assert all(item[0] == expected_digest
                   for item in gathered_preflight[:g])
        assert all(item[0] is None for item in gathered_preflight[g:])

        def world_finish() -> tuple[object, ...]:
            with _nvtx(args.nvtx, "c080_world_b0_b5"):
                values = tuple(
                    world_runtime._rail_balance_hybrid_vnode_finish(  # type: ignore[attr-defined]
                        world_invocation))
            return values

        world_outputs = _checked_phase(
            "fixed world B0..B5", control_group, world_finish)
        _checked_phase(
            "byte-exact world records/routes/demux",
            control_group,
            lambda: _verify_world_outputs(
                rank=rank,
                outputs=world_outputs,
                result=result,
                source_inputs=source_inputs,
                layout=layout,
                generation=generation,
            ),
        )

        def source_unshuffle() -> torch.Tensor | None:
            if rank >= g:
                return None
            assert source_runtime is not None
            proxy_return = world_outputs[0]
            reduce_seed = world_outputs[1]
            assert isinstance(proxy_return, torch.Tensor)
            assert isinstance(reduce_seed, torch.Tensor)
            with _nvtx(args.nvtx, "c080_source_return_unshuffle"):
                value = source_runtime._rail_balance_hybrid_return_unshuffle_test(  # type: ignore[attr-defined]
                    proxy_return, reduce_seed, source_invocation)
            assert isinstance(value, torch.Tensor)
            return value

        reduce_snapshot = _checked_phase(
            "production return unshuffle", control_group, source_unshuffle)

        def verify_reduce() -> None:
            if rank >= g:
                assert reduce_snapshot is None
                return
            assert isinstance(reduce_snapshot, torch.Tensor)
            assert reduce_snapshot.is_cuda
            assert reduce_snapshot.device.index == rank
            assert reduce_snapshot.dtype == torch.uint8
            assert reduce_snapshot.is_contiguous()
            assert torch.equal(
                reduce_snapshot.cpu(),
                _expected_reduce(
                    owner=rank,
                    result=result,
                    source_inputs=source_inputs,
                    layout=layout,
                ),
            )

        _checked_phase(
            "exact complete legacy reduce buffer",
            control_group,
            verify_reduce,
        )

        def source_epilogue() -> tuple[torch.Tensor, torch.Tensor] | None:
            if rank >= g:
                return None
            assert source_runtime is not None
            with _nvtx(args.nvtx, "c080_legacy_combine_epilogue"):
                values = tuple(
                    source_runtime._rail_balance_hybrid_combine_epilogue_test(  # type: ignore[attr-defined]
                        source_invocation))
            assert len(values) == 2
            assert all(isinstance(value, torch.Tensor) for value in values)
            return values  # type: ignore[return-value]

        combined = _checked_phase(
            "unchanged legacy combine epilogue",
            control_group,
            source_epilogue,
        )

        def verify_final() -> None:
            if rank >= g:
                assert combined is None
                return
            assert combined is not None
            combined_x, combined_weights = combined
            assert torch.equal(
                combined_x.cpu(),
                _expected_combined(rank, result, args.hidden))
            assert torch.equal(
                combined_weights.cpu(), source_inputs[rank][2])
            assert torch.equal(x.cpu(), source_inputs[rank][0])
            assert torch.equal(topk_idx.cpu(), source_inputs[rank][1])
            assert torch.equal(topk_weights.cpu(), source_inputs[rank][2])

        _checked_phase(
            "exact hierarchical final output and immutable inputs",
            control_group,
            verify_final,
        )
    finally:
        _abort_transactions(
            rank=rank,
            num_source_ranks=g,
            source_runtime=source_runtime,
            world_runtime=world_runtime,
            source_invocation=source_invocation,
            world_invocation=world_invocation,
            control_group=control_group,
            timeout=args.timeout,
        )


@torch.inference_mode()
def _worker(
    local_rank: int,
    num_local_ranks: int,
    args: argparse.Namespace,
) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, world_group = init_dist(
        local_rank, num_local_ranks, seed=args.generation)
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(seconds=args.timeout),
    )
    assert rank == local_rank and world_size == _WORLD_SIZE
    result = _oracle(GpuCase(
        args.case_name,
        args.fixture,
        args.hidden,
        args.generation,
        args.non_default_stream,
        args.repetitions,
    ))
    case = result.case
    g = case.topology.rails_per_node
    source_group = dist.new_group(
        ranks=list(range(g)),
        backend="nccl",
        timeout=timedelta(seconds=args.timeout),
        use_local_synchronization=False,
        device_id=torch.device(f"cuda:{local_rank}"),
        group_desc=f"c080-source-g{g}",
    )
    source_buffer = None
    world_buffer = None
    clean_shutdown = False
    try:
        source_inputs = _cpu_source_inputs(case, args.hidden)
        caller_stream = (
            torch.cuda.Stream(device=local_rank)
            if args.non_default_stream else None
        )
        with _caller_stream(caller_stream):
            x, topk_idx, topk_weights = _make_cuda_inputs(
                rank, result, source_inputs, args.hidden)
        layout, hybrid_layout = _layout_abis(case, args.hidden)
        alignment = int(_C.get_elastic_buffer_alignment())
        assert alignment == _ARENA_ALIGNMENT

        if rank < g:
            source_base = deep_ep.ElasticBuffer.get_buffer_size_hint(
                source_group,
                num_max_tokens_per_rank=case.num_max_tokens_per_rank,
                hidden=args.hidden,
                num_topk=case.num_topk,
                use_fp8_dispatch=False,
                allow_hybrid_mode=False,
                allow_multiple_reduction=True,
            )
            source_arena_offset = align(source_base, alignment)
            source_buffer = deep_ep.ElasticBuffer(
                source_group,
                num_bytes=align(
                    source_arena_offset + hybrid_layout[-1], alignment),
                num_max_tokens_per_rank=case.num_max_tokens_per_rank,
                hidden=args.hidden,
                num_topk=case.num_topk,
                allow_hybrid_mode=False,
                allow_multiple_reduction=True,
                prefer_overlap_with_compute=False,
                explicitly_destroy=True,
                num_gpu_timeout_secs=args.timeout,
                num_cpu_timeout_secs=args.timeout,
            )
            assert source_buffer.get_logical_domain_size() == (1, g)
            assert source_buffer.scaleup_rank_idx == rank
        else:
            source_arena_offset = -1
        source_offsets = _gather_objects(source_arena_offset, control_group)
        assert len(set(source_offsets[:g])) == 1
        assert tuple(source_offsets[g:]) == (-1,) * (_WORLD_SIZE - g)
        source_arena_offset = int(source_offsets[0])
        _monitored_barrier(control_group, args.timeout)

        world_base = deep_ep.ElasticBuffer.get_buffer_size_hint(
            world_group,
            num_max_tokens_per_rank=case.num_max_tokens_per_rank,
            hidden=args.hidden,
            num_topk=case.num_topk,
            use_fp8_dispatch=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=False,
        )
        world_arena_offset = align(world_base, alignment)
        world_buffer = deep_ep.ElasticBuffer(
            world_group,
            num_bytes=align(
                world_arena_offset + layout.vnode_arena_bytes
                + _ARENA_GUARD_BYTES,
                alignment,
            ),
            num_max_tokens_per_rank=case.num_max_tokens_per_rank,
            hidden=args.hidden,
            num_topk=case.num_topk,
            allow_hybrid_mode=False,
            allow_multiple_reduction=False,
            prefer_overlap_with_compute=False,
            explicitly_destroy=True,
            num_gpu_timeout_secs=args.timeout,
            num_cpu_timeout_secs=args.timeout,
        )
        assert world_buffer.get_logical_domain_size() == (1, _WORLD_SIZE)
        assert world_buffer.scaleup_rank_idx == rank
        _monitored_barrier(control_group, args.timeout)

        # Generic barrier() may cold-build its cubin.  Warm it before any
        # source publication so the later system-scope visibility barrier has
        # no fallible build step between producer completion and peer entry.
        def warm_source_visibility_barrier() -> None:
            if rank < g:
                assert source_buffer is not None
                source_buffer.runtime.barrier(  # type: ignore[attr-defined]
                    True, True, True)

        _checked_phase(
            "warm source LSA visibility barrier",
            control_group,
            warm_source_visibility_barrier,
        )

        signature = (
            args.case_name,
            args.hidden,
            args.generation,
            args.repetitions,
            args.non_default_stream,
            g,
            case.topology.num_nodes,
            case.num_topk,
            case.num_channels,
            case.num_max_tokens_per_rank,
            case.num_experts,
            case.proxy_capacity_per_egress,
            source_arena_offset,
            world_arena_offset,
            layout.vnode_arena_bytes,
        )
        signatures = _gather_objects(signature, control_group)
        _assert_equal(signatures)
        token_counts = _gather_objects(int(x.shape[0]), control_group)
        assert tuple(token_counts[:g]) == case.num_tokens_per_owner
        assert tuple(token_counts[g:]) == (0,) * (_WORLD_SIZE - g)

        generation = args.generation
        if args.case_name == _PLAN_FAULT_CASE:
            for plan_fault in ("corrupt", "missing"):
                with _caller_stream(caller_stream):
                    _run_transaction(
                        rank=rank,
                        args=args,
                        control_group=control_group,
                        source_buffer=source_buffer,
                        world_buffer=world_buffer,
                        result=result,
                        source_inputs=source_inputs,
                        x=x,
                        topk_idx=topk_idx,
                        topk_weights=topk_weights,
                        layout=layout,
                        source_arena_offset=source_arena_offset,
                        world_arena_offset=world_arena_offset,
                        generation=generation,
                        plan_fault=plan_fault,
                    )
                generation += 1

        for iteration in range(args.repetitions):
            with _caller_stream(caller_stream):
                if caller_stream is not None:
                    assert torch.cuda.current_stream().cuda_stream == \
                        caller_stream.cuda_stream
                _run_transaction(
                    rank=rank,
                    args=args,
                    control_group=control_group,
                    source_buffer=source_buffer,
                    world_buffer=world_buffer,
                    result=result,
                    source_inputs=source_inputs,
                    x=x,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    layout=layout,
                    source_arena_offset=source_arena_offset,
                    world_arena_offset=world_arena_offset,
                    generation=generation + iteration,
                )
        _monitored_barrier(control_group, args.timeout)
        clean_shutdown = True
    finally:
        # Both destroy calls are collective in their respective communicators.
        # Do not attempt a partial teardown after an uncoordinated failure; the
        # outer process-group watchdog owns that recovery path.
        if clean_shutdown and world_buffer is not None:
            _monitored_barrier(control_group, args.timeout)
            if rank < g:
                assert source_buffer is not None
                source_buffer.destroy()
                source_buffer = None
            _monitored_barrier(control_group, args.timeout)
            world_buffer.destroy()
            world_buffer = None
            _monitored_barrier(control_group, args.timeout)
            if rank == 0:
                print(
                    "PASS C080-D Hybrid vnode: "
                    f"{args.case_name}, G={g}, "
                    f"D={case.topology.num_nodes}, K={case.num_topk}, "
                    f"H={args.hidden}, N={case.num_tokens_per_owner}, "
                    f"moved={result.schedule.moved_copies}, "
                    f"repetitions={args.repetitions}",
                    flush=True,
                )
            dist.destroy_process_group()


def _run_watchdog(
    arguments: argparse.Namespace,
    spec: GpuCase,
    port: int,
) -> None:
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker-suite",
        "--case-name", spec.name,
        "--fixture", spec.fixture,
        "--hidden", str(spec.hidden),
        "--generation", str(spec.generation),
        "--repetitions", str(spec.repetitions),
        "--num-processes", str(arguments.num_processes),
        "--timeout", str(arguments.timeout),
        "--master-port", str(port),
        "--watchdog-seconds", str(arguments.watchdog_seconds),
    ]
    if arguments.nvtx:
        command.append("--nvtx")
    if spec.non_default_stream:
        command.append("--non-default-stream")
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
            "C080-D Hybrid vnode subprocess watchdog expired after "
            f"{arguments.watchdog_seconds}s for {spec.name}") from error
    if return_code:
        raise SystemExit(return_code)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C080-D strict 8-GPU Hybrid vnode closed-loop test")
    if not __debug__:
        parser.error("C080-D correctness checks require Python assertions")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--worker-suite", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--master-port", type=int, default=29937)
    parser.add_argument("--watchdog-seconds", type=int, default=1800)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--non-default-stream", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--case-name")
    parser.add_argument("--fixture", choices=(
                            "4x2", "4x2_empty_egress", "2x4", "rounding",
                            "all_zero"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--hidden", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--generation", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repetitions", type=int, default=1,
                        help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.num_processes != _WORLD_SIZE:
        parser.error("C080-D Hybrid vnode requires exactly 8 processes")
    if arguments.timeout <= 0 or arguments.watchdog_seconds <= 0 or \
            arguments.repetitions <= 0:
        parser.error("timeouts must be positive")
    return arguments


def main() -> None:
    arguments = _parse_args()
    all_specs = _specs()
    by_name = {spec.name: spec for spec in all_specs}
    if arguments.case_name is not None:
        if arguments.case_name not in by_name:
            raise SystemExit(
                "unknown --case-name; expected one of "
                + ", ".join(sorted(by_name)))
        selected = (by_name[arguments.case_name],)
    else:
        selected = all_specs

    if arguments.worker_suite:
        assert len(selected) == 1
        spec = selected[0]
        assert arguments.fixture == spec.fixture
        assert arguments.hidden == spec.hidden
        assert arguments.generation == spec.generation
        assert arguments.non_default_stream == spec.non_default_stream
        assert arguments.repetitions == spec.repetitions
    _assert_cpu_oracle(selected)
    print(
        "PASS C080-D Hybrid vnode CPU oracle: "
        + ", ".join(spec.name for spec in selected),
        flush=True,
    )
    if arguments.oracle_only:
        return

    required = (
        "_rail_balance_hybrid_plan_prepare",
        "_rail_balance_hybrid_plan_finish",
        "_rail_balance_hybrid_source_shuffle",
        "_rail_balance_hybrid_proxy_dispatch_snapshot",
        "_rail_balance_hybrid_return_unshuffle_test",
        "_rail_balance_hybrid_combine_epilogue_test",
        "_rail_balance_hybrid_plan_abort",
        "_rail_balance_hybrid_vnode_prepare",
        "_rail_balance_hybrid_vnode_finish",
        "_rail_balance_hybrid_vnode_abort",
    )
    runtime_type = getattr(_C, "ElasticBuffer", None)
    missing = [
        name for name in required
        if runtime_type is None or not hasattr(runtime_type, name)
    ]
    if missing:
        raise SystemExit(
            "rebuild the extension with the C080-D private API; missing "
            + ", ".join(missing))
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise SystemExit("C080-D Hybrid vnode requires 8 visible CUDA devices")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ.setdefault("EP_DISABLE_GIN", "1")
    if not arguments.worker_suite:
        for index, spec in enumerate(selected):
            _run_watchdog(arguments, spec, arguments.master_port + index)
        return

    os.environ["MASTER_PORT"] = str(arguments.master_port)
    torch.multiprocessing.spawn(
        _worker,
        args=(arguments.num_processes, arguments),
        nprocs=arguments.num_processes,
        join=True,
    )


if __name__ == "__main__":
    main()
