"""C060: deterministic 4+4 virtual-node rail-balance round trip.

This is a functional, pure-single-node test.  GPUs 0..3 are virtual source
owner/egress ranks and GPUs 4..7 are virtual destination ingress/expert ranks.
The private C060 launcher uses LSA copies in place of scaleout transport; this
test therefore makes no claim about GIN, QPs, NICs, or multi-node performance.

Run from the repository root after rebuilding the extension::

    PYTHONPATH=$PWD EP_DISABLE_GIN=1 \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      python -B tests/elastic/test_rail_balance_vnode.py --hidden 256

Use ``--hidden 7168`` for the production-sized functional record.
"""

from __future__ import annotations

import argparse
import os
import struct
import traceback
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Sequence, TypeVar

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import align
from rail_balance_reference import (
    FullEgressPlan,
    build_count_plan,
    build_destination_copies,
    counts_from_copies,
    materialize_full_egress_slots,
    materialize_static_slots,
)


_T = TypeVar("_T")

_WORLD_SIZE = 8
_NUM_SOURCE_RANKS = 4
_NUM_TOKENS = 32
_ACTIVE_TOKENS = (32, 24, 16, 8)
_NUM_TOPK = 4
_PHYSICAL_CAPACITY = 20
_GENERATION = 61
_EXPERT_BEGIN = 16
_EXPERTS_PER_RANK = 4
_TMA_ALIGNMENT = 32
_ARENA_ALIGNMENT = 2_097_152
_ARENA_GUARD_BYTES = 4096
_HEAD_CANARY = 0xDEADBEEF
_TAIL_CANARY = 0xC001D00D
_WEIGHTS = (0.5, 0.25, 0.125, 0.125)


def _expert_local_rank(owner: int, token: int, lane: int) -> int:
    """Rotate lane ownership so a lane is not accidentally treated as a GPU."""

    assert 0 <= owner < _NUM_SOURCE_RANKS
    assert 0 <= token < _NUM_TOKENS
    assert 0 <= lane < _NUM_TOPK
    return (owner + token + lane) % _NUM_SOURCE_RANKS


@dataclass(frozen=True, order=True)
class ExpectedRecord:
    owner: int
    token: int
    destination: int
    owner_ordinal: int
    egress: int
    channel: int
    logical_slot: int
    physical_slot: int
    moved: bool
    fingerprint: int

    @property
    def manifest(self) -> tuple[int, ...]:
        return (
            self.token,
            self.destination,
            self.owner_ordinal,
            self.egress,
            self.channel,
            self.logical_slot,
            self.physical_slot,
        )

    @property
    def identity(self) -> tuple[int, ...]:
        return (
            self.owner,
            self.token,
            self.destination,
            self.owner_ordinal,
            self.egress,
            self.channel,
            self.logical_slot,
            self.physical_slot,
            int(self.moved),
            self.fingerprint,
        )


@dataclass(frozen=True)
class Layout:
    record_bytes: int
    route_bytes: int
    rail_capacity: int
    expert_capacity: int
    arena_bytes: int
    head_bytes: int
    token_bytes: int
    descriptor_offset: int
    tail_offset: int


def _checked_local_phase(
    name: str,
    control_group: dist.ProcessGroup,
    function: Callable[[], _T],
) -> _T:
    """Turn a local assertion into one synchronized, rank-attributed error."""

    value = None
    try:
        value = function()
        failure = None
    except BaseException:
        failure = traceback.format_exc()

    failures: list[str | None] = [None] * dist.get_world_size(control_group)
    dist.all_gather_object(failures, failure, group=control_group)
    messages = [
        f"rank {rank}:\n{message}"
        for rank, message in enumerate(failures)
        if message is not None
    ]
    if messages:
        raise AssertionError(
            f"{name} failed on one or more ranks:\n" + "\n".join(messages))
    return value  # type: ignore[return-value]


def _fingerprint(owner: int, token: int, owner_ordinal: int) -> int:
    return (
        (_GENERATION << 48)
        | (owner << 40)
        | (token << 24)
        | owner_ordinal
    )


def _build_oracle() -> tuple[FullEgressPlan, tuple[ExpectedRecord, ...]]:
    destinations = [
        [[0] if token < _ACTIVE_TOKENS[owner] else [-1]
         for token in range(_NUM_TOKENS)]
        for owner in range(_NUM_SOURCE_RANKS)
    ]
    copies = build_destination_copies(
        destinations,
        num_destinations=1,
        num_channels=1,
        local_destination=None,
    )
    counts = counts_from_copies(
        copies,
        num_rails=_NUM_SOURCE_RANKS,
        num_destinations=1,
        num_channels=1,
    )
    assert counts == tuple((count,) for count in _ACTIVE_TOKENS)
    candidate = build_count_plan(counts, remainder_seed=_GENERATION)
    assert candidate.quota == ((20,), (20,), (20,), (20,))
    assert tuple(
        (segment.owner, segment.egress, segment.owner_begin, segment.count)
        for segment in candidate.segments
    ) == ((0, 2, 20, 4), (0, 3, 24, 8), (1, 3, 20, 4))
    logical = materialize_static_slots(
        copies,
        candidate,
        num_channels=1,
        channel_capacity=_PHYSICAL_CAPACITY,
        generation=_GENERATION,
    )
    assert logical.enabled and logical.moved_copies == 16
    full = materialize_full_egress_slots(
        logical,
        physical_capacity_per_egress=_PHYSICAL_CAPACITY,
    )
    assert full.enabled and full.bypass_reason is None
    assert full.records_per_egress == (_PHYSICAL_CAPACITY,) * 4
    assert len(full.assignments) == sum(_ACTIVE_TOKENS) == 80

    records = tuple(
        ExpectedRecord(
            owner=item.assignment.copy.owner,
            token=item.assignment.copy.token,
            destination=item.assignment.destination,
            owner_ordinal=item.assignment.copy.owner_ordinal,
            egress=item.egress,
            channel=item.assignment.channel,
            logical_slot=item.assignment.slot,
            physical_slot=item.physical_slot,
            moved=item.assignment.moved,
            fingerprint=_fingerprint(
                item.assignment.copy.owner,
                item.assignment.copy.token,
                item.assignment.copy.owner_ordinal,
            ),
        )
        for item in full.assignments
    )
    assert sum(record.moved for record in records) == 16
    assert len({record.identity for record in records}) == 80
    for lane in range(_NUM_TOPK):
        assert {
            _expert_local_rank(record.owner, record.token, lane)
            for record in records
        } == set(range(_NUM_SOURCE_RANKS))
    for egress in range(_NUM_SOURCE_RANKS):
        assert sorted(
            record.physical_slot for record in records
            if record.egress == egress
        ) == list(range(_PHYSICAL_CAPACITY))
    return full, records


def _source_inputs(
    owner: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU source fixture with exact BF16/dyadic arithmetic."""

    assert 0 <= owner < _NUM_SOURCE_RANKS
    cpu = torch.device("cpu")
    columns = torch.arange(hidden, dtype=torch.int64, device=cpu)
    x = torch.empty((_NUM_TOKENS, hidden), dtype=torch.bfloat16, device=cpu)
    topk_idx = torch.full(
        (_NUM_TOKENS, _NUM_TOPK), -1,
        dtype=deep_ep.topk_idx_t, device=cpu)
    topk_weights = torch.zeros(
        (_NUM_TOKENS, _NUM_TOPK), dtype=torch.float32, device=cpu)
    for token in range(_NUM_TOKENS):
        x[token] = (
            8 * ((5 * owner + 3 * token + (columns & 7)) & 7)
        ).to(torch.bfloat16)
        if token >= _ACTIVE_TOKENS[owner]:
            continue
        for lane, weight in enumerate(_WEIGHTS):
            expert_local_rank = _expert_local_rank(owner, token, lane)
            local_expert = (
                _EXPERTS_PER_RANK * expert_local_rank
                + ((3 * owner + token + lane) % _EXPERTS_PER_RANK)
            )
            topk_idx[token, lane] = _EXPERT_BEGIN + local_expert
            topk_weights[token, lane] = weight
    return x, topk_idx, topk_weights


def _expected_source_results(
    owner: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, topk_idx, _topk_weights = _source_inputs(owner, hidden)
    partials = torch.zeros(
        (_NUM_TOKENS, _NUM_TOPK, hidden),
        dtype=torch.bfloat16,
        device="cpu",
    )
    output = torch.zeros(
        (_NUM_TOKENS, hidden), dtype=torch.bfloat16, device="cpu")
    for token in range(_ACTIVE_TOKENS[owner]):
        reduced = torch.zeros(hidden, dtype=torch.float32, device="cpu")
        for lane, weight in enumerate(_WEIGHTS):
            local_expert = int(topk_idx[token, lane]) - _EXPERT_BEGIN
            partial = (
                (x[token].float() + 8 * (local_expert + 1)) * weight
            ).to(torch.bfloat16)
            partials[token, lane] = partial
            # Match the kernel's deterministic lane-order FP32 accumulation.
            reduced += partial.float()
        output[token] = reduced.to(torch.bfloat16)
    return partials, output


def _make_local_inputs(
    rank: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda", rank)
    if rank < _NUM_SOURCE_RANKS:
        x, topk_idx, topk_weights = _source_inputs(rank, hidden)
    else:
        x = torch.zeros(
            (_NUM_TOKENS, hidden), dtype=torch.bfloat16, device="cpu")
        topk_idx = torch.full(
            (_NUM_TOKENS, _NUM_TOPK), -1,
            dtype=deep_ep.topk_idx_t, device="cpu")
        topk_weights = torch.zeros(
            (_NUM_TOKENS, _NUM_TOPK), dtype=torch.float32, device="cpu")
    return x.to(device), topk_idx.to(device), topk_weights.to(device)


def _layout(hidden: int) -> Layout:
    vnode = tuple(int(value) for value in _C._get_rail_balance_vnode_layout(
        hidden,
        _NUM_TOPK,
        _PHYSICAL_CAPACITY,
        _NUM_SOURCE_RANKS,
        _NUM_TOKENS,
    ))
    assert len(vnode) == 13
    (record_bytes, route_bytes, rail_capacity, rail_ready_offset,
     rail_route_offset, expert_offset, expert_capacity,
     expert_ready_offset, expert_route_offset, owner_values_offset,
     owner_ready_offset, role_arena_bytes, arena_bytes) = vnode

    # Independently derive every layout field. Do not use one C++ getter to
    # validate another: an under-sized arena or overlapping role region could
    # otherwise become a common-mode host/device bug.
    hidden_bytes = hidden * torch.bfloat16.itemsize
    metadata_bytes = _NUM_TOPK * 2 * 4 + (1 + _NUM_TOPK) * 4
    expected_token_bytes = (
        align(hidden_bytes, _TMA_ALIGNMENT)
        + align(metadata_bytes, _TMA_ALIGNMENT))
    expected_record_bytes = align(
        32 + expected_token_bytes + 64 + 32, _TMA_ALIGNMENT)

    def expected_stage(capacity: int) -> tuple[int, int, int]:
        ready = expected_record_bytes * capacity
        route = align(
            ready + align(capacity * 4, _TMA_ALIGNMENT),
            _TMA_ALIGNMENT,
        )
        arena = align(route + capacity * 32, _ARENA_ALIGNMENT)
        return ready, route, arena

    expected_rail_capacity = _PHYSICAL_CAPACITY * (_NUM_TOPK + 1)
    expected_expert_capacity = (
        _NUM_SOURCE_RANKS * _PHYSICAL_CAPACITY * _NUM_TOPK)
    expected_rail_ready, expected_rail_route, expected_rail_arena = \
        expected_stage(expected_rail_capacity)
    expected_expert_ready_rel, expected_expert_route_rel, \
        expected_expert_arena = expected_stage(expected_expert_capacity)
    partial_capacity = _NUM_TOKENS * _NUM_TOPK
    expected_owner_ready_rel = align(
        hidden_bytes * partial_capacity, _TMA_ALIGNMENT)
    expected_owner_arena = align(
        expected_owner_ready_rel
        + align(partial_capacity * 4, _TMA_ALIGNMENT),
        _ARENA_ALIGNMENT,
    )
    expected_role_arena = max(expected_expert_arena, expected_owner_arena)
    expected_vnode = (
        expected_record_bytes,
        32,
        expected_rail_capacity,
        expected_rail_ready,
        expected_rail_route,
        expected_rail_arena,
        expected_expert_capacity,
        expected_rail_arena + expected_expert_ready_rel,
        expected_rail_arena + expected_expert_route_rel,
        expected_rail_arena,
        expected_rail_arena + expected_owner_ready_rel,
        expected_role_arena,
        expected_rail_arena + expected_role_arena,
    )
    assert vnode == expected_vnode
    assert route_bytes == 32
    assert rail_capacity == _PHYSICAL_CAPACITY * (_NUM_TOPK + 1) == 100
    assert expert_capacity == (
        _NUM_SOURCE_RANKS * _PHYSICAL_CAPACITY * _NUM_TOPK) == 320
    assert owner_values_offset == expert_offset
    assert 0 < rail_ready_offset < rail_route_offset < expert_offset
    assert expert_offset < expert_ready_offset < expert_route_offset
    assert 0 < owner_ready_offset < expert_offset + role_arena_bytes
    alignment = int(_C.get_elastic_buffer_alignment())
    assert expert_offset % alignment == arena_bytes % alignment == 0

    source = tuple(int(value) for value in
                   _C._get_rail_balance_source_shuffle_layout(
                       hidden, _NUM_TOPK, rail_capacity))
    (head_bytes, token_bytes, descriptor_offset, tail_offset,
     source_record_bytes, source_ready_offset, _source_arena_bytes) = source
    assert head_bytes == 32
    assert token_bytes == (
        align(hidden_bytes, _TMA_ALIGNMENT)
        + align(metadata_bytes, _TMA_ALIGNMENT))
    assert descriptor_offset == head_bytes + token_bytes
    assert tail_offset == descriptor_offset + 64
    assert record_bytes == source_record_bytes
    assert rail_ready_offset == source_ready_offset
    return Layout(
        record_bytes=record_bytes,
        route_bytes=route_bytes,
        rail_capacity=rail_capacity,
        expert_capacity=expert_capacity,
        arena_bytes=arena_bytes,
        head_bytes=head_bytes,
        token_bytes=token_bytes,
        descriptor_offset=descriptor_offset,
        tail_offset=tail_offset,
    )


def _manifest_for_rank(
    records: Sequence[ExpectedRecord],
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = [record for record in records if record.owner == rank]
    if local:
        manifest = torch.tensor(
            [record.manifest for record in local],
            dtype=torch.int32,
            device="cpu",
        )
        fingerprints = torch.tensor(
            [record.fingerprint for record in local],
            dtype=torch.int64,
            device="cpu",
        )
    else:
        manifest = torch.empty((0, 7), dtype=torch.int32, device="cpu")
        fingerprints = torch.empty((0,), dtype=torch.int64, device="cpu")
    device = torch.device("cuda", rank)
    return manifest.to(device), fingerprints.to(device)


def _verify_preflight(
    gathered: Sequence[tuple[list[list[int]], list[int]]],
    records: Sequence[ExpectedRecord],
) -> None:
    observed = []
    physical_keys = []
    for owner, (manifests, fingerprints) in enumerate(gathered):
        if owner >= _NUM_SOURCE_RANKS:
            assert not manifests and not fingerprints
            continue
        assert len(manifests) == len(fingerprints) == _ACTIVE_TOKENS[owner]
        for manifest, fingerprint in zip(manifests, fingerprints):
            assert len(manifest) == 7
            (token, destination, owner_ordinal, egress, channel,
             logical_slot, physical_slot) = manifest
            physical_keys.append((egress, physical_slot))
            observed.append((
                owner, token, destination, owner_ordinal, egress, channel,
                logical_slot, physical_slot, fingerprint,
            ))
    expected = [
        record.identity[:8] + (record.fingerprint,)
        for record in records
    ]
    # Physical slot is the stable protocol key; producer arrival order is not.
    assert len(physical_keys) == len(set(physical_keys)) == 80
    assert sorted(observed) == sorted(expected)


def _route_bytes(route: torch.Tensor) -> bytes:
    assert route.dtype == torch.uint8 and route.numel() == 32
    return route.contiguous().numpy().tobytes()


def _expected_route(record: ExpectedRecord, lane: int) -> tuple[int, ...]:
    expert_rank = (
        _NUM_SOURCE_RANKS
        + _expert_local_rank(record.owner, record.token, lane)
    )
    expert_slot = (
        (record.egress * _PHYSICAL_CAPACITY + record.physical_slot)
        * _NUM_TOPK + lane
    )
    return (
        record.fingerprint,
        _GENERATION,
        _NUM_SOURCE_RANKS + record.egress,
        record.physical_slot,
        lane,
        expert_rank,
        expert_slot,
    )


def _verify_route(
    route: torch.Tensor,
    record: ExpectedRecord,
    lane: int,
) -> None:
    assert struct.unpack("<Q6i", _route_bytes(route)) == \
        _expected_route(record, lane)


def _verify_record(
    record_bytes: torch.Tensor,
    expected: ExpectedRecord,
    expected_hidden: torch.Tensor,
    source_inputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    layout: Layout,
) -> None:
    assert record_bytes.dtype == torch.uint8
    assert tuple(record_bytes.shape) == (layout.record_bytes,)
    raw = record_bytes.contiguous().numpy().tobytes()
    assert struct.unpack_from("<8I", raw, 0) == (_HEAD_CANARY,) * 8
    assert struct.unpack_from(
        "<8I", raw, layout.tail_offset) == (_TAIL_CANARY,) * 8

    source_x, source_topk_idx, source_topk_weights = source_inputs[expected.owner]
    hidden_bytes = expected_hidden.numel() * torch.bfloat16.itemsize
    token_offset = layout.head_bytes
    expected_hidden_bytes = expected_hidden.contiguous().view(
        torch.uint8).numpy().tobytes()
    assert raw[token_offset:token_offset + hidden_bytes] == expected_hidden_bytes

    metadata_offset = token_offset + align(hidden_bytes, _TMA_ALIGNMENT)
    expected_topk = tuple(
        int(value) for value in source_topk_idx[expected.token].tolist())
    assert struct.unpack_from(
        f"<{_NUM_TOPK}i", raw, metadata_offset) == expected_topk
    weights_offset = metadata_offset + _NUM_TOPK * 4
    expected_weights = source_topk_weights[expected.token].contiguous().view(
        torch.uint8).numpy().tobytes()
    assert raw[weights_offset:weights_offset + _NUM_TOPK * 4] == \
        expected_weights
    src_global_offset = weights_offset + _NUM_TOPK * 4
    src_token_global_idx = expected.owner * _NUM_TOKENS + expected.token
    assert struct.unpack_from("<i", raw, src_global_offset)[0] == \
        src_token_global_idx
    assert struct.unpack_from(
        f"<{_NUM_TOPK}i", raw, src_global_offset + 4) == (-1,) * _NUM_TOPK

    descriptor = struct.unpack_from(
        "<Q14i", raw, layout.descriptor_offset)
    assert descriptor == (
        expected.fingerprint,
        expected.owner,
        expected.token,
        expected.destination,
        expected.owner_ordinal,
        expected.egress,
        expected.channel,
        expected.logical_slot,
        expected.physical_slot,
        _GENERATION,
        src_token_global_idx,
        0, 0, 0, 0,
    )

    # Freeze every padding byte as well as every semantic field.
    expected_raw = bytearray(layout.record_bytes)
    struct.pack_into("<8I", expected_raw, 0, *([_HEAD_CANARY] * 8))
    expected_raw[token_offset:token_offset + hidden_bytes] = \
        expected_hidden_bytes
    struct.pack_into(
        f"<{_NUM_TOPK}i", expected_raw, metadata_offset, *expected_topk)
    expected_raw[weights_offset:weights_offset + _NUM_TOPK * 4] = \
        expected_weights
    struct.pack_into(
        "<i", expected_raw, src_global_offset, src_token_global_idx)
    struct.pack_into(
        f"<{_NUM_TOPK}i", expected_raw, src_global_offset + 4,
        *([-1] * _NUM_TOPK),
    )
    struct.pack_into(
        "<Q14i", expected_raw, layout.descriptor_offset, *descriptor)
    struct.pack_into(
        "<8I", expected_raw, layout.tail_offset, *([_TAIL_CANARY] * 8))
    assert raw == bytes(expected_raw)


def _expected_global_coverage(
    records: Sequence[ExpectedRecord],
) -> list[tuple[str, int, int, int, int]]:
    coverage = []
    for record in records:
        for rank in (record.egress, _NUM_SOURCE_RANKS + record.egress):
            coverage.append((
                "rail-base", rank, record.physical_slot,
                record.fingerprint, -1,
            ))
            for lane in range(_NUM_TOPK):
                contribution_slot = (
                    _PHYSICAL_CAPACITY
                    + record.physical_slot * _NUM_TOPK + lane)
                coverage.append((
                    "rail-contribution", rank, contribution_slot,
                    record.fingerprint, lane,
                ))
        for lane in range(_NUM_TOPK):
            expert_rank = (
                _NUM_SOURCE_RANKS
                + _expert_local_rank(record.owner, record.token, lane)
            )
            expert_slot = (
                (record.egress * _PHYSICAL_CAPACITY + record.physical_slot)
                * _NUM_TOPK + lane)
            coverage.append((
                "expert", expert_rank, expert_slot,
                record.fingerprint, lane,
            ))
    return coverage


def _verify_local_outputs(
    rank: int,
    outputs: Sequence[torch.Tensor],
    records: Sequence[ExpectedRecord],
    source_inputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    expected_partials: Sequence[torch.Tensor],
    expected_outputs: Sequence[torch.Tensor],
    layout: Layout,
    hidden: int,
) -> list[tuple[str, int, int, int, int]]:
    assert len(outputs) == 12
    (combined_output, output_ready, owner_partials, owner_partial_ready,
     rail_records, rail_ready, rail_routes, expert_records, expert_ready,
     expert_routes, stage_status, arena_guard) = outputs
    for tensor in outputs:
        assert tensor.is_cuda and tensor.device.index == rank
        assert tensor.is_contiguous()
    assert combined_output.dtype == owner_partials.dtype == torch.bfloat16
    assert rail_records.dtype == rail_routes.dtype == torch.uint8
    assert expert_records.dtype == expert_routes.dtype == torch.uint8
    assert (output_ready.dtype == owner_partial_ready.dtype ==
            rail_ready.dtype == expert_ready.dtype ==
            stage_status.dtype == torch.int32)
    assert tuple(combined_output.shape) == (_NUM_TOKENS, hidden)
    assert tuple(owner_partials.shape) == (
        _NUM_TOKENS, _NUM_TOPK, hidden)
    assert tuple(rail_records.shape) == (
        layout.rail_capacity, layout.record_bytes)
    assert tuple(rail_ready.shape) == (layout.rail_capacity,)
    assert tuple(rail_routes.shape) == (layout.rail_capacity, 32)
    assert tuple(expert_records.shape) == (
        layout.expert_capacity, layout.record_bytes)
    assert tuple(expert_ready.shape) == (layout.expert_capacity,)
    assert tuple(expert_routes.shape) == (layout.expert_capacity, 32)
    assert tuple(stage_status.shape) == (6, layout.expert_capacity)
    assert tuple(arena_guard.shape) == (_ARENA_GUARD_BYTES,)
    assert arena_guard.dtype == torch.uint8

    cpu = [tensor.cpu() for tensor in outputs]
    (combined_output, output_ready, owner_partials, owner_partial_ready,
     rail_records, rail_ready, rail_routes, expert_records, expert_ready,
     expert_routes, stage_status, arena_guard) = cpu
    assert torch.count_nonzero(stage_status).item() == 0
    assert torch.equal(
        arena_guard,
        torch.full(
            (_ARENA_GUARD_BYTES,), 0xA5,
            dtype=torch.uint8, device="cpu"),
    )

    coverage: list[tuple[str, int, int, int, int]] = []
    local_egress = rank % _NUM_SOURCE_RANKS
    local_records = sorted(
        (record for record in records if record.egress == local_egress),
        key=lambda record: record.physical_slot,
    )
    assert len(local_records) == _PHYSICAL_CAPACITY
    assert torch.equal(
        rail_ready,
        torch.full(
            (layout.rail_capacity,), _GENERATION,
            dtype=torch.int32, device="cpu"),
    )
    for record in local_records:
        base_slot = record.physical_slot
        _verify_record(
            rail_records[base_slot], record,
            source_inputs[record.owner][0][record.token],
            source_inputs, layout)
        assert _route_bytes(rail_routes[base_slot]) == bytes(32)
        coverage.append((
            "rail-base", rank, base_slot, record.fingerprint, -1))
        for lane in range(_NUM_TOPK):
            contribution_slot = (
                _PHYSICAL_CAPACITY + base_slot * _NUM_TOPK + lane)
            _verify_record(
                rail_records[contribution_slot], record,
                expected_partials[record.owner][record.token, lane],
                source_inputs, layout)
            _verify_route(rail_routes[contribution_slot], record, lane)
            coverage.append((
                "rail-contribution", rank, contribution_slot,
                record.fingerprint, lane,
            ))

    if rank < _NUM_SOURCE_RANKS:
        active = _ACTIVE_TOKENS[rank]
        expected_ready = torch.zeros(
            _NUM_TOKENS, dtype=torch.int32, device="cpu")
        expected_ready[:active] = _GENERATION
        expected_partial_ready = torch.zeros(
            (_NUM_TOKENS, _NUM_TOPK), dtype=torch.int32, device="cpu")
        expected_partial_ready[:active] = _GENERATION
        assert torch.equal(output_ready, expected_ready)
        assert torch.equal(owner_partial_ready, expected_partial_ready)
        assert torch.equal(owner_partials, expected_partials[rank])
        assert torch.equal(combined_output, expected_outputs[rank])
        assert torch.count_nonzero(expert_records).item() == 0
        assert torch.count_nonzero(expert_ready).item() == 0
        assert torch.count_nonzero(expert_routes).item() == 0
    else:
        assert torch.count_nonzero(combined_output).item() == 0
        assert torch.count_nonzero(output_ready).item() == 0
        assert torch.count_nonzero(owner_partials).item() == 0
        assert torch.count_nonzero(owner_partial_ready).item() == 0

        expected_expert_slots = {}
        for record in records:
            for lane in range(_NUM_TOPK):
                expert_rank = (
                    _NUM_SOURCE_RANKS
                    + _expert_local_rank(record.owner, record.token, lane)
                )
                if expert_rank != rank:
                    continue
                expert_slot = (
                    (record.egress * _PHYSICAL_CAPACITY
                     + record.physical_slot)
                    * _NUM_TOPK + lane)
                assert expert_slot not in expected_expert_slots
                expected_expert_slots[expert_slot] = (record, lane)
        expected_ready = torch.zeros(
            layout.expert_capacity, dtype=torch.int32, device="cpu")
        for expert_slot in expected_expert_slots:
            expected_ready[expert_slot] = _GENERATION
        assert torch.equal(expert_ready, expected_ready)
        for expert_slot in range(layout.expert_capacity):
            expected = expected_expert_slots.get(expert_slot)
            if expected is None:
                assert torch.count_nonzero(expert_records[expert_slot]).item() == 0
                assert _route_bytes(expert_routes[expert_slot]) == bytes(32)
                continue
            record, lane = expected
            _verify_record(
                expert_records[expert_slot], record,
                source_inputs[record.owner][0][record.token],
                source_inputs, layout)
            _verify_route(expert_routes[expert_slot], record, lane)
            coverage.append((
                "expert", rank, expert_slot, record.fingerprint, lane))
    return coverage


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(
        local_rank, num_local_ranks, seed=_GENERATION)
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(seconds=args.timeout),
    )
    assert rank == local_rank and world_size == _WORLD_SIZE
    assert torch.cuda.current_device() == local_rank

    buffer = None
    clean_shutdown = False
    try:
        full, records = _checked_local_phase(
            "full-egress CPU oracle", control_group, _build_oracle)
        assert full.records_per_egress == (_PHYSICAL_CAPACITY,) * 4
        source_inputs = tuple(
            _source_inputs(owner, args.hidden)
            for owner in range(_NUM_SOURCE_RANKS))
        expected_results = tuple(
            _expected_source_results(owner, args.hidden)
            for owner in range(_NUM_SOURCE_RANKS))
        expected_partials = tuple(item[0] for item in expected_results)
        expected_outputs = tuple(item[1] for item in expected_results)

        x, topk_idx, topk_weights = _checked_local_phase(
            "deterministic local inputs", control_group,
            lambda: _make_local_inputs(rank, args.hidden),
        )
        source_snapshot = (
            x.cpu().clone(), topk_idx.cpu().clone(), topk_weights.cpu().clone())
        manifest, fingerprints = _manifest_for_rank(records, rank)
        quota = torch.full(
            (_NUM_SOURCE_RANKS, 1),
            _PHYSICAL_CAPACITY,
            dtype=torch.int32,
            device=torch.device("cuda", rank),
        )
        preflight = (manifest.cpu().tolist(), fingerprints.cpu().tolist())
        gathered_preflight = [None] * world_size
        dist.all_gather_object(
            gathered_preflight, preflight, group=control_group)
        _checked_local_phase(
            "full-egress manifest preflight", control_group,
            lambda: _verify_preflight(
                gathered_preflight, records),  # type: ignore[arg-type]
        )

        layout = _checked_local_phase(
            "vnode layout ABI", control_group,
            lambda: _layout(args.hidden),
        )
        base_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
            ep_group,
            num_max_tokens_per_rank=_NUM_TOKENS,
            hidden=args.hidden,
            num_topk=_NUM_TOPK,
            use_fp8_dispatch=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=False,
        )
        arena_alignment = int(_C.get_elastic_buffer_alignment())
        arena_offset = align(base_bytes, arena_alignment)
        assert arena_offset % arena_alignment == 0
        buffer = deep_ep.ElasticBuffer(
            ep_group,
            num_bytes=align(
                arena_offset + layout.arena_bytes + _ARENA_GUARD_BYTES,
                arena_alignment,
            ),
            num_max_tokens_per_rank=_NUM_TOKENS,
            hidden=args.hidden,
            num_topk=_NUM_TOPK,
            allow_hybrid_mode=False,
            allow_multiple_reduction=False,
            prefer_overlap_with_compute=False,
            explicitly_destroy=True,
            num_gpu_timeout_secs=args.timeout,
            num_cpu_timeout_secs=args.timeout,
        )
        assert buffer.get_logical_domain_size() == (1, _WORLD_SIZE)
        assert buffer.scaleout_rank_idx == 0 and buffer.scaleup_rank_idx == rank

        collective_config = (
            world_size, args.hidden, _NUM_TOPK, _PHYSICAL_CAPACITY,
            _GENERATION, _NUM_TOKENS, _NUM_SOURCE_RANKS,
            _EXPERT_BEGIN, _EXPERTS_PER_RANK, arena_offset,
            layout.arena_bytes,
        )
        configs = [None] * world_size
        dist.all_gather_object(configs, collective_config, group=control_group)
        _checked_local_phase(
            "collective configuration", control_group,
            lambda: assert_all_equal(configs),
        )
        dist.monitored_barrier(
            group=control_group,
            timeout=timedelta(seconds=args.timeout),
            wait_all_ranks=True,
        )

        # The private method contains eight all-rank LSA barriers. Let any
        # communication exception escape so mp.spawn can terminate peer ranks.
        outputs = buffer.runtime._rail_balance_vnode_roundtrip(
            x,
            topk_idx,
            topk_weights,
            manifest,
            fingerprints,
            quota,
            arena_offset,
            _PHYSICAL_CAPACITY,
            _PHYSICAL_CAPACITY,
            1,
            _GENERATION,
            _NUM_TOKENS,
            _NUM_SOURCE_RANKS,
            _EXPERT_BEGIN,
            _EXPERTS_PER_RANK,
        )
        torch.cuda.synchronize()
        local_coverage = _checked_local_phase(
            "local vnode payload/route/output verification", control_group,
            lambda: _verify_local_outputs(
                rank,
                outputs,
                records,
                source_inputs,
                expected_partials,
                expected_outputs,
                layout,
                args.hidden,
            ),
        )
        gathered_coverage = [None] * world_size
        dist.all_gather_object(
            gathered_coverage, local_coverage, group=control_group)

        def verify_global_coverage() -> None:
            observed = [
                item
                for rank_items in gathered_coverage
                for item in rank_items  # type: ignore[union-attr]
            ]
            expected = _expected_global_coverage(records)
            assert len(observed) == len(set(observed))
            assert sorted(observed) == sorted(expected)

        _checked_local_phase(
            "global order-independent record/route coverage",
            control_group,
            verify_global_coverage,
        )

        def verify_source_immutable() -> None:
            expected_x, expected_idx, expected_weights = source_snapshot
            assert torch.equal(x.cpu(), expected_x)
            assert torch.equal(topk_idx.cpu(), expected_idx)
            assert torch.equal(topk_weights.cpu(), expected_weights)

        _checked_local_phase(
            "source tensors remain immutable", control_group,
            verify_source_immutable,
        )
        del outputs, manifest, fingerprints, quota
        torch.cuda.synchronize()
        dist.monitored_barrier(
            group=control_group,
            timeout=timedelta(seconds=args.timeout),
            wait_all_ranks=True,
        )
        clean_shutdown = True
    finally:
        # ElasticBuffer.destroy() is collective.  Only enter it after every
        # rank has completed the final Gloo checkpoint.
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
                    "PASS C060 4x2 vnode round trip: "
                    f"hidden={args.hidden}, 80 full-egress records, "
                    "320 independent expert contributions, exact combine",
                    flush=True,
                )
            dist.destroy_process_group()


def assert_all_equal(values: Sequence[object]) -> None:
    assert values
    assert all(value == values[0] for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C060 deterministic 4+4 virtual-node round-trip test")
    if not __debug__:
        parser.error("C060 correctness checks require Python assertions")
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--master-port", type=int, default=29661)
    args = parser.parse_args()
    if args.num_processes != _WORLD_SIZE:
        parser.error("C060 4x2 requires exactly 8 processes")
    if args.hidden <= 0 or args.hidden % 16:
        parser.error("--hidden must be a positive multiple of 16")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("C060 4x2 requires at least 8 visible CUDA devices")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.master_port)
    # init_dist interprets WORLD_SIZE as number of nodes and RANK as node rank.
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
