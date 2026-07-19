"""C070: CPU oracle for replaying a frozen C061 vnode snapshot.

The replay path intentionally receives only the bytes and counters that survive
dispatch: token records, 32-byte routes, ready generations, quotas, and top-k
indices.  It has no planner manifest, move segments, or source routing table.
The frozen descriptor payloads below are protocol snapshots, not planner input.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass

import torch


_NUM_SOURCE_RANKS = 2
_NUM_DESTINATIONS = 3
_NUM_TOKENS = 10
_NUM_TOPK = 4
_DESTINATION_CAPACITY = 6
_BASE_CAPACITY = _NUM_DESTINATIONS * _DESTINATION_CAPACITY
_RAIL_CAPACITY = _BASE_CAPACITY * (_NUM_TOPK + 1)
_GENERATION = 62
_EXPERT_BEGIN = 16
_EXPERTS_PER_RANK = 4
_ALIGNMENT = 32
_HEAD_CANARY = 0xDEADBEEF
_TAIL_CANARY = 0xC001D00D
_WEIGHTS = (0.5, 0.25, 0.125, 0.125)
_QUOTA = ((6, 5, 6), (5, 6, 5))

# This is the top-k tensor copied from the passing C061 H200 snapshot.  Replay
# treats it as an input tensor; no destination-per-lane source table exists in
# this file.
_FROZEN_TOPK = (
    (
        (16, 30, 32, 23),
        (31, 33, 29, 17),
        (34, 39, 18, 23),
        (36, 19, 20, 35),
        (16, 21, 32, 37),
        (22, 33, 38, 17),
        (16, 21, 18, 23),
        (22, 19, 20, 17),
        (16, 21, 18, 23),
        (-1, -1, -1, -1),
    ),
    (
        (28, 34, 30, 18),
        (35, 36, 19, 29),
        (37, 25, 30, 32),
        (26, 31, 33, 38),
        (28, 34, 39, 27),
        (26, 31, 24, 29),
        (28, 25, 30, 27),
        (26, 31, 24, 29),
        (28, 25, 30, 27),
        (-1, -1, -1, -1),
    ),
)

# Exact 64-byte ``<Q14i`` descriptor regions captured from the 33 C061 base
# records, ordered by (egress, physical_slot).  Keeping these as bytes ensures
# the replay oracle cannot regenerate an answer by invoking the planner.
_FROZEN_DESCRIPTOR_HEX = (
    "0000000000003e0000000000000000000000000000000000000000000000000000000000000000003e0000000000000000000000000000000000000000000000",
    "0200000002003e0000000000020000000000000002000000000000000000000001000000010000003e0000000200000000000000000000000000000000000000",
    "0400000004003e0000000000040000000000000004000000000000000000000002000000020000003e0000000400000000000000000000000000000000000000",
    "0100000001003e0000000000010000000000000001000000000000000100000000000000030000003e0000000100000000000000000000000000000000000000",
    "0300000003003e0000000000030000000000000003000000000000000100000001000000040000003e0000000300000000000000000000000000000000000000",
    "0500000005003e0000000000050000000000000005000000000000000100000002000000050000003e0000000500000000000000000000000000000000000000",
    "0000000100003e0000000000000000000100000000000000000000000000000000000000060000003e0000000000000000000000000000000000000000000000",
    "0600000106103e0001000000060000000100000006000000000000000000000001000000070000003e0000001000000000000000000000000000000000000000",
    "0800000108103e0001000000080000000100000008000000000000000000000002000000080000003e0000001200000000000000000000000000000000000000",
    "0100000101003e0000000000010000000100000001000000000000000100000000000000090000003e0000000100000000000000000000000000000000000000",
    "0700000107103e00010000000700000001000000070000000000000001000000010000000a0000003e0000001100000000000000000000000000000000000000",
    "0000000200003e00000000000000000002000000000000000000000000000000000000000c0000003e0000000000000000000000000000000000000000000000",
    "0200000202003e00000000000200000002000000020000000000000000000000010000000d0000003e0000000200000000000000000000000000000000000000",
    "0400000204003e00000000000400000002000000040000000000000000000000020000000e0000003e0000000400000000000000000000000000000000000000",
    "0100000201003e00000000000100000002000000010000000000000001000000000000000f0000003e0000000100000000000000000000000000000000000000",
    "0300000203003e0000000000030000000200000003000000000000000100000001000000100000003e0000000300000000000000000000000000000000000000",
    "0500000205003e0000000000050000000200000005000000000000000100000002000000110000003e0000000500000000000000000000000000000000000000",
    "0000000000103e0001000000000000000000000000000000010000000000000000000000000000003e0000000a00000000000000000000000000000000000000",
    "0600000006003e0000000000060000000000000006000000010000000000000001000000010000003e0000000600000000000000000000000000000000000000",
    "0800000008003e0000000000080000000000000008000000010000000000000002000000020000003e0000000800000000000000000000000000000000000000",
    "0100000001103e0001000000010000000000000001000000010000000100000000000000030000003e0000000b00000000000000000000000000000000000000",
    "0700000007003e0000000000070000000000000007000000010000000100000001000000040000003e0000000700000000000000000000000000000000000000",
    "0000000100103e0001000000000000000100000000000000010000000000000000000000060000003e0000000a00000000000000000000000000000000000000",
    "0200000102103e0001000000020000000100000002000000010000000000000001000000070000003e0000000c00000000000000000000000000000000000000",
    "0400000104103e0001000000040000000100000004000000010000000000000002000000080000003e0000000e00000000000000000000000000000000000000",
    "0100000101103e0001000000010000000100000001000000010000000100000000000000090000003e0000000b00000000000000000000000000000000000000",
    "0300000103103e00010000000300000001000000030000000100000001000000010000000a0000003e0000000d00000000000000000000000000000000000000",
    "0500000105103e00010000000500000001000000050000000100000001000000020000000b0000003e0000000f00000000000000000000000000000000000000",
    "0000000200103e00010000000000000002000000000000000100000000000000000000000c0000003e0000000a00000000000000000000000000000000000000",
    "0200000202103e00010000000200000002000000020000000100000000000000010000000d0000003e0000000c00000000000000000000000000000000000000",
    "0400000204103e00010000000400000002000000040000000100000000000000020000000e0000003e0000000e00000000000000000000000000000000000000",
    "0100000201103e00010000000100000002000000010000000100000001000000000000000f0000003e0000000b00000000000000000000000000000000000000",
    "0300000203103e0001000000030000000200000003000000010000000100000001000000100000003e0000000d00000000000000000000000000000000000000",
)


def _align(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class RecordLayout:
    hidden: int
    record_bytes: int
    head_bytes: int
    token_bytes: int
    descriptor_offset: int
    tail_offset: int


@dataclass(frozen=True)
class ReplaySnapshot:
    layout: RecordLayout
    # First dimension is destination ingress index, i.e. rank-G.
    records: torch.Tensor
    routes: torch.Tensor
    ready: torch.Tensor
    quota: torch.Tensor
    topk_idx: torch.Tensor


@dataclass(frozen=True, order=True)
class ParsedDescriptor:
    fingerprint: int
    owner: int
    token: int
    destination: int
    owner_ordinal: int
    egress: int
    channel: int
    logical_slot: int
    physical_slot: int
    generation: int
    src_token_global_idx: int


@dataclass(frozen=True, order=True)
class ReplayContribution:
    owner: int
    token: int
    lane: int
    destination: int
    egress: int
    ingress_rank: int
    expert_rank: int
    expert_idx: int
    expert_slot: int
    physical_slot: int
    fingerprint: int


@dataclass(frozen=True)
class ReplayResult:
    base_descriptors: tuple[ParsedDescriptor, ...]
    contributions: tuple[ReplayContribution, ...]
    owner_partials: torch.Tensor
    combined_output: torch.Tensor


def _record_layout(hidden: int) -> RecordLayout:
    assert hidden > 0 and hidden % 16 == 0
    hidden_bytes = hidden * torch.bfloat16.itemsize
    metadata_bytes = _NUM_TOPK * 2 * 4 + (1 + _NUM_TOPK) * 4
    token_bytes = _align(hidden_bytes) + _align(metadata_bytes)
    descriptor_offset = 32 + token_bytes
    tail_offset = descriptor_offset + 64
    return RecordLayout(
        hidden=hidden,
        record_bytes=_align(tail_offset + 32),
        head_bytes=32,
        token_bytes=token_bytes,
        descriptor_offset=descriptor_offset,
        tail_offset=tail_offset,
    )


def _decode_descriptor(raw: bytes, offset: int) -> ParsedDescriptor:
    fields = struct.unpack_from("<Q14i", raw, offset)
    assert fields[11:] == (0, 0, 0, 0)
    return ParsedDescriptor(*fields[:11])


def _expert_from_topk(expert_idx: int) -> tuple[int, int, int]:
    assert _EXPERT_BEGIN <= expert_idx < (
        _EXPERT_BEGIN
        + _NUM_DESTINATIONS * _NUM_SOURCE_RANKS * _EXPERTS_PER_RANK)
    relative_expert = expert_idx - _EXPERT_BEGIN
    relative_rank, local_expert = divmod(
        relative_expert, _EXPERTS_PER_RANK)
    destination, rail = divmod(relative_rank, _NUM_SOURCE_RANKS)
    expert_rank = _NUM_SOURCE_RANKS + relative_rank
    assert 0 <= local_expert < _EXPERTS_PER_RANK
    return destination, expert_rank, rail


def _source_hidden(owner: int, token: int, hidden: int) -> torch.Tensor:
    columns = torch.arange(hidden, dtype=torch.int64, device="cpu")
    return (
        8 * ((5 * owner + 3 * token + (columns & 7)) & 7)
    ).to(torch.bfloat16)


def _weighted_partial(
    owner: int,
    token: int,
    lane: int,
    expert_idx: int,
    hidden: int,
) -> torch.Tensor:
    relative_expert = expert_idx - _EXPERT_BEGIN
    return (
        (_source_hidden(owner, token, hidden).float()
         + 8 * (relative_expert + 1)) * _WEIGHTS[lane]
    ).to(torch.bfloat16)


def _encode_record(
    descriptor_bytes: bytes,
    payload: torch.Tensor,
    topk_row: torch.Tensor,
    owner: int,
    token: int,
    layout: RecordLayout,
) -> torch.Tensor:
    assert len(descriptor_bytes) == 64
    assert payload.dtype == torch.bfloat16
    assert tuple(payload.shape) == (layout.hidden,)
    raw = bytearray(layout.record_bytes)
    struct.pack_into("<8I", raw, 0, *([_HEAD_CANARY] * 8))
    hidden_bytes = payload.contiguous().view(torch.uint8).numpy().tobytes()
    raw[layout.head_bytes:layout.head_bytes + len(hidden_bytes)] = hidden_bytes

    metadata_offset = layout.head_bytes + _align(len(hidden_bytes))
    topk = tuple(int(value) for value in topk_row.tolist())
    struct.pack_into(f"<{_NUM_TOPK}i", raw, metadata_offset, *topk)
    weights_offset = metadata_offset + _NUM_TOPK * 4
    struct.pack_into(f"<{_NUM_TOPK}f", raw, weights_offset, *_WEIGHTS)
    src_global_offset = weights_offset + _NUM_TOPK * 4
    struct.pack_into(
        "<i", raw, src_global_offset, owner * _NUM_TOKENS + token)
    struct.pack_into(
        f"<{_NUM_TOPK}i", raw, src_global_offset + 4,
        *([-1] * _NUM_TOPK),
    )
    raw[layout.descriptor_offset:layout.descriptor_offset + 64] = \
        descriptor_bytes
    struct.pack_into(
        "<8I", raw, layout.tail_offset, *([_TAIL_CANARY] * 8))
    return torch.tensor(list(raw), dtype=torch.uint8, device="cpu")


def _build_frozen_c061_snapshot(hidden: int) -> ReplaySnapshot:
    """Materialize frozen C061 protocol bytes without calling its planner."""

    layout = _record_layout(hidden)
    records = torch.zeros(
        (_NUM_DESTINATIONS * _NUM_SOURCE_RANKS,
         _RAIL_CAPACITY, layout.record_bytes),
        dtype=torch.uint8,
        device="cpu",
    )
    routes = torch.zeros(
        (_NUM_DESTINATIONS * _NUM_SOURCE_RANKS,
         _RAIL_CAPACITY, 32),
        dtype=torch.uint8,
        device="cpu",
    )
    ready = torch.zeros(
        (_NUM_DESTINATIONS * _NUM_SOURCE_RANKS, _RAIL_CAPACITY),
        dtype=torch.int32,
        device="cpu",
    )
    quota = torch.tensor(_QUOTA, dtype=torch.int32, device="cpu")
    topk_idx = torch.tensor(
        _FROZEN_TOPK, dtype=torch.int32, device="cpu")

    occupied_base: set[tuple[int, int]] = set()
    occupied_contribution: set[tuple[int, int]] = set()
    for descriptor_hex in _FROZEN_DESCRIPTOR_HEX:
        descriptor_bytes = bytes.fromhex(descriptor_hex)
        descriptor = _decode_descriptor(descriptor_bytes, 0)
        ingress_index = (
            descriptor.destination * _NUM_SOURCE_RANKS
            + descriptor.egress)
        base_slot = descriptor.physical_slot
        assert (ingress_index, base_slot) not in occupied_base
        occupied_base.add((ingress_index, base_slot))
        topk_row = topk_idx[descriptor.owner, descriptor.token]
        records[ingress_index, base_slot] = _encode_record(
            descriptor_bytes,
            _source_hidden(descriptor.owner, descriptor.token, hidden),
            topk_row,
            descriptor.owner,
            descriptor.token,
            layout,
        )
        ready[ingress_index, base_slot] = _GENERATION

        for lane, expert_idx_tensor in enumerate(topk_row):
            expert_idx = int(expert_idx_tensor)
            if expert_idx < 0:
                continue
            lane_destination, expert_rank, _rail = \
                _expert_from_topk(expert_idx)
            if lane_destination != descriptor.destination:
                continue
            contribution_slot = (
                _BASE_CAPACITY
                + descriptor.physical_slot * _NUM_TOPK
                + lane)
            assert (ingress_index, contribution_slot) not in \
                occupied_contribution
            occupied_contribution.add((ingress_index, contribution_slot))
            records[ingress_index, contribution_slot] = _encode_record(
                descriptor_bytes,
                _weighted_partial(
                    descriptor.owner,
                    descriptor.token,
                    lane,
                    expert_idx,
                    hidden,
                ),
                topk_row,
                descriptor.owner,
                descriptor.token,
                layout,
            )
            ingress_rank = _NUM_SOURCE_RANKS + ingress_index
            destination_slot = (
                descriptor.physical_slot % _DESTINATION_CAPACITY)
            expert_slot = (
                (descriptor.egress * _DESTINATION_CAPACITY
                 + destination_slot) * _NUM_TOPK + lane)
            route = struct.pack(
                "<Q6i",
                descriptor.fingerprint,
                _GENERATION,
                ingress_rank,
                descriptor.physical_slot,
                lane,
                expert_rank,
                expert_slot,
            )
            routes[ingress_index, contribution_slot] = torch.tensor(
                list(route), dtype=torch.uint8, device="cpu")
            ready[ingress_index, contribution_slot] = _GENERATION

    assert len(occupied_base) == 33
    assert len(occupied_contribution) == 72
    return ReplaySnapshot(
        layout=layout,
        records=records,
        routes=routes,
        ready=ready,
        quota=quota,
        topk_idx=topk_idx,
    )


def _raw_record(snapshot: ReplaySnapshot, ingress: int, slot: int) -> bytes:
    return snapshot.records[ingress, slot].contiguous().numpy().tobytes()


def _payload_from_record(raw: bytes, layout: RecordLayout) -> torch.Tensor:
    hidden_bytes = layout.hidden * torch.bfloat16.itemsize
    payload = bytearray(
        raw[layout.head_bytes:layout.head_bytes + hidden_bytes])
    return torch.frombuffer(payload, dtype=torch.bfloat16).clone()


def _validate_record_envelope(
    raw: bytes,
    descriptor: ParsedDescriptor,
    snapshot: ReplaySnapshot,
) -> None:
    layout = snapshot.layout
    assert len(raw) == layout.record_bytes
    assert struct.unpack_from("<8I", raw, 0) == (_HEAD_CANARY,) * 8
    assert struct.unpack_from(
        "<8I", raw, layout.tail_offset) == (_TAIL_CANARY,) * 8
    assert descriptor.generation == _GENERATION
    assert 0 <= descriptor.owner < _NUM_SOURCE_RANKS
    assert 0 <= descriptor.token < _NUM_TOKENS
    assert 0 <= descriptor.destination < _NUM_DESTINATIONS
    assert 0 <= descriptor.egress < _NUM_SOURCE_RANKS
    assert descriptor.src_token_global_idx == \
        descriptor.owner * _NUM_TOKENS + descriptor.token
    expected_fingerprint = (
        (_GENERATION << 48)
        | (descriptor.owner << 44)
        | (descriptor.token << 32)
        | (descriptor.destination << 24)
        | descriptor.owner_ordinal
    )
    assert descriptor.fingerprint == expected_fingerprint

    hidden_bytes = layout.hidden * torch.bfloat16.itemsize
    metadata_offset = layout.head_bytes + _align(hidden_bytes)
    record_topk = struct.unpack_from(
        f"<{_NUM_TOPK}i", raw, metadata_offset)
    assert record_topk == tuple(
        int(value) for value in
        snapshot.topk_idx[descriptor.owner, descriptor.token].tolist())
    weights_offset = metadata_offset + _NUM_TOPK * 4
    assert struct.unpack_from(
        f"<{_NUM_TOPK}f", raw, weights_offset) == _WEIGHTS
    src_global_offset = weights_offset + _NUM_TOPK * 4
    assert struct.unpack_from("<i", raw, src_global_offset)[0] == \
        descriptor.src_token_global_idx
    assert struct.unpack_from(
        f"<{_NUM_TOPK}i", raw, src_global_offset + 4
    ) == (-1,) * _NUM_TOPK


def replay_c061_snapshot(snapshot: ReplaySnapshot) -> ReplayResult:
    """Recover C061 routing solely from persisted replay inputs."""

    layout = snapshot.layout
    assert snapshot.records.dtype == snapshot.routes.dtype == torch.uint8
    assert snapshot.ready.dtype == snapshot.quota.dtype == torch.int32
    # The frozen byte fixture uses int32 indices, while DeepEP's public
    # routing input is int64. Record metadata is decoded independently below,
    # so both integer source representations are valid replay inputs.
    assert snapshot.topk_idx.dtype in (torch.int32, torch.int64)
    assert tuple(snapshot.records.shape) == (
        _NUM_DESTINATIONS * _NUM_SOURCE_RANKS,
        _RAIL_CAPACITY,
        layout.record_bytes,
    )
    assert tuple(snapshot.routes.shape) == (
        _NUM_DESTINATIONS * _NUM_SOURCE_RANKS,
        _RAIL_CAPACITY,
        32,
    )
    assert tuple(snapshot.ready.shape) == (
        _NUM_DESTINATIONS * _NUM_SOURCE_RANKS, _RAIL_CAPACITY)
    assert tuple(snapshot.quota.shape) == (
        _NUM_SOURCE_RANKS, _NUM_DESTINATIONS)
    assert tuple(snapshot.topk_idx.shape) == (
        _NUM_SOURCE_RANKS, _NUM_TOKENS, _NUM_TOPK)

    base_by_location: dict[tuple[int, int], ParsedDescriptor] = {}
    contribution_by_location: dict[
        tuple[int, int], tuple[ReplayContribution, ParsedDescriptor]
    ] = {}
    partials = torch.zeros(
        (_NUM_SOURCE_RANKS, _NUM_TOKENS, _NUM_TOPK, layout.hidden),
        dtype=torch.bfloat16,
        device="cpu",
    )
    partial_ready = torch.zeros(
        (_NUM_SOURCE_RANKS, _NUM_TOKENS, _NUM_TOPK),
        dtype=torch.bool,
        device="cpu",
    )

    for ingress_index in range(_NUM_DESTINATIONS * _NUM_SOURCE_RANKS):
        destination, egress = divmod(
            ingress_index, _NUM_SOURCE_RANKS)
        ingress_rank = _NUM_SOURCE_RANKS + ingress_index
        for slot in range(_RAIL_CAPACITY):
            generation = int(snapshot.ready[ingress_index, slot])
            route_raw = snapshot.routes[
                ingress_index, slot].contiguous().numpy().tobytes()
            raw = _raw_record(snapshot, ingress_index, slot)
            if generation == 0:
                assert not any(raw)
                assert route_raw == bytes(32)
                continue
            assert generation == _GENERATION
            descriptor = _decode_descriptor(raw, layout.descriptor_offset)
            _validate_record_envelope(raw, descriptor, snapshot)
            assert descriptor.destination == destination
            assert descriptor.egress == egress
            assert descriptor.physical_slot // _DESTINATION_CAPACITY == \
                destination

            if slot < _BASE_CAPACITY:
                assert slot == descriptor.physical_slot
                assert route_raw == bytes(32)
                key = (ingress_index, slot)
                assert key not in base_by_location
                base_by_location[key] = descriptor
                continue

            route = struct.unpack("<Q6i", route_raw)
            (fingerprint, route_generation, route_ingress_rank,
             physical_slot, lane, expert_rank, expert_slot) = route
            assert fingerprint == descriptor.fingerprint
            assert route_generation == _GENERATION
            assert route_ingress_rank == ingress_rank
            assert physical_slot == descriptor.physical_slot
            assert 0 <= lane < _NUM_TOPK
            assert slot == (
                _BASE_CAPACITY
                + descriptor.physical_slot * _NUM_TOPK
                + lane)

            expert_idx = int(snapshot.topk_idx[
                descriptor.owner, descriptor.token, lane])
            lane_destination, expected_expert_rank, _rail = \
                _expert_from_topk(expert_idx)
            assert lane_destination == descriptor.destination
            assert expert_rank == expected_expert_rank
            destination_slot = (
                descriptor.physical_slot % _DESTINATION_CAPACITY)
            expected_expert_slot = (
                (descriptor.egress * _DESTINATION_CAPACITY
                 + destination_slot) * _NUM_TOPK + lane)
            assert expert_slot == expected_expert_slot

            replayed = ReplayContribution(
                owner=descriptor.owner,
                token=descriptor.token,
                lane=lane,
                destination=descriptor.destination,
                egress=descriptor.egress,
                ingress_rank=ingress_rank,
                expert_rank=expert_rank,
                expert_idx=expert_idx,
                expert_slot=expert_slot,
                physical_slot=descriptor.physical_slot,
                fingerprint=descriptor.fingerprint,
            )
            location = (ingress_index, slot)
            assert location not in contribution_by_location
            contribution_by_location[location] = (replayed, descriptor)
            partial_key = (descriptor.owner, descriptor.token, lane)
            assert not bool(partial_ready[partial_key])
            partials[partial_key] = _payload_from_record(raw, layout)
            partial_ready[partial_key] = True

    # Quota is sufficient to validate every physical base prefix and hole;
    # planner segments are neither available nor needed.
    for destination in range(_NUM_DESTINATIONS):
        for egress in range(_NUM_SOURCE_RANKS):
            ingress_index = destination * _NUM_SOURCE_RANKS + egress
            expected_local_slots = set(
                range(int(snapshot.quota[egress, destination])))
            observed_local_slots = {
                physical % _DESTINATION_CAPACITY
                for (observed_ingress, physical) in base_by_location
                if observed_ingress == ingress_index
            }
            assert observed_local_slots == expected_local_slots

    # A contribution must name an existing base descriptor byte-for-byte, and
    # top-k determines the exact set of lanes belonging to that destination.
    expected_contribution_locations = set()
    for (ingress_index, physical_slot), descriptor in base_by_location.items():
        for lane, expert_idx_tensor in enumerate(
                snapshot.topk_idx[descriptor.owner, descriptor.token]):
            expert_idx = int(expert_idx_tensor)
            if expert_idx < 0:
                continue
            lane_destination, _expert_rank, _rail = \
                _expert_from_topk(expert_idx)
            if lane_destination != descriptor.destination:
                continue
            location = (
                ingress_index,
                _BASE_CAPACITY + physical_slot * _NUM_TOPK + lane,
            )
            expected_contribution_locations.add(location)
            replayed, contribution_descriptor = \
                contribution_by_location[location]
            assert contribution_descriptor == descriptor
            assert replayed.lane == lane
    assert set(contribution_by_location) == expected_contribution_locations

    expected_partial_ready = snapshot.topk_idx >= 0
    assert torch.equal(partial_ready, expected_partial_ready)
    combined = torch.zeros(
        (_NUM_SOURCE_RANKS, _NUM_TOKENS, layout.hidden),
        dtype=torch.bfloat16,
        device="cpu",
    )
    for owner in range(_NUM_SOURCE_RANKS):
        for token in range(_NUM_TOKENS):
            reduced = torch.zeros(
                layout.hidden, dtype=torch.float32, device="cpu")
            for lane in range(_NUM_TOPK):
                if partial_ready[owner, token, lane]:
                    reduced += partials[owner, token, lane].float()
            combined[owner, token] = reduced.to(torch.bfloat16)

    return ReplayResult(
        base_descriptors=tuple(sorted(base_by_location.values())),
        contributions=tuple(sorted(
            item[0] for item in contribution_by_location.values())),
        owner_partials=partials,
        combined_output=combined,
    )


def _independent_expected_output(
    topk_idx: torch.Tensor,
    hidden: int,
) -> torch.Tensor:
    expected = torch.zeros(
        (_NUM_SOURCE_RANKS, _NUM_TOKENS, hidden),
        dtype=torch.bfloat16,
        device="cpu",
    )
    for owner in range(_NUM_SOURCE_RANKS):
        for token in range(_NUM_TOKENS):
            reduced = torch.zeros(hidden, dtype=torch.float32, device="cpu")
            for lane in range(_NUM_TOPK):
                expert_idx = int(topk_idx[owner, token, lane])
                if expert_idx >= 0:
                    reduced += _weighted_partial(
                        owner, token, lane, expert_idx, hidden).float()
            expected[owner, token] = reduced.to(torch.bfloat16)
    return expected


def _run_cpu_replay_oracle(hidden: int) -> None:
    snapshot = _build_frozen_c061_snapshot(hidden)
    result = replay_c061_snapshot(snapshot)

    assert len(result.base_descriptors) == 33
    assert len(result.contributions) == 72
    assert len({
        (item.owner, item.token, item.lane)
        for item in result.contributions
    }) == 72
    assert sum(
        descriptor.owner != descriptor.egress
        for descriptor in result.base_descriptors
    ) == 6
    assert sum(
        item.owner != item.egress
        for item in result.contributions
    ) == 24

    expert_counts = [0] * (
        _NUM_DESTINATIONS * _NUM_SOURCE_RANKS)
    ingress_counts = [0] * (
        _NUM_DESTINATIONS * _NUM_SOURCE_RANKS)
    for item in result.contributions:
        expert_counts[item.expert_rank - _NUM_SOURCE_RANKS] += 1
        ingress_counts[item.ingress_rank - _NUM_SOURCE_RANKS] += 1
        assert item.destination == \
            (item.ingress_rank - _NUM_SOURCE_RANKS) // _NUM_SOURCE_RANKS
        assert item.destination == \
            (item.expert_rank - _NUM_SOURCE_RANKS) // _NUM_SOURCE_RANKS
    assert tuple(expert_counts) == (14, 11, 11, 17, 11, 8)
    assert tuple(ingress_counts) == (11, 14, 15, 13, 10, 9)

    for owner in range(_NUM_SOURCE_RANKS):
        for token in range(_NUM_TOKENS):
            for lane in range(_NUM_TOPK):
                expert_idx = int(snapshot.topk_idx[owner, token, lane])
                observed = result.owner_partials[owner, token, lane]
                if expert_idx < 0:
                    assert torch.count_nonzero(observed).item() == 0
                else:
                    assert torch.equal(
                        observed,
                        _weighted_partial(
                            owner, token, lane, expert_idx, hidden),
                    )

    expected = _independent_expected_output(snapshot.topk_idx, hidden)
    assert torch.equal(result.combined_output, expected)
    assert result.combined_output[0, 0, :8].tolist() == [
        59.0, 67.0, 75.0, 83.0, 91.0, 99.0, 107.0, 115.0,
    ]
    assert result.combined_output[1, 8, :8].tolist() == [
        139.0, 147.0, 155.0, 99.0, 107.0, 115.0, 123.0, 131.0,
    ]
    assert torch.count_nonzero(result.combined_output[:, 9]).item() == 0
    assert torch.count_nonzero(result.combined_output[:, :9]).item() > 0
    assert tuple(result.owner_partials.shape) == (
        _NUM_SOURCE_RANKS, _NUM_TOKENS, _NUM_TOPK, hidden)


def test_cpu_replay_oracle() -> None:
    _run_cpu_replay_oracle(hidden=128)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C070 CPU-only C061 snapshot replay oracle")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="explicitly select the only backend implemented at C070",
    )
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()
    if not __debug__:
        parser.error("C070 correctness checks require Python assertions")
    if args.hidden <= 0 or args.hidden % 16:
        parser.error("--hidden must be a positive multiple of 16")
    return args


def main() -> None:
    args = _parse_args()
    _run_cpu_replay_oracle(args.hidden)
    print(
        "PASS C070 CPU replay oracle: "
        f"hidden={args.hidden}, 33 descriptors, 72 routes, "
        "6 moved copies, fixed-lane BF16 combine",
        flush=True,
    )


if __name__ == "__main__":
    main()
