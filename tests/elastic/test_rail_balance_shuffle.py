"""C040: 8-GPU source-side rail-balance shuffle correctness test.

This is a functional test only.  It uses one process per GPU and the pure
single-node ElasticBuffer LSA team; it neither exercises GIN nor reports
performance numbers.

Run from the repository root after rebuilding the extension::

    PYTHONPATH=$PWD EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      python -B tests/elastic/test_rail_balance_shuffle.py
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
    DestinationCopy,
    PhysicalProxyPlan,
    build_count_plan,
    build_destination_copies,
    counts_from_copies,
    materialize_physical_proxy_slots,
    materialize_static_slots,
)


_T = TypeVar("_T")

_WORLD_SIZE = 8
_NUM_TOKENS = 32
_HIDDEN = 7168
_NUM_TOPK = 4
_NUM_DESTINATIONS = 4
_NUM_CHANNELS = 4
_EXPERTS_PER_DESTINATION = 8
_GENERATION = 17
_PHYSICAL_CAPACITY = 32
_TMA_ALIGNMENT = 32
_HEAD_CANARY = 0xDEADBEEF
_TAIL_CANARY = 0xC001D00D


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
    def coverage_key(self) -> tuple[int, ...]:
        return (
            self.egress,
            self.physical_slot,
            self.fingerprint,
            self.owner,
            self.token,
            self.destination,
            self.owner_ordinal,
            self.channel,
            self.logical_slot,
        )


def _checked_local_phase(
    name: str,
    control_group: dist.ProcessGroup,
    function: Callable[[], _T],
) -> _T:
    """Aggregate local-only failures before any rank enters the next phase."""

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
        raise AssertionError(f"{name} failed on one or more ranks:\n" + "\n".join(messages))
    return value  # type: ignore[return-value]


def _destination_lanes(owner: int, token: int) -> list[int]:
    """Adversarial routes with masks, duplicates, and multi-destination tokens."""

    del token
    destination = owner % _NUM_DESTINATIONS
    return [
        destination,
        destination,
        (destination + 1) % _NUM_DESTINATIONS,
        -1,
    ]


def _make_local_inputs(rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
    destination_rows = [
        _destination_lanes(rank, token) for token in range(_NUM_TOKENS)
    ]
    expert_rows = []
    weight_rows = []
    for token, destinations in enumerate(destination_rows):
        experts = []
        weights = []
        for lane, destination in enumerate(destinations):
            if destination < 0:
                experts.append(-1)
                weights.append(0.0)
            else:
                local_expert = (rank + token + lane) % _EXPERTS_PER_DESTINATION
                experts.append(
                    destination * _EXPERTS_PER_DESTINATION + local_expert)
                weights.append((rank + 1) * 0.01 + token * 0.001 + lane * 0.0001)
        expert_rows.append(experts)
        weight_rows.append(weights)

    device = torch.device("cuda", rank)
    topk_idx = torch.tensor(
        expert_rows, dtype=deep_ep.topk_idx_t, device=device)
    topk_weights = torch.tensor(
        weight_rows, dtype=torch.float32, device=device)
    decoded_destinations = torch.where(
        topk_idx >= 0,
        topk_idx // _EXPERTS_PER_DESTINATION,
        topk_idx,
    )
    assert decoded_destinations.cpu().tolist() == destination_rows
    values = torch.arange(
        _NUM_TOKENS * _HIDDEN, dtype=torch.int32, device=device)
    x = ((values + rank * 37) % 127).reshape(
        _NUM_TOKENS, _HIDDEN).to(torch.bfloat16)
    return x, topk_idx, topk_weights, destination_rows


def _local_destination_counts(destination_rows: Sequence[Sequence[int]], rank: int) -> torch.Tensor:
    counts = [0] * _NUM_DESTINATIONS
    for row in destination_rows:
        for destination in set(value for value in row if value >= 0):
            counts[destination] += 1
    return torch.tensor(counts, dtype=torch.int32, device=torch.device("cuda", rank))


def _all_gather_tensor(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    gathered = torch.empty(
        (world_size * local.shape[0], *local.shape[1:]),
        dtype=local.dtype,
        device=local.device,
    )
    dist.all_gather_into_tensor(gathered, local.contiguous(), group=group)
    return gathered.view(world_size, *local.shape)


def _fingerprint(copy: DestinationCopy) -> int:
    return (
        (_GENERATION << 48)
        | (copy.owner << 40)
        | (copy.token << 24)
        | (copy.destination << 16)
        | copy.owner_ordinal
    )


def _build_physical_oracle(
    destination_rows_by_rank: Sequence[Sequence[Sequence[int]]],
) -> tuple[PhysicalProxyPlan, tuple[ExpectedRecord, ...]]:
    copies = build_destination_copies(
        destination_rows_by_rank,
        num_destinations=_NUM_DESTINATIONS,
        num_channels=_NUM_CHANNELS,
        local_destination=None,
    )
    counts = counts_from_copies(
        copies,
        num_rails=_WORLD_SIZE,
        num_destinations=_NUM_DESTINATIONS,
        num_channels=_NUM_CHANNELS,
    )
    count_plan = build_count_plan(counts, remainder_seed=_GENERATION)
    logical = materialize_static_slots(
        copies,
        count_plan,
        num_channels=_NUM_CHANNELS,
        channel_capacity=_PHYSICAL_CAPACITY,
        generation=_GENERATION,
        channel_seed=_GENERATION,
        proxy_payload_capacity=_WORLD_SIZE * _PHYSICAL_CAPACITY,
        metadata_capacity=_WORLD_SIZE * _PHYSICAL_CAPACITY,
    )
    physical = materialize_physical_proxy_slots(
        logical,
        global_policy_budget=_WORLD_SIZE * _PHYSICAL_CAPACITY,
        physical_capacity_per_egress=_PHYSICAL_CAPACITY,
    )
    assert physical.enabled and physical.moved_copies > 0
    assert physical.moved_copies == _WORLD_SIZE * _PHYSICAL_CAPACITY
    assert physical.incoming_copies_per_egress == (
        _PHYSICAL_CAPACITY,) * _WORLD_SIZE
    assert all(item.assignment.moved for item in physical.assignments)

    # Derive the moved set directly from the count-plan segments.  This is an
    # independent integration oracle: the physical helper cannot make both
    # the manifest and expected records agree on a wrong moved copy or egress.
    copy_by_ordinal = {
        (copy.owner, copy.destination, copy.owner_ordinal): copy
        for copy in copies
    }
    segment_egress_by_copy = {}
    for segment in count_plan.segments:
        for owner_ordinal in range(
                segment.owner_begin, segment.owner_begin + segment.count):
            copy = copy_by_ordinal[
                (segment.owner, segment.destination, owner_ordinal)]
            assert copy.key not in segment_egress_by_copy
            segment_egress_by_copy[copy.key] = segment.egress
    physical_egress_by_copy = {
        item.assignment.copy.key: item.egress
        for item in physical.assignments
    }
    assert segment_egress_by_copy == physical_egress_by_copy
    for egress in range(_WORLD_SIZE):
        assert sorted(
            item.physical_slot
            for item in physical.assignments if item.egress == egress
        ) == list(range(_PHYSICAL_CAPACITY))

    records = tuple(ExpectedRecord(
        owner=item.assignment.copy.owner,
        token=item.assignment.copy.token,
        destination=item.assignment.copy.destination,
        owner_ordinal=item.assignment.copy.owner_ordinal,
        egress=item.egress,
        channel=item.assignment.channel,
        logical_slot=item.assignment.slot,
        physical_slot=item.physical_slot,
        fingerprint=_fingerprint(item.assignment.copy),
    ) for item in physical.assignments)

    # The route deliberately proves all C040 copy-granularity corner cases.
    valid_lanes = sum(
        destination >= 0
        for owner_rows in destination_rows_by_rank
        for row in owner_rows
        for destination in row
    )
    assert any(
        len(valid := [value for value in row if value >= 0]) > len(set(valid))
        for owner_rows in destination_rows_by_rank
        for row in owner_rows
    )
    assert len(copies) < valid_lanes  # Same-destination lanes are deduplicated.
    destinations_per_token: dict[tuple[int, int], set[int]] = {}
    owners_per_token: dict[int, set[int]] = {}
    for record in records:
        destinations_per_token.setdefault(
            (record.owner, record.token), set()).add(record.destination)
        owners_per_token.setdefault(record.token, set()).add(record.owner)
    assert any(len(values) > 1 for values in destinations_per_token.values())
    assert any(len(values) > 1 for values in owners_per_token.values())

    retained_keys = {
        assignment.copy.key
        for assignment in logical.assignments
        if not assignment.moved
    }
    assert retained_keys.isdisjoint(
        (record.owner, record.token, record.destination) for record in records)
    return physical, records


def _layout(physical_capacity: int = _PHYSICAL_CAPACITY) -> tuple[int, ...]:
    result = tuple(int(value) for value in
                   _C._get_rail_balance_source_shuffle_layout(
                       _HIDDEN, _NUM_TOPK, physical_capacity))
    (head_bytes, token_bytes, descriptor_offset, tail_offset,
     record_bytes, ready_offset, arena_bytes) = result
    hidden_bytes = _HIDDEN * torch.bfloat16.itemsize
    metadata_bytes = _NUM_TOPK * 2 * 4 + (1 + _NUM_TOPK) * 4
    expected_token_bytes = (
        align(hidden_bytes, _TMA_ALIGNMENT)
        + align(metadata_bytes, _TMA_ALIGNMENT)
    )
    assert head_bytes == 32
    assert token_bytes == expected_token_bytes
    assert descriptor_offset == head_bytes + token_bytes
    assert tail_offset == descriptor_offset + 64
    assert record_bytes == align(tail_offset + 32, _TMA_ALIGNMENT)
    assert ready_offset == record_bytes * physical_capacity
    assert arena_bytes == align(
        ready_offset + align(physical_capacity * 4, _TMA_ALIGNMENT),
        _C.get_elastic_buffer_alignment(),
    )
    return result


def _verify_gpu_count_plan(
    gathered_counts: torch.Tensor,
    physical: PhysicalProxyPlan,
) -> None:
    candidate = physical.logical_candidate.candidate
    quota, keep_count, segments, num_segments, moved_copies = \
        _C._build_rail_balance_plan(gathered_counts.contiguous(), _GENERATION)
    expected_quota = torch.tensor(
        candidate.quota, dtype=torch.int32, device="cpu")
    expected_keep = torch.tensor(
        candidate.keep_count, dtype=torch.int32, device="cpu")
    expected_segments = torch.full(
        (_NUM_DESTINATIONS, _WORLD_SIZE - 1, 4),
        -1,
        dtype=torch.int32,
        device="cpu",
    )
    expected_num_segments = torch.zeros(
        (_NUM_DESTINATIONS,), dtype=torch.int32, device="cpu")
    offsets = [0] * _NUM_DESTINATIONS
    for segment in candidate.segments:
        offset = offsets[segment.destination]
        expected_segments[segment.destination, offset] = torch.tensor(
            [segment.owner, segment.egress, segment.owner_begin, segment.count],
            dtype=torch.int32,
            device="cpu",
        )
        offsets[segment.destination] += 1
        expected_num_segments[segment.destination] += 1
    assert torch.equal(quota.cpu(), expected_quota)
    assert torch.equal(keep_count.cpu(), expected_keep)
    assert torch.equal(segments.cpu(), expected_segments)
    assert torch.equal(num_segments.cpu(), expected_num_segments)
    assert moved_copies.cpu().tolist() == [candidate.moved_copies]


def _verify_collective_preflight(
    gathered: Sequence[tuple[tuple[int, ...], list[list[int]], list[int]]],
    expected_records: Sequence[ExpectedRecord],
) -> None:
    configs = [item[0] for item in gathered]
    assert all(config == configs[0] for config in configs)

    observed = []
    physical_keys = []
    moved_per_owner = [0] * _WORLD_SIZE
    for owner, (_config, manifests, fingerprints) in enumerate(gathered):
        assert len(manifests) == len(fingerprints)
        for manifest, fingerprint in zip(manifests, fingerprints):
            assert len(manifest) == 7
            token, destination, owner_ordinal, egress, channel, logical_slot, physical_slot = manifest
            assert egress != owner
            physical_keys.append((egress, physical_slot, _GENERATION))
            moved_per_owner[owner] += 1
            observed.append((
                owner, token, destination, owner_ordinal, egress,
                channel, logical_slot, physical_slot, fingerprint,
            ))
    assert len(physical_keys) == len(set(physical_keys))
    assert all(count > 0 for count in moved_per_owner)
    expected = [(
        record.owner, record.token, record.destination, record.owner_ordinal,
        record.egress, record.channel, record.logical_slot,
        record.physical_slot, record.fingerprint,
    ) for record in expected_records]
    assert sorted(observed) == sorted(expected)
    for egress in range(_WORLD_SIZE):
        slots = sorted(
            physical_slot
            for key_egress, physical_slot, _generation in physical_keys
            if key_egress == egress
        )
        assert slots == list(range(_PHYSICAL_CAPACITY))


def _verify_empty_collective_preflight(
    gathered: Sequence[tuple[tuple[int, ...], list[list[int]], list[int]]],
) -> None:
    configs = [item[0] for item in gathered]
    assert all(config == configs[0] for config in configs)
    assert all(not manifests and not fingerprints
               for _config, manifests, fingerprints in gathered)


def _manifest_for_owner(
    records: Sequence[ExpectedRecord], owner: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = [record for record in records if record.owner == owner]
    manifest = torch.tensor(
        [record.manifest for record in local], dtype=torch.int32, device="cpu")
    if not local:
        manifest = torch.empty((0, 7), dtype=torch.int32, device="cpu")
    fingerprints = torch.tensor(
        [record.fingerprint for record in local], dtype=torch.int64, device="cpu")
    return manifest.cuda(owner), fingerprints.cuda(owner)


def _decode_and_verify_record(
    record: torch.Tensor,
    expected: ExpectedRecord,
    source_x: torch.Tensor,
    source_topk_idx: torch.Tensor,
    source_topk_weights: torch.Tensor,
    layout: tuple[int, ...],
) -> tuple[int, ...]:
    (head_bytes, _token_bytes, descriptor_offset, tail_offset,
     record_bytes, _ready_offset, _arena_bytes) = layout
    assert record.dtype == torch.uint8 and record.numel() == record_bytes
    raw = bytes(record.tolist())

    assert struct.unpack_from("<8I", raw, 0) == (_HEAD_CANARY,) * 8
    assert struct.unpack_from("<8I", raw, tail_offset) == (_TAIL_CANARY,) * 8

    hidden_bytes = _HIDDEN * torch.bfloat16.itemsize
    token_offset = head_bytes
    expected_hidden = bytes(
        source_x[expected.owner, expected.token].contiguous().view(torch.uint8).tolist())
    assert raw[token_offset:token_offset + hidden_bytes] == expected_hidden

    metadata_offset = token_offset + align(hidden_bytes, _TMA_ALIGNMENT)
    decoded_topk_idx = struct.unpack_from(
        f"<{_NUM_TOPK}i", raw, metadata_offset)
    expected_topk_idx = tuple(int(value) for value in
                              source_topk_idx[expected.owner, expected.token].tolist())
    assert decoded_topk_idx == expected_topk_idx

    weights_offset = metadata_offset + _NUM_TOPK * 4
    expected_weights = bytes(
        source_topk_weights[expected.owner, expected.token]
        .contiguous().view(torch.uint8).tolist())
    assert raw[weights_offset:weights_offset + _NUM_TOPK * 4] == expected_weights

    src_global_offset = weights_offset + _NUM_TOPK * 4
    src_token_global_idx = expected.owner * _NUM_TOKENS + expected.token
    assert struct.unpack_from("<i", raw, src_global_offset)[0] == src_token_global_idx
    assert struct.unpack_from(
        f"<{_NUM_TOPK}i", raw, src_global_offset + 4) == (-1,) * _NUM_TOPK

    descriptor = struct.unpack_from("<Q14i", raw, descriptor_offset)
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

    # Compare the entire record, not only semantic fields.  This freezes all
    # TokenLayout alignment padding and the descriptor/tail padding as zero,
    # which is part of the direct-final publication contract.
    expected_raw = bytearray(record_bytes)
    struct.pack_into("<8I", expected_raw, 0, *([_HEAD_CANARY] * 8))
    expected_raw[token_offset:token_offset + hidden_bytes] = expected_hidden
    struct.pack_into(
        f"<{_NUM_TOPK}i", expected_raw, metadata_offset, *expected_topk_idx)
    expected_raw[weights_offset:weights_offset + _NUM_TOPK * 4] = \
        expected_weights
    struct.pack_into("<i", expected_raw, src_global_offset, src_token_global_idx)
    struct.pack_into(
        f"<{_NUM_TOPK}i", expected_raw, src_global_offset + 4,
        *([-1] * _NUM_TOPK),
    )
    struct.pack_into("<Q14i", expected_raw, descriptor_offset, *descriptor)
    struct.pack_into(
        "<8I", expected_raw, tail_offset, *([_TAIL_CANARY] * 8))
    assert raw == bytes(expected_raw)
    return expected.coverage_key


def _verify_local_arena(
    rank: int,
    records: torch.Tensor,
    ready: torch.Tensor,
    expected_records: Sequence[ExpectedRecord],
    source_x: torch.Tensor,
    source_topk_idx: torch.Tensor,
    source_topk_weights: torch.Tensor,
    layout: tuple[int, ...],
    physical_capacity: int,
) -> list[tuple[int, ...]]:
    local_expected = sorted(
        (record for record in expected_records if record.egress == rank),
        key=lambda record: record.physical_slot,
    )
    assert [record.physical_slot for record in local_expected] == list(
        range(len(local_expected)))
    assert records.dtype == torch.uint8
    assert ready.dtype == torch.int32
    assert records.is_cuda and ready.is_cuda
    assert records.device.index == ready.device.index == rank
    assert records.is_contiguous() and ready.is_contiguous()
    assert tuple(records.shape) == (physical_capacity, layout[4])
    assert tuple(ready.shape) == (physical_capacity,)

    records_cpu = records.cpu()
    ready_cpu = ready.cpu()
    assert torch.equal(
        ready_cpu[:len(local_expected)],
        torch.full(
            (len(local_expected),), _GENERATION,
            dtype=torch.int32, device="cpu"),
    )
    assert torch.equal(
        ready_cpu[len(local_expected):],
        torch.zeros(
            (physical_capacity - len(local_expected),),
            dtype=torch.int32, device="cpu"),
    )
    assert torch.equal(
        records_cpu[len(local_expected):],
        torch.full(
            (physical_capacity - len(local_expected), layout[4]), 0xA5,
            dtype=torch.uint8, device="cpu"),
    )
    return [
        _decode_and_verify_record(
            records_cpu[index], expected, source_x, source_topk_idx,
            source_topk_weights, layout)
        for index, expected in enumerate(local_expected)
    ]


def _verify_bypass_arena(
    records: torch.Tensor,
    ready: torch.Tensor,
    layout: tuple[int, ...],
    physical_capacity: int,
) -> None:
    """A caller-side bypass launches no producer and publishes no generation."""

    assert records.dtype == torch.uint8
    assert ready.dtype == torch.int32
    assert records.is_cuda and ready.is_cuda
    assert records.device == ready.device
    assert records.is_contiguous() and ready.is_contiguous()
    assert tuple(records.shape) == (physical_capacity, layout[4])
    assert tuple(ready.shape) == (physical_capacity,)
    assert torch.equal(
        records.cpu(),
        torch.full(
            (physical_capacity, layout[4]), 0xA5,
            dtype=torch.uint8, device="cpu"),
    )
    assert torch.equal(
        ready.cpu(),
        torch.zeros(
            (physical_capacity,), dtype=torch.int32, device="cpu"),
    )


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
        x, topk_idx, topk_weights, local_destinations = _checked_local_phase(
            "input generation", control_group,
            lambda: _make_local_inputs(rank),
        )
        destinations_by_rank: list[list[list[int]] | None] = [None] * world_size
        dist.all_gather_object(
            destinations_by_rank, local_destinations, group=control_group)
        assert all(value is not None for value in destinations_by_rank)
        oracle, expected_records = _checked_local_phase(
            "CPU physical oracle", control_group,
            lambda: _build_physical_oracle(destinations_by_rank),  # type: ignore[arg-type]
        )

        local_counts = _local_destination_counts(local_destinations, rank)
        gathered_counts = _all_gather_tensor(local_counts, ep_group)
        def verify_gathered_counts() -> None:
            assert gathered_counts.cpu().tolist() == [
                list(row) for row in oracle.logical_candidate.candidate.counts
            ]

        _checked_local_phase(
            "NCCL count gather", control_group, verify_gathered_counts)
        _checked_local_phase(
            "GPU count planner equality", control_group,
            lambda: _verify_gpu_count_plan(gathered_counts, oracle),
        )

        source_x = _all_gather_tensor(x, ep_group).cpu()
        source_topk_idx = _all_gather_tensor(topk_idx, ep_group).cpu()
        source_topk_weights = _all_gather_tensor(topk_weights, ep_group).cpu()
        layout = _checked_local_phase("layout ABI", control_group, _layout)
        arena_bytes = layout[-1]
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
            num_bytes=base_bytes + arena_bytes,
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
        assert buffer.get_logical_domain_size() == (1, _WORLD_SIZE)
        assert buffer.scaleout_rank_idx == 0 and buffer.scaleup_rank_idx == rank

        manifest, fingerprints = _manifest_for_owner(expected_records, rank)
        num_recv_records = oracle.incoming_copies_per_egress[rank]
        assert num_recv_records == _PHYSICAL_CAPACITY
        collective_config = (
            world_size, _HIDDEN, _NUM_TOPK, _PHYSICAL_CAPACITY,
            _GENERATION, _NUM_TOKENS, arena_offset, arena_bytes,
            layout[4], layout[5],
        )
        local_preflight = (
            collective_config,
            manifest.cpu().tolist(),
            fingerprints.cpu().tolist(),
        )
        gathered_preflight = [None] * world_size
        dist.all_gather_object(
            gathered_preflight, local_preflight, group=control_group)
        _checked_local_phase(
            "collective config and physical-slot preflight", control_group,
            lambda: _verify_collective_preflight(
                gathered_preflight, expected_records),  # type: ignore[arg-type]
        )
        dist.monitored_barrier(
            group=control_group,
            timeout=timedelta(seconds=args.timeout),
            wait_all_ranks=True,
        )

        # This method contains the two collective LSA barriers; communication
        # errors must escape directly so mp.spawn can terminate peer ranks.
        records, ready = buffer.runtime._rail_balance_source_shuffle(
            x, topk_idx, topk_weights,
            manifest, fingerprints,
            arena_offset, _PHYSICAL_CAPACITY, _GENERATION,
            _NUM_TOKENS, _PHYSICAL_CAPACITY,
        )
        torch.cuda.synchronize()
        local_coverage = _checked_local_phase(
            "local raw-record verification", control_group,
            lambda: _verify_local_arena(
                rank, records, ready, expected_records,
                source_x, source_topk_idx, source_topk_weights, layout,
                _PHYSICAL_CAPACITY),
        )
        all_coverage: list[list[tuple[int, ...]] | None] = [None] * world_size
        dist.all_gather_object(
            all_coverage, local_coverage, group=control_group)

        def verify_global_coverage() -> None:
            observed = sorted(
                item for rank_items in all_coverage for item in rank_items)  # type: ignore[union-attr]
            expected = sorted(record.coverage_key for record in expected_records)
            assert observed == expected
            assert len(observed) == len(set(observed)) == oracle.moved_copies

        _checked_local_phase(
            "global exact coverage", control_group, verify_global_coverage)

        # Run the same valid producers with one unused physical slot per
        # egress.  Full-capacity readback must leave that slot poisoned at A5,
        # proving no producer wrote outside its statically assigned range.
        headroom_capacity = _PHYSICAL_CAPACITY + 1
        headroom_layout = _checked_local_phase(
            "headroom layout ABI", control_group,
            lambda: _layout(headroom_capacity),
        )
        assert headroom_layout[-1] <= arena_bytes
        headroom_config = (
            world_size, _HIDDEN, _NUM_TOPK, headroom_capacity,
            _GENERATION, _NUM_TOKENS, arena_offset, arena_bytes,
            headroom_layout[4], headroom_layout[5],
        )
        headroom_preflight = (
            headroom_config,
            manifest.cpu().tolist(),
            fingerprints.cpu().tolist(),
        )
        gathered_headroom = [None] * world_size
        dist.all_gather_object(
            gathered_headroom, headroom_preflight, group=control_group)
        _checked_local_phase(
            "headroom collective preflight", control_group,
            lambda: _verify_collective_preflight(
                gathered_headroom, expected_records),  # type: ignore[arg-type]
        )
        headroom_records, headroom_ready = \
            buffer.runtime._rail_balance_source_shuffle(
                x, topk_idx, topk_weights,
                manifest, fingerprints,
                arena_offset, headroom_capacity, _GENERATION,
                _NUM_TOKENS, headroom_capacity,
            )
        torch.cuda.synchronize()
        _checked_local_phase(
            "valid-path unused-record poison verification", control_group,
            lambda: _verify_local_arena(
                rank, headroom_records, headroom_ready, expected_records,
                source_x, source_topk_idx, source_topk_weights,
                headroom_layout, headroom_capacity),
        )

        def verify_source_inputs_unchanged() -> None:
            assert torch.equal(x.cpu(), source_x[rank])
            assert torch.equal(topk_idx.cpu(), source_topk_idx[rank])
            assert torch.equal(topk_weights.cpu(), source_topk_weights[rank])

        _checked_local_phase(
            "source inputs remain immutable", control_group,
            verify_source_inputs_unchanged,
        )
        del records, ready, headroom_records, headroom_ready
        del manifest, fingerprints

        # Capacity rejection is a caller-side atomic gate: no rank may enter
        # the producer with a partial manifest.  Reuse the populated arena,
        # shrink P by one, and prove the no-producer call leaves every payload
        # byte at the launcher's 0xA5 poison value and every ready word at zero.
        bypass_capacity = _PHYSICAL_CAPACITY - 1
        bypass_oracle = _checked_local_phase(
            "whole-plan physical-capacity bypass oracle", control_group,
            lambda: materialize_physical_proxy_slots(
                oracle.logical_candidate,
                global_policy_budget=oracle.logical_candidate.moved_copies,
                physical_capacity_per_egress=bypass_capacity,
            ),
        )
        assert not bypass_oracle.enabled
        assert bypass_oracle.bypass_reason == "physical_egress_capacity"
        assert bypass_oracle.assignments == ()
        assert bypass_oracle.moved_copies == 0
        assert bypass_oracle.incoming_copies_per_egress == (
            _PHYSICAL_CAPACITY,) * _WORLD_SIZE
        committed = bypass_oracle.committed_logical
        candidate = oracle.logical_candidate.candidate
        assert not committed.enabled
        assert committed.committed_quota == candidate.counts
        assert committed.committed_keep_count == candidate.counts
        assert committed.committed_segments == ()
        assert committed.assignments == ()
        assert committed.moved_copies == 0
        assert committed.bypass_reason == "physical_egress_capacity"

        bypass_layout = _checked_local_phase(
            "bypass layout ABI", control_group,
            lambda: _layout(bypass_capacity),
        )
        assert bypass_layout[-1] <= arena_bytes
        empty_manifest = torch.empty(
            (0, 7), dtype=torch.int32, device=torch.device("cuda", rank))
        empty_fingerprints = torch.empty(
            (0,), dtype=torch.int64, device=torch.device("cuda", rank))
        assert empty_manifest.numel() == empty_fingerprints.numel() == 0
        bypass_config = (
            world_size, _HIDDEN, _NUM_TOPK, bypass_capacity,
            _GENERATION + 1, _NUM_TOKENS, arena_offset, arena_bytes,
            bypass_layout[4], bypass_layout[5],
        )
        bypass_preflight = (
            bypass_config,
            empty_manifest.cpu().tolist(),
            empty_fingerprints.cpu().tolist(),
        )
        gathered_bypass = [None] * world_size
        dist.all_gather_object(
            gathered_bypass, bypass_preflight, group=control_group)
        _checked_local_phase(
            "bypass collective empty-manifest preflight", control_group,
            lambda: _verify_empty_collective_preflight(
                gathered_bypass),  # type: ignore[arg-type]
        )
        bypass_records, bypass_ready = \
            buffer.runtime._rail_balance_source_shuffle(
                x, topk_idx, topk_weights,
                empty_manifest, empty_fingerprints,
                arena_offset, bypass_capacity, _GENERATION + 1,
                _NUM_TOKENS, bypass_capacity,
            )
        torch.cuda.synchronize()
        _checked_local_phase(
            "GPU whole-plan bypass has no publication", control_group,
            lambda: _verify_bypass_arena(
                bypass_records, bypass_ready, bypass_layout,
                bypass_capacity),
        )
        _checked_local_phase(
            "source inputs remain immutable after bypass", control_group,
            verify_source_inputs_unchanged,
        )
        del bypass_records, bypass_ready, empty_manifest, empty_fingerprints
        torch.cuda.synchronize()
        dist.monitored_barrier(
            group=control_group,
            timeout=timedelta(seconds=args.timeout),
            wait_all_ranks=True,
        )
        clean_shutdown = True
    finally:
        # ElasticBuffer.destroy() is collective.  Only call it after every rank
        # has passed the final control-group checkpoint.
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
                    f"PASS C040 8x1 source shuffle: {oracle.moved_copies} "
                    f"moved records from all owners, "
                    f"incoming={oracle.incoming_copies_per_egress}, "
                    "exact-capacity and GPU bypass verified",
                    flush=True,
                )
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C040 8x1 rail-balance source-shuffle functional test")
    if not __debug__:
        parser.error("C040 correctness checks require Python assertions")
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--master-port", type=int, default=29641)
    args = parser.parse_args()
    if args.num_processes != _WORLD_SIZE:
        parser.error("C040 8x1 requires exactly 8 processes")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("C040 8x1 requires at least 8 visible CUDA devices")

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
