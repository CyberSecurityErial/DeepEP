"""Direct CPU tests for the frozen C080 Hybrid rail-balance contract."""

from __future__ import annotations

import random
import re
from dataclasses import fields, replace

from rail_balance_hybrid_reference import (
    HybridTransferSegment,
    build_hybrid_rail_schedule,
    build_hybrid_rail_schedule_from_destinations,
    enumerate_resolved_copies,
    map_topk_experts_to_destinations,
    resolve_hybrid_copy,
    validate_hybrid_rail_schedule,
)
from rail_balance_reference import build_count_plan


_C061_ROUTES = (
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


def _assert_raises(error_type, pattern, function):
    try:
        function()
    except error_type as error:
        assert re.search(pattern, str(error)), str(error)
    else:
        names = (
            "/".join(item.__name__ for item in error_type)
            if isinstance(error_type, tuple) else error_type.__name__
        )
        raise AssertionError(f"expected {names}")


def _build(routes, *, destinations, channels, max_tokens, **kwargs):
    return build_hybrid_rail_schedule_from_destinations(
        routes,
        num_destinations=destinations,
        num_channels=channels,
        num_max_tokens_per_rank=max_tokens,
        **kwargs,
    )


def _assert_resolver_invariants(routes, schedule):
    validate_hybrid_rail_schedule(schedule)
    assert schedule.enabled

    # Keep the Hybrid destination-bucketed ABI locked to the canonical C030
    # minimum-move plan while checking egress_begin independently below.
    canonical = build_count_plan(
        schedule.count, remainder_seed=schedule.remainder_seed)
    assert schedule.count == canonical.counts
    assert schedule.quota == canonical.quota
    assert schedule.keep_count == canonical.keep_count
    assert [
        (destination, segment.owner, segment.egress,
         segment.owner_begin, segment.count)
        for destination, destination_segments in enumerate(schedule.segments)
        for segment in destination_segments
    ] == [
        (segment.destination, segment.owner, segment.egress,
         segment.owner_begin, segment.count)
        for segment in canonical.segments
    ]

    records = enumerate_resolved_copies(routes, schedule)
    assert len(records) == sum(sum(row) for row in schedule.count)
    assert len({record.copy.key for record in records}) == len(records)
    assert sum(record.resolution.moved for record in records) == schedule.moved_copies

    # Owner ordinals are dense in the frozen channel-major source order.
    for owner in range(schedule.num_rails):
        for destination in range(schedule.num_destinations):
            ordinals = sorted(
                record.copy.owner_ordinal
                for record in records
                if record.copy.owner == owner
                and record.copy.destination == destination
            )
            assert ordinals == list(range(schedule.count[owner][destination]))

    # One moved-only physical namespace per egress, dense and collision-free.
    for egress in range(schedule.num_rails):
        proxy_slots = sorted(
            record.resolution.proxy_slot
            for record in records
            if record.resolution.moved
            and record.resolution.egress == egress
        )
        assert proxy_slots == list(range(schedule.proxy_required[egress]))

    proxy_keys = [
        (record.resolution.egress, record.resolution.proxy_slot)
        for record in records
        if record.resolution.moved
    ]
    assert len(proxy_keys) == len(set(proxy_keys))

    # Retained [0, r) and moved [r, r + m) slots form one dense namespace for
    # every (egress, channel, destination) group.
    remote_keys = []
    for egress in range(schedule.num_rails):
        for channel in range(schedule.num_channels):
            for destination in range(schedule.num_destinations):
                slots = sorted(
                    record.resolution.remote_slot
                    for record in records
                    if record.resolution.egress == egress
                    and record.resolution.channel == channel
                    and record.resolution.destination == destination
                )
                expected = (
                    schedule.retained[egress][channel][destination]
                    + schedule.moved[egress][channel][destination]
                )
                assert slots == list(range(expected))
                remote_keys.extend(
                    (egress, channel, destination, slot) for slot in slots)
    assert len(remote_keys) == len(set(remote_keys))

    # Segment egress ordinals and moved-channel prefixes describe exactly the
    # same dense incoming stream.
    for egress in range(schedule.num_rails):
        for destination in range(schedule.num_destinations):
            incoming = sorted(
                record.resolution.incoming_ordinal
                for record in records
                if record.resolution.moved
                and record.resolution.egress == egress
                and record.resolution.destination == destination
            )
            expected = (
                schedule.quota[egress][destination]
                - schedule.keep_count[egress][destination]
            )
            assert incoming == list(range(expected))

    # Count/quota conservation and the exact move identity hold per dst.
    for destination in range(schedule.num_destinations):
        counts = [
            schedule.count[owner][destination]
            for owner in range(schedule.num_rails)
        ]
        quotas = [
            schedule.quota[owner][destination]
            for owner in range(schedule.num_rails)
        ]
        assert sum(counts) == sum(quotas)
        assert max(quotas) - min(quotas) <= 1
        moved = sum(max(count - quota, 0) for count, quota in zip(counts, quotas))
        l1_half = sum(abs(count - quota) for count, quota in zip(counts, quotas)) // 2
        assert moved == l1_half
        assert moved == sum(
            segment.count
            for segment in schedule.segments[destination]
        )


def test_c061_channel_major_golden_and_capacity():
    schedule = _build(
        _C061_ROUTES,
        destinations=3,
        channels=2,
        max_tokens=10,
        remainder_seed=62,
        proxy_capacity_per_egress=3,
    )
    assert schedule.channel_capacity == 5
    assert schedule.remainder_seed == 0
    assert schedule.channel_count == (
        ((5, 1, 3), (4, 1, 3)),
        ((1, 5, 3), (1, 4, 2)),
    )
    assert schedule.owner_channel_prefix == (
        ((0, 0, 0), (5, 1, 3)),
        ((0, 0, 0), (1, 5, 3)),
    )
    assert schedule.count == ((9, 2, 6), (2, 9, 5))
    assert schedule.quota == ((6, 5, 6), (5, 6, 5))
    assert schedule.keep_count == ((6, 2, 6), (2, 6, 5))
    assert schedule.segments == (
        (HybridTransferSegment(0, 1, 6, 3, 0),),
        (HybridTransferSegment(1, 0, 6, 3, 0),),
        (),
    )
    assert schedule.num_segments == (1, 1, 0)
    assert schedule.retained == (
        ((5, 1, 3), (1, 1, 3)),
        ((1, 5, 3), (1, 1, 2)),
    )
    assert schedule.moved == (
        ((0, 3, 0), (0, 0, 0)),
        ((3, 0, 0), (0, 0, 0)),
    )
    assert schedule.moved_channel_prefix == (
        ((0, 0, 0), (0, 3, 3), (0, 0, 0)),
        ((0, 3, 3), (0, 0, 0), (0, 0, 0)),
    )
    assert schedule.group_prefix == (
        ((0, 0, 3), (3, 3, 3)),
        ((0, 3, 3), (3, 3, 3)),
    )
    assert schedule.proxy_required == (3, 3)
    assert schedule.moved_copies == 6

    records = enumerate_resolved_copies(_C061_ROUTES, schedule)
    assert len(records) == 33
    moved = [
        (
            record.copy.owner,
            record.copy.token,
            record.copy.destination,
            record.resolution.egress,
            record.resolution.channel,
            record.resolution.remote_slot,
            record.resolution.proxy_slot,
        )
        for record in records
        if record.resolution.moved
    ]
    assert moved == [
        (0, 3, 0, 1, 0, 1, 0),
        (0, 5, 0, 1, 0, 2, 1),
        (0, 7, 0, 1, 0, 3, 2),
        (1, 3, 1, 0, 0, 1, 0),
        (1, 5, 1, 0, 0, 2, 1),
        (1, 7, 1, 0, 0, 3, 2),
    ]
    _assert_resolver_invariants(_C061_ROUTES, schedule)

    failed = _build(
        _C061_ROUTES,
        destinations=3,
        channels=2,
        max_tokens=10,
        remainder_seed=62,
        proxy_capacity_per_egress=2,
    )
    assert not failed.enabled
    assert failed.failure_reason == "proxy_capacity_per_egress_exceeded"
    assert failed.proxy_required == (3, 3)
    assert failed.moved_copies == 6
    assert failed.segments == schedule.segments
    _assert_raises(
        RuntimeError,
        "disabled schedule",
        lambda: resolve_hybrid_copy(
            failed,
            owner=0,
            source_channel=1,
            destination=0,
            channel_local_ordinal=1,
        ),
    )


def test_zero_tokens_and_no_manifest_in_schedule():
    routes = ((), (), (), ())
    schedule = _build(
        routes,
        destinations=5,
        channels=3,
        max_tokens=7,
        proxy_capacity_per_egress=0,
    )
    assert schedule.enabled
    assert schedule.channel_capacity == 3
    assert schedule.count == ((0, 0, 0, 0, 0),) * 4
    assert schedule.quota == schedule.count
    assert schedule.keep_count == schedule.count
    assert schedule.segments == ((),) * 5
    assert schedule.num_segments == (0,) * 5
    assert schedule.proxy_required == (0,) * 4
    assert schedule.moved_copies == 0
    assert enumerate_resolved_copies(routes, schedule) == ()
    field_names = {field.name for field in fields(schedule)}
    assert "copies" not in field_names
    assert "assignments" not in field_names
    assert "manifest" not in field_names
    _assert_resolver_invariants(routes, schedule)


def test_t_less_than_channels_and_empty_owners():
    routes = (
        ((0, 1), (0,), (1,)),
        (),
        ((1,),),
    )
    schedule = _build(
        routes,
        destinations=2,
        channels=8,
        max_tokens=3,
        remainder_seed=4,
    )
    assert schedule.channel_capacity == 1
    assert schedule.num_tokens_per_owner == (3, 0, 1)
    assert all(
        value <= 1
        for owner in schedule.channel_count
        for channel in owner
        for value in channel
    )
    _assert_resolver_invariants(routes, schedule)


def test_nondivisible_t_and_channel_prefixes():
    routes = (
        tuple((0, 1) if token % 2 else (0,) for token in range(7)),
        tuple((1,) for _token in range(7)),
        tuple((0,) if token in (1, 4, 6) else (1,) for token in range(7)),
    )
    schedule = _build(
        routes,
        destinations=2,
        channels=3,
        max_tokens=7,
        remainder_seed=9,
    )
    assert schedule.channel_capacity == 3
    for owner in range(schedule.num_rails):
        for destination in range(schedule.num_destinations):
            prefix = 0
            for channel in range(schedule.num_channels):
                assert schedule.owner_channel_prefix[owner][channel][destination] == prefix
                prefix += schedule.channel_count[owner][channel][destination]
    _assert_resolver_invariants(routes, schedule)


def test_duplicate_local_exclusion_and_lane_permutation():
    routes = (
        (
            (2, 0, 0, 1, 2, 2),
            (3, 3, 2, 2, 1, 1),
            (2, 2, 2, 2),
        ),
        (
            (0, 2, 0, 2),
            (3, 2, 3, 2),
        ),
    )
    permuted = tuple(
        tuple(tuple(reversed(lanes)) for lanes in owner) for owner in routes
    )
    schedule = _build(
        routes,
        destinations=4,
        channels=2,
        max_tokens=3,
        local_destination=2,
        remainder_seed=3,
    )
    permuted_schedule = _build(
        permuted,
        destinations=4,
        channels=2,
        max_tokens=3,
        local_destination=2,
        remainder_seed=3,
    )
    assert schedule == permuted_schedule
    assert schedule.count == ((1, 2, 0, 1), (1, 0, 0, 1))
    records = enumerate_resolved_copies(routes, schedule)
    assert len(records) == 6
    assert not any(record.copy.destination == 2 for record in records)
    assert len({record.copy.key for record in records}) == 6
    _assert_resolver_invariants(routes, schedule)


def test_egress_begin_accumulates_across_multiple_owner_segments():
    routes = (
        ((0,), (0,), (0,)),
        ((0,), (0,), (0,)),
        (),
    )
    schedule = _build(
        routes,
        destinations=1,
        channels=2,
        max_tokens=3,
    )
    assert schedule.count == ((3,), (3,), (0,))
    assert schedule.quota == ((2,), (2,), (2,))
    assert schedule.segments == (
        (
            HybridTransferSegment(0, 2, 2, 1, 0),
            HybridTransferSegment(1, 2, 2, 1, 1),
        ),
    )
    _assert_resolver_invariants(routes, schedule)


def test_one_owner_segment_can_split_across_egresses():
    routes = (
        ((0,),) * 6,
        (),
        (),
    )
    schedule = _build(
        routes,
        destinations=1,
        channels=2,
        max_tokens=6,
    )
    assert schedule.count == ((6,), (0,), (0,))
    assert schedule.quota == ((2,), (2,), (2,))
    assert schedule.segments == (
        (
            HybridTransferSegment(0, 1, 2, 2, 0),
            HybridTransferSegment(0, 2, 4, 2, 0),
        ),
    )
    _assert_resolver_invariants(routes, schedule)


def test_balanced_plan_is_a_true_zero_move_plan():
    routes = tuple(tuple((0, 1) for _ in range(5)) for _ in range(4))
    schedule = _build(
        routes,
        destinations=2,
        channels=3,
        max_tokens=5,
        proxy_capacity_per_egress=0,
    )
    assert schedule.enabled
    assert schedule.count == schedule.quota == schedule.keep_count
    assert schedule.segments == ((), ())
    assert schedule.moved_copies == 0
    assert schedule.proxy_required == (0, 0, 0, 0)
    _assert_resolver_invariants(routes, schedule)


def test_randomized_schedule_and_resolver_invariants():
    for seed in range(120):
        randomizer = random.Random(0xC080 + seed)
        rails = randomizer.randint(1, 5)
        destinations = randomizer.randint(1, 6)
        max_tokens = randomizer.randint(1, 15)
        channels = randomizer.randint(1, 9)
        local_destination = (
            randomizer.randrange(destinations)
            if randomizer.random() < 0.35 else None
        )
        routes = []
        for _owner in range(rails):
            tokens = []
            for _token in range(randomizer.randint(0, max_tokens)):
                lanes = []
                for _lane in range(randomizer.randint(0, 8)):
                    if lanes and randomizer.random() < 0.35:
                        lanes.append(randomizer.choice(lanes))
                    else:
                        lanes.append(randomizer.randrange(destinations))
                tokens.append(tuple(lanes))
            routes.append(tuple(tokens))
        routes = tuple(routes)
        kwargs = dict(
            destinations=destinations,
            channels=channels,
            max_tokens=max_tokens,
            local_destination=local_destination,
            remainder_seed=randomizer.randint(0, 1000),
        )
        schedule = _build(routes, **kwargs)
        _assert_resolver_invariants(routes, schedule)

        exact_capacity = max(schedule.proxy_required)
        capacity_schedule = _build(
            routes,
            proxy_capacity_per_egress=exact_capacity,
            **kwargs,
        )
        assert capacity_schedule.enabled
        assert capacity_schedule.proxy_required == schedule.proxy_required
        if exact_capacity:
            failed = _build(
                routes,
                proxy_capacity_per_egress=exact_capacity - 1,
                **kwargs,
            )
            assert not failed.enabled
            assert failed.failure_reason == "proxy_capacity_per_egress_exceeded"
            assert failed.proxy_required == schedule.proxy_required


def test_strict_expert_mapping_and_force_entry_contract():
    topk_idx = (
        (
            (0, 1, 8, 9),
            (8, 16, 17, 18),
            (16, 17, 18, 19),
        ),
        (
            (2, 3, 10, 11),
            (8, 9, 10, 11),
        ),
    )
    mapped = map_topk_experts_to_destinations(
        topk_idx,
        num_topk=4,
        num_experts=24,
        num_scaleout_ranks=3,
        local_scaleout_rank=0,
    )
    assert mapped == (
        ((1,), (1, 2), (2,)),
        ((1,), (1,)),
    )
    schedule = build_hybrid_rail_schedule(
        topk_idx,
        num_topk=4,
        num_experts=24,
        num_scaleout_ranks=3,
        local_scaleout_rank=0,
        num_channels=2,
        num_max_tokens_per_rank=3,
        proxy_capacity_per_egress=4,
    )
    assert schedule.local_destination == 0
    assert all(row[0] == 0 for row in schedule.count)
    _assert_resolver_invariants(mapped, schedule)

    common = dict(
        num_topk=2,
        num_experts=8,
        num_scaleout_ranks=2,
        local_scaleout_rank=0,
    )
    invalid = (
        ((((0,),),), "exactly num_topk"),
        ((((-1, 0),),), "at least 0"),
        ((((0, 8),),), "outside"),
        ((((0, 0),),), "distinct"),
        ((((False, 1),),), "integer"),
    )
    for value, pattern in invalid:
        _assert_raises(
            ValueError,
            pattern,
            lambda value=value: map_topk_experts_to_destinations(
                value, **common),
        )
    _assert_raises(
        ValueError,
        "32 limit",
        lambda: map_topk_experts_to_destinations(
            ((tuple(range(33)),),),
            num_topk=33,
            num_experts=64,
            num_scaleout_ranks=2,
            local_scaleout_rank=0,
        ),
    )
    _assert_raises(
        ValueError,
        "divisible",
        lambda: map_topk_experts_to_destinations(
            (((0, 1),),),
            num_topk=2,
            num_experts=7,
            num_scaleout_ranks=2,
            local_scaleout_rank=0,
        ),
    )
    _assert_raises(
        ValueError,
        r"scaleout_ranks \* local_rails",
        lambda: map_topk_experts_to_destinations(
            (((0, 1),), ((2, 3),)),
            num_topk=2,
            num_experts=6,
            num_scaleout_ranks=2,
            local_scaleout_rank=0,
        ),
    )
    assert map_topk_experts_to_destinations(
        (((2, 4),),),
        num_topk=2,
        num_experts=6,
        num_scaleout_ranks=3,
        local_scaleout_rank=0,
    ) == (((1, 2),),)
    _assert_raises(
        ValueError,
        "at least 1",
        lambda: build_hybrid_rail_schedule(
            (((0, 1),),),
            num_topk=2,
            num_experts=8,
            num_scaleout_ranks=2,
            local_scaleout_rank=0,
            num_channels=1,
            num_max_tokens_per_rank=1,
            proxy_capacity_per_egress=0,
        ),
    )


def test_input_validation_and_corruption_detection():
    class IntSubclass(int):
        pass

    valid = (((0,),), ((0,),))
    common = dict(destinations=1, channels=1, max_tokens=1)
    _assert_raises(
        ValueError,
        "at least one owner",
        lambda: _build((), **common),
    )
    _assert_raises(
        ValueError,
        "must be an integer",
        lambda: _build(valid, destinations=1, channels=True, max_tokens=1),
    )
    _assert_raises(
        ValueError,
        "outside",
        lambda: _build((((1,),),), **common),
    )
    _assert_raises(
        ValueError,
        "exceeding",
        lambda: _build((((0,), (0,)),), **common),
    )
    _assert_raises(
        ValueError,
        "must be a sequence",
        lambda: _build(((0,),), **common),
    )
    _assert_raises(
        ValueError,
        "masked destinations",
        lambda: _build((((-1,),),), **common),
    )
    _assert_raises(
        ValueError,
        "32-rail limit",
        lambda: _build(tuple(() for _ in range(33)), **common),
    )
    _assert_raises(
        ValueError,
        "1024 limit",
        lambda: _build(valid, destinations=1, channels=1025, max_tokens=1),
    )
    _assert_raises(
        ValueError,
        "32 limit",
        lambda: _build(valid, destinations=33, channels=1, max_tokens=1),
    )
    _assert_raises(
        ValueError,
        "signed int32 prefix ABI",
        lambda: _build(
            tuple(() for _ in range(32)),
            destinations=32,
            channels=1,
            max_tokens=1 << 21,
        ),
    )

    assert _build(
        tuple(() for _ in range(32)),
        destinations=1,
        channels=1,
        max_tokens=1,
    ).num_rails == 32
    assert _build(
        ((),), destinations=1, channels=1024, max_tokens=1,
    ).num_channels == 1024
    assert _build(
        ((),), destinations=32, channels=1, max_tokens=1,
    ).num_destinations == 32
    assert _build(
        valid,
        **common,
        remainder_seed=1 << 40,
    ).remainder_seed == 0
    _assert_raises(
        ValueError,
        "signed int64 host ABI",
        lambda: _build(valid, **common, remainder_seed=1 << 70),
    )
    _assert_raises(
        ValueError,
        "must be an integer",
        lambda: _build(valid, **common, remainder_seed=IntSubclass(0)),
    )

    schedule = _build(valid, **common)
    _assert_raises(
        ValueError,
        "outside the source group",
        lambda: resolve_hybrid_copy(
            schedule,
            owner=0,
            source_channel=0,
            destination=0,
            channel_local_ordinal=1,
        ),
    )
    corrupt_prefix = list(list(row) for row in schedule.group_prefix[0])
    corrupt_prefix[0][0] += 1
    corrupt = replace(
        schedule,
        group_prefix=(
            tuple(tuple(row) for row in corrupt_prefix),
            schedule.group_prefix[1],
        ),
    )
    _assert_raises(
        ValueError,
        "group_prefix",
        lambda: validate_hybrid_rail_schedule(corrupt),
    )
    _assert_raises(
        ValueError,
        "signed int32",
        lambda: validate_hybrid_rail_schedule(replace(
            schedule,
            proxy_required=((1 << 31), schedule.proxy_required[1]),
        )),
    )


if __name__ == "__main__":
    tests = sorted(
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} rail-balance Hybrid CPU reference tests")
