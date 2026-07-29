import itertools
import random
import re
from dataclasses import replace

from rail_balance_reference import (
    DestinationCopy,
    advance_contiguous_ready_tail,
    build_count_plan,
    build_destination_copies,
    counts_from_copies,
    materialize_destination_segmented_egress_slots,
    materialize_full_egress_slots,
    materialize_physical_proxy_slots,
    materialize_static_slots,
    simulate_contiguous_ready_publication,
    validate_count_plan,
    validate_static_slot_plan,
)


def _column(matrix, destination=0):
    return [row[destination] for row in matrix]


def _minimum_moved_by_quota_enumeration(counts):
    num_rails = len(counts)
    total = sum(counts)
    base, remainder = divmod(total, num_rails)
    best = None
    for winners in itertools.combinations(range(num_rails), remainder):
        winner_set = set(winners)
        quota = [base + int(rail in winner_set) for rail in range(num_rails)]
        moved = sum(max(count - target, 0) for count, target in zip(counts, quota))
        best = moved if best is None else min(best, moved)
    return best


def _copies_from_specs(specs, num_channels):
    ordinals = {}
    copies = []
    for owner, token, destination in sorted(specs):
        group = owner, destination
        ordinal = ordinals.get(group, 0)
        copies.append(DestinationCopy(
            owner=owner,
            token=token,
            destination=destination,
            source_channel=token % num_channels,
            owner_ordinal=ordinal,
        ))
        ordinals[group] = ordinal + 1
    return tuple(copies)


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


def _single_destination_logical_plan(counts, generation=0):
    copies = _copies_from_specs([
        (owner, token, 0)
        for owner, count in enumerate(counts)
        for token in range(count)
    ], num_channels=1)
    candidate = build_count_plan([[count] for count in counts])
    logical = materialize_static_slots(
        copies,
        candidate,
        num_channels=1,
        channel_capacity=max(sum(counts), 1),
        generation=generation,
    )
    assert logical.enabled
    return logical


def _assert_atomic_physical_bypass(result, logical, reason):
    assert not result.enabled
    assert result.bypass_reason == reason
    assert result.assignments == ()
    assert result.moved_copies == 0
    assert result.logical_candidate == logical
    committed = result.committed_logical
    assert not committed.enabled
    assert committed.bypass_reason == reason
    assert committed.assignments == ()
    assert committed.moved_copies == 0
    assert committed.committed_quota == logical.candidate.counts
    assert committed.committed_keep_count == logical.candidate.counts
    assert committed.committed_segments == ()


def test_full_egress_pack_covers_retained_and_moved_in_compact_prefixes():
    logical = _single_destination_logical_plan(
        [32, 24, 16, 8], generation=61)
    full = materialize_full_egress_slots(
        logical, physical_capacity_per_egress=20)
    assert full.enabled and full.bypass_reason is None
    assert full.records_per_egress == (20, 20, 20, 20)
    assert len(full.assignments) == 80
    assert sum(item.assignment.moved for item in full.assignments) == 16
    assert sum(not item.assignment.moved for item in full.assignments) == 64
    assert {item.assignment.copy.key for item in full.assignments} == {
        assignment.copy.key for assignment in logical.assignments
    }
    for egress in range(4):
        assert [
            item.physical_slot for item in full.assignments
            if item.egress == egress
        ] == list(range(20))


def test_full_egress_capacity_failure_is_atomic_and_strict():
    logical = _single_destination_logical_plan(
        [32, 24, 16, 8], generation=61)
    rejected = materialize_full_egress_slots(
        logical, physical_capacity_per_egress=19)
    assert not rejected.enabled
    assert rejected.bypass_reason == "full_egress_physical_capacity"
    assert rejected.assignments == ()
    assert rejected.records_per_egress == (0, 0, 0, 0)
    assert rejected.logical_candidate == logical
    _assert_raises(
        ValueError,
        "physical_capacity_per_egress must be an integer",
        lambda: materialize_full_egress_slots(
            logical, physical_capacity_per_egress=True),
    )


def _c061_segmented_logical_plan():
    counts = ((9, 2, 6), (2, 9, 5))
    copies = _copies_from_specs([
        (owner, token, destination)
        for owner, row in enumerate(counts)
        for destination, count in enumerate(row)
        for token in range(count)
    ], num_channels=2)
    candidate = build_count_plan(counts, remainder_seed=62)
    assert candidate.quota == ((6, 5, 6), (5, 6, 5))
    assert candidate.keep_count == ((6, 2, 6), (2, 6, 5))
    assert tuple(
        (item.destination, item.owner, item.egress,
         item.owner_begin, item.count)
        for item in candidate.segments
    ) == ((0, 0, 1, 6, 3), (1, 1, 0, 6, 3))
    assert candidate.moved_copies == 6
    logical = materialize_static_slots(
        copies,
        candidate,
        num_channels=2,
        channel_capacity=6,
        generation=62,
    )
    assert logical.enabled
    return logical


def test_destination_segmented_pack_discriminates_per_destination_balance():
    logical = _c061_segmented_logical_plan()
    # An aggregate-only policy sees an already balanced 17/16 split and would
    # move nothing; the canonical per-destination plan must still move six.
    assert tuple(map(sum, logical.candidate.counts)) == (17, 16)
    segmented = materialize_destination_segmented_egress_slots(
        logical, capacity_per_destination=6)
    assert segmented.enabled and segmented.bypass_reason is None
    assert segmented.records_per_egress_destination == (
        (6, 5, 6), (5, 6, 5))
    assert segmented.physical_capacity_per_egress == 18
    assert len(segmented.assignments) == 33
    observed = [set(), set()]
    for item in segmented.assignments:
        destination = item.assignment.destination
        assert item.physical_slot // 6 == destination
        assert item.physical_slot % 6 < \
            segmented.records_per_egress_destination[
                item.egress][destination]
        observed[item.egress].add(item.physical_slot)
    assert set(range(18)) - observed[0] == {11}
    assert set(range(18)) - observed[1] == {5, 17}


def test_destination_segmented_capacity_failure_is_atomic():
    logical = _c061_segmented_logical_plan()
    rejected = materialize_destination_segmented_egress_slots(
        logical, capacity_per_destination=5)
    assert not rejected.enabled
    assert rejected.bypass_reason == \
        "destination_segment_physical_capacity"
    assert rejected.assignments == ()
    assert rejected.records_per_egress_destination == ((0, 0, 0),) * 2
    assert rejected.logical_candidate == logical
    _assert_raises(
        ValueError,
        "capacity_per_destination must be an integer",
        lambda: materialize_destination_segmented_egress_slots(
            logical, capacity_per_destination=False),
    )


def test_normalization_deduplicates_masks_excludes_local_and_ignores_lane_order():
    routes = [
        [[0, 2, 1, 2, -1], [2, 1, 2]],
        [[1, 1, 0], [-1, -1, -1]],
    ]
    permuted = [[list(reversed(token)) for token in owner] for owner in routes]
    copies = build_destination_copies(
        routes, num_destinations=3, num_channels=3, local_destination=0)
    assert copies == build_destination_copies(
        permuted, num_destinations=3, num_channels=3, local_destination=0)
    assert [copy.key for copy in copies] == [
        (0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 1, 2), (1, 0, 1),
    ]
    assert counts_from_copies(
        copies, num_rails=2, num_destinations=3, num_channels=3) == (
            (0, 2, 2),
            (0, 1, 0),
        )


def test_count_aware_remainder_avoids_unnecessary_moves_and_preserves_identity():
    skewed = build_count_plan([[4], [0], [0]], remainder_seed=1)
    assert _column(skewed.quota) == [2, 1, 1]
    assert skewed.moved_copies == 2

    balanced = build_count_plan([[1], [2]], remainder_seed=0)
    assert balanced.quota == balanced.counts
    assert balanced.moved_copies == 0


def test_exact_quota_is_move_minimal_by_exhaustive_enumeration():
    for remainder_seed in range(6):
        for num_rails in range(1, 6):
            for counts in itertools.product(range(4), repeat=num_rails):
                plan = build_count_plan([[count] for count in counts],
                                        remainder_seed=remainder_seed)
                validate_count_plan(plan)
                assert plan.moved_copies == _minimum_moved_by_quota_enumeration(counts)
                if max(counts) - min(counts) <= 1:
                    assert _column(plan.quota) == list(counts)
                    assert plan.moved_copies == 0


def test_segments_are_stable_contiguous_suffixes():
    cases = [
        [[0], [0], [0], [0]],
        [[1], [0], [0], [0]],
        [[4], [1], [1], [1]],
        [[3], [2], [2], [0], [0]],
        [[9], [0], [0], [0]],
        [[6], [5], [4], [3], [2], [1], [0], [0]],
    ]
    for counts in cases:
        first = build_count_plan(counts, remainder_seed=3)
        second = build_count_plan(counts, remainder_seed=3)
        assert first == second
        assert sum(segment.count for segment in first.segments) == first.moved_copies
        assert len(first.segments) <= len(counts) - 1
        validate_count_plan(first)


def test_per_destination_rotation_is_canonical_and_zero_rows_do_not_change_shape():
    counts = [[6, 6, 6, 6], [0, 0, 0, 0],
              [0, 0, 0, 0], [0, 0, 0, 0]]
    plan = build_count_plan(counts, remainder_seed=2)
    assert plan == build_count_plan(counts, remainder_seed=2)
    assert _column(plan.quota, 0) == [2, 1, 2, 1]
    assert _column(plan.quota, 1) == [2, 1, 1, 2]
    assert _column(plan.quota, 2) == [2, 2, 1, 1]
    validate_count_plan(plan)


def test_static_slots_share_retained_and_proxy_namespace_and_restripe_channels():
    # Owner 1 retains token 0 and 2 in channel 0, filling it.  The incoming
    # proxy copy must use channel 1 even though its deterministic search starts
    # at the full channel.
    copies = _copies_from_specs([
        (0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0),
        (1, 0, 0), (1, 2, 0),
    ], num_channels=2)
    plan = build_count_plan([[4], [2]])
    slots = materialize_static_slots(
        copies,
        plan,
        num_channels=2,
        channel_capacity=2,
        proxy_payload_capacity=1,
        generation=7,
    )
    assert slots.enabled
    assert slots.moved_copies == 1
    moved = [assignment for assignment in slots.assignments if assignment.moved]
    assert len(moved) == 1
    assert (moved[0].egress, moved[0].channel, moved[0].slot) == (1, 1, 0)
    assert len({assignment.slot_key for assignment in slots.assignments}) == len(slots.assignments)


def test_proxy_segment_stripes_across_least_occupied_channels():
    copies = _copies_from_specs(
        [(0, token, 0) for token in range(8)], num_channels=4)
    plan = build_count_plan([[8], [0]])
    slots = materialize_static_slots(
        copies,
        plan,
        num_channels=4,
        channel_capacity=8,
        proxy_payload_capacity=4,
    )
    assert slots.enabled
    moved = [assignment for assignment in slots.assignments if assignment.moved]
    assert [(assignment.channel, assignment.slot) for assignment in moved] == [
        (0, 0), (1, 0), (2, 0), (3, 0),
    ]


def test_proxy_slots_start_after_retained_slots_in_same_channel():
    copies = _copies_from_specs([
        (0, 0, 0), (0, 1, 0), (0, 2, 0),
        (1, 0, 0),
    ], num_channels=1)
    plan = build_count_plan([[3], [1]])
    slots = materialize_static_slots(
        copies, plan, num_channels=1, channel_capacity=2,
        proxy_payload_capacity=1, generation=4)
    assert slots.enabled
    egress_one = [assignment for assignment in slots.assignments if assignment.egress == 1]
    assert [(assignment.moved, assignment.slot) for assignment in egress_one] == [
        (False, 0), (True, 1),
    ]


def test_two_owners_with_same_token_index_get_unique_proxy_slots():
    copies = _copies_from_specs([
        (0, 0, 0), (0, 1, 0), (0, 2, 0),
        (1, 0, 0), (1, 1, 0), (1, 2, 0),
    ], num_channels=2)
    plan = build_count_plan([[3], [3], [0]])
    slots = materialize_static_slots(
        copies, plan, num_channels=2, channel_capacity=2,
        proxy_payload_capacity=2, generation=9)
    assert slots.enabled
    moved = [assignment for assignment in slots.assignments if assignment.moved]
    assert len(moved) == 2
    assert {assignment.copy.owner for assignment in moved} == {0, 1}
    assert len({assignment.slot_key for assignment in moved}) == 2


def test_physical_global_budget_can_fail_while_every_egress_fits():
    # Two moved copies land on different egresses.  P=1 is sufficient for each
    # physical arena, but B=1 is insufficient for the node-wide move policy.
    logical = _single_destination_logical_plan([4, 0, 0], generation=21)
    assert logical.moved_copies == 2
    physical = materialize_physical_proxy_slots(
        logical,
        global_policy_budget=1,
        physical_capacity_per_egress=1,
    )
    assert physical.incoming_copies_per_egress == (0, 1, 1)
    _assert_atomic_physical_bypass(
        physical, logical, "global_policy_budget")


def test_physical_single_egress_can_fail_while_global_budget_fits():
    # The node-wide budget admits both moves, but egress 1 only has one record.
    logical = _single_destination_logical_plan([4, 0], generation=22)
    assert logical.moved_copies == 2
    physical = materialize_physical_proxy_slots(
        logical,
        global_policy_budget=2,
        physical_capacity_per_egress=1,
    )
    assert physical.incoming_copies_per_egress == (0, 2)
    _assert_atomic_physical_bypass(
        physical, logical, "physical_egress_capacity")


def test_physical_total_may_exceed_per_egress_capacity_when_distributed():
    # P is a per-egress allocation, not a second global budget.  Both egresses
    # can independently use physical slot zero.
    logical = _single_destination_logical_plan([4, 0, 0], generation=23)
    physical = materialize_physical_proxy_slots(
        logical,
        global_policy_budget=2,
        physical_capacity_per_egress=1,
    )
    assert physical.enabled
    assert physical.moved_copies == 2 > physical.physical_capacity_per_egress
    assert physical.incoming_copies_per_egress == (0, 1, 1)
    assert [(assignment.egress, assignment.physical_slot)
            for assignment in physical.assignments] == [(1, 0), (2, 0)]
    assert len({assignment.physical_slot_key
                for assignment in physical.assignments}) == 2
    assert all(assignment.assignment.moved
               for assignment in physical.assignments)
    assert physical.committed_logical == logical


def test_physical_exact_boundaries_are_stable_unique_and_moved_only():
    logical = _single_destination_logical_plan([4, 0], generation=24)
    physical = materialize_physical_proxy_slots(
        logical,
        global_policy_budget=2,
        physical_capacity_per_egress=2,
    )
    assert physical.enabled
    assert physical.moved_copies == physical.global_policy_budget == 2
    assert physical.incoming_copies_per_egress == (0, 2)
    assert [(assignment.egress, assignment.physical_slot)
            for assignment in physical.assignments] == [(1, 0), (1, 1)]
    assert len({assignment.physical_slot_key
                for assignment in physical.assignments}) == 2
    assert all(assignment.assignment.moved
               for assignment in physical.assignments)
    retained_keys = {
        assignment.copy.key
        for assignment in logical.assignments
        if not assignment.moved
    }
    assert retained_keys.isdisjoint(
        assignment.assignment.copy.key
        for assignment in physical.assignments
    )

    # Physical packing must not depend on the order in which a caller supplies
    # already-materialized logical assignments.
    reversed_logical = replace(
        logical, assignments=tuple(reversed(logical.assignments)))
    replay = materialize_physical_proxy_slots(
        reversed_logical,
        global_policy_budget=2,
        physical_capacity_per_egress=2,
    )
    assert replay.enabled
    assert replay.assignments == physical.assignments


def test_capacity_failure_bypasses_the_whole_plan_without_partial_assignments():
    copies = _copies_from_specs([
        (0, 0, 0), (0, 1, 0), (0, 2, 0),
        (0, 0, 1), (0, 1, 1), (0, 2, 1),
        (1, 0, 0), (1, 0, 1),
    ], num_channels=1)
    plan = build_count_plan([[3, 3], [1, 1]])
    assert plan.moved_copies == 2
    bypassed = materialize_static_slots(
        copies, plan, num_channels=1, channel_capacity=2,
        proxy_payload_capacity=1, generation=5)
    assert not bypassed.enabled
    assert bypassed.bypass_reason == "proxy_payload_capacity"
    assert bypassed.moved_copies == 0
    assert bypassed.assignments == ()
    assert bypassed.committed_segments == ()
    assert bypassed.committed_quota == plan.counts


def test_channel_capacity_exact_boundary_passes_and_one_less_bypasses():
    copies = _copies_from_specs([
        (0, 0, 0), (0, 1, 0), (0, 2, 0),
        (1, 0, 0),
    ], num_channels=1)
    plan = build_count_plan([[3], [1]])
    passed = materialize_static_slots(
        copies, plan, num_channels=1, channel_capacity=2,
        proxy_payload_capacity=1)
    assert passed.enabled

    failed = materialize_static_slots(
        copies, plan, num_channels=1, channel_capacity=1,
        proxy_payload_capacity=1)
    assert not failed.enabled
    assert failed.bypass_reason in {"retained_channel_capacity", "channel_capacity"}
    assert failed.committed_quota == plan.counts
    assert failed.committed_keep_count == plan.counts
    assert failed.committed_segments == ()
    assert failed.moved_copies == 0
    assert failed.assignments == ()


def test_metadata_capacity_failure_is_an_atomic_identity_bypass():
    copies = _copies_from_specs([
        (0, 0, 0), (0, 1, 0), (0, 2, 0),
        (1, 0, 0),
    ], num_channels=1)
    candidate = build_count_plan([[3], [1]])
    bypassed = materialize_static_slots(
        copies,
        candidate,
        num_channels=1,
        channel_capacity=2,
        proxy_payload_capacity=1,
        metadata_capacity=0,
    )
    assert not bypassed.enabled
    assert bypassed.bypass_reason == "metadata_capacity"
    assert bypassed.committed_quota == candidate.counts
    assert bypassed.committed_keep_count == candidate.counts
    assert bypassed.committed_segments == ()
    assert bypassed.assignments == ()
    assert bypassed.moved_copies == 0


def test_already_balanced_plan_is_identity_bypass():
    copies = _copies_from_specs([(0, 0, 0), (1, 0, 0)], num_channels=1)
    plan = build_count_plan([[1], [1]])
    slots = materialize_static_slots(
        copies, plan, num_channels=1, channel_capacity=1, generation=11)
    assert not slots.enabled
    assert slots.bypass_reason == "already_balanced"
    assert slots.assignments == ()
    assert slots.committed_quota == plan.counts


def test_duplicate_copy_and_noncanonical_ordinal_are_input_errors():
    duplicate = DestinationCopy(0, 0, 0, 0, 0)
    _assert_raises(
        ValueError,
        "duplicate",
        lambda: counts_from_copies(
            [duplicate, duplicate], num_rails=1, num_destinations=1, num_channels=1)
    )

    _assert_raises(
        ValueError,
        "ordinal",
        lambda: counts_from_copies(
            [DestinationCopy(0, 0, 0, 0, 1)],
            num_rails=1, num_destinations=1, num_channels=1)
    )


def test_copy_count_mismatch_is_rejected_before_materialization():
    copies = _copies_from_specs([(0, 0, 0)], num_channels=1)
    candidate = build_count_plan([[2], [0]])
    _assert_raises(
        ValueError,
        "do not match",
        lambda: materialize_static_slots(
            copies, candidate, num_channels=1, channel_capacity=2)
    )


def test_seeded_random_routes_preserve_copy_and_slot_invariants():
    for seed in range(64):
        rng = random.Random(seed)
        num_rails = rng.randint(1, 8)
        num_destinations = rng.randint(1, 17)
        num_channels = rng.randint(1, 4)
        local_destination = rng.randrange(num_destinations)
        routes = []
        for _owner in range(num_rails):
            owner_tokens = []
            for _token in range(rng.randint(0, 16)):
                owner_tokens.append([
                    rng.randint(-2, num_destinations - 1)
                    for _lane in range(rng.randint(0, 8))
                ])
            routes.append(owner_tokens)

        copies = build_destination_copies(
            routes,
            num_destinations=num_destinations,
            num_channels=num_channels,
            local_destination=local_destination,
        )
        counts = counts_from_copies(
            copies,
            num_rails=num_rails,
            num_destinations=num_destinations,
            num_channels=num_channels,
        )
        candidate = build_count_plan(counts, remainder_seed=seed)
        validate_count_plan(candidate)

        slots = materialize_static_slots(
            copies,
            candidate,
            num_channels=num_channels,
            channel_capacity=max(len(copies), 1),
            proxy_payload_capacity=candidate.moved_copies,
            metadata_capacity=len(candidate.segments),
            generation=seed,
            channel_seed=seed,
        )
        assert slots == materialize_static_slots(
            tuple(reversed(copies)),
            candidate,
            num_channels=num_channels,
            channel_capacity=max(len(copies), 1),
            proxy_payload_capacity=candidate.moved_copies,
            metadata_capacity=len(candidate.segments),
            generation=seed,
            channel_seed=seed,
        )

        if candidate.moved_copies == 0:
            assert not slots.enabled
            assert slots.bypass_reason == "already_balanced"
            continue

        assert slots.enabled
        assert len(slots.assignments) == len(copies)
        assert {assignment.copy.key for assignment in slots.assignments} == {
            copy.key for copy in copies
        }
        assert len({assignment.slot_key for assignment in slots.assignments}) == len(copies)
        assigned_counts = [[0] * num_destinations for _ in range(num_rails)]
        moved_ordinals = set()
        for assignment in slots.assignments:
            assigned_counts[assignment.egress][assignment.destination] += 1
            keep_count = candidate.keep_count[
                assignment.copy.owner][assignment.destination]
            if assignment.moved:
                assert assignment.copy.owner_ordinal >= keep_count
                moved_ordinals.add((
                    assignment.copy.owner,
                    assignment.destination,
                    assignment.copy.owner_ordinal,
                ))
            else:
                assert assignment.egress == assignment.copy.owner
                assert assignment.copy.owner_ordinal < keep_count
        assert tuple(tuple(row) for row in assigned_counts) == candidate.quota
        segment_ordinals = {
            (segment.owner, segment.destination, ordinal)
            for segment in candidate.segments
            for ordinal in range(
                segment.owner_begin, segment.owner_begin + segment.count)
        }
        assert moved_ordinals == segment_ordinals


def test_zero_destination_shape_is_a_deterministic_identity():
    candidate = build_count_plan([[], [], []], remainder_seed=4)
    validate_count_plan(candidate)
    assert candidate.counts == ((), (), ())
    assert candidate.quota == candidate.counts
    assert candidate.moved_copies == 0


def test_corrupted_noncanonical_plan_is_rejected_before_materialization():
    candidate = build_count_plan([[4], [0], [0]], remainder_seed=1)
    corrupted = replace(candidate, quota=((1,), (2,), (1,)))
    _assert_raises(
        AssertionError,
        "canonical minimum-move quota",
        lambda: validate_count_plan(corrupted),
    )
    copies = _copies_from_specs(
        [(0, token, 0) for token in range(4)], num_channels=1)
    _assert_raises(
        AssertionError,
        "canonical minimum-move quota",
        lambda: materialize_static_slots(
            copies,
            corrupted,
            num_channels=1,
            channel_capacity=4,
            proxy_payload_capacity=4,
        ),
    )


def test_scalar_and_copy_fields_require_strict_integers():
    candidate = build_count_plan([[3], [1]])
    copies = _copies_from_specs([
        (0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0),
    ], num_channels=1)
    _assert_raises(
        ValueError,
        "generation must be an integer",
        lambda: materialize_static_slots(
            copies, candidate, num_channels=1, channel_capacity=2,
            generation=0.5),
    )
    _assert_raises(
        ValueError,
        "channel_seed must be an integer",
        lambda: materialize_static_slots(
            copies, candidate, num_channels=1, channel_capacity=2,
            channel_seed=True),
    )
    _assert_raises(
        ValueError,
        "num_channels must be at least 1",
        lambda: counts_from_copies(
            [], num_rails=1, num_destinations=0, num_channels=0),
    )
    _assert_raises(
        ValueError,
        "copy owner must be an integer",
        lambda: counts_from_copies(
            [DestinationCopy(0.0, 0, 0, 0, 0)],
            num_rails=1, num_destinations=1, num_channels=1),
    )
    _assert_raises(
        ValueError,
        "remainder_seed must be an integer",
        lambda: build_count_plan([[1]], remainder_seed=False),
    )
    logical = _single_destination_logical_plan([3, 1])
    _assert_raises(
        ValueError,
        "global_policy_budget must be an integer",
        lambda: materialize_physical_proxy_slots(
            logical,
            global_policy_budget=True,
            physical_capacity_per_egress=1,
        ),
    )
    _assert_raises(
        ValueError,
        "physical_capacity_per_egress must be an integer",
        lambda: materialize_physical_proxy_slots(
            logical,
            global_policy_budget=1,
            physical_capacity_per_egress=0.5,
        ),
    )


def _mutation_logical_plan():
    copies = _copies_from_specs([
        (owner, token, destination)
        for owner, count in enumerate((3, 1))
        for token in range(count)
        for destination in range(2)
    ], num_channels=1)
    candidate = build_count_plan([[3, 3], [1, 1]])
    logical = materialize_static_slots(
        copies, candidate, num_channels=1, channel_capacity=2,
        proxy_payload_capacity=2, generation=41)
    assert logical.enabled
    validate_static_slot_plan(logical)
    return logical


def _replace_assignment(plan, old, new):
    return replace(plan, assignments=tuple(
        new if assignment == old else assignment
        for assignment in plan.assignments))


def test_static_validator_rejects_assignment_field_mutations():
    logical = _mutation_logical_plan()
    moved = next(a for a in logical.assignments if a.moved)
    retained = next(a for a in logical.assignments if not a.moved)

    corruptions = [
        (moved, replace(
            moved, destination=1 - moved.destination),
         "destination differs"),
        (moved, replace(
            moved, channel=logical.num_channels),
         "channel is out of range"),
        (moved, replace(
            moved, slot=logical.channel_capacity),
         "slot is out of range"),
        (moved, replace(
            moved, generation=logical.generation + 1),
         "generation differs"),
        (moved, replace(
            moved, egress=moved.copy.owner),
         "moved copy must use"),
        (retained, replace(
            retained, egress=1 - retained.copy.owner),
         "retained copy must stay"),
        (moved, replace(moved, moved=False),
         "retained copy must stay|segment copy was marked retained"),
        (moved, replace(moved, moved=1),
         "moved must be a boolean"),
        (moved, replace(
            moved,
            copy=replace(
                moved.copy,
                source_channel=logical.num_channels)),
         "source channel is out of range"),
    ]
    for old, bad_assignment, pattern in corruptions:
        corrupted = _replace_assignment(logical, old, bad_assignment)
        _assert_raises(
            (AssertionError, ValueError), pattern,
            lambda corrupted=corrupted: validate_static_slot_plan(corrupted))


def test_static_validator_rejects_slot_collision_and_missing_copy():
    logical = _mutation_logical_plan()
    moved = next(a for a in logical.assignments if a.moved)
    retained = next(
        a for a in logical.assignments
        if not a.moved and a.egress == moved.egress
        and a.destination == moved.destination and a.channel == moved.channel)
    collision = _replace_assignment(
        logical, moved, replace(moved, slot=retained.slot))
    _assert_raises(
        AssertionError, "slot collision",
        lambda: validate_static_slot_plan(collision))

    missing = replace(logical, assignments=logical.assignments[:-1])
    _assert_raises(
        AssertionError, "do not cover every source copy",
        lambda: validate_static_slot_plan(missing))


def test_static_validator_rejects_commit_mutations():
    logical = _mutation_logical_plan()
    corruptions = [
        (replace(logical, committed_quota=logical.candidate.counts),
         "commit candidate quota"),
        (replace(logical, committed_keep_count=logical.candidate.counts),
         "commit candidate keep"),
        (replace(logical, committed_segments=()),
         "commit candidate segments"),
        (replace(logical, moved_copies=logical.moved_copies - 1),
         "moved-copy count"),
    ]
    for corrupted, pattern in corruptions:
        _assert_raises(
            AssertionError, pattern,
            lambda corrupted=corrupted: validate_static_slot_plan(corrupted))


def test_static_validator_rejects_disabled_partial_commit():
    logical = _mutation_logical_plan()
    copies = tuple(a.copy for a in sorted(
        logical.assignments,
        key=lambda a: (
            a.copy.owner, a.copy.destination,
            a.copy.owner_ordinal, a.copy.token)))
    disabled = materialize_static_slots(
        copies,
        logical.candidate,
        num_channels=logical.num_channels,
        channel_capacity=logical.channel_capacity,
        proxy_payload_capacity=0,
        generation=logical.generation,
    )
    assert not disabled.enabled
    validate_static_slot_plan(disabled)

    _assert_raises(
        AssertionError, "identity quota",
        lambda: validate_static_slot_plan(replace(
            disabled, committed_quota=logical.candidate.quota)))
    _assert_raises(
        AssertionError, "atomic identity commit",
        lambda: validate_static_slot_plan(replace(
            disabled, assignments=(logical.assignments[0],))))


def test_physical_helper_validates_logical_plan_before_packing():
    logical = _mutation_logical_plan()
    moved = next(a for a in logical.assignments if a.moved)
    corrupted = _replace_assignment(
        logical, moved,
        replace(moved, destination=1 - moved.destination))
    _assert_raises(
        AssertionError,
        "destination differs",
        lambda: materialize_physical_proxy_slots(
            corrupted,
            global_policy_budget=logical.moved_copies,
            physical_capacity_per_egress=logical.moved_copies,
        ),
    )


def test_contiguous_ready_tail_exhausts_all_six_slot_publication_orders():
    for order in itertools.permutations(range(6)):
        trace = simulate_contiguous_ready_publication(
            order, generation=29)
        assert len(trace.tail_history) == 7
        assert trace.tail_history[-1] == 6
        assert trace.final_ready == (29,) * 6
        for step, tail in enumerate(trace.tail_history):
            published = set(order[:step])
            expected = next(
                (slot for slot in range(6) if slot not in published), 6)
            assert tail == expected
            assert all(slot in published for slot in range(tail))
        assert all(
            observation.contiguous_tail < observation.max_ready_plus_one
            for observation in trace.holes)


def test_contiguous_ready_tail_ignores_stale_generation_and_forced_holes():
    order = (1, 3, 5, 7, 0, 2, 4, 6)
    trace = simulate_contiguous_ready_publication(
        order,
        generation=29,
        initial_ready=(17,) * 8,
    )
    assert trace.tail_history == (0, 0, 0, 0, 0, 2, 4, 6, 8)
    assert trace.stale_front_observations == 4
    assert trace.holes[0].contiguous_tail == 0
    assert trace.holes[0].max_ready_plus_one == 2
    assert trace.final_ready == (29,) * 8


def test_contiguous_ready_tail_discriminates_unsafe_shortcuts():
    stale = (17,) * 8
    assert advance_contiguous_ready_tail(
        stale, generation=29, start_tail=0) == 0
    assert advance_contiguous_ready_tail(
        (30, 29), generation=29, start_tail=0) == 0

    trace = simulate_contiguous_ready_publication(
        (7, 0, 1, 2, 3, 4, 5, 6), generation=29)
    assert trace.tail_history[1] == 0
    assert trace.holes[0].max_ready_plus_one == 8

    # Each tempting shortcut would cross a real hole in one of the cases:
    # boolean-ready accepts stale 17, >= accepts future 30, and max-ready+1
    # would publish 8 immediately after slot 7 arrives.
    assert next(index for index, value in enumerate(stale) if value != 29) == 0
    assert 30 >= 29 and 30 != 29
    assert max((7,)) + 1 > trace.tail_history[1]


def test_contiguous_ready_tail_handles_empty_and_partial_publication():
    assert advance_contiguous_ready_tail(
        (), generation=29, start_tail=0) == 0
    empty = simulate_contiguous_ready_publication((), generation=29)
    assert empty.tail_history == (0,)
    assert empty.final_ready == ()

    # Slots 2 and 3 are visible, but the missing slot 1 pins the safe tail.
    partial = (29, 0, 29, 29)
    assert advance_contiguous_ready_tail(
        partial, generation=29, start_tail=0) == 1
    assert advance_contiguous_ready_tail(
        partial, generation=29, start_tail=1) == 1
    completed = (29, 29, 29, 29)
    assert advance_contiguous_ready_tail(
        completed, generation=29, start_tail=1) == 4


def test_contiguous_ready_tail_rejects_unproven_start_prefix():
    invalid_calls = [
        lambda: advance_contiguous_ready_tail(
            (17, 17), generation=29, start_tail=2),
        lambda: advance_contiguous_ready_tail(
            (0, 29), generation=29, start_tail=1),
        lambda: advance_contiguous_ready_tail(
            (29, 17, 29), generation=29, start_tail=2),
    ]
    for call in invalid_calls:
        _assert_raises(
            ValueError,
            "start_tail is not backed by a current-generation ready prefix",
            call,
        )


def test_contiguous_ready_protocol_rejects_invalid_state():
    invalid_calls = [
        lambda: simulate_contiguous_ready_publication(
            (0, 0), generation=1),
        lambda: simulate_contiguous_ready_publication(
            (0, 2), generation=1),
        lambda: simulate_contiguous_ready_publication(
            (False,), generation=1),
        lambda: simulate_contiguous_ready_publication(
            (0,), generation=0),
        lambda: simulate_contiguous_ready_publication(
            (0,), generation=2, initial_ready=(2,)),
        lambda: simulate_contiguous_ready_publication(
            (0,), generation=2, initial_ready=()),
        lambda: advance_contiguous_ready_tail(
            (0,), generation=1, start_tail=2),
        lambda: advance_contiguous_ready_tail(
            (0,), generation=True, start_tail=0),
        lambda: advance_contiguous_ready_tail(
            (0,), generation=0x80000000, start_tail=0),
        lambda: advance_contiguous_ready_tail(
            (False,), generation=1, start_tail=0),
        lambda: advance_contiguous_ready_tail(
            (-1,), generation=1, start_tail=0),
        lambda: advance_contiguous_ready_tail(
            (0x80000000,), generation=1, start_tail=0),
        lambda: advance_contiguous_ready_tail(
            (0,), generation=1, start_tail=False),
        lambda: advance_contiguous_ready_tail(
            (0,), generation=1, start_tail=-1),
        lambda: simulate_contiguous_ready_publication(
            (0,), generation=0x80000000),
        lambda: simulate_contiguous_ready_publication(
            (0,), generation=1, initial_ready=(0x80000000,)),
    ]
    for call in invalid_calls:
        _assert_raises(ValueError, ".+", call)


if __name__ == "__main__":
    tests = sorted(
        (name, function) for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} rail-balance planner tests")
