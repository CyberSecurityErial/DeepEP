"""Focused CPU tests for Hybrid rail-balance policy selection.

Run with::

    PYTHONPATH=. python -B tests/elastic/test_rail_balance_hybrid_policy.py

Policies only choose the quota target.  The compact segment builder, channel
packing, resolver, and data-plane schedule remain shared by every policy.
"""

from __future__ import annotations

from dataclasses import fields

from rail_balance_hybrid_reference import (
    HybridTransferSegment,
    build_hybrid_rail_schedule_from_destinations,
    enumerate_resolved_copies,
    validate_hybrid_rail_schedule,
)


def _routes_from_counts(counts):
    routes = []
    for owner_counts in counts:
        owner = []
        for destination, count in enumerate(owner_counts):
            owner.extend(((destination,),) * count)
        routes.append(tuple(owner))
    return tuple(routes)


def _build(
    counts,
    *,
    policy="all",
    threshold_percent=0,
    remainder_seed=0,
):
    routes = _routes_from_counts(counts)
    max_tokens = max(1, max(len(owner) for owner in routes))
    schedule = build_hybrid_rail_schedule_from_destinations(
        routes,
        num_destinations=len(counts[0]),
        num_channels=3,
        num_max_tokens_per_rank=max_tokens,
        remainder_seed=remainder_seed,
        policy=policy,
        threshold_percent=threshold_percent,
    )
    return routes, schedule


def _column(matrix, destination=0):
    return tuple(row[destination] for row in matrix)


def _expect_value_error(function, fragment):
    try:
        function()
    except ValueError as error:
        assert fragment in str(error), str(error)
        return
    raise AssertionError("expected ValueError")


def test_all_zero_threshold_preserves_the_previous_golden() -> None:
    # This distribution cannot lower its peak of two, but the historical exact
    # all-rail planner still fills rail three.  threshold=0 means that the gate
    # is disabled, so the old quota and segment remain byte-for-byte stable.
    counts = ((2,), (2,), (1,), (0,))
    _routes, implicit = _build(counts)
    _routes, explicit = _build(counts, policy="all", threshold_percent=0)
    assert implicit == explicit
    assert explicit.policy == "all"
    assert explicit.threshold_percent == 0
    assert _column(explicit.quota) == (2, 1, 1, 1)
    assert explicit.segments == ((HybridTransferSegment(1, 3, 1, 1, 0),),)
    assert explicit.moved_copies == 1


def test_active_uses_only_original_rails_3_5_7() -> None:
    counts = tuple((value,) for value in (0, 0, 0, 12, 0, 3, 0, 6))
    _routes, all_schedule = _build(counts, policy="all", remainder_seed=1)
    _routes, active = _build(counts, policy="active", remainder_seed=1)

    assert _column(all_schedule.quota) == (2, 3, 3, 3, 2, 3, 2, 3)
    assert _column(active.quota) == (0, 0, 0, 7, 0, 7, 0, 7)
    assert active.segments == (
        (
            HybridTransferSegment(3, 5, 7, 4, 0),
            HybridTransferSegment(3, 7, 11, 1, 0),
        ),
    )
    assert active.proxy_required == (0, 0, 0, 0, 0, 4, 0, 1)
    assert active.moved_copies == 5
    assert {egress for egress, quota in enumerate(_column(active.quota)) if quota} == {
        3,
        5,
        7,
    }


def test_threshold_uses_strict_greater_than_and_zero_disables_gate() -> None:
    counts = ((6,), (2,), (0,), (0,))
    _routes, disabled = _build(counts, policy="active", threshold_percent=0)
    _routes, below = _build(counts, policy="active", threshold_percent=49)
    _routes, equal = _build(counts, policy="active", threshold_percent=50)

    assert _column(disabled.quota) == (4, 4, 0, 0)
    assert _column(below.quota) == (4, 4, 0, 0)
    # observed_max=6 and target=4: 6*100 == 4*(100+50), so strict >
    # rejects the rewrite and the compact plan is the identity.
    assert equal.quota == equal.count
    assert equal.keep_count == equal.count
    assert equal.segments == ((),)
    assert equal.moved_copies == 0


def test_adaptive_recruitment_is_ring_deterministic() -> None:
    counts = ((8,), (0,), (0,), (0,))
    _routes, seed_two = _build(
        counts,
        policy="adaptive",
        threshold_percent=99,
        remainder_seed=2,
    )
    _routes, seed_three = _build(
        counts,
        policy="adaptive",
        threshold_percent=99,
        remainder_seed=3,
    )
    _routes, exact_boundary = _build(
        counts,
        policy="adaptive",
        threshold_percent=100,
        remainder_seed=2,
    )

    # 8 -> 4 exceeds 99%, so exactly the first inactive rail in the ring joins.
    # 4 -> 3 does not exceed 99%, so expansion stops.
    assert _column(seed_two.quota) == (4, 0, 4, 0)
    assert _column(seed_three.quota) == (4, 0, 0, 4)
    assert seed_two.segments == ((HybridTransferSegment(0, 2, 4, 4, 0),),)
    assert seed_three.segments == ((HybridTransferSegment(0, 3, 4, 4, 0),),)
    # 8 -> 4 is exactly 100%; strict > rejects recruitment and the final gate.
    assert exact_boundary.quota == exact_boundary.count
    assert exact_boundary.segments == ((),)


def test_adaptive_stops_at_the_first_failed_marginal_step() -> None:
    counts = ((8,), (0,), (0,), (0,))
    _routes, schedule = _build(
        counts,
        policy="adaptive",
        threshold_percent=40,
        remainder_seed=0,
    )
    # The discrete tails are 8, 4, 3, 2.  The first step exceeds 40%, the
    # second does not, and the greedy policy stops at two rails.  It does not
    # jump over that failed step even though the later 3 -> 2 ratio is larger.
    assert _column(schedule.quota) == (4, 4, 0, 0)
    assert schedule.moved_copies == 4


def test_adaptive_expands_3_5_7_in_ring_order() -> None:
    counts = tuple((value,) for value in (0, 0, 0, 12, 0, 3, 0, 6))
    _routes, no_recruitment = _build(
        counts,
        policy="adaptive",
        threshold_percent=20,
        remainder_seed=1,
    )
    _routes, recruited = _build(
        counts,
        policy="adaptive",
        threshold_percent=16,
        remainder_seed=1,
    )

    assert _column(no_recruitment.quota) == (0, 0, 0, 7, 0, 7, 0, 7)
    # total=21 has selected-set tails 7,6,5,4,3,3.  At 16%, rails
    # 1,2,4,6 join in ring order; rail 0 is rejected because 3 cannot fall.
    assert _column(recruited.quota) == (0, 3, 3, 3, 3, 3, 3, 3)
    assert {
        egress for egress, quota in enumerate(_column(recruited.quota)) if quota
    } == {1, 2, 3, 4, 5, 6, 7}


def test_every_policy_conserves_counts_and_uses_shared_segments() -> None:
    counts = (
        (8, 0, 0),
        (2, 4, 0),
        (0, 1, 0),
        (0, 0, 0),
    )
    settings = (
        ("all", 0),
        ("active", 0),
        ("adaptive", 0),
        ("all", 25),
        ("active", 25),
        ("adaptive", 25),
        ("adaptive", 3100),
    )
    for policy, threshold_percent in settings:
        routes, schedule = _build(
            counts,
            policy=policy,
            threshold_percent=threshold_percent,
            remainder_seed=3,
        )
        validate_hybrid_rail_schedule(schedule)
        assert schedule.policy == policy
        assert schedule.threshold_percent == threshold_percent
        assert _column(schedule.count, 2) == (0, 0, 0, 0)
        assert _column(schedule.quota, 2) == (0, 0, 0, 0)
        assert schedule.segments[2] == ()

        expected_moved = 0
        for destination in range(schedule.num_destinations):
            source = _column(schedule.count, destination)
            target = _column(schedule.quota, destination)
            assert sum(source) == sum(target)
            moved = sum(max(count - quota, 0) for count, quota in zip(source, target))
            assert moved == sum(
                segment.count for segment in schedule.segments[destination]
            )
            expected_moved += moved
        assert schedule.moved_copies == expected_moved

        records = enumerate_resolved_copies(routes, schedule)
        assert len(records) == sum(sum(row) for row in schedule.count)
        assert (
            sum(record.resolution.moved for record in records) == schedule.moved_copies
        )

        # Selection is planner-local.  Every policy exports the same compact
        # schedule/data-plane fields rather than an eligible-mask side channel.
        field_names = {field.name for field in fields(schedule)}
        assert "selected" not in field_names
        assert "eligible" not in field_names
        assert "selected_mask" not in field_names
        assert "eligible_mask" not in field_names


def test_policy_configuration_validation() -> None:
    counts = ((1,), (0,))
    for value in (None, "ALL", 0):
        _expect_value_error(
            lambda value=value: _build(counts, policy=value),
            "policy",
        )
    for value in (True, -1, 3101, 1.0):
        _expect_value_error(
            lambda value=value: _build(counts, threshold_percent=value),
            "threshold_percent",
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
    print(f"PASS {len(tests)} Hybrid rail-balance policy tests")
