"""C0-C7 contracts for the endpoint-aware RailBalance reference planner."""

from __future__ import annotations

import random
from dataclasses import replace

from rail_balance_hop_reference import (
    HopFlow,
    HopPlannerConfig,
    PlanCapacityError,
    build_hop_plan,
    endpoint_rails,
    local_forward_count,
    materialize_hop_records,
    validate_hop_plan,
)


def _config(
    mode: str,
    *,
    rails: int = 4,
    destinations: int = 2,
    chunk: int = 4,
    threshold: float = 0.0,
    ratio: float = 0.0,
    penalty: float = 0.0,
    seed: int = 0,
    capacity: int | None = None,
) -> HopPlannerConfig:
    return HopPlannerConfig(
        num_rails=rails,
        num_destinations=destinations,
        mode=mode,
        chunk_size=chunk,
        two_hop_threshold=threshold,
        max_two_hop_ratio=ratio,
        hop_penalty=penalty,
        planner_seed=seed,
        proxy_capacity_per_egress=capacity,
    )


def _raises(exception: type[Exception], text: str, function) -> None:
    try:
        function()
    except exception as error:
        assert text in str(error)
    else:
        raise AssertionError(f"expected {exception.__name__}")


def test_c0_balanced_stays_endpoint_only() -> None:
    flows = [HopFlow(owner, 1, ((owner + 1) % 4,), 8) for owner in range(4)]
    one_hop = build_hop_plan(flows, _config("one_hop"))
    adaptive = build_hop_plan(flows, _config("adaptive", ratio=0.5))

    assert one_hop.pair_load[1] == (8, 8, 8, 8)
    assert adaptive.pair_load == one_hop.pair_load
    assert adaptive.two_hop_units == 0
    assert adaptive.extra_local_forward_units == 0


def test_c1_offdiagonal_hotspot_is_solved_by_one_hop() -> None:
    flows = [
        HopFlow(0, 1, (1,), 16),
        HopFlow(0, 1, (2,), 16),
        HopFlow(0, 1, (3,), 16),
    ]
    off = build_hop_plan(flows, _config("off"))
    one_hop = build_hop_plan(flows, _config("one_hop"))

    assert max(one_hop.pair_load[1]) < max(off.pair_load[1])
    assert one_hop.units_for("dst_forward") > 0
    assert one_hop.units_for("src_forward") > 0
    assert one_hop.two_hop_units == 0
    for assignment in one_hop.assignments:
        assert assignment.egress in {assignment.owner, *assignment.targets}


def test_c2_diagonal_hotspot_requires_two_hop() -> None:
    flows = [HopFlow(0, 1, (0,), 32)]
    one_hop = build_hop_plan(flows, _config("one_hop"))
    adaptive = build_hop_plan(flows, _config("adaptive", ratio=0.5))

    assert one_hop.pair_load[1] == (32, 0, 0, 0)
    assert one_hop.units_for("direct") == 32
    assert adaptive.two_hop_units == 16
    assert max(adaptive.pair_load[1]) < max(one_hop.pair_load[1])
    assert adaptive.extra_local_forward_units == 2 * adaptive.two_hop_units


def test_c3_closed_block_escapes_without_spraying_everything() -> None:
    flows = [HopFlow(0, 1, (1,), 32), HopFlow(1, 1, (0,), 32)]
    one_hop = build_hop_plan(flows, _config("one_hop"))
    adaptive = build_hop_plan(flows, _config("adaptive", ratio=0.25))

    assert one_hop.pair_load[1][2:] == (0, 0)
    assert adaptive.pair_load[1][2] + adaptive.pair_load[1][3] > 0
    assert adaptive.two_hop_units == 16
    assert adaptive.two_hop_units < adaptive.total_units
    assert max(adaptive.pair_load[1]) < max(one_hop.pair_load[1])


def test_c4_gain_and_penalty_reject_unneeded_two_hop() -> None:
    balanced = [HopFlow(owner, 1, ((owner + 1) % 4,), 8) for owner in range(4)]
    assert build_hop_plan(
        balanced, _config("adaptive", ratio=1.0)
    ).two_hop_units == 0

    diagonal = [HopFlow(0, 1, (0,), 32)]
    assert build_hop_plan(
        diagonal, _config("adaptive", threshold=0.2, ratio=1.0)
    ).two_hop_units == 0
    assert build_hop_plan(
        diagonal, _config("adaptive", ratio=1.0, penalty=1.0)
    ).two_hop_units == 0


def test_c5_two_hop_cap_handles_a_partial_chunk_deterministically() -> None:
    flows = [HopFlow(0, 1, (0,), 11)]
    config = _config("adaptive", chunk=4, ratio=0.25)
    first = build_hop_plan(flows, config)
    second = build_hop_plan(flows, config)

    assert first == second
    assert first.two_hop_units == 2
    assert first.two_hop_units <= int(first.total_units * config.max_two_hop_ratio)
    assert sum(item.count for item in first.assignments) == 11


def test_c6_capacity_and_corruption_fail_the_complete_plan() -> None:
    flows = [HopFlow(0, 1, (1,), 16)]
    _raises(
        PlanCapacityError,
        "complete plan proxy load",
        lambda: build_hop_plan(flows, _config("one_hop", capacity=1)),
    )

    plan = build_hop_plan(flows, _config("one_hop", capacity=32))
    original = plan.assignments[0]
    corrupt = replace(
        plan,
        assignments=(replace(original, path_kind="two_hop"), *plan.assignments[1:]),
    )
    _raises(
        ValueError,
        "path_kind disagrees",
        lambda: validate_hop_plan(flows, corrupt),
    )


def test_c7_random_small_oracle_properties() -> None:
    for case_seed in range(1500):
        rng = random.Random(case_seed)
        rails = rng.randint(2, 4)
        destinations = rng.randint(1, 4)
        flows = []
        for _ in range(rng.randint(1, 6)):
            targets = tuple(sorted(rng.sample(
                range(rails), rng.randint(1, min(rails, 3))
            )))
            flows.append(HopFlow(
                owner=rng.randrange(rails),
                destination=rng.randrange(destinations),
                targets=targets,
                count=rng.randint(1, 9),
            ))
        ratio = rng.choice((0.0, 0.125, 0.25, 0.5))
        mode = rng.choice(("one_hop", "adaptive"))
        config = _config(
            mode,
            rails=rails,
            destinations=destinations,
            chunk=rng.randint(1, 4),
            threshold=rng.choice((0.0, 0.05, 0.2)),
            ratio=ratio,
            penalty=rng.choice((0.0, 0.25, 1.0)),
            seed=rng.randrange(32),
        )
        try:
            first = build_hop_plan(flows, config)
            second = build_hop_plan(flows, config)
            assert first == second
            validate_hop_plan(flows, first)
            assert first.total_units == sum(flow.count for flow in flows)
            if mode == "one_hop":
                assert all(
                    item.egress in endpoint_rails(flows[item.flow_index])
                    for item in first.assignments
                )
            else:
                assert first.two_hop_units <= int(first.total_units * ratio)
        except Exception as error:
            raise AssertionError(
                f"random case failed: seed={case_seed}, config={config}, flows={flows}"
            ) from error


def test_multitarget_payload_is_not_duplicated() -> None:
    flows = [HopFlow(0, 1, (1, 2), 16)]
    plan = build_hop_plan(flows, _config("one_hop"))

    assert plan.total_units == 16
    assert sum(item.count for item in plan.assignments) == 16
    assert all(item.egress in (0, 1, 2) for item in plan.assignments)
    assert plan.two_hop_units == 0
    expected_local = sum(
        item.count * local_forward_count(item.owner, item.targets, item.egress)
        for item in plan.assignments
    )
    assert plan.local_forward_units == expected_local


def test_topk_materializer_retains_target_mask_and_node_dedup() -> None:
    # D=2, G=4, two experts/rank. Experts 8, 10, and 11 all live on node 1;
    # the latter two share local rank 1 and must still produce one payload.
    topk_idx = (
        ((8, 10, 11, 0),),
        ((9, 14, 2, 3),),
        ((4, 5, 12, 15),),
        ((1, 6, 13, 7),),
    )
    records = materialize_hop_records(
        topk_idx,
        num_experts=16,
        num_destinations=2,
        num_rails=4,
        local_destination=0,
    )

    assert records[0][0][0].destination == 1
    assert records[0][0][0].target_mask == 0b0011
    assert records[0][0][1].target_mask == 0
    assert records[1][0][0].target_mask == 0b1001
    assert records[2][0][0].target_mask == 0b1100
    assert records[3][0][0].target_mask == 0b0100
    assert sum(record.target_mask != 0 for owner in records
               for token in owner for record in token) == 4


def test_legacy_exact_can_select_a_nonendpoint_rail() -> None:
    flows = [HopFlow(0, 1, (0,), 16)]
    legacy = build_hop_plan(flows, _config("legacy_exact"))
    one_hop = build_hop_plan(flows, _config("one_hop"))

    assert legacy.two_hop_units > 0
    assert one_hop.two_hop_units == 0


def test_invalid_contracts_raise_without_fallback() -> None:
    _raises(
        ValueError,
        "sorted and unique",
        lambda: HopFlow(0, 0, (1, 1), 1),
    )
    flow = [HopFlow(0, 0, (0,), 1)]
    _raises(
        ValueError,
        "two_hop_threshold",
        lambda: build_hop_plan(flow, _config("adaptive", threshold=1.1)),
    )
    _raises(
        ValueError,
        "target is outside",
        lambda: build_hop_plan(
            [HopFlow(0, 0, (4,), 1)], _config("one_hop")
        ),
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
    print(f"PASS {len(tests)} hop-aware RailBalance reference tests")
