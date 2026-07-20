"""Pure-CPU boundary tests for the C080 Hybrid vnode round-trip oracle."""

from __future__ import annotations

from dataclasses import replace

from rail_balance_hybrid_vnode_reference import (
    VnodeTopology,
    build_2x4_case,
    build_4x2_case,
    derive_reduce_row,
    run_vnode_roundtrip,
)


def _expect_failure(function, exception=ValueError) -> None:
    try:
        function()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def _assert_route_coordinates(case, result) -> None:
    topology = case.topology
    for route in result.routes:
        assert route.source_physical == topology.physical(0, route.owner)
        assert route.egress_physical == topology.physical(0, route.egress)
        assert route.ingress_physical == topology.physical(
            route.destination, route.egress)
        assert route.return_physical == route.egress_physical
        assert route.final_owner_physical == route.source_physical
        assert all(
            topology.coordinates(physical)[0] == route.destination
            for physical in route.expert_physicals
        )
        if route.moved:
            assert route.egress != route.owner and route.proxy_slot >= 0
        else:
            assert route.proxy_slot == -1


def test_physical_rank_mapping_is_node_major() -> None:
    for rails, nodes in ((4, 2), (2, 4)):
        topology = VnodeTopology(rails, nodes)
        assert topology.world_size == 8
        assert tuple(
            tuple(topology.physical(node, rail) for rail in range(rails))
            for node in range(nodes)
        ) == tuple(
            tuple(range(node * rails, (node + 1) * rails))
            for node in range(nodes)
        )
        for physical_rank in range(8):
            node, rail = topology.coordinates(physical_rank)
            assert topology.physical(node, rail) == physical_rank

    _expect_failure(lambda: VnodeTopology(0, 2))
    _expect_failure(lambda: VnodeTopology(2, 1))
    _expect_failure(lambda: VnodeTopology(True, 4))
    _expect_failure(lambda: VnodeTopology(2, 4).physical(4, 0))
    _expect_failure(lambda: VnodeTopology(2, 4).physical(0, 2))
    _expect_failure(lambda: VnodeTopology(2, 4).coordinates(8))


def test_reduce_row_rank_and_topk_layouts() -> None:
    # D <= K selects rank layout; row is destination and needs no lane scan.
    assert derive_reduce_row(
        destination=1,
        topk_idx=(0, 8, 1, 9),
        num_destinations=2,
        num_experts=16,
    ) == 1

    # D > K selects top-k layout.  The correct row is the last/highest
    # matching lane, not the first one.  This is the audit's minimal counter-
    # example: destinations [2, 1, 2] for d=2 must produce row two.
    assert derive_reduce_row(
        destination=2,
        topk_idx=(8, 4, 9),
        num_destinations=4,
        num_experts=16,
    ) == 2
    assert derive_reduce_row(
        destination=3,
        topk_idx=(4, 12),
        num_destinations=4,
        num_experts=16,
    ) == 1
    _expect_failure(lambda: derive_reduce_row(
        destination=2,
        topk_idx=(4, 12),
        num_destinations=4,
        num_experts=16,
    ))
    _expect_failure(lambda: derive_reduce_row(
        destination=4,
        topk_idx=(4, 12),
        num_destinations=4,
        num_experts=16,
    ))


def test_4x2_d_le_k_complete_round_trip() -> None:
    case = build_4x2_case()
    result = run_vnode_roundtrip(case)
    _assert_route_coordinates(case, result)
    schedule = result.schedule
    assert case.topology.rails_per_node == 4
    assert case.topology.num_nodes == 2 <= case.num_topk == 4
    assert case.num_tokens_per_owner == (4, 2, 1, 1)
    assert schedule.count == (
        (0, 4),
        (0, 2),
        (0, 0),
        (0, 0),
    )
    assert schedule.quota == (
        (0, 2),
        (0, 2),
        (0, 1),
        (0, 1),
    )
    assert schedule.moved_copies == 2
    assert schedule.proxy_required == (0, 0, 1, 1)

    remote = [route for route in result.routes if route.is_remote]
    retained = [route for route in remote if not route.moved]
    moved = [route for route in remote if route.moved]
    local = [route for route in result.routes if not route.is_remote]
    assert len(remote) == 6
    assert len(retained) == 4
    assert len(moved) == 2
    assert len(local) == 8
    assert {(route.egress, route.proxy_slot) for route in moved} == {
        (2, 0), (3, 0)}
    assert {route.ingress_physical for route in moved} == {6, 7}
    assert {route.ingress_physical for route in retained} == {4, 5}
    assert all(route.reduce_row == route.destination
               for route in result.routes)
    assert all(route.proxy_slot == -1 for route in retained + local)
    assert result.combined == result.direct
    assert all(any(value != 0 for value in token)
               for owner in result.combined for token in owner)


def test_2x4_d_gt_k_complete_round_trip() -> None:
    case = build_2x4_case()
    result = run_vnode_roundtrip(case)
    _assert_route_coordinates(case, result)
    schedule = result.schedule
    assert case.topology.rails_per_node == 2
    assert case.topology.num_nodes == 4 > case.num_topk == 2
    assert case.num_tokens_per_owner == (6, 6)
    assert schedule.count == (
        (0, 5, 2, 1),
        (0, 1, 2, 5),
    )
    assert schedule.quota == (
        (0, 3, 2, 3),
        (0, 3, 2, 3),
    )
    assert schedule.moved_copies == 4
    assert schedule.proxy_required == (2, 2)

    remote = [route for route in result.routes if route.is_remote]
    retained = [route for route in remote if not route.moved]
    moved = [route for route in remote if route.moved]
    local = [route for route in result.routes if not route.is_remote]
    assert len(remote) == 16
    assert len(retained) == 12
    assert len(moved) == 4
    assert len(local) == 6
    assert {(route.egress, route.proxy_slot) for route in moved} == {
        (0, 0), (0, 1), (1, 0), (1, 1)}
    assert {route.ingress_physical for route in moved} == {3, 6}
    assert all(0 <= route.reduce_row < case.num_topk
               for route in result.routes)
    for route in result.routes:
        matching = [
            lane for lane, expert in enumerate(
                case.topk_idx[route.owner][route.token])
            if expert // (case.num_experts // case.topology.num_nodes)
            == route.destination
        ]
        assert matching and route.reduce_row == matching[-1]

    # Both fixtures with repeated destinations select the highest lane.
    owner_zero_repeat = next(
        route for route in result.routes
        if route.owner == 0 and route.token == 0 and route.destination == 1)
    owner_one_repeat = next(
        route for route in result.routes
        if route.owner == 1 and route.token == 0 and route.destination == 3)
    assert owner_zero_repeat.topk_lanes == (0, 1)
    assert owner_one_repeat.topk_lanes == (0, 1)
    assert owner_zero_repeat.reduce_row == owner_one_repeat.reduce_row == 1
    assert result.combined == result.direct


def test_capacity_and_fixture_validation_fail_closed() -> None:
    case = build_2x4_case()
    _expect_failure(
        lambda: run_vnode_roundtrip(replace(
            case, proxy_capacity_per_egress=1)),
        RuntimeError,
    )
    _expect_failure(lambda: run_vnode_roundtrip(replace(
        case,
        source_values=(case.source_values[0][:-1], case.source_values[1]),
    )))
    duplicate = list(case.topk_idx[0][0])
    duplicate[1] = duplicate[0]
    owner_zero = list(case.topk_idx[0])
    owner_zero[0] = tuple(duplicate)
    _expect_failure(lambda: run_vnode_roundtrip(replace(
        case,
        topk_idx=(tuple(owner_zero), case.topk_idx[1]),
    )))


def main() -> None:
    tests = (
        test_physical_rank_mapping_is_node_major,
        test_reduce_row_rank_and_topk_layouts,
        test_4x2_d_le_k_complete_round_trip,
        test_2x4_d_gt_k_complete_round_trip,
        test_capacity_and_fixture_validation_fail_closed,
    )
    for test in tests:
        test()
    print(
        "PASS C080 vnode CPU oracle: 5 tests; 4x2 D<=K, 2x4 D>K, "
        "retained/moved/proxy p, highest-lane reduce rows, synthetic "
        "expert/combine/unshuffle round trip",
        flush=True,
    )


if __name__ == "__main__":
    main()
