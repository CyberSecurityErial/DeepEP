"""Pure-CPU boundary tests for the C080 Hybrid vnode round-trip oracle."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import torch

from rail_balance_hybrid_vnode_reference import (
    VnodeTopology,
    build_2x4_case,
    build_4x2_case,
    build_rounding_sensitive_case,
    combine_reduce_bf16,
    derive_reduce_row,
    round_bfloat16,
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


def _assert_remote_unique_topk(case) -> None:
    experts_per_destination = case.num_experts // case.topology.num_nodes
    for owner in case.topk_idx:
        for row in owner:
            assert len(row) == len(set(row)) == case.num_topk
            assert all(expert >= experts_per_destination for expert in row)


def _segments(schedule) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(tuple(
        (segment.owner, segment.egress, segment.owner_begin,
         segment.count, segment.egress_begin)
        for segment in bucket
    ) for bucket in schedule.segments)


def _dense_occupancy(result) -> dict[
        tuple[int, int, int], tuple[int, int, bool, int]]:
    occupancy = {}
    for route in result.routes:
        key = (route.egress, route.destination, route.dense_slot)
        assert key not in occupancy
        occupancy[key] = (
            route.owner, route.token, route.moved, route.proxy_slot)
    return occupancy


def _assert_dense_prefix(result) -> None:
    occupancy = _dense_occupancy(result)
    schedule = result.schedule
    for egress in range(schedule.num_rails):
        for destination in range(1, schedule.num_destinations):
            slots = sorted(
                dense for (candidate_egress, candidate_destination, dense)
                in occupancy
                if candidate_egress == egress and
                candidate_destination == destination
            )
            assert slots == list(range(schedule.quota[egress][destination]))


def _combined_constants(case, result) -> tuple[tuple[Fraction, ...], ...]:
    constants = []
    for owner, tokens in enumerate(result.combined):
        owner_constants = []
        for token, output in enumerate(tokens):
            source = case.source_values[owner][token]
            deltas = tuple(value - base for value, base in zip(output, source))
            assert deltas and all(delta == deltas[0] for delta in deltas)
            owner_constants.append(deltas[0])
        constants.append(tuple(owner_constants))
    return tuple(constants)


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


def test_bfloat16_rounding_and_combine_order() -> None:
    # BF16 RN-even: the first halfway value rounds to an even low mantissa;
    # the second rounds upward to the next even mantissa.
    assert round_bfloat16(Fraction(257, 256)) == Fraction(1)
    assert round_bfloat16(Fraction(259, 256)) == Fraction(65, 64)

    # Two valid slots use BF16 hadd and lose the half-ULP. Three valid slots
    # use ordered FP32 accumulation, retain both increments, then cast once.
    one = (Fraction(1),)
    half_ulp = (Fraction(1, 256),)
    assert combine_reduce_bf16((one, half_ulp)) == (Fraction(1),)
    assert combine_reduce_bf16(
        (one, half_ulp, half_ulp)) == (Fraction(129, 128),)


def test_rounding_sensitive_legacy_two_level_is_authoritative() -> None:
    case = build_rounding_sensitive_case()
    result = run_vnode_roundtrip(case)
    _assert_remote_unique_topk(case)
    _assert_route_coordinates(case, result)
    _assert_dense_prefix(result)

    assert case.topk_idx == (((4, 7, 11),), ())
    assert result.reduce_rows == ((
        ((Fraction(0),) * 3,
         (Fraction(1),) * 3,
         (Fraction(1, 256),) * 3),
    ), ())
    assert result.combined == (((Fraction(1),) * 3,), ())
    assert result.direct == (((Fraction(129, 128),) * 3,), ())
    assert result.combined != result.direct

    # Cross-check the exact oracle against host torch BF16: the two legacy
    # rounding boundaries produce 1, while a wrong flat three-slot FP32
    # reduction produces the next BF16 value, 129/128.
    partials = torch.tensor(
        [1.0, 1.0 / 256.0, 1.0 / 256.0], dtype=torch.bfloat16)
    destination_one = partials[0] + partials[1]
    two_level = destination_one + partials[2]
    flat = partials.float().sum().to(torch.bfloat16)
    assert two_level.item() == 1.0
    assert flat.item() == 129.0 / 128.0
    assert two_level.view(torch.int16).item() == 0x3f80
    assert flat.view(torch.int16).item() == 0x3f81


def test_4x2_d_le_k_complete_round_trip() -> None:
    case = build_4x2_case()
    result = run_vnode_roundtrip(case)
    _assert_remote_unique_topk(case)
    _assert_route_coordinates(case, result)
    _assert_dense_prefix(result)
    schedule = result.schedule
    assert case.topology.rails_per_node == 4
    assert case.topology.num_nodes == 2 <= case.num_topk == 4
    assert case.num_tokens_per_owner == (4, 2, 1, 1)
    assert case.proxy_capacity_per_egress == 1
    assert case.topk_idx == (
        ((8, 10, 12, 14), (11, 13, 15, 8),
         (12, 14, 9, 11), (15, 8, 10, 12)),
        ((14, 9, 11, 13), (8, 10, 12, 14)),
        ((13, 15, 8, 10),),
        ((10, 12, 14, 9),),
    )
    assert schedule.channel_count == (
        ((0, 2), (0, 2)),
        ((0, 1), (0, 1)),
        ((0, 1), (0, 0)),
        ((0, 1), (0, 0)),
    )
    assert schedule.owner_channel_prefix == (
        ((0, 0), (0, 2)),
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1)),
    )
    assert schedule.count == (
        (0, 4),
        (0, 2),
        (0, 1),
        (0, 1),
    )
    assert sum(sum(row) for row in schedule.count) == 8
    assert schedule.quota == (
        (0, 2),
        (0, 2),
        (0, 2),
        (0, 2),
    )
    assert schedule.keep_count == (
        (0, 2), (0, 2), (0, 1), (0, 1))
    assert _segments(schedule) == (
        (),
        ((0, 2, 2, 1, 0), (0, 3, 3, 1, 0)),
    )
    assert schedule.num_segments == (0, 2)
    assert schedule.retained == (
        ((0, 2), (0, 0)),
        ((0, 1), (0, 1)),
        ((0, 1), (0, 0)),
        ((0, 1), (0, 0)),
    )
    assert schedule.moved == (
        ((0, 0), (0, 0)),
        ((0, 0), (0, 0)),
        ((0, 1), (0, 0)),
        ((0, 1), (0, 0)),
    )
    assert schedule.moved_channel_prefix == (
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (0, 1, 1)),
        ((0, 0, 0), (0, 1, 1)),
    )
    assert schedule.group_prefix == (
        ((0, 0), (0, 0)),
        ((0, 0), (0, 0)),
        ((0, 0), (1, 1)),
        ((0, 0), (1, 1)),
    )
    assert schedule.moved_copies == 2
    assert schedule.proxy_required == (0, 0, 1, 1)

    remote = [route for route in result.routes if route.is_remote]
    retained = [route for route in remote if not route.moved]
    moved = [route for route in remote if route.moved]
    local = [route for route in result.routes if not route.is_remote]
    assert len(remote) == 8
    assert len(retained) == 6
    assert len(moved) == 2
    assert not local
    assert all(route.destination > 0 for route in result.routes)
    assert {
        (route.owner, route.token, route.destination, route.egress,
         route.proxy_slot, route.dense_slot, route.reduce_row)
        for route in moved
    } == {
        (0, 1, 1, 2, 0, 1, 1),
        (0, 3, 1, 3, 0, 1, 1),
    }
    assert {route.ingress_physical for route in moved} == {6, 7}
    assert {route.ingress_physical for route in retained} == {4, 5, 6, 7}
    assert all(route.reduce_row == route.destination
               for route in result.routes)
    assert all(route.proxy_slot == -1 for route in retained)
    assert _dense_occupancy(result) == {
        (0, 1, 0): (0, 0, False, -1),
        (0, 1, 1): (0, 2, False, -1),
        (1, 1, 0): (1, 0, False, -1),
        (1, 1, 1): (1, 1, False, -1),
        (2, 1, 0): (2, 0, False, -1),
        (2, 1, 1): (0, 1, True, 0),
        (3, 1, 0): (3, 0, False, -1),
        (3, 1, 1): (0, 3, True, 0),
    }
    assert {route.vnode_slot for route in result.routes} == {0, 1}
    assert result.combined == result.direct
    assert _combined_constants(case, result) == (
        (Fraction(22), Fraction(37), Fraction(40), Fraction(42)),
        (Fraction(42), Fraction(22)),
        (Fraction(44),),
        (Fraction(31),),
    )
    assert all(any(value != 0 for value in token)
               for owner in result.combined for token in owner)


def test_2x4_d_gt_k_complete_round_trip() -> None:
    case = build_2x4_case()
    result = run_vnode_roundtrip(case)
    _assert_remote_unique_topk(case)
    _assert_route_coordinates(case, result)
    _assert_dense_prefix(result)
    schedule = result.schedule
    assert case.topology.rails_per_node == 2
    assert case.topology.num_nodes == 4 > case.num_topk == 2
    assert case.num_tokens_per_owner == (6, 6)
    assert case.proxy_capacity_per_egress == 4
    assert case.topk_idx == (
        ((4, 10), (14, 13), (5, 11), (15, 12), (12, 14), (14, 13)),
        ((7, 8), (4, 10), (6, 9), (5, 11), (7, 8), (4, 10)),
    )
    assert schedule.channel_count == (
        ((0, 2, 2, 1), (0, 0, 0, 3)),
        ((0, 3, 3, 0), (0, 3, 3, 0)),
    )
    assert schedule.owner_channel_prefix == (
        ((0, 0, 0, 0), (0, 2, 2, 1)),
        ((0, 0, 0, 0), (0, 3, 3, 0)),
    )
    assert schedule.count == (
        (0, 2, 2, 4),
        (0, 6, 6, 0),
    )
    assert sum(sum(row) for row in schedule.count) == 20
    assert schedule.quota == (
        (0, 4, 4, 2),
        (0, 4, 4, 2),
    )
    assert schedule.keep_count == (
        (0, 2, 2, 2),
        (0, 4, 4, 0),
    )
    assert _segments(schedule) == (
        (), ((1, 0, 4, 2, 0),), ((1, 0, 4, 2, 0),),
        ((0, 1, 2, 2, 0),))
    assert schedule.num_segments == (0, 1, 1, 1)
    assert schedule.retained == (
        ((0, 2, 2, 1), (0, 0, 0, 1)),
        ((0, 3, 3, 0), (0, 1, 1, 0)),
    )
    assert schedule.moved == (
        ((0, 1, 1, 0), (0, 1, 1, 0)),
        ((0, 0, 0, 2), (0, 0, 0, 0)),
    )
    assert schedule.moved_channel_prefix == (
        ((0, 0, 0), (0, 1, 2), (0, 1, 2), (0, 0, 0)),
        ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 2, 2)),
    )
    assert schedule.group_prefix == (
        ((0, 0, 1, 2), (2, 2, 3, 4)),
        ((0, 0, 0, 0), (2, 2, 2, 2)),
    )
    assert schedule.moved_copies == 6
    assert schedule.proxy_required == (4, 2)

    remote = [route for route in result.routes if route.is_remote]
    retained = [route for route in remote if not route.moved]
    moved = [route for route in remote if route.moved]
    local = [route for route in result.routes if not route.is_remote]
    assert len(remote) == 20
    assert len(retained) == 14
    assert len(moved) == 6
    assert not local
    assert all(route.destination > 0 for route in result.routes)
    assert {
        (route.owner, route.token, route.destination, route.egress,
         route.target_channel, route.proxy_slot, route.remote_slot,
         route.dense_slot, route.reduce_row)
        for route in moved
    } == {
        (0, 3, 3, 1, 0, 0, 0, 0, 1),
        (0, 5, 3, 1, 0, 1, 1, 1, 1),
        (1, 3, 1, 0, 0, 0, 2, 2, 0),
        (1, 3, 2, 0, 0, 1, 2, 2, 1),
        (1, 5, 1, 0, 1, 2, 0, 3, 0),
        (1, 5, 2, 0, 1, 3, 0, 3, 1),
    }
    assert {route.ingress_physical for route in moved} == {2, 4, 7}
    assert {
        route.proxy_slot for route in moved if route.egress == 0
    } == {0, 1, 2, 3}
    assert {
        route.proxy_slot for route in moved if route.egress == 1
    } == {0, 1}
    assert any(
        route.moved and route.target_channel == 1 and
        schedule.moved_channel_prefix[
            route.egress][route.destination][route.target_channel] > 0 and
        schedule.group_prefix[
            route.egress][route.target_channel][route.destination] > 0
        for route in result.routes)
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

    # A moved repeated-destination copy selects the highest lane, while the
    # channel-one copies exercise nonzero moved and group prefixes.
    repeated = next(
        route for route in result.routes
        if route.owner == 0 and route.token == 3 and route.destination == 3)
    assert repeated.topk_lanes == (0, 1)
    assert repeated.reduce_row == 1
    assert repeated.moved
    assert _dense_occupancy(result) == {
        (0, 1, 0): (0, 0, False, -1),
        (0, 1, 1): (0, 2, False, -1),
        (0, 1, 2): (1, 3, True, 0),
        (0, 1, 3): (1, 5, True, 2),
        (0, 2, 0): (0, 0, False, -1),
        (0, 2, 1): (0, 2, False, -1),
        (0, 2, 2): (1, 3, True, 1),
        (0, 2, 3): (1, 5, True, 3),
        (0, 3, 0): (0, 4, False, -1),
        (0, 3, 1): (0, 1, False, -1),
        (1, 1, 0): (1, 0, False, -1),
        (1, 1, 1): (1, 2, False, -1),
        (1, 1, 2): (1, 4, False, -1),
        (1, 1, 3): (1, 1, False, -1),
        (1, 2, 0): (1, 0, False, -1),
        (1, 2, 1): (1, 2, False, -1),
        (1, 2, 2): (1, 4, False, -1),
        (1, 2, 3): (1, 1, False, -1),
        (1, 3, 0): (0, 3, True, 0),
        (1, 3, 1): (0, 5, True, 1),
    }
    assert {
        route.vnode_slot for route in result.routes
    } == {0, 1, 2, 3, 6, 7, 8, 9, 12, 13}
    assert result.combined == result.direct
    assert _combined_constants(case, result) == (
        (Fraction(20), Fraction(86), Fraction(28), Fraction(90),
         Fraction(76), Fraction(86)),
        (Fraction(34), Fraction(20), Fraction(30), Fraction(28),
         Fraction(34), Fraction(20)),
    )


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

    local = list(case.topk_idx[0][0])
    local[0] = 0
    owner_zero = list(case.topk_idx[0])
    owner_zero[0] = tuple(local)
    _expect_failure(lambda: run_vnode_roundtrip(replace(
        case,
        topk_idx=(tuple(owner_zero), case.topk_idx[1]),
    )))


def main() -> None:
    tests = (
        test_physical_rank_mapping_is_node_major,
        test_reduce_row_rank_and_topk_layouts,
        test_bfloat16_rounding_and_combine_order,
        test_rounding_sensitive_legacy_two_level_is_authoritative,
        test_4x2_d_le_k_complete_round_trip,
        test_2x4_d_gt_k_complete_round_trip,
        test_capacity_and_fixture_validation_fail_closed,
    )
    for test in tests:
        test()
    print(
        "PASS C080 vnode CPU oracle: 7 tests; remote-only 4x2 D<=K, "
        "remote-only 2x4 D>K, exact BF16 expert/reduction, retained/moved, "
        "two-level rounding, nonzero channel/group prefixes, dense vnode "
        "slots, proxy p, highest-lane combine/unshuffle",
        flush=True,
    )


if __name__ == "__main__":
    main()
