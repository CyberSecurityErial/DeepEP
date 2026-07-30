"""Endpoint-aware vnode route and inverse-combine contracts."""

from __future__ import annotations

from fractions import Fraction

from rail_balance_hop_reference import HopPlannerConfig
from rail_balance_hybrid_vnode_reference import (
    VnodeRoundTripCase,
    VnodeTopology,
    run_hop_vnode_roundtrip,
    run_vnode_roundtrip,
)


def _case(name: str, owner_targets: tuple[tuple[tuple[int, ...], ...], ...]):
    topology = VnodeTopology(rails_per_node=4, num_nodes=2)
    num_topk = len(next(
        targets for owner in owner_targets for targets in owner
    ))
    experts_per_rank = 2
    topk_idx = tuple(
        tuple(
            tuple(
                topology.physical(1, target) * experts_per_rank + lane % 2
                for lane, target in enumerate(targets)
            )
            for targets in owner
        )
        for owner in owner_targets
    )
    weights = tuple(
        Fraction(1, num_topk) for _ in range(num_topk)
    )
    topk_weights = tuple(
        tuple(weights for _ in owner) for owner in topk_idx
    )
    source_values = tuple(
        tuple(
            tuple(Fraction(8 * (rank + token + column)) for column in range(3))
            for token in range(len(tokens))
        )
        for rank, tokens in enumerate(topk_idx)
    )
    return VnodeRoundTripCase(
        name=name,
        topology=topology,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        source_values=source_values,
        num_topk=num_topk,
        num_channels=1,
        num_max_tokens_per_rank=16,
        num_experts=topology.world_size * experts_per_rank,
        experts_per_physical_rank=experts_per_rank,
        proxy_capacity_per_egress=32,
        remainder_seed=0,
    )


def _config(mode: str, *, ratio: float = 0.0) -> HopPlannerConfig:
    return HopPlannerConfig(
        num_rails=4,
        num_destinations=2,
        mode=mode,
        chunk_size=1,
        max_two_hop_ratio=ratio,
    )


def test_direct_path_has_no_local_forward_and_combine_is_identical() -> None:
    case = _case("direct", ((((0,),) * 4), (), (), ()))
    result = run_hop_vnode_roundtrip(case, _config("one_hop"))

    assert {route.path_kind for route in result.routes} == {"direct"}
    assert all(route.source_forward is None for route in result.routes)
    assert all(not route.destination_forwards for route in result.routes)
    assert result.combined == run_vnode_roundtrip(case).combined


def test_one_hop_chooses_both_endpoint_paths_and_inverts_them() -> None:
    targets = tuple((target,) for target in (1, 2, 3) * 4)
    case = _case("offdiag", (targets, (), (), ()))
    result = run_hop_vnode_roundtrip(case, _config("one_hop"))

    kinds = {route.path_kind for route in result.routes}
    assert {"dst_forward", "src_forward"} <= kinds
    assert "two_hop" not in kinds
    for route in result.routes:
        assert route.egress in {route.owner, *route.targets}
        assert route.combine_destination_forwards == tuple(
            (target, ingress)
            for ingress, target in route.destination_forwards
        )
        if route.source_forward is None:
            assert route.combine_source_forward is None
        else:
            assert route.combine_source_forward == (
                route.source_forward[1], route.source_forward[0]
            )


def test_adaptive_diagonal_uses_bounded_two_hop_and_both_sides() -> None:
    case = _case("diagonal", ((((0,),) * 12), (), (), ()))
    result = run_hop_vnode_roundtrip(
        case, _config("adaptive", ratio=0.5)
    )
    two_hop = [route for route in result.routes if route.path_kind == "two_hop"]

    assert len(two_hop) == 6
    assert result.plan.two_hop_ratio == 0.5
    for route in two_hop:
        assert route.egress not in {route.owner, *route.targets}
        assert route.source_forward is not None
        assert len(route.destination_forwards) == 1
        assert route.combine_source_forward == (
            route.source_forward[1], route.source_forward[0]
        )


def test_multitarget_token_remains_one_network_payload() -> None:
    case = _case("multitarget", ((((1, 2),) * 8), (), (), ()))
    result = run_hop_vnode_roundtrip(case, _config("one_hop"))

    assert len(result.routes) == 8
    assert result.plan.total_units == 8
    assert all(route.targets == (1, 2) for route in result.routes)
    assert sum(len(route.destination_target_physicals) for route in result.routes) == 16
    assert result.combined == run_vnode_roundtrip(case).combined
