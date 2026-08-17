import math

import pytest
import torch

from deep_ep.utils.incast import (
    plan_incast_rail_masks,
    plan_pairwise_incast_waves,
)
from deep_ep.buffers.elastic import _pack_incast_weighted_quotas


def _unsigned(mask: torch.Tensor) -> int:
    return int(mask.item()) & 0xFFFFFFFF


def _popcount(mask: torch.Tensor) -> int:
    return _unsigned(mask).bit_count()


def test_four_node_pairwise_waves_match_incast_free_schedule():
    assert plan_pairwise_incast_waves(4) == (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )


@pytest.mark.parametrize('num_nodes', [2, 3, 4, 5, 8, 16, 32, 64])
def test_pairwise_waves_cover_every_pair_once_with_fan_in_one(num_nodes):
    waves = plan_pairwise_incast_waves(num_nodes)
    expected_waves = num_nodes - 1 if num_nodes % 2 == 0 else num_nodes
    assert len(waves) == expected_waves

    pairs = []
    for wave in waves:
        participants = [node for pair in wave for node in pair]
        assert len(participants) == len(set(participants))
        pairs.extend(wave)

    assert len(pairs) == num_nodes * (num_nodes - 1) // 2
    assert len(set(pairs)) == len(pairs)
    assert set(pairs) == {
        (lhs, rhs)
        for lhs in range(num_nodes)
        for rhs in range(lhs + 1, num_nodes)
    }


@pytest.mark.parametrize(('num_nodes', 'budget'), [(4, 2), (5, 2), (8, 3)])
def test_pairwise_wave_grouping_respects_peer_budget(num_nodes, budget):
    waves = plan_pairwise_incast_waves(
        num_nodes, max_peers_per_wave=budget)
    factors = num_nodes - 1 if num_nodes % 2 == 0 else num_nodes
    assert len(waves) == math.ceil(factors / budget)

    all_pairs = []
    for wave in waves:
        degrees = [0] * num_nodes
        for lhs, rhs in wave:
            degrees[lhs] += 1
            degrees[rhs] += 1
        assert max(degrees) <= budget
        all_pairs.extend(wave)
    assert len(set(all_pairs)) == num_nodes * (num_nodes - 1) // 2


def test_pairwise_wave_epoch_rotates_rounds_without_changing_pairs():
    plan0 = plan_pairwise_incast_waves(4, epoch=0)
    plan1 = plan_pairwise_incast_waves(4, epoch=1)
    assert plan0 != plan1
    assert set(pair for wave in plan0 for pair in wave) == set(
        pair for wave in plan1 for pair in wave)


@pytest.mark.parametrize(
    ('args', 'error'),
    [
        ((1,), ValueError),
        ((True,), TypeError),
        ((4, 0), ValueError),
        ((4, True), TypeError),
    ],
)
def test_invalid_pairwise_wave_inputs(args, error):
    num_nodes = args[0]
    kwargs = {} if len(args) == 1 else {'max_peers_per_wave': args[1]}
    with pytest.raises(error):
        plan_pairwise_incast_waves(num_nodes, **kwargs)


def test_three_source_incast_gets_disjoint_full_rail_partition():
    demand = torch.zeros((4, 4), dtype=torch.float64)
    demand[:3, 3] = 100

    plan = plan_incast_rail_masks(demand, 8)
    masks = [_unsigned(plan[source, 3]) for source in range(3)]

    assert plan.dtype == torch.int32
    assert plan.device.type == 'cpu'
    assert sorted(mask.bit_count() for mask in masks) == [2, 3, 3]
    assert masks[0] | masks[1] | masks[2] == 0xFF
    assert masks[0] & masks[1] == 0
    assert masks[0] & masks[2] == 0
    assert masks[1] & masks[2] == 0
    assert torch.count_nonzero(torch.diag(plan)) == 0


def test_skewed_incast_allocates_more_rails_to_hotter_sources():
    demand = torch.zeros((4, 4), dtype=torch.float64)
    demand[0, 3] = 80
    demand[1, 3] = 20
    demand[2, 3] = 10

    plan = plan_incast_rail_masks(demand, 8)
    counts = [_popcount(plan[source, 3]) for source in range(3)]

    assert counts[0] > counts[1] >= counts[2] >= 1
    assert sum(counts) == 8


def test_bounded_overlap_preserves_more_source_rail_parallelism():
    demand = torch.zeros((4, 4), dtype=torch.float64)
    demand[:3, 3] = 100

    plan = plan_incast_rail_masks(demand, 8, rail_overlap=1.5)
    masks = [_unsigned(plan[source, 3]) for source in range(3)]
    memberships = [
        sum((mask >> rail) & 1 for mask in masks)
        for rail in range(8)
    ]

    assert [mask.bit_count() for mask in masks] == [4, 4, 4]
    assert sum(memberships) == 12
    assert max(memberships) == 2
    assert min(memberships) == 1


def test_weighted_overlap_two_has_exact_sink_balance():
    demand = torch.zeros((4, 4), dtype=torch.float64)
    demand[:3, 3] = 1024
    plan = plan_incast_rail_masks(demand, 8, rail_overlap=2.0)
    controls = [
        int(_pack_incast_weighted_quotas(plan, 8, source)[3].item())
        & 0xFFFFFFFF
        for source in range(3)
    ]
    weights = [
        [(control >> (4 * rail)) & 0xF for rail in range(8)]
        for control in controls
    ]

    assert [sum(row) for row in weights] == [48, 48, 48]
    assert all(sum(weight > 0 for weight in row) in (5, 6)
               for row in weights)
    assert all(sum(weights[source][rail] for source in range(3)) == 18
               for rail in range(8))
    assert all(sum(weights[source][rail] > 0 for source in range(3)) == 2
               for rail in range(8))


def test_epoch_rotates_identities_without_changing_allocations():
    demand = torch.zeros((4, 4), dtype=torch.float32)
    demand[:3, 3] = torch.tensor([80, 20, 10])

    plan0 = plan_incast_rail_masks(demand, 8, epoch=0)
    plan1 = plan_incast_rail_masks(demand, 8, epoch=1)

    assert not torch.equal(plan0, plan1)
    assert [
        _popcount(plan0[source, 3]) for source in range(3)
    ] == [
        _popcount(plan1[source, 3]) for source in range(3)
    ]


def test_more_sources_than_rails_uses_every_rail_and_limits_each_flow_to_one():
    demand = torch.zeros((7, 7), dtype=torch.float64)
    demand[:6, 6] = torch.tensor([60, 50, 40, 30, 20, 10])

    plan = plan_incast_rail_masks(demand, 2)
    masks = [_unsigned(plan[source, 6]) for source in range(6)]

    assert all(mask.bit_count() == 1 for mask in masks)
    assert set(masks) == {1, 2}


def test_zero_demand_and_32_rail_encoding():
    zero = plan_incast_rail_masks(torch.zeros((2, 2)), 8)
    assert torch.count_nonzero(zero) == 0

    demand = torch.zeros((2, 2))
    demand[0, 1] = 1
    plan = plan_incast_rail_masks(demand, 32)
    assert _unsigned(plan[0, 1]) == 0xFFFFFFFF


@pytest.mark.parametrize(
    ('demand', 'num_rails', 'error'),
    [
        (torch.zeros(2), 8, ValueError),
        (torch.zeros((2, 3)), 8, ValueError),
        (torch.tensor([[0.0, -1.0], [0.0, 0.0]]), 8, ValueError),
        (torch.tensor([[0.0, float('nan')], [0.0, 0.0]]), 8, ValueError),
        (torch.zeros((2, 2)), 0, ValueError),
        (torch.zeros((2, 2)), 33, ValueError),
    ],
)
def test_invalid_inputs(demand, num_rails, error):
    with pytest.raises(error):
        plan_incast_rail_masks(demand, num_rails)


@pytest.mark.parametrize('rail_overlap', [0.5, float('nan'), float('inf')])
def test_invalid_rail_overlap(rail_overlap):
    with pytest.raises(ValueError):
        plan_incast_rail_masks(
            torch.zeros((2, 2)), 8, rail_overlap=rail_overlap)
