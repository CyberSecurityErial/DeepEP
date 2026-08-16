import pytest
import torch

from deep_ep.utils.incast import plan_incast_rail_masks


def _unsigned(mask: torch.Tensor) -> int:
    return int(mask.item()) & 0xFFFFFFFF


def _popcount(mask: torch.Tensor) -> int:
    return _unsigned(mask).bit_count()


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
