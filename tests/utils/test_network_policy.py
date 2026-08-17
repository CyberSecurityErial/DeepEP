import pytest

from deep_ep.utils.network_policy import (
    NetworkPolicy,
    NetworkPolicySelector,
    PolicyProfilePoint,
    TrafficFeatures,
    extract_traffic_features,
)


def test_extract_traffic_features_separates_rail_and_pair_skew():
    demand = [
        [0, 80, 20, 0],
        [20, 0, 80, 0],
        [80, 20, 0, 0],
        [0, 0, 0, 0],
    ]
    rails = [
        [80, 20],
        [50, 50],
        [50, 50],
        [0, 0],
    ]
    features = extract_traffic_features(demand, rails, num_qps=17)
    assert features.total_tokens == 300
    assert features.rail_max_over_mean == pytest.approx(1.6)
    assert features.max_node_fan_in == 2
    assert features.pair_max_over_mean == pytest.approx(1.6)


def _features(rail_skew, fan_in, pair_skew, qps=129):
    return TrafficFeatures(
        total_tokens=65536,
        rail_max_over_mean=rail_skew,
        max_node_fan_in=fan_in,
        pair_max_over_mean=pair_skew,
        num_qps=qps,
        num_nodes=4,
    )


def test_selector_merges_disjoint_sweet_spots():
    rail_point = PolicyProfilePoint(
        _features(5.7, 1, 1.0),
        {
            NetworkPolicy.OFF: 14000,
            NetworkPolicy.RAIL: 5300,
            NetworkPolicy.FANIN: 15000,
        },
    )
    fanin_point = PolicyProfilePoint(
        _features(1.0, 3, 2.0, qps=17),
        {
            NetworkPolicy.OFF: 9000,
            NetworkPolicy.RAIL: 9300,
            NetworkPolicy.FANIN: 7000,
        },
    )
    selector = NetworkPolicySelector([rail_point, fanin_point])
    assert selector.select(_features(5.5, 1, 1.0)) == NetworkPolicy.RAIL
    assert selector.select(
        _features(1.1, 3, 2.1, qps=17)) == NetworkPolicy.FANIN


def test_selector_keeps_off_without_minimum_gain_and_hysteresis_stops_flap():
    point = PolicyProfilePoint(
        _features(1.0, 3, 1.0),
        {
            NetworkPolicy.OFF: 1000,
            NetworkPolicy.RAIL: 995,
            NetworkPolicy.FANIN: 1005,
        },
    )
    selector = NetworkPolicySelector(
        [point], min_gain=0.02, switch_margin=0.02)
    assert selector.select(point.features) == NetworkPolicy.OFF

    faster = PolicyProfilePoint(
        point.features,
        {
            NetworkPolicy.OFF: 1000,
            NetworkPolicy.RAIL: 970,
            NetworkPolicy.FANIN: 965,
        },
    )
    selector = NetworkPolicySelector(
        [faster], min_gain=0.02, switch_margin=0.02)
    assert selector.select(
        point.features, current=NetworkPolicy.RAIL) == NetworkPolicy.RAIL
    decision = selector.decide(
        point.features, current=NetworkPolicy.RAIL)
    assert decision.policy == NetworkPolicy.RAIL
    assert decision.reference_distance == 0
    assert decision.predicted_gain_over_off == pytest.approx(1000 / 970 - 1)
    assert decision.reason.startswith('hysteresis:')
