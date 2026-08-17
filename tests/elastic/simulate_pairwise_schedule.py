"""Simulate EchoP's topology-independent pairwise incast schedule."""

import argparse
import json
import math

from deep_ep.utils.incast import plan_pairwise_incast_waves


def _parse_positive_csv(value: str) -> list[int]:
    result = [int(item) for item in value.split(',')]
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError('expected positive comma-separated integers')
    return result


def simulate(num_nodes: int, peer_budget: int) -> dict[str, object]:
    waves = plan_pairwise_incast_waves(
        num_nodes, max_peers_per_wave=peer_budget)
    expected_pairs = num_nodes * (num_nodes - 1) // 2
    all_pairs = [pair for wave in waves for pair in wave]
    max_fan_in = 0
    active_fractions = []
    for wave in waves:
        degrees = [0] * num_nodes
        for lhs, rhs in wave:
            degrees[lhs] += 1
            degrees[rhs] += 1
        max_fan_in = max(max_fan_in, max(degrees))
        active_fractions.append(
            sum(degree > 0 for degree in degrees) / num_nodes)

    factors = num_nodes - 1 if num_nodes % 2 == 0 else num_nodes
    result = {
        'num_nodes': num_nodes,
        'peer_budget': peer_budget,
        'num_waves': len(waves),
        'theoretical_num_waves': math.ceil(
            factors / min(peer_budget, num_nodes - 1)),
        'num_pairs': len(all_pairs),
        'expected_pairs': expected_pairs,
        'unique_pair_coverage': len(set(all_pairs)) == expected_pairs,
        'max_node_fan_in_per_wave': max_fan_in,
        'simultaneous_alltoall_fan_in': num_nodes - 1,
        'fan_in_reduction': (num_nodes - 1) / max_fan_in,
        'mean_active_node_fraction': sum(active_fractions) / len(waves),
        'min_active_node_fraction': min(active_fractions),
    }
    assert result['unique_pair_coverage']
    assert result['num_pairs'] == expected_pairs
    assert result['num_waves'] == result['theoretical_num_waves']
    assert max_fan_in <= min(peer_budget, num_nodes - 1)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nodes', type=_parse_positive_csv,
        default=_parse_positive_csv('4,5,8,16,32,64'))
    parser.add_argument(
        '--peer-budgets', type=_parse_positive_csv,
        default=_parse_positive_csv('1,2'))
    args = parser.parse_args()

    results = [
        simulate(num_nodes, budget)
        for num_nodes in args.nodes
        if num_nodes >= 2
        for budget in args.peer_budgets
    ]
    for result in results:
        print('RESULT ' + json.dumps(result))

    print('\n| Nodes | Peer budget | Waves | Max fan-in | '
          'Fan-in reduction | Mean active nodes |')
    print('| ---: | ---: | ---: | ---: | ---: | ---: |')
    for result in results:
        print(
            f"| {result['num_nodes']} | {result['peer_budget']} | "
            f"{result['num_waves']} | {result['max_node_fan_in_per_wave']} | "
            f"{result['fan_in_reduction']:.1f}× | "
            f"{100 * result['mean_active_node_fraction']:.1f}% |")
