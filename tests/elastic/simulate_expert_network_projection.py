"""Show that globally balanced experts do not imply balanced source Rails.

The construction is intentionally stronger than the minimum counterexample:
every source node sends the same number of tokens, every destination node
receives the same number, and every one of the global experts receives the
same number.  Only the physical source-GPU/Rail histogram is changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics


def proportional_counts(total: int, weights: list[float]) -> list[int]:
    exact = [total * weight / sum(weights) for weight in weights]
    counts = [math.floor(value) for value in exact]
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    assert sum(counts) == total
    return counts


def zipf_counts(total: int, alpha: float, size: int) -> list[int]:
    return proportional_counts(
        total, [(index + 1) ** (-alpha) for index in range(size)])


def max_mean(values: list[int]) -> float:
    return max(values) / statistics.fmean(values)


def coefficient_of_variation(values: list[int]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean


def simulate(
        num_nodes: int, rails_per_node: int, tokens_per_node: int,
        alpha: float, seed: int = 20260818) -> dict[str, object]:
    num_experts = num_nodes * rails_per_node
    remote_experts_per_source = (num_nodes - 1) * rails_per_node
    if tokens_per_node % remote_experts_per_source:
        raise ValueError(
            'tokens-per-node must be divisible by all remote experts so the '
            'global expert histogram can be exactly uniform')

    source_rail_counts = [[0] * rails_per_node for _ in range(num_nodes)]
    expert_counts = [0] * num_experts
    node_pair_counts = [[0] * num_nodes for _ in range(num_nodes)]
    contribution = tokens_per_node // remote_experts_per_source

    for source_node in range(num_nodes):
        targets = [
            destination_node * rails_per_node + destination_expert
            for destination_node in range(num_nodes)
            if destination_node != source_node
            for destination_expert in range(rails_per_node)
            for _ in range(contribution)
        ]
        assert len(targets) == tokens_per_node
        random.Random(seed + source_node).shuffle(targets)

        rail_counts = zipf_counts(tokens_per_node, alpha, rails_per_node)
        source_rail_counts[source_node] = rail_counts
        begin = 0
        for count in rail_counts:
            for target in targets[begin:begin + count]:
                expert_counts[target] += 1
                node_pair_counts[source_node][target // rails_per_node] += 1
            begin += count
        assert begin == tokens_per_node

    node_egress = [sum(row) for row in node_pair_counts]
    node_ingress = [
        sum(node_pair_counts[source][destination]
            for source in range(num_nodes))
        for destination in range(num_nodes)
    ]
    flat_rails = [count for row in source_rail_counts for count in row]
    expected_expert_count = (
        (num_nodes - 1) * contribution)

    audit = {
        'all_global_experts_exactly_equal': (
            min(expert_counts) == max(expert_counts) == expected_expert_count),
        'all_node_egress_exactly_equal': (
            min(node_egress) == max(node_egress) == tokens_per_node),
        'all_node_ingress_exactly_equal': (
            min(node_ingress) == max(node_ingress) == tokens_per_node),
        'all_traffic_is_remote': all(
            node_pair_counts[node][node] == 0 for node in range(num_nodes)),
        'token_conservation': (
            sum(expert_counts) == num_nodes * tokens_per_node ==
            sum(flat_rails)),
    }
    assert all(audit.values())

    return {
        'alpha': alpha,
        'global_expert_max_mean': max_mean(expert_counts),
        'global_expert_cv': coefficient_of_variation(expert_counts),
        'node_ingress_max_mean': max_mean(node_ingress),
        'node_egress_max_mean': max_mean(node_egress),
        'source_rail_max_mean': max_mean(flat_rails),
        'source_rail_cv': coefficient_of_variation(flat_rails),
        'hottest_source_rail_fraction': max(flat_rails) / tokens_per_node,
        'global_expert_counts': expert_counts,
        'node_ingress_counts': node_ingress,
        'node_egress_counts': node_egress,
        'source_rail_counts': source_rail_counts,
        'audit': audit,
    }


def parse_alphas(value: str) -> list[float]:
    alphas = [float(item) for item in value.split(',')]
    if not alphas or any(alpha < 0 or not math.isfinite(alpha)
                         for alpha in alphas):
        raise argparse.ArgumentTypeError('alphas must be finite and non-negative')
    return alphas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=4)
    parser.add_argument('--rails-per-node', type=int, default=8)
    parser.add_argument('--tokens-per-node', type=int, default=24576)
    parser.add_argument(
        '--alphas', type=parse_alphas,
        default=parse_alphas('0,0.75,1.5,2.25'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.nodes < 2 or args.rails_per_node < 1 or args.tokens_per_node < 1:
        parser.error('nodes >= 2, rails >= 1 and tokens > 0 are required')

    results = [
        simulate(
            args.nodes, args.rails_per_node, args.tokens_per_node, alpha)
        for alpha in args.alphas
    ]
    payload = {
        'definition': (
            'Global expert balance means every one of all global experts '
            'receives exactly the same number of tokens.'),
        'construction': (
            'Every source and destination node has equal total traffic; only '
            'the source GPU/Rail marginal follows Zipf(alpha).'),
        'config': {
            'num_nodes': args.nodes,
            'rails_per_node': args.rails_per_node,
            'num_global_experts': args.nodes * args.rails_per_node,
            'tokens_per_node': args.tokens_per_node,
            'alphas': args.alphas,
        },
        'results': results,
    }

    for result in results:
        print('RESULT ' + json.dumps({
            key: value for key, value in result.items()
            if key not in (
                'global_expert_counts', 'node_ingress_counts',
                'node_egress_counts', 'source_rail_counts')
        }, sort_keys=True))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / 'expert_vs_network.json'
        csv_path = args.output_dir / 'expert_vs_network.csv'
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
        with csv_path.open('w', newline='', encoding='utf-8') as handle:
            fieldnames = [
                'alpha', 'global_expert_max_mean', 'global_expert_cv',
                'node_ingress_max_mean', 'node_egress_max_mean',
                'source_rail_max_mean', 'source_rail_cv',
                'hottest_source_rail_fraction',
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow({key: result[key] for key in fieldnames})


if __name__ == '__main__':
    main()
