"""Model FAST global-control scaling from its published scheduler timings.

This is a control-plane simulation, not a reproduction of FAST data-plane
throughput.  Published points are kept separate from interpolation and
out-of-range projection so plots cannot silently present modeled values as
measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FAST_PUBLISHED_PLAN_US = {
    4: 25.0,
    8: 221.0,
    12: 805.0,
    40: 77_000.0,
}
FAST_SOURCE = 'https://homes.cs.washington.edu/~arvind/papers/fast-yiran.pdf'


def parse_positive_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(',')]
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError('expected positive comma-separated ints')
    return result


def parse_positive_floats(value: str) -> list[float]:
    result = [float(item) for item in value.split(',')]
    if not result or any(item <= 0 or not math.isfinite(item)
                         for item in result):
        raise argparse.ArgumentTypeError(
            'expected positive finite comma-separated floats')
    return result


def fit_power_law(points: dict[int, float]) -> tuple[float, float]:
    xs = [math.log(node) for node in points]
    ys = [math.log(value) for value in points.values()]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    exponent = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / sum((x - x_mean) ** 2 for x in xs)
    coefficient = math.exp(y_mean - exponent * x_mean)
    return coefficient, exponent


def interpolate_published(node: int) -> float:
    """Log-log interpolate inside the measured domain."""
    anchors = sorted(FAST_PUBLISHED_PLAN_US)
    for left, right in zip(anchors, anchors[1:]):
        if left <= node <= right:
            fraction = (
                math.log(node / left) / math.log(right / left))
            return math.exp(
                math.log(FAST_PUBLISHED_PLAN_US[left]) * (1 - fraction) +
                math.log(FAST_PUBLISHED_PLAN_US[right]) * fraction)
    raise ValueError('interpolation requested outside published domain')


def plan_time(node: int, exponent: float) -> tuple[float, str]:
    if node in FAST_PUBLISHED_PLAN_US:
        return FAST_PUBLISHED_PLAN_US[node], 'published_measurement'
    if min(FAST_PUBLISHED_PLAN_US) < node < max(FAST_PUBLISHED_PLAN_US):
        return interpolate_published(node), 'log_interpolation'
    if node > max(FAST_PUBLISHED_PLAN_US):
        anchor_node = max(FAST_PUBLISHED_PLAN_US)
        anchor_us = FAST_PUBLISHED_PLAN_US[anchor_node]
        return (
            anchor_us * (node / anchor_node) ** exponent,
            'out_of_range_empirical_projection')
    coefficient, _ = fit_power_law(FAST_PUBLISHED_PLAN_US)
    return coefficient * node ** exponent, 'out_of_range_empirical_projection'


def simulate(
        node: int, gpus_per_node: int, bytes_per_count: int,
        bandwidth_gbps: float, collective_stage_us: float,
        traffic_epochs_ms: list[float], exponent: float) -> dict[str, object]:
    num_gpus = node * gpus_per_node

    # Lower-bound state model: one count for every GPU-to-GPU demand cell.
    # If an implementation tracks multiple experts per destination GPU, its
    # actual matrix is larger, so this deliberately favors FAST.
    global_count_cells = num_gpus * num_gpus
    matrix_bytes_per_gpu = global_count_cells * bytes_per_count
    echop_source_local_bytes = gpus_per_node * bytes_per_count

    planner_us, evidence = plan_time(node, exponent)

    # Optimistic synchronization floor: perfect tree, full link bandwidth,
    # no contention, and no straggler.  It is not claimed as a measurement.
    transfer_floor_us = (
        matrix_bytes_per_gpu * 8 / (bandwidth_gbps * 1_000))
    tree_stages = math.ceil(math.log2(node)) if node > 1 else 0
    synchronization_floor_us = (
        transfer_floor_us + tree_stages * collective_stage_us)
    reaction_us = planner_us + synchronization_floor_us

    result: dict[str, object] = {
        'num_nodes': node,
        'num_gpus': num_gpus,
        'timing_evidence': evidence,
        'fast_plan_us': planner_us,
        'fast_gpu_demand_cells': global_count_cells,
        'fast_matrix_bytes_per_gpu': matrix_bytes_per_gpu,
        'fast_sync_optimistic_floor_us': synchronization_floor_us,
        'fast_reaction_optimistic_us': reaction_us,
        'echop_rail_local_count_cells': gpus_per_node,
        'echop_rail_local_state_bytes': echop_source_local_bytes,
        'fast_to_echop_state_ratio': (
            matrix_bytes_per_gpu / echop_source_local_bytes),
        'echop_global_sync_bytes': 0,
    }
    for epoch_ms in traffic_epochs_ms:
        result[f'reaction_fraction_of_{epoch_ms:g}ms_epoch'] = min(
            1.0, reaction_us / (epoch_ms * 1_000))
    if node >= 40:
        # This is an algorithmic O(N^5) stress envelope anchored at the last
        # published timing, not an expected runtime prediction.
        result['o_n5_stress_envelope_us'] = (
            FAST_PUBLISHED_PLAN_US[40] * (node / 40) ** 5)
    else:
        result['o_n5_stress_envelope_us'] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nodes', type=parse_positive_ints,
        default=parse_positive_ints('4,8,12,16,20,32,40,64,80,128'))
    parser.add_argument('--gpus-per-node', type=int, default=8)
    parser.add_argument('--bytes-per-count', type=int, default=4)
    parser.add_argument('--bandwidth-gbps', type=float, default=400.0)
    parser.add_argument('--collective-stage-us', type=float, default=2.0)
    parser.add_argument(
        '--traffic-epochs-ms', type=parse_positive_floats,
        default=parse_positive_floats('100,250,500'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.gpus_per_node < 1 or args.bytes_per_count < 1:
        parser.error('gpus-per-node and bytes-per-count must be positive')

    coefficient, exponent = fit_power_law(FAST_PUBLISHED_PLAN_US)
    results = [
        simulate(
            node, args.gpus_per_node, args.bytes_per_count,
            args.bandwidth_gbps, args.collective_stage_us,
            args.traffic_epochs_ms, exponent)
        for node in args.nodes
    ]
    payload = {
        'scope': 'control-plane scaling simulation only',
        'source': FAST_SOURCE,
        'published_fast_plan_us': FAST_PUBLISHED_PLAN_US,
        'assumptions': {
            'global_state': (
                'one 32-bit count per GPU-to-GPU demand cell; this is a '
                'lower bound when multiple experts reside on a GPU'),
            'sync_model': (
                'optimistic binary tree at full configured bandwidth, no '
                'contention and no straggler'),
            'projection': (
                'power-law fit is used only beyond 40 nodes and is never '
                'labeled as a measurement'),
            'echop': (
                'RailBalance keeps one local count per source Rail and has '
                'no global synchronization'),
        },
        'fit': {
            'coefficient': coefficient,
            'exponent': exponent,
        },
        'config': {
            'nodes': args.nodes,
            'gpus_per_node': args.gpus_per_node,
            'bytes_per_count': args.bytes_per_count,
            'bandwidth_gbps': args.bandwidth_gbps,
            'collective_stage_us': args.collective_stage_us,
            'traffic_epochs_ms': args.traffic_epochs_ms,
        },
        'results': results,
    }

    for result in results:
        print('RESULT ' + json.dumps(result, sort_keys=True))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'fast_control_scaling.json').write_text(
            json.dumps(payload, indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
        with (args.output_dir / 'fast_control_scaling.csv').open(
                'w', newline='', encoding='utf-8') as handle:
            fieldnames = list(results[0])
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


if __name__ == '__main__':
    main()
