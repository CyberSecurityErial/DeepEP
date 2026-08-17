"""Merge measured Rail/FanIn sweet spots into an EchoP CPU lookup table."""

import argparse
from collections import defaultdict
import glob
import json
import statistics
import time

from deep_ep.utils.network_policy import (
    NetworkPolicy,
    NetworkPolicySelector,
    PolicyProfilePoint,
    extract_traffic_features,
)


POLICY_BY_VARIANT = {
    'native': NetworkPolicy.OFF,
    'rail_only': NetworkPolicy.RAIL,
    'fanin_only': NetworkPolicy.FANIN,
}


def _scenario_key(result):
    return (
        result.get('case'),
        result.get('tokens_parameter'),
        result.get('incast_total_mode'),
        result.get('alltoall_flow_shape', 'directed-zipf'),
        result.get('source_alpha'),
        result.get('rail_alpha'),
        result.get('rail_phase'),
        result.get('expert_alpha'),
        result.get('fan_in'),
        result.get('num_qps', 0),
    )


def _load_results(patterns):
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                if line.startswith('RESULT '):
                    result = json.loads(line[7:])
                    result['_path'] = path
                    yield result


def build_profile(results):
    grouped = defaultdict(lambda: defaultdict(list))
    exemplars = {}
    for result in results:
        policy = POLICY_BY_VARIANT.get(result.get('variant'))
        if policy is None:
            continue
        key = _scenario_key(result)
        grouped[key][policy].append(float(result['roundtrip_us']))
        exemplars[key] = result

    points = []
    rows = []
    required = set(NetworkPolicy)
    for key, samples in sorted(grouped.items(), key=lambda item: str(item[0])):
        if set(samples) != required:
            continue
        result = exemplars[key]
        qps = int(result.get('num_qps', 0)) or 129
        features = extract_traffic_features(
            result['demand'], result['source_rail_counts_by_node'],
            num_qps=qps)
        latencies = {
            policy: statistics.median(values)
            for policy, values in samples.items()
        }
        point = PolicyProfilePoint(features, latencies)
        winner = point.best_policy(min_gain=0.02)
        off_us = latencies[NetworkPolicy.OFF]
        winner_us = latencies[winner]
        points.append(point)
        rows.append({
            'scenario': {
                'case': key[0],
                'tokens_parameter': key[1],
                'flow_shape': key[3],
                'source_alpha': key[4],
                'rail_alpha': key[5],
                'rail_phase': key[6],
                'fan_in': key[8],
                'num_qps': qps,
            },
            'features': features.__dict__,
            'latency_us': {
                policy.value: latency
                for policy, latency in latencies.items()
            },
            'winner': winner.value,
            'speedup_over_off': off_us / winner_us,
        })
    return points, rows


def _selector_cost_us(selector, points, iterations=10000):
    samples = []
    current = NetworkPolicy.OFF
    for index in range(iterations):
        features = points[index % len(points)].features
        start = time.perf_counter_ns()
        current = selector.select(features, current=current)
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    return statistics.median(samples), sorted(samples)[int(0.99 * len(samples))]


def _markdown(rows, median_hook_us, p99_hook_us):
    lines = [
        '# EchoP Auto Policy Sweet-Spot Table',
        '',
        f'CPU lookup overhead: median {median_hook_us:.3f} us, '
        f'P99 {p99_hook_us:.3f} us (separately reported, asynchronous).',
        '',
        '| Size | Rail skew | Fan-in | Pair skew | QPs | Off us | Rail us | FanIn us | Auto | Speedup |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |',
    ]
    for row in rows:
        feature = row['features']
        latency = row['latency_us']
        lines.append(
            f"| {feature['total_tokens']} | "
            f"{feature['rail_max_over_mean']:.3f} | "
            f"{feature['max_node_fan_in']} | "
            f"{feature['pair_max_over_mean']:.3f} | "
            f"{feature['num_qps']} | {latency['off']:.3f} | "
            f"{latency['rail']:.3f} | {latency['fanin']:.3f} | "
            f"{row['winner']} | {row['speedup_over_off']:.3f}x |")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', help='log paths or glob patterns')
    parser.add_argument('--json-out', required=True)
    parser.add_argument('--markdown-out', required=True)
    args = parser.parse_args()

    points, rows = build_profile(_load_results(args.inputs))
    if not points:
        raise SystemExit('no complete off/rail/fanin scenario groups found')
    selector = NetworkPolicySelector(points, min_gain=0.02, switch_margin=0.02)
    median_hook_us, p99_hook_us = _selector_cost_us(selector, points)
    payload = {
        'schema': 1,
        'theory': 'Expert-to-Node-to-Rail hierarchical bottleneck selection',
        'selection': 'nearest calibrated feature row',
        'min_gain': 0.02,
        'switch_margin': 0.02,
        'cpu_hook_median_us': median_hook_us,
        'cpu_hook_p99_us': p99_hook_us,
        'rows': rows,
    }
    with open(args.json_out, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
    with open(args.markdown_out, 'w', encoding='utf-8') as handle:
        handle.write(_markdown(rows, median_hook_us, p99_hook_us))
    print(json.dumps({
        'complete_rows': len(rows),
        'cpu_hook_median_us': median_hook_us,
        'cpu_hook_p99_us': p99_hook_us,
    }))
