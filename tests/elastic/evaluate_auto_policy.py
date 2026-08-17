"""Evaluate a frozen EchoP policy table on held-out benchmark logs."""

import argparse
from collections import defaultdict
import glob
import json
import statistics

from deep_ep.utils.network_policy import (
    NetworkPolicy,
    NetworkPolicySelector,
    PolicyProfilePoint,
    TrafficFeatures,
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
                    yield json.loads(line[7:])


def _load_selector(path):
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    points = []
    for row in payload['rows']:
        points.append(PolicyProfilePoint(
            features=TrafficFeatures(**row['features']),
            latency_us={
                NetworkPolicy(policy): float(latency)
                for policy, latency in row['latency_us'].items()
            },
        ))
    return NetworkPolicySelector(
        points,
        min_gain=float(payload.get('min_gain', 0.02)),
        switch_margin=float(payload.get('switch_margin', 0.02))), payload


def evaluate(selector, results):
    samples = defaultdict(lambda: defaultdict(list))
    exemplars = {}
    for result in results:
        policy = POLICY_BY_VARIANT.get(result.get('variant'))
        if policy is None:
            continue
        key = _scenario_key(result)
        samples[key][policy].append(float(result['roundtrip_us']))
        exemplars[key] = result

    rows = []
    required = set(NetworkPolicy)
    for key, policy_samples in sorted(samples.items(), key=lambda item: str(item[0])):
        if set(policy_samples) != required:
            continue
        result = exemplars[key]
        qps = int(result.get('num_qps', 0)) or 129
        features = extract_traffic_features(
            result['demand'], result['source_rail_counts_by_node'],
            num_qps=qps)
        latency = {
            policy: statistics.median(values)
            for policy, values in policy_samples.items()
        }
        decision = selector.decide(features)
        oracle_policy = min(latency, key=latency.get)
        rows.append({
            'scenario': {
                'case': key[0],
                'tokens_parameter': key[1],
                'source_alpha': key[4],
                'rail_alpha': key[5],
                'rail_phase': key[6],
                'num_qps': qps,
            },
            'features': features.__dict__,
            'latency_us': {
                policy.value: value for policy, value in latency.items()
            },
            'auto': decision.policy.value,
            'oracle': oracle_policy.value,
            'auto_regret': (
                latency[decision.policy] / latency[oracle_policy] - 1.0),
            'reference_distance': decision.reference_distance,
            'reason': decision.reason,
        })
    if not rows:
        raise ValueError('no complete held-out off/rail/fanin groups found')

    totals = {
        policy: sum(row['latency_us'][policy.value] for row in rows)
        for policy in NetworkPolicy
    }
    auto_total = sum(row['latency_us'][row['auto']] for row in rows)
    oracle_total = sum(row['latency_us'][row['oracle']] for row in rows)
    summary = {
        'points': len(rows),
        'auto_total_us': auto_total,
        'oracle_total_us': oracle_total,
        'auto_regret': auto_total / oracle_total - 1.0,
        'auto_oracle_match_rate': (
            sum(row['auto'] == row['oracle'] for row in rows) / len(rows)),
        'auto_within_gain_gate_rate': (
            sum(row['auto_regret'] <= selector.min_gain for row in rows) /
            len(rows)),
        'fixed_total_us': {
            policy.value: total for policy, total in totals.items()
        },
        'auto_speedup_over_fixed': {
            policy.value: total / auto_total
            for policy, total in totals.items()
        },
    }
    return rows, summary


def _markdown(rows, summary):
    lines = [
        '# EchoP Auto Held-out Evaluation',
        '',
        '| Case | Size | Rail skew | Fan-in | Pair skew | QPs | Off us | Rail us | FanIn us | Auto | Oracle | Regret |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |',
    ]
    for row in rows:
        scenario = row['scenario']
        feature = row['features']
        latency = row['latency_us']
        lines.append(
            f"| {scenario['case']} | {scenario['tokens_parameter']} | "
            f"{feature['rail_max_over_mean']:.3f} | "
            f"{feature['max_node_fan_in']} | "
            f"{feature['pair_max_over_mean']:.3f} | "
            f"{feature['num_qps']} | {latency['off']:.3f} | "
            f"{latency['rail']:.3f} | {latency['fanin']:.3f} | "
            f"{row['auto']} | {row['oracle']} | "
            f"{100 * row['auto_regret']:.2f}% |")
    lines.extend([
        '',
        f"Auto/oracle aggregate regret: {100 * summary['auto_regret']:.2f}%.",
        f"Oracle match rate: {100 * summary['auto_oracle_match_rate']:.1f}%.",
        f"Within gain gate: "
        f"{100 * summary['auto_within_gain_gate_rate']:.1f}%.",
        '',
        '| Fixed policy | Total us | Fixed/Auto |',
        '| --- | ---: | ---: |',
    ])
    for policy in NetworkPolicy:
        lines.append(
            f"| {policy.value} | "
            f"{summary['fixed_total_us'][policy.value]:.3f} | "
            f"{summary['auto_speedup_over_fixed'][policy.value]:.3f}x |")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--heldout', nargs='+', required=True)
    parser.add_argument('--json-out', required=True)
    parser.add_argument('--markdown-out', required=True)
    args = parser.parse_args()

    selector, calibration = _load_selector(args.calibration)
    rows, summary = evaluate(selector, _load_results(args.heldout))
    payload = {
        'schema': 1,
        'calibration': args.calibration,
        'calibration_rows': len(calibration['rows']),
        'rows': rows,
        'summary': summary,
    }
    with open(args.json_out, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
    with open(args.markdown_out, 'w', encoding='utf-8') as handle:
        handle.write(_markdown(rows, summary))
    print(json.dumps(summary))
