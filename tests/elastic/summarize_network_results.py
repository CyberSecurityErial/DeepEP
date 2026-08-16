"""Merge EchoP RESULT JSONL logs and compute policy speedups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


KEYS = (
    'case', 'tokens_parameter', 'incast_total_mode', 'source_alpha',
    'rail_alpha', 'rail_phase', 'expert_alpha', 'fan_in', 'plan_state',
)
INVARIANTS = (
    'remote_tokens_global', 'payload_mib_global', 'demand',
    'source_node_counts', 'source_rail_counts_by_node',
    'source_to_sink_rail_counts_by_node', 'target_expert_counts',
)


def _lines(path: str):
    if path == '-':
        yield from sys.stdin
        return
    yield from Path(path).read_text().splitlines()


def _read(paths: list[str]) -> list[dict]:
    results = []
    for path in paths:
        for line in _lines(path):
            line = line.strip()
            if line.startswith('RESULT '):
                result = json.loads(line[len('RESULT '):])
                result['_source_log'] = path
                results.append(result)
    return results


def _value(result: dict, metric: str) -> float:
    if metric in result:
        return float(result[metric])
    if metric == 'roundtrip_plus_amortized_control_us':
        return float(result['ep_total_plus_amortized_control_us'])
    raise KeyError(f'missing metric {metric!r} in {result.get("_source_log")}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+')
    parser.add_argument(
        '--metric', default='roundtrip_plus_amortized_control_us')
    args = parser.parse_args()

    groups: dict[tuple, dict[str, dict]] = {}
    for result in _read(args.logs):
        key = tuple(result.get(field) for field in KEYS)
        policy = result['policy']
        if policy in groups.setdefault(key, {}):
            raise ValueError(f'duplicate policy {policy!r} for key {key!r}')
        groups[key][policy] = result

    fields = list(KEYS) + [
        'remote_tokens_global', 'payload_mib_global',
        'off_us', 'active_us', 'incast_us',
        'off_over_active', 'active_over_incast', 'best_baseline_over_incast',
        'incast_control_us_per_step', 'incast_moved_ratio',
        'incast_sink_rail_max_over_mean',
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        policies = groups[key]
        reference = next(iter(policies.values()))
        for invariant in INVARIANTS:
            expected = reference.get(invariant)
            for policy, result in policies.items():
                if result.get(invariant) != expected:
                    raise ValueError(
                        f'{invariant} changed across policies for {key}: '
                        f'{policy}')
        times = {
            policy: _value(result, args.metric)
            for policy, result in policies.items()
        }
        off = times.get('off')
        active = times.get('active')
        incast = times.get('incast')
        row = dict(zip(KEYS, key))
        row.update({
            'remote_tokens_global': reference.get('remote_tokens_global'),
            'payload_mib_global': reference.get('payload_mib_global'),
            'off_us': off,
            'active_us': active,
            'incast_us': incast,
            'off_over_active': (
                off / active if off is not None and active else None),
            'active_over_incast': (
                active / incast if active is not None and incast else None),
            'best_baseline_over_incast': (
                min(off, active) / incast
                if off is not None and active is not None and incast else None),
            'incast_control_us_per_step': (
                policies.get('incast', {}).get(
                    'control_amortized_us_per_step')),
            'incast_moved_ratio': (
                policies.get('incast', {}).get('predicted_moved_ratio')),
            'incast_sink_rail_max_over_mean': (
                policies.get('incast', {}).get(
                    'sink_rail_max_over_mean_after_policy')),
        })
        writer.writerow(row)


if __name__ == '__main__':
    main()
