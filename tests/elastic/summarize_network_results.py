"""Merge EchoP RESULT JSONL logs and compute policy speedups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


KEYS = (
    'case', 'tokens_parameter', 'incast_total_mode',
    'alltoall_flow_shape', 'source_alpha',
    'rail_alpha', 'rail_phase', 'expert_alpha', 'fan_in', 'plan_state',
    'num_qps',
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


def _variant(result: dict) -> str:
    if 'variant' in result:
        return result['variant']
    policy = result['policy']
    if policy == 'off':
        return 'native'
    if policy == 'active':
        return 'rail_only'
    if policy == 'incast':
        return 'joint' if result.get('incast_weighted_quotas') else 'fanin_only'
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+')
    parser.add_argument(
        '--metric', default='roundtrip_us',
        help=(
            'primary latency metric; CPU control is asynchronous and is '
            'reported separately rather than added to GPU data-path latency'))
    args = parser.parse_args()

    groups: dict[tuple, dict[str, dict]] = {}
    for result in _read(args.logs):
        key = tuple(result.get(field) for field in KEYS)
        variant = _variant(result)
        if variant in groups.setdefault(key, {}):
            raise ValueError(f'duplicate variant {variant!r} for key {key!r}')
        groups[key][variant] = result

    fields = list(KEYS) + [
        'remote_tokens_global', 'payload_mib_global',
        'native_us', 'rail_only_us', 'fanin_only_us', 'joint_us',
        'native_over_rail_only', 'native_over_fanin_only',
        'native_over_joint', 'best_single_over_joint',
        'joint_beats_best_single', 'additive_margin_us',
        'multiplicative_margin_us', 'joint_control_us_per_step',
        'fanin_moved_ratio', 'joint_moved_ratio',
        'fanin_sink_rail_max_over_mean',
        'joint_sink_rail_max_over_mean',
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
        native = times.get('native')
        rail_only = times.get('rail_only')
        fanin_only = times.get('fanin_only')
        joint = times.get('joint')
        best_single = (
            min(rail_only, fanin_only)
            if rail_only is not None and fanin_only is not None else None)
        additive_expected = (
            rail_only + fanin_only - native
            if native is not None and rail_only is not None and
            fanin_only is not None else None)
        multiplicative_expected = (
            rail_only * fanin_only / native
            if native and rail_only is not None and fanin_only is not None
            else None)
        row = dict(zip(KEYS, key))
        row.update({
            'remote_tokens_global': reference.get('remote_tokens_global'),
            'payload_mib_global': reference.get('payload_mib_global'),
            'native_us': native,
            'rail_only_us': rail_only,
            'fanin_only_us': fanin_only,
            'joint_us': joint,
            'native_over_rail_only': (
                native / rail_only if native is not None and rail_only else None),
            'native_over_fanin_only': (
                native / fanin_only if native is not None and fanin_only else None),
            'native_over_joint': (
                native / joint if native is not None and joint else None),
            'best_single_over_joint': (
                best_single / joint if best_single is not None and joint else None),
            'joint_beats_best_single': (
                joint < best_single if joint is not None and
                best_single is not None else None),
            'additive_margin_us': (
                additive_expected - joint
                if additive_expected is not None and joint is not None else None),
            'multiplicative_margin_us': (
                multiplicative_expected - joint
                if multiplicative_expected is not None and joint is not None else None),
            'joint_control_us_per_step': (
                policies.get('joint', {}).get(
                    'control_amortized_us_per_step')),
            'fanin_moved_ratio': (
                policies.get('fanin_only', {}).get('predicted_moved_ratio')),
            'joint_moved_ratio': (
                policies.get('joint', {}).get('predicted_moved_ratio')),
            'fanin_sink_rail_max_over_mean': (
                policies.get('fanin_only', {}).get(
                    'sink_rail_max_over_mean_after_policy')),
            'joint_sink_rail_max_over_mean': (
                policies.get('joint', {}).get(
                    'sink_rail_max_over_mean_after_policy')),
        })
        writer.writerow(row)


if __name__ == '__main__':
    main()
