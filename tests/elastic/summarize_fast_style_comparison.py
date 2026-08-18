"""Audit and summarize FAST-style versus EchoP on identical traffic."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import glob
import json
from pathlib import Path


VARIANTS = (
    'native', 'rail_only', 'fanin_only', 'joint', 'fast_style_global')
DIRECT_VARIANTS = ('echop_local_matching', 'fast_style_global')
TRAFFIC_FIELDS = (
    'remote_tokens_global', 'demand', 'source_node_counts',
    'source_rail_counts_by_node', 'target_expert_counts')


def _load(patterns):
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                if line.startswith('RESULT '):
                    result = json.loads(line[7:])
                    result['_path'] = path
                    yield result


def _key(result):
    return (
        result.get('case'), result.get('tokens_parameter'),
        result.get('source_alpha'), result.get('rail_alpha'),
        result.get('expert_alpha'), result.get('fan_in'),
        result.get('rail_phase'),
        result.get('alltoall_flow_shape', 'directed-zipf'))


def summarize(results):
    groups = defaultdict(dict)
    duplicates = []
    for result in results:
        variant = result.get('variant')
        if variant not in VARIANTS:
            continue
        key = _key(result)
        if variant in groups[key]:
            duplicates.append((key, variant))
        groups[key][variant] = result
    if duplicates:
        raise ValueError(f'duplicate results: {duplicates[:3]}')

    rows = []
    incomplete = []
    for key, variants in sorted(groups.items(), key=lambda item: str(item[0])):
        # Existing EchoP logs may contain a larger calibration matrix than a
        # FAST-style scout.  Only keys with a FAST result are comparisons;
        # once FAST is present, require every EchoP ablation for that key.
        if 'fast_style_global' not in variants:
            continue
        missing = sorted(set(VARIANTS) - set(variants))
        if missing:
            incomplete.append((key, missing))
            continue
        reference = variants['native']
        for variant in VARIANTS[1:]:
            candidate = variants[variant]
            for field in TRAFFIC_FIELDS:
                if candidate.get(field) != reference.get(field):
                    raise ValueError(
                        f'traffic mismatch key={key} variant={variant} '
                        f'field={field}')

        latency = {
            variant: float(variants[variant]['roundtrip_us'])
            for variant in VARIANTS
        }
        echop_variants = VARIANTS[:-1]
        echop_best = min(echop_variants, key=latency.get)
        fast_us = latency['fast_style_global']
        echop_us = latency[echop_best]
        rows.append({
            'source_alpha': key[2],
            'rail_alpha': key[3],
            'native_us': latency['native'],
            'rail_only_us': latency['rail_only'],
            'matching_only_us': latency['fanin_only'],
            'joint_us': latency['joint'],
            'fast_style_us': fast_us,
            'echop_best_policy': echop_best,
            'echop_best_us': echop_us,
            'echop_over_fast_speedup': fast_us / echop_us,
            'echop_vs_fast_latency_reduction_pct':
                100.0 * (fast_us - echop_us) / fast_us,
            'fast_plan_cpu_us': float(
                variants['fast_style_global']['plan_cpu_median_max_us']),
            'fast_num_waves': int(
                variants['fast_style_global']['pairwise_num_waves']),
            'global_demand_bytes_u32': int(
                variants['fast_style_global']['global_demand_bytes_u32']),
        })
    if incomplete:
        raise ValueError(f'incomplete groups: {incomplete[:3]}')
    if not rows:
        raise ValueError('no complete FAST-style/EchoP groups found')
    return rows


def summarize_direct(results):
    groups = defaultdict(dict)
    for result in results:
        variant = result.get('variant')
        if variant not in DIRECT_VARIANTS:
            continue
        key = _key(result)
        if variant in groups[key]:
            raise ValueError(f'duplicate result: {key} {variant}')
        groups[key][variant] = result
    rows = []
    for key, variants in sorted(groups.items(), key=lambda item: str(item[0])):
        if set(variants) != set(DIRECT_VARIANTS):
            raise ValueError(f'incomplete direct comparison: {key}')
        local = variants['echop_local_matching']
        fast = variants['fast_style_global']
        for field in TRAFFIC_FIELDS:
            if local.get(field) != fast.get(field):
                raise ValueError(f'traffic mismatch: {key} {field}')
        local_us = float(local['roundtrip_us'])
        fast_us = float(fast['roundtrip_us'])
        local_plan_us = float(local['plan_cpu_median_max_us'])
        fast_plan_us = float(fast['plan_cpu_median_max_us'])
        rows.append({
            'source_alpha': key[2], 'rail_alpha': key[3],
            'echop_local_us': local_us, 'fast_style_us': fast_us,
            'fast_over_echop': fast_us / local_us,
            'echop_plan_cpu_us': local_plan_us,
            'fast_plan_cpu_us': fast_plan_us,
            'fast_over_echop_plan': fast_plan_us / local_plan_us,
            'traffic_equal': True,
        })
    if not rows:
        raise ValueError('no direct EchoP/FAST-style groups found')
    return rows


def write_direct_markdown(path, rows):
    latency_ratio = (
        sum(row['fast_style_us'] for row in rows) /
        sum(row['echop_local_us'] for row in rows))
    plan_ratios = sorted(row['fast_over_echop_plan'] for row in rows)
    lines = [
        '# EchoP source-local matching vs FAST-style global planning', '',
        'Both planners use the same DeepEP data plane. FAST-style is an '
        'independent implementation of the published scheduling idea, not '
        'the official FAST runtime.', '',
        f'- Complete, traffic-identical points: {len(rows)}',
        f'- Aggregate FAST-style / EchoP GPU latency: {latency_ratio:.4f}x',
        f'- Median FAST-style / EchoP CPU planning time: '
        f'{plan_ratios[len(plan_ratios) // 2]:.3f}x', '',
        '| Node-pair alpha | Rail alpha | EchoP us | FAST-style us | '
        'FAST/EchoP | FAST/EchoP plan |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            f"| {row['source_alpha']:.2f} | {row['rail_alpha']:.2f} | "
            f"{row['echop_local_us']:.3f} | {row['fast_style_us']:.3f} | "
            f"{row['fast_over_echop']:.4f}x | "
            f"{row['fast_over_echop_plan']:.3f}x |")
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    wins = [row for row in rows if row['echop_best_us'] < row['fast_style_us']]
    losses = [row for row in rows if row['echop_best_us'] > row['fast_style_us']]
    weighted_fast = sum(row['fast_style_us'] for row in rows)
    weighted_echop = sum(row['echop_best_us'] for row in rows)
    lines = [
        '# Experiment 8: FAST-style global planning vs EchoP', '',
        'This is an independent, dependency-free reimplementation of the '
        'published FAST planning idea, not the official FAST runtime.', '',
        f'- Complete traffic points: {len(rows)}',
        f'- EchoP faster points: {len(wins)}',
        f'- FAST-style faster points: {len(losses)}',
        f'- Aggregate FAST-style / EchoP speedup: '
        f'{weighted_fast / weighted_echop:.3f}x', '',
        '| Node-pair alpha | Rail alpha | FAST-style us | EchoP best | '
        'EchoP us | FAST/EchoP |',
        '| ---: | ---: | ---: | --- | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            f"| {row['source_alpha']:.2f} | {row['rail_alpha']:.2f} | "
            f"{row['fast_style_us']:.3f} | {row['echop_best_policy']} | "
            f"{row['echop_best_us']:.3f} | "
            f"{row['echop_over_fast_speedup']:.3f}x |")
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--csv-out', required=True)
    parser.add_argument('--markdown-out', required=True)
    parser.add_argument('--direct', action='store_true')
    args = parser.parse_args()
    rows = (summarize_direct(_load(args.inputs)) if args.direct
            else summarize(_load(args.inputs)))
    write_csv(args.csv_out, rows)
    if args.direct:
        write_direct_markdown(args.markdown_out, rows)
        print(json.dumps({
            'points': len(rows),
            'aggregate_fast_over_echop': (
                sum(row['fast_style_us'] for row in rows) /
                sum(row['echop_local_us'] for row in rows)),
            'median_fast_over_echop_plan': sorted(
                row['fast_over_echop_plan'] for row in rows)[len(rows) // 2],
        }, sort_keys=True))
        return
    write_markdown(args.markdown_out, rows)
    print(json.dumps({
        'points': len(rows),
        'echop_faster_points': sum(
            row['echop_best_us'] < row['fast_style_us'] for row in rows),
        'aggregate_fast_over_echop': (
            sum(row['fast_style_us'] for row in rows) /
            sum(row['echop_best_us'] for row in rows)),
    }, sort_keys=True))


if __name__ == '__main__':
    main()
