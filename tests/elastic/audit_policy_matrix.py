"""Audit policy variants on byte-for-byte identical benchmark traffic."""

from __future__ import annotations

import argparse
import csv
import json


TRAFFIC_FIELDS = (
    'remote_tokens_global', 'demand', 'source_node_counts',
    'source_rail_counts_by_node', 'target_expert_counts')


def load(path):
    results = {}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if not line.startswith('RESULT '):
                continue
            result = json.loads(line[7:])
            key = (
                result['tokens_parameter'], result['source_alpha'],
                result['rail_alpha'], result['expert_alpha'], result['fan_in'])
            if key in results:
                raise ValueError(f'duplicate point in {path}: {key}')
            results[key] = result
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+', metavar='NAME=PATH')
    parser.add_argument('--csv-out')
    parser.add_argument('--summary-out')
    args = parser.parse_args()
    named = {}
    for item in args.logs:
        name, path = item.split('=', 1)
        named[name] = load(path)
    reference_name = next(iter(named))
    reference = named[reference_name]
    if any(set(results) != set(reference) for results in named.values()):
        raise ValueError('point keys differ across variants')
    for key, expected in reference.items():
        for name, results in named.items():
            for field in TRAFFIC_FIELDS:
                if results[key].get(field) != expected.get(field):
                    raise ValueError(
                        f'traffic mismatch: point={key} variant={name} field={field}')

    rows = []
    for key in sorted(reference):
        latency = {
            name: float(results[key]['roundtrip_us'])
            for name, results in named.items()
        }
        row = dict(zip(
            ('size', 'source_alpha', 'rail_alpha', 'expert_alpha', 'fan_in'), key))
        row.update({f'{name}_us': value for name, value in latency.items()})
        if {'rail', 'matching', 'joint'} <= set(latency):
            best_single = min(latency['rail'], latency['matching'])
            row['best_single_us'] = best_single
            row['joint_over_best_single'] = best_single / latency['joint']
        rows.append(row)
    summary = {'points': len(rows), 'traffic_equal': True}
    if rows and 'joint_over_best_single' in rows[0]:
        ratios = [row['joint_over_best_single'] for row in rows]
        summary.update({
            'joint_wins': sum(ratio > 1 for ratio in ratios),
            'joint_wins_gt_3pct': sum(ratio > 1.03 for ratio in ratios),
            'joint_wins_gt_5pct': sum(ratio > 1.05 for ratio in ratios),
            'joint_ratio_min': min(ratios),
            'joint_ratio_median': sorted(ratios)[len(ratios) // 2],
            'joint_ratio_max': max(ratios),
            'aggregate_best_single_over_joint': (
                sum(row['best_single_us'] for row in rows) /
                sum(row['joint_us'] for row in rows)),
        })
    print('SUMMARY ' + json.dumps(summary, sort_keys=True))
    for row in rows:
        print('ROW ' + json.dumps(row, sort_keys=True))
    if args.csv_out:
        with open(args.csv_out, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    if args.summary_out:
        with open(args.summary_out, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write('\n')


if __name__ == '__main__':
    main()
