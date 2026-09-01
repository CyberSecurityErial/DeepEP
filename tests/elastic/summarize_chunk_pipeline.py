"""Flatten chunk-interleaving benchmark logs into a reproducible CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


FIELDS = (
    'tokens_parameter', 'equivalent_input_tokens_per_gpu', 'source_alpha',
    'rail_alpha', 'policy', 'pairwise_planner', 'pairwise_execution',
    'pairwise_chunk_divisor', 'pairwise_chunk_tokens',
    'pairwise_num_waves', 'roundtrip_us', 'roundtrip_p99_us',
    'plan_cpu_median_max_us', 'control_amortized_us_per_step',
    'payload_mib_global', 'ep_total_gbps_global', '_source_log',
)


def _read(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.startswith('RESULT '):
                continue
            row = json.loads(line[len('RESULT '):])
            row['_source_log'] = path
            # The four-node experiment uses eight GPUs per node, Top-8, and
            # models the expected 75% cross-node share.  Thus one original
            # token/GPU contributes 8 * 8 * 0.75 = 48 remote route entries
            # per node.
            row['equivalent_input_tokens_per_gpu'] = (
                float(row['tokens_parameter']) / 48.0)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+')
    args = parser.parse_args()

    rows = sorted(
        _read(args.logs),
        key=lambda row: (
            row.get('tokens_parameter', 0), row.get('source_alpha', 0),
            row.get('rail_alpha', 0), row.get('pairwise_planner', ''),
            row.get('pairwise_execution', '')))
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for result in rows:
        writer.writerow({field: result.get(field) for field in FIELDS})


if __name__ == '__main__':
    main()
