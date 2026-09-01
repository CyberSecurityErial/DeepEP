#!/usr/bin/env python3
"""Flatten rail_balance_bench key=value logs into one CSV."""

import argparse
import csv
from pathlib import Path
import sys


FIELDS = (
    "engine", "launch", "measurement", "traffic", "operation", "sm",
    "record_bytes", "records_per_rail", "max_8rail_us", "p95_us",
    "valid_copy_GB/s", "cross_rail_records", "logical_nvlink_GB/s",
    "source_log",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for path_string in args.logs:
        path = Path(path_string)
        for line in path.read_text().splitlines():
            if not line.startswith("engine="):
                continue
            row = {}
            for item in line.split():
                key, value = item.split("=", 1)
                row[key] = value
            # Logs produced before the extended benchmark used these defaults.
            row.setdefault("traffic", "skew")
            row.setdefault("operation", "balance")
            row["source_log"] = str(path)
            writer.writerow({field: row.get(field, "") for field in FIELDS})


if __name__ == "__main__":
    main()
