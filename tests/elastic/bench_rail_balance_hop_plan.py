#!/usr/bin/env python3
"""Profiler-free local microbenchmark for the hop-aware GPU planner API."""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import torch

import deep_ep._C as _C
from bench_rail_balance_hop import _write_json


_UNUSED = -(1 << 32)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _records(tokens: int, mode: str) -> torch.Tensor:
    records = torch.full((8, tokens, 8), _UNUSED, dtype=torch.int64, device="cuda")
    if mode == "adaptive":
        records[0, :, 0] = (1 << 32) | 1
    else:
        for token in range(tokens):
            records[0, token, 0] = (1 << 32) | (1 << (1 + token % 7))
    return records


def _run(
    records: torch.Tensor, mode: str, channels: int, planner_chunk_size: int
) -> None:
    tokens = records.size(1)
    _C._build_rail_balance_hop_one_hop_plan(
        records,
        channels,
        2,
        tokens,
        8 * tokens,
        0,
        0,
        25 if mode == "adaptive" else 0,
        0,
        planner_chunk_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("one_hop", "adaptive", "both"), default="both"
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=(8, 32, 128, 512))
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--planner-chunk-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steady", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(*args.tokens, args.steady) <= 0
        or args.warmup < 0
        or not 1 <= args.channels <= 256
        or args.planner_chunk_size < 1
    ):
        parser.error(
            "tokens/steady must be positive, warmup nonnegative, and "
            "channels in [1, 256]"
        )

    torch.cuda.set_device(args.device)
    modes = ("one_hop", "adaptive") if args.mode == "both" else (args.mode,)
    rows = []
    for mode in modes:
        for tokens in args.tokens:
            records = _records(tokens, mode)
            for _ in range(args.warmup):
                _run(records, mode, args.channels, args.planner_chunk_size)
            samples = []
            for sample in range(args.steady):
                if args.nvtx:
                    torch.cuda.nvtx.range_push(
                        f"rail_balance_hop_plan/{mode}/N{tokens}_sample{sample}"
                    )
                started = time.perf_counter_ns()
                _run(records, mode, args.channels, args.planner_chunk_size)
                samples.append((time.perf_counter_ns() - started) / 1000)
                if args.nvtx:
                    torch.cuda.nvtx.range_pop()
            rows.append(
                {
                    "mode": mode,
                    "G": 8,
                    "N": tokens,
                    "K": 8,
                    "C": args.channels,
                    "D": 2,
                    "planner_chunk_size": args.planner_chunk_size,
                    "samples_us": samples,
                    "median_us": statistics.median(samples),
                    "p95_us": _percentile(samples, 95),
                    "p99_us": _percentile(samples, 99),
                    "mean_us": statistics.fmean(samples),
                    "stdev_us": statistics.pstdev(samples),
                }
            )
    report = {
        "schema_version": 1,
        "claim_scope": "single_node_diagnostic",
        "measurement_scope": "private API allocation + planner kernel + status sync",
        "device": args.device,
        "warmup": args.warmup,
        "steady": args.steady,
        "nvtx": args.nvtx,
        "rows": rows,
    }
    _write_json(args.output_json, report)
    print(f"PASS hop planner microbenchmark -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
