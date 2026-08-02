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


def _records(tokens: int, pattern: str) -> torch.Tensor:
    records = torch.full((8, tokens, 8), _UNUSED, dtype=torch.int64, device="cuda")
    if pattern == "balanced":
        for owner in range(8):
            records[owner, :, 0] = (1 << 32) | (1 << ((owner + 1) % 8))
    elif pattern == "singleton":
        records[0, :, 0] = (1 << 32) | 1
    elif pattern == "multitarget":
        records[:4, :, 0] = (1 << 32) | 0b1111
    else:
        for token in range(tokens):
            records[0, token, 0] = (1 << 32) | (1 << (1 + token % 7))
    return records


def _run(
    records: torch.Tensor,
    mode: str,
    channels: int,
    planner_chunk_size: int,
    activation_threshold_percent: int,
) -> tuple[torch.Tensor, ...]:
    tokens = records.size(1)
    return tuple(
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
            activation_threshold_percent,
        )
    )


def _check_status(outputs: tuple[torch.Tensor, ...]) -> None:
    if not outputs or int(outputs[-1].item()) != 0:
        status = None if not outputs else int(outputs[-1].item())
        raise RuntimeError(f"hop planner failed with status {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("one_hop", "adaptive", "both"), default="both"
    )
    parser.add_argument(
        "--pattern",
        choices=("balanced", "rotating", "singleton", "multitarget"),
        default="rotating",
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=(8, 32, 128, 512))
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--planner-chunk-size", type=int, default=1)
    parser.add_argument("--activation-threshold-percent", type=int, default=0)
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
        or not 0 <= args.activation_threshold_percent <= 3100
        or (
            args.activation_threshold_percent > 0
            and args.planner_chunk_size == 1
        )
    ):
        parser.error(
            "tokens/steady must be positive, warmup nonnegative, and "
            "channels in [1, 256], activation threshold in [0, 3100], and "
            "a positive activation threshold requires planner chunk > 1"
        )

    torch.cuda.set_device(args.device)
    modes = ("one_hop", "adaptive") if args.mode == "both" else (args.mode,)
    rows = []
    for tokens in args.tokens:
        records = _records(tokens, args.pattern)
        for mode in modes:
            for _ in range(args.warmup):
                _check_status(
                    _run(
                        records,
                        mode,
                        args.channels,
                        args.planner_chunk_size,
                        args.activation_threshold_percent,
                    )
                )
            samples = []
            for sample in range(args.steady):
                if args.nvtx:
                    torch.cuda.nvtx.range_push(
                        f"rail_balance_hop_plan/{mode}/N{tokens}_sample{sample}"
                    )
                started = time.perf_counter_ns()
                outputs = _run(
                    records,
                    mode,
                    args.channels,
                    args.planner_chunk_size,
                    args.activation_threshold_percent,
                )
                samples.append((time.perf_counter_ns() - started) / 1000)
                _check_status(outputs)
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
                    "workload_pattern": args.pattern,
                    "planner_chunk_size": args.planner_chunk_size,
                    "activation_threshold_percent":
                        args.activation_threshold_percent,
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
        "status_checked": True,
        "nvtx": args.nvtx,
        "activation_threshold_percent": args.activation_threshold_percent,
        "rows": rows,
    }
    _write_json(args.output_json, report)
    print(f"PASS hop planner microbenchmark -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
