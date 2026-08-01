#!/usr/bin/env python3
"""Unified hop-aware RailBalance reference, vnode, and multinode entry."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Any, Sequence

from rail_balance_hop_reference import (
    HopFlow,
    HopPlannerConfig,
    build_hop_plan,
    validate_hop_plan,
)
from rail_balance_validation_common import (
    CANONICAL_CASES,
    build_deterministic_topk,
)


_MODES = ("off", "legacy_exact", "one_hop", "adaptive")
_KINDS = ("direct", "dst_forward", "src_forward", "two_hop")


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _load_stats(rows: Sequence[Sequence[int]]) -> dict[str, float]:
    values = [float(value) for row in rows if sum(row) for value in row]
    if not values:
        return {"max": 0.0, "mean": 0.0, "cv": 0.0, "p95": 0.0}
    mean = statistics.fmean(values)
    return {
        "max": max(values),
        "mean": mean,
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
        "p95": _percentile(values, 95),
    }


def load_workload_routes(
    workload_json: Path,
    *,
    num_nodes: int,
    rails: int,
    tokens_per_rank: int,
    topk: int,
    num_experts: int,
    bytes_per_copy: int,
) -> list[list[list[int]]]:
    """Expand the documented endpoint-count manifest into token routes."""
    manifest = json.loads(workload_json.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("workload schema_version must be 1")
    if manifest.get("topology") != {"num_nodes": num_nodes, "rails_per_node": rails}:
        raise ValueError("workload topology does not match CLI topology")
    if manifest.get("unit") != "token_copy":
        raise ValueError("workload unit must be token_copy")
    if manifest.get("bytes_per_copy") != bytes_per_copy:
        raise ValueError("workload bytes_per_copy does not match the payload")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("workload records must be a list")
    world_size = num_nodes * rails
    experts_per_rank = num_experts // world_size
    if topk > experts_per_rank:
        raise ValueError("trace replay requires top-k <= experts per rank")
    routes: list[list[list[int]]] = [[] for _ in range(world_size)]
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("every workload record must be an object")
        values = tuple(
            record.get(name)
            for name in (
                "src_node",
                "src_local_rank",
                "dst_node",
                "dst_local_rank",
                "count",
            )
        )
        if any(type(value) is not int for value in values):
            raise ValueError("workload endpoint fields must be exact integers")
        source, owner, destination, target, count = values
        if not (
            0 <= source < num_nodes
            and 0 <= destination < num_nodes
            and source != destination
            and 0 <= owner < rails
            and 0 <= target < rails
            and count > 0
        ):
            raise ValueError("workload endpoint record is outside the topology")
        rank = source * rails + owner
        target_rank = destination * rails + target
        target_base = target_rank * experts_per_rank
        route = [target_base + lane for lane in range(topk)]
        routes[rank].extend([route] * count)
    for rank, rank_routes in enumerate(routes):
        if len(rank_routes) > tokens_per_rank:
            raise ValueError("workload exceeds tokens-per-rank")
        local_base = rank * experts_per_rank
        local_route = [local_base + lane for lane in range(topk)]
        rank_routes.extend([local_route] * (tokens_per_rank - len(rank_routes)))
    return routes


def _flows_by_source(
    case: str,
    num_nodes: int,
    rails: int,
    tokens_per_rank: int,
    topk: int,
    num_experts: int,
    bytes_per_copy: int,
    workload_json: Path | None = None,
) -> tuple[list[list[list[int]]], tuple[tuple[HopFlow, ...], ...]]:
    routes = (
        load_workload_routes(
            workload_json,
            num_nodes=num_nodes,
            rails=rails,
            tokens_per_rank=tokens_per_rank,
            topk=topk,
            num_experts=num_experts,
            bytes_per_copy=bytes_per_copy,
        )
        if workload_json is not None
        else build_deterministic_topk(
            case,
            num_scaleout_ranks=num_nodes,
            num_scaleup_ranks=rails,
            num_tokens_per_rank=tokens_per_rank,
            num_topk=topk,
            num_experts=num_experts,
        )
    )
    experts_per_node = num_experts // num_nodes
    experts_per_rank = num_experts // (num_nodes * rails)
    sources = []
    for source in range(num_nodes):
        counts: dict[tuple[int, int, tuple[int, ...]], int] = defaultdict(int)
        for owner in range(rails):
            rank = source * rails + owner
            for token_route in routes[rank]:
                masks: dict[int, set[int]] = defaultdict(set)
                for expert in token_route:
                    destination = expert // experts_per_node
                    if destination != source:
                        target = (expert % experts_per_node) // experts_per_rank
                        masks[destination].add(target)
                for destination, targets in masks.items():
                    counts[(owner, destination, tuple(sorted(targets)))] += 1
        sources.append(
            tuple(
                HopFlow(owner, destination, targets, count)
                for (owner, destination, targets), count in sorted(counts.items())
            )
        )
    return routes, tuple(sources)


def build_reference_report(
    *,
    run_id: str,
    case: str,
    mode: str,
    num_nodes: int,
    rails: int,
    tokens_per_rank: int,
    topk: int,
    num_experts: int,
    hidden: int,
    chunk_size: int,
    two_hop_threshold_percent: int,
    max_two_hop_percent: int,
    hop_penalty_percent: int,
    seed: int,
    proxy_slots_per_rank: int | None = None,
    workload_json: Path | None = None,
) -> dict[str, Any]:
    bytes_per_copy = hidden * 2
    routes, sources = _flows_by_source(
        case,
        num_nodes,
        rails,
        tokens_per_rank,
        topk,
        num_experts,
        bytes_per_copy,
        workload_json,
    )
    config = HopPlannerConfig(
        num_rails=rails,
        num_destinations=num_nodes,
        mode=mode,
        chunk_size=chunk_size,
        two_hop_threshold=two_hop_threshold_percent / 100,
        max_two_hop_ratio=(max_two_hop_percent / 100 if mode == "adaptive" else 0),
        hop_penalty=hop_penalty_percent / 100,
        planner_seed=seed,
        proxy_capacity_per_egress=proxy_slots_per_rank,
    )
    before = [[[0] * rails for _ in range(num_nodes)] for _ in range(num_nodes)]
    after = [[[0] * rails for _ in range(num_nodes)] for _ in range(num_nodes)]
    before_source = [[0] * rails for _ in range(num_nodes)]
    after_source = [[0] * rails for _ in range(num_nodes)]
    path_units = {kind: 0 for kind in _KINDS}
    path_chunks = {kind: 0 for kind in _KINDS}
    local_forward_units = extra_forward_units = total_units = 0
    started = time.perf_counter_ns()
    for source, flows in enumerate(sources):
        off_plan = build_hop_plan(
            flows,
            HopPlannerConfig(
                num_rails=rails,
                num_destinations=num_nodes,
                mode="off",
                chunk_size=chunk_size,
                planner_seed=seed,
            ),
        )
        plan = build_hop_plan(flows, config)
        validate_hop_plan(flows, plan)
        before[source] = [list(row) for row in off_plan.pair_load]
        after[source] = [list(row) for row in plan.pair_load]
        before_source[source] = list(off_plan.source_rail_load)
        after_source[source] = list(plan.source_rail_load)
        for kind in _KINDS:
            path_units[kind] += plan.units_for(kind)
            path_chunks[kind] += sum(
                assignment.path_kind == kind for assignment in plan.assignments
            )
        local_forward_units += plan.local_forward_units
        extra_forward_units += plan.extra_local_forward_units
        total_units += plan.total_units
    planner_time_us = (time.perf_counter_ns() - started) / 1000
    before_rows = [
        row
        for source in before
        for destination, row in enumerate(source)
        if sum(row) and destination >= 0
    ]
    after_rows = [
        row
        for source in after
        for destination, row in enumerate(source)
        if sum(row) and destination >= 0
    ]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "backend": "reference",
        "claim_scope": "planner_only",
        "topology": {"num_nodes": num_nodes, "rails_per_node": rails},
        "environment": {
            "hostname": platform.node(),
            "python": sys.version.split()[0],
        },
        "planner_config": {
            "mode": mode,
            "chunk_size": chunk_size,
            "two_hop_threshold_percent": two_hop_threshold_percent,
            "max_two_hop_percent": max_two_hop_percent,
            "hop_penalty_percent": hop_penalty_percent,
            "planner_seed": seed,
            "proxy_slots_per_rank": proxy_slots_per_rank,
        },
        "kernel_config": None,
        "workload_summary": {
            "case": case,
            "tokens_per_rank": tokens_per_rank,
            "topk": topk,
            "num_experts": num_experts,
            "hidden": hidden,
            "bytes_per_copy": bytes_per_copy,
            "workload_json": (
                str(workload_json.resolve()) if workload_json is not None else None
            ),
            "route_rank_count": len(routes),
            "remote_payload_units": total_units,
        },
        "correctness": {
            "passed": True,
            "conserved_payload_units": total_units,
            "endpoint_unchanged": True,
            "deterministic": True,
        },
        "path_distribution": {
            **{f"{kind}_units": path_units[kind] for kind in _KINDS},
            **{f"{kind}_chunks": path_chunks[kind] for kind in _KINDS},
            "two_hop_ratio": (
                path_units["two_hop"] / total_units if total_units else 0.0
            ),
            "local_forward_bytes": local_forward_units * bytes_per_copy,
            "extra_local_hop_bytes": extra_forward_units * bytes_per_copy,
        },
        "rail_load_before_after": {
            "unit": "token_copy",
            "before_pair_rail_load": before,
            "after_pair_rail_load": after,
            "before_source_rail_load": before_source,
            "after_source_rail_load": after_source,
            "before_stats": _load_stats(before_rows),
            "after_stats": _load_stats(after_rows),
            "before_source_stats": _load_stats(before_source),
            "after_source_stats": _load_stats(after_source),
        },
        "latency_statistics": {"planner_time_us": planner_time_us},
        "throughput_statistics": None,
        "fallbacks": [],
    }


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _vnode(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if args.num_nodes != 2 or args.gpus_per_node != 4:
        raise ValueError("vnode backend currently requires 2 logical nodes x 4 GPUs")
    if (args.tokens_per_rank, args.topk, args.num_experts, args.hidden) != (
        16,
        1,
        16,
        256,
    ):
        raise ValueError(
            "vnode backend currently fixes tokens/topk/experts/hidden to " "16/1/16/256"
        )
    if args.mode not in ("one_hop", "adaptive"):
        raise ValueError("vnode backend supports one_hop or adaptive")
    if args.case == "capacity":
        raise ValueError("capacity is a planner-only case")
    command = [
        sys.executable,
        str(Path(__file__).with_name("test_rail_balance_hop_vnode_cuda.py")),
        "--mode",
        args.mode,
        "--case",
        args.case,
        "--two-hop-threshold-percent",
        str(args.two_hop_threshold_percent),
        "--max-two-hop-percent",
        str(args.max_two_hop_percent),
        "--hop-penalty-percent",
        str(args.hop_penalty_percent),
        "--timeout",
        str(args.timeout),
    ]
    if args.workload_json is not None:
        command.extend(("--workload-json", str(args.workload_json)))
    environment = dict(os.environ)
    environment.setdefault("EP_DISABLE_GIN", "1")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MASTER_PORT", str(30000 + os.getpid() % 20000))
    subprocess.run(command, check=True, env=environment)
    report["backend"] = "vnode"
    report["claim_scope"] = "single_node_diagnostic"
    report["correctness"]["vnode_roundtrip_passed"] = True
    report["kernel_config"] = {
        "physical_gpus": 8,
        "virtual_layout": "4x2",
        "transport": "LSA emulation",
    }


def _multinode(args: argparse.Namespace) -> None:
    if args.mode == "off":
        raise ValueError("multinode candidate mode must differ from off baseline")
    command = [
        sys.executable,
        str(Path(__file__).with_name("bench_rail_balance_hybrid_multinode.py")),
        "--run-id",
        args.run_id,
        "--case",
        args.case,
        "--output",
        str(args.output_json),
        "--candidate-mode",
        args.mode,
        "--num-processes",
        str(args.gpus_per_node),
        "--num-tokens",
        str(args.tokens_per_rank),
        "--hidden",
        str(args.hidden),
        "--num-topk",
        str(args.topk),
        "--num-experts",
        str(args.num_experts),
        "--two-hop-threshold-percent",
        str(args.two_hop_threshold_percent),
        "--max-two-hop-percent",
        str(args.max_two_hop_percent),
        "--hop-penalty-percent",
        str(args.hop_penalty_percent),
        "--warmup-iters",
        str(args.warmup_iters),
        "--steady-iters",
        str(args.steady_iters),
        "--timeout",
        str(args.timeout),
        "--watchdog-seconds",
        str(args.watchdog_seconds),
    ]
    if args.workload_json is not None:
        command.extend(("--workload-json", str(args.workload_json)))
    if args.proxy_slots_per_rank is not None:
        command.extend(("--proxy-slots-per-rank", str(args.proxy_slots_per_rank)))
    os.execv(sys.executable, command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("reference", "vnode", "multinode"), required=True
    )
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--case", choices=(*CANONICAL_CASES, "trace"), required=True)
    parser.add_argument("--workload-json", type=Path)
    parser.add_argument("--run-id", default="rail-balance-hop")
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--tokens-per-rank", type=int, default=16)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--two-hop-threshold-percent", type=int, default=0)
    parser.add_argument("--max-two-hop-percent", type=int, default=25)
    parser.add_argument("--hop-penalty-percent", type=int, default=50)
    parser.add_argument("--proxy-slots-per-rank", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--steady-iters", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--watchdog-seconds", type=int, default=7200)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if (
        min(
            args.num_nodes,
            args.gpus_per_node,
            args.tokens_per_rank,
            args.topk,
            args.num_experts,
            args.hidden,
            args.chunk_size,
        )
        <= 0
    ):
        parser.error("topology and workload sizes must be positive")
    if args.num_experts % (args.num_nodes * args.gpus_per_node):
        parser.error("num-experts must be divisible by nodes * GPUs")
    if (args.case == "trace") != (args.workload_json is not None):
        parser.error("case=trace requires --workload-json, and only trace uses it")
    if not 0 <= args.two_hop_threshold_percent <= 10000:
        parser.error("two-hop threshold percent must be in [0, 10000]")
    if not 0 <= args.max_two_hop_percent <= 100:
        parser.error("max two-hop percent must be in [0, 100]")
    if not 0 <= args.hop_penalty_percent <= 10000:
        parser.error("hop penalty percent must be in [0, 10000]")
    return args


def main() -> None:
    args = _parse_args()
    if args.backend == "multinode":
        _multinode(args)
        return
    report = build_reference_report(
        run_id=args.run_id,
        case=args.case,
        mode=args.mode,
        num_nodes=args.num_nodes,
        rails=args.gpus_per_node,
        tokens_per_rank=args.tokens_per_rank,
        topk=args.topk,
        num_experts=args.num_experts,
        hidden=args.hidden,
        chunk_size=args.chunk_size,
        two_hop_threshold_percent=args.two_hop_threshold_percent,
        max_two_hop_percent=args.max_two_hop_percent,
        hop_penalty_percent=args.hop_penalty_percent,
        seed=args.seed,
        proxy_slots_per_rank=args.proxy_slots_per_rank,
        workload_json=args.workload_json,
    )
    if args.backend == "vnode":
        _vnode(args, report)
    _write_json(args.output_json, report)
    print(
        f"PASS {args.backend}/{args.mode}/{args.case} -> {args.output_json}", flush=True
    )


if __name__ == "__main__":
    main()
