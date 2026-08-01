#!/usr/bin/env python3
"""Profiler-free public Hybrid baseline/candidate round-trip benchmark.

Run once per node. ``WORLD_SIZE``/``RANK`` are node count/node index.  The
baseline is native DeepEP V2 ``rail_balance='off'``. Correctness probes are
outside timing, and every timed RailBalance dispatch ticket is consumed exactly
once by its matching combine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

import run_rail_balance_hybrid_multinode as c105
from bench_rail_balance_hop import build_reference_report, load_workload_routes
from rail_balance_validation_common import (
    build_deterministic_topk,
    build_validation_bundle,
)


_CASES = (
    "balanced",
    "two_hot",
    "one_hot",
    "offdiag_hot",
    "diag_hot",
    "closed_block",
    "trace",
)
_CANDIDATE_MODES = ("force", "legacy_exact", "one_hop", "adaptive")
_CUDA_METRICS = ("dispatch_cuda_ms", "combine_cuda_ms", "roundtrip_cuda_ms")
_ALL_METRICS = _CUDA_METRICS + (
    "dispatch_host_ns",
    "combine_host_ns",
    "roundtrip_host_ns",
)
_PRODUCTION_PLANNER_CHUNK_SIZE = 8
_PRODUCTION_PLANNER_SEED = 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _runtime_proxy_capacity(
    requested: int | None, oracle_capacity: int, num_tokens: int
) -> int:
    """Return a runtime-safe proxy capacity for retained-token staging."""
    if requested is not None and requested < num_tokens:
        raise ValueError(
            "--proxy-slots-per-rank must be at least --num-tokens; "
            "retained-token staging is indexed by the local token index"
        )
    return max(num_tokens, oracle_capacity) if requested is None else requested


def _gather(value: Any, group: dist.ProcessGroup) -> list[Any]:
    values: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def _capture(command: Sequence[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
        return {
            "argv": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "argv": list(command),
            "returncode": None,
            "error": repr(error),
            "stdout": "",
            "stderr": "",
        }


def _node_snapshot(local_rank: int) -> dict[str, Any] | None:
    if local_rank:
        return None
    process_table = _capture(("ps", "-eo", "pid=,ppid=,pgid=,cmd="))
    return {
        "node_rank": int(os.environ["RANK"]),
        "hostname": platform.node(),
        "gpu_state": _capture(
            (
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,pstate,"
                "temperature.gpu,power.draw,power.limit,clocks.current.sm,"
                "clocks.current.memory,clocks_throttle_reasons.active,"
                "mig.mode.current,compute_mode",
                "--format=csv,noheader,nounits",
            )
        ),
        "compute_apps": _capture(
            (
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            )
        ),
        "process_table": process_table,
        "mps_processes": [
            row
            for row in process_table["stdout"].splitlines()
            if "nvidia-cuda-mps-control" in row or "nvidia-cuda-mps-server" in row
        ],
    }


def _snapshot_reasons(
    snapshots: Sequence[dict[str, Any]],
    workers: Sequence[dict[str, Any]],
    num_processes: int,
) -> list[str]:
    allowed: dict[tuple[int, str], set[int]] = {}
    for worker in workers:
        key = (worker["node_rank"], worker["hostname"])
        allowed.setdefault(key, set()).add(worker["pid"])
    reasons = []
    for snapshot in snapshots:
        host = snapshot["hostname"]
        node_key = (snapshot["node_rank"], host)
        for name in ("gpu_state", "compute_apps", "process_table"):
            if snapshot[name].get("returncode") != 0:
                reasons.append(f"{host}: {name} unavailable")
        gpu_rows = [
            row for row in snapshot["gpu_state"]["stdout"].splitlines() if row.strip()
        ]
        if len(gpu_rows) != num_processes:
            reasons.append(
                f"{host}: expected {num_processes} GPUs, got {len(gpu_rows)}"
            )
        for row in gpu_rows:
            fields = [field.strip() for field in row.split(",")]
            if len(fields) != 13:
                reasons.append(f"{host}: invalid GPU state row")
                continue
            if fields[4] != "P0":
                reasons.append(f"{host}: GPU {fields[0]} is not in P0")
            if fields[10].lower() not in ("0", "0x0000000000000000", "not active"):
                reasons.append(f"{host}: GPU {fields[0]} throttle is active")
            if fields[11] != "Disabled":
                reasons.append(f"{host}: GPU {fields[0]} MIG is not disabled")
            if fields[12] != "Default":
                reasons.append(f"{host}: GPU {fields[0]} compute mode is not Default")
        if snapshot["mps_processes"]:
            reasons.append(f"{host}: CUDA MPS is active")
        pids = set()
        for row in snapshot["compute_apps"]["stdout"].splitlines():
            fields = [field.strip() for field in row.split(",")]
            try:
                pids.add(int(fields[1]))
            except (IndexError, ValueError):
                reasons.append(f"{host}: invalid compute-app row")
        unexpected = sorted(pids - allowed.get(node_key, set()))
        if unexpected:
            reasons.append(f"{host}: unexpected GPU PIDs {unexpected}")
    return reasons


def _snapshot_drift_reasons(
    before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]
) -> list[str]:
    def identities(rows: Sequence[dict[str, Any]]) -> dict[int, list[str]]:
        return {
            row["node_rank"]: [
                ",".join(field.strip() for field in gpu.split(",")[:4])
                for gpu in row["gpu_state"]["stdout"].splitlines()
                if gpu.strip()
            ]
            for row in rows
        }

    return (
        []
        if identities(before) == identities(after)
        else ["GPU UUID, model, or driver changed during measurement"]
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    values = [float(value) for value in values]
    _require(
        values and all(math.isfinite(value) for value in values),
        "performance samples must be finite",
    )
    mean, std = statistics.fmean(values), statistics.pstdev(values)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "mean": mean,
        "std_population": std,
        "cv_population": 0.0 if mean == 0 else std / mean,
        "min": min(values),
        "max": max(values),
    }


def _order(repeats: int, candidate: str = "force") -> list[tuple[int, str, int, str]]:
    blocks = []
    for repeat in range(repeats):
        for pattern, modes in (
            ("ABBA", ("off", candidate, candidate, "off")),
            ("BAAB", (candidate, "off", "off", candidate)),
        ):
            blocks.extend(
                (repeat, pattern, position, mode)
                for position, mode in enumerate(modes)
            )
    return blocks


def _roundtrip(
    buffer: Any,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    args: argparse.Namespace,
    events: tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event] | None,
) -> tuple[dict[str, int], torch.Tensor, torch.Tensor | None]:
    start, middle, end = events or (None, None, None)
    if start is not None:
        start.record()
    begin = time.perf_counter_ns()
    recv_x, _, recv_weights, handle, _ = buffer.dispatch(
        x,
        topk_idx,
        topk_weights,
        num_experts=args.num_experts,
        num_max_tokens_per_rank=args.num_tokens,
        expert_alignment=1,
        num_sms=args.num_sms,
        num_qps=0,
        do_handle_copy=True,
        do_cpu_sync=True,
    )
    dispatch_end = time.perf_counter_ns()
    if middle is not None:
        middle.record()
    combine_begin = time.perf_counter_ns()
    combined_x, combined_weights, _ = buffer.combine(
        recv_x, handle, topk_weights=recv_weights, num_sms=0, num_qps=0
    )
    combine_end = time.perf_counter_ns()
    if end is not None:
        end.record()
    _require(
        combined_x.is_cuda and (combined_weights is None or combined_weights.is_cuda),
        "combine output is not CUDA-resident",
    )
    return (
        {
            "dispatch_host_ns": dispatch_end - begin,
            "combine_host_ns": combine_end - combine_begin,
            "roundtrip_host_ns": combine_end - begin,
        },
        combined_x,
        combined_weights,
    )


def _measure_block(
    buffer: Any,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    args: argparse.Namespace,
    group: dist.ProcessGroup,
    rank: int,
    block_index: int,
    repeat: int,
    pattern: str,
    position: int,
    mode: str,
) -> dict[str, Any]:
    dist.monitored_barrier(
        group=group, timeout=timedelta(seconds=args.timeout), wait_all_ranks=True
    )
    torch.cuda.synchronize()
    for _ in range(args.warmup_iters):
        _roundtrip(buffer, x, topk_idx, topk_weights, args, None)
    torch.cuda.synchronize()
    events = [
        tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
        for _ in range(args.steady_iters)
    ]
    dist.monitored_barrier(
        group=group, timeout=timedelta(seconds=args.timeout), wait_all_ranks=True
    )
    torch.cuda.synchronize()
    host_rows = []
    last_output = None
    for item in events:
        host, combined_x, combined_weights = _roundtrip(
            buffer, x, topk_idx, topk_weights, args, item
        )
        host_rows.append(host)
        last_output = (combined_x, combined_weights)
    # No steady-step synchronization: one synchronization at the block boundary.
    torch.cuda.synchronize()
    assert last_output is not None and last_output[1] is not None
    last_digest = c105._digest(last_output[0], last_output[1])
    rank_digests = _gather(last_digest, group)
    local = []
    for iteration, (host, (start, middle, end)) in enumerate(
        zip(host_rows, events, strict=True)
    ):
        local.append(
            {
                "rank": rank,
                "iteration": iteration,
                **host,
                "dispatch_cuda_ms": float(start.elapsed_time(middle)),
                "combine_cuda_ms": float(middle.elapsed_time(end)),
                "roundtrip_cuda_ms": float(start.elapsed_time(end)),
            }
        )
    rank_rows = _gather(local, group)
    samples = []
    logical_tokens = dist.get_world_size(group) * args.num_tokens
    for iteration in range(args.steady_iters):
        raw = [rows[iteration] for rows in rank_rows]
        _require(
            [row["rank"] for row in raw] == list(range(len(rank_rows))),
            "rank samples are not ordered",
        )
        rank_max = {
            metric: max(float(row[metric]) for row in raw) for metric in _ALL_METRICS
        }
        samples.append(
            {
                "iteration": iteration,
                "rank_raw": raw,
                "rank_max": rank_max,
                "logical_input_tokens_per_second": logical_tokens
                * 1000
                / rank_max["roundtrip_cuda_ms"],
            }
        )
    summary = {
        metric: _stats([sample["rank_max"][metric] for sample in samples])
        for metric in _ALL_METRICS
    }
    summary["logical_input_tokens_per_second"] = _stats(
        [sample["logical_input_tokens_per_second"] for sample in samples]
    )
    resolved_num_sms = args.num_sms or buffer.get_theoretical_num_sms(
        args.num_experts, args.num_topk
    )
    return {
        "block_index": block_index,
        "pair_index": block_index // 2,
        "order_repeat": repeat,
        "pattern": pattern,
        "pattern_position": position,
        "mode": mode,
        "last_output_rank_digests": rank_digests,
        "samples": samples,
        "summary": summary,
        "resolved_execution": {
            "num_sms": resolved_num_sms,
            "num_qps": buffer.get_theoretical_num_qps(resolved_num_sms),
            "num_allocated_qps": buffer.num_allocated_qps,
        },
    }


def _aggregate(blocks: Sequence[dict[str, Any]], mode: str) -> dict[str, Any]:
    selected = [block for block in blocks if block["mode"] == mode]
    result = {}
    for metric in _ALL_METRICS:
        pooled = [
            sample["rank_max"][metric]
            for block in selected
            for sample in block["samples"]
        ]
        block_medians = [block["summary"][metric]["median"] for block in selected]
        result[metric] = {
            "pooled_raw": _stats(pooled),
            "block_medians": block_medians,
            "median_of_block_medians": statistics.median(block_medians),
        }
    return result


def _compare(
    aggregate: dict[str, Any], candidate_mode: str = "force"
) -> dict[str, Any]:
    comparison = {}
    for metric in _CUDA_METRICS:
        baseline = aggregate["off"][metric]["median_of_block_medians"]
        candidate = aggregate[candidate_mode][metric]["median_of_block_medians"]
        comparison[metric] = {
            "baseline_median_ms": baseline,
            "candidate_median_ms": candidate,
            "speedup_baseline_over_candidate": baseline / candidate,
            "candidate_latency_delta_percent": 100 * (candidate / baseline - 1),
        }
    return comparison


def _paired_roundtrip(
    blocks: Sequence[dict[str, Any]], candidate_mode: str = "force"
) -> list[dict[str, Any]]:
    pairs = []
    for pair_index in sorted({block["pair_index"] for block in blocks}):
        pair = [block for block in blocks if block["pair_index"] == pair_index]
        _require(
            len(pair) == 2
            and {item["mode"] for item in pair} == {"off", candidate_mode},
            "each adjacent pair must contain baseline and candidate",
        )
        medians = {
            item["mode"]: item["summary"]["roundtrip_cuda_ms"]["median"]
            for item in pair
        }
        pairs.append(
            {
                "pair_index": pair_index,
                "block_indices": [item["block_index"] for item in pair],
                "pattern": pair[0]["pattern"],
                "speedup_baseline_over_candidate": (
                    medians["off"] / medians[candidate_mode]
                ),
            }
        )
    return pairs


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


def _benchmark_manifest(args: argparse.Namespace) -> dict[str, Any]:
    workload_sha256 = None
    if args.workload_json is not None:
        workload_sha256 = hashlib.sha256(
            args.workload_json.read_bytes()
        ).hexdigest()
    return {
        "run_id": args.run_id,
        "case": args.case,
        "output": str(args.output),
        "num_processes": args.num_processes,
        "warmup_iters": args.warmup_iters,
        "steady_iters": args.steady_iters,
        "order_repeats": args.order_repeats,
        "candidate_mode": args.candidate_mode,
        "workload_json": (
            str(args.workload_json.resolve())
            if args.workload_json is not None else None
        ),
        "workload_sha256": workload_sha256,
        "two_hop_threshold_percent": args.two_hop_threshold_percent,
        "max_two_hop_percent": args.max_two_hop_percent,
        "hop_penalty_percent": args.hop_penalty_percent,
        "expanded_order": [
            row[3] for row in _order(args.order_repeats, args.candidate_mode)
        ],
        "allow_contended_smoke": args.allow_contended_smoke,
        "diagnostic_profiler": args.diagnostic_profiler,
        "jit_cache_dir": os.environ.get("EP_JIT_CACHE_DIR"),
        "timeout": args.timeout,
        "watchdog_seconds": args.watchdog_seconds,
    }


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    import deep_ep
    from deep_ep.buffers import elastic as elastic_module
    from deep_ep.utils.envs import init_dist

    _C = importlib.import_module("deep_ep._C")
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(seconds=args.timeout),
    )
    original_host = elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE
    original_capability = _C._rail_balance_force_available
    live_buffer = None
    try:
        c105._phase(
            "production capability gate",
            group,
            args.timeout,
            lambda: (
                _require(original_host is False, "host force capability must be off"),
                _require(
                    original_capability() is False,
                    "compiled force capability must be off",
                ),
            ),
        )
        identity = c105._phase(
            "cluster identity",
            group,
            args.timeout,
            lambda: c105._consensus_identity(c105._local_identity(args, _C), group),
        )
        benchmark_manifest = c105._phase(
            "benchmark manifest",
            group,
            args.timeout,
            lambda: c105._consensus_identity(_benchmark_manifest(args), group),
        )
        node_count = int(os.environ["WORLD_SIZE"])
        _require(
            world_size == node_count * num_local_ranks,
            "WORLD does not match nodes * local processes",
        )
        _require(
            rank == int(os.environ["RANK"]) * num_local_ranks + local_rank,
            "global rank ordering is not server-major",
        )
        routes = (
            load_workload_routes(
                args.workload_json,
                num_nodes=node_count,
                rails=num_local_ranks,
                tokens_per_rank=args.num_tokens,
                topk=args.num_topk,
                num_experts=args.num_experts,
                bytes_per_copy=args.hidden * 2,
            )
            if args.workload_json is not None
            else build_deterministic_topk(
                args.case,
                num_scaleout_ranks=node_count,
                num_scaleup_ranks=num_local_ranks,
                num_tokens_per_rank=args.num_tokens,
                num_topk=args.num_topk,
                num_experts=args.num_experts,
            )
        )
        x, topk_idx, topk_weights = c105._make_input(
            rank,
            routes,
            args.num_tokens,
            args.hidden,
            torch.device("cuda", local_rank),
            deep_ep.topk_idx_t,
        )
        reference = c105._phase(
            "dispatch reference",
            group,
            args.timeout,
            lambda: c105._dispatch_reference(
                x, topk_idx, topk_weights, args.num_tokens, args.num_experts
            ),
        )
        source_y = c105._phase(
            "combine source",
            group,
            args.timeout,
            lambda: c105._source_combine_input(
                topk_idx, args.num_tokens, args.num_topk, args.hidden
            ),
        )

        def combine_reference():
            from deep_ep.utils.refs import combine as ref_combine

            return ref_combine(
                source_y,
                topk_idx,
                node_count,
                num_local_ranks,
                args.num_experts,
                None,
                True,
                True,
            )

        combined_reference = c105._phase(
            "combine reference", group, args.timeout, combine_reference
        )
        oracle = (
            None
            if args.workload_json is not None
            else build_validation_bundle(
                run_id=f"{args.run_id}/benchmark",
                case=args.case,
                modes=("off", "force"),
                num_scaleout_ranks=node_count,
                num_scaleup_ranks=num_local_ranks,
                num_tokens_per_rank=args.num_tokens,
                num_topk=args.num_topk,
                num_experts=args.num_experts,
                hidden=args.hidden,
                num_channels=1,
                proxy_slots_per_rank=args.proxy_slots_per_rank,
                policy=args.rail_policy,
                threshold_percent=args.rail_threshold_percent,
            )
        )
        capacity = _runtime_proxy_capacity(
            args.proxy_slots_per_rank,
            (
                oracle["config"]["proxy_slots_per_rank"]
                if oracle is not None
                else num_local_ranks * args.num_tokens
            ),
            args.num_tokens,
        )
        candidate_mode = args.candidate_mode
        modes = ("off", candidate_mode)
        elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = True
        _C._rail_balance_force_available = lambda: True

        probes = {}
        for mode in modes:
            live_buffer = c105._phase(
                f"{mode} probe constructor",
                group,
                args.timeout,
                lambda mode=mode: deep_ep.ElasticBuffer(
                    ep_group, **c105._constructor_kwargs(args, mode, capacity)
                ),
            )
            probes[mode] = c105._run_round_trip(
                live_buffer,
                x,
                topk_idx,
                topk_weights,
                reference,
                combined_reference,
                args.num_tokens,
                args.num_topk,
                args.hidden,
                args.num_experts,
                args.num_sms,
                group,
                args.timeout,
            )
            _, timed_x, timed_weights = _roundtrip(
                live_buffer, x, topk_idx, topk_weights, args, None
            )
            torch.cuda.synchronize()
            _require(timed_weights is not None, "timed-path probe returned no weights")
            probes[mode]["timed_path_rank_digests"] = _gather(
                c105._digest(timed_x, timed_weights), group
            )
            c105._destroy(live_buffer, group, args.timeout)
            live_buffer = None
        c105._phase(
            "probe A/B",
            group,
            args.timeout,
            lambda: (
                _require(
                    torch.equal(
                        probes["off"]["combined_x"],
                        probes[candidate_mode]["combined_x"]
                    ),
                    "baseline/candidate probe payload differs",
                ),
                _require(
                    torch.equal(
                        probes["off"]["combined_weights"],
                        probes[candidate_mode]["combined_weights"],
                    ),
                    "baseline/candidate probe weights differ",
                ),
            ),
        )
        if oracle is not None:
            oracle = build_validation_bundle(
                run_id=f"{args.run_id}/benchmark",
                case=args.case,
                modes=("off", "force"),
                num_scaleout_ranks=node_count,
                num_scaleup_ranks=num_local_ranks,
                num_tokens_per_rank=args.num_tokens,
                num_topk=args.num_topk,
                num_experts=args.num_experts,
                hidden=args.hidden,
                num_channels=int(probes[candidate_mode]["num_channels"]),
                proxy_slots_per_rank=capacity,
                policy=args.rail_policy,
                threshold_percent=args.rail_threshold_percent,
            )

        workers = _gather(
            {
                "rank": rank,
                "node_rank": int(os.environ["RANK"]),
                "hostname": platform.node(),
                "pid": os.getpid(),
            },
            group,
        )
        pre = [
            row for row in _gather(_node_snapshot(local_rank), group) if row is not None
        ]
        pre_reasons = _snapshot_reasons(pre, workers, num_local_ranks)
        if pre_reasons and not args.allow_contended_smoke:
            raise RuntimeError(
                "performance environment rejected before timing: "
                + "; ".join(pre_reasons)
            )
        blocks = []
        for block_index, (repeat, pattern, position, mode) in enumerate(
            _order(args.order_repeats, candidate_mode)
        ):
            live_buffer = c105._phase(
                f"block {block_index} {mode} constructor",
                group,
                args.timeout,
                lambda mode=mode: deep_ep.ElasticBuffer(
                    ep_group, **c105._constructor_kwargs(args, mode, capacity)
                ),
            )
            blocks.append(
                _measure_block(
                    live_buffer,
                    x,
                    topk_idx,
                    topk_weights,
                    args,
                    group,
                    rank,
                    block_index,
                    repeat,
                    pattern,
                    position,
                    mode,
                )
            )
            c105._destroy(live_buffer, group, args.timeout)
            live_buffer = None
        post = [
            row for row in _gather(_node_snapshot(local_rank), group) if row is not None
        ]
        post_identity = c105._phase(
            "post-measurement cluster identity",
            group,
            args.timeout,
            lambda: c105._consensus_identity(c105._local_identity(args, _C), group),
        )
        _require(
            identity == post_identity,
            "Git, extension, hardware, or config identity changed",
        )
        for block in blocks:
            _require(
                block["last_output_rank_digests"]
                == probes[block["mode"]]["timed_path_rank_digests"],
                f"timed {block['mode']} output differs from its correctness probe",
            )
        reasons = sorted(
            set(
                pre_reasons
                + _snapshot_reasons(post, workers, num_local_ranks)
                + _snapshot_drift_reasons(pre, post)
            )
        )
        if args.warmup_iters < 10:
            reasons.append("warmup_iters < 10")
        if args.steady_iters < 100:
            reasons.append("steady_iters < 100")
        if os.environ.get("EP_USE_NVIDIA_TOOLS") not in (None, "", "0"):
            reasons.append("EP_USE_NVIDIA_TOOLS declares profiler instrumentation")
        if args.diagnostic_profiler:
            reasons.append("diagnostic_profiler was declared")
        if args.allow_contended_smoke:
            reasons.append("allow_contended_smoke was requested")
        aggregate = {mode: _aggregate(blocks, mode) for mode in modes}
        comparison = _compare(aggregate, candidate_mode)
        comparison["paired_roundtrip"] = _paired_roundtrip(blocks, candidate_mode)
        hop_reference = build_reference_report(
            run_id=args.run_id,
            case=args.case,
            mode=(
                "legacy_exact"
                if candidate_mode in ("force", "legacy_exact")
                else candidate_mode
            ),
            num_nodes=node_count,
            rails=num_local_ranks,
            tokens_per_rank=args.num_tokens,
            topk=args.num_topk,
            num_experts=args.num_experts,
            hidden=args.hidden,
            chunk_size=_PRODUCTION_PLANNER_CHUNK_SIZE,
            two_hop_threshold_percent=args.two_hop_threshold_percent,
            max_two_hop_percent=args.max_two_hop_percent,
            hop_penalty_percent=args.hop_penalty_percent,
            seed=_PRODUCTION_PLANNER_SEED,
            proxy_slots_per_rank=capacity,
            workload_json=args.workload_json,
        )
        report = {
            "schema_version": 1,
            "run_id": args.run_id,
            "git_commit": identity["git_commit"],
            "backend": "multinode",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "claim_scope": "real_multinode",
            "baseline": {
                "mode": "off",
                "definition": "native DeepEP V2 Hybrid data path",
            },
            "candidate": {
                "mode": candidate_mode,
                "policy": args.rail_policy,
                "threshold_percent": args.rail_threshold_percent,
                "two_hop_threshold_percent": args.two_hop_threshold_percent,
                "max_two_hop_percent": args.max_two_hop_percent,
                "hop_penalty_percent": args.hop_penalty_percent,
            },
            "identity": identity,
            "topology": hop_reference["topology"],
            "planner_config": hop_reference["planner_config"],
            "kernel_config": {
                "num_sms": args.num_sms,
                "num_allocated_qps": args.num_allocated_qps,
                "resolved_num_sms": probes[candidate_mode]["resolved_num_sms"],
                "resolved_num_qps": probes[candidate_mode]["resolved_num_qps"],
                "resolved_num_allocated_qps": probes[candidate_mode][
                    "resolved_num_allocated_qps"
                ],
                "baseline_resolved": {
                    "num_sms": probes["off"]["resolved_num_sms"],
                    "num_qps": probes["off"]["resolved_num_qps"],
                    "num_allocated_qps": probes["off"]["resolved_num_allocated_qps"],
                },
            },
            "workload_summary": hop_reference["workload_summary"],
            "correctness": {
                "passed": True,
                "off_candidate_equal": True,
                "endpoint_unchanged": True,
            },
            "path_distribution": {
                "source": "python_endpoint_oracle_expected",
                **hop_reference["path_distribution"],
            },
            "rail_load_before_after": hop_reference["rail_load_before_after"],
            "post_measurement_identity": post_identity,
            "benchmark_manifest": benchmark_manifest,
            "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "workload": {
                "case": args.case,
                "node_count": node_count,
                "local_processes": num_local_ranks,
                "tokens_per_rank": args.num_tokens,
                "hidden": args.hidden,
                "topk": args.num_topk,
                "experts": args.num_experts,
                "num_sms": args.num_sms,
                "num_allocated_qps": args.num_allocated_qps,
                "proxy_slots_per_rank": capacity,
                "seed": args.seed,
            },
            "measurement": {
                "profiler_contract": (
                    "diagnostic"
                    if args.diagnostic_profiler
                    else "declared direct and unwrapped; external wrappers are not auto-detected"
                ),
                "warmup_per_block": args.warmup_iters,
                "steady_per_block": args.steady_iters,
                "order": [block["mode"] for block in blocks],
                "timing": (
                    "CUDA events; no synchronization inside the steady loop, "
                    "one synchronization at its block boundary"
                ),
                "distributed_truth": "max rank-local elapsed per iteration",
                "cuda_metric_scope": (
                    "current-stream CUDA-event call envelope, including GPU idle "
                    "while host/world gates run; not summed kernel execution time"
                ),
            },
            "correctness_probe": {
                "timed": False,
                "baseline_candidate_equal": True,
                "off_digest": probes["off"]["global_digest"],
                "candidate_digest": probes[candidate_mode]["global_digest"],
            },
            "environment": {
                "workers": workers,
                "pre": pre,
                "post": post,
                "residual_risks": ["transient co-tenants and profiler auto-detection"],
            },
            "eligibility": {
                "performance_claim_eligible": not reasons,
                "reasons": reasons,
            },
            "traffic_oracle": (
                oracle["records"]["force"]["plan"]
                if oracle is not None
                and candidate_mode in ("force", "legacy_exact")
                else None
            ),
            "blocks": blocks,
            "aggregate": aggregate,
            "comparison": comparison,
            "latency_statistics": {
                "aggregate": aggregate,
                "comparison": comparison,
            },
            "throughput_statistics": {
                mode: aggregate[mode]["roundtrip_cuda_ms"]
                for mode in modes
            },
            "fallbacks": reasons,
        }

        def write() -> None:
            if rank == 0:
                _write_json(args.output, report)
                result = comparison["roundtrip_cuda_ms"]
                print(
                    f"PASS off/{candidate_mode} benchmark: "
                    f"off={result['baseline_median_ms']*1e3:.3f} us, "
                    f"candidate={result['candidate_median_ms']*1e3:.3f} us, "
                    f"speedup={result['speedup_baseline_over_candidate']:.4f}x, "
                    f"eligible={not reasons} -> {args.output}",
                    flush=True,
                )

        c105._phase("write report", group, args.timeout, write)
        dist.destroy_process_group()
    finally:
        elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = original_host
        _C._rail_balance_force_available = original_capability
        if live_buffer is not None:
            with suppress(BaseException):
                live_buffer.destroy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    if not __debug__:
        parser.error("benchmark correctness checks require Python assertions")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", required=True, choices=_CASES)
    parser.add_argument("--workload-json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--candidate-mode", choices=_CANDIDATE_MODES, default="force"
    )
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-sms", type=int, default=0)
    parser.add_argument("--num-allocated-qps", type=int, default=0)
    parser.add_argument("--proxy-slots-per-rank", type=int)
    parser.add_argument(
        "--rail-policy", choices=("all", "active", "adaptive"), default="all"
    )
    parser.add_argument("--rail-threshold-percent", type=int, default=0)
    parser.add_argument("--two-hop-threshold-percent", type=int, default=0)
    parser.add_argument("--max-two-hop-percent", type=int, default=25)
    parser.add_argument("--hop-penalty-percent", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--steady-iters", type=int, default=100)
    parser.add_argument(
        "--order-repeats",
        type=int,
        default=1,
        help="repeat one ABBA plus one BAAB cycle",
    )
    parser.add_argument(
        "--allow-contended-smoke",
        action="store_true",
        help="continue after environment rejection; result is diagnostic-only",
    )
    parser.add_argument(
        "--diagnostic-profiler",
        action="store_true",
        help="declare an externally profiled run; never performance eligible",
    )
    parser.add_argument("--sl-idx", type=int, default=3)
    parser.add_argument("--seed", type=int, default=106)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--watchdog-seconds", type=int, default=7200)
    parser.add_argument("--worker-suite", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        c105._validate_launch_environment()
    except ValueError as error:
        parser.error(str(error))
    if args.num_processes < 2 or torch.cuda.device_count() < args.num_processes:
        parser.error("at least two visible local CUDA devices are required")
    if args.num_tokens <= 0 or args.hidden <= 0 or args.hidden % 256:
        parser.error(
            "tokens must be positive; hidden must be a positive multiple of 256"
        )
    if (
        args.num_topk <= 0
        or args.num_sms < 0
        or args.num_sms == 1
        or args.num_allocated_qps < 0
    ):
        parser.error("top-k, SM count, and allocated QP count are invalid")
    if not 0 <= args.rail_threshold_percent <= 3100:
        parser.error("rail threshold percent must be in [0, 3100]")
    if not 0 <= args.two_hop_threshold_percent <= 10000:
        parser.error("two-hop threshold percent must be in [0, 10000]")
    if not 0 <= args.max_two_hop_percent <= 100:
        parser.error("max two-hop percent must be in [0, 100]")
    if not 0 <= args.hop_penalty_percent <= 10000:
        parser.error("hop penalty percent must be in [0, 10000]")
    if (args.case == "trace") != (args.workload_json is not None):
        parser.error("case=trace requires --workload-json, and only trace uses it")
    if args.warmup_iters < 0 or args.steady_iters <= 0 or args.order_repeats <= 0:
        parser.error("warmup must be nonnegative; steady/repeats must be positive")
    if args.timeout <= 0 or args.watchdog_seconds <= args.timeout:
        parser.error("watchdog must exceed the positive operation timeout")
    return args


def main() -> None:
    args = _parse_args()
    if not args.worker_suite:
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--worker-suite",
            *sys.argv[1:],
        ]
        process = subprocess.Popen(command, start_new_session=True)
        try:
            return_code = process.wait(timeout=args.watchdog_seconds)
        except subprocess.TimeoutExpired as error:
            c105._terminate_process_group(process)
            raise RuntimeError("benchmark watchdog expired") from error
        if return_code:
            c105._terminate_process_group(process, grace_seconds=0)
            raise SystemExit(return_code)
        return
    torch.multiprocessing.spawn(
        _worker, args=(args.num_processes, args), nprocs=args.num_processes, join=True
    )


if __name__ == "__main__":
    main()
