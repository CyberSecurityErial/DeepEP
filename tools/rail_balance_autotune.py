#!/usr/bin/env python3
"""Build and freeze finite, offline RailBalance tuning candidates.

This tool never imports DeepEP and never launches a benchmark.  CPU reference
planning removes invalid or Pareto-dominated routing policies; real multinode
measurements remain the only source of performance truth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


_ROOT = Path(__file__).resolve().parents[1]
_ELASTIC_TESTS = _ROOT / "tests" / "elastic"
sys.path.insert(0, str(_ELASTIC_TESTS))

build_reference_report = importlib.import_module(
    "bench_rail_balance_hop"
).build_reference_report


SCHEMA_VERSION = 1
PRODUCTION_PLANNER_SEED = 0
PRODUCTION_PLANNER_CHUNK_SIZE = 8
MINIMUM_FREEZE_RUNS = 5
BENCHMARK_SL_INDEX = 3
BENCHMARK_DATA_SEED = 106
BENCHMARK_CASES = (
    "balanced",
    "two_hot",
    "one_hot",
    "offdiag_hot",
    "diag_hot",
    "closed_block",
    "trace",
)
_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_keys(value: object, keys: set[str], name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    _require(set(value) == keys, f"{name} fields must be {sorted(keys)}")
    return value


def _integer_list(value: object, name: str, *, minimum: int, maximum: int) -> list[int]:
    _require(isinstance(value, list) and value, f"{name} must be a nonempty list")
    _require(
        all(type(item) is int and minimum <= item <= maximum for item in value),
        f"{name} values must be integers in [{minimum}, {maximum}]",
    )
    _require(len(value) == len(set(value)), f"{name} contains duplicates")
    return value


def _validate_space(space: object) -> None:
    space = _exact_keys(
        space,
        {
            "schema_version",
            "frozen_default_commit",
            "algorithm",
            "execution_presets",
            "proxy_slots_per_rank",
            "max_pareto_candidates",
        },
        "search space",
    )
    _require(space["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
    _require(
        isinstance(space["frozen_default_commit"], str)
        and _COMMIT.fullmatch(space["frozen_default_commit"]) is not None,
        "frozen_default_commit must be a full lowercase Git commit",
    )
    algorithm = _exact_keys(
        space["algorithm"],
        {
            "modes",
            "two_hop_threshold_percent",
            "max_two_hop_percent",
            "hop_penalty_percent",
            "planner_seed",
            "planner_chunk_size",
        },
        "algorithm",
    )
    _require(
        isinstance(algorithm["modes"], list)
        and algorithm["modes"]
        and len(algorithm["modes"]) == len(set(algorithm["modes"]))
        and set(algorithm["modes"]) <= {"one_hop", "adaptive"},
        "algorithm.modes must contain unique one_hop/adaptive values",
    )
    _integer_list(
        algorithm["two_hop_threshold_percent"],
        "algorithm.two_hop_threshold_percent",
        minimum=0,
        maximum=10000,
    )
    _integer_list(
        algorithm["max_two_hop_percent"],
        "algorithm.max_two_hop_percent",
        minimum=0,
        maximum=100,
    )
    _integer_list(
        algorithm["hop_penalty_percent"],
        "algorithm.hop_penalty_percent",
        minimum=0,
        maximum=10000,
    )
    _require(
        algorithm["planner_seed"] == PRODUCTION_PLANNER_SEED,
        f"planner_seed must match production value {PRODUCTION_PLANNER_SEED}",
    )
    _require(
        algorithm["planner_chunk_size"] == PRODUCTION_PLANNER_CHUNK_SIZE,
        "planner_chunk_size must match production value "
        f"{PRODUCTION_PLANNER_CHUNK_SIZE}",
    )

    presets = space["execution_presets"]
    _require(
        isinstance(presets, list) and len(presets) == 1,
        "v1 requires exactly one V2-auto execution preset",
    )
    preset_ids: set[str] = set()
    for index, raw in enumerate(presets):
        preset = _exact_keys(
            raw,
            {"id", "strategy", "num_sms", "num_allocated_qps"},
            f"execution_presets[{index}]",
        )
        _require(
            isinstance(preset["id"], str) and _ID.fullmatch(preset["id"]),
            "execution preset id is invalid",
        )
        _require(preset["id"] not in preset_ids, "duplicate execution preset id")
        preset_ids.add(preset["id"])
        _require(preset["strategy"] == "v2_auto", "v1 only supports V2 auto")
        _require(
            type(preset["num_sms"]) is int
            and preset["num_sms"] >= 0
            and preset["num_sms"] != 1,
            "num_sms must be 0 or at least 2",
        )
        _require(
            type(preset["num_allocated_qps"]) is int
            and preset["num_allocated_qps"] >= 0,
            "num_allocated_qps must be nonnegative",
        )
        _require(
            preset["num_sms"] == preset["num_allocated_qps"] == 0,
            "v2_auto requires num_sms=num_allocated_qps=0",
        )

    slots = space["proxy_slots_per_rank"]
    _require(
        slots is None or (type(slots) is int and slots > 0),
        "proxy_slots_per_rank must be null or a positive integer",
    )
    _require(
        type(space["max_pareto_candidates"]) is int
        and 1 <= space["max_pareto_candidates"] <= 64,
        "max_pareto_candidates must be in [1, 64]",
    )


def _enumerate_algorithm_configs(space: dict[str, Any]) -> list[dict[str, int | str]]:
    _validate_space(space)
    algorithm = space["algorithm"]
    rows: list[dict[str, int | str]] = []
    for mode in algorithm["modes"]:
        if mode == "one_hop":
            combinations = ((0, 0, 0),)
        else:
            combinations = itertools.product(
                algorithm["two_hop_threshold_percent"],
                algorithm["max_two_hop_percent"],
                algorithm["hop_penalty_percent"],
            )
        for threshold, cap, penalty in combinations:
            if mode == "adaptive" and cap == 0:
                continue
            rows.append(
                {
                    "mode": mode,
                    "two_hop_threshold_percent": threshold,
                    "max_two_hop_percent": cap,
                    "hop_penalty_percent": penalty,
                    "planner_seed": algorithm["planner_seed"],
                    "planner_chunk_size": algorithm["planner_chunk_size"],
                }
            )
    return rows


def _pareto_frontier(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep policies not dominated in pair/source peaks and added hop bytes."""

    def objectives(row: dict[str, Any]) -> tuple[float, float, int]:
        oracle = row["oracle"]
        return (
            float(oracle["pair_peak"]),
            float(oracle["source_peak"]),
            int(oracle["extra_local_hop_bytes"]),
        )

    ordered = sorted(
        rows,
        key=lambda row: (
            *objectives(row),
            json.dumps(row["algorithm"], sort_keys=True),
        ),
    )
    frontier: list[dict[str, Any]] = []
    seen_objectives: set[tuple[float, int]] = set()
    for row in ordered:
        current = objectives(row)
        if current in seen_objectives:
            continue
        seen_objectives.add(current)
        dominated = any(
            other is not row
            and all(left <= right for left, right in zip(objectives(other), current))
            and objectives(other) != current
            for other in ordered
        )
        if not dominated:
            frontier.append(row)
    return frontier


def _bounded_frontier(rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    if len(rows) <= budget:
        return rows
    rows = sorted(
        rows,
        key=lambda row: (
            float(row["oracle"]["pair_peak"]),
            float(row["oracle"]["source_peak"]),
            -int(row["oracle"]["extra_local_hop_bytes"]),
        ),
    )
    if budget == 1:
        return [rows[0]]
    indices = {
        round(position * (len(rows) - 1) / (budget - 1)) for position in range(budget)
    }
    return [rows[index] for index in sorted(indices)]


def _select_candidate_rows(
    rows: list[dict[str, Any]], budget: int
) -> list[dict[str, Any]]:
    """Reserve one slot for the one-hop causal control, then sample Pareto rows."""

    controls = [row for row in rows if row["algorithm"]["mode"] == "one_hop"]
    _require(len(controls) == 1, "search space must produce one one-hop control")
    control = controls[0]
    if budget == 1:
        return [control]
    frontier = [row for row in _pareto_frontier(rows) if row is not control]
    selected = [control, *_bounded_frontier(frontier, budget - 1)]
    return sorted(
        selected,
        key=lambda row: (
            float(row["oracle"]["pair_peak"]),
            float(row["oracle"]["source_peak"]),
            int(row["oracle"]["extra_local_hop_bytes"]),
            json.dumps(row["algorithm"], sort_keys=True),
        ),
    )


def _candidate_id(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return f"rb_{hashlib.sha256(encoded).hexdigest()[:12]}"


def _object_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _plan_identity(
    frozen_default_commit: str,
    source_commit: str,
    search_space_sha256: str,
    workload: dict[str, Any],
    acceptance_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "frozen_default_commit": frozen_default_commit,
        "source_commit": source_commit,
        "search_space_sha256": search_space_sha256,
        "workload": workload,
        "acceptance_contract": acceptance_contract,
    }


def _build_benchmark_argv(
    candidate: dict[str, Any],
    workload: dict[str, Any],
    output: Path,
    run_id: str,
    python_executable: str = sys.executable,
) -> list[str]:
    profile = candidate["profile"]
    argv = [
        python_executable,
        "-B",
        str(_ELASTIC_TESTS / "bench_rail_balance_hybrid_multinode.py"),
        "--run-id",
        run_id,
        "--case",
        workload["case"],
        "--output",
        str(output),
        "--candidate-mode",
        profile["mode"],
        "--num-processes",
        str(workload["gpus_per_node"]),
        "--num-tokens",
        str(workload["tokens_per_rank"]),
        "--hidden",
        str(workload["hidden"]),
        "--num-topk",
        str(workload["topk"]),
        "--num-experts",
        str(workload["num_experts"]),
        "--num-sms",
        str(profile["num_sms"]),
        "--num-allocated-qps",
        str(profile["num_allocated_qps"]),
        "--rail-policy",
        "all",
        "--rail-threshold-percent",
        "0",
        "--two-hop-threshold-percent",
        str(profile["two_hop_threshold_percent"]),
        "--max-two-hop-percent",
        str(profile["max_two_hop_percent"]),
        "--hop-penalty-percent",
        str(profile["hop_penalty_percent"]),
        "--warmup-iters",
        str(workload["warmup_iters"]),
        "--steady-iters",
        str(workload["steady_iters"]),
        "--order-repeats",
        "1",
        "--sl-idx",
        str(BENCHMARK_SL_INDEX),
        "--seed",
        str(BENCHMARK_DATA_SEED),
    ]
    if workload["workload_json"] is not None:
        argv.extend(("--workload-json", workload["workload_json"]))
    if profile["proxy_slots_per_rank"] is not None:
        argv.extend(("--proxy-slots-per-rank", str(profile["proxy_slots_per_rank"])))
    return argv


def _load_json_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return _load_json_with_sha256(path)[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _source_identity(frozen_default_commit: str) -> dict[str, Any]:
    source_commit = _git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", frozen_default_commit, source_commit),
        cwd=_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _require(
        ancestor.returncode == 0,
        "frozen_default_commit is not an ancestor of the current source",
    )
    changed_paths = tuple(
        filter(
            None,
            _git("diff", "--name-only", f"{frozen_default_commit}..HEAD").splitlines(),
        )
    )
    return {
        "source_commit": source_commit,
        "source_clean": not bool(_git("status", "--porcelain")),
        "source_paths_changed_since_frozen": list(changed_paths),
        "production_paths_changed_since_frozen": [
            path for path in changed_paths if path.startswith(("csrc/", "deep_ep/"))
        ],
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    space = _load_json(args.space)
    _validate_space(space)
    source_identity = _source_identity(space["frozen_default_commit"])
    workload_json = args.workload_json.resolve() if args.workload_json else None
    _require(
        (args.case == "trace") == (workload_json is not None),
        "case=trace requires --workload-json, and only trace accepts it",
    )
    _require(
        args.num_experts % (args.num_nodes * args.gpus_per_node) == 0,
        "num_experts must be divisible by nodes * GPUs",
    )
    _require(
        2 <= args.num_nodes <= 32 and 2 <= args.gpus_per_node <= 32,
        "benchmark topology dimensions must be in [2, 32]",
    )
    _require(args.topk <= 32, "topk must be at most 32")
    _require(args.num_experts <= 2048, "num_experts must be at most 2048")
    _require(
        args.num_experts // (args.num_nodes * args.gpus_per_node) <= 256,
        "experts per rank must be at most 256",
    )
    _require(
        args.num_nodes * args.gpus_per_node * args.tokens_per_rank < (1 << 31),
        "global token index must fit int32",
    )
    _require(args.hidden % 256 == 0, "hidden must be a multiple of 256")
    _require(
        args.warmup_iters >= 10 and args.steady_iters >= 100,
        "claim attempts require at least 10 warmup and 100 steady iterations",
    )
    if args.case == "trace":
        _require(
            args.topk <= args.num_experts // (args.num_nodes * args.gpus_per_node),
            "trace replay requires topk <= experts per rank",
        )
    _require(
        math.isfinite(args.minimum_speedup) and args.minimum_speedup > 1.0,
        "minimum_speedup must be finite and greater than 1",
    )
    _require(
        args.minimum_runs >= MINIMUM_FREEZE_RUNS,
        f"minimum_runs must be at least {MINIMUM_FREEZE_RUNS}",
    )
    acceptance_contract = {
        "minimum_aggregate_speedup": args.minimum_speedup,
        "minimum_paired_median_speedup": args.minimum_speedup,
        "minimum_p95_speedup": 1.0,
        "minimum_distinct_attempts": args.minimum_runs,
    }
    workload = {
        "case": args.case,
        "trace_semantics": (
            "endpoint_count_v1_diagnostic"
            if workload_json is not None
            else "synthetic_diagnostic"
        ),
        "workload_json": str(workload_json) if workload_json else None,
        "workload_json_sha256": _sha256(workload_json) if workload_json else None,
        "num_nodes": args.num_nodes,
        "gpus_per_node": args.gpus_per_node,
        "tokens_per_rank": args.tokens_per_rank,
        "topk": args.topk,
        "num_experts": args.num_experts,
        "hidden": args.hidden,
        "warmup_iters": args.warmup_iters,
        "steady_iters": args.steady_iters,
    }
    evaluated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, algorithm in enumerate(_enumerate_algorithm_configs(space)):
        try:
            report = build_reference_report(
                run_id=f"autotune-oracle-{index}",
                case=args.case,
                mode=str(algorithm["mode"]),
                num_nodes=args.num_nodes,
                rails=args.gpus_per_node,
                tokens_per_rank=args.tokens_per_rank,
                topk=args.topk,
                num_experts=args.num_experts,
                hidden=args.hidden,
                chunk_size=int(algorithm["planner_chunk_size"]),
                two_hop_threshold_percent=int(algorithm["two_hop_threshold_percent"]),
                max_two_hop_percent=int(algorithm["max_two_hop_percent"]),
                hop_penalty_percent=int(algorithm["hop_penalty_percent"]),
                seed=int(algorithm["planner_seed"]),
                proxy_slots_per_rank=space["proxy_slots_per_rank"],
                workload_json=workload_json,
            )
        except ValueError as error:
            rejected.append({"algorithm": algorithm, "reason": str(error)})
            continue
        evaluated.append(
            {
                "algorithm": algorithm,
                "oracle": {
                    "pair_peak": report["rail_load_before_after"]["after_stats"]["max"],
                    "pair_cv": report["rail_load_before_after"]["after_stats"]["cv"],
                    "source_peak": report["rail_load_before_after"][
                        "after_source_stats"
                    ]["max"],
                    "source_cv": report["rail_load_before_after"]["after_source_stats"][
                        "cv"
                    ],
                    "extra_local_hop_bytes": report["path_distribution"][
                        "extra_local_hop_bytes"
                    ],
                    "two_hop_ratio": report["path_distribution"]["two_hop_ratio"],
                },
            }
        )
    _require(evaluated, "all algorithm candidates were rejected by the CPU oracle")
    frontier = _select_candidate_rows(evaluated, space["max_pareto_candidates"])
    search_space_sha256 = _sha256(args.space)
    plan_identity = _plan_identity(
        space["frozen_default_commit"],
        source_identity["source_commit"],
        search_space_sha256,
        workload,
        acceptance_contract,
    )
    results_dir = args.results_dir.resolve()
    _require(
        not results_dir.is_relative_to(_ROOT)
        and not args.output.resolve().is_relative_to(_ROOT),
        "plan output and results directory must be outside the Git worktree",
    )
    candidates = []
    for row, execution in itertools.product(frontier, space["execution_presets"]):
        profile = {
            **row["algorithm"],
            "execution_preset": execution["id"],
            "execution_strategy": execution["strategy"],
            "num_sms": execution["num_sms"],
            "num_allocated_qps": execution["num_allocated_qps"],
            "proxy_slots_per_rank": space["proxy_slots_per_rank"],
        }
        identifier = _candidate_id(
            {
                "plan_identity": plan_identity,
                "profile": profile,
                "oracle": row["oracle"],
            }
        )
        candidate = {
            "id": identifier,
            "profile": profile,
            "oracle": row["oracle"],
        }
        candidate["environment"] = {
            "WORLD_SIZE": str(args.num_nodes),
            "RANK": "<NODE_RANK>",
            "MASTER_ADDR": "<MASTER_ADDR>",
            "MASTER_PORT": "<MASTER_PORT>",
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(index) for index in range(args.gpus_per_node)
            ),
            "EP_DISABLE_GIN": "0",
            "OMP_NUM_THREADS": "1",
        }
        candidate["attempts"] = []
        for attempt in range(1, acceptance_contract["minimum_distinct_attempts"] + 1):
            run_id = f"{identifier}-attempt-{attempt:02d}"
            result_path = results_dir / f"{run_id}.json"
            candidate["attempts"].append(
                {
                    "run_id": run_id,
                    "result_path": str(result_path),
                    "argv": _build_benchmark_argv(
                        candidate, workload, result_path, run_id
                    ),
                }
            )
        candidates.append(candidate)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "rail_balance_autotune_plan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_default_commit": space["frozen_default_commit"],
        **source_identity,
        "planner_side_effects": "writes plan only; does not launch benchmarks",
        "performance_claim_eligible": False,
        "selection_scope": "endpoint_count_or_synthetic_diagnostic",
        "acceptance_contract": acceptance_contract,
        "search_space_sha256": search_space_sha256,
        "plan_identity": plan_identity,
        "workload": workload,
        "evaluated_algorithm_count": len(evaluated),
        "evaluated_algorithms": evaluated,
        "rejected": rejected,
        "candidates": candidates,
    }


def _result_metrics(report: dict[str, Any]) -> dict[str, float]:
    candidate_mode = report["candidate"]["mode"]
    blocks = report.get("blocks")
    manifest = report["benchmark_manifest"]
    expected_order = manifest["expanded_order"]
    steady_iters = manifest["steady_iters"]
    world_size = (
        report["workload"]["node_count"] * report["workload"]["local_processes"]
    )
    _require(
        isinstance(blocks, list)
        and len(blocks) == len(expected_order)
        and [block.get("mode") for block in blocks] == expected_order,
        "result timing blocks do not match the pre-registered order",
    )
    by_mode: dict[str, list[float]] = {"off": [], candidate_mode: []}
    pooled: dict[str, list[float]] = {"off": [], candidate_mode: []}
    by_pair: dict[int, dict[str, float]] = {}
    for block_index, block in enumerate(blocks):
        mode = block.get("mode")
        _require(mode in by_mode, "result block has an unexpected mode")
        expected_pattern = "ABBA" if block_index < 4 else "BAAB"
        _require(
            block.get("block_index") == block_index
            and block.get("pair_index") == block_index // 2
            and block.get("order_repeat") == 0
            and block.get("pattern") == expected_pattern
            and block.get("pattern_position") == block_index % 4,
            "result block pairing identity is invalid",
        )
        samples = block.get("samples")
        _require(
            isinstance(samples, list) and len(samples) == steady_iters,
            "result block has the wrong steady sample count",
        )
        values = []
        for iteration, sample in enumerate(samples):
            rank_raw = sample.get("rank_raw")
            _require(
                sample.get("iteration") == iteration
                and isinstance(rank_raw, list)
                and [row.get("rank") for row in rank_raw] == list(range(world_size)),
                "result rank-raw sample is incomplete",
            )
            raw_max = max(float(row["roundtrip_cuda_ms"]) for row in rank_raw)
            reported_max = float(sample["rank_max"]["roundtrip_cuda_ms"])
            _require(raw_max == reported_max, "rank-max sample disagrees with rank raw")
            values.append(raw_max)
        _require(
            all(math.isfinite(value) and value > 0 for value in values),
            "result contains invalid raw timing samples",
        )
        median = statistics.median(values)
        by_mode[mode].append(median)
        pooled[mode].extend(values)
        pair_index = block.get("pair_index")
        _require(type(pair_index) is int, "result block pair index is invalid")
        _require(mode not in by_pair.setdefault(pair_index, {}), "duplicate pair mode")
        by_pair[pair_index][mode] = median

    pairs = []
    for pair in by_pair.values():
        _require(set(pair) == set(by_mode), "timing pair is incomplete")
        pairs.append(pair["off"] / pair[candidate_mode])

    def percentile(values: list[float], quantile: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    baseline_median = statistics.median(by_mode["off"])
    candidate_median = statistics.median(by_mode[candidate_mode])
    return {
        "aggregate_speedup": baseline_median / candidate_median,
        "paired_median_speedup": statistics.median(pairs),
        "p95_speedup": percentile(pooled["off"], 0.95)
        / percentile(pooled[candidate_mode], 0.95),
    }


def _validate_plan_for_freeze(plan: dict[str, Any]) -> None:
    _require(plan.get("schema_version") == SCHEMA_VERSION, "invalid plan schema")
    _require(plan.get("kind") == "rail_balance_autotune_plan", "invalid plan kind")
    _require(plan.get("source_clean") is True, "autotune plan used a dirty tree")
    source_commit = plan.get("source_commit")
    _require(
        isinstance(source_commit, str) and _COMMIT.fullmatch(source_commit),
        "autotune plan source commit is invalid",
    )
    workload = plan.get("workload")
    _require(isinstance(workload, dict), "autotune plan workload is invalid")
    frozen_default_commit = plan.get("frozen_default_commit")
    search_space_sha256 = plan.get("search_space_sha256")
    _require(
        isinstance(frozen_default_commit, str)
        and _COMMIT.fullmatch(frozen_default_commit),
        "frozen default commit is invalid",
    )
    _require(
        isinstance(search_space_sha256, str) and _SHA256.fullmatch(search_space_sha256),
        "search-space hash is invalid",
    )
    contract = _exact_keys(
        plan.get("acceptance_contract"),
        {
            "minimum_aggregate_speedup",
            "minimum_paired_median_speedup",
            "minimum_p95_speedup",
            "minimum_distinct_attempts",
        },
        "acceptance_contract",
    )
    _require(
        type(contract["minimum_distinct_attempts"]) is int
        and contract["minimum_distinct_attempts"] >= MINIMUM_FREEZE_RUNS,
        f"minimum_distinct_attempts must be at least {MINIMUM_FREEZE_RUNS}",
    )
    for name in (
        "minimum_aggregate_speedup",
        "minimum_paired_median_speedup",
        "minimum_p95_speedup",
    ):
        value = contract[name]
        lower_bound = 1.0
        _require(
            type(value) in (int, float)
            and math.isfinite(value)
            and (
                value >= lower_bound
                if name == "minimum_p95_speedup"
                else value > lower_bound
            ),
            f"{name} is invalid",
        )
    candidates = plan.get("candidates")
    _require(isinstance(candidates, list) and candidates, "plan has no candidates")
    expected_plan_identity = _plan_identity(
        frozen_default_commit,
        source_commit,
        search_space_sha256,
        workload,
        contract,
    )
    _require(
        plan.get("plan_identity") == expected_plan_identity,
        "plan identity is corrupt",
    )
    identifiers: set[str] = set()
    for candidate in candidates:
        _require(isinstance(candidate, dict), "plan candidate is invalid")
        profile = _exact_keys(
            candidate.get("profile"),
            {
                "mode",
                "two_hop_threshold_percent",
                "max_two_hop_percent",
                "hop_penalty_percent",
                "planner_seed",
                "planner_chunk_size",
                "execution_preset",
                "execution_strategy",
                "num_sms",
                "num_allocated_qps",
                "proxy_slots_per_rank",
            },
            "candidate profile",
        )
        _require(profile["mode"] in {"one_hop", "adaptive"}, "profile mode is invalid")
        _require(
            profile["planner_seed"] == PRODUCTION_PLANNER_SEED
            and profile["planner_chunk_size"] == PRODUCTION_PLANNER_CHUNK_SIZE,
            "profile planner identity differs from production",
        )
        _require(
            isinstance(profile["execution_preset"], str)
            and _ID.fullmatch(profile["execution_preset"]) is not None
            and profile["execution_strategy"] == "v2_auto"
            and profile["num_sms"] == profile["num_allocated_qps"] == 0,
            "v1 profile must use V2-auto SM/QP selection",
        )
        _require(
            type(profile["two_hop_threshold_percent"]) is int
            and 0 <= profile["two_hop_threshold_percent"] <= 10000
            and type(profile["max_two_hop_percent"]) is int
            and 0 <= profile["max_two_hop_percent"] <= 100
            and type(profile["hop_penalty_percent"]) is int
            and 0 <= profile["hop_penalty_percent"] <= 10000,
            "profile hop controls are invalid",
        )
        if profile["mode"] == "one_hop":
            _require(
                profile["two_hop_threshold_percent"]
                == profile["max_two_hop_percent"]
                == profile["hop_penalty_percent"]
                == 0,
                "one-hop profile must normalize all two-hop controls",
            )
        _require(
            profile["proxy_slots_per_rank"] is None
            or (
                type(profile["proxy_slots_per_rank"]) is int
                and profile["proxy_slots_per_rank"] > 0
            ),
            "profile proxy capacity is invalid",
        )
        expected = _candidate_id(
            {
                "plan_identity": expected_plan_identity,
                "profile": profile,
                "oracle": candidate.get("oracle"),
            }
        )
        _require(candidate.get("id") == expected, "candidate identity is corrupt")
        _require(expected not in identifiers, "candidate id is duplicated")
        identifiers.add(expected)
        attempts = candidate.get("attempts")
        _require(
            isinstance(attempts, list)
            and len(attempts) == contract["minimum_distinct_attempts"],
            "candidate attempt manifest is invalid",
        )
        expected_run_ids = {
            f"{expected}-attempt-{index:02d}" for index in range(1, len(attempts) + 1)
        }
        _require(
            {attempt.get("run_id") for attempt in attempts} == expected_run_ids,
            "candidate attempt IDs are invalid",
        )


def _freeze_profile(
    plan: dict[str, Any],
    candidate_id: str,
    reports: Sequence[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    _validate_plan_for_freeze(plan)
    contract = plan["acceptance_contract"]
    _require(
        len(reports) >= contract["minimum_distinct_attempts"],
        "not enough distinct result files",
    )
    matching = [row for row in plan["candidates"] if row["id"] == candidate_id]
    _require(len(matching) == 1, "candidate id is absent or duplicated")
    candidate = matching[0]
    profile = candidate["profile"]
    commits: set[str] = set()
    evidence = []
    evidence_hashes: set[str] = set()
    speedups = []
    resolved_proxy_slots: set[int] = set()
    resolved_num_sms: set[int] = set()
    resolved_num_qps: set[int] = set()
    resolved_allocated_qps: set[int] = set()
    measurement_identities: set[str] = set()
    run_ids: set[str] = set()
    attempts = {attempt["run_id"]: attempt for attempt in candidate["attempts"]}
    seen_paths: set[Path] = set()
    for path, report in reports:
        path = path.resolve()
        _require(path not in seen_paths, "duplicate result path")
        seen_paths.add(path)
        run_id = report.get("run_id")
        _require(isinstance(run_id, str), "result run_id is invalid")
        _require(run_id not in run_ids, "duplicate result run_id")
        run_ids.add(run_id)
        _require(run_id in attempts, "result run_id was not pre-registered")
        _require(
            path == Path(attempts[run_id]["result_path"]).resolve(),
            "result path differs from the pre-registered attempt",
        )
        _require(
            report.get("claim_scope") == "real_multinode",
            "result is not real_multinode",
        )
        _require(
            report.get("correctness", {}).get("passed") is True
            and report["correctness"].get("off_candidate_equal") is True,
            "result correctness failed",
        )
        _require(
            report.get("eligibility", {}).get("performance_claim_eligible") is True,
            "result is not performance-claim eligible",
        )
        result_candidate = report.get("candidate", {})
        result_kernel = report.get("kernel_config", {})
        expected = {
            "mode": profile["mode"],
            "two_hop_threshold_percent": profile["two_hop_threshold_percent"],
            "max_two_hop_percent": profile["max_two_hop_percent"],
            "hop_penalty_percent": profile["hop_penalty_percent"],
            "num_sms": profile["num_sms"],
            "num_allocated_qps": profile["num_allocated_qps"],
        }
        actual = {
            "mode": result_candidate.get("mode"),
            "two_hop_threshold_percent": result_candidate.get(
                "two_hop_threshold_percent"
            ),
            "max_two_hop_percent": result_candidate.get("max_two_hop_percent"),
            "hop_penalty_percent": result_candidate.get("hop_penalty_percent"),
            "num_sms": result_kernel.get("num_sms"),
            "num_allocated_qps": result_kernel.get("num_allocated_qps"),
        }
        _require(actual == expected, "result configuration does not match candidate")
        _require(
            result_candidate.get("policy") == "all"
            and result_candidate.get("threshold_percent") == 0,
            "result used a different RailBalance policy or threshold",
        )
        for key, target in (
            ("resolved_num_sms", resolved_num_sms),
            ("resolved_num_qps", resolved_num_qps),
            ("resolved_num_allocated_qps", resolved_allocated_qps),
        ):
            value = result_kernel.get(key)
            _require(type(value) is int and value > 0, f"result {key} is invalid")
            target.add(value)
        _require(
            result_kernel["resolved_num_qps"]
            <= result_kernel["resolved_num_allocated_qps"],
            "resolved V2 QPs exceed allocated QPs",
        )
        _require(
            result_kernel.get("baseline_resolved")
            == {
                "num_sms": result_kernel["resolved_num_sms"],
                "num_qps": result_kernel["resolved_num_qps"],
                "num_allocated_qps": result_kernel["resolved_num_allocated_qps"],
            },
            "off baseline and candidate did not use the same V2-auto resources",
        )
        expected_resolution = {
            "num_sms": result_kernel["resolved_num_sms"],
            "num_qps": result_kernel["resolved_num_qps"],
            "num_allocated_qps": result_kernel["resolved_num_allocated_qps"],
        }
        _require(
            isinstance(report.get("blocks"), list)
            and bool(report["blocks"])
            and all(
                block.get("resolved_execution") == expected_resolution
                for block in report["blocks"]
            ),
            "timed blocks did not use the recorded V2-auto resources",
        )
        result_planner = report.get("planner_config", {})
        _require(
            result_planner.get("planner_seed") == profile["planner_seed"]
            and result_planner.get("chunk_size") == profile["planner_chunk_size"],
            "result planner identity does not match candidate",
        )
        if profile["proxy_slots_per_rank"] is not None:
            _require(
                report.get("workload", {}).get("proxy_slots_per_rank")
                == profile["proxy_slots_per_rank"],
                "result proxy capacity does not match candidate",
            )
        result_workload = report.get("workload", {})
        expected_workload = {
            "case": plan["workload"]["case"],
            "node_count": plan["workload"]["num_nodes"],
            "local_processes": plan["workload"]["gpus_per_node"],
            "tokens_per_rank": plan["workload"]["tokens_per_rank"],
            "hidden": plan["workload"]["hidden"],
            "topk": plan["workload"]["topk"],
            "experts": plan["workload"]["num_experts"],
        }
        _require(
            {key: result_workload.get(key) for key in expected_workload}
            == expected_workload,
            "result workload does not match autotune plan",
        )
        benchmark_manifest = report.get("benchmark_manifest", {})
        _require(
            benchmark_manifest.get("run_id") == run_id
            and Path(benchmark_manifest.get("output", "")).resolve() == path,
            "result manifest does not match its pre-registered attempt",
        )
        _require(
            benchmark_manifest.get("workload_sha256")
            == plan["workload"].get("workload_json_sha256"),
            "result route trace differs from autotune plan",
        )
        expected_order = [
            "off",
            profile["mode"],
            profile["mode"],
            "off",
            profile["mode"],
            "off",
            "off",
            profile["mode"],
        ]
        _require(
            benchmark_manifest.get("warmup_iters") == plan["workload"]["warmup_iters"]
            and benchmark_manifest.get("steady_iters")
            == plan["workload"]["steady_iters"]
            and benchmark_manifest.get("order_repeats") == 1
            and benchmark_manifest.get("expanded_order") == expected_order,
            "result measurement schedule differs from autotune plan",
        )
        resolved_slots = result_workload.get("proxy_slots_per_rank")
        _require(
            type(resolved_slots) is int and resolved_slots > 0,
            "result proxy capacity is invalid",
        )
        resolved_proxy_slots.add(resolved_slots)
        metrics = _result_metrics(report)
        speedup = metrics["aggregate_speedup"]
        _require(
            all(math.isfinite(value) and value > 0 for value in metrics.values()),
            "result performance metrics are invalid",
        )
        _require(
            speedup > contract["minimum_aggregate_speedup"]
            and metrics["paired_median_speedup"]
            > contract["minimum_paired_median_speedup"],
            "median speedup gate failed",
        )
        _require(
            metrics["p95_speedup"] >= contract["minimum_p95_speedup"],
            "candidate pooled p95 regressed against its V2 baseline",
        )
        identity = report.get("identity")
        identity_fields = {
            "git_commit",
            "git_dirty",
            "extension_path",
            "extension_sha256",
            "torch",
            "cuda",
            "nccl",
            "gpu",
            "compute_capability",
            "node_count",
            "local_processes",
            "config",
            "environment",
        }
        _require(
            isinstance(identity, dict) and identity_fields <= set(identity),
            "result measurement identity is incomplete",
        )
        identity_config = identity["config"]
        expected_identity_config = {
            "case": plan["workload"]["case"],
            "num_tokens": plan["workload"]["tokens_per_rank"],
            "hidden": plan["workload"]["hidden"],
            "num_topk": plan["workload"]["topk"],
            "num_experts": plan["workload"]["num_experts"],
            "num_sms": profile["num_sms"],
            "num_allocated_qps": profile["num_allocated_qps"],
            "proxy_slots_per_rank": profile["proxy_slots_per_rank"],
            "rail_balance_policy": "all",
            "rail_balance_threshold_percent": 0,
            "candidate_mode": profile["mode"],
            "two_hop_threshold_percent": profile["two_hop_threshold_percent"],
            "max_two_hop_percent": profile["max_two_hop_percent"],
            "hop_penalty_percent": profile["hop_penalty_percent"],
            "sl_idx": BENCHMARK_SL_INDEX,
            "seed": BENCHMARK_DATA_SEED,
        }
        _require(
            identity_config == expected_identity_config,
            "result identity config differs from the pre-registered command",
        )
        measurement_identities.add(_object_sha256(identity))
        commit = report.get("git_commit")
        _require(
            isinstance(commit, str) and _COMMIT.fullmatch(commit),
            "result Git commit is invalid",
        )
        _require(
            commit == plan["source_commit"], "result commit differs from autotune plan"
        )
        _require(
            identity["git_commit"] == commit and identity["git_dirty"] is False,
            "result top-level and measurement Git identities differ",
        )
        commits.add(commit)
        speedups.append(speedup)
        evidence_hash = _sha256(path)
        _require(evidence_hash not in evidence_hashes, "duplicate result content")
        evidence_hashes.add(evidence_hash)
        evidence.append({"path": str(path), "sha256": evidence_hash, **metrics})
    _require(len(commits) == 1, "results were produced by different commits")
    _require(
        len(resolved_proxy_slots) == 1, "results resolved different proxy capacities"
    )
    _require(len(resolved_num_sms) == 1, "results resolved different V2 SM counts")
    _require(len(resolved_num_qps) == 1, "results resolved different V2 QP counts")
    _require(
        len(resolved_allocated_qps) == 1,
        "results resolved different allocated QP counts",
    )
    _require(
        len(measurement_identities) == 1,
        "results have different binary, hardware, or configuration identities",
    )
    _require(
        run_ids == {attempt["run_id"] for attempt in candidate["attempts"]},
        "results do not match the pre-registered attempt IDs",
    )
    semantics = plan["workload"].get("trace_semantics")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "rail_balance_frozen_profile",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_default_commit": plan["frozen_default_commit"],
        "candidate_commit": next(iter(commits)),
        "candidate_id": candidate_id,
        "plan_canonical_sha256": _object_sha256(plan),
        "profile": profile,
        "resolved_proxy_slots_per_rank": next(iter(resolved_proxy_slots)),
        "resolved_v2_execution": {
            "num_sms": next(iter(resolved_num_sms)),
            "num_qps": next(iter(resolved_num_qps)),
            "num_allocated_qps": next(iter(resolved_allocated_qps)),
        },
        "measurement_identity_sha256": next(iter(measurement_identities)),
        "measurement_identity": identity,
        "oracle": candidate["oracle"],
        "workload": plan["workload"],
        "acceptance_contract": contract,
        "evidence": evidence,
        "speedup_summary": {
            "minimum": min(speedups),
            "median": statistics.median(speedups),
            "maximum": max(speedups),
        },
        "activation": "explicit_only",
        "production_consumed": False,
        "production_promotion_eligible": False,
        "claim_scope": (
            "real_multinode_endpoint_count_diagnostic"
            if semantics == "endpoint_count_v1_diagnostic"
            else "real_multinode_synthetic_diagnostic"
        ),
        "missing_promotion_evidence": [
            "continuous_token_route_trace_v2",
            "held_out_confirmatory_runs",
            "runtime_path_and_proxy_counters",
            "direct_one_hop_vs_adaptive_paired_ablation",
        ],
        "v2_auto_reused": profile["execution_strategy"] == "v2_auto",
    }


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="CPU-oracle prune and emit commands")
    plan.add_argument("--space", required=True, type=Path)
    plan.add_argument("--case", required=True, choices=BENCHMARK_CASES)
    plan.add_argument("--workload-json", type=Path)
    plan.add_argument("--num-nodes", type=_positive, default=2)
    plan.add_argument("--gpus-per-node", type=_positive, default=8)
    plan.add_argument("--tokens-per-rank", type=_positive, default=4096)
    plan.add_argument("--topk", type=_positive, default=8)
    plan.add_argument("--num-experts", type=_positive, default=256)
    plan.add_argument("--hidden", type=_positive, default=7168)
    plan.add_argument("--warmup-iters", type=_positive, default=10)
    plan.add_argument("--steady-iters", type=_positive, default=100)
    plan.add_argument("--minimum-speedup", type=float, required=True)
    plan.add_argument("--minimum-runs", type=_positive, default=MINIMUM_FREEZE_RUNS)
    plan.add_argument("--results-dir", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)

    freeze = subparsers.add_parser(
        "freeze", help="freeze an explicitly chosen candidate"
    )
    freeze.add_argument("--plan", required=True, type=Path)
    freeze.add_argument("--candidate-id", required=True)
    freeze.add_argument("--result", required=True, action="append", type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan":
        result = _plan(args)
    else:
        plan, plan_file_sha256 = _load_json_with_sha256(args.plan)
        result = _freeze_profile(
            plan,
            args.candidate_id,
            [(path, _load_json(path)) for path in args.result],
        )
        result["plan_file_sha256"] = plan_file_sha256
    _write_json(args.output.resolve(), result)
    print(f"PASS {args.command} -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
