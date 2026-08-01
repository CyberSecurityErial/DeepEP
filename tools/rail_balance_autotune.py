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
_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


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
        type(algorithm["planner_seed"]) is int and algorithm["planner_seed"] >= 0,
        "planner_seed must be a nonnegative integer",
    )
    _require(
        type(algorithm["planner_chunk_size"]) is int
        and algorithm["planner_chunk_size"] >= 1,
        "planner_chunk_size must be a positive integer",
    )

    presets = space["execution_presets"]
    _require(isinstance(presets, list) and presets, "execution_presets is empty")
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
        _require(
            preset["strategy"] in {"v2_auto", "explicit_evidence"},
            "execution strategy must be v2_auto or explicit_evidence",
        )
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
        if preset["strategy"] == "v2_auto":
            _require(
                preset["num_sms"] == preset["num_allocated_qps"] == 0,
                "v2_auto requires num_sms=num_allocated_qps=0",
            )
        else:
            _require(
                preset["num_sms"] >= 2,
                "explicit_evidence requires an explicit num_sms",
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
    """Keep policies not dominated in Rail peak and added local-hop bytes."""

    def objectives(row: dict[str, Any]) -> tuple[float, int]:
        oracle = row["oracle"]
        return float(oracle["pair_peak"]), int(oracle["extra_local_hop_bytes"])

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
            and objectives(other)[0] <= current[0]
            and objectives(other)[1] <= current[1]
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
            -int(row["oracle"]["extra_local_hop_bytes"]),
        ),
    )
    if budget == 1:
        return [rows[0]]
    indices = {
        round(position * (len(rows) - 1) / (budget - 1)) for position in range(budget)
    }
    return [rows[index] for index in sorted(indices)]


def _candidate_id(profile: dict[str, Any]) -> str:
    encoded = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    return f"rb_{hashlib.sha256(encoded).hexdigest()[:12]}"


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
    ]
    if workload["workload_json"] is not None:
        argv.extend(("--workload-json", workload["workload_json"]))
    if profile["proxy_slots_per_rank"] is not None:
        argv.extend(("--proxy-slots-per-rank", str(profile["proxy_slots_per_rank"])))
    return argv


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


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
    return {
        "source_commit": source_commit,
        "source_clean": not bool(_git("status", "--porcelain")),
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
    workload = {
        "case": args.case,
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
                    "extra_local_hop_bytes": report["path_distribution"][
                        "extra_local_hop_bytes"
                    ],
                    "two_hop_ratio": report["path_distribution"]["two_hop_ratio"],
                },
            }
        )
    _require(evaluated, "all algorithm candidates were rejected by the CPU oracle")
    frontier = _bounded_frontier(
        _pareto_frontier(evaluated), space["max_pareto_candidates"]
    )
    results_dir = args.results_dir.resolve()
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
        identifier = _candidate_id(profile)
        result_path = results_dir / f"{identifier}.json"
        candidate = {
            "id": identifier,
            "profile": profile,
            "oracle": row["oracle"],
            "result_path": str(result_path),
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
        candidate["argv"] = _build_benchmark_argv(
            candidate, workload, result_path, identifier
        )
        candidates.append(candidate)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "rail_balance_autotune_plan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_default_commit": space["frozen_default_commit"],
        **source_identity,
        "production_code_modified": False,
        "performance_claim_eligible": False,
        "selection_scope": "cpu_oracle_pareto_only",
        "search_space_sha256": _sha256(args.space),
        "workload": workload,
        "evaluated_algorithm_count": len(evaluated),
        "rejected": rejected,
        "candidates": candidates,
    }


def _result_speedup(report: dict[str, Any]) -> float:
    return float(
        report["comparison"]["roundtrip_cuda_ms"]["speedup_baseline_over_candidate"]
    )


def _freeze_profile(
    plan: dict[str, Any],
    candidate_id: str,
    reports: Sequence[tuple[Path, dict[str, Any]]],
    minimum_speedup: float,
    minimum_runs: int = 2,
) -> dict[str, Any]:
    _require(minimum_runs >= 2, "minimum_runs must be at least 2")
    _require(len(reports) >= minimum_runs, "not enough independent result files")
    _require(
        plan.get("source_clean") is True, "autotune plan was made from a dirty tree"
    )
    _require(
        isinstance(plan.get("source_commit"), str)
        and _COMMIT.fullmatch(plan["source_commit"]),
        "autotune plan source commit is invalid",
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
    seen_paths: set[Path] = set()
    for path, report in reports:
        path = path.resolve()
        _require(path not in seen_paths, "duplicate result path")
        seen_paths.add(path)
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
            benchmark_manifest.get("workload_sha256")
            == plan["workload"].get("workload_json_sha256"),
            "result route trace differs from autotune plan",
        )
        resolved_slots = result_workload.get("proxy_slots_per_rank")
        _require(
            type(resolved_slots) is int and resolved_slots > 0,
            "result proxy capacity is invalid",
        )
        resolved_proxy_slots.add(resolved_slots)
        speedup = _result_speedup(report)
        _require(
            math.isfinite(speedup) and speedup > minimum_speedup, "speedup gate failed"
        )
        commit = report.get("git_commit")
        _require(
            isinstance(commit, str) and _COMMIT.fullmatch(commit),
            "result Git commit is invalid",
        )
        _require(
            commit == plan["source_commit"], "result commit differs from autotune plan"
        )
        commits.add(commit)
        speedups.append(speedup)
        evidence_hash = _sha256(path)
        _require(evidence_hash not in evidence_hashes, "duplicate result content")
        evidence_hashes.add(evidence_hash)
        evidence.append(
            {"path": str(path), "sha256": evidence_hash, "speedup": speedup}
        )
    _require(len(commits) == 1, "results were produced by different commits")
    _require(
        len(resolved_proxy_slots) == 1, "results resolved different proxy capacities"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "rail_balance_frozen_profile",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_default_commit": plan["frozen_default_commit"],
        "candidate_commit": next(iter(commits)),
        "candidate_id": candidate_id,
        "profile": profile,
        "resolved_proxy_slots_per_rank": next(iter(resolved_proxy_slots)),
        "oracle": candidate["oracle"],
        "workload": plan["workload"],
        "evidence": evidence,
        "speedup_summary": {
            "minimum": min(speedups),
            "median": statistics.median(speedups),
            "maximum": max(speedups),
        },
        "activation": "explicit_only",
        "production_consumed": False,
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
    plan.add_argument("--case", required=True)
    plan.add_argument("--workload-json", type=Path)
    plan.add_argument("--num-nodes", type=_positive, default=2)
    plan.add_argument("--gpus-per-node", type=_positive, default=8)
    plan.add_argument("--tokens-per-rank", type=_positive, default=4096)
    plan.add_argument("--topk", type=_positive, default=8)
    plan.add_argument("--num-experts", type=_positive, default=256)
    plan.add_argument("--hidden", type=_positive, default=7168)
    plan.add_argument("--warmup-iters", type=_positive, default=10)
    plan.add_argument("--steady-iters", type=_positive, default=100)
    plan.add_argument("--results-dir", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)

    freeze = subparsers.add_parser(
        "freeze", help="freeze an explicitly chosen candidate"
    )
    freeze.add_argument("--plan", required=True, type=Path)
    freeze.add_argument("--candidate-id", required=True)
    freeze.add_argument("--result", required=True, action="append", type=Path)
    freeze.add_argument("--minimum-speedup", type=float, required=True)
    freeze.add_argument("--minimum-runs", type=_positive, default=5)
    freeze.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan":
        result = _plan(args)
    else:
        _require(
            math.isfinite(args.minimum_speedup) and args.minimum_speedup > 1.0,
            "minimum_speedup must be finite and greater than 1",
        )
        plan = _load_json(args.plan)
        result = _freeze_profile(
            plan,
            args.candidate_id,
            [(path, _load_json(path)) for path in args.result],
            args.minimum_speedup,
            args.minimum_runs,
        )
        result["plan_sha256"] = _sha256(args.plan)
    _write_json(args.output.resolve(), result)
    print(f"PASS {args.command} -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
