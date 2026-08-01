"""CPU-only contracts for the offline RailBalance autotuner."""

from __future__ import annotations

import importlib.util
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[2]
_PROGRAM = _ROOT / "tools" / "rail_balance_autotune.py"
_SPEC = importlib.util.spec_from_file_location("rail_balance_autotune", _PROGRAM)
assert _SPEC is not None and _SPEC.loader is not None
autotune = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = autotune
_SPEC.loader.exec_module(autotune)

_COMMIT = "4e002766d4be7c319687e127a69c1811efe7fe49"


def _space() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "frozen_default_commit": _COMMIT,
        "algorithm": {
            "modes": ["one_hop", "adaptive"],
            "two_hop_threshold_percent": [2, 5],
            "max_two_hop_percent": [10, 25],
            "hop_penalty_percent": [25, 50],
            "planner_seed": 0,
            "planner_chunk_size": 8,
        },
        "execution_presets": [
            {
                "id": "v2_auto",
                "strategy": "v2_auto",
                "num_sms": 0,
                "num_allocated_qps": 0,
            }
        ],
        "proxy_slots_per_rank": 512,
        "max_pareto_candidates": 8,
    }


def _profile() -> dict[str, Any]:
    return {
        "mode": "adaptive",
        "two_hop_threshold_percent": 5,
        "max_two_hop_percent": 25,
        "hop_penalty_percent": 50,
        "planner_seed": 0,
        "planner_chunk_size": 8,
        "execution_preset": "v2_auto",
        "execution_strategy": "v2_auto",
        "num_sms": 0,
        "num_allocated_qps": 0,
        "proxy_slots_per_rank": 512,
    }


def _workload() -> dict[str, Any]:
    return {
        "case": "trace",
        "trace_semantics": "endpoint_count_v1_diagnostic",
        "workload_json": None,
        "workload_json_sha256": None,
        "num_nodes": 2,
        "gpus_per_node": 8,
        "tokens_per_rank": 4096,
        "hidden": 7168,
        "topk": 8,
        "num_experts": 256,
        "warmup_iters": 10,
        "steady_iters": 100,
    }


def _contract() -> dict[str, Any]:
    return {
        "minimum_aggregate_speedup": 1.05,
        "minimum_paired_median_speedup": 1.05,
        "minimum_p95_speedup": 1.0,
        "minimum_distinct_attempts": 5,
    }


def _candidate(result_directory: Path = Path("/tmp")) -> dict[str, Any]:
    profile = _profile()
    workload = _workload()
    contract = _contract()
    oracle = {
        "pair_peak": 10,
        "pair_cv": 0.1,
        "source_peak": 12,
        "source_cv": 0.2,
        "extra_local_hop_bytes": 1024,
        "two_hop_ratio": 0.05,
    }
    plan_identity = autotune._plan_identity(
        _COMMIT, _COMMIT, "b" * 64, workload, contract
    )
    identifier = autotune._candidate_id(
        {"plan_identity": plan_identity, "profile": profile, "oracle": oracle}
    )
    return {
        "id": identifier,
        "profile": profile,
        "oracle": oracle,
        "environment": {},
        "attempts": [
            {
                "run_id": f"{identifier}-attempt-{index:02d}",
                "result_path": str(
                    result_directory / f"{identifier}-attempt-{index:02d}.json"
                ),
                "argv": [],
            }
            for index in range(1, 6)
        ],
    }


def _plan(result_directory: Path = Path("/tmp")) -> dict[str, Any]:
    workload = _workload()
    contract = _contract()
    return {
        "schema_version": 1,
        "kind": "rail_balance_autotune_plan",
        "frozen_default_commit": _COMMIT,
        "source_commit": _COMMIT,
        "source_clean": True,
        "search_space_sha256": "b" * 64,
        "workload": workload,
        "acceptance_contract": contract,
        "plan_identity": autotune._plan_identity(
            _COMMIT, _COMMIT, "b" * 64, workload, contract
        ),
        "candidates": [_candidate(result_directory)],
    }


def _report(speedup: float = 1.10, run_id: str = "run-0") -> dict[str, Any]:
    profile = _profile()
    modes = [
        "off",
        "adaptive",
        "adaptive",
        "off",
        "adaptive",
        "off",
        "off",
        "adaptive",
    ]
    resolved = {"num_sms": 64, "num_qps": 129, "num_allocated_qps": 129}
    blocks = []
    for index, mode in enumerate(modes):
        value = 1.0 if mode == "off" else 1.0 / speedup
        blocks.append(
            {
                "block_index": index,
                "pair_index": index // 2,
                "order_repeat": 0,
                "pattern": "ABBA" if index < 4 else "BAAB",
                "pattern_position": index % 4,
                "mode": mode,
                "samples": [
                    {
                        "iteration": iteration,
                        "rank_raw": [
                            {"rank": rank, "roundtrip_cuda_ms": value}
                            for rank in range(16)
                        ],
                        "rank_max": {"roundtrip_cuda_ms": value},
                    }
                    for iteration in range(100)
                ],
                "resolved_execution": resolved,
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "claim_scope": "real_multinode",
        "git_commit": _COMMIT,
        "candidate": {
            "mode": profile["mode"],
            "policy": "all",
            "threshold_percent": 0,
            "two_hop_threshold_percent": profile["two_hop_threshold_percent"],
            "max_two_hop_percent": profile["max_two_hop_percent"],
            "hop_penalty_percent": profile["hop_penalty_percent"],
        },
        "kernel_config": {
            "num_sms": profile["num_sms"],
            "num_allocated_qps": profile["num_allocated_qps"],
            "resolved_num_sms": 64,
            "resolved_num_qps": 129,
            "resolved_num_allocated_qps": 129,
            "baseline_resolved": {
                "num_sms": 64,
                "num_qps": 129,
                "num_allocated_qps": 129,
            },
        },
        "identity": {
            "git_commit": _COMMIT,
            "git_dirty": False,
            "extension_path": "/tmp/deep_ep.so",
            "extension_sha256": "a" * 64,
            "torch": "test",
            "cuda": "test",
            "nccl": "test",
            "gpu": "NVIDIA H200",
            "compute_capability": [9, 0],
            "node_count": 2,
            "local_processes": 8,
            "config": {
                "case": "trace",
                "num_tokens": 4096,
                "hidden": 7168,
                "num_topk": 8,
                "num_experts": 256,
                "num_sms": 0,
                "num_allocated_qps": 0,
                "proxy_slots_per_rank": 512,
                "rail_balance_policy": "all",
                "rail_balance_threshold_percent": 0,
                "candidate_mode": "adaptive",
                "two_hop_threshold_percent": 5,
                "max_two_hop_percent": 25,
                "hop_penalty_percent": 50,
                "sl_idx": 3,
                "seed": 106,
            },
            "environment": {},
        },
        "planner_config": {"planner_seed": 0, "chunk_size": 8},
        "benchmark_manifest": {
            "workload_sha256": None,
            "warmup_iters": 10,
            "steady_iters": 100,
            "order_repeats": 1,
            "expanded_order": [
                "off",
                "adaptive",
                "adaptive",
                "off",
                "adaptive",
                "off",
                "off",
                "adaptive",
            ],
        },
        "workload": {
            "case": "trace",
            "node_count": 2,
            "local_processes": 8,
            "tokens_per_rank": 4096,
            "hidden": 7168,
            "topk": 8,
            "experts": 256,
            "proxy_slots_per_rank": profile["proxy_slots_per_rank"],
        },
        "correctness": {"passed": True, "off_candidate_equal": True},
        "eligibility": {"performance_claim_eligible": True, "reasons": []},
        "comparison": {
            "roundtrip_cuda_ms": {
                "speedup_baseline_over_candidate": speedup,
            },
            "paired_roundtrip": [
                {"speedup_baseline_over_candidate": speedup},
                {"speedup_baseline_over_candidate": speedup},
            ],
        },
        "aggregate": {
            "off": {"roundtrip_cuda_ms": {"pooled_raw": {"p95": 1.0}}},
            "adaptive": {"roundtrip_cuda_ms": {"pooled_raw": {"p95": 1.0 / speedup}}},
        },
        "blocks": blocks,
    }


def _write_report(directory: Path, name: str, report: dict[str, Any]) -> Path:
    path = directory / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


class RailBalanceAutotuneTest(unittest.TestCase):
    def test_validate_space_schema_and_v2_auto(self) -> None:
        autotune._validate_space(_space())
        layout = (
            _ROOT / "deep_ep/include/deep_ep/common/rail_balance_hybrid_layout.cuh"
        ).read_text(encoding="utf-8")
        self.assertIn("kDefaultHopPlannerChunkSize = 8;", layout)

        wrong_schema = _space()
        wrong_schema["schema_version"] = 2
        with self.assertRaises(ValueError):
            autotune._validate_space(wrong_schema)

        non_auto = _space()
        non_auto["execution_presets"][0]["num_sms"] = 8
        with self.assertRaises(ValueError):
            autotune._validate_space(non_auto)

        labeled_auto = _space()
        labeled_auto["execution_presets"][0]["id"] = "auto"
        autotune._validate_space(labeled_auto)

        wrong_seed = _space()
        wrong_seed["algorithm"]["planner_seed"] = 1
        with self.assertRaises(ValueError):
            autotune._validate_space(wrong_seed)

        wrong_chunk = _space()
        wrong_chunk["algorithm"]["planner_chunk_size"] = 4
        with self.assertRaises(ValueError):
            autotune._validate_space(wrong_chunk)

    def test_enumerate_normalizes_one_hop_two_hop_fields(self) -> None:
        configs = autotune._enumerate_algorithm_configs(_space())
        one_hop = [row for row in configs if row["mode"] == "one_hop"]
        adaptive = [row for row in configs if row["mode"] == "adaptive"]

        self.assertEqual(len(one_hop), 1)
        self.assertEqual(
            (
                one_hop[0]["two_hop_threshold_percent"],
                one_hop[0]["max_two_hop_percent"],
                one_hop[0]["hop_penalty_percent"],
            ),
            (0, 0, 0),
        )
        self.assertEqual(one_hop[0]["planner_seed"], 0)
        self.assertEqual(one_hop[0]["planner_chunk_size"], 8)
        self.assertEqual(len(adaptive), 2 * 2 * 2)

    def test_pareto_frontier_uses_pair_source_and_hop_cost(self) -> None:
        def row(
            name: str, pair_peak: int, source_peak: int, extra_bytes: int
        ) -> dict[str, Any]:
            return {
                "algorithm": {"name": name, "mode": "adaptive"},
                "oracle": {
                    "pair_peak": pair_peak,
                    "source_peak": source_peak,
                    "extra_local_hop_bytes": extra_bytes,
                },
            }

        rows = [
            row("low_pair", 8, 12, 100),
            row("low_source", 10, 8, 100),
            row("low_hops", 10, 12, 0),
            row("dominated", 11, 13, 100),
        ]
        frontier = autotune._pareto_frontier(rows)
        self.assertEqual(
            {item["algorithm"]["name"] for item in frontier},
            {"low_pair", "low_source", "low_hops"},
        )

    def test_candidate_selection_keeps_one_hop_control(self) -> None:
        control = {
            "algorithm": {"mode": "one_hop"},
            "oracle": {
                "pair_peak": 100,
                "source_peak": 100,
                "extra_local_hop_bytes": 100,
            },
        }
        adaptive = {
            "algorithm": {"mode": "adaptive"},
            "oracle": {
                "pair_peak": 10,
                "source_peak": 10,
                "extra_local_hop_bytes": 10,
            },
        }
        selected = autotune._select_candidate_rows([control, adaptive], 2)
        self.assertIn(control, selected)

    def test_v2_auto_benchmark_argv_uses_zero_sms_and_qps(self) -> None:
        workload = {
            "case": "trace",
            "workload_json": "/traces/held-out.json",
            "gpus_per_node": 8,
            "tokens_per_rank": 4096,
            "hidden": 7168,
            "topk": 8,
            "num_experts": 256,
            "warmup_iters": 10,
            "steady_iters": 100,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            argv = autotune._build_benchmark_argv(
                _candidate(),
                workload,
                output,
                "run-01",
                python_executable="/python",
            )

        self.assertEqual(argv[0], "/python")
        self.assertEqual(argv[argv.index("--run-id") + 1], "run-01")
        self.assertEqual(argv[argv.index("--output") + 1], str(output))
        self.assertEqual(argv[argv.index("--candidate-mode") + 1], "adaptive")
        self.assertEqual(argv[argv.index("--num-sms") + 1], "0")
        self.assertEqual(argv[argv.index("--num-allocated-qps") + 1], "0")

    def test_freeze_rejects_bad_evidence_and_accepts_five_diagnostic_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = _plan(directory)
            candidate = _candidate(directory)

            def freeze(reports: list[dict[str, Any]]) -> dict[str, Any]:
                reports = [copy.deepcopy(report) for report in reports]
                attempts = candidate["attempts"]
                for index, report in enumerate(reports):
                    attempt = attempts[index]
                    report["run_id"] = attempt["run_id"]
                    report["benchmark_manifest"]["run_id"] = attempt["run_id"]
                    report["benchmark_manifest"]["output"] = attempt["result_path"]
                inputs = [
                    (
                        _write_report(
                            directory, Path(attempts[index]["result_path"]).name, report
                        ),
                        report,
                    )
                    for index, report in enumerate(reports)
                ]
                return autotune._freeze_profile(plan, candidate["id"], inputs)

            valid = [_report(value) for value in (1.10, 1.12, 1.14, 1.16, 1.20)]

            with self.assertRaises(ValueError):
                freeze(valid[:2])

            ineligible = _report()
            ineligible["eligibility"]["performance_claim_eligible"] = False
            with self.assertRaises(ValueError):
                freeze([ineligible, *valid[1:]])

            mismatch = _report()
            mismatch["candidate"]["max_two_hop_percent"] = 10
            with self.assertRaises(ValueError):
                freeze([mismatch, *valid[1:]])

            with self.assertRaises(ValueError):
                freeze([_report(1.01), *valid[1:]])

            false_summary = _report(1.01)
            false_summary["comparison"]["roundtrip_cuda_ms"][
                "speedup_baseline_over_candidate"
            ] = 99.0
            with self.assertRaises(ValueError):
                freeze([false_summary, *valid[1:]])

            bad_pairing = _report(1.20)
            bad_pairing["blocks"][2]["pair_index"] = 0
            with self.assertRaises(ValueError):
                freeze([bad_pairing, *valid[1:]])

            p95_regression = _report(1.20)
            for block in p95_regression["blocks"]:
                if block["mode"] == "adaptive":
                    for sample in block["samples"]:
                        sample["rank_max"]["roundtrip_cuda_ms"] = 1.10
                        for row in sample["rank_raw"]:
                            row["roundtrip_cuda_ms"] = 1.10
            with self.assertRaises(ValueError):
                freeze([p95_regression, *valid[1:]])

            identity_mismatch = _report()
            identity_mismatch["identity"]["gpu"] = "different"
            with self.assertRaises(ValueError):
                freeze([identity_mismatch, *valid[1:]])

            corrupt_plan = _plan()
            corrupt_plan["candidates"][0]["id"] = "rb_corrupt"
            with self.assertRaises(ValueError):
                autotune._freeze_profile(corrupt_plan, "rb_corrupt", [])

            explicit_plan = _plan()
            explicit_plan["candidates"][0]["profile"]["execution_strategy"] = (
                "explicit"
            )
            with self.assertRaises(ValueError):
                autotune._freeze_profile(
                    explicit_plan, explicit_plan["candidates"][0]["id"], []
                )

            profile = freeze(valid)

        self.assertIs(profile["production_consumed"], False)
        self.assertIs(profile["production_promotion_eligible"], False)
        self.assertIs(profile["v2_auto_reused"], True)
        self.assertEqual(profile["candidate_id"], candidate["id"])
        self.assertEqual(len(profile["evidence"]), 5)
        self.assertEqual(profile["resolved_v2_execution"]["num_sms"], 64)
        self.assertEqual(
            profile["speedup_summary"],
            {"minimum": 1.10, "median": 1.14, "maximum": 1.20},
        )


if __name__ == "__main__":
    unittest.main()
