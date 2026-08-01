"""CPU-only contracts for the offline RailBalance autotuner."""

from __future__ import annotations

import importlib.util
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


def _candidate() -> dict[str, Any]:
    return {
        "id": "rb_test",
        "profile": _profile(),
        "oracle": {
            "pair_peak": 10,
            "pair_cv": 0.1,
            "extra_local_hop_bytes": 1024,
            "two_hop_ratio": 0.05,
        },
        "result_path": "/tmp/rb_test.json",
        "environment": {},
        "argv": [],
    }


def _plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "rail_balance_autotune_plan",
        "frozen_default_commit": _COMMIT,
        "source_commit": _COMMIT,
        "source_clean": True,
        "workload": {
            "case": "trace",
            "num_nodes": 2,
            "gpus_per_node": 8,
            "tokens_per_rank": 4096,
            "hidden": 7168,
            "topk": 8,
            "num_experts": 256,
        },
        "candidates": [_candidate()],
    }


def _report(speedup: float = 1.10) -> dict[str, Any]:
    profile = _profile()
    return {
        "schema_version": 1,
        "claim_scope": "real_multinode",
        "git_commit": _COMMIT,
        "candidate": {
            "mode": profile["mode"],
            "two_hop_threshold_percent": profile["two_hop_threshold_percent"],
            "max_two_hop_percent": profile["max_two_hop_percent"],
            "hop_penalty_percent": profile["hop_penalty_percent"],
        },
        "kernel_config": {
            "num_sms": profile["num_sms"],
            "num_allocated_qps": profile["num_allocated_qps"],
        },
        "benchmark_manifest": {"workload_sha256": None},
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
            }
        },
    }


def _write_report(directory: Path, name: str, report: dict[str, Any]) -> Path:
    path = directory / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


class RailBalanceAutotuneTest(unittest.TestCase):
    def test_validate_space_schema_and_v2_auto(self) -> None:
        autotune._validate_space(_space())

        wrong_schema = _space()
        wrong_schema["schema_version"] = 2
        with self.assertRaises(ValueError):
            autotune._validate_space(wrong_schema)

        non_auto = _space()
        non_auto["execution_presets"][0]["num_sms"] = 8
        with self.assertRaises(ValueError):
            autotune._validate_space(non_auto)

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

    def test_pareto_frontier_uses_peak_and_extra_hop_bytes(self) -> None:
        def row(name: str, pair_peak: int, extra_bytes: int) -> dict[str, Any]:
            return {
                "algorithm": {"name": name},
                "oracle": {
                    "pair_peak": pair_peak,
                    "extra_local_hop_bytes": extra_bytes,
                },
            }

        rows = [
            row("low_peak", 8, 100),
            row("middle", 9, 50),
            row("low_hops", 10, 0),
            row("dominated", 11, 100),
        ]
        frontier = autotune._pareto_frontier(rows)
        self.assertEqual(
            {item["algorithm"]["name"] for item in frontier},
            {"low_peak", "middle", "low_hops"},
        )

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

    def test_freeze_rejects_bad_evidence_and_accepts_two_real_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def freeze(reports: list[dict[str, Any]]) -> dict[str, Any]:
                inputs = [
                    (_write_report(directory, f"run-{index}.json", report), report)
                    for index, report in enumerate(reports)
                ]
                return autotune._freeze_profile(
                    _plan(), "rb_test", inputs, minimum_speedup=1.05
                )

            ineligible = _report()
            ineligible["eligibility"]["performance_claim_eligible"] = False
            with self.assertRaises(ValueError):
                freeze([ineligible, _report(1.11)])

            mismatch = _report()
            mismatch["candidate"]["max_two_hop_percent"] = 10
            with self.assertRaises(ValueError):
                freeze([mismatch, _report(1.11)])

            with self.assertRaises(ValueError):
                freeze([_report(1.01), _report(1.11)])

            profile = freeze([_report(1.10), _report(1.20)])

        self.assertIs(profile["production_consumed"], False)
        self.assertIs(profile["v2_auto_reused"], True)
        self.assertEqual(profile["candidate_id"], "rb_test")
        self.assertEqual(len(profile["evidence"]), 2)
        self.assertEqual(
            profile["speedup_summary"],
            {"minimum": 1.10, "median": 1.15, "maximum": 1.20},
        )


if __name__ == "__main__":
    unittest.main()
