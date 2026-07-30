"""CPU contracts for the unified hop-aware benchmark entry."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from bench_rail_balance_hop import _write_json, build_reference_report


_COMMON = {
    "run_id": "hop-bench-test",
    "num_nodes": 2,
    "rails": 4,
    "tokens_per_rank": 16,
    "topk": 1,
    "num_experts": 16,
    "hidden": 256,
    "chunk_size": 1,
    "two_hop_threshold_percent": 0,
    "max_two_hop_percent": 50,
    "hop_penalty_percent": 0,
    "seed": 0,
}


def _report(case: str, mode: str):
    return build_reference_report(case=case, mode=mode, **_COMMON)


def test_offdiagonal_hotspot_is_solved_without_two_hop() -> None:
    report = _report("offdiag_hot", "one_hop")
    paths = report["path_distribution"]
    loads = report["rail_load_before_after"]
    assert paths["dst_forward_units"] > 0
    assert paths["src_forward_units"] > 0
    assert paths["two_hop_units"] == 0
    assert loads["after_stats"]["max"] < loads["before_stats"]["max"]


def test_diagonal_hotspot_uses_only_bounded_adaptive_escape() -> None:
    one_hop = _report("diag_hot", "one_hop")
    adaptive = _report("diag_hot", "adaptive")
    assert one_hop["path_distribution"]["two_hop_units"] == 0
    assert (
        one_hop["rail_load_before_after"]["after_stats"]["max"]
        == one_hop["rail_load_before_after"]["before_stats"]["max"]
    )
    assert adaptive["path_distribution"]["two_hop_ratio"] == 0.5
    assert adaptive["path_distribution"]["extra_local_hop_bytes"] > 0
    assert (
        adaptive["rail_load_before_after"]["after_stats"]["max"]
        < adaptive["rail_load_before_after"]["before_stats"]["max"]
    )


def test_json_schema_is_atomic_and_claim_scope_is_explicit() -> None:
    report = _report("balanced", "adaptive")
    required = {
        "schema_version",
        "run_id",
        "backend",
        "claim_scope",
        "topology",
        "environment",
        "planner_config",
        "kernel_config",
        "workload_summary",
        "correctness",
        "path_distribution",
        "rail_load_before_after",
        "latency_statistics",
        "throughput_statistics",
        "fallbacks",
    }
    assert required <= report.keys()
    assert report["claim_scope"] == "planner_only"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.json"
        _write_json(path, report)
        assert json.loads(path.read_text(encoding="utf-8")) == report
        assert not list(path.parent.glob(".*.tmp"))


def test_trace_manifest_replays_the_same_diagonal_endpoint_matrix() -> None:
    manifest = (
        Path(__file__).with_name("workloads")
        / "rail_balance"
        / "diagonal_twohop_required.json"
    )
    report = build_reference_report(
        case="trace", mode="adaptive", workload_json=manifest, **_COMMON
    )
    assert report["path_distribution"]["two_hop_ratio"] == 0.5
    assert report["workload_summary"]["workload_json"] == str(manifest.resolve())
    assert report["correctness"]["conserved_payload_units"] == 32


if __name__ == "__main__":
    tests = sorted(
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} unified hop benchmark CPU contracts")
