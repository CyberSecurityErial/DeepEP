"""CPU contracts for the unified hop-aware benchmark entry."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch

from bench_rail_balance_hop import _write_json, build_reference_report
from bench_rail_balance_hop_plan import _check_status
from bench_rail_balance_hybrid_lsa import (
    _REPOSITORY_ROOT,
    _git_manifest,
    _hop_plan_diagnostics,
    _movement_from_owner_to_egress,
)


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


def test_activation_gate_bypasses_balanced_but_keeps_hot_sources() -> None:
    balanced = build_reference_report(
        case="balanced",
        mode="one_hop",
        rail_threshold_percent=20,
        **_COMMON,
    )
    hot = build_reference_report(
        case="offdiag_hot",
        mode="one_hop",
        rail_threshold_percent=20,
        **_COMMON,
    )
    assert balanced["planner_config"]["activated_sources"] == 0
    assert (
        balanced["rail_load_before_after"]["after_pair_rail_load"]
        == balanced["rail_load_before_after"]["before_pair_rail_load"]
    )
    assert hot["planner_config"]["activated_sources"] == 2
    assert (
        hot["rail_load_before_after"]["after_stats"]["max"]
        < hot["rail_load_before_after"]["before_stats"]["max"]
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


def test_forwarding_accounting_counts_nonzero_owner_target_bits() -> None:
    records = torch.tensor(
        [[[(1 << 32) | 0b11]], [[(1 << 32) | 0b10]]], dtype=torch.int64
    )
    resolutions = torch.full((2, 1, 1, 4), -1, dtype=torch.int32)
    resolutions[..., 0] = torch.tensor([[[0]], [[1]]], dtype=torch.int32)
    values = (
        records,
        resolutions,
        torch.tensor([[0, 0], [1, 1]], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([1, 1, 0, 0], dtype=torch.int32),
        torch.tensor(0, dtype=torch.int32),
        torch.tensor(0, dtype=torch.int32),
    )

    class FakeRuntime:
        def _rail_balance_hop_plan_snapshot(self, invocation_id: int):
            assert invocation_id == 17
            return values

    diagnostics = _hop_plan_diagnostics(FakeRuntime(), 17, 0)
    assert diagnostics["source_forward_units"] == 0
    assert diagnostics["destination_forward_units"] == 1
    assert diagnostics["minimum_local_forward_units"] == 1
    assert diagnostics["extra_local_forward_units"] == 0


def test_git_manifest_is_bound_to_the_deepep_repository() -> None:
    expected = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        cwd=_REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    expected_dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            check=True,
            cwd=_REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.splitlines()
    )
    original = Path.cwd()
    try:
        os.chdir("/")
        manifest = _git_manifest()
    finally:
        os.chdir(original)
    assert manifest["commit"] == expected
    assert manifest["dirty"] is expected_dirty


def test_private_planner_status_is_checked_fail_closed() -> None:
    _check_status((torch.tensor([0], dtype=torch.int32),))
    for outputs, expected in (
        ((), "status None"),
        ((torch.tensor([2], dtype=torch.int32),), "status 2"),
    ):
        try:
            _check_status(outputs)
        except RuntimeError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"planner status was accepted: {outputs!r}")


def test_measured_hop_movement_uses_owner_egress_matrix() -> None:
    owner_to_egress = [[0 for _ in range(8)] for _ in range(8)]
    owner_to_egress[1][2] = 3
    owner_to_egress[3][2] = 1
    source = _movement_from_owner_to_egress(
        owner_to_egress, logical_token_bytes=16, stage="source"
    )
    returned = _movement_from_owner_to_egress(
        owner_to_egress, logical_token_bytes=16, stage="return"
    )
    assert source["selected_moved_copies_per_rank"] == [0, 3, 0, 1, 0, 0, 0, 0]
    assert returned["selected_moved_copies_per_rank"] == [0, 0, 4, 0, 0, 0, 0, 0]
    assert source["selected_logical_bytes_aggregate"] == 64
    assert returned["selected_logical_bytes_aggregate"] == 64
    for kwargs in (
        {"logical_token_bytes": 16, "stage": "finish"},
        {"logical_token_bytes": -1, "stage": "source"},
    ):
        try:
            _movement_from_owner_to_egress(owner_to_egress, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid movement arguments accepted: {kwargs}")


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
