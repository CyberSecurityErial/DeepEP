"""CPU-only evidence-contract tests for the Hybrid off/force benchmark.

Run without pytest or CUDA initialization::

    PYTHONPATH=.:tests/elastic python3 -B \
      tests/elastic/test_rail_balance_hybrid_multinode_bench.py
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import bench_rail_balance_hybrid_multinode as bench


def _block(mode: str, values: tuple[float, ...]) -> dict:
    samples = [
        {"rank_max": {metric: value for metric in bench._ALL_METRICS}}
        for value in values
    ]
    return {
        "mode": mode,
        "samples": samples,
        "summary": {
            metric: bench._stats([row["rank_max"][metric] for row in samples])
            for metric in bench._ALL_METRICS
        },
    }


def _snapshot() -> tuple[list[dict], list[dict]]:
    host = "h200-node-0"
    return [
        {
            "node_rank": 0,
            "hostname": host,
            "gpu_state": {
                "returncode": 0,
                "stdout": (
                    "0, GPU-0, NVIDIA H200, 570, P0, 35, 100, 700, 1900, "
                    "1593, 0, Disabled, Default\n"
                ),
            },
            "compute_apps": {
                "returncode": 0,
                "stdout": "GPU-0, 111, python, 1024\n",
            },
            "process_table": {"returncode": 0, "stdout": "111 1 111 python\n"},
            "mps_processes": [],
        }
    ], [{"node_rank": 0, "hostname": host, "pid": 111}]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(nodes) == 1
    return nodes[0]


def _source(source: str, node: ast.AST) -> str:
    result = ast.get_source_segment(source, node)
    assert result is not None
    return result


def test_order_is_symmetric_and_each_neighbor_pair_is_off_force() -> None:
    rows = bench._order(2)
    assert len(rows) == 16
    assert sum(row[3] == "off" for row in rows) == 8
    assert sum(row[3] == "force" for row in rows) == 8
    for begin in range(0, len(rows), 8):
        cycle = rows[begin : begin + 8]
        assert tuple(row[3] for row in cycle[:4]) == ("off", "force", "force", "off")
        assert tuple(row[3] for row in cycle[4:]) == ("force", "off", "off", "force")
    for begin in range(0, len(rows), 2):
        assert {rows[begin][3], rows[begin + 1][3]} == {"off", "force"}

    adaptive = bench._order(1, "adaptive")
    assert tuple(row[3] for row in adaptive[:4]) == (
        "off", "adaptive", "adaptive", "off")
    assert tuple(row[3] for row in adaptive[4:]) == (
        "adaptive", "off", "off", "adaptive")


def test_stats_percentiles_and_invalid_samples() -> None:
    stats = bench._stats((1, 2, 3, 4, 5))
    assert stats["count"] == 5 and stats["median"] == stats["mean"] == 3
    assert math.isclose(stats["p95"], 4.8)
    assert math.isclose(stats["p99"], 4.96)
    assert math.isclose(stats["std_population"], math.sqrt(2))
    for values in ((), (1, float("nan")), (1, float("inf"))):
        try:
            bench._stats(values)
        except AssertionError:
            continue
        raise AssertionError(f"invalid samples were accepted: {values}")


def test_compare_reports_deepep_v2_over_candidate_speedup() -> None:
    blocks = [
        _block("off", (10, 12)),
        _block("force", (5, 6)),
        _block("force", (5, 6)),
        _block("off", (10, 12)),
    ]
    aggregate = {mode: bench._aggregate(blocks, mode) for mode in ("off", "force")}
    result = bench._compare(aggregate)["roundtrip_cuda_ms"]
    assert result == {
        "baseline_median_ms": 11.0,
        "candidate_median_ms": 5.5,
        "speedup_baseline_over_candidate": 2.0,
        "candidate_latency_delta_percent": -50.0,
    }
    losing = {
        "off": bench._aggregate([_block("off", (5,))], "off"),
        "force": bench._aggregate([_block("force", (10,))], "force"),
    }
    result = bench._compare(losing)["roundtrip_cuda_ms"]
    assert result["speedup_baseline_over_candidate"] == 0.5
    assert result["candidate_latency_delta_percent"] == 100


def test_snapshot_reasons_fail_closed() -> None:
    snapshots, workers = _snapshot()
    assert bench._snapshot_reasons(snapshots, workers, 1) == []
    snapshot = snapshots[0]
    fields = [item.strip() for item in snapshot["gpu_state"]["stdout"].split(",")]
    fields[10:13] = ["0x1", "Enabled", "Exclusive_Process"]
    snapshot["gpu_state"]["stdout"] = ", ".join(fields) + "\n"
    snapshot["compute_apps"]["stdout"] += "GPU-0, 999, foreign, 2048\n"
    snapshot["mps_processes"] = ["222 1 222 nvidia-cuda-mps-server"]
    reasons = "\n".join(bench._snapshot_reasons(snapshots, workers, 1))
    for fragment in (
        "unexpected GPU PIDs [999]",
        "CUDA MPS is active",
        "throttle is active",
        "MIG is not disabled",
        "compute mode is not Default",
    ):
        assert fragment in reasons


def test_source_keeps_v2_baseline_selectable_candidate_and_clean_steady_loop() -> None:
    source = Path(bench.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    worker = _source(source, _function(tree, "_worker"))
    assert '"mode": "off"' in worker
    assert '"native DeepEP V2 Hybrid data path"' in worker
    assert 'candidate_mode = args.candidate_mode' in worker
    assert 'modes = ("off", candidate_mode)' in worker
    assert 'probes["off"]["combined_x"]' in worker
    assert 'probes[candidate_mode]["combined_x"]' in worker
    assert '"benchmark manifest"' in worker
    assert "_benchmark_manifest(args)" in worker
    assert '"post-measurement cluster identity"' in worker
    assert 'block["last_output_rank_digests"]' in worker
    assert '["timed_path_rank_digests"]' in worker

    roundtrip = _source(source, _function(tree, "_roundtrip"))
    assert "synchronize(" not in roundtrip
    measure = _function(tree, "_measure_block")
    steady_loops = [
        node
        for node in ast.walk(measure)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "events"
    ]
    assert len(steady_loops) == 1
    steady = _source(source, steady_loops[0])
    assert "_roundtrip(" in steady and "synchronize(" not in steady


if __name__ == "__main__":
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} Hybrid multinode benchmark CPU contracts")
