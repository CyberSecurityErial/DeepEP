"""CPU-only safety contracts for the hop-aware campaign supervisor."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

from rail_balance_campaign_schema import ManifestError, load_manifest, validate_manifest
from run_rail_balance_hop_campaign import (
    _ROOT,
    _append_jsonl,
    _atomic_json,
    _campaign_lock,
    _child_environment,
    _declared_artifacts,
    _directory_bytes,
    _finalization_reasons,
    _foreign_profiler_rows,
    _git_identity,
    _gpu_snapshot_reasons,
    _process_group_members,
    _run_attempt,
    _safe_output_dir,
    _terminal_exit,
    _terminate_process_group,
)


_MANIFEST = Path(__file__).with_name("experiments") / "hop_local8_sm90_v1.json"


def _valid_manifest() -> dict[str, object]:
    contract = {
        name: "frozen"
        for name in (
            "interface",
            "supported_inputs",
            "target_platform",
            "reference",
            "build",
            "correctness",
            "benchmark",
            "profile",
            "numerical_tolerance",
            "objective_and_budget",
        )
    }
    contract["target_platform"] = {
        "gpu_arch": "SM90",
        "required_gpus": 8,
        "topology": "single node",
        "driver": "570.172.08",
        "cuda_toolkit": "12.8 build 35404655",
        "pytorch": "2.11.0+cu128",
        "python": "3.11.15",
        "nsys": "2024.6.2.225-246235244400v0",
        "ncu": "2025.1.1.0 build 35528883",
        "compute_sanitizer": "2025.1.0.0 build 35351055",
    }
    contract["objective_and_budget"] = {
        "minimum_candidates_per_round": 2,
        "max_candidates_per_round": 4,
    }
    python = "/home/chen/.cache/deepep-sjlgpt/bin/python"
    cpu = {
        "id": "cpu",
        "kind": "correctness",
        "resource": "cpu",
        "argv": [python, "-B", "tests/elastic/test_rail_balance_hop_plan.py"],
        "timeout_seconds": 1,
        "depends_on": [],
    }
    compile_stage = {
        "id": "compile",
        "kind": "compile",
        "resource": "exclusive_8gpu",
        "argv": [
            python,
            "-B",
            "tests/elastic/run_rail_balance_build_warmup.py",
        ],
        "timeout_seconds": 1,
        "depends_on": ["cpu"],
    }
    gpu = {
        "id": "gpu",
        "kind": "correctness",
        "resource": "exclusive_8gpu",
        "argv": [
            python,
            "-B",
            "tests/elastic/test_rail_balance_hop_one_hop_cuda.py",
        ],
        "timeout_seconds": 1,
        "depends_on": ["compile"],
    }
    sanitizer = {
        "id": "sanitizer",
        "kind": "sanitizer",
        "resource": "exclusive_8gpu",
        "argv": [
            "/usr/local/cuda/bin/compute-sanitizer",
            "--tool",
            "memcheck",
            python,
            "-B",
            "tests/elastic/test_rail_balance_hop_one_hop_cuda.py",
        ],
        "timeout_seconds": 1,
        "depends_on": ["gpu"],
    }
    benchmark = {
        "id": "benchmark",
        "kind": "benchmark",
        "resource": "exclusive_8gpu",
        "argv": [
            python,
            "-B",
            "tests/elastic/bench_rail_balance_hybrid_lsa.py",
        ],
        "timeout_seconds": 1,
        "depends_on": ["sanitizer"],
    }
    nsys = {
        "id": "nsys",
        "kind": "profile_nsys",
        "resource": "exclusive_8gpu",
        "argv": [
            "/usr/local/cuda/bin/nsys",
            "profile",
            python,
            "-B",
            "tests/elastic/bench_rail_balance_hybrid_lsa.py",
        ],
        "timeout_seconds": 1,
        "depends_on": ["benchmark"],
    }
    return {
        "schema_version": 1,
        "campaign_id": "test-campaign",
        "candidate": {
            "id": "candidate-1",
            "parent": "leader-0",
            "primary_change": "one change",
            "hypothesis": "one hypothesis",
            "expected_profile_metrics": ["duration"],
            "risks": ["regression"],
        },
        "frozen_contract": contract,
        "resources": {
            "required_gpu_count": 8,
            "require_all_host_gpus": True,
            "expected_gpu_index_uuid_mapping": [
                {
                    "index": index,
                    "uuid": f"GPU-00000000-0000-0000-0000-{index:012x}",
                }
                for index in range(8)
            ],
            "home_path": "/home/chen",
            "home_hard_limit_bytes": 300_000_000_000,
            "artifact_budget_bytes": 1,
            "lock_path": "/tmp/deepep-rail-balance-hop-campaign.lock",
        },
        "stages": [cpu, compile_stage, gpu, sanitizer, benchmark, nsys],
        "leaderboard_path": ".cache/rail_balance/hop_campaign/leaderboard.jsonl",
    }


def _expect_manifest_error(manifest: dict[str, object]) -> None:
    try:
        validate_manifest(manifest)
    except ManifestError:
        pass
    else:
        raise AssertionError("unsafe manifest was accepted")


def test_checked_in_manifest_is_strict_and_complete() -> None:
    manifest, raw_hash, canonical_hash = load_manifest(_MANIFEST)
    assert manifest["campaign_id"] == "hop-local8-sm90-v1"
    assert len(manifest["stages"]) == 25
    assert len(raw_hash) == len(canonical_hash) == 64
    gpu_jit_caches = [
        stage["env"]["EP_JIT_CACHE_DIR"]
        for stage in manifest["stages"]
        if "EP_JIT_CACHE_DIR" in stage.get("env", {})
    ]
    assert len(gpu_jit_caches) == 16
    assert all(value == "{stage_dir}/jit" for value in gpu_jit_caches)
    benchmark_stages = [
        stage for stage in manifest["stages"] if stage["kind"] == "benchmark"
    ]
    assert [stage["id"] for stage in benchmark_stages] == [
        "bench-abba-a1",
        "bench-abba-b1",
        "bench-abba-b2",
        "bench-abba-a2",
        "bench-baab-b3",
        "bench-baab-a3",
        "bench-baab-a4",
        "bench-baab-b4",
    ]
    for previous, current in zip(benchmark_stages, benchmark_stages[1:]):
        assert previous["id"] in current["depends_on"]
    profile = next(stage for stage in manifest["stages"] if stage["id"] == "profile-nsys")
    assert profile["enabled"] is False


def test_manifest_rejects_ambiguous_json() -> None:
    for payload in (
        b'{"schema_version":1,"schema_version":1}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_bytes(payload)
            try:
                load_manifest(path)
            except ManifestError:
                pass
            else:
                raise AssertionError(f"ambiguous JSON was accepted: {payload!r}")


def test_manifest_rejects_links_unsafe_mode_and_oversize() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        payload = json.dumps(_valid_manifest()).encode("utf-8")
        target = root / "target.json"
        target.write_bytes(payload)

        symlink = root / "symlink.json"
        symlink.symlink_to(target)
        try:
            load_manifest(symlink)
        except ManifestError:
            pass
        else:
            raise AssertionError("manifest reader followed a symbolic link")

        hardlink = root / "hardlink.json"
        os.link(target, hardlink)
        try:
            load_manifest(target)
        except ManifestError:
            pass
        else:
            raise AssertionError("manifest reader accepted a hard-linked file")
        hardlink.unlink()

        target.chmod(0o666)
        try:
            load_manifest(target)
        except ManifestError:
            pass
        else:
            raise AssertionError("manifest reader accepted a writable file")

        oversized = root / "oversized.json"
        oversized.write_bytes(b" " * ((1 << 20) + 1))
        try:
            load_manifest(oversized)
        except ManifestError:
            pass
        else:
            raise AssertionError("manifest reader accepted an oversized file")


def test_manifest_rejects_command_environment_and_path_escape() -> None:
    invalid = copy.deepcopy(_valid_manifest())
    invalid["stages"][0]["argv"][0] = "/bin/rm"
    _expect_manifest_error(invalid)

    invalid = copy.deepcopy(_valid_manifest())
    invalid["stages"][0]["argv"] = [
        "/home/chen/.cache/deepep-sjlgpt/bin/python",
        "-c",
        "print('unsafe')",
    ]
    _expect_manifest_error(invalid)

    invalid = copy.deepcopy(_valid_manifest())
    invalid["stages"][0]["env"] = {"LD_PRELOAD": "/tmp/inject.so"}
    _expect_manifest_error(invalid)

    invalid = copy.deepcopy(_valid_manifest())
    invalid["stages"][0]["artifacts"] = ["../../outside"]
    _expect_manifest_error(invalid)

    invalid = copy.deepcopy(_valid_manifest())
    invalid["stages"][0]["argv"].append("{root}/../outside")
    _expect_manifest_error(invalid)

    invalid = copy.deepcopy(_valid_manifest())
    invalid["stages"][4]["argv"].extend(
        ["--json-out", "docs/unsafe-report.json"]
    )
    _expect_manifest_error(invalid)


def test_manifest_rejects_weakened_gates() -> None:
    invalid = copy.deepcopy(_valid_manifest())
    invalid["resources"]["home_hard_limit_bytes"] = 300_000_000_001
    _expect_manifest_error(invalid)

    invalid = copy.deepcopy(_valid_manifest())
    invalid["stages"][2]["depends_on"] = ["cpu"]
    _expect_manifest_error(invalid)

    invalid = copy.deepcopy(_valid_manifest())
    invalid["leaderboard_path"] = "docs/rail_balance/leaderboard.jsonl"
    _expect_manifest_error(invalid)


def test_gpu_snapshot_rejects_any_compute_process() -> None:
    rows = "\n".join(
        f"{index}, GPU-{index}, Hopper, 143771, Default, Disabled"
        for index in range(8)
    )
    snapshot = {
        "gpu_state": {"returncode": 0, "stdout": rows},
        "compute_apps": {"returncode": 0, "stdout": ""},
        "process_table": {"returncode": 0, "stdout": ""},
        "app_rows": [],
        "mps_process_rows": [],
        "profiler_process_rows": [],
    }
    assert _gpu_snapshot_reasons(snapshot, 8) == []
    expected_mapping = [
        {"index": index, "uuid": f"GPU-{index}"} for index in range(8)
    ]
    assert _gpu_snapshot_reasons(snapshot, 8, expected_mapping) == []
    expected_mapping[7] = {"index": 7, "uuid": "GPU-other"}
    assert "gpu_index_uuid_mapping_mismatch" in _gpu_snapshot_reasons(
        snapshot, 8, expected_mapping
    )
    snapshot["app_rows"] = [{"pid": 123, "process_name": "qwen"}]
    assert "gpu_compute_processes_present:[123]" in _gpu_snapshot_reasons(
        snapshot, 8
    )


def test_child_gpu_visibility_is_forced_by_resource_class() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "CUDA_VISIBLE_DEVICES": "7",
            "LD_PRELOAD": "/tmp/unsafe.so",
            "PYTHONSTARTUP": "/tmp/unsafe.py",
            "UNRELATED_SECRET": "must-not-propagate",
        },
        clear=False,
    ):
        cpu = _child_environment({"resource": "cpu"}, {}, 8)
        gpu = _child_environment(
            {"resource": "exclusive_8gpu"},
            {"CUDA_VISIBLE_DEVICES": "7"},
            8,
        )
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    assert gpu["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert cpu["DEEP_EP_CAMPAIGN_SUPERVISED"] == "1"
    assert gpu["DEEP_EP_CAMPAIGN_SUPERVISED"] == "1"
    assert "LD_PRELOAD" not in cpu and "PYTHONSTARTUP" not in cpu
    assert "UNRELATED_SECRET" not in cpu
    assert cpu["PATH"].startswith("/home/chen/.cache/deepep-sjlgpt/bin:")


def test_profiler_monitor_only_allows_owned_profiler_stages() -> None:
    rows = [
        {"pid": 10, "pgid": 100, "comm": "nsys"},
        {"pid": 20, "pgid": 200, "comm": "ncu"},
    ]
    assert _foreign_profiler_rows(rows, "benchmark", 100) == rows
    assert _foreign_profiler_rows(rows, "correctness", 100) == rows
    assert _foreign_profiler_rows(rows, "profile_nsys", 100) == [rows[1]]
    assert _foreign_profiler_rows(rows, "sanitizer", 100) == [rows[1]]


def test_final_gpu_gate_rejects_without_starting_a_child() -> None:
    stage = {
        "id": "gpu-final-gate",
        "kind": "correctness",
        "resource": "exclusive_8gpu",
        "argv": [
            "/home/chen/.cache/deepep-sjlgpt/bin/python",
            "-B",
            "tests/elastic/test_rail_balance_hop_one_hop_cuda.py",
        ],
        "timeout_seconds": 1,
        "depends_on": [],
    }
    cache_root = _ROOT / ".cache" / "rail_balance" / "campaign-test-temp"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_root) as directory, mock.patch(
        "run_rail_balance_hop_campaign._home_usage",
        return_value={"path": "/home/chen", "bytes": 1, "command": {}},
    ), mock.patch(
        "run_rail_balance_hop_campaign._gpu_snapshot",
        return_value={"occupied": True},
    ), mock.patch(
        "run_rail_balance_hop_campaign._gpu_snapshot_reasons",
        return_value=["gpu_compute_processes_present:[123]"],
    ), mock.patch(
        "run_rail_balance_hop_campaign.subprocess.Popen"
    ) as popen:
        result = _run_attempt(
            stage=stage,
            attempt=1,
            output_dir=Path(directory),
            artifact_budget_bytes=1_000_000,
            home_path=Path("/home/chen"),
            home_hard_limit_bytes=300_000_000_000,
            monitor_exclusive_gpus=True,
            required_gpu_count=8,
            expected_gpu_mapping=[],
        )
    popen.assert_not_called()
    assert not result["process_started"]
    assert result["foreign_gpu_process_detected"]
    assert result["launch_gpu_reasons"] == [
        "gpu_compute_processes_present:[123]"
    ]


def test_stage_group_cleanup_reaps_nested_descendants() -> None:
    code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        (sys.executable, "-c", code),
        start_new_session=True,
    )
    deadline = time.monotonic() + 2
    while (
        len(_process_group_members(process.pid)) < 2
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert len(_process_group_members(process.pid)) >= 2
    _terminate_process_group(process)
    assert not _process_group_members(process.pid)


def test_artifact_symlink_escape_is_rejected() -> None:
    cache_root = _ROOT / ".cache" / "rail_balance" / "campaign-test-temp"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_root) as directory:
        stage_dir = Path(directory)
        outside = stage_dir.parent / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        (stage_dir / "escape").symlink_to(outside)
        rows, errors = _declared_artifacts(
            {"artifacts": ["escape"]}, stage_dir
        )
        assert not rows and errors
        try:
            _directory_bytes(stage_dir)
        except RuntimeError as error:
            assert "symlink" in str(error)
        else:
            raise AssertionError("artifact-tree accounting followed a symlink")
        (stage_dir / "escape").unlink()
        outside.unlink()

        hardlink_source = stage_dir.parent / "hardlink-source.txt"
        hardlink_source.write_text("outside", encoding="utf-8")
        os.link(hardlink_source, stage_dir / "hardlink")
        try:
            _directory_bytes(stage_dir)
        except RuntimeError as error:
            assert "hard-linked" in str(error)
        else:
            raise AssertionError("artifact-tree accounting accepted a hard link")
        hardlink_source.unlink()


def test_atomic_json_preserves_previous_file_on_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.json"
        path.write_text('{"old":true}\n', encoding="utf-8")
        try:
            _atomic_json(path, {"invalid": float("nan")})
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite JSON was accepted")
        assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}


def test_campaign_lock_rejects_unsafe_existing_files() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        lock = root / "lock"
        lock.write_text("unsafe\n", encoding="utf-8")
        lock.chmod(0o644)
        try:
            with _campaign_lock(lock):
                pass
        except RuntimeError:
            pass
        else:
            raise AssertionError("campaign accepted a group/world-readable lock")

        lock.unlink()
        target = root / "target"
        target.write_text("unsafe\n", encoding="utf-8")
        target.chmod(0o600)
        lock.symlink_to(target)
        try:
            with _campaign_lock(lock):
                pass
        except RuntimeError:
            pass
        else:
            raise AssertionError("campaign followed a lock symlink")


def test_leaderboard_append_is_parseable_and_idempotent() -> None:
    row = {
        "campaign_id": "campaign",
        "run_id": "run",
        "candidate": {"id": "candidate"},
        "source_identity_sha256": "a" * 64,
        "manifest_canonical_sha256": "b" * 64,
        "status": "failed",
        "artifact_dir": "/artifact",
        "sha256sums_sha256": "c" * 64,
        "created_utc": "first",
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "leaderboard.jsonl"
        entry_id, appended = _append_jsonl(path, row)
        assert appended
        repeated_id, repeated = _append_jsonl(
            path, {**row, "created_utc": "retry"}
        )
        assert repeated_id == entry_id and not repeated
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["entry_id"] == entry_id


def test_leaderboard_rejects_unsafe_existing_files() -> None:
    row = {
        "campaign_id": "campaign",
        "run_id": "run",
        "candidate": {"id": "candidate"},
        "source_identity_sha256": "a" * 64,
        "manifest_canonical_sha256": "b" * 64,
        "status": "failed",
        "artifact_dir": "/artifact",
        "sha256sums_sha256": "c" * 64,
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "leaderboard.jsonl"
        path.write_text("", encoding="utf-8")
        path.chmod(0o644)
        try:
            _append_jsonl(path, row)
        except RuntimeError:
            pass
        else:
            raise AssertionError("leaderboard accepted an unsafe mode")

        path.chmod(0o600)
        hardlink = root / "leaderboard-hardlink.jsonl"
        os.link(path, hardlink)
        try:
            _append_jsonl(path, row)
        except RuntimeError:
            pass
        else:
            raise AssertionError("leaderboard accepted a hard-linked file")


def test_git_identity_is_bound_to_repository_from_other_cwd() -> None:
    expected = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=_ROOT, text=True
    ).strip()
    original = Path.cwd()
    try:
        os.chdir("/")
        identity = _git_identity()
    finally:
        os.chdir(original)
    assert identity["commit"] == expected


def test_output_directory_is_confined_to_ignored_campaign_root() -> None:
    accepted = _ROOT / ".cache" / "rail_balance" / "campaigns" / "unit" / "run"
    assert _safe_output_dir(accepted) == accepted.resolve()
    for rejected in (Path("/tmp/campaign"), _ROOT / "docs" / "campaign"):
        try:
            _safe_output_dir(rejected)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"unsafe output path accepted: {rejected}")


def test_supervisor_import_does_not_import_torch_or_deep_ep() -> None:
    code = (
        "import sys; import run_rail_balance_hop_campaign; "
        "assert 'torch' not in sys.modules; assert 'deep_ep' not in sys.modules"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parent)
    subprocess.run(
        (sys.executable, "-B", "-c", code),
        cwd=_ROOT,
        env=environment,
        check=True,
    )


def test_terminal_exit_does_not_report_failed_campaign_as_success() -> None:
    assert _terminal_exit("complete", False) == ("PASS", 0)
    assert _terminal_exit("scaffold_only", True) == ("SCAFFOLD", 0)
    assert _terminal_exit("partial", True) == ("SCAFFOLD", 0)
    for status in (
        "scaffold_only",
        "partial",
        "failed",
        "blocked_disk",
        "environment_lost",
    ):
        assert _terminal_exit(status, False) == ("FAIL", 3)


def test_finalization_reserves_space_before_writing_evidence() -> None:
    assert not _finalization_reasons(
        artifact_bytes=10,
        artifact_budget_bytes=20_000_000,
        home_bytes=270_000_000_000,
        home_hard_limit_bytes=300_000_000_000,
    )
    assert "insufficient_artifact_finalization_reserve" in _finalization_reasons(
        artifact_bytes=10_000_001,
        artifact_budget_bytes=20_000_000,
        home_bytes=270_000_000_000,
        home_hard_limit_bytes=300_000_000_000,
    )
    assert "insufficient_home_finalization_reserve" in _finalization_reasons(
        artifact_bytes=10,
        artifact_budget_bytes=20_000_000,
        home_bytes=295_000_000_001,
        home_hard_limit_bytes=300_000_000_000,
    )


if __name__ == "__main__":
    tests = sorted(
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} hop campaign CPU contracts")
