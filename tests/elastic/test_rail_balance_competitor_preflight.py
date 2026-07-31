"""CPU-only safety contracts for the local competitor preflight."""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import rail_balance_competitor_preflight as preflight


_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = (
    Path(__file__).with_name("experiments")
    / "rail_balance_competitor_local_v1.json"
)


def _run(cwd: Path, *argv: str) -> str:
    return subprocess.check_output(
        argv,
        cwd=cwd,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _expect_error(function: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    try:
        function(*args, **kwargs)
    except preflight.PreflightError as error:
        return str(error)
    raise AssertionError("unsafe competitor preflight input was accepted")


def _manifest() -> dict[str, Any]:
    value, _raw, _raw_sha, _canonical_sha = preflight.load_manifest(_MANIFEST)
    return value


class GitFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "competitor"
        self.repo.mkdir()
        _run(self.repo, "git", "init", "-b", "main")
        _run(self.repo, "git", "config", "user.email", "preflight@example.invalid")
        _run(self.repo, "git", "config", "user.name", "Preflight Test")
        subtree = self.repo / "ep"
        subtree.mkdir()
        (subtree / "kernel.cu").write_text("// frozen\n", encoding="utf-8")
        _run(self.repo, "git", "add", "ep/kernel.cu")
        _run(self.repo, "git", "commit", "-m", "frozen")
        commit = _run(self.repo, "git", "rev-parse", "HEAD")
        source_tree = _run(self.repo, "git", "rev-parse", "HEAD^{tree}")
        subtree_tree = _run(self.repo, "git", "rev-parse", "HEAD:ep")
        content = preflight._tracked_subtree_content(self.repo, "ep")
        build_inputs = preflight._tracked_build_input_content(self.repo, (".",))
        self.competitor: dict[str, Any] = {
            "id": "fixture",
            "checkout": str(self.repo),
            "commit": commit,
            "source_tree": source_tree,
            "subtree": "ep",
            "subtree_tree": subtree_tree,
            "subtree_content_sha256": content["sha256"],
            "build_source_policy": "PINNED_GIT_OBJECTS_TO_FRESH_STAGE",
            "source_materialization": {
                "method": "PINNED_GIT_OBJECTS_ONLY",
                "paths": ["."],
                "destination": "{stage_root}/fixture/source-copy",
                "expected_content_sha256": build_inputs["sha256"],
            },
            "future_build_stages": [],
        }


def _command(
    stdout: str = "", *, returncode: int | None = 0, stderr: str = ""
) -> dict[str, Any]:
    return {
        "argv": [],
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": None if returncode is not None else "unavailable",
        "duration_ns": 1,
    }


def _gpu_mapping_output(manifest: dict[str, Any]) -> str:
    return "".join(
        f"{row['index']}, {row['uuid']}, NVIDIA L20X, 143771, Default, Disabled\n"
        for row in manifest["resources"]["expected_gpu_mapping"]
    )


def _topology_output(manifest: dict[str, Any]) -> str:
    columns = [f"GPU{index}" for index in range(8)] + [
        row["topology_label"] for row in manifest["network"]["nics"]
    ]
    lines = ["        " + " ".join(columns) + " CPU Affinity NUMA Affinity"]
    for gpu_index in range(8):
        gpu_links = ["X" if peer == gpu_index else "NV18" for peer in range(8)]
        nic_links = ["PIX" if nic == gpu_index else "SYS" for nic in range(8)]
        lines.append(
            f"GPU{gpu_index} "
            + " ".join(gpu_links + nic_links)
            + f" 0-47,96-143 {gpu_index // 4} N/A"
        )
    return "\n".join(lines) + "\n"


def test_real_manifest_has_exact_schema_and_frozen_build_safety() -> None:
    manifest = _manifest()
    assert manifest["mode"] == "check_only"
    assert manifest["execution_contract"] == {
        "gpu_launch_policy": "FORBIDDEN",
        "build_execution_status": "NOT_RUN",
        "future_build_executor_status": "NOT_IMPLEMENTED",
        "performance_claim_allowed": False,
        "checkout_write_policy": "FORBIDDEN",
        "stage_root_policy": "FRESH_EMPTY_OWNER_0700_NO_SYMLINKS_NO_REUSE",
        "source_materialization_policy": "PINNED_GIT_OBJECTS_ONLY",
    }
    competitors = {item["id"]: item for item in manifest["competitors"]}
    uccl = competitors["uccl-ep"]
    assert uccl["build_source_policy"] == "PINNED_GIT_OBJECTS_TO_FRESH_STAGE"
    assert uccl["source_materialization"]["paths"] == ["ep", "include"]
    uccl_stage = uccl["future_build_stages"][0]
    assert uccl_stage["cwd"].startswith("{stage_root}/uccl/source-copy/")
    assert "all" in uccl_stage["argv"]
    assert "install" not in uccl_stage["argv"]
    assert "DETECTED_SM=90" in uccl_stage["argv"]
    assert "SM=90" in uccl_stage["argv"]
    assert "GPU_NAME=NVIDIA L20X" in uccl_stage["argv"]
    assert uccl_stage["env"]["CUDA_VISIBLE_DEVICES"] == ""
    guard = uccl_stage["gpu_query_guard"]
    assert guard is not None
    guard_payload = bytes.fromhex(guard["content_hex"])
    assert preflight.hashlib.sha256(guard_payload).hexdigest() == guard["sha256"]
    assert guard_payload == b"#!/bin/sh\nexit 97\n"
    assert uccl_stage["env"]["PATH"].split(":", 1)[0] == str(
        Path(guard["path"]).parent
    )

    nccl = competitors["nccl-ep"]
    assert nccl["build_source_policy"] == "PINNED_GIT_OBJECTS_TO_FRESH_STAGE"
    assert nccl["source_materialization"]["paths"] == ["."]
    for stage in nccl["future_build_stages"]:
        joined = "\n".join(stage["argv"])
        assert "BUILDDIR={stage_root}/nccl-ep/" in joined
        assert "/home/chen/workspace/infra/" not in joined
        assert "{stage_root}/nccl-ep/source-copy" in joined
        assert stage["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert "src.build" in nccl["future_build_stages"][0]["argv"]
    ep_stage = nccl["future_build_stages"][1]
    assert "NCCL_EP_BUILDDIR={stage_root}/nccl-ep/nccl-ep" in ep_stage["argv"]
    assert "ep_test" in ep_stage["argv"]
    assert "ep_bench" in ep_stage["argv"]


def test_uccl_eager_make_detection_is_intercepted_by_frozen_toolshim() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = _manifest()
        uccl = next(item for item in manifest["competitors"] if item["id"] == "uccl-ep")
        stage = uccl["future_build_stages"][0]
        guard = stage["gpu_query_guard"]
        assert guard is not None
        shim_dir = root / "toolshim"
        shim_dir.mkdir()
        shim = shim_dir / "nvidia-smi"
        shim.write_bytes(bytes.fromhex(guard["content_hex"]))
        shim.chmod(guard["mode"])

        makefile = root / "Makefile"
        makefile.write_text(
            "DETECTED_SM := $(shell nvidia-smi --query-gpu=compute_cap)\n"
            "SM ?= $(DETECTED_SM)\n"
            "GPU_NAME ?= $(shell nvidia-smi --query-gpu=name)\n"
            "all:\n"
            "\t@printf '%s|%s|%s\\n' '$(DETECTED_SM)' '$(SM)' '$(GPU_NAME)'\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            (
                "/usr/bin/make",
                "-f",
                str(makefile),
                "-n",
                "all",
                "DETECTED_SM=90",
                "GPU_NAME=NVIDIA L20X",
            ),
            cwd=directory,
            env={
                "HOME": str(root),
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": f"{shim_dir}:/usr/bin:/bin",
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        # GNU make eagerly expands the Makefile's ``:=`` RHS even when a
        # command-line value wins.  The leading frozen shim therefore matters:
        # DETECTED_SM/SM/GPU_NAME alone do not prevent an NVML query.
        assert "nvidia-smi" not in completed.stderr
        assert "'90' '90' 'NVIDIA L20X'" in completed.stdout


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        duplicate = root / "duplicate.json"
        duplicate.write_text(
            '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
        )
        error = _expect_error(preflight.load_manifest, duplicate)
        assert error == "duplicate JSON key: schema_version"

        nonfinite = root / "nonfinite.json"
        nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
        error = _expect_error(preflight.load_manifest, nonfinite)
        assert error == "non-finite JSON constant is forbidden: NaN"

        writable = root / "writable.json"
        writable.write_bytes(_MANIFEST.read_bytes())
        writable.chmod(0o660)
        assert "manifest is group/other writable" in _expect_error(
            preflight.load_manifest, writable
        )


def test_exact_schema_and_pinned_tool_path_reject_escape() -> None:
    manifest = _manifest()
    unknown = copy.deepcopy(manifest)
    unknown["unknown"] = True
    assert "fields are incomplete or unknown" in _expect_error(
        preflight.validate_manifest, unknown
    )

    escaped = copy.deepcopy(manifest)
    git_tool = next(tool for tool in escaped["tools"] if tool["id"] == "git")
    git_tool["path"] = "/tmp/git"
    error = _expect_error(preflight.validate_manifest, escaped)
    assert "differs from the pinned git identity" in error


def test_checkout_rejects_symlink_and_stale_head_but_ignores_untracked() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = GitFixture(Path(directory))
        stale = copy.deepcopy(fixture.competitor)
        stale["commit"] = "0" * 40
        assert _expect_error(preflight._inspect_checkout, stale) == (
            "fixture HEAD is stale"
        )

        link = fixture.root / "competitor-link"
        link.symlink_to(fixture.repo, target_is_directory=True)
        linked = copy.deepcopy(fixture.competitor)
        linked["checkout"] = str(link)
        error = _expect_error(preflight._inspect_checkout, linked)
        assert "checkout contains a symlink" in error

        (fixture.repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        result = preflight._inspect_checkout(fixture.competitor)
        assert result["git_status_execution_status"] == "NOT_RUN"
        assert result["worktree_untracked_policy"] == "IGNORED_NOT_BUILD_INPUT"


def test_checkout_content_hash_detects_assume_unchanged_edit() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = GitFixture(Path(directory))
        _run(
            fixture.repo,
            "git",
            "update-index",
            "--assume-unchanged",
            "ep/kernel.cu",
        )
        (fixture.repo / "ep/kernel.cu").write_text("// hidden edit\n", encoding="utf-8")
        error = _expect_error(preflight._inspect_checkout, fixture.competitor)
        assert error == "fixture subtree content SHA256 changed"


def test_checkout_build_input_closure_detects_hidden_edit_outside_subtree() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = GitFixture(Path(directory))
        source = fixture.repo / "src/core.cc"
        source.parent.mkdir()
        source.write_text("// core\n", encoding="utf-8")
        _run(fixture.repo, "git", "add", "src/core.cc")
        _run(fixture.repo, "git", "commit", "-m", "core input")
        fixture.competitor["commit"] = _run(
            fixture.repo, "git", "rev-parse", "HEAD"
        )
        fixture.competitor["source_tree"] = _run(
            fixture.repo, "git", "rev-parse", "HEAD^{tree}"
        )
        fixture.competitor["source_materialization"][
            "expected_content_sha256"
        ] = preflight._tracked_build_input_content(fixture.repo, (".",))["sha256"]

        _run(fixture.repo, "git", "update-index", "--assume-unchanged", "src/core.cc")
        source.write_text("// hidden core edit\n", encoding="utf-8")
        error = _expect_error(preflight._inspect_checkout, fixture.competitor)
        assert error == "fixture build input closure SHA256 changed"


def test_ignored_worktree_file_is_never_a_future_build_input() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = GitFixture(Path(directory))
        excluded = fixture.repo / ".git/info/exclude"
        with excluded.open("a", encoding="utf-8") as output:
            output.write("src/ignored.cc\n")
        ignored = fixture.repo / "src/ignored.cc"
        ignored.parent.mkdir()
        ignored.write_text("$(shell false)\n", encoding="utf-8")
        result = preflight._inspect_checkout(fixture.competitor)
        assert result["git_status_execution_status"] == "NOT_RUN"
        assert result["source_materialization"]["method"] == (
            "PINNED_GIT_OBJECTS_ONLY"
        )
        assert all(
            row["path"] != "src/ignored.cc"
            for row in result["build_input_closure"]["files"]
        )


def test_checkout_inspection_never_invokes_git_conversion_filters() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = GitFixture(Path(directory))
        marker = fixture.root / "filter-ran"
        filter_script = fixture.root / "filter.sh"
        filter_script.write_text(
            f"#!/bin/sh\ntouch {marker}\ncat\n",
            encoding="utf-8",
        )
        filter_script.chmod(0o700)
        info_attributes = fixture.repo / ".git/info/attributes"
        info_attributes.write_text("ep/kernel.cu filter=evil\n", encoding="utf-8")
        _run(
            fixture.repo,
            "git",
            "config",
            "filter.evil.clean",
            str(filter_script),
        )
        _run(fixture.repo, "git", "config", "filter.evil.required", "true")
        tracked = fixture.repo / "ep/kernel.cu"
        tracked_stat = tracked.stat()
        os.utime(
            tracked,
            ns=(tracked_stat.st_atime_ns, tracked_stat.st_mtime_ns + 1_000_000_000),
        )

        result = preflight._inspect_checkout(fixture.competitor)
        assert result["status"] == "PASSED"
        assert result["git_status_execution_status"] == "NOT_RUN"
        assert not marker.exists()


def test_git_probe_disables_lazy_fetch_hooks_and_replace_objects() -> None:
    completed = mock.Mock(returncode=0, stdout=b"ok\n", stderr=b"")
    with mock.patch.object(preflight.subprocess, "run", return_value=completed) as run:
        assert preflight._git(Path("/tmp/fixture"), "rev-parse", "HEAD") == "ok\n"
    command = run.call_args.args[0]
    environment = run.call_args.kwargs["env"]
    assert "protocol.allow=never" in command
    assert "core.hooksPath=/dev/null" in command
    assert environment["GIT_NO_LAZY_FETCH"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert run.call_args.kwargs["timeout"] == preflight.COMMAND_TIMEOUT_SECONDS

    timeout = subprocess.TimeoutExpired(command, preflight.COMMAND_TIMEOUT_SECONDS)
    with mock.patch.object(preflight.subprocess, "run", side_effect=timeout):
        error = _expect_error(
            preflight._git,
            Path("/tmp/fixture"),
            "rev-parse",
            "HEAD",
        )
    assert error == "pinned Git timed out after 180 seconds"


def test_checkout_post_recheck_rejects_mid_probe_tamper() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = GitFixture(Path(directory))
        before = [preflight._inspect_checkout(fixture.competitor)]
        _run(
            fixture.repo,
            "git",
            "update-index",
            "--assume-unchanged",
            "ep/kernel.cu",
        )
        (fixture.repo / "ep/kernel.cu").write_text(
            "// changed during probes\n", encoding="utf-8"
        )
        error = _expect_error(
            preflight._recheck_checkouts,
            before,
            [fixture.competitor],
        )
        assert error == "fixture subtree content SHA256 changed"


def test_subtree_hash_accepts_only_contained_tracked_symlinks() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = GitFixture(Path(directory))
        target = fixture.repo / "ep/target.py"
        target.write_text("value = 1\n", encoding="utf-8")
        link = fixture.repo / "ep/link.py"
        link.symlink_to("target.py")
        _run(fixture.repo, "git", "add", "ep/target.py", "ep/link.py")
        _run(fixture.repo, "git", "commit", "-m", "contained link")
        result = preflight._tracked_subtree_content(fixture.repo, "ep")
        assert result["tracked_symlink_count"] == 1

        link.unlink()
        link.symlink_to("../../outside")
        _run(fixture.repo, "git", "add", "ep/link.py")
        _run(fixture.repo, "git", "commit", "-m", "escaping link")
        assert "tracked symlink escapes or is dangling" in _expect_error(
            preflight._tracked_subtree_content, fixture.repo, "ep"
        )


def test_output_is_atomic_and_no_clobber() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "result.json"
        digest = preflight._atomic_json_no_clobber(output, {"status": "NOT_RUN"})
        raw = output.read_bytes()
        assert digest == preflight.hashlib.sha256(raw).hexdigest()
        assert json.loads(raw) == {"status": "NOT_RUN"}
        assert output.stat().st_mode & 0o777 == 0o444
        before = raw
        error = _expect_error(
            preflight._atomic_json_no_clobber,
            output,
            {"status": "STATIC_CHECKS_PASSED_NO_BUILD_AUTHORITY"},
        )
        assert "immutable output already exists" in error
        assert output.read_bytes() == before


def test_cli_output_cannot_write_source_or_frozen_checkout() -> None:
    good = preflight.OUTPUT_ROOT / "run-01/result.json"
    assert preflight._validate_cli_output_path(good) == good
    for unsafe in (
        _ROOT / "result.json",
        Path("/home/chen/workspace/infra/nccl-ep-v0.1.0/result.json"),
        preflight.OUTPUT_ROOT / "result.json",
        preflight.OUTPUT_ROOT / "run-01/not-result.json",
    ):
        _expect_error(preflight._validate_cli_output_path, unsafe)


def test_disk_budget_is_decimal_and_fail_closed() -> None:
    resources = _manifest()["resources"]
    exact = preflight._disk_probe(_command("291000000000 /home/chen\n"), resources)
    assert exact["projected_home_bytes"] == 300_000_000_000
    assert exact["status"] == "PASSED"
    over = preflight._disk_probe(
        _command("291000000001 /home/chen\n"), resources
    )
    assert over["status"] == "BLOCKED_DISK"
    invalid = preflight._disk_probe(_command("not-a-number\n"), resources)
    assert invalid["status"] == "BLOCKED_DISK"
    for ambiguous in (
        "-9000000000 /home/chen\n",
        "0 /not-home\n",
        "0 /home/chen\n999999999999 /home/chen\n",
    ):
        assert preflight._disk_probe(
            _command(ambiguous), resources
        )["status"] == "BLOCKED_DISK"


def test_gpu_snapshot_records_pids_and_mapping_fail_closed() -> None:
    manifest = _manifest()
    mapping = manifest["resources"]["expected_gpu_mapping"]
    processes = (
        f"{mapping[0]['uuid']}, 3047501, python, 71123\n"
        f"{mapping[1]['uuid']}, 3047502, python, 71123\n"
    )
    result = preflight._gpu_probe(
        _command(_gpu_mapping_output(manifest)),
        _command(processes),
        mapping,
    )
    assert result["mapping_status"] == "PASSED"
    assert result["availability_status"] == "WAITING_GPU"
    assert result["occupied_pids"] == [3047501, 3047502]

    bad_mapping = _gpu_mapping_output(manifest).replace(
        mapping[0]["uuid"], mapping[1]["uuid"], 1
    )
    bad = preflight._gpu_probe(_command(bad_mapping), _command(), mapping)
    assert bad["mapping_status"] == "BLOCKED_GPU_MAPPING"
    assert "gpu_index_uuid_mapping_mismatch" in bad["reasons"]

    unknown = preflight._gpu_probe(
        _command(_gpu_mapping_output(manifest)),
        _command(returncode=1, stderr="NVML failed"),
        mapping,
    )
    assert unknown["mapping_status"] == "PASSED"
    assert unknown["process_status"] == "BLOCKED"
    assert unknown["availability_status"] == "UNKNOWN"
    assert "gpu_process_command_failed" in unknown["reasons"]


def test_topology_requires_nv18_and_matching_pix() -> None:
    manifest = _manifest()
    good = preflight._topology_probe(
        _command(_topology_output(manifest)), manifest["network"]["nics"]
    )
    assert good["status"] == "PASSED"
    bad_output = _topology_output(manifest).replace("GPU0 X NV18", "GPU0 X SYS", 1)
    bad = preflight._topology_probe(
        _command(bad_output), manifest["network"]["nics"]
    )
    assert bad["status"] == "BLOCKED_TOPOLOGY"


def test_nanobind_has_distinct_blocker() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dependencies = [
            {
                "id": "uccl-nanobind",
                "competitor_id": "uccl-ep",
                "kind": "file",
                "required": True,
                "visibility_paths": [str(root / "missing-nanobind.py")],
                "missing_status": "BLOCKED_DEPENDENCY_NANOBIND",
            }
        ]
        result = preflight._probe_dependencies(dependencies)
        assert result["status"] == "BLOCKED"
        assert result["blocker_statuses"] == ["BLOCKED_DEPENDENCY_NANOBIND"]


def test_dependency_contract_cannot_be_removed_optional_or_substituted() -> None:
    manifest = _manifest()

    missing = copy.deepcopy(manifest)
    missing["dependencies"].pop()
    assert "exact sorted frozen dependency set" in _expect_error(
        preflight.validate_manifest, missing
    )

    optional = copy.deepcopy(manifest)
    optional["dependencies"][0]["required"] = False
    assert ".required must remain true" in _expect_error(
        preflight.validate_manifest, optional
    )

    substituted = copy.deepcopy(manifest)
    substituted["dependencies"][0]["visibility_paths"] = ["/tmp/cuda.h"]
    assert "differs from its frozen dependency contract" in _expect_error(
        preflight.validate_manifest, substituted
    )


def test_future_build_environment_is_exact_and_rejects_make_injection() -> None:
    manifest = _manifest()
    competitors = {item["id"]: item for item in manifest["competitors"]}

    makefiles = copy.deepcopy(manifest)
    makefiles["competitors"][0]["future_build_stages"][0]["env"][
        "MAKEFILES"
    ] = "/tmp/injected.mk"
    assert "frozen future build environment" in _expect_error(
        preflight.validate_manifest, makefiles
    )

    preload = copy.deepcopy(manifest)
    preload["competitors"][1]["future_build_stages"][0]["env"][
        "LD_PRELOAD"
    ] = "/tmp/injected.so"
    assert "frozen future build environment" in _expect_error(
        preflight.validate_manifest, preload
    )

    assert (
        competitors["uccl-ep"]["future_build_stages"][0]["env"]
        == preflight._EXPECTED_BUILD_ENV["uccl-ep-extension"]
    )
    assert preflight._select_status([]) == (
        "STATIC_CHECKS_PASSED_NO_BUILD_AUTHORITY"
    )


def test_future_source_output_and_host_contracts_are_exact() -> None:
    manifest = _manifest()

    worktree_build = copy.deepcopy(manifest)
    worktree_build["competitors"][0]["build_source_policy"] = (
        "FROZEN_CHECKOUT_WITH_EXTERNAL_BUILDDIRS"
    )
    assert "build_source_policy is unsafe" in _expect_error(
        preflight.validate_manifest, worktree_build
    )

    bad_materialization = copy.deepcopy(manifest)
    bad_materialization["competitors"][0]["source_materialization"]["paths"] = [
        "contrib/nccl_ep"
    ]
    assert "differs from the frozen closure" in _expect_error(
        preflight.validate_manifest, bad_materialization
    )

    escaped_output = copy.deepcopy(manifest)
    escaped_output["competitors"][0]["future_build_stages"][0][
        "expected_outputs"
    ][0] = "{stage_root}/../../frozen-checkout/owned"
    assert "traverses or is ambiguous" in _expect_error(
        preflight.validate_manifest, escaped_output
    )

    changed_tool = copy.deepcopy(manifest)
    changed_tool["tools"][0]["sha256"] = "0" * 64
    assert "differs from the pinned du identity" in _expect_error(
        preflight.validate_manifest, changed_tool
    )

    changed_nic = copy.deepcopy(manifest)
    changed_nic["network"]["nics"][0]["state"] = "1: DOWN"
    assert "network NIC contract differs" in _expect_error(
        preflight.validate_manifest, changed_nic
    )


def test_run_preflight_dispatches_only_frozen_static_probe_argv() -> None:
    manifest = _manifest()
    _loaded, _raw, raw_sha256, canonical_sha256 = preflight.load_manifest(
        _MANIFEST
    )
    captured: list[list[str]] = []

    def fake_capture(argv: Any, _timeout: int) -> dict[str, Any]:
        command = list(argv)
        captured.append(command)
        if command == manifest["probes"]["du_argv"]:
            return _command("280000000000 /home/chen\n")
        if command == manifest["probes"]["gpu_mapping_argv"]:
            return _command(_gpu_mapping_output(manifest))
        if command == manifest["probes"]["gpu_process_argv"]:
            return _command()
        if command == manifest["probes"]["topology_argv"]:
            return _command(_topology_output(manifest))
        if command == manifest["probes"]["ibv_devices_argv"]:
            return _command()
        raise AssertionError(f"unexpected command: {command}")

    passed_network = {
        "nics": [],
        "ibv_devices": {},
        "ibv_devinfo": {"execution_status": "NOT_RUN"},
        "reasons": [],
        "status": "PASSED",
    }
    with mock.patch.object(
        preflight,
        "_inspect_tool",
        side_effect=lambda tool: {"id": tool["id"], "status": "PASSED"},
    ), mock.patch.object(
        preflight,
        "_inspect_checkout",
        side_effect=lambda item: {"id": item["id"], "status": "PASSED"},
    ), mock.patch.object(
        preflight,
        "_network_probe",
        return_value=passed_network,
    ):
        result = preflight.run_preflight(
            manifest,
            manifest_path=_MANIFEST,
            manifest_raw_sha256=raw_sha256,
            manifest_canonical_sha256=canonical_sha256,
            capture=fake_capture,
        )
    assert captured == [
        manifest["probes"]["du_argv"],
        manifest["probes"]["gpu_mapping_argv"],
        manifest["probes"]["gpu_process_argv"],
        manifest["probes"]["topology_argv"],
        manifest["probes"]["ibv_devices_argv"],
    ]
    build_argvs = {
        tuple(stage["argv"])
        for competitor in manifest["competitors"]
        for stage in competitor["future_build_stages"]
    }
    assert not build_argvs & {tuple(argv) for argv in captured}
    assert result["execution_contract"]["build_execution_status"] == "NOT_RUN"
    assert result["execution_contract"]["performance_claim_allowed"] is False
    assert result["probe_capture_identity"] == "TEST_DOUBLE_CAPTURE"
    assert result["interpreter"]["sys_executable"] == os.path.abspath(sys.executable)
    assert result["runtime_closure"]["status"] == "TEST_NOT_ATTESTED"
    assert "BLOCKED_DEPENDENCY_NANOBIND" in result["status_observations"]


def test_run_preflight_rejects_forged_manifest_hash_and_interpreter() -> None:
    manifest = _manifest()
    _loaded, _raw, raw_sha256, canonical_sha256 = preflight.load_manifest(
        _MANIFEST
    )
    assert "raw manifest SHA256 is stale or forged" in _expect_error(
        preflight.run_preflight,
        manifest,
        manifest_path=_MANIFEST,
        manifest_raw_sha256="0" * 64,
        manifest_canonical_sha256=canonical_sha256,
    )
    with mock.patch.object(preflight.sys, "executable", "/nonexistent/python"):
        assert "cannot resolve" in _expect_error(
            preflight._inspect_interpreter,
            require_isolated_startup=False,
        )
    interpreter = preflight._inspect_interpreter(require_isolated_startup=False)
    assert interpreter["sha256"] == (
        preflight._PINNED_TOOL_IDENTITIES["python"][2]
    )
    assert interpreter["startup_contract_status"] == "TEST_NONATTESTED"
    assert "preflight interpreter must be /usr/bin/python3" in _expect_error(
        preflight._inspect_interpreter,
        require_isolated_startup=True,
    )
    assert raw_sha256 != "0" * 64


def test_process_launch_binds_kernel_argv_and_empty_environment() -> None:
    output = preflight.OUTPUT_ROOT / "launch-contract/result.json"
    command = [
        preflight._PINNED_PREFLIGHT_INTERPRETER[0],
        "-I",
        "-S",
        "-B",
        str(preflight.PROGRAM_PATH),
        "--manifest",
        str(preflight.MANIFEST_PATH),
        "--output",
        str(output),
    ]
    environment = dict(preflight._EXPECTED_PROCESS_ENVIRONMENT)

    def fields(_path: Path, name: str) -> tuple[bytes, list[str]]:
        values = (
            command
            if name == "process cmdline"
            else [f"{key}={value}" for key, value in environment.items()]
        )
        raw = b"\0".join(value.encode() for value in values) + b"\0"
        return raw, values

    with mock.patch.object(
        preflight, "_read_proc_nul_fields", side_effect=fields
    ), mock.patch.dict(os.environ, environment, clear=True):
        result = preflight._inspect_process_launch(
            manifest_path=preflight.MANIFEST_PATH,
            output_path=output,
        )
        assert result["status"] == "PINNED_ARGV_EMPTY_ENV"

        command.insert(1, "-c")
        assert "frozen production argv" in _expect_error(
            preflight._inspect_process_launch,
            manifest_path=preflight.MANIFEST_PATH,
            output_path=output,
        )
        command.pop(1)
        environment["LD_PRELOAD"] = "/tmp/injected.so"
        assert "empty-environment allowlist" in _expect_error(
            preflight._inspect_process_launch,
            manifest_path=preflight.MANIFEST_PATH,
            output_path=output,
        )


def test_proc_self_exe_cannot_be_spoofed_by_argv_zero() -> None:
    expected = preflight._PINNED_PREFLIGHT_INTERPRETER[1]
    with mock.patch.object(preflight.os, "readlink", return_value=expected):
        assert preflight._inspect_proc_executable(expected)["realpath"] == expected
    with mock.patch.object(preflight.os, "readlink", return_value="/usr/bin/false"):
        assert "/proc/self/exe differs" in _expect_error(
            preflight._inspect_proc_executable, expected
        )


def test_module_is_stdlib_only_and_forbids_accelerator_imports() -> None:
    source_path = Path(preflight.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/false\n")
    tree = ast.parse(source, filename=str(source_path))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_git"
        ):
            literal_arguments = {
                argument.value
                for argument in node.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
            }
            assert "status" not in literal_arguments
    assert not imported_roots & {"torch", "deep_ep", "uccl", "numpy"}
    assert imported_roots <= set(sys.stdlib_module_names) | {"__future__"}


if __name__ == "__main__":
    tests = sorted(
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    os.chdir(_ROOT)
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} competitor preflight CPU contracts")
