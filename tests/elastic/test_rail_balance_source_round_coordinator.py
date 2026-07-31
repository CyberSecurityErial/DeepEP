"""CPU-only contracts for the source-version round coordinator."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import rail_balance_source_round_coordinator as coordinator


_ROOT = Path(__file__).resolve().parents[2]


def _run(cwd: Path, *arguments: str) -> str:
    return subprocess.check_output(
        arguments,
        cwd=cwd,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _run(repo, "git", "add", "-A")
    _run(repo, "git", "commit", "-m", message)
    return _run(repo, "git", "rev-parse", "HEAD")


def _expect_error(function, *arguments, **keywords) -> None:
    try:
        function(*arguments, **keywords)
    except coordinator.SourceRoundError:
        pass
    else:
        raise AssertionError("unsafe source-round input was accepted")


def _variant(
    variant_id: str,
    worktree: Path,
    commit: str,
    contract_sha256: str,
) -> dict:
    return {
        "id": variant_id,
        "worktree": str(worktree),
        "source_commit": commit,
        "frozen_execution_contract_sha256": contract_sha256,
        "build_bundle_ref": f"bundles/{variant_id}/build",
        "gate_bundle_ref": f"bundles/{variant_id}/gates",
    }


def _fixture(root: Path, candidate_count: int = 2) -> tuple[dict, Path]:
    control = root / "control"
    control.mkdir()
    _run(control, "git", "init", "-b", "main")
    _run(control, "git", "config", "user.email", "source-round@example.invalid")
    _run(control, "git", "config", "user.name", "Source Round Test")
    for harness_path in coordinator.REQUIRED_HARNESS_PATHS:
        _write(control / harness_path, f"# frozen harness: {harness_path}\n")
    _write(control / "kernel.cu", "// parent\n")
    parent_commit = _commit(control, "parent")

    candidate_commits = []
    for index in range(candidate_count):
        branch = f"candidate-{index + 1}"
        _run(control, "git", "switch", "-c", branch, parent_commit)
        _write(control / "kernel.cu", f"// candidate {index + 1}\n")
        candidate_commits.append(_commit(control, branch))
    _run(control, "git", "switch", "--detach", parent_commit)

    parent_worktree = root / "parent-worktree"
    _run(control, "git", "worktree", "add", "--force", "--detach", str(parent_worktree), parent_commit)
    candidate_worktrees = []
    for index, commit in enumerate(candidate_commits):
        worktree = root / f"candidate-worktree-{index + 1}"
        _run(control, "git", "worktree", "add", "--detach", str(worktree), commit)
        candidate_worktrees.append(worktree)

    artifact_root = root / "artifacts"
    artifact_root.mkdir()
    contract = {
        "algorithm": {"hop_mode": "adaptive", "max_two_hop_percent": 25},
        "shape": {"world_size": 8, "tokens": 1024, "topk": 4, "hidden": 256},
        "timing": {"warmup": 10, "measure": 100, "clock": "cuda_event"},
        "toolchain": {"cuda": "12.8", "driver": "570.172.08", "arch": "sm90"},
        "gpu_mapping": [
            {
                "index": index,
                "uuid": f"GPU-00000000-0000-0000-0000-{index:012x}",
            }
            for index in range(8)
        ],
    }
    contract_sha256 = coordinator.canonical_sha256(contract)
    parent = _variant("parent", parent_worktree, parent_commit, contract_sha256)
    candidates = []
    for index, (commit, worktree) in enumerate(zip(candidate_commits, candidate_worktrees)):
        candidate_id = f"candidate-{index + 1}"
        candidate = _variant(candidate_id, worktree, commit, contract_sha256)
        candidate.update(
            {
                "declared_patch": {
                    "base_commit": parent_commit,
                    "diff_sha256": coordinator.declared_diff_sha256(control, parent_commit, commit),
                    "changed_files": ["kernel.cu"],
                },
                "primary_change": f"single CUDA change {index + 1}",
                "hypothesis": "reduce exposed kernel duration",
                "expected_profile_metrics": ["lower kernel duration"],
                "risks": ["register pressure"],
            }
        )
        candidates.append(candidate)
    harness_sha256 = {
        path: hashlib.sha256((control / path).read_bytes()).hexdigest()
        for path in coordinator.REQUIRED_HARNESS_PATHS
    }
    manifest = {
        "schema_version": 1,
        "round_id": "round-a",
        "mode": "check_only",
        "execution_status": "NOT_RUN",
        "coordinator": {
            "control_worktree": str(control),
            "control_commit": parent_commit,
            "artifact_root": str(artifact_root),
            "lock_path": coordinator.SOURCE_ROUND_LOCK,
        },
        "frozen_execution_contract": contract,
        "source_policy": {
            "parent_id": "parent",
            "allowed_candidate_files": ["kernel.cu"],
            "harness_sha256": harness_sha256,
        },
        "parent": parent,
        "candidates": candidates,
    }
    manifest_path = root / "source-round.json"
    _write(manifest_path, json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return manifest, manifest_path


def _load_preflight_plan(manifest_path: Path) -> tuple[dict, dict, dict]:
    manifest, raw_sha256, canonical_sha256 = coordinator.load_manifest(manifest_path)
    checked = coordinator.preflight(
        manifest,
        current_directory=Path(manifest["coordinator"]["control_worktree"]),
    )
    plan = coordinator.materialize_plan(manifest, raw_sha256, canonical_sha256, checked)
    return manifest, checked, plan


def test_valid_plan_has_one_global_pairwise_abba_baab_schedule() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest, manifest_path = _fixture(root, candidate_count=3)
        fake_bin = root / "fake-bin"
        fake_bin.mkdir()
        _write(fake_bin / "git", "#!/bin/sh\nexit 99\n")
        (fake_bin / "git").chmod(0o755)
        original_path = os.environ.get("PATH")
        try:
            os.environ["PATH"] = str(fake_bin)
            _, checked, plan = _load_preflight_plan(manifest_path)
        finally:
            if original_path is None:
                del os.environ["PATH"]
            else:
                os.environ["PATH"] = original_path
        assert checked["status"] == "passed"
        assert checked["harness_checkout_count"] == 5
        assert checked["coordinator_git"]["path"] == "/usr/bin/git"
        assert len(checked["coordinator_git"]["sha256"]) == 64
        assert plan["execution_status"] == "NOT_RUN"
        assert plan["promotion"] == {
            "eligible": False,
            "candidate_id": None,
            "reason": "no benchmark block has been executed live",
        }
        assert len(plan["blocks"]) == 4 + 4 * len(manifest["candidates"])
        assert [block["variant_id"] for block in plan["blocks"]] == [
            "parent",
            "candidate-1",
            "candidate-2",
            "candidate-3",
            "candidate-1",
            "candidate-2",
            "candidate-3",
            "parent",
            "candidate-1",
            "candidate-2",
            "candidate-3",
            "parent",
            "parent",
            "candidate-1",
            "candidate-2",
            "candidate-3",
        ]
        expected = [
            "parent",
            "candidate",
            "candidate",
            "parent",
            "candidate",
            "parent",
            "parent",
            "candidate",
        ]
        assert all(order == expected for order in plan["pairwise_orders"].values())
        assert len({block["run_id"] for block in plan["blocks"]}) == len(plan["blocks"])
        assert len({block["artifact_dir"] for block in plan["blocks"]}) == len(plan["blocks"])
        assert len({block["raw_benchmark_artifact"] for block in plan["blocks"]}) == len(plan["blocks"])
        assert len({block["jit_dir"] for block in plan["blocks"]}) == len(plan["blocks"])
        for block in plan["blocks"]:
            attempt_dir = (
                f"{block['artifact_dir']}/stages/bench-abba-a1/attempt-01"
            )
            assert block["raw_benchmark_artifact"] == f"{attempt_dir}/report.json"
            assert block["jit_dir"] == f"{attempt_dir}/jit"
        parent_blocks = [block for block in plan["blocks"] if block["role"] == "parent"]
        assert len({block["build_bundle_ref"] for block in parent_blocks}) == 1
        assert len({block["gate_bundle_ref"] for block in parent_blocks}) == 1
        assert all(block["live_execution_required"] for block in plan["blocks"])
        assert all(not block["raw_benchmark_reuse_allowed"] for block in plan["blocks"])


def test_schema_rejects_source_identity_contract_and_candidate_count_attacks() -> None:
    with tempfile.TemporaryDirectory() as directory:
        manifest, _ = _fixture(Path(directory))
        mutations = []
        changed = copy.deepcopy(manifest)
        changed["execution_status"] = "COMPLETE"
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["candidates"] = changed["candidates"][:1]
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["candidates"][1]["source_commit"] = changed["candidates"][0]["source_commit"]
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["candidates"][0]["frozen_execution_contract_sha256"] = "0" * 64
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["candidates"][0]["declared_patch"]["changed_files"] = [
            coordinator.REQUIRED_HARNESS_PATHS[0]
        ]
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        del changed["source_policy"]["harness_sha256"][
            "tests/elastic/run_rail_balance_hop_campaign.py"
        ]
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["frozen_execution_contract"]["gpu_mapping"][7]["uuid"] = changed["frozen_execution_contract"]["gpu_mapping"][0]["uuid"]
        mutations.append(changed)
        changed = copy.deepcopy(manifest)
        changed["coordinator"]["lock_path"] = "/tmp/user-selected.lock"
        mutations.append(changed)
        for mutation in mutations:
            _expect_error(coordinator.validate_manifest, mutation)


def test_strict_json_rejects_duplicates_nonfinite_and_symlink_components() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        duplicate = root / "duplicate.json"
        _write(duplicate, '{"schema_version":1,"schema_version":1}\n')
        nonfinite = root / "nonfinite.json"
        _write(nonfinite, '{"value":NaN}\n')
        for path in (duplicate, nonfinite):
            _expect_error(coordinator.load_manifest, path)
        real = root / "real"
        real.mkdir()
        target = real / "manifest.json"
        _write(target, "{}\n")
        linked_parent = root / "linked"
        linked_parent.symlink_to(real, target_is_directory=True)
        _expect_error(coordinator.load_manifest, linked_parent / "manifest.json")

        fixture_root = root / "owner-controlled"
        fixture_root.mkdir()
        valid_manifest, valid_path = _fixture(fixture_root)
        del valid_manifest
        hardlink = root / "manifest-hardlink.json"
        os.link(valid_path, hardlink)
        _expect_error(coordinator.load_manifest, valid_path)
        hardlink.unlink()
        valid_path.chmod(0o666)
        _expect_error(coordinator.load_manifest, valid_path)

        oversized = root / "oversized.json"
        oversized.write_bytes(b" " * (coordinator._MAX_JSON_BYTES + 1))
        _expect_error(coordinator.load_manifest, oversized)


def test_preflight_rejects_dirty_stale_patch_wrong_control_and_bundle_symlink() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest, _ = _fixture(root)
        control = Path(manifest["coordinator"]["control_worktree"])
        _expect_error(coordinator.preflight, manifest, current_directory=root)

        stale = copy.deepcopy(manifest)
        stale["candidates"][0]["declared_patch"]["diff_sha256"] = "0" * 64
        _expect_error(coordinator.preflight, stale, current_directory=control)

        candidate_worktree = Path(manifest["candidates"][0]["worktree"])
        _write(candidate_worktree / "untracked.txt", "dirty\n")
        _expect_error(coordinator.preflight, manifest, current_directory=control)
        (candidate_worktree / "untracked.txt").unlink()

        linked_worktree = root / "linked-candidate-worktree"
        linked_worktree.symlink_to(candidate_worktree, target_is_directory=True)
        linked = copy.deepcopy(manifest)
        linked["candidates"][0]["worktree"] = str(linked_worktree)
        _expect_error(coordinator.preflight, linked, current_directory=control)

        artifact_root = Path(manifest["coordinator"]["artifact_root"])
        (artifact_root / "bundles").mkdir()
        (artifact_root / "bundles" / "candidate-1").symlink_to(root, target_is_directory=True)
        _expect_error(coordinator.preflight, manifest, current_directory=control)


def test_preflight_rejects_different_git_common_dir_and_nonancestor_parent() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest, _ = _fixture(root)
        control = Path(manifest["coordinator"]["control_worktree"])

        foreign = root / "foreign"
        _run(root, "git", "clone", "--quiet", str(control), str(foreign))
        _run(foreign, "git", "checkout", "--detach", manifest["candidates"][1]["source_commit"])
        different_common = copy.deepcopy(manifest)
        different_common["candidates"][1]["worktree"] = str(foreign)
        _expect_error(coordinator.preflight, different_common, current_directory=control)

        nonancestor = copy.deepcopy(manifest)
        old_parent = copy.deepcopy(nonancestor["parent"])
        first_candidate = copy.deepcopy(nonancestor["candidates"][0])
        nonancestor["parent"] = {
            **old_parent,
            "worktree": first_candidate["worktree"],
            "source_commit": first_candidate["source_commit"],
        }
        nonancestor["candidates"][0] = {
            **first_candidate,
            "worktree": old_parent["worktree"],
            "source_commit": old_parent["source_commit"],
            "declared_patch": {
                "base_commit": first_candidate["source_commit"],
                "diff_sha256": coordinator.declared_diff_sha256(
                    control,
                    first_candidate["source_commit"],
                    old_parent["source_commit"],
                ),
                "changed_files": ["kernel.cu"],
            },
        }
        for candidate in nonancestor["candidates"]:
            candidate["declared_patch"]["base_commit"] = first_candidate["source_commit"]
            if candidate["id"] == "candidate-2":
                candidate["declared_patch"]["diff_sha256"] = coordinator.declared_diff_sha256(
                    control,
                    first_candidate["source_commit"],
                    candidate["source_commit"],
                )
        _expect_error(coordinator.preflight, nonancestor, current_directory=control)


def test_preflight_rejects_a_candidate_stacked_on_its_sibling() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest, _ = _fixture(root)
        control = Path(manifest["coordinator"]["control_worktree"])
        parent_commit = manifest["parent"]["source_commit"]
        first_commit = manifest["candidates"][0]["source_commit"]
        _run(control, "git", "switch", "-c", "stacked-candidate", first_commit)
        _write(control / "kernel.cu", "// stacked candidate\n")
        stacked_commit = _commit(control, "stacked candidate")
        _run(control, "git", "switch", "--detach", parent_commit)
        stacked_worktree = root / "stacked-worktree"
        _run(
            control,
            "git",
            "worktree",
            "add",
            "--detach",
            str(stacked_worktree),
            stacked_commit,
        )
        stacked = copy.deepcopy(manifest)
        stacked["candidates"][1]["worktree"] = str(stacked_worktree)
        stacked["candidates"][1]["source_commit"] = stacked_commit
        stacked["candidates"][1]["declared_patch"] = {
            "base_commit": parent_commit,
            "diff_sha256": coordinator.declared_diff_sha256(
                control,
                parent_commit,
                stacked_commit,
            ),
            "changed_files": ["kernel.cu"],
        }
        _expect_error(coordinator.preflight, stacked, current_directory=control)


def test_artifact_confinement_and_atomic_no_clobber_are_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact_root = root / "artifacts"
        artifact_root.mkdir()
        outside = root / "outside.json"
        _expect_error(coordinator._safe_output_path, outside, artifact_root)
        target = artifact_root / "target.json"
        _write(target, "old\n")
        link = artifact_root / "link.json"
        link.symlink_to(target)
        _expect_error(coordinator._safe_output_path, link, artifact_root)
        output = artifact_root / "plan.json"
        safe = coordinator._safe_output_path(output, artifact_root)
        coordinator._atomic_json_no_clobber(safe, {"execution_status": "NOT_RUN"})
        original = output.read_bytes()
        _expect_error(coordinator._atomic_json_no_clobber, output, {"bad": math.nan})
        _expect_error(coordinator._atomic_json_no_clobber, output, {"replacement": True})
        assert output.read_bytes() == original


def test_check_only_cli_writes_not_run_plan_and_imports_no_cuda_packages() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest, manifest_path = _fixture(root)
        control = Path(manifest["coordinator"]["control_worktree"])
        output = Path(manifest["coordinator"]["artifact_root"]) / "plan.json"
        original = Path.cwd()
        try:
            os.chdir(control)
            captured = StringIO()
            with redirect_stdout(captured):
                coordinator.main(
                    (
                        "--manifest",
                        str(manifest_path),
                        "--output",
                        str(output),
                        "--check-only",
                    )
                )
        finally:
            os.chdir(original)
        plan = json.loads(output.read_text(encoding="utf-8"))
        assert plan["execution_status"] == "NOT_RUN"
        assert plan["promotion"]["eligible"] is False
        assert captured.getvalue().startswith("NOT_RUN source-version round plan")

        code = (
            "import sys; import rail_balance_source_round_coordinator; "
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


def test_source_round_uses_campaign_gpu_lease_and_fails_on_contention() -> None:
    assert coordinator.SOURCE_ROUND_LOCK == "/tmp/deepep-rail-balance-hop-campaign.lock"
    descriptor = os.open(
        coordinator.SOURCE_ROUND_LOCK,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def contend() -> None:
            with coordinator._global_lock(coordinator.SOURCE_ROUND_LOCK):
                raise AssertionError("contended global GPU lease was acquired")

        _expect_error(contend)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


if __name__ == "__main__":
    tests = sorted(
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} source-round coordinator CPU contracts")
