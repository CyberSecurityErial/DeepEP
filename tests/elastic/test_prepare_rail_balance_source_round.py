"""CPU-only contracts for the source-round manifest freezer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import prepare_rail_balance_source_round as preparer
import rail_balance_source_round_coordinator as coordinator
import rail_balance_source_round_executor as executor


_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_SOURCE = "csrc/kernels/internode_ll.cu"


def _run(cwd: Path, *arguments: str) -> str:
    return subprocess.check_output(
        arguments,
        cwd=cwd,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _git_read(cwd: Path, *arguments: str, text: bool = False) -> bytes | str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.check_output(
        (coordinator.PINNED_GIT, "--no-pager", "-C", str(cwd), *arguments),
        env=environment,
        stderr=subprocess.PIPE,
        text=text,
    )


def _git_state(fixture: "Fixture") -> dict[str, dict[str, bytes | str]]:
    result: dict[str, dict[str, bytes | str]] = {}
    for worktree in (
        fixture.control,
        fixture.parent_worktree,
        *fixture.candidate_worktrees,
    ):
        raw_index = str(
            _git_read(worktree, "rev-parse", "--git-path", "index", text=True)
        ).strip()
        index_path = Path(raw_index)
        if not index_path.is_absolute():
            index_path = worktree / index_path
        result[str(worktree)] = {
            "head": str(_git_read(worktree, "rev-parse", "HEAD", text=True)).strip(),
            "status": bytes(
                _git_read(
                    worktree,
                    "status",
                    "--porcelain=v1",
                    "-z",
                    "--untracked-files=all",
                )
            ),
            "refs": bytes(_git_read(worktree, "show-ref", "--head")),
            "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        }
    return result


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _run(repo, "git", "add", "-A")
    _run(repo, "git", "commit", "-m", message)
    return _run(repo, "git", "rev-parse", "HEAD")


def _contract() -> dict[str, Any]:
    template = json.loads(
        (_ROOT / preparer.CAMPAIGN_TEMPLATE_PATH).read_text(encoding="utf-8")
    )
    return {
        "algorithm": {
            "hop_mode": "adaptive",
            "two_hop_threshold_percent": 0,
            "max_two_hop_percent": 25,
            "hop_penalty_percent": 50,
        },
        "shape": {
            "stage": "source",
            "case_name": "c100_volume_h256",
            "world_size": 8,
            "hidden": 256,
            "input_iteration": 100,
            "init_dist_seed": 100,
            "remainder_seed": 100,
            "tokens_per_rank": [1024] * 8,
            "num_topk": 4,
            "num_channels": 256,
            "num_experts": 72,
            "num_destinations": 9,
            "local_destination": 0,
            "proxy_capacity_per_egress": 896,
            "dtype": "bfloat16",
            "interference_mode": "none",
            "interference_compute_shape": [2048, 2048, 2048],
        },
        "timing": {
            "warmup_iterations": 10,
            "steady_iterations": 100,
            "clock": "time.perf_counter_ns common monotonic host clock",
            "stage_truth": "max(stage_end)-min(stage_start) across 8 ranks",
            "percentile_method": "linear interpolation over sorted samples",
            "std_method": "population",
            "logical_token_bytes": 576,
            "logical_bytes_per_iteration": 1,
            "logical_bandwidth_denominator": "global target-stage span",
        },
        "toolchain": {"cuda": "12.8", "arch": "sm90"},
        "gpu_mapping": [
            dict(row)
            for row in template["resources"]["expected_gpu_index_uuid_mapping"]
        ],
    }


class Fixture:
    def __init__(
        self,
        root: Path,
        *,
        candidate_count: int = 2,
        stacked: bool = False,
        forbidden: bool = False,
    ) -> None:
        self.root = root
        self.control = root / "control"
        self.control.mkdir()
        _run(self.control, "git", "init", "-b", "main")
        _run(self.control, "git", "config", "user.email", "round@example.invalid")
        _run(self.control, "git", "config", "user.name", "Round Test")
        for relative in preparer.HARNESS_PATHS:
            destination = self.control / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_ROOT / relative, destination)
        _write(self.control / _ALLOWED_SOURCE, "// parent\n")
        self.parent_commit = _commit(self.control, "parent")

        self.candidate_commits: list[str] = []
        base = self.parent_commit
        for index in range(candidate_count):
            candidate_base = base if stacked and index > 0 else self.parent_commit
            _run(
                self.control,
                "git",
                "switch",
                "-c",
                f"candidate-{index + 1}",
                candidate_base,
            )
            if forbidden and index == 0:
                _write(self.control / "forbidden.cu", "// forbidden candidate\n")
            else:
                _write(
                    self.control / _ALLOWED_SOURCE,
                    f"// candidate {index + 1}\n",
                )
            commit = _commit(self.control, f"candidate-{index + 1}")
            self.candidate_commits.append(commit)
            base = commit
        _run(self.control, "git", "switch", "--detach", self.parent_commit)

        self.parent_worktree = root / "parent-worktree"
        _run(
            self.control,
            "git",
            "worktree",
            "add",
            "--detach",
            str(self.parent_worktree),
            self.parent_commit,
        )
        self.candidate_worktrees = []
        for index, commit in enumerate(self.candidate_commits):
            path = root / f"candidate-worktree-{index + 1}"
            _run(
                self.control,
                "git",
                "worktree",
                "add",
                "--detach",
                str(path),
                commit,
            )
            self.candidate_worktrees.append(path)

        self.artifact_root = root / "artifacts"
        self.artifact_root.mkdir(mode=0o700)
        self.spec = {
            "schema_version": 1,
            "round_id": f"round-{candidate_count}",
            "control_worktree": str(self.control),
            "artifact_root": str(self.artifact_root),
            "frozen_execution_contract": _contract(),
            "allowed_candidate_files": [_ALLOWED_SOURCE],
            "parent": {"id": "parent", "worktree": str(self.parent_worktree)},
            "candidates": [
                {
                    "id": f"candidate-{index + 1}",
                    "worktree": str(worktree),
                    "primary_change": f"one CUDA change {index + 1}",
                    "hypothesis": "reduce exposed source-stage latency",
                    "expected_profile_metrics": ["lower source kernel duration"],
                    "risks": ["register pressure"],
                }
                for index, worktree in enumerate(self.candidate_worktrees)
            ],
        }
        self.spec_path = root / "human-spec.json"
        self.write_spec()

    def write_spec(self) -> None:
        self.spec_path.write_text(
            json.dumps(self.spec, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def outputs(self, suffix: str = "") -> tuple[Path, Path]:
        return (
            self.artifact_root / f"source-round{suffix}.json",
            self.artifact_root / f"plan{suffix}.json",
        )


def _invoke_copied_preparer(
    fixture: Fixture,
    spec_path: Path,
    manifest_path: Path,
    plan_path: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{fixture.control / 'tests/elastic'}:{fixture.control}"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        (
            sys.executable,
            "-B",
            str(fixture.control / preparer.PREPARER_PATH),
            "--spec",
            str(spec_path),
            "--manifest-output",
            str(manifest_path),
            "--plan-output",
            str(plan_path),
        ),
        cwd=fixture.control,
        env=environment,
        text=True,
        capture_output=True,
    )


def _expect_error(function: Callable[..., Any], *args: Any) -> str:
    try:
        function(*args)
    except (
        preparer.PreparationError,
        coordinator.SourceRoundError,
        executor.ExecutorError,
    ) as error:
        return str(error)
    raise AssertionError("unsafe source-round preparation was accepted")


def test_import_is_stdlib_only() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parent)
    subprocess.run(
        (
            sys.executable,
            "-B",
            "-c",
            "import sys; import prepare_rail_balance_source_round; "
            "assert 'torch' not in sys.modules; assert 'deep_ep' not in sys.modules",
        ),
        cwd=_ROOT,
        env=environment,
        check=True,
    )


def test_two_three_four_candidate_specs_freeze_exact_manifest_and_plan() -> None:
    assert len(preparer.HARNESS_PATHS) == 20
    for candidate_count in (2, 3, 4):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), candidate_count=candidate_count)
            manifest_path, plan_path = fixture.outputs()
            git_state_before = _git_state(fixture)
            completed = _invoke_copied_preparer(
                fixture,
                fixture.spec_path,
                manifest_path,
                plan_path,
            )
            assert completed.returncode == 0, completed.stderr
            manifest, raw_sha, canonical_sha = coordinator.load_manifest(manifest_path)
            plan, _, plan_raw_sha = executor._strict_json(plan_path)
            executor._validate_plan(manifest, raw_sha, canonical_sha, plan)
            assert plan_raw_sha == hashlib.sha256(plan_path.read_bytes()).hexdigest()
            assert len(plan["blocks"]) == 4 + 4 * candidate_count
            assert manifest["source_policy"]["harness_sha256"] == {
                relative: hashlib.sha256(
                    (fixture.control / relative).read_bytes()
                ).hexdigest()
                for relative in preparer.HARNESS_PATHS
            }
            assert [row["source_commit"] for row in manifest["candidates"]] == (
                fixture.candidate_commits
            )
            assert all(
                row["declared_patch"]["changed_files"] == [_ALLOWED_SOURCE]
                for row in manifest["candidates"]
            )
            assert stat_mode(manifest_path) == 0o600
            assert stat_mode(plan_path) == 0o600
            assert manifest_path.stat().st_nlink == plan_path.stat().st_nlink == 1
            rendered = completed.stdout
            assert "PREPARED_NOT_RUN" in rendered
            assert "EXECUTOR_CHECK_ONLY:" in rendered
            assert "EXECUTOR_LIVE:" in rendered
            assert f"--confirm-round-id round-{candidate_count}" in rendered
            check_only = next(
                line.removeprefix("EXECUTOR_CHECK_ONLY: ")
                for line in rendered.splitlines()
                if line.startswith("EXECUTOR_CHECK_ONLY: ")
            )
            check_environment = os.environ.copy()
            check_environment["CUDA_VISIBLE_DEVICES"] = ""
            checked = subprocess.run(
                ("bash", "-c", check_only),
                check=True,
                text=True,
                capture_output=True,
                env=check_environment,
            )
            assert checked.stderr == ""
            assert (
                checked.stdout.strip()
                == f"NOT_RUN source-round executor check-only: round=round-{candidate_count} "
                f"blocks={4 + 4 * candidate_count} launch=false"
            )
            assert _git_state(fixture) == git_state_before


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_dirty_stacked_and_forbidden_candidates_fail_before_publication() -> None:
    attacks = ("dirty", "stacked", "forbidden")
    for attack in attacks:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(
                Path(directory),
                stacked=attack == "stacked",
                forbidden=attack == "forbidden",
            )
            if attack == "dirty":
                _write(fixture.candidate_worktrees[0] / "untracked.txt", "dirty\n")
            manifest_path, plan_path = fixture.outputs()
            completed = _invoke_copied_preparer(
                fixture,
                fixture.spec_path,
                manifest_path,
                plan_path,
            )
            assert completed.returncode == 2
            expected = {
                "dirty": "must be clean, including untracked files",
                "stacked": "must be one independent commit above parent",
                "forbidden": "changes a file outside allowed_candidate_files",
            }[attack]
            assert expected in completed.stderr
            assert not manifest_path.exists()
            assert not plan_path.exists()


def test_duplicate_nonfinite_and_symlink_specs_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        duplicate = fixture.root / "duplicate.json"
        duplicate.write_text(
            '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
        )
        nonfinite = fixture.root / "nonfinite.json"
        nonfinite.write_text('{"schema_version":NaN}\n', encoding="utf-8")
        symlink = fixture.root / "spec-link.json"
        symlink.symlink_to(fixture.spec_path)
        invalid_specs = (
            (duplicate, "duplicate JSON key: schema_version"),
            (nonfinite, "non-finite JSON constant is forbidden: NaN"),
            (symlink, "human spec contains a symlink"),
        )
        for index, (spec_path, expected) in enumerate(invalid_specs):
            manifest_path, plan_path = fixture.outputs(f"-{index}")
            completed = _invoke_copied_preparer(
                fixture,
                spec_path,
                manifest_path,
                plan_path,
            )
            assert completed.returncode == 2
            assert expected in completed.stderr
            assert not manifest_path.exists()
            assert not plan_path.exists()


def test_manifest_and_plan_outputs_are_independently_no_clobber() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        manifest_path, plan_path = fixture.outputs()
        manifest_path.write_text("sentinel\n", encoding="utf-8")
        completed = _invoke_copied_preparer(
            fixture,
            fixture.spec_path,
            manifest_path,
            plan_path,
        )
        assert completed.returncode == 2
        assert (
            completed.stderr
            == "ERROR: output already exists; check-only plans are immutable\n"
        )
        assert manifest_path.read_text(encoding="utf-8") == "sentinel\n"
        assert not plan_path.exists()

        manifest_path.unlink()
        plan_path.write_text("plan sentinel\n", encoding="utf-8")
        completed = _invoke_copied_preparer(
            fixture,
            fixture.spec_path,
            manifest_path,
            plan_path,
        )
        assert completed.returncode == 2
        assert (
            completed.stderr
            == "ERROR: output already exists; check-only plans are immutable\n"
        )
        assert not manifest_path.exists()
        assert plan_path.read_text(encoding="utf-8") == "plan sentinel\n"


def test_second_link_oserror_leaves_orphan_plan_without_terminal_manifest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path = root / "source-round.json"
        plan_path = root / "plan.json"
        manifest_payload = b'{"terminal":"manifest"}\n'
        plan_payload = b'{"status":"NOT_RUN"}\n'
        original_link = preparer.os.link
        calls = 0

        def fail_second_link(*args: Any, **kwargs: Any) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic second-link race")
            original_link(*args, **kwargs)

        preparer.os.link = fail_second_link
        try:
            _expect_error(
                preparer._publish_pair_no_clobber,
                manifest_path,
                manifest_payload,
                plan_path,
                plan_payload,
            )
        finally:
            preparer.os.link = original_link
        assert calls == 2
        assert not manifest_path.exists()
        assert plan_path.read_bytes() == plan_payload
        assert {path.name for path in root.iterdir()} == {plan_path.name}


def test_async_publish_failures_leave_orphan_and_never_remove_replacement() -> None:
    for exception in (KeyboardInterrupt(), SystemExit(17)):
        for replace_plan in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path = root / "source-round.json"
                plan_path = root / "plan.json"
                manifest_payload = b'{"terminal":"manifest"}\n'
                plan_payload = b'{"status":"NOT_RUN"}\n'
                replacement = b"replacement owned by another actor\n"
                original_link = preparer.os.link
                calls = 0

                def interrupt_second_link(
                    *args: Any,
                    _replace_plan: bool = replace_plan,
                    _plan_path: Path = plan_path,
                    _replacement: bytes = replacement,
                    _exception: BaseException = exception,
                    _original_link: Callable[..., Any] = original_link,
                    **kwargs: Any,
                ) -> None:
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        if _replace_plan:
                            _plan_path.unlink()
                            _plan_path.write_bytes(_replacement)
                        raise _exception
                    _original_link(*args, **kwargs)

                preparer.os.link = interrupt_second_link
                try:
                    try:
                        preparer._publish_pair_no_clobber(
                            manifest_path,
                            manifest_payload,
                            plan_path,
                            plan_payload,
                        )
                    except BaseException as observed:
                        assert observed is exception
                    else:
                        raise AssertionError(
                            "asynchronous publish failure was swallowed"
                        )
                finally:
                    preparer.os.link = original_link
                assert calls == 2
                assert not manifest_path.exists()
                assert plan_path.read_bytes() == (
                    replacement if replace_plan else plan_payload
                )
                assert {path.name for path in root.iterdir()} == {plan_path.name}


def test_actual_control_cwd_and_preparer_file_are_mandatory() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        manifest_path, plan_path = fixture.outputs()
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            f"{fixture.control / 'tests/elastic'}:{fixture.control}"
        )
        wrong_cwd = subprocess.run(
            (
                sys.executable,
                "-B",
                str(fixture.control / preparer.PREPARER_PATH),
                "--spec",
                str(fixture.spec_path),
                "--manifest-output",
                str(manifest_path),
                "--plan-output",
                str(plan_path),
            ),
            cwd=_ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        assert wrong_cwd.returncode == 2
        assert (
            wrong_cwd.stderr
            == "ERROR: preparer must run from the neutral control worktree\n"
        )

        environment["PYTHONPATH"] = f"{_ROOT / 'tests/elastic'}:{_ROOT}"
        wrong_file = subprocess.run(
            (
                sys.executable,
                "-B",
                str(_ROOT / preparer.PREPARER_PATH),
                "--spec",
                str(fixture.spec_path),
                "--manifest-output",
                str(manifest_path),
                "--plan-output",
                str(plan_path),
            ),
            cwd=fixture.control,
            env=environment,
            text=True,
            capture_output=True,
        )
        assert wrong_file.returncode == 2
        assert (
            wrong_file.stderr
            == "ERROR: running preparer module is not the frozen control-worktree file\n"
        )
        assert not manifest_path.exists()
        assert not plan_path.exists()


def test_direct_cli_imports_helpers_without_writing_bytecode() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        cache = fixture.root / "pycache-prefix"
        environment = os.environ.copy()
        environment.pop("PYTHONDONTWRITEBYTECODE", None)
        environment["PYTHONPYCACHEPREFIX"] = str(cache)
        environment["PYTHONPATH"] = (
            f"{fixture.control / 'tests/elastic'}:{fixture.control}"
        )
        completed = subprocess.run(
            (
                sys.executable,
                str(fixture.control / preparer.PREPARER_PATH),
                "--help",
            ),
            cwd=fixture.control,
            env=environment,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr
        local_modules = {
            Path(relative).stem
            for relative in preparer.HARNESS_PATHS
            if relative.endswith(".py")
        }
        local_bytecode = [
            path
            for path in cache.rglob("*.pyc")
            if any(path.name.startswith(f"{module}.") for module in local_modules)
        ]
        assert local_bytecode == []


def test_held_directory_rejects_path_replacement_and_mode_drift() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output_parent = root / "output"
        output_parent.mkdir(mode=0o700)
        held = preparer._hold_directory(output_parent, "output parent")
        try:
            displaced = root / "displaced-output"
            output_parent.rename(displaced)
            output_parent.mkdir(mode=0o700)
            error = _expect_error(preparer._verify_held_directory, held)
            assert error == "output parent path was replaced after preparation began"
        finally:
            os.close(held.descriptor)

        artifact_root = root / "artifacts"
        artifact_root.mkdir(mode=0o700)
        held = preparer._hold_directory(
            artifact_root,
            "artifact_root",
            exact_mode=0o700,
        )
        try:
            artifact_root.chmod(0o755)
            error = _expect_error(preparer._verify_held_directory, held)
            assert error == "held artifact_root must have mode 0700"
        finally:
            os.close(held.descriptor)


if __name__ == "__main__":
    tests = sorted(
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} source-round preparer CPU contracts")
