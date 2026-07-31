#!/usr/bin/env python3
"""Freeze a human-authored source-round spec into immutable CPU-only inputs.

This preparer never creates or changes a Git worktree, launches a benchmark,
or selects a candidate.  It derives objective identities from already-clean
control, parent, and candidate worktrees, runs the existing coordinator
preflight, and publishes a source manifest plus its exact NOT_RUN plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import sys
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Sequence

# The CLI is read-only with respect to every Git checkout.  In particular,
# importing its local stdlib-only helpers must not create ignored __pycache__
# files when the script is invoked directly without ``python -B``.
sys.dont_write_bytecode = True

import rail_balance_source_round_coordinator as coordinator  # noqa: E402
import rail_balance_source_round_executor as source_executor  # noqa: E402


SPEC_SCHEMA_VERSION = 1
PREPARER_PATH = "tests/elastic/prepare_rail_balance_source_round.py"
CAMPAIGN_TEMPLATE_PATH = "tests/elastic/experiments/hop_local8_sm90_v1.json"
PINNED_PYTHON = "/home/chen/.cache/deepep-sjlgpt/bin/python"
MAX_SPEC_BYTES = 8 * 1024 * 1024
HARNESS_PATHS = tuple(sorted({*coordinator.REQUIRED_HARNESS_PATHS, PREPARER_PATH}))
_RUNTIME_MODULE_PATHS: tuple[tuple[str, ModuleType, str], ...] = (
    (
        "source-round coordinator",
        coordinator,
        "tests/elastic/rail_balance_source_round_coordinator.py",
    ),
    (
        "source-round executor",
        source_executor,
        source_executor.SOURCE_ROUND_EXECUTOR_PATH,
    ),
    (
        "campaign round evaluator",
        source_executor.round_evaluator,
        source_executor.EVALUATOR_PATH,
    ),
    (
        "campaign schema",
        source_executor.campaign_schema,
        source_executor.SCHEMA_PATH,
    ),
    (
        "campaign runner",
        source_executor.campaign_runner,
        source_executor.RUNNER_PATH,
    ),
)
_RUNTIME_CALLABLE_PATHS: tuple[tuple[str, Callable[..., Any], str], ...] = (
    (
        "coordinator.validate_manifest",
        coordinator.validate_manifest,
        "tests/elastic/rail_balance_source_round_coordinator.py",
    ),
    (
        "coordinator.preflight",
        coordinator.preflight,
        "tests/elastic/rail_balance_source_round_coordinator.py",
    ),
    (
        "coordinator.materialize_plan",
        coordinator.materialize_plan,
        "tests/elastic/rail_balance_source_round_coordinator.py",
    ),
    (
        "executor._validate_live_contract",
        source_executor._validate_live_contract,
        source_executor.SOURCE_ROUND_EXECUTOR_PATH,
    ),
    (
        "executor._validate_plan",
        source_executor._validate_plan,
        source_executor.SOURCE_ROUND_EXECUTOR_PATH,
    ),
)


class PreparationError(ValueError):
    """The human spec cannot be frozen into an executable source round."""


@dataclass(frozen=True)
class _HeldDirectory:
    name: str
    path: Path
    descriptor: int
    device: int
    inode: int
    owner: int
    mode: int
    exact_mode: int | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def _directory_identity(
    descriptor: int,
    name: str,
    *,
    exact_mode: int | None,
) -> tuple[int, int, int, int]:
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise PreparationError(f"cannot inspect held {name}: {error}") from error
    mode = stat.S_IMODE(metadata.st_mode)
    _require(stat.S_ISDIR(metadata.st_mode), f"held {name} is not a directory")
    _require(metadata.st_uid == os.geteuid(), f"held {name} has a foreign owner")
    if exact_mode is None:
        _require(
            mode & 0o022 == 0,
            f"held {name} is group/other writable",
        )
    else:
        _require(mode == exact_mode, f"held {name} must have mode {exact_mode:04o}")
    return metadata.st_dev, metadata.st_ino, metadata.st_uid, mode


def _hold_directory(
    path: Path,
    name: str,
    *,
    exact_mode: int | None = None,
) -> _HeldDirectory:
    descriptor = coordinator._open_absolute_directory_no_follow(path)
    try:
        device, inode, owner, mode = _directory_identity(
            descriptor,
            name,
            exact_mode=exact_mode,
        )
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise
    return _HeldDirectory(
        name=name,
        path=path,
        descriptor=descriptor,
        device=device,
        inode=inode,
        owner=owner,
        mode=mode,
        exact_mode=exact_mode,
    )


def _verify_held_directory(held: _HeldDirectory) -> None:
    expected = (held.device, held.inode, held.owner, held.mode)
    observed = _directory_identity(
        held.descriptor,
        held.name,
        exact_mode=held.exact_mode,
    )
    _require(observed == expected, f"held {held.name} identity or mode changed")
    reopened = coordinator._open_absolute_directory_no_follow(held.path)
    try:
        path_observed = _directory_identity(
            reopened,
            f"reopened {held.name}",
            exact_mode=held.exact_mode,
        )
    finally:
        with suppress(OSError):
            os.close(reopened)
    _require(
        path_observed == expected,
        f"{held.name} path was replaced after preparation began",
    )


@contextmanager
def _hold_publication_directories(
    artifact_root: Path,
    manifest_parent: Path,
    plan_parent: Path,
) -> Iterator[tuple[_HeldDirectory, _HeldDirectory, _HeldDirectory]]:
    held: list[_HeldDirectory] = []
    try:
        held.append(_hold_directory(artifact_root, "artifact_root", exact_mode=0o700))
        held.append(_hold_directory(manifest_parent, "manifest output parent"))
        held.append(_hold_directory(plan_parent, "plan output parent"))
        for directory in held:
            _verify_held_directory(directory)
        yield held[0], held[1], held[2]
    finally:
        for directory in reversed(held):
            with suppress(OSError):
                os.close(directory.descriptor)


def _load_spec(path: Path) -> tuple[dict[str, Any], Path]:
    path = coordinator._absolute_without_resolving(path)
    coordinator._assert_existing_path_has_no_symlink(path, "human spec")
    raw = coordinator._read_regular_no_follow(
        path,
        MAX_SPEC_BYTES,
        owner_controlled=True,
    )
    try:
        value = json.loads(
            raw,
            object_pairs_hook=coordinator._unique_object,
            parse_constant=coordinator._reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise PreparationError(f"invalid human spec {path}: {error}") from error
    _require(isinstance(value, dict), "human spec root must be an object")
    _validate_spec(value)
    return value, path


def _validate_spec(spec: dict[str, Any]) -> None:
    coordinator._validate_json_tree(spec, "human spec")
    coordinator._exact_fields(
        spec,
        {
            "schema_version",
            "round_id",
            "control_worktree",
            "artifact_root",
            "frozen_execution_contract",
            "allowed_candidate_files",
            "parent",
            "candidates",
        },
        "human spec",
    )
    _require(
        spec["schema_version"] == SPEC_SCHEMA_VERSION,
        "unsupported human spec schema_version",
    )
    coordinator._slug(spec["round_id"], "human spec.round_id")
    control = coordinator._absolute_path(
        spec["control_worktree"], "human spec.control_worktree"
    )
    artifact_root = coordinator._absolute_path(
        spec["artifact_root"], "human spec.artifact_root"
    )
    parent = coordinator._exact_fields(
        spec["parent"], {"id", "worktree"}, "human spec.parent"
    )
    parent_id = coordinator._slug(parent["id"], "human spec.parent.id")
    parent_path = coordinator._absolute_path(
        parent["worktree"], "human spec.parent.worktree"
    )
    allowed = coordinator._sorted_unique_relative(
        spec["allowed_candidate_files"],
        "human spec.allowed_candidate_files",
    )
    _require(
        not set(allowed) & set(HARNESS_PATHS),
        "allowed candidate files overlap the frozen harness",
    )
    contract = spec["frozen_execution_contract"]
    _require(isinstance(contract, dict), "frozen execution contract is not an object")

    candidates = spec["candidates"]
    _require(
        isinstance(candidates, list) and 2 <= len(candidates) <= 4,
        "human spec requires two to four candidates",
    )
    candidate_ids: list[str] = []
    candidate_paths: list[Path] = []
    for index, candidate in enumerate(candidates):
        name = f"human spec.candidates[{index}]"
        coordinator._exact_fields(
            candidate,
            {
                "id",
                "worktree",
                "primary_change",
                "hypothesis",
                "expected_profile_metrics",
                "risks",
            },
            name,
        )
        candidate_ids.append(coordinator._slug(candidate["id"], f"{name}.id"))
        candidate_paths.append(
            coordinator._absolute_path(candidate["worktree"], f"{name}.worktree")
        )
        for field in ("primary_change", "hypothesis"):
            coordinator._nonempty_string(candidate[field], f"{name}.{field}")
        for field in ("expected_profile_metrics", "risks"):
            values = candidate[field]
            _require(
                isinstance(values, list) and bool(values),
                f"{name}.{field} must be a nonempty list",
            )
            for item_index, item in enumerate(values):
                coordinator._nonempty_string(item, f"{name}.{field}[{item_index}]")
    _require(
        candidate_ids == sorted(set(candidate_ids)),
        "candidate IDs must be sorted, unique, and deterministic",
    )
    _require(parent_id not in candidate_ids, "candidate ID collides with parent ID")
    paths = [control, parent_path, *candidate_paths]
    _require(
        len({str(path) for path in paths}) == len(paths),
        "control, parent, and candidate worktree paths must be distinct",
    )
    _require(
        all(path != artifact_root for path in paths),
        "artifact root collides with a worktree path",
    )


def _inspect_head(path: Path, name: str) -> dict[str, Any]:
    realpath, _ = coordinator._existing_directory(path, name)
    commit = str(
        coordinator._git(realpath, "rev-parse", "--verify", "HEAD^{commit}")
    ).strip()
    return coordinator._inspect_worktree(realpath, commit, name)


def _artifact_root(path: Path) -> Path:
    root, metadata = coordinator._existing_directory(path, "artifact_root")
    _require(metadata.st_uid == os.geteuid(), "artifact_root has a foreign owner")
    _require(
        stat.S_IMODE(metadata.st_mode) == 0o700,
        "artifact_root must have mode 0700",
    )
    return root


def _runtime_path(value: object, name: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{name} has no source path")
    try:
        return Path(value).resolve(strict=True)
    except OSError as error:
        raise PreparationError(f"cannot resolve {name} source path: {error}") from error


def _verify_runtime_code_identity(control_worktree: Path) -> None:
    expected_preparer = control_worktree / PREPARER_PATH
    _require(
        _runtime_path(__file__, "preparer module") == expected_preparer,
        "running preparer module is not the frozen control-worktree file",
    )
    _require(
        _runtime_path(
            derive_manifest.__code__.co_filename,
            "derive_manifest code",
        )
        == expected_preparer,
        "derive_manifest code is not loaded from the frozen control worktree",
    )
    _require(
        source_executor.source_coordinator is coordinator,
        "executor and preparer loaded different coordinator modules",
    )
    for name, module, relative in _RUNTIME_MODULE_PATHS:
        expected = control_worktree / relative
        _require(
            _runtime_path(getattr(module, "__file__", None), name) == expected,
            f"{name} is not loaded from the frozen control worktree",
        )
    for name, function, relative in _RUNTIME_CALLABLE_PATHS:
        code = getattr(function, "__code__", None)
        _require(code is not None, f"{name} has no Python code identity")
        _require(
            _runtime_path(code.co_filename, name) == control_worktree / relative,
            f"{name} code is not loaded from the frozen control worktree",
        )


def derive_manifest(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive and preflight a source manifest without writing or mutating state."""

    _validate_spec(spec)
    try:
        current_directory = Path.cwd().resolve(strict=True)
    except OSError as error:
        raise PreparationError(
            f"cannot resolve preparer execution identity: {error}"
        ) from error
    control_worktree, _ = coordinator._existing_directory(
        Path(spec["control_worktree"]),
        "control_worktree",
    )
    _require(
        current_directory == control_worktree,
        "preparer must run from the neutral control worktree",
    )
    _verify_runtime_code_identity(control_worktree)
    control = _inspect_head(control_worktree, "control_worktree")
    parent = _inspect_head(Path(spec["parent"]["worktree"]), "parent.worktree")
    candidate_heads = [
        _inspect_head(
            Path(candidate["worktree"]),
            f"candidates[{index}].worktree",
        )
        for index, candidate in enumerate(spec["candidates"])
    ]
    artifact_root = _artifact_root(Path(spec["artifact_root"]))
    contract = spec["frozen_execution_contract"]
    contract_sha256 = coordinator.canonical_sha256(contract)
    harness = {
        relative: coordinator._hash_harness(Path(control["worktree"]), relative)
        for relative in HARNESS_PATHS
    }

    parent_id = spec["parent"]["id"]
    parent_row = {
        "id": parent_id,
        "worktree": parent["worktree"],
        "source_commit": parent["source_commit"],
        "frozen_execution_contract_sha256": contract_sha256,
        "build_bundle_ref": f"bundles/{parent_id}/build",
        "gate_bundle_ref": f"bundles/{parent_id}/gates",
    }
    candidates = []
    for declared, observed in zip(spec["candidates"], candidate_heads):
        changed_files, diff_sha256 = coordinator._git_diff(
            Path(parent["worktree"]),
            parent["source_commit"],
            observed["source_commit"],
        )
        candidate_id = declared["id"]
        candidates.append(
            {
                "id": candidate_id,
                "worktree": observed["worktree"],
                "source_commit": observed["source_commit"],
                "frozen_execution_contract_sha256": contract_sha256,
                "build_bundle_ref": f"bundles/{candidate_id}/build",
                "gate_bundle_ref": f"bundles/{candidate_id}/gates",
                "declared_patch": {
                    "base_commit": parent["source_commit"],
                    "diff_sha256": diff_sha256,
                    "changed_files": changed_files,
                },
                "primary_change": declared["primary_change"],
                "hypothesis": declared["hypothesis"],
                "expected_profile_metrics": declared["expected_profile_metrics"],
                "risks": declared["risks"],
            }
        )

    manifest = {
        "schema_version": coordinator.SCHEMA_VERSION,
        "round_id": spec["round_id"],
        "mode": "check_only",
        "execution_status": "NOT_RUN",
        "coordinator": {
            "control_worktree": control["worktree"],
            "control_commit": control["source_commit"],
            "artifact_root": str(artifact_root),
            "lock_path": coordinator.SOURCE_ROUND_LOCK,
        },
        "frozen_execution_contract": contract,
        "source_policy": {
            "parent_id": parent_id,
            "allowed_candidate_files": spec["allowed_candidate_files"],
            "harness_sha256": harness,
        },
        "parent": parent_row,
        "candidates": candidates,
    }
    coordinator.validate_manifest(manifest)
    source_executor._validate_live_contract(manifest)
    checked = coordinator.preflight(
        manifest,
        current_directory=Path(control["worktree"]),
    )
    return manifest, checked


def _canonical_payload(value: object) -> bytes:
    return coordinator._canonical_bytes(value) + b"\n"


def _stage_payload(
    parent_descriptor: int,
    destination: Path,
    payload: bytes,
    label: str,
) -> str:
    temporary = f".{destination.name}.{os.getpid()}.{time.time_ns()}.{label}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_descriptor,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary, dir_fd=parent_descriptor)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return temporary


def _publish_pair_no_clobber(
    manifest_path: Path,
    manifest_payload: bytes,
    plan_path: Path,
    plan_payload: bytes,
    *,
    held_directories: Sequence[_HeldDirectory] | None = None,
) -> None:
    """Publish plan first and the no-clobber manifest terminal marker last.

    Any failure between the links can leave an orphan plan; a late failure can
    leave both final names.  Final names are never rolled back: cleanup by
    pathname could delete a concurrent actor's replacement.  Without the
    terminal manifest, an orphan plan is unusable and makes a same-name retry
    fail closed.  Uncatchable process death can also leave hidden stage files.
    """

    owned_directories: list[_HeldDirectory] = []
    if held_directories is None:
        try:
            owned_directories.append(
                _hold_directory(manifest_path.parent, "manifest output parent")
            )
            owned_directories.append(
                _hold_directory(plan_path.parent, "plan output parent")
            )
        except BaseException:
            for directory in reversed(owned_directories):
                with suppress(OSError):
                    os.close(directory.descriptor)
            raise
        directories: tuple[_HeldDirectory, ...] = tuple(owned_directories)
    else:
        directories = tuple(held_directories)
        _require(
            len(directories) >= 2,
            "publisher requires held manifest and plan output parents",
        )
    manifest_parent = directories[-2]
    plan_parent = directories[-1]
    _require(
        manifest_parent.path == manifest_path.parent,
        "held manifest output parent does not match destination",
    )
    _require(
        plan_parent.path == plan_path.parent,
        "held plan output parent does not match destination",
    )
    manifest_temporary: str | None = None
    plan_temporary: str | None = None
    try:
        for directory in directories:
            _verify_held_directory(directory)
        manifest_temporary = _stage_payload(
            manifest_parent.descriptor,
            manifest_path,
            manifest_payload,
            "manifest",
        )
        plan_temporary = _stage_payload(
            plan_parent.descriptor,
            plan_path,
            plan_payload,
            "plan",
        )
        for directory in directories:
            _verify_held_directory(directory)
        os.link(
            plan_temporary,
            plan_path.name,
            src_dir_fd=plan_parent.descriptor,
            dst_dir_fd=plan_parent.descriptor,
            follow_symlinks=False,
        )
        os.unlink(plan_temporary, dir_fd=plan_parent.descriptor)
        plan_temporary = None
        os.fsync(plan_parent.descriptor)
        for directory in directories:
            _verify_held_directory(directory)
        os.link(
            manifest_temporary,
            manifest_path.name,
            src_dir_fd=manifest_parent.descriptor,
            dst_dir_fd=manifest_parent.descriptor,
            follow_symlinks=False,
        )
        os.unlink(manifest_temporary, dir_fd=manifest_parent.descriptor)
        manifest_temporary = None
        os.fsync(manifest_parent.descriptor)
        for directory in directories:
            _verify_held_directory(directory)
    except OSError as error:
        raise PreparationError(
            "cannot complete plan-first source-round terminal publication; "
            "final names are never rolled back, so inspect for a terminal "
            f"manifest or orphan plan: {error}"
        ) from error
    finally:
        if manifest_temporary is not None:
            with suppress(OSError):
                os.unlink(manifest_temporary, dir_fd=manifest_parent.descriptor)
        if plan_temporary is not None:
            with suppress(OSError):
                os.unlink(plan_temporary, dir_fd=plan_parent.descriptor)
        for directory in reversed(owned_directories):
            with suppress(OSError):
                os.close(directory.descriptor)


def prepare(
    spec_path: Path,
    manifest_output: Path,
    plan_output: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Freeze, recheck, and terminal-publish one source round."""

    spec, _ = _load_spec(spec_path)
    artifact_root = _artifact_root(Path(spec["artifact_root"]))
    manifest_path = coordinator._safe_output_path(manifest_output, artifact_root)
    plan_path = coordinator._safe_output_path(plan_output, artifact_root)
    _require(manifest_path != plan_path, "manifest and plan outputs must differ")

    with _hold_publication_directories(
        artifact_root,
        manifest_path.parent,
        plan_path.parent,
    ) as held_directories:
        manifest, checked = derive_manifest(spec)
        manifest_payload = _canonical_payload(manifest)
        manifest_raw_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        manifest_canonical_sha256 = coordinator.canonical_sha256(manifest)
        plan = coordinator.materialize_plan(
            manifest,
            manifest_raw_sha256,
            manifest_canonical_sha256,
            checked,
        )
        source_executor._validate_plan(
            manifest,
            manifest_raw_sha256,
            manifest_canonical_sha256,
            plan,
        )
        post_checked = coordinator.preflight(
            manifest,
            current_directory=Path(manifest["coordinator"]["control_worktree"]),
        )
        _require(
            post_checked == checked,
            "source worktrees drifted during preparation",
        )
        _publish_pair_no_clobber(
            manifest_path,
            manifest_payload,
            plan_path,
            _canonical_payload(plan),
            held_directories=held_directories,
        )
    return manifest, plan, manifest_path, plan_path


def _executor_command(
    control_worktree: Path,
    manifest_path: Path,
    plan_path: Path,
    round_id: str,
    *,
    live: bool,
) -> str:
    pythonpath = f"{control_worktree}/tests/elastic:{control_worktree}"
    arguments = [
        PINNED_PYTHON,
        "-B",
        str(control_worktree / source_executor.SOURCE_ROUND_EXECUTOR_PATH),
        "--source-manifest",
        str(manifest_path),
        "--plan",
        str(plan_path),
        "--campaign-template",
        str(control_worktree / CAMPAIGN_TEMPLATE_PATH),
    ]
    if live:
        arguments.extend(("--live", "--confirm-round-id", round_id))
    return (
        f"cd -- {shlex.quote(str(control_worktree))} && "
        f"PYTHONPATH={shlex.quote(pythonpath)} " + shlex.join(arguments)
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest, plan, manifest_path, plan_path = prepare(
        args.spec,
        args.manifest_output,
        args.plan_output,
    )
    control = Path(manifest["coordinator"]["control_worktree"])
    print(
        "PREPARED_NOT_RUN source round: "
        f"round={manifest['round_id']} blocks={len(plan['blocks'])} "
        f"manifest={manifest_path} plan={plan_path}",
        flush=True,
    )
    print(
        "EXECUTOR_CHECK_ONLY: "
        + _executor_command(
            control,
            manifest_path,
            plan_path,
            manifest["round_id"],
            live=False,
        ),
        flush=True,
    )
    print(
        "EXECUTOR_LIVE: "
        + _executor_command(
            control,
            manifest_path,
            plan_path,
            manifest["round_id"],
            live=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        PreparationError,
        coordinator.SourceRoundError,
        source_executor.ExecutorError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error
