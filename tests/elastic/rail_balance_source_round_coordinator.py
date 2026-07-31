#!/usr/bin/env python3
"""Fail-closed, check-only planner for CUDA source-version rounds.

This module is deliberately stdlib-only.  It validates clean Git worktrees and
materializes the one global interleaving required for a formal parent versus
two-to-four candidate round.  It never compiles code, launches a benchmark, or
promotes a candidate.  A successful result therefore remains ``NOT_RUN``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Iterator, Sequence


SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
# This must be the same global GPU lease used by run_rail_balance_hop_campaign.
# A separate coordinator lock would permit two nominally supervised 8-GPU jobs
# to overlap.
SOURCE_ROUND_LOCK = "/tmp/deepep-rail-balance-hop-campaign.lock"
PINNED_GIT = "/usr/bin/git"
REQUIRED_HARNESS_PATHS = (
    "tests/elastic/bench_rail_balance_hop.py",
    "tests/elastic/bench_rail_balance_hybrid_lsa.py",
    "tests/elastic/experiments/hop_local8_sm90_v1.json",
    "tests/elastic/rail_balance_campaign_round.py",
    "tests/elastic/rail_balance_campaign_schema.py",
    "tests/elastic/rail_balance_source_round_coordinator.py",
    "tests/elastic/rail_balance_source_round_executor.py",
    "tests/elastic/run_rail_balance_build_warmup.py",
    "tests/elastic/run_rail_balance_hop_campaign.py",
    "tests/elastic/test_rail_balance_hop_bench.py",
    "tests/elastic/test_rail_balance_hop_one_hop_cuda.py",
    "tests/elastic/test_rail_balance_hop_plan.py",
    "tests/elastic/test_rail_balance_hop_vnode_cuda.py",
    "tests/elastic/test_rail_balance_hybrid_api.py",
    "tests/elastic/test_rail_balance_hybrid_combine_codegen.py",
    "tests/elastic/test_rail_balance_hybrid_dispatch_codegen.py",
    "tests/elastic/test_rail_balance_hybrid_layout.py",
    "tests/elastic/test_rail_balance_hybrid_policy.py",
    "tests/elastic/test_rail_balance_hybrid_public_lifecycle.py",
)
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_HASHED_FILE_BYTES = 256 * 1024 * 1024
_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,47}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GPU_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)


class SourceRoundError(ValueError):
    """The source-round manifest or local checkout is unsafe or ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceRoundError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceRoundError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SourceRoundError(f"non-finite JSON constant is forbidden: {value}")


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SourceRoundError(f"value is not strict JSON: {error}") from error


def canonical_sha256(value: object) -> str:
    """Return the semantic identity used for frozen contracts and plans."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _read_regular_no_follow(
    path: Path, maximum_bytes: int, *, owner_controlled: bool = False
) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        if owner_controlled:
            _require(before.st_uid == os.geteuid(), f"file has a foreign owner: {path}")
            _require(before.st_nlink == 1, f"file is hard-linked: {path}")
            _require(
                before.st_mode & 0o022 == 0,
                f"file is group/other writable: {path}",
            )
        _require(before.st_size <= maximum_bytes, f"file is too large: {path}")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        _require(len(payload) <= maximum_bytes, f"file is too large: {path}")
        after = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"file changed while being read: {path}",
        )
        _require(len(payload) == after.st_size, f"short read: {path}")
        return payload
    except OSError as error:
        raise SourceRoundError(f"cannot read {path}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def load_manifest(path: Path) -> tuple[dict[str, Any], str, str]:
    """Read strict JSON without following a final-component symlink."""

    path = _absolute_without_resolving(path)
    _assert_existing_path_has_no_symlink(path, "manifest")
    raw = _read_regular_no_follow(
        path, _MAX_JSON_BYTES, owner_controlled=True
    )
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise SourceRoundError(f"invalid JSON manifest {path}: {error}") from error
    _require(isinstance(value, dict), "manifest root must be an object")
    validate_manifest(value)
    return (
        value,
        hashlib.sha256(raw).hexdigest(),
        canonical_sha256(value),
    )


def _validate_json_tree(value: object, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"{name} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str), f"{name} has a non-string key")
            _validate_json_tree(item, f"{name}.{key}")
        return
    raise SourceRoundError(f"{name} contains a non-JSON value")


def _exact_fields(value: object, fields: set[str], name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    _require(set(value) == fields, f"{name} fields are incomplete or unknown")
    return value


def _nonempty_string(value: object, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} is empty")
    _require("\x00" not in value, f"{name} contains NUL")
    return value


def _slug(value: object, name: str) -> str:
    value = _nonempty_string(value, name)
    _require(bool(_SLUG.fullmatch(value)), f"{name} is not a safe slug")
    return value


def _sha256_text(value: object, name: str) -> str:
    value = _nonempty_string(value, name)
    _require(bool(_SHA256.fullmatch(value)), f"{name} is not a lowercase SHA256")
    return value


def _git_object(value: object, name: str) -> str:
    value = _nonempty_string(value, name)
    _require(bool(_GIT_OBJECT.fullmatch(value)), f"{name} is not a full Git object ID")
    return value


def _absolute_path(value: object, name: str) -> Path:
    raw = _nonempty_string(value, name)
    path = Path(raw)
    _require(path.is_absolute(), f"{name} must be absolute")
    normalized = _absolute_without_resolving(path)
    _require(str(path) == str(normalized), f"{name} must be normalized")
    return path


def _safe_relative(value: object, name: str) -> str:
    value = _nonempty_string(value, name)
    _require(
        "\\" not in value and all(ord(character) >= 32 and ord(character) != 127 for character in value),
        f"{name} contains an unsafe path character",
    )
    path = Path(value)
    _require(not path.is_absolute(), f"{name} must be relative")
    _require(value == path.as_posix(), f"{name} must use normalized POSIX separators")
    _require(path.parts and all(part not in {"", ".", ".."} for part in path.parts), f"{name} traverses or is ambiguous")
    return value


def _sorted_unique_relative(values: object, name: str, *, nonempty: bool = True) -> list[str]:
    _require(isinstance(values, list), f"{name} must be a list")
    result = [_safe_relative(value, f"{name}[{index}]") for index, value in enumerate(values)]
    if nonempty:
        _require(bool(result), f"{name} must not be empty")
    _require(result == sorted(set(result)), f"{name} must be sorted and unique")
    return result


def _validate_variant(
    value: object,
    name: str,
    contract_sha256: str,
    *,
    candidate: bool,
) -> dict[str, Any]:
    common = {
        "id",
        "worktree",
        "source_commit",
        "frozen_execution_contract_sha256",
        "build_bundle_ref",
        "gate_bundle_ref",
    }
    candidate_fields = {
        "declared_patch",
        "primary_change",
        "hypothesis",
        "expected_profile_metrics",
        "risks",
    }
    variant = _exact_fields(value, common | (candidate_fields if candidate else set()), name)
    variant_id = _slug(variant["id"], f"{name}.id")
    _absolute_path(variant["worktree"], f"{name}.worktree")
    _git_object(variant["source_commit"], f"{name}.source_commit")
    _require(
        _sha256_text(
            variant["frozen_execution_contract_sha256"],
            f"{name}.frozen_execution_contract_sha256",
        )
        == contract_sha256,
        f"{name} does not pin the shared frozen execution contract",
    )
    expected_build = f"bundles/{variant_id}/build"
    expected_gate = f"bundles/{variant_id}/gates"
    _require(
        _safe_relative(variant["build_bundle_ref"], f"{name}.build_bundle_ref")
        == expected_build,
        f"{name}.build_bundle_ref must be {expected_build}",
    )
    _require(
        _safe_relative(variant["gate_bundle_ref"], f"{name}.gate_bundle_ref")
        == expected_gate,
        f"{name}.gate_bundle_ref must be {expected_gate}",
    )
    if candidate:
        for field in ("primary_change", "hypothesis"):
            _nonempty_string(variant[field], f"{name}.{field}")
        for field in ("expected_profile_metrics", "risks"):
            values = variant[field]
            _require(isinstance(values, list) and bool(values), f"{name}.{field} must be a nonempty list")
            for index, item in enumerate(values):
                _nonempty_string(item, f"{name}.{field}[{index}]")
        patch = _exact_fields(
            variant["declared_patch"],
            {"base_commit", "diff_sha256", "changed_files"},
            f"{name}.declared_patch",
        )
        _git_object(patch["base_commit"], f"{name}.declared_patch.base_commit")
        _sha256_text(patch["diff_sha256"], f"{name}.declared_patch.diff_sha256")
        _sorted_unique_relative(patch["changed_files"], f"{name}.declared_patch.changed_files")
    return variant


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the semantic schema without touching Git or the filesystem."""

    _validate_json_tree(manifest, "manifest")
    fields = {
        "schema_version",
        "round_id",
        "mode",
        "execution_status",
        "coordinator",
        "frozen_execution_contract",
        "source_policy",
        "parent",
        "candidates",
    }
    _exact_fields(manifest, fields, "manifest")
    _require(manifest["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
    _slug(manifest["round_id"], "round_id")
    _require(manifest["mode"] == "check_only", "mode must be check_only")
    _require(manifest["execution_status"] == "NOT_RUN", "execution_status must be NOT_RUN")

    coordinator = _exact_fields(
        manifest["coordinator"],
        {"control_worktree", "control_commit", "artifact_root", "lock_path"},
        "coordinator",
    )
    _absolute_path(coordinator["control_worktree"], "coordinator.control_worktree")
    _git_object(coordinator["control_commit"], "coordinator.control_commit")
    _absolute_path(coordinator["artifact_root"], "coordinator.artifact_root")
    _require(coordinator["lock_path"] == SOURCE_ROUND_LOCK, f"lock_path must be {SOURCE_ROUND_LOCK}")

    contract = _exact_fields(
        manifest["frozen_execution_contract"],
        {"algorithm", "shape", "timing", "toolchain", "gpu_mapping"},
        "frozen_execution_contract",
    )
    for field in ("algorithm", "shape", "timing", "toolchain"):
        _require(isinstance(contract[field], dict) and bool(contract[field]), f"frozen_execution_contract.{field} must be nonempty")
    mapping = contract["gpu_mapping"]
    _require(isinstance(mapping, list) and len(mapping) == 8, "gpu_mapping must pin exactly eight GPUs")
    indices: list[int] = []
    uuids: list[str] = []
    for ordinal, row in enumerate(mapping):
        row = _exact_fields(row, {"index", "uuid"}, f"gpu_mapping[{ordinal}]")
        _require(type(row["index"]) is int, f"gpu_mapping[{ordinal}].index must be an integer")
        indices.append(row["index"])
        uuid = _nonempty_string(row["uuid"], f"gpu_mapping[{ordinal}].uuid")
        _require(bool(_GPU_UUID.fullmatch(uuid)), f"gpu_mapping[{ordinal}].uuid is invalid")
        uuids.append(uuid)
    _require(indices == list(range(8)), "gpu_mapping indices must be exactly 0..7 in order")
    _require(len(set(uuids)) == 8, "gpu_mapping UUIDs must be unique")
    contract_sha256 = canonical_sha256(contract)

    policy = _exact_fields(
        manifest["source_policy"],
        {"parent_id", "allowed_candidate_files", "harness_sha256"},
        "source_policy",
    )
    parent_id = _slug(policy["parent_id"], "source_policy.parent_id")
    allowed = _sorted_unique_relative(policy["allowed_candidate_files"], "source_policy.allowed_candidate_files")
    harness = policy["harness_sha256"]
    _require(isinstance(harness, dict) and bool(harness), "source_policy.harness_sha256 must be nonempty")
    harness_paths = []
    for raw_path, digest in harness.items():
        path = _safe_relative(raw_path, "source_policy.harness_sha256 key")
        harness_paths.append(path)
        _sha256_text(digest, f"source_policy.harness_sha256[{path}]")
    _require(harness_paths == sorted(harness_paths), "harness_sha256 keys must be sorted")
    missing_harness = set(REQUIRED_HARNESS_PATHS) - set(harness_paths)
    _require(
        not missing_harness,
        "harness_sha256 omits required experiment infrastructure: "
        f"{sorted(missing_harness)}",
    )
    _require(not set(allowed) & set(harness_paths), "candidate files cannot include frozen harness files")

    parent = _validate_variant(manifest["parent"], "parent", contract_sha256, candidate=False)
    _require(parent["id"] == parent_id, "parent.id must match source_policy.parent_id")
    candidates = manifest["candidates"]
    _require(isinstance(candidates, list) and 2 <= len(candidates) <= 4, "a round requires two to four candidates")
    validated = [
        _validate_variant(candidate, f"candidates[{index}]", contract_sha256, candidate=True)
        for index, candidate in enumerate(candidates)
    ]
    candidate_ids = [candidate["id"] for candidate in validated]
    _require(candidate_ids == sorted(set(candidate_ids)), "candidate IDs must be sorted, unique, and deterministic")
    _require(parent_id not in candidate_ids, "candidate ID collides with parent ID")
    commits = [parent["source_commit"], *(candidate["source_commit"] for candidate in validated)]
    _require(len(set(commits)) == len(commits), "parent and candidate source commits must be unique")
    bundle_refs = []
    for variant in (parent, *validated):
        bundle_refs.extend((variant["build_bundle_ref"], variant["gate_bundle_ref"]))
    _require(len(set(bundle_refs)) == len(bundle_refs), "build and gate bundle refs must be unique between source variants")
    for index, candidate in enumerate(validated):
        patch = candidate["declared_patch"]
        _require(patch["base_commit"] == parent["source_commit"], f"candidates[{index}] patch base is not the parent commit")
        _require(set(patch["changed_files"]) <= set(allowed), f"candidates[{index}] changes a file outside allowed_candidate_files")


def _assert_existing_path_has_no_symlink(path: Path, name: str) -> os.stat_result:
    path = _absolute_without_resolving(path)
    cursor = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            cursor /= part
            metadata = os.lstat(cursor)
            _require(not stat.S_ISLNK(metadata.st_mode), f"{name} contains a symlink: {cursor}")
        return os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise SourceRoundError(f"cannot inspect {name} {path}: {error}") from error


def _existing_directory(path: Path, name: str) -> tuple[Path, os.stat_result]:
    metadata = _assert_existing_path_has_no_symlink(path, name)
    _require(stat.S_ISDIR(metadata.st_mode), f"{name} is not a directory: {path}")
    resolved = path.resolve(strict=True)
    _require(resolved == path, f"{name} is not a canonical realpath: {path}")
    return resolved, metadata


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _check_planned_path(root: Path, relative: str, name: str) -> Path:
    path = root / relative
    _require(_is_below(path, root) and path != root, f"{name} escapes artifact_root")
    cursor = root
    for part in Path(relative).parts:
        cursor /= part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            break
        except OSError as error:
            raise SourceRoundError(f"cannot inspect {name} {cursor}: {error}") from error
        _require(not stat.S_ISLNK(metadata.st_mode), f"{name} contains a symlink: {cursor}")
        _require(stat.S_ISDIR(metadata.st_mode), f"{name} has a non-directory prefix: {cursor}")
    return path


def _git(worktree: Path, *arguments: str, text: bool = True) -> str | bytes:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    command = (
        PINNED_GIT,
        "--no-pager",
        "-c",
        "core.fsmonitor=false",
        "-C",
        str(worktree),
        *arguments,
    )
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            env=environment,
            check=False,
        )
    except OSError as error:
        raise SourceRoundError(f"cannot execute Git preflight: {error}") from error
    if result.returncode != 0:
        stderr = result.stderr if text else result.stderr.decode("utf-8", "replace")
        raise SourceRoundError(f"Git preflight failed ({' '.join(command)}): {stderr.strip()}")
    return result.stdout


def _git_is_ancestor(worktree: Path, parent: str, candidate: str) -> bool:
    result = subprocess.run(
        (
            PINNED_GIT,
            "--no-pager",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(worktree),
            "merge-base",
            "--is-ancestor",
            parent,
            candidate,
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise SourceRoundError(
            "Git ancestry preflight failed: "
            + result.stderr.decode("utf-8", "replace").strip()
        )
    return result.returncode == 0


def _git_tool_identity() -> dict[str, Any]:
    path = Path(PINNED_GIT)
    metadata = _assert_existing_path_has_no_symlink(path, "pinned Git")
    _require(stat.S_ISREG(metadata.st_mode), "pinned Git is not a regular file")
    _require(metadata.st_mode & 0o111 != 0, "pinned Git is not executable")
    _require(metadata.st_uid == 0, "pinned Git must be owned by root")
    _require(metadata.st_mode & 0o022 == 0, "pinned Git cannot be group/other writable")
    payload = _read_regular_no_follow(path, 32 * 1024 * 1024)
    try:
        version = subprocess.check_output(
            (PINNED_GIT, "--version"),
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SourceRoundError(f"cannot identify pinned Git: {error}") from error
    _require(version.startswith("git version "), "pinned Git returned an invalid version")
    return {
        "path": PINNED_GIT,
        "realpath": str(path.resolve(strict=True)),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "version": version,
    }


def _git_diff(worktree: Path, parent: str, candidate: str) -> tuple[list[str], str]:
    name_payload = _git(
        worktree,
        "diff",
        "--name-only",
        "-z",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        parent,
        candidate,
        "--",
        text=False,
    )
    assert isinstance(name_payload, bytes)
    try:
        names = [item.decode("utf-8") for item in name_payload.split(b"\0") if item]
    except UnicodeDecodeError as error:
        raise SourceRoundError("candidate diff contains a non-UTF-8 path") from error
    for index, path in enumerate(names):
        _safe_relative(path, f"candidate Git diff path[{index}]")
    _require(names == sorted(set(names)), "Git diff paths are not sorted and unique")
    diff_payload = _git(
        worktree,
        "diff",
        "--binary",
        "--full-index",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        parent,
        candidate,
        "--",
        text=False,
    )
    assert isinstance(diff_payload, bytes)
    return names, hashlib.sha256(diff_payload).hexdigest()


def declared_diff_sha256(worktree: Path, parent: str, candidate: str) -> str:
    """Helper for authoring a pinned candidate manifest."""

    return _git_diff(worktree, parent, candidate)[1]


def _hash_harness(worktree: Path, relative: str) -> str:
    path = worktree / relative
    _require(_is_below(path, worktree), f"harness path escapes worktree: {relative}")
    _assert_existing_path_has_no_symlink(path, f"harness {relative}")
    return hashlib.sha256(_read_regular_no_follow(path, _MAX_HASHED_FILE_BYTES)).hexdigest()


def _inspect_worktree(path: Path, declared_commit: str, name: str) -> dict[str, Any]:
    realpath, metadata = _existing_directory(path, name)
    top = Path(str(_git(realpath, "rev-parse", "--show-toplevel")).strip()).resolve(strict=True)
    _require(top == realpath, f"{name} is not the Git worktree top level")
    commit = str(_git(realpath, "rev-parse", "--verify", "HEAD^{commit}")).strip()
    _require(commit == declared_commit, f"{name} HEAD does not match declared source_commit")
    tree = str(_git(realpath, "rev-parse", "--verify", "HEAD^{tree}")).strip()
    common_raw = Path(str(_git(realpath, "rev-parse", "--git-common-dir")).strip())
    common = common_raw if common_raw.is_absolute() else realpath / common_raw
    common = _absolute_without_resolving(common)
    common_realpath, common_metadata = _existing_directory(common, f"{name} git common-dir")
    status_payload = _git(realpath, "status", "--porcelain=v1", "-z", "--untracked-files=all", text=False)
    assert isinstance(status_payload, bytes)
    _require(not status_payload, f"{name} must be clean, including untracked files")
    return {
        "worktree": str(realpath),
        "worktree_device": metadata.st_dev,
        "worktree_inode": metadata.st_ino,
        "source_commit": commit,
        "source_tree": tree,
        "git_common_dir": str(common_realpath),
        "git_common_device": common_metadata.st_dev,
        "git_common_inode": common_metadata.st_ino,
    }


def preflight(manifest: dict[str, Any], *, current_directory: Path | None = None) -> dict[str, Any]:
    """Inspect worktrees and patches without launching any experiment process."""

    validate_manifest(manifest)
    git_tool_pre = _git_tool_identity()
    coordinator = manifest["coordinator"]
    control_path = Path(coordinator["control_worktree"])
    current = _absolute_without_resolving(current_directory or Path.cwd())
    current_realpath, _ = _existing_directory(current, "current_directory")
    _require(current_realpath == control_path, "check-only coordinator must run from the neutral control checkout")
    control = _inspect_worktree(control_path, coordinator["control_commit"], "control_worktree")
    parent_manifest = manifest["parent"]
    parent = _inspect_worktree(Path(parent_manifest["worktree"]), parent_manifest["source_commit"], "parent.worktree")
    candidates: list[dict[str, Any]] = []
    for index, candidate_manifest in enumerate(manifest["candidates"]):
        candidates.append(
            _inspect_worktree(
                Path(candidate_manifest["worktree"]),
                candidate_manifest["source_commit"],
                f"candidates[{index}].worktree",
            )
        )

    identities = [control, parent, *candidates]
    common_keys = {
        (identity["git_common_dir"], identity["git_common_device"], identity["git_common_inode"])
        for identity in identities
    }
    _require(len(common_keys) == 1, "all control, parent, and candidate checkouts must share one Git common-dir identity")
    worktree_paths = {identity["worktree"] for identity in identities}
    worktree_inodes = {
        (identity["worktree_device"], identity["worktree_inode"])
        for identity in identities
    }
    _require(
        len(worktree_paths) == len(identities)
        and len(worktree_inodes) == len(identities),
        "control, parent, and candidate worktrees must have unique realpath/dev/inode identities",
    )
    source_commits = {identity["source_commit"] for identity in (parent, *candidates)}
    source_trees = {identity["source_tree"] for identity in (parent, *candidates)}
    _require(
        len(source_commits) == 1 + len(candidates)
        and len(source_trees) == 1 + len(candidates),
        "parent and candidates must have independently unique commit and tree source identities",
    )
    _require(
        control["source_commit"] not in {
            candidate["source_commit"] for candidate in candidates
        },
        "neutral control checkout cannot point at a candidate source commit",
    )

    artifact_root, _ = _existing_directory(Path(coordinator["artifact_root"]), "artifact_root")
    for identity in identities:
        worktree = Path(identity["worktree"])
        _require(
            not _is_below(artifact_root, worktree) and not _is_below(worktree, artifact_root),
            "artifact_root must be disjoint from every Git worktree",
        )
    planned_bundle_paths: list[Path] = []
    for variant in (parent_manifest, *manifest["candidates"]):
        for field in ("build_bundle_ref", "gate_bundle_ref"):
            planned_bundle_paths.append(
                _check_planned_path(artifact_root, variant[field], f"{variant['id']}.{field}")
            )
    _require(len(set(planned_bundle_paths)) == len(planned_bundle_paths), "planned bundle paths are not unique")

    harness_expected = manifest["source_policy"]["harness_sha256"]
    harness_observed: dict[str, dict[str, str]] = {}
    for identity in identities:
        observed = {
            path: _hash_harness(Path(identity["worktree"]), path)
            for path in harness_expected
        }
        _require(observed == harness_expected, f"frozen harness hashes differ in {identity['worktree']}")
        harness_observed[identity["worktree"]] = observed

    parent_commit = parent["source_commit"]
    allowed = set(manifest["source_policy"]["allowed_candidate_files"])
    patch_checks = []
    for candidate_manifest, candidate in zip(manifest["candidates"], candidates):
        candidate_commit = candidate["source_commit"]
        _require(
            _git_is_ancestor(Path(parent["worktree"]), parent_commit, candidate_commit),
            f"parent is not an ancestor of candidate {candidate_manifest['id']}",
        )
        distance_text = str(
            _git(
                Path(parent["worktree"]),
                "rev-list",
                "--count",
                f"{parent_commit}..{candidate_commit}",
            )
        ).strip()
        _require(
            distance_text == "1",
            f"candidate {candidate_manifest['id']} must be one independent commit above parent",
        )
        parent_row = str(
            _git(
                Path(parent["worktree"]),
                "rev-list",
                "--parents",
                "-n",
                "1",
                candidate_commit,
            )
        ).strip().split()
        _require(
            parent_row == [candidate_commit, parent_commit],
            f"candidate {candidate_manifest['id']} must have the shared parent as its only direct parent",
        )
        changed_files, diff_sha256 = _git_diff(Path(parent["worktree"]), parent_commit, candidate_commit)
        declared = candidate_manifest["declared_patch"]
        _require(changed_files == declared["changed_files"], f"candidate {candidate_manifest['id']} changed-file declaration is stale")
        _require(diff_sha256 == declared["diff_sha256"], f"candidate {candidate_manifest['id']} diff hash is stale")
        _require(set(changed_files) <= allowed, f"candidate {candidate_manifest['id']} changes a forbidden file")
        patch_checks.append(
            {
                "candidate_id": candidate_manifest["id"],
                "base_commit": parent_commit,
                "candidate_commit": candidate_commit,
                "commit_distance": 1,
                "direct_single_parent": True,
                "changed_files": changed_files,
                "diff_sha256": diff_sha256,
            }
        )

    git_tool_post = _git_tool_identity()
    _require(git_tool_post == git_tool_pre, "pinned Git identity changed during preflight")
    return {
        "status": "passed",
        "scope": "git_and_plan_only_no_cuda_execution",
        "control": control,
        "parent": parent,
        "candidates": [
            {"id": manifest_candidate["id"], **identity}
            for manifest_candidate, identity in zip(manifest["candidates"], candidates)
        ],
        "git_common_identity": {
            "realpath": control["git_common_dir"],
            "device": control["git_common_device"],
            "inode": control["git_common_inode"],
        },
        "coordinator_git": git_tool_pre,
        "artifact_root": str(artifact_root),
        "harness_sha256": harness_expected,
        "harness_checkout_count": len(harness_observed),
        "patches": patch_checks,
    }


def _schedule_roles(candidate_ids: list[str]) -> list[tuple[str, str, int]]:
    schedule: list[tuple[str, str, int]] = [("parent", "parent", 0)]
    schedule.extend(("candidate", candidate_id, 0) for candidate_id in candidate_ids)
    schedule.extend(("candidate", candidate_id, 1) for candidate_id in candidate_ids)
    schedule.append(("parent", "parent", 1))
    schedule.extend(("candidate", candidate_id, 2) for candidate_id in candidate_ids)
    schedule.extend(("parent", "parent", repetition) for repetition in (2, 3))
    schedule.extend(("candidate", candidate_id, 3) for candidate_id in candidate_ids)
    return schedule


def materialize_plan(
    manifest: dict[str, Any],
    manifest_raw_sha256: str,
    manifest_canonical_sha256: str,
    preflight_result: dict[str, Any],
) -> dict[str, Any]:
    """Create a deterministic NOT_RUN schedule from a passed preflight."""

    validate_manifest(manifest)
    _sha256_text(manifest_raw_sha256, "manifest_raw_sha256")
    _sha256_text(manifest_canonical_sha256, "manifest_canonical_sha256")
    _require(preflight_result.get("status") == "passed", "plan requires a passed preflight")
    contract_sha256 = canonical_sha256(manifest["frozen_execution_contract"])
    variants = {
        manifest["parent"]["id"]: manifest["parent"],
        **{candidate["id"]: candidate for candidate in manifest["candidates"]},
    }
    candidate_ids = [candidate["id"] for candidate in manifest["candidates"]]
    blocks = []
    for ordinal, (role, schedule_id, repetition) in enumerate(_schedule_roles(candidate_ids)):
        variant_id = manifest["parent"]["id"] if role == "parent" else schedule_id
        variant = variants[variant_id]
        run_id = f"{manifest['round_id']}-{ordinal:02d}-{variant_id}-{repetition}"
        _slug(run_id, f"generated run_id[{ordinal}]")
        artifact_dir = f"runs/{run_id}"
        attempt_dir = f"{artifact_dir}/stages/bench-abba-a1/attempt-01"
        blocks.append(
            {
                "ordinal": ordinal,
                "role": role,
                "variant_id": variant_id,
                "repetition": repetition,
                "run_id": run_id,
                "worktree": variant["worktree"],
                "source_commit": variant["source_commit"],
                "build_bundle_ref": variant["build_bundle_ref"],
                "gate_bundle_ref": variant["gate_bundle_ref"],
                "frozen_execution_contract_sha256": contract_sha256,
                "artifact_dir": artifact_dir,
                "raw_benchmark_artifact": f"{attempt_dir}/report.json",
                "jit_dir": f"{attempt_dir}/jit",
                "execution_status": "NOT_RUN",
                "live_execution_required": True,
                "raw_benchmark_reuse_allowed": False,
                "jit_reuse_allowed": False,
            }
        )
    run_ids = [block["run_id"] for block in blocks]
    artifact_dirs = [block["artifact_dir"] for block in blocks]
    raw_paths = [block["raw_benchmark_artifact"] for block in blocks]
    jit_paths = [block["jit_dir"] for block in blocks]
    _require(len(set(run_ids)) == len(blocks), "generated run IDs are not unique")
    _require(len(set(artifact_dirs)) == len(blocks), "artifact directories cannot be shared")
    _require(len(set(raw_paths)) == len(blocks), "raw benchmark artifacts cannot be shared")
    _require(len(set(jit_paths)) == len(blocks), "JIT directories cannot be shared")
    pairwise_orders = {}
    expected = ["parent", "candidate", "candidate", "parent", "candidate", "parent", "parent", "candidate"]
    for candidate_id in candidate_ids:
        order = [
            block["role"]
            for block in blocks
            if block["role"] == "parent" or block["variant_id"] == candidate_id
        ]
        _require(order == expected, f"generated pairwise schedule is not ABBA+BAAB for {candidate_id}")
        pairwise_orders[candidate_id] = order

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "round_id": manifest["round_id"],
        "mode": "check_only_plan",
        "execution_status": "NOT_RUN",
        "claim_scope": "source_round_preflight_and_schedule_only",
        "promotion": {
            "eligible": False,
            "candidate_id": None,
            "reason": "no benchmark block has been executed live",
        },
        "manifest_raw_sha256": manifest_raw_sha256,
        "manifest_canonical_sha256": manifest_canonical_sha256,
        "frozen_execution_contract_sha256": contract_sha256,
        "lock_contract": {
            "path": SOURCE_ROUND_LOCK,
            "holder": "neutral_control_checkout",
            "required_scope": "entire_round_through_all_descendant_process_exit_and_final_evidence_fsync",
        },
        "reuse_contract": {
            "parent_build_bundle_may_be_shared_across_parent_blocks": True,
            "parent_gate_bundle_may_be_shared_across_parent_blocks": True,
            "raw_benchmark_artifacts_may_be_shared": False,
            "jit_directories_may_be_shared": False,
            "bundle_refs_status": "PLANNED_NOT_VALIDATED_AS_BUILD_OUTPUT",
        },
        "sealed_parent_contract": (
            "the parent must be a clean sealed commit containing the exact frozen "
            "experiment infrastructure; an older CUDA baseline must be reconstructed "
            "on that infrastructure instead of weakening harness hashes"
        ),
        "preflight": preflight_result,
        "schedule_formula": "P0,C1_0..Cn_0,C1_1..Cn_1,P1,C1_2..Cn_2,P2,P3,C1_3..Cn_3",
        "pairwise_orders": pairwise_orders,
        "blocks": blocks,
        "next_required_action": "execute every block live in order under one lock, then aggregate immutable raw evidence",
    }


def _safe_output_path(path: Path, artifact_root: Path) -> Path:
    path = _absolute_without_resolving(path)
    _require(_is_below(path, artifact_root) and path != artifact_root, "output must be below artifact_root")
    parent, _ = _existing_directory(path.parent, "output parent")
    _require(_is_below(parent, artifact_root), "output parent escapes artifact_root")
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise SourceRoundError(f"cannot inspect output {path}: {error}") from error
    else:
        _require(not stat.S_ISLNK(metadata.st_mode), "output cannot be a symlink")
        raise SourceRoundError("output already exists; check-only plans are immutable")
    return path


def _atomic_json_no_clobber(path: Path, value: object) -> None:
    payload = _canonical_bytes(value) + b"\n"
    parent_descriptor = _open_absolute_directory_no_follow(path.parent)
    temporary_name = f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise SourceRoundError("output appeared concurrently; refusing to clobber it") from error
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)


def _open_absolute_directory_no_follow(path: Path) -> int:
    """Open every path component with O_NOFOLLOW and return the final fd."""

    path = _absolute_without_resolving(path)
    _require(path.is_absolute(), "directory path must be absolute")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise SourceRoundError(
            f"cannot open output directory without following symlinks: {path}: {error}"
        ) from error


@contextmanager
def _global_lock(path: str) -> Iterator[None]:
    _require(path == SOURCE_ROUND_LOCK, "unexpected source-round lock path")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), "source-round lock is not a regular file")
        _require(metadata.st_uid == os.geteuid(), "source-round lock has a foreign owner")
        _require(metadata.st_nlink == 1, "source-round lock must not be hard-linked")
        _require(stat.S_IMODE(metadata.st_mode) == 0o600, "source-round lock permissions must be 0600")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SourceRoundError("another source-version round holds the global lock") from error
        yield
    except OSError as error:
        raise SourceRoundError(f"cannot acquire source-round lock: {error}") from error
    finally:
        if descriptor is not None:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def build_check_only_plan(
    manifest_path: Path,
    *,
    current_directory: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load, preflight, and plan; return the manifest and immutable plan."""

    manifest, raw_sha256, canonical_manifest_sha256 = load_manifest(manifest_path)
    with _global_lock(manifest["coordinator"]["lock_path"]):
        checked = preflight(manifest, current_directory=current_directory)
        plan = materialize_plan(manifest, raw_sha256, canonical_manifest_sha256, checked)
    return manifest, plan


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest, raw_sha256, canonical_manifest_sha256 = load_manifest(args.manifest)
    with _global_lock(manifest["coordinator"]["lock_path"]):
        checked = preflight(manifest)
        plan = materialize_plan(
            manifest,
            raw_sha256,
            canonical_manifest_sha256,
            checked,
        )
        artifact_root, _ = _existing_directory(
            Path(manifest["coordinator"]["artifact_root"]),
            "artifact_root",
        )
        output = _safe_output_path(args.output, artifact_root)
        _atomic_json_no_clobber(output, plan)
    print(
        "NOT_RUN source-version round plan: "
        f"preflight=passed blocks={len(plan['blocks'])} promotion_eligible=false -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except SourceRoundError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error
