#!/usr/bin/env python3
"""Fail-closed CPU evaluator for one formal source-version CUDA round.

The evaluator never imports Torch/DeepEP and never launches a child process.
It consumes two to four candidates measured against one sealed parent build.
The round has four shared live parent blocks and four live blocks per candidate,
scheduled globally as P0,C1_0..Cn_0,C1_1..Cn_1,P1,C1_2..Cn_2,P2,P3,
C1_3..Cn_3.  Projecting that schedule onto any candidate gives the exact
P/C/C/P/C/P/P/C order.  A single-binary algorithm-mode A/B is deliberately not
a source-leader round.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import time
from contextlib import suppress
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rail_balance_campaign_schema import (
    ManifestError as CampaignManifestError,
    validate_manifest as validate_campaign_manifest,
)
from rail_balance_source_round_coordinator import (
    SourceRoundError,
    validate_manifest as validate_source_round_manifest,
)


ROUND_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 64 * 1024 * 1024
_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40,64}\Z")
_ABBA_BAAB = (
    "parent",
    "candidate",
    "candidate",
    "parent",
    "candidate",
    "parent",
    "parent",
    "candidate",
)
_WORKLOAD_FIELDS = {
    "stage",
    "case_name",
    "world_size",
    "hidden",
    "input_iteration",
    "init_dist_seed",
    "remainder_seed",
    "tokens_per_rank",
    "num_topk",
    "num_channels",
    "num_experts",
    "num_destinations",
    "local_destination",
    "proxy_capacity_per_egress",
    "dtype",
    "interference_mode",
    "interference_compute_shape",
}
_ALGORITHM_FIELDS = {
    "hop_mode",
    "two_hop_threshold_percent",
    "max_two_hop_percent",
    "hop_penalty_percent",
}
_TIMER_FIELDS = {
    "clock",
    "stage_truth",
    "percentile_method",
    "std_method",
    "logical_token_bytes",
    "logical_bytes_per_iteration",
    "logical_bandwidth_denominator",
}
_CAMPAIGN_CANDIDATE_FIELDS = {
    "id",
    "parent",
    "primary_change",
    "hypothesis",
    "expected_profile_metrics",
    "risks",
}
_SOURCE_REF_FIELDS = {
    "ref_id",
    "commit",
    "working_tree_identity_sha256",
    "worktree_realpath",
    "worktree_device",
    "worktree_inode",
    "build_bundle_sha256",
    "code_identity_sha256",
}
_BLOCK_FIELDS = {
    "ordinal",
    "global_ordinal",
    "role",
    "artifact_root_realpath",
    "artifact_root_device",
    "artifact_root_inode",
    "campaign_result_raw_sha256",
    "campaign_manifest_raw_sha256",
    "finalized_raw_sha256",
    "coordinator_record",
    "coordinator_record_raw_sha256",
    "stage_id",
    "attempt",
    "artifact",
}
_CANDIDATE_FIELDS = _CAMPAIGN_CANDIDATE_FIELDS | {
    "frozen_contract_sha256",
    "source_ref",
    "benchmark_blocks",
}
_SOURCE_ROUND_BINDING_FIELDS = {
    "manifest_path",
    "manifest_raw_sha256",
    "manifest_canonical_sha256",
    "plan_path",
    "plan_raw_sha256",
    "plan_canonical_sha256",
    "coordinator_code_sha256",
    "source_round_executor_sha256",
    "control_commit",
    "parent_commit",
}
_FINALIZED_FIELDS = {
    "schema_version",
    "created_utc",
    "campaign_id",
    "run_id",
    "campaign_status",
    "claim_scope",
    "terminal_label",
    "terminal_exit_code",
    "round_evaluation_allowed",
    "result_sha256",
    "sha256sums_sha256",
    "source_identity_sha256",
    "manifest_raw_sha256",
    "leaderboard_entry_id",
}
_COORDINATOR_FIELDS = {
    "schema_version",
    "round_id",
    "variant_id",
    "source_round_manifest_raw_sha256",
    "source_round_plan_raw_sha256",
    "source_round_plan_ordinal",
    "source_round_run_id",
    "source_round_coordinator_sha256",
    "source_round_executor_sha256",
    "source_round_control_commit",
    "source_round_parent_commit",
    "ordinal",
    "global_ordinal",
    "role",
    "boot_id",
    "lock_identity_sha256",
    "monotonic_start_ns",
    "monotonic_end_ns",
    "artifact_root_realpath",
    "artifact_root_device",
    "artifact_root_inode",
    "worktree_realpath",
    "worktree_device",
    "worktree_inode",
    "source_commit",
    "source_identity_sha256",
    "campaign_result_raw_sha256",
    "campaign_finalized_raw_sha256",
    "benchmark_report_sha256",
    "gpu_mapping_sha256",
    "terminal_evidence_complete",
}

# This is one publication rule, not a post-hoc knob.  The manifest must carry
# this exact object so its hash is retained with the evidence.
FORMAL_NOISE_POLICY: dict[str, int | float] = {
    "relative_floor": 0.005,
    "minimum_improvement_fraction": 0.01,
    "mad_multiplier": 2.0,
    "minimum_winning_pairs": 3,
    "maximum_pooled_cv": 0.10,
    "maximum_parent_baseline_drift_fraction": 0.02,
}


class RoundError(ValueError):
    """The round contract itself is ambiguous or unsafe."""


class CandidateEvidenceError(RoundError):
    """One candidate is invalid without invalidating its siblings."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RoundError(message)


def _candidate_require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateEvidenceError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RoundError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise RoundError(f"non-finite JSON constant is forbidden: {value}")


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise RoundError(f"value is not strict canonical JSON: {error}") from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalized_toolchain(value: object) -> dict[str, Any]:
    """Drop only the runner's per-snapshot timestamp from toolchain identity."""

    _require(isinstance(value, dict), "toolchain snapshot is not an object")
    normalized = deepcopy(value)
    created = normalized.pop("created_utc", None)
    if created is not None:
        _require(
            isinstance(created, str) and bool(created),
            "toolchain created_utc is invalid",
        )
    _require(bool(normalized), "normalized toolchain snapshot is empty")
    return normalized


def _file_snapshot(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _sha256_file_snapshot(path: Path) -> tuple[str, tuple[int, ...]]:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and not before.st_mode & (stat.S_IWGRP | stat.S_IWOTH),
            f"code artifact is linked, writable, or non-regular: {path}",
        )
        for chunk in iter(lambda: os.read(descriptor, 1 << 20), b""):
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        snapshot = _file_snapshot(before)
        _require(
            snapshot == _file_snapshot(after) == _file_snapshot(current),
            f"code artifact changed while hashing: {path}",
        )
    finally:
        os.close(descriptor)
    return digest.hexdigest(), snapshot


def _sha256_file(path: Path) -> str:
    return _sha256_file_snapshot(path)[0]


def evaluator_sha256() -> str:
    return _sha256_file(Path(__file__))


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _decode_json(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise RoundError(f"invalid JSON artifact {name}: {error}") from error
    if not isinstance(value, dict):
        raise RoundError(f"JSON artifact root must be an object: {name}")
    return value


def _read_descriptor(descriptor: int, name: str) -> tuple[dict[str, Any], str, int]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RoundError(f"JSON artifact is not a regular file: {name}")
    if before.st_nlink != 1 or before.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise RoundError(f"JSON artifact is linked or group/world-writable: {name}")
    if before.st_size > _MAX_JSON_BYTES:
        raise RoundError(f"JSON artifact exceeds {_MAX_JSON_BYTES} bytes: {name}")
    chunks: list[bytes] = []
    remaining = _MAX_JSON_BYTES + 1
    while remaining:
        chunk = os.read(descriptor, min(1 << 20, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > _MAX_JSON_BYTES:
        raise RoundError(f"JSON artifact exceeds {_MAX_JSON_BYTES} bytes: {name}")
    after = os.fstat(descriptor)
    if _file_snapshot(before) != _file_snapshot(after):
        raise RoundError(f"JSON artifact changed while reading: {name}")
    return _decode_json(raw, name), hashlib.sha256(raw).hexdigest(), len(raw)


def _read_strict_json(path: Path) -> tuple[dict[str, Any], str]:
    path = _absolute_without_resolving(path)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        value, digest, _ = _read_descriptor(descriptor, str(path))
        return value, digest
    except OSError as error:
        raise RoundError(f"cannot read JSON artifact {path}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_dir_beneath(
    root: Path,
    parts: Sequence[str],
    *,
    root_identity: tuple[int, int] | None = None,
) -> int:
    descriptor = os.open(
        root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        if root_identity is not None:
            metadata = os.fstat(descriptor)
            _candidate_require(
                (metadata.st_dev, metadata.st_ino) == root_identity,
                "artifact root changed while opening a finalized artifact",
            )
        for part in parts:
            child = os.open(
                part,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_json_beneath(
    root: Path,
    relative: str,
    *,
    root_identity: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], str, int]:
    path = Path(relative)
    _candidate_require(
        not path.is_absolute() and bool(path.parts) and ".." not in path.parts,
        f"artifact path is unsafe: {relative}",
    )
    directory: int | None = None
    descriptor: int | None = None
    try:
        directory = _open_dir_beneath(
            root,
            path.parts[:-1],
            root_identity=root_identity,
        )
        descriptor = os.open(
            path.parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory,
        )
        value, digest, size = _read_descriptor(descriptor, f"{root}/{relative}")
        return value, digest, size
    except OSError as error:
        raise CandidateEvidenceError(
            f"cannot safely read campaign artifact {root}/{relative}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)


def _read_bytes_beneath(
    root: Path,
    relative: str,
    *,
    root_identity: tuple[int, int] | None = None,
) -> tuple[bytes, str]:
    path = Path(relative)
    _candidate_require(
        not path.is_absolute() and bool(path.parts) and ".." not in path.parts,
        f"artifact path is unsafe: {relative}",
    )
    directory: int | None = None
    descriptor: int | None = None
    try:
        directory = _open_dir_beneath(
            root,
            path.parts[:-1],
            root_identity=root_identity,
        )
        descriptor = os.open(
            path.parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory,
        )
        before = os.fstat(descriptor)
        _candidate_require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and not before.st_mode & (stat.S_IWGRP | stat.S_IWOTH),
            "artifact is linked, writable, or non-regular",
        )
        _candidate_require(before.st_size <= _MAX_JSON_BYTES, "artifact is too large")
        raw = b""
        while len(raw) <= _MAX_JSON_BYTES:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            raw += chunk
        _candidate_require(len(raw) <= _MAX_JSON_BYTES, "artifact is too large")
        after = os.fstat(descriptor)
        _candidate_require(
            _file_snapshot(before) == _file_snapshot(after),
            "artifact changed while reading",
        )
        return raw, hashlib.sha256(raw).hexdigest()
    except OSError as error:
        raise CandidateEvidenceError(
            f"cannot safely read campaign artifact {root}/{relative}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)


def _atomic_json(path: Path, value: object, *, overwrite: bool = True) -> None:
    path = _absolute_without_resolving(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(_canonical_bytes(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _nonempty(value: object, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} is empty")
    _require("\x00" not in value, f"{name} contains NUL")
    return value


def _slug(value: object, name: str) -> str:
    value = _nonempty(value, name)
    _require(bool(_SLUG.fullmatch(value)), f"{name} is not a safe slug")
    return value


def _sha(value: object, name: str) -> str:
    value = _nonempty(value, name)
    _require(bool(_SHA256.fullmatch(value)), f"{name} is not a lowercase SHA256")
    return value


def _commit(value: object, name: str) -> str:
    value = _nonempty(value, name)
    _require(bool(_COMMIT.fullmatch(value)), f"{name} is not a Git commit")
    return value


def _positive_int(value: object, name: str) -> int:
    _require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


def _safe_relative(value: object, name: str) -> str:
    value = _nonempty(value, name)
    _require(
        "\\" not in value
        and all(ord(character) >= 32 and ord(character) != 127 for character in value),
        f"{name} contains an unsafe path character",
    )
    path = Path(value)
    _require(
        not path.is_absolute()
        and value == path.as_posix()
        and bool(path.parts)
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{name} is unsafe",
    )
    return value


def _validate_campaign_candidate(value: object, name: str) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == _CAMPAIGN_CANDIDATE_FIELDS,
        f"{name} fields are incomplete or unknown",
    )
    _slug(value["id"], f"{name}.id")
    _slug(value["parent"], f"{name}.parent")
    for field in ("primary_change", "hypothesis"):
        _nonempty(value[field], f"{name}.{field}")
    for field in ("expected_profile_metrics", "risks"):
        _require(isinstance(value[field], list) and value[field], f"{name}.{field} is empty")
        for index, item in enumerate(value[field]):
            _nonempty(item, f"{name}.{field}[{index}]")
    return value


def _validate_source_ref(value: object, name: str) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == _SOURCE_REF_FIELDS,
        f"{name} fields are incomplete or unknown",
    )
    _slug(value["ref_id"], f"{name}.ref_id")
    _commit(value["commit"], f"{name}.commit")
    for field in (
        "working_tree_identity_sha256",
        "build_bundle_sha256",
        "code_identity_sha256",
    ):
        _sha(value[field], f"{name}.{field}")
    realpath = Path(_nonempty(value["worktree_realpath"], f"{name}.worktree_realpath"))
    _require(realpath.is_absolute(), f"{name}.worktree_realpath must be absolute")
    for field in ("worktree_device", "worktree_inode"):
        _positive_int(value[field], f"{name}.{field}")
    return value


def _load_source_round_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    binding = manifest["source_round_binding"]
    _require(
        isinstance(binding, dict) and set(binding) == _SOURCE_ROUND_BINDING_FIELDS,
        "source_round_binding fields are incomplete or unknown",
    )
    paths: dict[str, Path] = {}
    for field in ("manifest_path", "plan_path"):
        path = Path(_nonempty(binding[field], f"source_round_binding.{field}"))
        _require(path.is_absolute(), f"source_round_binding.{field} must be absolute")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise RoundError(f"source-round {field} is unavailable: {error}") from error
        _require(resolved == path, f"source_round_binding.{field} is not a canonical realpath")
        paths[field] = path
    for field in (
        "manifest_raw_sha256",
        "manifest_canonical_sha256",
        "plan_raw_sha256",
        "plan_canonical_sha256",
        "coordinator_code_sha256",
        "source_round_executor_sha256",
    ):
        _sha(binding[field], f"source_round_binding.{field}")
    for field in ("control_commit", "parent_commit"):
        _commit(binding[field], f"source_round_binding.{field}")

    source_manifest, source_manifest_raw = _read_strict_json(paths["manifest_path"])
    _require(
        source_manifest_raw == binding["manifest_raw_sha256"],
        "source-round manifest raw hash differs",
    )
    _require(
        canonical_sha256(source_manifest) == binding["manifest_canonical_sha256"],
        "source-round manifest canonical hash differs",
    )
    try:
        validate_source_round_manifest(source_manifest)
    except SourceRoundError as error:
        raise RoundError(f"source-round manifest is invalid: {error}") from error

    plan, plan_raw = _read_strict_json(paths["plan_path"])
    _require(plan_raw == binding["plan_raw_sha256"], "source-round plan raw hash differs")
    _require(
        canonical_sha256(plan) == binding["plan_canonical_sha256"],
        "source-round plan canonical hash differs",
    )
    evaluator_path = Path(__file__)
    evaluator_sha, evaluator_snapshot = _sha256_file_snapshot(evaluator_path)
    coordinator_path = evaluator_path.with_name(
        "rail_balance_source_round_coordinator.py"
    )
    coordinator_sha, coordinator_snapshot = _sha256_file_snapshot(coordinator_path)
    _require(
        coordinator_sha == binding["coordinator_code_sha256"],
        "source-round coordinator code hash differs",
    )
    executor_path = evaluator_path.with_name("rail_balance_source_round_executor.py")
    executor_sha, executor_snapshot = _sha256_file_snapshot(executor_path)
    _require(
        executor_sha == binding["source_round_executor_sha256"],
        "source-round executor code hash differs",
    )

    round_candidates = manifest["candidates"]
    source_candidates = source_manifest["candidates"]
    round_candidate_ids = [candidate["id"] for candidate in round_candidates]
    source_candidate_ids = [candidate["id"] for candidate in source_candidates]
    _require(
        source_candidate_ids == round_candidate_ids,
        "source-round candidate order/identity differs",
    )
    _require(source_manifest["round_id"] == manifest["round_id"], "source-round round_id differs")
    _require(
        binding["control_commit"] == source_manifest["coordinator"]["control_commit"],
        "source-round bound control commit differs",
    )
    _require(
        binding["parent_commit"] == source_manifest["parent"]["source_commit"],
        "source-round bound parent commit differs",
    )
    _require(source_manifest["parent"]["id"] == manifest["parent"], "source-round parent differs")
    _require(
        source_manifest["parent"]["source_commit"] == manifest["parent_ref"]["commit"],
        "source-round parent commit differs",
    )
    _require(
        source_manifest["parent"]["worktree"] == manifest["parent_ref"]["worktree_realpath"],
        "source-round parent worktree differs",
    )
    for index, (round_candidate, source_candidate) in enumerate(
        zip(round_candidates, source_candidates)
    ):
        source_ref = round_candidate["source_ref"]
        _require(
            source_candidate["source_commit"] == source_ref["commit"],
            f"source-round candidate {index} commit differs",
        )
        _require(
            source_candidate["worktree"] == source_ref["worktree_realpath"],
            f"source-round candidate {index} worktree differs",
        )
        for field in (
            "primary_change",
            "hypothesis",
            "expected_profile_metrics",
            "risks",
        ):
            _require(
                source_candidate[field] == round_candidate[field],
                f"source-round candidate {index} {field} differs",
            )

    required_plan_fields = {
        "schema_version",
        "round_id",
        "mode",
        "execution_status",
        "claim_scope",
        "promotion",
        "manifest_raw_sha256",
        "manifest_canonical_sha256",
        "frozen_execution_contract_sha256",
        "lock_contract",
        "reuse_contract",
        "sealed_parent_contract",
        "preflight",
        "schedule_formula",
        "pairwise_orders",
        "blocks",
        "next_required_action",
    }
    _require(set(plan) == required_plan_fields, "source-round plan fields differ")
    _require(plan["schema_version"] == 1, "source-round plan schema differs")
    _require(plan["round_id"] == manifest["round_id"], "source-round plan round_id differs")
    _require(plan["mode"] == "check_only_plan", "source-round plan mode differs")
    _require(plan["execution_status"] == "NOT_RUN", "source-round plan status differs")
    _require(
        plan["claim_scope"] == "source_round_preflight_and_schedule_only",
        "source-round plan claim scope differs",
    )
    _require(
        plan["promotion"]
        == {
            "eligible": False,
            "candidate_id": None,
            "reason": "no benchmark block has been executed live",
        },
        "check-only source-round plan cannot authorize promotion",
    )
    _require(
        plan["manifest_raw_sha256"] == binding["manifest_raw_sha256"]
        and plan["manifest_canonical_sha256"] == binding["manifest_canonical_sha256"],
        "source-round plan does not bind its manifest",
    )
    source_contract_sha = canonical_sha256(
        source_manifest["frozen_execution_contract"]
    )
    _require(
        plan["frozen_execution_contract_sha256"] == source_contract_sha,
        "source-round frozen execution contract differs",
    )
    source_contract = source_manifest["frozen_execution_contract"]
    measurement = manifest["measurement_contract"]
    expected_timing = {
        "warmup_iterations": measurement["warmup_iterations"],
        "steady_iterations": measurement["steady_iterations"],
        **measurement["timer"],
    }
    _require(
        source_contract["algorithm"] == measurement["algorithm"]
        and source_contract["shape"] == measurement["workload"]
        and source_contract["timing"] == expected_timing
        and "created_utc" not in source_contract["toolchain"]
        and canonical_sha256(_normalized_toolchain(source_contract["toolchain"]))
        == measurement["toolchain_sha256"]
        and canonical_sha256(source_contract["gpu_mapping"])
        == measurement["gpu_mapping_sha256"],
        "source-round frozen execution contract does not match the round measurement contract",
    )
    _require(
        plan["schedule_formula"]
        == "P0,C1_0..Cn_0,C1_1..Cn_1,P1,C1_2..Cn_2,P2,P3,C1_3..Cn_3",
        "source-round schedule formula differs",
    )
    _require(
        plan["lock_contract"]
        == {
            "path": "/tmp/deepep-rail-balance-hop-campaign.lock",
            "holder": "neutral_control_checkout",
            "required_scope": "entire_round_through_all_descendant_process_exit_and_final_evidence_fsync",
        },
        "source-round global lock contract differs",
    )
    _require(
        plan["reuse_contract"]
        == {
            "parent_build_bundle_may_be_shared_across_parent_blocks": True,
            "parent_gate_bundle_may_be_shared_across_parent_blocks": True,
            "raw_benchmark_artifacts_may_be_shared": False,
            "jit_directories_may_be_shared": False,
            "bundle_refs_status": "PLANNED_NOT_VALIDATED_AS_BUILD_OUTPUT",
        },
        "source-round evidence reuse contract differs",
    )

    preflight = plan["preflight"]
    _require(
        isinstance(preflight, dict)
        and set(preflight)
        == {
            "status",
            "scope",
            "control",
            "parent",
            "candidates",
            "git_common_identity",
            "coordinator_git",
            "artifact_root",
            "harness_sha256",
            "harness_checkout_count",
            "patches",
        }
        and preflight.get("status") == "passed"
        and preflight.get("scope") == "git_and_plan_only_no_cuda_execution",
        "source-round Git preflight did not pass",
    )
    source_artifact_root = Path(source_manifest["coordinator"]["artifact_root"])
    try:
        resolved_source_artifact_root = source_artifact_root.resolve(strict=True)
        source_artifact_root_stat = os.stat(
            resolved_source_artifact_root, follow_symlinks=False
        )
    except OSError as error:
        raise RoundError(f"source-round artifact root is unavailable: {error}") from error
    _require(
        resolved_source_artifact_root == source_artifact_root
        and stat.S_ISDIR(source_artifact_root_stat.st_mode)
        and not source_artifact_root_stat.st_mode
        & (stat.S_IWGRP | stat.S_IWOTH)
        and preflight.get("artifact_root") == str(source_artifact_root),
        "source-round preflight artifact root differs",
    )
    control = preflight.get("control")
    checked_parent = preflight.get("parent")
    checked_candidates = preflight.get("candidates")
    _require(
        isinstance(control, dict)
        and isinstance(checked_parent, dict)
        and isinstance(checked_candidates, list)
        and len(checked_candidates) == len(source_candidates),
        "source-round preflight identities are incomplete",
    )
    _require(
        control.get("worktree") == source_manifest["coordinator"]["control_worktree"]
        and control.get("source_commit") == source_manifest["coordinator"]["control_commit"],
        "source-round control checkout proof differs",
    )
    _require(
        checked_parent.get("worktree") == manifest["parent_ref"]["worktree_realpath"]
        and checked_parent.get("source_commit") == manifest["parent_ref"]["commit"]
        and checked_parent.get("worktree_device") == manifest["parent_ref"]["worktree_device"]
        and checked_parent.get("worktree_inode") == manifest["parent_ref"]["worktree_inode"],
        "source-round parent preflight proof differs",
    )
    for index, (round_candidate, checked) in enumerate(
        zip(round_candidates, checked_candidates)
    ):
        source_ref = round_candidate["source_ref"]
        _require(
            checked.get("id") == round_candidate["id"]
            and checked.get("worktree") == source_ref["worktree_realpath"]
            and checked.get("source_commit") == source_ref["commit"]
            and checked.get("worktree_device") == source_ref["worktree_device"]
            and checked.get("worktree_inode") == source_ref["worktree_inode"],
            f"source-round candidate {index} preflight proof differs",
        )
    common = preflight.get("git_common_identity")
    _require(isinstance(common, dict), "source-round Git common-dir proof is missing")
    common_identity = (
        common.get("realpath"),
        common.get("device"),
        common.get("inode"),
    )
    for checked in (control, checked_parent, *checked_candidates):
        _require(
            (
                checked.get("git_common_dir"),
                checked.get("git_common_device"),
                checked.get("git_common_inode"),
            )
            == common_identity,
            "source-round worktrees do not share the preflight Git common-dir",
        )

    patch_checks = preflight.get("patches")
    _require(
        isinstance(patch_checks, list) and len(patch_checks) == len(source_candidates),
        "source-round patch proofs are incomplete",
    )
    allowed_files = set(source_manifest["source_policy"]["allowed_candidate_files"])
    for index, (source_candidate, patch) in enumerate(
        zip(source_candidates, patch_checks)
    ):
        declared = source_candidate["declared_patch"]
        _require(
            isinstance(patch, dict)
            and patch.get("candidate_id") == source_candidate["id"]
            and patch.get("base_commit") == manifest["parent_ref"]["commit"]
            and patch.get("candidate_commit") == source_candidate["source_commit"]
            and patch.get("commit_distance") == 1
            and patch.get("direct_single_parent") is True
            and patch.get("changed_files") == declared["changed_files"]
            and patch.get("diff_sha256") == declared["diff_sha256"],
            f"source-round candidate {index} direct-parent/diff proof differs",
        )
        _require(
            set(patch["changed_files"]) <= allowed_files,
            f"source-round candidate {index} changes an unrelated file",
        )

    harness = source_manifest["source_policy"]["harness_sha256"]
    expected_harness_hashes = {
        "tests/elastic/rail_balance_campaign_round.py": evaluator_sha,
        "tests/elastic/rail_balance_source_round_coordinator.py": coordinator_sha,
        "tests/elastic/rail_balance_source_round_executor.py": executor_sha,
        "tests/elastic/run_rail_balance_hop_campaign.py": manifest["measurement_contract"]["campaign_runner_sha256"],
        "tests/elastic/rail_balance_campaign_schema.py": manifest["measurement_contract"]["campaign_schema_sha256"],
        "tests/elastic/bench_rail_balance_hybrid_lsa.py": manifest["measurement_contract"]["benchmark_harness_sha256"],
    }
    for path, expected in expected_harness_hashes.items():
        _require(harness.get(path) == expected, f"source-round harness hash differs: {path}")
    _require(
        preflight.get("harness_sha256") == harness
        and preflight.get("harness_checkout_count") == 2 + len(source_candidates),
        "source-round cross-checkout harness proof differs",
    )

    schedule: list[tuple[str, str, int]] = [("parent", manifest["parent"], 0)]
    schedule.extend(("candidate", candidate_id, 0) for candidate_id in round_candidate_ids)
    schedule.extend(("candidate", candidate_id, 1) for candidate_id in round_candidate_ids)
    schedule.append(("parent", manifest["parent"], 1))
    schedule.extend(("candidate", candidate_id, 2) for candidate_id in round_candidate_ids)
    schedule.extend(("parent", manifest["parent"], repetition) for repetition in (2, 3))
    schedule.extend(("candidate", candidate_id, 3) for candidate_id in round_candidate_ids)
    plan_blocks = plan["blocks"]
    _require(
        isinstance(plan_blocks, list) and len(plan_blocks) == len(schedule),
        "source-round plan block count differs",
    )
    variants = {
        source_manifest["parent"]["id"]: source_manifest["parent"],
        **{candidate["id"]: candidate for candidate in source_candidates},
    }
    plan_block_fields = {
        "ordinal",
        "role",
        "variant_id",
        "repetition",
        "run_id",
        "worktree",
        "source_commit",
        "build_bundle_ref",
        "gate_bundle_ref",
        "frozen_execution_contract_sha256",
        "artifact_dir",
        "raw_benchmark_artifact",
        "jit_dir",
        "execution_status",
        "live_execution_required",
        "raw_benchmark_reuse_allowed",
        "jit_reuse_allowed",
    }
    plan_run_ids: set[str] = set()
    plan_artifact_dirs: set[str] = set()
    plan_raw_paths: set[str] = set()
    plan_jit_paths: set[str] = set()
    for ordinal, (block, expected) in enumerate(zip(plan_blocks, schedule)):
        role, variant_id, repetition = expected
        variant = variants[variant_id]
        expected_run_id = (
            f"{manifest['round_id']}-{ordinal:02d}-{variant_id}-{repetition}"
        )
        expected_artifact_dir = f"runs/{expected_run_id}"
        expected_attempt_dir = (
            f"{expected_artifact_dir}/stages/bench-abba-a1/attempt-01"
        )
        _require(
            isinstance(block, dict)
            and set(block) == plan_block_fields
            and block.get("ordinal") == ordinal
            and block.get("role") == role
            and block.get("variant_id") == variant_id
            and block.get("repetition") == repetition
            and block.get("worktree") == variant["worktree"]
            and block.get("source_commit") == variant["source_commit"]
            and block.get("build_bundle_ref") == variant["build_bundle_ref"]
            and block.get("gate_bundle_ref") == variant["gate_bundle_ref"]
            and block.get("frozen_execution_contract_sha256") == source_contract_sha
            and block.get("run_id") == expected_run_id
            and block.get("artifact_dir") == expected_artifact_dir
            and block.get("raw_benchmark_artifact")
            == f"{expected_attempt_dir}/report.json"
            and block.get("jit_dir") == f"{expected_attempt_dir}/jit"
            and block.get("execution_status") == "NOT_RUN"
            and block.get("live_execution_required") is True
            and block.get("raw_benchmark_reuse_allowed") is False
            and block.get("jit_reuse_allowed") is False,
            f"source-round plan block {ordinal} differs",
        )
        _slug(block["run_id"], f"source-round plan block {ordinal}.run_id")
        _safe_relative(
            block["artifact_dir"], f"source-round plan block {ordinal}.artifact_dir"
        )
        _safe_relative(
            block["raw_benchmark_artifact"],
            f"source-round plan block {ordinal}.raw_benchmark_artifact",
        )
        _safe_relative(
            block["jit_dir"], f"source-round plan block {ordinal}.jit_dir"
        )
        for collection, value, label in (
            (plan_run_ids, block["run_id"], "run_id"),
            (plan_artifact_dirs, block["artifact_dir"], "artifact_dir"),
            (plan_raw_paths, block["raw_benchmark_artifact"], "raw benchmark path"),
            (plan_jit_paths, block["jit_dir"], "JIT path"),
        ):
            _require(value not in collection, f"source-round plan reuses a {label}")
            collection.add(value)
    expected_projection = list(_ABBA_BAAB)
    _require(
        plan["pairwise_orders"]
        == {candidate_id: expected_projection for candidate_id in round_candidate_ids},
        "source-round pairwise schedules differ",
    )
    return {
        "binding": binding,
        "manifest": source_manifest,
        "plan": plan,
        "code_snapshots": {
            "round evaluator": {
                "path": evaluator_path,
                "sha256": evaluator_sha,
                "identity": evaluator_snapshot,
            },
            "source-round coordinator": {
                "path": coordinator_path,
                "sha256": coordinator_sha,
                "identity": coordinator_snapshot,
            },
            "source-round executor": {
                "path": executor_path,
                "sha256": executor_sha,
                "identity": executor_snapshot,
            },
        },
    }


def _verify_code_snapshots(binding_state: dict[str, Any]) -> None:
    snapshots = binding_state.get("code_snapshots")
    _require(isinstance(snapshots, dict), "source-round code snapshots are missing")
    for name, expected in snapshots.items():
        _require(isinstance(expected, dict), f"{name} snapshot is invalid")
        path = expected.get("path")
        _require(isinstance(path, Path), f"{name} snapshot path is invalid")
        digest, identity = _sha256_file_snapshot(path)
        _require(
            digest == expected.get("sha256")
            and identity == expected.get("identity"),
            f"{name} changed during round evaluation",
        )


def _validate_round_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "round_id",
        "comparison_axis",
        "parent",
        "parent_campaign_candidate",
        "parent_ref",
        "source_round_binding",
        "frozen_contract_sha256",
        "measurement_contract",
        "noise_policy",
        "parent_blocks",
        "candidates",
    }
    _require(set(manifest) == required, "round manifest fields are incomplete or unknown")
    _require(manifest["schema_version"] == ROUND_SCHEMA_VERSION, "unsupported schema_version")
    _slug(manifest["round_id"], "round_id")
    _require(
        manifest["comparison_axis"] == "source_version",
        "formal leader promotion requires comparison_axis=source_version",
    )
    parent = _slug(manifest["parent"], "parent")
    parent_candidate = _validate_campaign_candidate(
        manifest["parent_campaign_candidate"], "parent_campaign_candidate"
    )
    _require(parent_candidate["id"] == parent, "parent campaign identity differs")
    parent_ref = _validate_source_ref(manifest["parent_ref"], "parent_ref")
    _require(parent_ref["ref_id"] == parent, "parent_ref.ref_id differs from parent")
    contract = _sha(manifest["frozen_contract_sha256"], "frozen_contract_sha256")

    measurement = manifest["measurement_contract"]
    measurement_fields = {
        "report_schema_version",
        "claim_scope",
        "metric",
        "warmup_iterations",
        "steady_iterations",
        "world_size",
        "order",
        "timer",
        "workload",
        "algorithm",
        "campaign_runner_sha256",
        "campaign_schema_sha256",
        "round_evaluator_sha256",
        "benchmark_harness_sha256",
        "loaded_libraries_sha256",
        "toolchain_sha256",
        "gpu_mapping_sha256",
        "semantic_config_sha256",
        "normalized_execution_identity_sha256",
        "coordinator_boot_id",
        "coordinator_lock_identity_sha256",
    }
    _require(
        isinstance(measurement, dict) and set(measurement) == measurement_fields,
        "measurement_contract fields are incomplete or unknown",
    )
    _require(measurement["report_schema_version"] == 4, "formal C100 report schema must be 4")
    _require(measurement["claim_scope"] == "checked_adapter_only", "C100 claim_scope differs")
    _require(measurement["metric"] == "stage_truth_ns", "metric must be stage_truth_ns")
    _require(measurement["warmup_iterations"] == 10, "formal warmup must be exactly 10")
    _require(measurement["steady_iterations"] == 100, "formal steady count must be exactly 100")
    _require(measurement["world_size"] == 8, "formal local round requires 8 GPUs")
    _require(tuple(measurement["order"]) == _ABBA_BAAB, "order must be exact ABBA+BAAB")
    _require(
        isinstance(measurement["timer"], dict)
        and set(measurement["timer"]) == _TIMER_FIELDS,
        "timer fields are incomplete or unknown",
    )
    _require(
        isinstance(measurement["workload"], dict)
        and set(measurement["workload"]) == _WORKLOAD_FIELDS,
        "workload fields are incomplete or unknown",
    )
    _require(
        measurement["workload"]["world_size"] == 8,
        "workload world_size differs",
    )
    _require(
        isinstance(measurement["algorithm"], dict)
        and set(measurement["algorithm"]) == _ALGORITHM_FIELDS,
        "algorithm fields are incomplete or unknown",
    )
    for field in (
        "campaign_runner_sha256",
        "campaign_schema_sha256",
        "round_evaluator_sha256",
        "benchmark_harness_sha256",
        "loaded_libraries_sha256",
        "toolchain_sha256",
        "gpu_mapping_sha256",
        "semantic_config_sha256",
        "normalized_execution_identity_sha256",
        "coordinator_lock_identity_sha256",
    ):
        _sha(measurement[field], f"measurement_contract.{field}")
    _nonempty(measurement["coordinator_boot_id"], "measurement_contract.coordinator_boot_id")
    _require(
        measurement["round_evaluator_sha256"] == evaluator_sha256(),
        "round evaluator code differs from the frozen contract",
    )
    _require(
        manifest["noise_policy"] == FORMAL_NOISE_POLICY,
        "noise_policy must equal the fixed formal publication rule",
    )

    candidates = manifest["candidates"]
    _require(isinstance(candidates, list) and 2 <= len(candidates) <= 4, "round requires 2-4 candidates")
    candidate_count = len(candidates)
    ids: set[str] = set()
    commits = {parent_ref["commit"]}
    trees = {parent_ref["working_tree_identity_sha256"]}
    builds = {parent_ref["build_bundle_sha256"]}
    codes = {parent_ref["code_identity_sha256"]}
    worktrees = {(parent_ref["worktree_device"], parent_ref["worktree_inode"])}
    global_ordinals: set[int] = set()
    artifact_root_paths: set[str] = set()
    artifact_root_identities: set[tuple[int, int]] = set()
    coordinator_paths: set[str] = set()
    result_hashes: set[str] = set()
    finalized_hashes: set[str] = set()
    coordinator_hashes: set[str] = set()

    def validate_block(
        block: object,
        *,
        name: str,
        ordinal: int,
        role: str,
        global_ordinal: int,
    ) -> None:
        _require(
            isinstance(block, dict) and set(block) == _BLOCK_FIELDS,
            f"{name} fields differ",
        )
        _require(block["ordinal"] == ordinal, f"{name}.ordinal differs")
        _require(block["role"] == role, f"{name}.role differs")
        _require(
            block["global_ordinal"] == global_ordinal,
            f"{name}.global_ordinal violates the global round schedule",
        )
        _require(
            block["global_ordinal"] not in global_ordinals,
            "global block ordinal is duplicated",
        )
        global_ordinals.add(block["global_ordinal"])
        for field in (
            "campaign_result_raw_sha256",
            "campaign_manifest_raw_sha256",
            "finalized_raw_sha256",
            "coordinator_record_raw_sha256",
        ):
            _sha(block[field], f"{name}.{field}")
        _slug(block["stage_id"], f"{name}.stage_id")
        _require(block["attempt"] == 1, f"{name} must select attempt 1")
        _safe_relative(block["artifact"], f"{name}.artifact")
        coordinator_path = Path(
            _nonempty(block["coordinator_record"], f"{name}.coordinator_record")
        )
        _require(
            coordinator_path.is_absolute(),
            f"{name}.coordinator_record must be an absolute control-plane path",
        )
        root = Path(
            _nonempty(block["artifact_root_realpath"], f"{name}.artifact_root_realpath")
        )
        _require(root.is_absolute(), f"{name}.artifact_root_realpath must be absolute")
        _positive_int(block["artifact_root_device"], f"{name}.artifact_root_device")
        _positive_int(block["artifact_root_inode"], f"{name}.artifact_root_inode")
        root_path = str(root)
        root_identity = (
            block["artifact_root_device"],
            block["artifact_root_inode"],
        )
        _require(
            root_path not in artifact_root_paths
            and root_identity not in artifact_root_identities,
            "round reuses an artifact root path or inode",
        )
        artifact_root_paths.add(root_path)
        artifact_root_identities.add(root_identity)
        _require(
            str(coordinator_path) not in coordinator_paths,
            "round reuses a coordinator record path",
        )
        coordinator_paths.add(str(coordinator_path))
        for collection, value, label in (
            (result_hashes, block["campaign_result_raw_sha256"], "campaign result"),
            (finalized_hashes, block["finalized_raw_sha256"], "FINALIZED record"),
            (
                coordinator_hashes,
                block["coordinator_record_raw_sha256"],
                "coordinator record",
            ),
        ):
            _require(value not in collection, f"round reuses a {label}")
            collection.add(value)

    parent_blocks = manifest["parent_blocks"]
    _require(
        isinstance(parent_blocks, list) and len(parent_blocks) == 4,
        "round needs exactly four shared parent blocks",
    )
    parent_global_ordinals = (0, 2 * candidate_count + 1, 3 * candidate_count + 2, 3 * candidate_count + 3)
    for ordinal, (block, global_ordinal) in enumerate(
        zip(parent_blocks, parent_global_ordinals)
    ):
        validate_block(
            block,
            name=f"parent_blocks[{ordinal}]",
            ordinal=ordinal,
            role="parent",
            global_ordinal=global_ordinal,
        )

    for index, candidate in enumerate(candidates):
        name = f"candidates[{index}]"
        _require(
            isinstance(candidate, dict) and set(candidate) == _CANDIDATE_FIELDS,
            f"{name} fields are incomplete or unknown",
        )
        identity = _validate_campaign_candidate(
            {key: candidate[key] for key in _CAMPAIGN_CANDIDATE_FIELDS},
            name,
        )
        candidate_id = identity["id"]
        _require(candidate_id not in ids and candidate_id != parent, f"duplicate candidate id: {candidate_id}")
        ids.add(candidate_id)
        _require(identity["parent"] == parent, f"{name}.parent differs")
        _require(candidate["frozen_contract_sha256"] == contract, f"{name} contract differs")
        source = _validate_source_ref(candidate["source_ref"], f"{name}.source_ref")
        _require(source["ref_id"] == candidate_id, f"{name} source ref differs")
        for collection, value, label in (
            (commits, source["commit"], "commit"),
            (trees, source["working_tree_identity_sha256"], "source tree"),
            (builds, source["build_bundle_sha256"], "build bundle"),
            (codes, source["code_identity_sha256"], "code identity"),
            (worktrees, (source["worktree_device"], source["worktree_inode"]), "worktree"),
        ):
            _require(value not in collection, f"{name} reuses a parent/sibling {label}")
            collection.add(value)
        blocks = candidate["benchmark_blocks"]
        _require(
            isinstance(blocks, list) and len(blocks) == 4,
            f"{name} needs exactly four candidate blocks",
        )
        candidate_global_ordinals = (
            1 + index,
            1 + candidate_count + index,
            2 + 2 * candidate_count + index,
            4 + 3 * candidate_count + index,
        )
        for ordinal, (block, global_ordinal) in enumerate(
            zip(blocks, candidate_global_ordinals)
        ):
            validate_block(
                block,
                name=f"{name}.benchmark_blocks[{ordinal}]",
                ordinal=ordinal,
                role="candidate",
                global_ordinal=global_ordinal,
            )
    _require(
        global_ordinals == set(range(4 + 4 * candidate_count)),
        "global block ordinals must be one dense round-wide sequence",
    )
    _load_source_round_binding(manifest)


def load_round_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest, digest = _read_strict_json(path)
    _validate_round_manifest(manifest)
    return manifest, digest


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise CandidateEvidenceError("cannot summarize empty samples")
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(float(item)) for item in values):
        raise CandidateEvidenceError("statistics require nonempty finite samples")
    numeric = [float(item) for item in values]
    mean = statistics.fmean(numeric)
    deviation = statistics.pstdev(numeric)
    return {
        "count": len(numeric),
        "median": float(statistics.median(numeric)),
        "p95": _percentile(numeric, 95),
        "p99": _percentile(numeric, 99),
        "mean": mean,
        "std_population": deviation,
        "cv_population": 0.0 if mean == 0 else deviation / mean,
        "min": min(numeric),
        "max": max(numeric),
    }


def _scaled_mad(values: Sequence[float]) -> float:
    center = float(statistics.median(values))
    return 1.4826 * float(statistics.median(abs(float(item) - center) for item in values))


def _stats_match(actual: object, expected: dict[str, float | int]) -> None:
    _candidate_require(isinstance(actual, dict), "C100 summary is not an object")
    for name, expected_value in expected.items():
        _candidate_require(name in actual, f"C100 summary lacks {name}")
        if name == "count":
            _candidate_require(actual[name] == expected_value, "C100 summary count differs")
            continue
        value = actual[name]
        _candidate_require(type(value) in (int, float) and math.isfinite(float(value)), f"C100 summary {name} invalid")
        tolerance = max(1e-9, abs(float(expected_value)) * 1e-12)
        _candidate_require(math.isclose(float(value), float(expected_value), rel_tol=0, abs_tol=tolerance), f"C100 summary {name} differs")


def _normalize_file_identity(value: object, name: str) -> dict[str, int | str]:
    _candidate_require(isinstance(value, dict), f"{name} is missing")
    sha = value.get("sha256")
    size = value.get("size_bytes")
    _candidate_require(isinstance(sha, str) and bool(_SHA256.fullmatch(sha)), f"{name}.sha256 invalid")
    _candidate_require(type(size) is int and size >= 0, f"{name}.size_bytes invalid")
    return {"sha256": sha, "size_bytes": size}


def _normalize_sources(value: object, name: str) -> dict[str, dict[str, int | str]]:
    _candidate_require(isinstance(value, dict) and value, f"{name} is empty")
    result: dict[str, dict[str, int | str]] = {}
    for path, identity in value.items():
        _candidate_require(isinstance(path, str) and bool(path), f"{name} path invalid")
        result[path] = _normalize_file_identity(identity, f"{name}[{path}]")
    return result


def _normalize_libraries(value: object, name: str) -> list[dict[str, int | str]]:
    _candidate_require(isinstance(value, list) and value, f"{name} is empty")
    rows = [
        _normalize_file_identity(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]
    return sorted(rows, key=lambda row: (str(row["sha256"]), int(row["size_bytes"])))


def _normalize_jit(value: object, name: str) -> list[dict[str, int | str]]:
    _candidate_require(isinstance(value, dict), f"{name} is missing")
    artifacts = value.get("artifacts")
    _candidate_require(isinstance(artifacts, list), f"{name}.artifacts missing")
    rows = []
    for index, artifact in enumerate(artifacts):
        normalized = _normalize_file_identity(artifact, f"{name}.artifacts[{index}]")
        relative = artifact.get("relative_path") if isinstance(artifact, dict) else None
        _candidate_require(isinstance(relative, str) and bool(relative), f"{name} relative_path invalid")
        rows.append({"relative_path": relative, **normalized})
    return sorted(rows, key=lambda row: str(row["relative_path"]))


def _normalize_command(argv: object) -> list[str]:
    _candidate_require(isinstance(argv, list) and all(isinstance(item, str) for item in argv), "C100 command argv invalid")
    result: list[str] = []
    cursor = 0
    while cursor < len(argv):
        item = argv[cursor]
        if item in {"--master-port", "--json-out"}:
            _candidate_require(cursor + 1 < len(argv), f"C100 command lacks {item} value")
            cursor += 2
            continue
        result.append(item)
        cursor += 1
    return result


def _validate_identity(
    report: dict[str, Any],
    measurement: dict[str, Any],
    source_ref: dict[str, Any],
    *,
    expected_jit_root: Path,
    expected_report_path: Path,
) -> dict[str, str]:
    identity = report.get("identity")
    _candidate_require(isinstance(identity, dict), "C100 identity is missing")
    for field in (
        "code_identity_stable_during_run",
        "jit_cache_empty_before_run",
        "jit_identity_stable_after_cold",
    ):
        _candidate_require(identity.get(field) is True, f"C100 identity gate failed: {field}")
    semantic = identity.get("semantic_config")
    semantic_sha = identity.get("semantic_config_sha256")
    _candidate_require(isinstance(semantic, dict), "C100 semantic_config missing")
    _candidate_require(canonical_sha256(semantic) == semantic_sha, "C100 semantic_config hash differs")
    _candidate_require(semantic_sha == measurement["semantic_config_sha256"], "C100 semantic config differs across round")

    pre = identity.get("pre_measurement")
    post = identity.get("post_measurement")
    _candidate_require(isinstance(pre, dict) and isinstance(post, dict), "C100 pre/post identity missing")
    pre_extension = _normalize_file_identity(pre.get("extension"), "pre extension")
    post_extension = _normalize_file_identity(post.get("extension"), "post extension")
    _candidate_require(pre_extension == post_extension, "C100 extension changed during measurement")
    pre_sources = _normalize_sources(pre.get("sources"), "pre sources")
    post_sources = _normalize_sources(post.get("sources"), "post sources")
    _candidate_require(pre_sources == post_sources, "C100 sources changed during measurement")
    pre_libraries = _normalize_libraries(
        pre.get("loaded_libraries"), "pre loaded libraries"
    )
    post_libraries = _normalize_libraries(
        post.get("loaded_libraries"), "post loaded libraries"
    )
    _candidate_require(
        pre_libraries == post_libraries,
        "C100 loaded CUDA/NCCL libraries changed during measurement",
    )
    _candidate_require(
        canonical_sha256(pre_libraries) == measurement["loaded_libraries_sha256"],
        "C100 loaded CUDA/NCCL libraries differ across the round",
    )
    pre_jit_value = pre.get("jit_cache")
    cold_jit_value = identity.get("after_cold_transaction_jit")
    post_jit_value = post.get("jit_cache")
    for value, name, cache_existed in (
        (pre_jit_value, "pre JIT", False),
        (cold_jit_value, "cold JIT", True),
        (post_jit_value, "post JIT", True),
    ):
        _candidate_require(isinstance(value, dict), f"{name} is missing")
        _candidate_require(
            value.get("root") == str(expected_jit_root)
            and value.get("cache_existed") is cache_existed,
            f"{name} root/cache state differs from the source-round plan",
        )
    pre_jit = _normalize_jit(pre_jit_value, "pre JIT")
    cold_jit = _normalize_jit(cold_jit_value, "cold JIT")
    post_jit = _normalize_jit(post_jit_value, "post JIT")
    _candidate_require(pre_jit == [], "C100 JIT cache was not empty before the block")
    _candidate_require(cold_jit == post_jit, "C100 JIT identity changed after cold transaction")
    cubins = [row for row in post_jit if str(row["relative_path"]).endswith("/kernel.cubin")]
    _candidate_require(bool(cubins), "C100 JIT identity contains no cubin")

    environment = report.get("environment")
    command = report.get("command")
    _candidate_require(isinstance(environment, dict) and isinstance(environment.get("variables"), dict), "C100 environment variables missing")
    _candidate_require(isinstance(command, dict), "C100 command missing")
    _candidate_require(
        command.get("cwd") == source_ref["worktree_realpath"],
        "C100 command cwd differs from the role worktree",
    )
    argv = command.get("argv")
    _candidate_require(isinstance(argv, list), "C100 command argv invalid")
    _candidate_require(
        environment["variables"].get("EP_JIT_CACHE_DIR")
        == str(expected_jit_root),
        "C100 EP_JIT_CACHE_DIR differs from the source-round plan",
    )
    _candidate_require(
        _argv_value(argv, "--json-out") == str(expected_report_path),
        "C100 command --json-out differs from the source-round plan",
    )
    master_port = _argv_value(argv, "--master-port")
    _candidate_require(
        master_port.isdecimal() and 1 <= int(master_port) <= 65535,
        "C100 command --master-port is invalid",
    )
    execution = {
        "semantic_config_sha256": semantic_sha,
        "environment": environment["variables"],
        "command": argv,
        "jit_cache_root": post.get("jit_cache", {}).get("root"),
    }
    raw_execution_sha = identity.get("execution_config_sha256")
    _candidate_require(canonical_sha256(execution) == raw_execution_sha, "C100 raw execution_config hash differs")
    normalized_environment = dict(environment["variables"])
    normalized_environment.pop("EP_JIT_CACHE_DIR", None)
    normalized_execution = {
        "semantic_config_sha256": semantic_sha,
        "environment": normalized_environment,
        "command": _normalize_command(argv),
    }
    normalized_execution_sha = canonical_sha256(normalized_execution)
    _candidate_require(
        normalized_execution_sha == measurement["normalized_execution_identity_sha256"],
        "C100 normalized execution identity differs across blocks",
    )
    harness_matches = [
        item["sha256"]
        for path, item in pre_sources.items()
        if path.endswith("tests/elastic/bench_rail_balance_hybrid_lsa.py")
    ]
    _candidate_require(len(harness_matches) == 1, "C100 benchmark harness source identity is ambiguous")
    _candidate_require(harness_matches[0] == measurement["benchmark_harness_sha256"], "C100 benchmark harness differs")
    build_bundle = canonical_sha256(
        {
            "source_commit": source_ref["commit"],
            "source_identity_sha256": source_ref["working_tree_identity_sha256"],
            "extension": pre_extension,
            "sources": pre_sources,
            "toolchain_sha256": measurement["toolchain_sha256"],
            "campaign_runner_sha256": measurement["campaign_runner_sha256"],
            "campaign_schema_sha256": measurement["campaign_schema_sha256"],
        }
    )
    code_identity = canonical_sha256({"build_bundle_sha256": build_bundle, "jit_cubins": cubins})
    _candidate_require(build_bundle == source_ref["build_bundle_sha256"], "C100 build bundle differs from source_ref")
    _candidate_require(code_identity == source_ref["code_identity_sha256"], "C100 code identity differs from source_ref")
    return {
        "semantic_config_sha256": semantic_sha,
        "raw_execution_identity_sha256": str(raw_execution_sha),
        "normalized_execution_identity_sha256": normalized_execution_sha,
        "extension_identity_sha256": canonical_sha256(pre_extension),
        "sources_identity_sha256": canonical_sha256(pre_sources),
        "jit_cubin_identity_sha256": canonical_sha256(cubins),
        "build_bundle_sha256": build_bundle,
        "code_identity_sha256": code_identity,
    }


def _validate_raw_sample(
    sample: object, category: str, index: int
) -> tuple[float, int, int]:
    _candidate_require(isinstance(sample, dict), "C100 raw sample is not an object")
    _candidate_require(sample.get("category") == category and sample.get("category_index") == index, "C100 sample identity differs")
    rows = sample.get("rank_raw")
    _candidate_require(isinstance(rows, list) and len(rows) == 8, "C100 rank_raw width differs")
    starts: list[int] = []
    ends: list[int] = []
    durations: list[int] = []
    ranks: list[int] = []
    for row in rows:
        _candidate_require(isinstance(row, dict), "C100 rank row invalid")
        rank = row.get("rank")
        start = row.get("stage_start_ns")
        end = row.get("stage_end_ns")
        timing = row.get("timings_ns")
        _candidate_require(type(rank) is int, "C100 rank invalid")
        _candidate_require(type(start) is int and type(end) is int and 0 <= start < end <= 2**63 - 1, "C100 timestamps invalid")
        _candidate_require(isinstance(timing, dict) and type(timing.get("stage")) is int, "C100 stage timing missing")
        duration = timing["stage"]
        _candidate_require(duration == end - start and duration > 0, "C100 duration differs from timestamps")
        ranks.append(rank)
        starts.append(start)
        ends.append(end)
        durations.append(duration)
    _candidate_require(ranks == list(range(8)), "C100 ranks are not ordered 0..7")
    span = max(ends) - min(starts)
    _candidate_require(sample.get("stage_truth_ns") == span and sample.get("stage_global_span_ns") == span, "C100 global span differs")
    _candidate_require(sample.get("stage_rank_max_duration_ns") == max(durations), "C100 rank max differs")
    rank_max = sample.get("rank_max_ns")
    _candidate_require(isinstance(rank_max, dict) and rank_max.get("stage") == max(durations), "C100 rank_max_ns differs")
    return float(span), min(starts), max(ends)


def _subset(value: object, fields: set[str], name: str) -> dict[str, Any]:
    _candidate_require(isinstance(value, dict), f"C100 {name} is not an object")
    missing = fields - set(value)
    _candidate_require(not missing, f"C100 {name} lacks {sorted(missing)}")
    return {field: value[field] for field in sorted(fields)}


def _validate_c100_report(
    report: dict[str, Any],
    measurement: dict[str, Any],
    source_ref: dict[str, Any],
    *,
    expected_jit_root: Path,
    expected_report_path: Path,
) -> dict[str, Any]:
    _candidate_require(report.get("schema_version") == 4, "C100 schema differs")
    _candidate_require(report.get("claim_scope") == "checked_adapter_only", "C100 claim_scope differs")
    run_id = report.get("run_id")
    _candidate_require(
        isinstance(run_id, str) and bool(run_id.strip()), "C100 run_id missing"
    )
    git = report.get("git")
    _candidate_require(isinstance(git, dict) and git.get("dirty") is False, "C100 git identity is dirty/missing")
    _candidate_require(git.get("commit") == source_ref["commit"], "C100 commit differs from role source")
    observed = report.get("measurement")
    _candidate_require(isinstance(observed, dict), "C100 measurement missing")
    for field, expected in (
        ("baseline_collection_eligible", True),
        ("measurement_depth_ok", True),
        ("persistent_report_requested", True),
        ("profiler_mode_declared", "disabled"),
        ("extra_synchronize_inside_stage", False),
        ("warmup_iterations", 10),
        ("steady_iterations", 100),
    ):
        _candidate_require(observed.get(field) == expected, f"C100 measurement {field} differs")
    for field, expected in measurement["timer"].items():
        _candidate_require(observed.get(field) == expected, f"C100 timer {field} differs")
    environment = report.get("environment")
    runtime = environment.get("runtime_state") if isinstance(environment, dict) else None
    _candidate_require(isinstance(runtime, dict), "C100 runtime_state missing")
    for field in ("system_commands_ok", "mps_processes_absent", "profiler_processes_absent"):
        _candidate_require(runtime.get(field) is True, f"C100 runtime gate failed: {field}")
    _candidate_require(runtime.get("unexpected_compute_app_pids") == [], "C100 unexpected GPU processes")
    gpu = runtime.get("gpu_state_validation")
    _candidate_require(isinstance(gpu, dict) and gpu.get("valid") is True, "C100 GPU state invalid")
    workload = _subset(report.get("shape"), _WORKLOAD_FIELDS, "shape")
    algorithm = _subset(report.get("shape"), _ALGORITHM_FIELDS, "shape")
    _candidate_require(workload == measurement["workload"], "C100 workload differs")
    _candidate_require(algorithm == measurement["algorithm"], "P/C algorithm semantics differ")
    warm = report.get("warm")
    steady = report.get("steady")
    _candidate_require(isinstance(warm, list) and len(warm) == 10, "C100 warm sample count differs")
    _candidate_require(isinstance(steady, list) and len(steady) == 100, "C100 steady sample count differs")
    warm_records = [
        _validate_raw_sample(item, "warm", index) for index, item in enumerate(warm)
    ]
    steady_records = [
        _validate_raw_sample(item, "steady", index)
        for index, item in enumerate(steady)
    ]
    all_records = [*warm_records, *steady_records]
    intervals = [(start, end) for _, start, end in all_records]
    _candidate_require(
        len(intervals) == len(set(intervals))
        and all(
            intervals[index][0] >= intervals[index - 1][1]
            for index in range(1, len(intervals))
        ),
        "C100 warm/steady global intervals repeat, overlap, or execute out of order",
    )
    warm_samples = [duration for duration, _, _ in warm_records]
    steady_samples = [duration for duration, _, _ in steady_records]
    stats = _stats(steady_samples)
    summary = report.get("steady_summary")
    _candidate_require(isinstance(summary, dict), "C100 steady_summary missing")
    _stats_match(summary.get("stage_truth_ns"), stats)
    identity = _validate_identity(
        report,
        measurement,
        source_ref,
        expected_jit_root=expected_jit_root,
        expected_report_path=expected_report_path,
    )
    return {
        "run_id": run_id,
        "git_commit": git["commit"],
        "workload_sha256": canonical_sha256(workload),
        "algorithm_sha256": canonical_sha256(algorithm),
        "warm_samples_ns": warm_samples,
        "samples_ns": steady_samples,
        "sample_intervals_ns": [list(interval) for interval in intervals],
        "stats": stats,
        **identity,
    }


def _source_identity(value: object, name: str, expected: dict[str, Any]) -> dict[str, Any]:
    _candidate_require(isinstance(value, dict), f"{name} missing")
    _candidate_require(value.get("commit") == expected["commit"], f"{name}.commit differs")
    _candidate_require(value.get("dirty") is False, f"{name} is dirty")
    _candidate_require(value.get("status_porcelain") == [], f"{name} status is not clean")
    _candidate_require(value.get("untracked_files") == [], f"{name} has untracked files")
    for field in ("tracked_diff_sha256", "untracked_tree_sha256", "working_tree_identity_sha256"):
        item = value.get(field)
        _candidate_require(isinstance(item, str) and bool(_SHA256.fullmatch(item)), f"{name}.{field} invalid")
    _candidate_require(value["working_tree_identity_sha256"] == expected["working_tree_identity_sha256"], f"{name} tree differs")
    return value


def _stages_by_id(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stages = campaign.get("stages")
    _candidate_require(isinstance(stages, list), "campaign stages missing")
    result: dict[str, dict[str, Any]] = {}
    for stage in stages:
        _candidate_require(isinstance(stage, dict) and isinstance(stage.get("id"), str), "campaign stage invalid")
        _candidate_require(stage["id"] not in result, "campaign stage duplicated")
        result[stage["id"]] = stage
    return result


def _attempt_passed(value: object) -> bool:
    return isinstance(value, dict) and value.get("passed") is True and value.get("returncode") == 0


def _gate_summary(
    stages: dict[str, dict[str, Any]], manifest_stages: list[dict[str, Any]], kind: str
) -> dict[str, Any]:
    plans = [stage for stage in manifest_stages if stage["kind"] == kind and stage.get("enabled", True) and stage["resource"] != "multinode"]
    if not plans:
        return {"passed": False, "reason": f"no enabled local {kind} stage", "stages": []}
    rows = []
    for plan in plans:
        stage = stages[plan["id"]]
        attempts = stage.get("attempts")
        passed = stage.get("status") == "passed" and isinstance(attempts, list) and bool(attempts) and all(_attempt_passed(item) for item in attempts)
        rows.append({"id": plan["id"], "status": stage.get("status"), "attempt_count": len(attempts) if isinstance(attempts, list) else None, "passed": passed})
    all_passed = all(row["passed"] for row in rows)
    return {"passed": all_passed, "reason": None if all_passed else f"one or more {kind} stages did not pass", "stages": rows}


def _parse_sha256sums(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CandidateEvidenceError("SHA256SUMS is not UTF-8") from error
    result: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("  ", 1)
        _candidate_require(len(parts) == 2 and bool(_SHA256.fullmatch(parts[0])), "SHA256SUMS row invalid")
        relative = _safe_relative(parts[1], "SHA256SUMS path")
        _candidate_require(relative == Path(relative).as_posix(), "SHA256SUMS path is not canonical")
        _candidate_require(relative not in result, "SHA256SUMS path duplicated")
        result[relative] = parts[0]
    return result


def _audit_artifact_tree(
    root: Path, expected: dict[str, str], *, root_identity: tuple[int, int]
) -> None:
    """Verify that SHA256SUMS covers the complete regular, unlinked artifact tree."""

    observed: dict[str, str] = {}

    def visit(directory: int, prefix: tuple[str, ...]) -> None:
        for name in sorted(os.listdir(directory)):
            relative_parts = (*prefix, name)
            relative = "/".join(relative_parts)
            try:
                before = os.stat(name, dir_fd=directory, follow_symlinks=False)
            except OSError as error:
                raise CandidateEvidenceError(
                    f"cannot inspect artifact tree entry {root}/{relative}: {error}"
                ) from error
            if stat.S_ISDIR(before.st_mode):
                _candidate_require(
                    not before.st_mode & (stat.S_IWGRP | stat.S_IWOTH),
                    f"artifact tree contains a group/world-writable directory: {root}/{relative}",
                )
                child: int | None = None
                try:
                    child = os.open(
                        name,
                        os.O_RDONLY
                        | os.O_CLOEXEC
                        | os.O_DIRECTORY
                        | os.O_NOFOLLOW,
                        dir_fd=directory,
                    )
                    opened = os.fstat(child)
                    _candidate_require(
                        (opened.st_dev, opened.st_ino)
                        == (before.st_dev, before.st_ino),
                        f"artifact directory changed while opening: {relative}",
                    )
                    visit(child, relative_parts)
                except OSError as error:
                    raise CandidateEvidenceError(
                        f"cannot safely open artifact directory {root}/{relative}: {error}"
                    ) from error
                finally:
                    if child is not None:
                        os.close(child)
                continue
            _candidate_require(
                stat.S_ISREG(before.st_mode)
                and before.st_nlink == 1
                and not before.st_mode & (stat.S_IWGRP | stat.S_IWOTH),
                f"artifact tree contains a linked or special file: {root}/{relative}",
            )
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                opened = os.fstat(descriptor)
                _candidate_require(
                    stat.S_ISREG(opened.st_mode)
                    and opened.st_nlink == 1
                    and (opened.st_dev, opened.st_ino)
                    == (before.st_dev, before.st_ino),
                    f"artifact file changed while opening: {relative}",
                )
                digest = hashlib.sha256()
                while chunk := os.read(descriptor, 1 << 20):
                    digest.update(chunk)
                after = os.fstat(descriptor)
                _candidate_require(
                    (
                        opened.st_size,
                        opened.st_mtime_ns,
                        opened.st_ctime_ns,
                    )
                    == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    f"artifact file changed while hashing: {relative}",
                )
            except OSError as error:
                raise CandidateEvidenceError(
                    f"cannot safely hash artifact file {root}/{relative}: {error}"
                ) from error
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            if relative not in {"SHA256SUMS", "FINALIZED.json"}:
                observed[relative] = digest.hexdigest()

    root_descriptor: int | None = None
    try:
        root_descriptor = os.open(
            root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        opened_root = os.fstat(root_descriptor)
        _candidate_require(
            (opened_root.st_dev, opened_root.st_ino) == root_identity,
            "artifact root changed before the finalized tree audit",
        )
        visit(root_descriptor, ())
    except OSError as error:
        raise CandidateEvidenceError(
            f"cannot safely walk finalized artifact tree {root}: {error}"
        ) from error
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)
    _candidate_require(
        observed == expected,
        "SHA256SUMS does not exactly cover the finalized artifact tree",
    )


def _leaderboard_entry_id(
    *,
    campaign: dict[str, Any],
    raw_manifest: dict[str, Any],
    source: dict[str, Any],
    artifact_root: Path,
    sha256sums_sha256: str,
) -> str:
    identity = {
        "campaign_id": campaign.get("campaign_id"),
        "run_id": campaign.get("run_id"),
        "candidate": campaign.get("candidate"),
        "source_identity_sha256": source["working_tree_identity_sha256"],
        "manifest_canonical_sha256": canonical_sha256(raw_manifest),
        "status": campaign.get("status"),
        "artifact_dir": str(artifact_root),
        "sha256sums_sha256": sha256sums_sha256,
    }
    return canonical_sha256(identity)


def _argv_value(argv: list[str], option: str) -> str:
    positions = [index for index, item in enumerate(argv) if item == option]
    _candidate_require(len(positions) == 1 and positions[0] + 1 < len(argv), f"benchmark argv {option} invalid")
    return argv[positions[0] + 1]


def _validate_source_ref_on_disk(source: dict[str, Any]) -> None:
    path = Path(source["worktree_realpath"])
    try:
        resolved = path.resolve(strict=True)
        metadata = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise CandidateEvidenceError(f"source worktree unavailable: {path}: {error}") from error
    _candidate_require(str(resolved) == str(path), "source worktree realpath differs")
    _candidate_require(stat.S_ISDIR(metadata.st_mode), "source worktree is not a directory")
    _candidate_require(metadata.st_dev == source["worktree_device"] and metadata.st_ino == source["worktree_inode"], "source worktree device/inode differs")


def _audit_control_plane_block(
    *,
    manifest: dict[str, Any],
    binding_state: dict[str, Any],
    candidate: dict[str, Any] | None,
    block: dict[str, Any],
) -> dict[str, Any]:
    owner = candidate
    if block["role"] == "parent":
        owner = manifest["candidates"][0]
    _candidate_require(owner is not None, "candidate control block lacks an owner")
    context = _resolve_block(
        manifest=manifest,
        binding_state=binding_state,
        candidate=owner,
        block=block,
    )
    coordinator = context["coordinator"]
    row = {
        "global_ordinal": block["global_ordinal"],
        "role": block["role"],
        "variant_id": coordinator["variant_id"],
        "terminal_status": (
            "success" if context["campaign_complete"] else "failure"
        ),
        "monotonic_start_ns": coordinator["monotonic_start_ns"],
        "monotonic_end_ns": coordinator["monotonic_end_ns"],
        "coordinator_record_raw_sha256": block[
            "coordinator_record_raw_sha256"
        ],
        "campaign_id": context["campaign_id"],
        "campaign_run_id": context["campaign_run_id"],
    }
    if context["campaign_complete"]:
        evidence = context["evidence"]
        row.update(
            {
                "report_sha256": context["report_sha256"],
                "report_run_id": evidence["run_id"],
                "sample_intervals_ns": evidence["sample_intervals_ns"],
            }
        )
    return row


def _audit_round_control_plane(
    manifest: dict[str, Any], binding_state: dict[str, Any]
) -> dict[str, Any]:
    specs: list[tuple[dict[str, Any] | None, dict[str, Any]]] = [
        (None, block) for block in manifest["parent_blocks"]
    ]
    specs.extend(
        (candidate, block)
        for candidate in manifest["candidates"]
        for block in candidate["benchmark_blocks"]
    )
    rows = [
        _audit_control_plane_block(
            manifest=manifest,
            binding_state=binding_state,
            candidate=candidate,
            block=block,
        )
        for candidate, block in specs
    ]
    ordered = sorted(rows, key=lambda row: int(row["global_ordinal"]))
    _candidate_require(
        [row["global_ordinal"] for row in ordered] == list(range(len(ordered))),
        "control-plane global ordinals are not dense",
    )
    _candidate_require(
        all(
            int(ordered[index]["monotonic_start_ns"])
            >= int(ordered[index - 1]["monotonic_end_ns"])
            for index in range(1, len(ordered))
        ),
        "control-plane blocks overlap or execute out of global order",
    )
    for field in ("campaign_id", "campaign_run_id"):
        values = [row[field] for row in ordered]
        _candidate_require(
            len(values) == len(set(values)),
            f"control-plane reuses a {field}",
        )
    successful = [row for row in ordered if row["terminal_status"] == "success"]
    for field in ("report_sha256", "report_run_id"):
        values = [row[field] for row in successful]
        _candidate_require(
            len(values) == len(set(values)),
            f"control-plane reuses a successful-block {field}",
        )
    sample_intervals = [
        tuple(interval)
        for row in successful
        for interval in row["sample_intervals_ns"]
    ]
    _candidate_require(
        len(sample_intervals) == len(set(sample_intervals))
        and all(
            sample_intervals[index][0] >= sample_intervals[index - 1][1]
            for index in range(1, len(sample_intervals))
        ),
        "successful-block raw samples repeat, overlap, or execute out of global order",
    )
    _verify_code_snapshots(binding_state)
    source_manifest = binding_state["manifest"]
    return {
        "passed": True,
        "block_count": len(ordered),
        "global_ordinals": [row["global_ordinal"] for row in ordered],
        "terminal_failures": sum(row["terminal_status"] == "failure" for row in ordered),
        "source_round_manifest_raw_sha256": binding_state["binding"]["manifest_raw_sha256"],
        "source_round_plan_raw_sha256": binding_state["binding"]["plan_raw_sha256"],
        "source_round_coordinator_sha256": binding_state["binding"]["coordinator_code_sha256"],
        "source_round_executor_sha256": binding_state["binding"][
            "source_round_executor_sha256"
        ],
        "control_commit": source_manifest["coordinator"]["control_commit"],
        "parent_commit": source_manifest["parent"]["source_commit"],
    }


def _resolve_block(
    *,
    manifest: dict[str, Any],
    binding_state: dict[str, Any],
    candidate: dict[str, Any],
    block: dict[str, Any],
) -> dict[str, Any]:
    measurement = manifest["measurement_contract"]
    source = manifest["parent_ref"] if block["role"] == "parent" else candidate["source_ref"]
    expected_campaign_candidate = manifest["parent_campaign_candidate"] if block["role"] == "parent" else {key: candidate[key] for key in _CAMPAIGN_CANDIDATE_FIELDS}
    plan_block = binding_state["plan"]["blocks"][block["global_ordinal"]]
    source_artifact_root = Path(
        binding_state["manifest"]["coordinator"]["artifact_root"]
    )
    expected_root = source_artifact_root / plan_block["artifact_dir"]
    expected_coordinator_path = (
        source_artifact_root
        / "coordinator"
        / f"block-{block['global_ordinal']:02d}.json"
    )
    _candidate_require(
        plan_block["ordinal"] == block["global_ordinal"]
        and plan_block["role"] == block["role"]
        and plan_block["variant_id"] == source["ref_id"]
        and plan_block["worktree"] == source["worktree_realpath"]
        and plan_block["source_commit"] == source["commit"],
        "round block differs from its source-round plan slot",
    )
    _candidate_require(
        Path(block["artifact_root_realpath"]) == expected_root,
        "campaign artifact root differs from the source-round plan",
    )
    _candidate_require(
        Path(block["coordinator_record"]) == expected_coordinator_path,
        "coordinator record path differs from the source-round control path",
    )
    _validate_source_ref_on_disk(source)
    root = Path(block["artifact_root_realpath"])
    try:
        resolved_root = root.resolve(strict=True)
        root_stat = os.stat(resolved_root, follow_symlinks=False)
    except OSError as error:
        raise CandidateEvidenceError(f"campaign artifact root unavailable: {root}: {error}") from error
    _candidate_require(str(resolved_root) == str(root), "artifact root realpath differs")
    _candidate_require(
        stat.S_ISDIR(root_stat.st_mode)
        and not root_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH),
        "artifact root is not a private regular directory",
    )
    _candidate_require(root_stat.st_dev == block["artifact_root_device"] and root_stat.st_ino == block["artifact_root_inode"], "artifact root device/inode differs")
    root_identity = (root_stat.st_dev, root_stat.st_ino)

    campaign, result_sha, _ = _read_json_beneath(
        root, "result.json", root_identity=root_identity
    )
    _candidate_require(result_sha == block["campaign_result_raw_sha256"], "campaign result hash differs")
    raw_manifest, raw_manifest_sha, _ = _read_json_beneath(
        root, "manifest.raw.json", root_identity=root_identity
    )
    _candidate_require(raw_manifest_sha == block["campaign_manifest_raw_sha256"], "campaign raw manifest hash differs")
    saved_manifest, saved_manifest_sha, _ = _read_json_beneath(
        root, "manifest.json", root_identity=root_identity
    )
    _candidate_require(raw_manifest == saved_manifest, "saved campaign manifest differs from raw input")
    try:
        validate_campaign_manifest(raw_manifest)
    except CampaignManifestError as error:
        raise CandidateEvidenceError(f"campaign manifest fails production schema: {error}") from error
    _candidate_require(raw_manifest["candidate"] == expected_campaign_candidate, "campaign candidate identity differs")
    _candidate_require(canonical_sha256(raw_manifest["frozen_contract"]) == manifest["frozen_contract_sha256"], "campaign frozen contract differs")
    _candidate_require(canonical_sha256(raw_manifest["resources"]["expected_gpu_index_uuid_mapping"]) == measurement["gpu_mapping_sha256"], "campaign GPU mapping differs")

    _candidate_require(campaign.get("schema_version") == 1, "campaign result schema differs")
    _candidate_require(campaign.get("candidate") == expected_campaign_candidate, "campaign result candidate differs")
    _candidate_require(
        campaign.get("run_id") == plan_block["run_id"],
        "campaign run_id differs from the source-round plan",
    )
    _candidate_require(
        campaign.get("performance_claim_allowed") is False,
        "campaign result must not self-authorize a performance claim",
    )
    _candidate_require(campaign.get("manifest_raw_sha256") == raw_manifest_sha, "campaign manifest hash differs")
    _candidate_require(campaign.get("manifest_canonical_sha256") == canonical_sha256(raw_manifest), "campaign canonical manifest hash differs")
    _candidate_require(campaign.get("runner_sha256") == measurement["campaign_runner_sha256"], "campaign runner differs")
    _candidate_require(campaign.get("schema_sha256") == measurement["campaign_schema_sha256"], "campaign schema differs")
    pre_source = _source_identity(campaign.get("source_identity"), "campaign source", source)
    post_source = _source_identity(campaign.get("source_identity_post"), "campaign post source", source)
    _candidate_require(pre_source == post_source and campaign.get("source_drift") is False, "campaign source drifted")
    stages = _stages_by_id(campaign)
    plans = raw_manifest["stages"]
    _candidate_require(
        [(item.get("id"), item.get("kind"), item.get("resource")) for item in campaign["stages"]]
        == [(item["id"], item["kind"], item["resource"]) for item in plans],
        "campaign stage results differ from frozen plan",
    )
    for plan_item in plans:
        result_item = stages[plan_item["id"]]
        if result_item.get("status") == "passed":
            _candidate_require(
                all(
                    stages[dependency].get("status") == "passed"
                    for dependency in plan_item["depends_on"]
                ),
                f"passed stage has a non-passing dependency: {plan_item['id']}",
            )
    gates = {kind: _gate_summary(stages, plans, kind) for kind in ("compile", "correctness", "sanitizer")}

    finalized, finalized_sha, _ = _read_json_beneath(
        root, "FINALIZED.json", root_identity=root_identity
    )
    _candidate_require(finalized_sha == block["finalized_raw_sha256"], "FINALIZED hash differs")
    _candidate_require(set(finalized) == _FINALIZED_FIELDS, "FINALIZED fields differ")
    _candidate_require(finalized.get("schema_version") == 1, "FINALIZED schema differs")
    _candidate_require(finalized.get("campaign_id") == campaign.get("campaign_id") and finalized.get("run_id") == campaign.get("run_id"), "FINALIZED campaign identity differs")
    _candidate_require(finalized.get("campaign_status") == campaign.get("status"), "FINALIZED campaign status differs")
    _candidate_require(finalized.get("claim_scope") == campaign.get("claim_scope"), "FINALIZED claim_scope differs")
    _candidate_require(finalized.get("result_sha256") == result_sha, "FINALIZED result hash differs")
    _candidate_require(finalized.get("source_identity_sha256") == source["working_tree_identity_sha256"], "FINALIZED source identity differs")
    _candidate_require(finalized.get("manifest_raw_sha256") == raw_manifest_sha, "FINALIZED manifest hash differs")
    _candidate_require(isinstance(finalized.get("leaderboard_entry_id"), str) and bool(_SHA256.fullmatch(finalized["leaderboard_entry_id"])), "FINALIZED leaderboard id invalid")
    sums_raw, sums_sha = _read_bytes_beneath(
        root, "SHA256SUMS", root_identity=root_identity
    )
    _candidate_require(finalized.get("sha256sums_sha256") == sums_sha, "FINALIZED SHA256SUMS hash differs")
    sums = _parse_sha256sums(sums_raw)
    _audit_artifact_tree(
        root,
        sums,
        root_identity=(block["artifact_root_device"], block["artifact_root_inode"]),
    )
    _candidate_require(sums.get("result.json") == result_sha, "SHA256SUMS result hash differs")
    _candidate_require(sums.get("manifest.raw.json") == raw_manifest_sha, "SHA256SUMS manifest hash differs")
    _candidate_require(
        sums.get("manifest.json") == saved_manifest_sha,
        "SHA256SUMS saved manifest hash differs",
    )
    toolchain, toolchain_raw_sha, _ = _read_json_beneath(
        root,
        "environment/toolchain.json",
        root_identity=root_identity,
    )
    _candidate_require(
        sums.get("environment/toolchain.json") == toolchain_raw_sha,
        "SHA256SUMS toolchain hash differs",
    )
    _candidate_require(
        canonical_sha256(_normalized_toolchain(toolchain))
        == measurement["toolchain_sha256"],
        "campaign normalized toolchain differs",
    )
    _candidate_require(
        finalized.get("leaderboard_entry_id")
        == _leaderboard_entry_id(
            campaign=campaign,
            raw_manifest=raw_manifest,
            source=source,
            artifact_root=root,
            sha256sums_sha256=sums_sha,
        ),
        "FINALIZED leaderboard entry id differs",
    )

    complete = (
        campaign.get("status") == "complete"
        and campaign.get("claim_scope") == "local_candidate_gate"
        and finalized.get("terminal_label") == "PASS"
        and finalized.get("terminal_exit_code") == 0
        and finalized.get("round_evaluation_allowed") is True
    )
    failed = (
        campaign.get("status") != "complete"
        and finalized.get("terminal_label") == "FAIL"
        and type(finalized.get("terminal_exit_code")) is int
        and finalized["terminal_exit_code"] != 0
        and finalized.get("round_evaluation_allowed") is False
    )
    _candidate_require(
        complete or failed,
        "block lacks an unambiguous FINALIZED terminal status",
    )
    coordinator_path = Path(block["coordinator_record"])
    try:
        resolved_coordinator_path = coordinator_path.resolve(strict=True)
        coordinator_stat = os.stat(resolved_coordinator_path, follow_symlinks=False)
    except OSError as error:
        raise CandidateEvidenceError(
            f"coordinator record is unavailable: {coordinator_path}: {error}"
        ) from error
    _candidate_require(
        resolved_coordinator_path == coordinator_path
        and stat.S_ISREG(coordinator_stat.st_mode)
        and coordinator_stat.st_nlink == 1,
        "coordinator record path is linked or non-regular",
    )
    coordinator, coordinator_sha = _read_strict_json(coordinator_path)
    _candidate_require(coordinator_sha == block["coordinator_record_raw_sha256"], "coordinator record hash differs")
    _candidate_require(set(coordinator) == _COORDINATOR_FIELDS, "coordinator record fields differ")
    expected_coordinator = {
        "schema_version": 1,
        "round_id": manifest["round_id"],
        "variant_id": source["ref_id"],
        "source_round_manifest_raw_sha256": manifest["source_round_binding"]["manifest_raw_sha256"],
        "source_round_plan_raw_sha256": manifest["source_round_binding"]["plan_raw_sha256"],
        "source_round_plan_ordinal": block["global_ordinal"],
        "source_round_run_id": plan_block["run_id"],
        "source_round_coordinator_sha256": manifest["source_round_binding"]["coordinator_code_sha256"],
        "source_round_executor_sha256": manifest["source_round_binding"][
            "source_round_executor_sha256"
        ],
        "source_round_control_commit": manifest["source_round_binding"]["control_commit"],
        "source_round_parent_commit": manifest["source_round_binding"]["parent_commit"],
        "ordinal": block["ordinal"],
        "global_ordinal": block["global_ordinal"],
        "role": block["role"],
        "boot_id": measurement["coordinator_boot_id"],
        "lock_identity_sha256": measurement["coordinator_lock_identity_sha256"],
        "artifact_root_realpath": block["artifact_root_realpath"],
        "artifact_root_device": block["artifact_root_device"],
        "artifact_root_inode": block["artifact_root_inode"],
        "worktree_realpath": source["worktree_realpath"],
        "worktree_device": source["worktree_device"],
        "worktree_inode": source["worktree_inode"],
        "source_commit": source["commit"],
        "source_identity_sha256": source["working_tree_identity_sha256"],
        "campaign_result_raw_sha256": result_sha,
        "campaign_finalized_raw_sha256": finalized_sha,
        "gpu_mapping_sha256": measurement["gpu_mapping_sha256"],
    }
    for field, expected in expected_coordinator.items():
        _candidate_require(coordinator.get(field) == expected, f"coordinator {field} differs")
    start = coordinator.get("monotonic_start_ns")
    end = coordinator.get("monotonic_end_ns")
    _candidate_require(type(start) is int and type(end) is int and 0 <= start < end <= 2**63 - 1, "coordinator monotonic interval invalid")

    context = {
        "artifact_root": str(root),
        "campaign_id": campaign.get("campaign_id"),
        "campaign_run_id": campaign.get("run_id"),
        "source_ref": source,
        "gates": gates,
        "campaign_complete": complete,
        "coordinator": coordinator,
        "result_sha256": result_sha,
        "finalized_sha256": finalized_sha,
    }
    if not complete:
        _candidate_require(coordinator.get("terminal_evidence_complete") is True, "failed block lacks terminal evidence")
        _candidate_require(coordinator.get("benchmark_report_sha256") is None, "failed block claims benchmark report")
        return context
    _candidate_require(coordinator.get("terminal_evidence_complete") is True, "block terminal evidence incomplete")
    _candidate_require(
        campaign.get("gpu_preflight_reasons") == []
        and campaign.get("campaign_preflight_reasons") == [],
        "complete campaign retains a preflight failure",
    )
    for plan_item in plans:
        result_item = stages[plan_item["id"]]
        if not plan_item.get("enabled", True) or plan_item["resource"] == "multinode":
            continue
        result_attempts = result_item.get("attempts")
        _candidate_require(
            result_item.get("status") == "passed"
            and isinstance(result_attempts, list)
            and bool(result_attempts)
            and all(_attempt_passed(item) for item in result_attempts),
            f"complete campaign has a non-passing stage: {plan_item['id']}",
        )
    for gate in gates.values():
        _candidate_require(gate["passed"], "complete campaign contains a failed formal gate")
    enabled_benchmarks = [plan for plan in plans if plan["kind"] == "benchmark" and plan.get("enabled", True)]
    _candidate_require(len(enabled_benchmarks) == 1 and enabled_benchmarks[0]["id"] == block["stage_id"], "single-block campaign benchmark plan differs")
    _candidate_require(all(not plan.get("enabled", True) for plan in plans if str(plan["kind"]).startswith("profile_")), "formal block must be profiler-free")
    plan = enabled_benchmarks[0]
    _candidate_require(plan.get("repeat", 1) == 1 and block["artifact"] in plan.get("artifacts", []), "benchmark plan allows cherry-picking")
    argv = plan["argv"]
    expected_options = {
        "--stage": str(measurement["workload"]["stage"]),
        "--case-name": str(measurement["workload"]["case_name"]),
        "--hop-mode": str(measurement["algorithm"]["hop_mode"]),
        "--two-hop-threshold-percent": str(
            measurement["algorithm"]["two_hop_threshold_percent"]
        ),
        "--max-two-hop-percent": str(measurement["algorithm"]["max_two_hop_percent"]),
        "--hop-penalty-percent": str(
            measurement["algorithm"]["hop_penalty_percent"]
        ),
        "--warmup-iters": "10",
        "--steady-iters": "100",
        "--json-out": f"{{stage_dir}}/{block['artifact']}",
    }
    for option, expected in expected_options.items():
        _candidate_require(_argv_value(argv, option) == expected, f"benchmark plan {option} differs")
    stage = stages[block["stage_id"]]
    attempts = stage.get("attempts")
    _candidate_require(stage.get("status") == "passed" and isinstance(attempts, list) and len(attempts) == 1 and _attempt_passed(attempts[0]), "benchmark attempt did not pass exactly once")
    attempt = attempts[0]
    declared = attempt.get("declared_artifacts")
    _candidate_require(isinstance(declared, list), "benchmark declared artifacts missing")
    matches = [item for item in declared if isinstance(item, dict) and item.get("path") == block["artifact"]]
    _candidate_require(len(matches) == 1, "benchmark artifact declaration ambiguous")
    report_relative = f"stages/{block['stage_id']}/attempt-01/{block['artifact']}"
    planned_report_relative = Path(plan_block["raw_benchmark_artifact"]).relative_to(
        plan_block["artifact_dir"]
    ).as_posix()
    planned_jit_relative = Path(plan_block["jit_dir"]).relative_to(
        plan_block["artifact_dir"]
    ).as_posix()
    _candidate_require(
        block["stage_id"] == "bench-abba-a1"
        and report_relative == planned_report_relative
        and planned_jit_relative
        == f"stages/{block['stage_id']}/attempt-01/jit",
        "benchmark report/JIT path differs from the source-round plan",
    )
    report, report_sha, report_bytes = _read_json_beneath(
        root,
        report_relative,
        root_identity=root_identity,
    )
    _candidate_require(matches[0].get("sha256") == report_sha and matches[0].get("bytes") == report_bytes, "benchmark artifact hash/bytes differ")
    _candidate_require(sums.get(report_relative) == report_sha, "SHA256SUMS report hash differs")
    _candidate_require(coordinator.get("benchmark_report_sha256") == report_sha, "coordinator report hash differs")
    evidence = _validate_c100_report(
        report,
        measurement,
        source,
        expected_jit_root=root / planned_jit_relative,
        expected_report_path=root / planned_report_relative,
    )
    sample_intervals = evidence["sample_intervals_ns"]
    _candidate_require(
        sample_intervals[0][0] >= start and sample_intervals[-1][1] <= end,
        "C100 raw samples fall outside the coordinator monotonic interval",
    )
    started = attempt.get("started_utc")
    finished = attempt.get("finished_utc")
    for value, name in ((started, "started_utc"), (finished, "finished_utc")):
        _candidate_require(isinstance(value, str), f"benchmark {name} missing")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise CandidateEvidenceError(f"benchmark {name} invalid") from error
        _candidate_require(parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed), f"benchmark {name} is not UTC")
    _candidate_require(datetime.fromisoformat(started) < datetime.fromisoformat(finished), "benchmark UTC interval inverted")
    return {
        **context,
        "report_path": str(root / report_relative),
        "report_sha256": report_sha,
        "started_utc": started,
        "finished_utc": finished,
        "evidence": evidence,
    }


def _analyze_candidate(
    manifest: dict[str, Any],
    binding_state: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    contexts = []
    aggregate_gates: dict[str, dict[str, Any]] = {
        kind: {"passed": True, "reason": None, "stages": []}
        for kind in ("compile", "correctness", "sanitizer")
    }
    parents = manifest["parent_blocks"]
    candidates = candidate["benchmark_blocks"]
    projected_blocks = (
        parents[0],
        candidates[0],
        candidates[1],
        parents[1],
        candidates[2],
        parents[2],
        parents[3],
        candidates[3],
    )
    for projection_ordinal, block in enumerate(projected_blocks):
        context = _resolve_block(
            manifest=manifest,
            binding_state=binding_state,
            candidate=candidate,
            block=block,
        )
        for kind, gate in context["gates"].items():
            aggregate_gates[kind]["passed"] &= gate["passed"]
            aggregate_gates[kind]["stages"].append(
                {
                    "projection_ordinal": projection_ordinal,
                    "global_ordinal": block["global_ordinal"],
                    **gate,
                }
            )
            if not gate["passed"]:
                aggregate_gates[kind]["reason"] = gate["reason"]
        contexts.append(context)
        directly_failed = [
            kind
            for kind in ("compile", "correctness", "sanitizer")
            if any(
                stage.get("status") == "failed"
                for block_gate in aggregate_gates[kind]["stages"]
                for stage in block_gate["stages"]
            )
        ]
        if directly_failed:
            kind = directly_failed[0]
            return aggregate_gates, {
                "status": f"{kind}_failed",
                "failure_reason": aggregate_gates[kind]["reason"],
            }
        incomplete_gates = [
            kind
            for kind in ("compile", "correctness", "sanitizer")
            if not aggregate_gates[kind]["passed"]
        ]
        if incomplete_gates:
            kind = incomplete_gates[0]
            return aggregate_gates, {
                "status": f"{kind}_failed",
                "failure_reason": aggregate_gates[kind]["reason"],
            }
        if not context["campaign_complete"]:
            return aggregate_gates, {"status": "campaign_failed", "failure_reason": "campaign lacks a successful FINALIZED terminal commit"}

    roots = [context["artifact_root"] for context in contexts]
    campaigns = [(context["campaign_id"], context["campaign_run_id"]) for context in contexts]
    report_paths = [context["report_path"] for context in contexts]
    report_hashes = [context["report_sha256"] for context in contexts]
    run_ids = [context["evidence"]["run_id"] for context in contexts]
    for values, name in ((roots, "artifact roots"), (campaigns, "campaign runs"), (report_paths, "report paths"), (report_hashes, "report hashes"), (run_ids, "C100 run_ids")):
        _candidate_require(len(values) == len(set(values)), f"candidate reuses {name}")
    intervals = [(context["coordinator"]["monotonic_start_ns"], context["coordinator"]["monotonic_end_ns"]) for context in contexts]
    _candidate_require(all(intervals[index][0] >= intervals[index - 1][1] for index in range(1, len(intervals))), "actual P/C/C/P/C/P/P/C blocks overlap or execute out of order")
    _candidate_require(tuple(context["coordinator"]["role"] for context in contexts) == _ABBA_BAAB, "actual coordinator role order differs")

    blocks = []
    for ordinal, context in enumerate(contexts):
        blocks.append(
            {
                "ordinal": ordinal,
                "global_ordinal": context["coordinator"]["global_ordinal"],
                "role": context["coordinator"]["role"],
                "artifact_root": context["artifact_root"],
                "campaign_id": context["campaign_id"],
                "campaign_run_id": context["campaign_run_id"],
                "report_path": context["report_path"],
                "report_sha256": context["report_sha256"],
                "coordinator_monotonic_start_ns": intervals[ordinal][0],
                "coordinator_monotonic_end_ns": intervals[ordinal][1],
                **context["evidence"],
            }
        )
    parent_blocks = [block for block in blocks if block["role"] == "parent"]
    candidate_blocks = [block for block in blocks if block["role"] == "candidate"]
    parent_medians = [float(block["stats"]["median"]) for block in parent_blocks]
    candidate_medians = [float(block["stats"]["median"]) for block in candidate_blocks]
    parent_pooled = [value for block in parent_blocks for value in block["samples_ns"]]
    candidate_pooled = [value for block in candidate_blocks for value in block["samples_ns"]]
    pairs = []
    for pair_index in range(4):
        pair = blocks[pair_index * 2 : pair_index * 2 + 2]
        by_role = {block["role"]: block for block in pair}
        _candidate_require(set(by_role) == {"parent", "candidate"}, "adjacent blocks are not a P/C pair")
        parent_median = float(by_role["parent"]["stats"]["median"])
        candidate_median = float(by_role["candidate"]["stats"]["median"])
        pairs.append(
            {
                "pair_index": pair_index,
                "block_ordinals": [block["ordinal"] for block in pair],
                "parent_median_ns": parent_median,
                "candidate_median_ns": candidate_median,
                "speedup_parent_over_candidate": parent_median / candidate_median,
                "improvement_fraction": 1 - candidate_median / parent_median,
            }
        )
    improvements = [float(pair["improvement_fraction"]) for pair in pairs]
    parent_center = float(statistics.median(parent_medians))
    candidate_center = float(statistics.median(candidate_medians))
    parent_relative_mad = _scaled_mad(parent_medians) / parent_center
    candidate_relative_mad = _scaled_mad(candidate_medians) / candidate_center
    pair_mad = _scaled_mad(improvements)
    policy = FORMAL_NOISE_POLICY
    robust_noise = max(float(policy["relative_floor"]), parent_relative_mad, candidate_relative_mad, pair_mad)
    required = max(float(policy["minimum_improvement_fraction"]), float(policy["mad_multiplier"]) * robust_noise)
    median_improvement = float(statistics.median(improvements))
    winning_pairs = sum(item > float(policy["relative_floor"]) for item in improvements)
    parent_stats = _stats(parent_pooled)
    candidate_stats = _stats(candidate_pooled)
    cv_ok = max(float(parent_stats["cv_population"]), float(candidate_stats["cv_population"])) <= float(policy["maximum_pooled_cv"])
    significant = median_improvement > required and winning_pairs >= int(policy["minimum_winning_pairs"]) and cv_ok
    return aggregate_gates, {
        "status": "significant" if significant else "not_significant",
        "failure_reason": None,
        "source_ref": candidate["source_ref"],
        "blocks": blocks,
        "pooled": {"parent": parent_stats, "candidate": candidate_stats},
        "block_medians_ns": {"parent": parent_medians, "candidate": candidate_medians},
        "pairs": pairs,
        "noise": {
            "relative_floor": policy["relative_floor"],
            "parent_relative_scaled_mad": parent_relative_mad,
            "candidate_relative_scaled_mad": candidate_relative_mad,
            "paired_improvement_scaled_mad": pair_mad,
            "robust_noise_fraction": robust_noise,
            "required_improvement_fraction": required,
            "median_paired_improvement_fraction": median_improvement,
            "winning_pairs": winning_pairs,
            "pooled_cv_ok": cv_ok,
            "significant": significant,
        },
        "aggregate_speedup_parent_over_candidate": parent_center / candidate_center,
    }


def evaluate_round(manifest_path: Path) -> dict[str, Any]:
    manifest_path = _absolute_without_resolving(manifest_path)
    manifest, manifest_raw_sha = load_round_manifest(manifest_path)
    control_plane_failure: str | None = None
    binding_state: dict[str, Any] | None = None
    try:
        binding_state = _load_source_round_binding(manifest)
        control_plane_evidence = _audit_round_control_plane(manifest, binding_state)
    except (
        CandidateEvidenceError,
        RoundError,
        OSError,
        RuntimeError,
        RecursionError,
        OverflowError,
        KeyError,
        TypeError,
        IndexError,
        ValueError,
    ) as error:
        control_plane_failure = "round_control_plane_invalid"
        control_plane_evidence = {
            "passed": False,
            "failure_reason": str(error),
        }
    verdicts = []
    for candidate in manifest["candidates"]:
        base = {key: candidate[key] for key in _CAMPAIGN_CANDIDATE_FIELDS | {"frozen_contract_sha256", "source_ref"}}
        try:
            _candidate_require(
                binding_state is not None,
                "source-round binding could not be reloaded for candidate analysis",
            )
            gates, analysis = _analyze_candidate(manifest, binding_state, candidate)
            verdicts.append({**base, "gates": gates, "analysis": analysis})
        except (
            CandidateEvidenceError,
            RoundError,
            CampaignManifestError,
            OSError,
            RuntimeError,
            RecursionError,
            OverflowError,
            KeyError,
            TypeError,
            IndexError,
            ValueError,
        ) as error:
            verdicts.append(
                {
                    **base,
                    "gates": {
                        "compile": {"passed": False, "reason": "artifact invalid", "stages": []},
                        "correctness": {"passed": False, "reason": "not evaluated", "stages": []},
                        "sanitizer": {"passed": False, "reason": "not evaluated", "stages": []},
                    },
                    "analysis": {"status": "evidence_invalid", "failure_reason": str(error)},
                }
            )

    fully_measured = [row for row in verdicts if row["analysis"]["status"] in {"significant", "not_significant"}]
    global_failure: str | None = control_plane_failure
    global_evidence: dict[str, Any] = {
        "control_plane_audit": control_plane_evidence
    }
    if fully_measured:
        projected_parent_blocks = [
            block
            for block in fully_measured[0]["analysis"]["blocks"]
            if block["role"] == "parent"
        ]
        parent_medians = [
            float(block["stats"]["median"]) for block in projected_parent_blocks
        ]
        parent_center = float(statistics.median(parent_medians))
        parent_drift = (max(parent_medians) - min(parent_medians)) / parent_center
        global_evidence["parent_baseline_medians_ns"] = parent_medians
        global_evidence["parent_baseline_drift_fraction"] = parent_drift
        global_evidence["parent_baseline_drift_limit_fraction"] = FORMAL_NOISE_POLICY["maximum_parent_baseline_drift_fraction"]
        if parent_drift > float(FORMAL_NOISE_POLICY["maximum_parent_baseline_drift_fraction"]):
            global_failure = "parent_baseline_drift_exceeded"
        all_blocks = [block for row in fully_measured for block in row["analysis"]["blocks"]]
        parent_global_ordinals = {
            int(block["global_ordinal"]) for block in manifest["parent_blocks"]
        }
        blocks_by_global_ordinal: dict[int, dict[str, Any]] = {}
        for block in all_blocks:
            global_ordinal = int(block["global_ordinal"])
            previous = blocks_by_global_ordinal.get(global_ordinal)
            if previous is None:
                blocks_by_global_ordinal[global_ordinal] = block
                continue
            if (
                global_ordinal not in parent_global_ordinals
                or block["role"] != "parent"
                or previous != block
            ):
                global_failure = "shared_parent_evidence_mismatch"
        unique_blocks = list(blocks_by_global_ordinal.values())
        for field in ("artifact_root", "campaign_run_id", "report_path", "report_sha256", "run_id"):
            values = [block[field] for block in unique_blocks]
            if len(values) != len(set(values)):
                global_failure = f"round_reuses_{field}"
        ordered = sorted(unique_blocks, key=lambda block: int(block["global_ordinal"]))
        if any(
            int(ordered[index]["coordinator_monotonic_start_ns"])
            < int(ordered[index - 1]["coordinator_monotonic_end_ns"])
            for index in range(1, len(ordered))
        ):
            global_failure = "round_global_chronology_overlaps"
        expected_measured_blocks = 4 + 4 * len(fully_measured)
        if len(unique_blocks) != expected_measured_blocks:
            global_failure = "round_measured_block_count_differs"
        global_evidence["unique_measured_block_count"] = len(unique_blocks)
        global_evidence["expected_measured_block_count"] = expected_measured_blocks
        global_evidence["shared_parent_block_count"] = len(parent_global_ordinals)

    if binding_state is not None:
        try:
            _verify_code_snapshots(binding_state)
        except (RoundError, OSError, RuntimeError, TypeError, ValueError) as error:
            global_failure = "round_control_plane_invalid"
            global_evidence["control_plane_audit"] = {
                "passed": False,
                "failure_reason": str(error),
            }

    eligible = [row for row in verdicts if row["analysis"]["status"] == "significant"]
    winner = None
    leader_separation: dict[str, Any] = {"evaluated": False, "passed": None}
    if global_failure is None and eligible:
        eligible.sort(
            key=lambda row: (
                float(row["analysis"]["pooled"]["candidate"]["median"]),
                float(row["analysis"]["pooled"]["candidate"]["p95"]),
                str(row["id"]),
            )
        )
        if len(eligible) == 1:
            winner = eligible[0]
        else:
            fastest, runner_up = eligible[:2]
            fastest_median = float(fastest["analysis"]["pooled"]["candidate"]["median"])
            runner_up_median = float(runner_up["analysis"]["pooled"]["candidate"]["median"])
            separation = 1 - fastest_median / runner_up_median
            required = max(
                float(fastest["analysis"]["noise"]["required_improvement_fraction"]),
                float(runner_up["analysis"]["noise"]["required_improvement_fraction"]),
            )
            leader_separation = {
                "evaluated": True,
                "fastest_candidate_id": fastest["id"],
                "runner_up_candidate_id": runner_up["id"],
                "separation_fraction": separation,
                "required_separation_fraction": required,
                "passed": separation > required,
            }
            if separation > required:
                winner = fastest

    for verdict in verdicts:
        analysis = verdict["analysis"]
        status = analysis["status"]
        if status == "significant":
            if global_failure is not None:
                analysis["verdict"] = f"not_promoted_{global_failure}"
            elif winner is not None and verdict["id"] == winner["id"]:
                analysis["verdict"] = "promoted_absolute_fastest"
            elif winner is None:
                analysis["verdict"] = "not_promoted_leader_separation_within_noise"
            else:
                analysis["verdict"] = "significant_not_absolute_fastest"
        elif status == "not_significant":
            analysis["verdict"] = "not_promoted_noise_gate"
        elif status in {"compile_failed", "correctness_failed", "sanitizer_failed", "campaign_failed"}:
            analysis["verdict"] = f"not_promoted_{status}"
        else:
            analysis["verdict"] = "not_promoted_invalid_evidence"

    failure_statuses = {"compile_failed", "correctness_failed", "sanitizer_failed", "campaign_failed", "evidence_invalid"}
    failure_count = sum(row["analysis"]["status"] in failure_statuses for row in verdicts)
    parent = manifest["parent"]
    leader_after = parent if winner is None else winner["id"]
    return {
        "schema_version": ROUND_SCHEMA_VERSION,
        "round_id": manifest["round_id"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim_scope": "formal_source_version_cpu_evidence_aggregation",
        "comparison_axis": "source_version",
        "source_leader_promotion_allowed": (
            global_failure is None and winner is not None
        ),
        "algorithm_mode_diagnostic_is_not_source_evidence": True,
        "round_manifest_raw_sha256": manifest_raw_sha,
        "round_manifest_canonical_sha256": canonical_sha256(manifest),
        "round_evaluator_sha256": evaluator_sha256(),
        "source_round_binding": manifest["source_round_binding"],
        "frozen_contract_sha256": manifest["frozen_contract_sha256"],
        "measurement_contract_sha256": canonical_sha256(manifest["measurement_contract"]),
        "noise_policy": FORMAL_NOISE_POLICY,
        "noise_policy_sha256": canonical_sha256(FORMAL_NOISE_POLICY),
        "parent_ref": manifest["parent_ref"],
        "candidate_count": len(verdicts),
        "candidate_failure_count": failure_count,
        "candidates": verdicts,
        "round_wide_evidence": {"failure_reason": global_failure, **global_evidence},
        "leader_before": parent,
        "leader_after": leader_after,
        "leader_separation": leader_separation,
        "promotion": {
            "promoted": winner is not None,
            "candidate_id": None if winner is None else winner["id"],
            "rule": "coordinator-preflight-bound distinct-source finalized global 4+4N schedule; compile/correctness/sanitizer pass; paired gain exceeds fixed robust noise; parent blocks agree; absolute candidate median wins only beyond top-two noise",
        },
        "status": "complete_with_failures" if failure_count or global_failure else "complete",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if _absolute_without_resolving(args.output).exists():
        parser.error("output already exists; round evidence is immutable")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_round(args.manifest)
    _atomic_json(args.output, result, overwrite=False)
    print(
        "PASS formal source round: "
        f"status={result['status']} leader={result['leader_before']}"
        f"->{result['leader_after']} -> {_absolute_without_resolving(args.output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
