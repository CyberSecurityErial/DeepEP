#!/usr/bin/env python3
"""Fail-closed live writer for one formal CUDA source-version round.

The default mode is check-only and never launches a child.  Live execution
requires both ``--live`` and an exact ``--confirm-round-id``.  In live mode one
inherited flock lease covers every block, all runner descendants, immutable
coordinator records, the evaluator-ready round manifest, and the final fsync.
This parent remains stdlib-only and never imports Torch or DeepEP.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import rail_balance_campaign_round as round_evaluator
import rail_balance_campaign_schema as campaign_schema
import rail_balance_source_round_coordinator as source_coordinator
import run_rail_balance_hop_campaign as campaign_runner


SCHEMA_VERSION = 1
SOURCE_ROUND_EXECUTOR_PATH = "tests/elastic/rail_balance_source_round_executor.py"
RUNNER_PATH = "tests/elastic/run_rail_balance_hop_campaign.py"
SCHEMA_PATH = "tests/elastic/rail_balance_campaign_schema.py"
EVALUATOR_PATH = "tests/elastic/rail_balance_campaign_round.py"
BENCHMARK_PATH = "tests/elastic/bench_rail_balance_hybrid_lsa.py"
CAMPAIGN_TEMPLATE_PATH = "tests/elastic/experiments/hop_local8_sm90_v1.json"
FIXED_BENCHMARK_STAGE = "bench-abba-a1"
FIXED_VNODE_GATES = ("gpu-vnode-one-hop", "gpu-vnode-adaptive")
FORMAL_C100_CASES = {
    "c100_volume_h256",
    "c100_volume_h7168",
    *{
        f"c100_matrix_{pattern}_h{hidden}"
        for pattern in ("fanout", "fanin", "mesh", "rot1", "rot4")
        for hidden in (256, 7168)
    },
}
PINNED_PYTHON = "/home/chen/.cache/deepep-sjlgpt/bin/python"
BLOCK_ARTIFACT_BUDGET_BYTES = 750_000_000
HOME_EMERGENCY_RESERVE_BYTES = 5_000_000_000
ROUND_FINALIZATION_RESERVE_BYTES = 20_000_000
ROUND_MANIFEST_NAME = "formal-round-manifest.json"
ROUND_TERMINAL_NAME = "SOURCE_ROUND_FINALIZED.json"
MAX_JSON_BYTES = 64 * 1024 * 1024
LOCK_RECORD_MAX_BYTES = 16 * 1024
LOCK_PROTOCOL = "deepep-source-round-flock-lease-v1"
LOCK_PATH = source_coordinator.SOURCE_ROUND_LOCK
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_LEASE_FIELDS = {
    "schema_version",
    "protocol",
    "round_id",
    "plan_raw_sha256",
    "executor_pid",
    "executor_code_sha256",
    "nonce_sha256",
    "boot_id",
    "lock_path",
    "lock_device",
    "lock_inode",
    "shared_offset",
    "artifact_root",
    "artifact_root_device",
    "artifact_root_inode",
    "output_dir",
    "run_id",
    "campaign_manifest_path",
    "campaign_manifest_raw_sha256",
    "round_lock_identity_sha256",
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
_SOURCE_IDENTITY_REQUIRED_FIELDS = {
    "commit",
    "dirty",
    "status_porcelain",
    "tracked_diff_sha256",
    "untracked_tree_sha256",
    "untracked_files",
    "working_tree_identity_sha256",
}


class ExecutorError(RuntimeError):
    """The live round cannot continue without weakening its evidence."""


class RoundBlocked(ExecutorError):
    def __init__(self, status: str, reasons: list[str]) -> None:
        super().__init__(f"{status}: {', '.join(reasons)}")
        self.status = status
        self.reasons = reasons


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutorError(message)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ExecutorError(f"value is not strict JSON: {error}") from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, maximum_bytes: int = 512 * 1024 * 1024) -> str:
    return _sha256_bytes(_read_regular_no_follow(path, maximum_bytes))


def _sha256_open_file(descriptor: int, maximum_bytes: int) -> str:
    before = os.fstat(descriptor)
    _require(
        stat.S_ISREG(before.st_mode)
        and before.st_uid == os.geteuid()
        and before.st_nlink == 1
        and before.st_size <= maximum_bytes,
        "opened executable is linked, foreign, non-regular, or too large",
    )
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(descriptor, min(1 << 20, before.st_size - offset), offset)
        _require(bool(chunk), "short read while hashing opened executable")
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "opened executable changed while being hashed",
    )
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ExecutorError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ExecutorError(f"non-finite JSON constant is forbidden: {value}")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_regular_no_follow(path: Path, maximum_bytes: int) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        _require(before.st_uid == os.geteuid(), f"foreign-owned file: {path}")
        _require(before.st_nlink == 1, f"hard-linked file: {path}")
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
        raise ExecutorError(f"cannot read {path}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _strict_json(path: Path) -> tuple[dict[str, Any], bytes, str]:
    raw = _read_regular_no_follow(path, MAX_JSON_BYTES)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise ExecutorError(f"invalid strict JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value, raw, _sha256_bytes(raw)


def _open_directory_no_follow(path: Path) -> int:
    path = _absolute(path)
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
        metadata = os.fstat(descriptor)
        _require(metadata.st_uid == os.geteuid(), f"foreign-owned directory: {path}")
        return descriptor
    except (OSError, ExecutorError) as error:
        os.close(descriptor)
        if isinstance(error, ExecutorError):
            raise
        raise ExecutorError(f"cannot open directory without symlinks {path}: {error}") from error


def _atomic_bytes_no_clobber(path: Path, payload: bytes) -> str:
    path = _absolute(path)
    parent_fd = _open_directory_no_follow(path.parent)
    temporary = f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor: int | None = None
    try:
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ExecutorError(f"immutable output already exists: {path}")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(
                temporary,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise ExecutorError(f"output appeared concurrently: {path}") from error
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return _sha256_bytes(payload)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=parent_fd)
        os.close(parent_fd)


def _atomic_json_no_clobber(path: Path, value: object) -> str:
    return _atomic_bytes_no_clobber(path, _canonical_bytes(value) + b"\n")


def _ensure_owned_directory(path: Path) -> Path:
    path = _absolute(path)
    path.mkdir(mode=0o700, parents=False, exist_ok=True)
    descriptor = _open_directory_no_follow(path)
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISDIR(metadata.st_mode), f"not a directory: {path}")
        _require(metadata.st_uid == os.geteuid(), f"foreign-owned directory: {path}")
    finally:
        os.close(descriptor)
    return path


def _ensure_directory_chain(root: Path, relative: str) -> Path:
    """Create an owned directory chain beneath an already trusted root."""

    rel = _safe_relative(relative, "directory path")
    root = root.resolve(strict=True)
    cursor = root
    for component in rel.parts:
        cursor = _ensure_owned_directory(cursor / component)
    return cursor


def _safe_relative(value: str, name: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{name} is empty")
    path = Path(value)
    _require(
        not path.is_absolute()
        and value == path.as_posix()
        and all(part not in ("", ".", "..") for part in path.parts),
        f"{name} is not a safe relative path",
    )
    return path


def _read_beneath(root: Path, relative: str) -> tuple[bytes, str]:
    rel = _safe_relative(relative, "artifact path")
    root = root.resolve(strict=True)
    cursor = root
    for component in rel.parts[:-1]:
        cursor /= component
        try:
            metadata = os.lstat(cursor)
        except OSError as error:
            raise ExecutorError(f"artifact ancestor unavailable {cursor}: {error}") from error
        _require(
            stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.geteuid(),
            f"artifact ancestor is linked, foreign, or not a directory: {cursor}",
        )
    path = root / rel
    payload = _read_regular_no_follow(path, MAX_JSON_BYTES)
    return payload, _sha256_bytes(payload)


def _json_beneath(root: Path, relative: str) -> tuple[dict[str, Any], str]:
    payload, digest = _read_beneath(root, relative)
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise ExecutorError(f"invalid JSON artifact {relative}: {error}") from error
    _require(isinstance(value, dict), f"artifact is not an object: {relative}")
    return value, digest


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeError) as error:
        raise ExecutorError(f"cannot read boot ID: {error}") from error
    _require(bool(value), "boot ID is empty")
    return value


def _home_usage_bytes() -> int:
    try:
        result = subprocess.run(
            ("/usr/bin/du", "-sx", "--block-size=1", "--", "/home/chen"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ExecutorError(f"cannot measure /home/chen usage: {error}") from error
    _require(result.returncode == 0, f"du failed: {result.stderr.strip()}")
    try:
        return int(result.stdout.split()[0])
    except (IndexError, ValueError) as error:
        raise ExecutorError("du returned an invalid byte count") from error


def _round_disk_preflight(block_count: int, hard_limit: int) -> dict[str, int]:
    _require(hard_limit == campaign_schema.HOME_HARD_LIMIT_BYTES, "home limit differs")
    used = _home_usage_bytes()
    block_reserve = block_count * BLOCK_ARTIFACT_BUDGET_BYTES
    projected = (
        used
        + block_reserve
        + HOME_EMERGENCY_RESERVE_BYTES
        + ROUND_FINALIZATION_RESERVE_BYTES
    )
    return {
        "home_usage_bytes": used,
        "block_count": block_count,
        "per_block_artifact_budget_bytes": BLOCK_ARTIFACT_BUDGET_BYTES,
        "all_blocks_artifact_reserve_bytes": block_reserve,
        "emergency_reserve_bytes": HOME_EMERGENCY_RESERVE_BYTES,
        "finalization_reserve_bytes": ROUND_FINALIZATION_RESERVE_BYTES,
        "projected_home_bytes": projected,
        "home_hard_limit_bytes": hard_limit,
    }


def _validate_plan(
    source_manifest: dict[str, Any],
    source_raw_sha: str,
    source_canonical_sha: str,
    plan: dict[str, Any],
) -> None:
    expected = source_coordinator.materialize_plan(
        source_manifest,
        source_raw_sha,
        source_canonical_sha,
        plan.get("preflight"),
    )
    _require(plan == expected, "source-round plan is not the exact coordinator output")
    blocks = plan["blocks"]
    _require(len(blocks) == 4 + 4 * len(source_manifest["candidates"]), "block count differs")
    artifact_root = Path(source_manifest["coordinator"]["artifact_root"])
    for ordinal, block in enumerate(blocks):
        run_id = block["run_id"]
        expected_dir = f"runs/{run_id}"
        expected_attempt = f"{expected_dir}/stages/{FIXED_BENCHMARK_STAGE}/attempt-01"
        _require(block["ordinal"] == ordinal, f"plan block {ordinal} ordinal differs")
        _require(block["artifact_dir"] == expected_dir, f"plan block {ordinal} root differs")
        _require(
            block["raw_benchmark_artifact"] == f"{expected_attempt}/report.json",
            f"plan block {ordinal} report path differs",
        )
        _require(block["jit_dir"] == f"{expected_attempt}/jit", f"plan block {ordinal} JIT path differs")
        for field in ("artifact_dir", "raw_benchmark_artifact", "jit_dir"):
            relative = _safe_relative(block[field], f"blocks[{ordinal}].{field}")
            _require((artifact_root / relative).is_absolute(), "artifact path is not bound")


def _campaign_candidate(
    source_manifest: dict[str, Any], variant_id: str
) -> dict[str, Any]:
    parent_id = source_manifest["parent"]["id"]
    if variant_id == parent_id:
        return {
            "id": parent_id,
            "parent": "sealed-baseline",
            "primary_change": "sealed parent source baseline",
            "hypothesis": "parent blocks measure the unchanged source baseline",
            "expected_profile_metrics": ["baseline profile retained for attribution"],
            "risks": ["baseline drift invalidates the full source round"],
        }
    matches = [row for row in source_manifest["candidates"] if row["id"] == variant_id]
    _require(len(matches) == 1, f"unknown plan variant: {variant_id}")
    row = matches[0]
    return {
        "id": row["id"],
        "parent": parent_id,
        "primary_change": row["primary_change"],
        "hypothesis": row["hypothesis"],
        "expected_profile_metrics": deepcopy(row["expected_profile_metrics"]),
        "risks": deepcopy(row["risks"]),
    }


def _validate_live_contract(source_manifest: dict[str, Any]) -> None:
    """Reject coordinator-generic contracts that cannot drive this executor."""

    contract = source_manifest.get("frozen_execution_contract")
    _require(isinstance(contract, dict), "frozen_execution_contract is missing")
    algorithm = contract.get("algorithm")
    shape = contract.get("shape")
    timing = contract.get("timing")
    _require(
        isinstance(algorithm, dict)
        and set(algorithm) == round_evaluator._ALGORITHM_FIELDS,
        "live algorithm contract must contain exactly hop_mode, "
        "two_hop_threshold_percent, max_two_hop_percent, and hop_penalty_percent",
    )
    _require(
        algorithm["hop_mode"] in {"one_hop", "adaptive"},
        "live algorithm hop_mode must be one_hop or adaptive",
    )
    for field, maximum in (
        ("two_hop_threshold_percent", 10_000),
        ("max_two_hop_percent", 100),
        ("hop_penalty_percent", 10_000),
    ):
        value = algorithm[field]
        _require(
            type(value) is int and 0 <= value <= maximum,
            f"live algorithm {field} must be an integer in [0, {maximum}]",
        )
    _require(
        algorithm["hop_mode"] == "adaptive"
        or algorithm["max_two_hop_percent"] == 0,
        "one_hop live contract requires max_two_hop_percent=0",
    )
    _require(
        isinstance(shape, dict) and set(shape) == round_evaluator._WORKLOAD_FIELDS,
        "live shape contract does not contain the exact formal C100 workload fields",
    )
    _require(shape["world_size"] == 8, "live shape world_size must be exactly 8")
    _require(
        shape["stage"] in {"source", "return"}
        and isinstance(shape["case_name"], str)
        and shape["case_name"] in FORMAL_C100_CASES,
        "live shape stage/case_name is invalid",
    )
    for field in (
        "hidden",
        "num_topk",
        "num_channels",
        "num_experts",
        "num_destinations",
        "proxy_capacity_per_egress",
    ):
        _require(
            type(shape[field]) is int and shape[field] > 0,
            f"live shape {field} must be a positive integer",
        )
    for field in ("input_iteration", "init_dist_seed", "remainder_seed"):
        _require(
            type(shape[field]) is int and shape[field] >= 0,
            f"live shape {field} must be a nonnegative integer",
        )
    _require(
        type(shape["local_destination"]) is int
        and 0 <= shape["local_destination"] < shape["num_destinations"],
        "live shape local_destination is out of range",
    )
    _require(
        isinstance(shape["tokens_per_rank"], list)
        and len(shape["tokens_per_rank"]) == 8
        and all(type(item) is int and item > 0 for item in shape["tokens_per_rank"]),
        "live shape tokens_per_rank must contain eight positive integers",
    )
    _require(shape["dtype"] == "bfloat16", "live shape dtype must be bfloat16")
    _require(
        shape["interference_mode"] == "none",
        "formal profiler-free source round requires interference_mode=none",
    )
    _require(
        isinstance(shape["interference_compute_shape"], list)
        and len(shape["interference_compute_shape"]) == 3
        and all(
            type(item) is int and item > 0
            for item in shape["interference_compute_shape"]
        ),
        "live shape interference_compute_shape must have three positive integers",
    )
    _require(
        isinstance(timing, dict)
        and set(timing)
        == {
            "warmup_iterations",
            "steady_iterations",
            *round_evaluator._TIMER_FIELDS,
        },
        "live timing contract does not contain the exact formal timer fields",
    )
    _require(
        timing["warmup_iterations"] == 10
        and timing["steady_iterations"] == 100,
        "live timing requires exactly 10 warmup and 100 steady iterations",
    )
    for field in (
        "clock",
        "stage_truth",
        "percentile_method",
        "std_method",
        "logical_bandwidth_denominator",
    ):
        _require(
            isinstance(timing[field], str) and bool(timing[field]),
            f"live timing {field} must be a nonempty string",
        )
    _require(
        type(timing["logical_token_bytes"]) is int
        and timing["logical_token_bytes"] > 0
        and type(timing["logical_bytes_per_iteration"]) is int
        and timing["logical_bytes_per_iteration"] >= 0,
        "live timing logical byte counts are invalid",
    )
    _require(
        isinstance(contract.get("toolchain"), dict) and bool(contract["toolchain"]),
        "live toolchain contract must be a nonempty object",
    )
    _require(
        "created_utc" not in contract["toolchain"],
        "live toolchain contract must be normalized without created_utc",
    )


def _set_argv_option(argv: list[str], option: str, value: str) -> None:
    positions = [index for index, item in enumerate(argv) if item == option]
    _require(len(positions) <= 1, f"benchmark argv duplicates {option}")
    if positions:
        index = positions[0]
        _require(index + 1 < len(argv), f"benchmark argv lacks {option} value")
        argv[index + 1] = value
    else:
        argv.extend((option, value))


def _block_manifest(
    template: dict[str, Any],
    source_manifest: dict[str, Any],
    plan_block: dict[str, Any],
) -> dict[str, Any]:
    _validate_live_contract(source_manifest)
    manifest = deepcopy(template)
    manifest["campaign_id"] = f"source-{plan_block['run_id']}"
    manifest["candidate"] = _campaign_candidate(source_manifest, plan_block["variant_id"])
    manifest["resources"]["artifact_budget_bytes"] = BLOCK_ARTIFACT_BUDGET_BYTES
    contract = source_manifest["frozen_execution_contract"]
    algorithm = contract["algorithm"]
    shape = contract["shape"]
    timing = contract["timing"]
    benchmarks = [stage for stage in manifest["stages"] if stage["kind"] == "benchmark"]
    selected = [stage for stage in benchmarks if stage["id"] == FIXED_BENCHMARK_STAGE]
    _require(len(selected) == 1, f"template lacks unique {FIXED_BENCHMARK_STAGE}")
    fixed_dependencies = list(selected[0]["depends_on"])
    _require(
        fixed_dependencies == list(FIXED_VNODE_GATES),
        "fixed benchmark dependencies are not the two frozen vnode gates",
    )
    for stage in manifest["stages"]:
        if stage["kind"] == "benchmark":
            if stage["id"] != FIXED_BENCHMARK_STAGE:
                stage["enabled"] = False
                stage["skip_reason"] = "disabled by single-block formal source-round executor"
                continue
            stage.pop("enabled", None)
            stage.pop("skip_reason", None)
            stage["depends_on"] = fixed_dependencies
            stage["repeat"] = 1
            stage["artifacts"] = ["report.json"]
            argv = stage["argv"]
            _set_argv_option(argv, "--stage", str(shape["stage"]))
            _set_argv_option(argv, "--case-name", str(shape["case_name"]))
            _set_argv_option(argv, "--hop-mode", str(algorithm["hop_mode"]))
            _set_argv_option(
                argv,
                "--two-hop-threshold-percent",
                str(algorithm["two_hop_threshold_percent"]),
            )
            _set_argv_option(
                argv,
                "--max-two-hop-percent",
                str(algorithm["max_two_hop_percent"]),
            )
            _set_argv_option(
                argv,
                "--hop-penalty-percent",
                str(algorithm["hop_penalty_percent"]),
            )
            _set_argv_option(
                argv,
                "--warmup-iters",
                str(timing["warmup_iterations"]),
            )
            _set_argv_option(
                argv,
                "--steady-iters",
                str(timing["steady_iterations"]),
            )
            _set_argv_option(
                argv,
                "--interference-mode",
                str(shape["interference_mode"]),
            )
            _set_argv_option(argv, "--master-port", str(30301 + plan_block["ordinal"]))
            _set_argv_option(argv, "--json-out", "{stage_dir}/report.json")
            environment = dict(stage.get("env", {}))
            environment["EP_JIT_CACHE_DIR"] = "{stage_dir}/jit"
            stage["env"] = environment
        elif str(stage["kind"]).startswith("profile_"):
            stage["enabled"] = False
            stage["skip_reason"] = "profiling follows formal source-round selection"
    try:
        campaign_schema.validate_manifest(manifest)
    except campaign_schema.ManifestError as error:
        raise ExecutorError(f"generated block manifest is invalid: {error}") from error
    enabled = [
        stage
        for stage in manifest["stages"]
        if stage["kind"] == "benchmark" and stage.get("enabled", True)
    ]
    _require(
        len(enabled) == 1 and enabled[0]["id"] == FIXED_BENCHMARK_STAGE,
        "generated block does not contain exactly one fixed benchmark",
    )
    return manifest


@contextmanager
def _whole_round_lock(path: Path) -> Iterator[tuple[int, os.stat_result]]:
    _require(str(path) == LOCK_PATH, "source-round lock path differs")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        metadata = os.fstat(descriptor)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.geteuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == 0o600,
            "whole-round lock must be a 0600 owned one-link regular file",
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ExecutorError("another campaign or source round holds the global lock") from error
        yield descriptor, metadata
    except OSError as error:
        raise ExecutorError(f"cannot acquire whole-round lock: {error}") from error
    finally:
        if descriptor is not None:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _assert_lock_retained(descriptor: int, metadata: os.stat_result) -> None:
    current = os.fstat(descriptor)
    _require(
        (current.st_dev, current.st_ino) == (metadata.st_dev, metadata.st_ino),
        "whole-round lock FD identity changed",
    )
    probe: int | None = None
    try:
        probe = os.open(LOCK_PATH, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(probe, fcntl.LOCK_UN)
        raise ExecutorError("whole-round flock was released by a child")
    finally:
        if probe is not None:
            os.close(probe)


def _write_lease_record(
    descriptor: int,
    record: dict[str, Any],
) -> tuple[str, int]:
    _require(set(record) == _LEASE_FIELDS, "lease record fields differ")
    payload = _canonical_bytes(record) + b"\n"
    _require(len(payload) <= LOCK_RECORD_MAX_BYTES, "lease record is too large")
    os.ftruncate(descriptor, 0)
    written = os.pwrite(descriptor, payload, 0)
    _require(written == len(payload), "short write to whole-round lease")
    os.fsync(descriptor)
    shared_offset = int(record["shared_offset"])
    os.lseek(descriptor, shared_offset, os.SEEK_SET)
    return canonical_sha256(record), shared_offset


def _round_lock_identity(record: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "protocol": record["protocol"],
            "round_id": record["round_id"],
            "plan_raw_sha256": record["plan_raw_sha256"],
            "executor_pid": record["executor_pid"],
            "executor_code_sha256": record["executor_code_sha256"],
            "nonce_sha256": record["nonce_sha256"],
            "boot_id": record["boot_id"],
            "lock_path": record["lock_path"],
            "lock_device": record["lock_device"],
            "lock_inode": record["lock_inode"],
            "artifact_root": record["artifact_root"],
            "artifact_root_device": record["artifact_root_device"],
            "artifact_root_inode": record["artifact_root_inode"],
        }
    )


def _gpu_preflight(mapping: list[dict[str, Any]]) -> dict[str, Any]:
    return campaign_runner._stable_gpu_preflight(8, mapping)


def _wait_process_group_empty(process_group: int, timeout_seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        present = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "stat").read_text(encoding="ascii")
                closing = raw.rfind(")")
                fields = raw[closing + 1 :].split()
            except (OSError, UnicodeError):
                continue
            if closing > 0 and len(fields) > 2 and int(fields[2]) == process_group:
                present = True
                break
        if not present:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _terminate_owned_runner(process: subprocess.Popen[bytes]) -> None:
    # The session leader may already have exited while one of its descendants
    # remains in the inherited process group.  Always address the owned PGID;
    # never infer group emptiness from the leader's return code.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)
    if _wait_process_group_empty(process.pid, timeout_seconds=5):
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)
    _require(
        _wait_process_group_empty(process.pid, timeout_seconds=5),
        f"owned campaign process group {process.pid} survived SIGKILL",
    )


def _child_environment(
    descriptor: int,
    worktree: Path,
    round_id: str,
    plan_sha: str,
    nonce: str,
    lease_identity: str,
) -> dict[str, str]:
    return {
        "HOME": "/home/chen",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": "chen",
        "PATH": "/home/chen/.cache/deepep-sjlgpt/bin:/usr/local/cuda/bin:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{worktree}/tests/elastic:{worktree}",
        "SHELL": "/bin/bash",
        "TMPDIR": "/tmp",
        "TZ": "UTC",
        "USER": "chen",
        "DEEP_EP_SOURCE_ROUND_LEASE_FD": str(descriptor),
        "DEEP_EP_SOURCE_ROUND_LEASE_NONCE": nonce,
        "DEEP_EP_SOURCE_ROUND_LEASE_PARENT_PID": str(os.getpid()),
        "DEEP_EP_SOURCE_ROUND_LEASE_PLAN_SHA256": plan_sha,
        "DEEP_EP_SOURCE_ROUND_LEASE_ROUND_ID": round_id,
        "DEEP_EP_SOURCE_ROUND_LEASE_IDENTITY_SHA256": lease_identity,
    }


def _launch_runner(
    *,
    descriptor: int,
    worktree: Path,
    manifest_path: Path,
    plan_block: dict[str, Any],
    campaign_root: Path,
    round_id: str,
    plan_sha: str,
    nonce: str,
    lease_identity: str,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
    expected_runner_sha256: str,
) -> int:
    runner = worktree / RUNNER_PATH
    runner_fd: int | None = None
    try:
        runner_fd = os.open(runner, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise ExecutorError(f"cannot open frozen campaign runner: {error}") from error
    try:
        _require(
            _sha256_open_file(runner_fd, 16 * 1024 * 1024)
            == expected_runner_sha256,
            "campaign runner hash differs at launch",
        )
    except BaseException:
        os.close(runner_fd)
        raise
    argv = [
        PINNED_PYTHON,
        "-B",
        f"/proc/self/fd/{runner_fd}",
        "--manifest",
        str(manifest_path),
        "--run-id",
        plan_block["run_id"],
        "--output-dir",
        str(campaign_root),
        "--source-round-lease",
        "--source-round-lease-fd",
        str(descriptor),
        "--source-round-lease-round-id",
        round_id,
        "--source-round-lease-plan-sha256",
        plan_sha,
    ]
    stdout_fd: int | None = None
    stderr_fd: int | None = None
    process: subprocess.Popen[bytes] | None = None
    returncode: int | None = None
    descendants_leaked = False
    try:
        stdout_fd = os.open(
            stdout_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        stderr_fd = os.open(
            stderr_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        process = subprocess.Popen(
            argv,
            cwd=worktree,
            env=_child_environment(
                descriptor, worktree, round_id, plan_sha, nonce, lease_identity
            ),
            stdin=subprocess.DEVNULL,
            stdout=stdout_fd,
            stderr=stderr_fd,
            pass_fds=(descriptor, runner_fd),
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise ExecutorError(
                f"campaign runner timed out for block {plan_block['ordinal']}"
            ) from error
    finally:
        cleanup_error: BaseException | None = None
        log_error: BaseException | None = None
        # Containment is the first finally action.  An ENOSPC/EIO while syncing
        # logs must never bypass owned process-group cleanup.
        if process is not None:
            try:
                if process.poll() is None:
                    _terminate_owned_runner(process)
                elif not _wait_process_group_empty(process.pid):
                    descendants_leaked = True
                    _terminate_owned_runner(process)
                _require(
                    _wait_process_group_empty(process.pid),
                    f"campaign runner descendants remain for block "
                    f"{plan_block['ordinal']}",
                )
            except BaseException as error:
                cleanup_error = error
        for output_fd in (stdout_fd, stderr_fd):
            if output_fd is None:
                continue
            try:
                os.fsync(output_fd)
            except BaseException as error:
                if log_error is None:
                    log_error = error
            finally:
                with suppress(OSError):
                    os.close(output_fd)
        with suppress(OSError):
            os.close(runner_fd)
        if cleanup_error is not None:
            raise cleanup_error
        if log_error is not None:
            raise ExecutorError(f"cannot fsync runner logs: {log_error}") from log_error
    _require(process is not None, "campaign runner did not start")
    _require(returncode is not None, "campaign runner has no terminal return code")
    _require(
        not descendants_leaked,
        f"campaign runner leaked descendants for block {plan_block['ordinal']}",
    )
    _require(
        _sha256_file(runner, 16 * 1024 * 1024) == expected_runner_sha256,
        "campaign runner path changed across launch",
    )
    return returncode


def _round_candidate_call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except round_evaluator.CandidateEvidenceError as error:
        raise ExecutorError(f"campaign evidence failed formal validation: {error}") from error


def _validate_source_identity(
    value: object, *, commit: str, name: str
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} is missing")
    _require(
        set(value) >= _SOURCE_IDENTITY_REQUIRED_FIELDS,
        f"{name} fields are incomplete",
    )
    _require(value.get("commit") == commit, f"{name} commit differs")
    _require(value.get("dirty") is False, f"{name} is dirty")
    _require(value.get("status_porcelain") == [], f"{name} status is not clean")
    _require(value.get("untracked_files") == [], f"{name} has untracked files")
    for field in (
        "tracked_diff_sha256",
        "untracked_tree_sha256",
        "working_tree_identity_sha256",
    ):
        _require(
            isinstance(value.get(field), str) and bool(_SHA256.fullmatch(value[field])),
            f"{name} {field} is invalid",
        )
    return value


def _stage_results(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stages = campaign.get("stages")
    _require(isinstance(stages, list), "campaign stage results are missing")
    result: dict[str, dict[str, Any]] = {}
    for row in stages:
        _require(
            isinstance(row, dict) and isinstance(row.get("id"), str),
            "campaign stage result is invalid",
        )
        _require(row["id"] not in result, "campaign stage result is duplicated")
        result[row["id"]] = row
    return result


def _validate_finalized_is_last(campaign_root: Path, finalized_mtime_ns: int) -> None:
    for path in campaign_root.rglob("*"):
        metadata = os.lstat(path)
        _require(not stat.S_ISLNK(metadata.st_mode), f"artifact tree has symlink: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            _require(metadata.st_uid == os.geteuid(), f"foreign artifact directory: {path}")
            continue
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.geteuid()
            and metadata.st_nlink == 1,
            f"artifact tree has linked, foreign, or special file: {path}",
        )
        if path != campaign_root / "FINALIZED.json":
            _require(
                metadata.st_mtime_ns <= finalized_mtime_ns,
                f"artifact was modified after FINALIZED.json: {path}",
            )


def _validate_campaign_evidence(
    *,
    campaign_root: Path,
    plan_block: dict[str, Any],
    expected_manifest: dict[str, Any],
    expected_manifest_raw_sha256: str,
    expected_runner_sha256: str,
    expected_schema_sha256: str,
    expected_gpu_mapping_sha256: str,
    expected_lease_identity_sha256: str,
    child_returncode: int,
) -> dict[str, Any]:
    """Validate one runner terminal chain before publishing its control record."""

    expected_root = _absolute(campaign_root)
    try:
        resolved_root = campaign_root.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(f"campaign root is unavailable: {error}") from error
    _require(resolved_root == expected_root, "campaign root is not a canonical realpath")
    root_fd = _open_directory_no_follow(resolved_root)
    try:
        root_stat = os.fstat(root_fd)
    finally:
        os.close(root_fd)

    raw_manifest, _, raw_manifest_sha = _strict_json(resolved_root / "manifest.raw.json")
    saved_manifest, _, saved_manifest_sha = _strict_json(resolved_root / "manifest.json")
    _require(raw_manifest == expected_manifest, "runner raw manifest differs from generated input")
    _require(saved_manifest == expected_manifest, "runner saved manifest differs from generated input")
    _require(
        raw_manifest_sha == expected_manifest_raw_sha256,
        "runner raw manifest hash differs from generated input",
    )
    try:
        campaign_schema.validate_manifest(raw_manifest)
    except campaign_schema.ManifestError as error:
        raise ExecutorError(f"runner manifest no longer passes schema: {error}") from error
    _require(
        canonical_sha256(
            raw_manifest["resources"]["expected_gpu_index_uuid_mapping"]
        )
        == expected_gpu_mapping_sha256,
        "runner manifest GPU mapping differs",
    )

    campaign, _, result_sha = _strict_json(resolved_root / "result.json")
    _require(campaign.get("schema_version") == 1, "campaign result schema differs")
    _require(campaign.get("run_id") == plan_block["run_id"], "campaign run_id differs")
    _require(
        campaign.get("campaign_id") == expected_manifest["campaign_id"],
        "campaign_id differs",
    )
    _require(campaign.get("candidate") == expected_manifest["candidate"], "candidate differs")
    _require(campaign.get("performance_claim_allowed") is False, "campaign self-authorized a claim")
    _require(campaign.get("manifest_raw_sha256") == raw_manifest_sha, "campaign manifest hash differs")
    _require(
        campaign.get("manifest_canonical_sha256") == canonical_sha256(raw_manifest),
        "campaign canonical manifest hash differs",
    )
    _require(campaign.get("runner_sha256") == expected_runner_sha256, "runner hash differs")
    _require(campaign.get("schema_sha256") == expected_schema_sha256, "schema hash differs")
    pre_source = _validate_source_identity(
        campaign.get("source_identity"), commit=plan_block["source_commit"], name="source identity"
    )
    post_source = _validate_source_identity(
        campaign.get("source_identity_post"),
        commit=plan_block["source_commit"],
        name="post source identity",
    )
    _require(pre_source == post_source, "source identity changed during campaign")
    _require(campaign.get("source_drift") is False, "campaign reports source drift")

    finalized, _, finalized_sha = _strict_json(resolved_root / "FINALIZED.json")
    _require(set(finalized) == _FINALIZED_FIELDS, "FINALIZED fields differ")
    expected_final = {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "run_id": campaign["run_id"],
        "campaign_status": campaign.get("status"),
        "claim_scope": campaign.get("claim_scope"),
        "result_sha256": result_sha,
        "source_identity_sha256": pre_source["working_tree_identity_sha256"],
        "manifest_raw_sha256": raw_manifest_sha,
    }
    for field, value in expected_final.items():
        _require(finalized.get(field) == value, f"FINALIZED {field} differs")
    _require(
        isinstance(finalized.get("created_utc"), str)
        and isinstance(finalized.get("leaderboard_entry_id"), str)
        and bool(_SHA256.fullmatch(finalized["leaderboard_entry_id"])),
        "FINALIZED metadata is invalid",
    )
    sums_raw = _read_regular_no_follow(resolved_root / "SHA256SUMS", MAX_JSON_BYTES)
    sums_sha = _sha256_bytes(sums_raw)
    _require(finalized.get("sha256sums_sha256") == sums_sha, "FINALIZED SHA256SUMS differs")
    sums = _round_candidate_call(round_evaluator._parse_sha256sums, sums_raw)
    _round_candidate_call(
        round_evaluator._audit_artifact_tree,
        resolved_root,
        sums,
        root_identity=(root_stat.st_dev, root_stat.st_ino),
    )
    _require(sums.get("result.json") == result_sha, "SHA256SUMS result differs")
    _require(sums.get("manifest.raw.json") == raw_manifest_sha, "SHA256SUMS raw manifest differs")
    _require(sums.get("manifest.json") == saved_manifest_sha, "SHA256SUMS saved manifest differs")
    toolchain, _, toolchain_raw_sha = _strict_json(
        resolved_root / "environment" / "toolchain.json"
    )
    _require(
        sums.get("environment/toolchain.json") == toolchain_raw_sha,
        "SHA256SUMS toolchain differs",
    )
    lease, _, lease_raw_sha = _strict_json(
        resolved_root / "environment" / "source-round-lease.json"
    )
    _require(
        sums.get("environment/source-round-lease.json") == lease_raw_sha,
        "SHA256SUMS lease evidence differs",
    )
    _require(
        lease.get("identity_sha256") == expected_lease_identity_sha256
        and lease.get("inherited_flock_validated") is True,
        "runner did not retain the expected inherited lease evidence",
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
    _require(complete or failed, "campaign terminal status is ambiguous")
    _require(
        (complete and child_returncode == 0)
        or (failed and child_returncode != 0),
        "runner return code disagrees with FINALIZED",
    )
    stages = _stage_results(campaign)
    _require(
        [(row.get("id"), row.get("kind"), row.get("resource")) for row in campaign["stages"]]
        == [(row["id"], row["kind"], row["resource"]) for row in raw_manifest["stages"]],
        "campaign stage result order differs from manifest",
    )
    report_relative = Path(plan_block["raw_benchmark_artifact"]).relative_to(
        plan_block["artifact_dir"]
    ).as_posix()
    report: dict[str, Any] | None = None
    report_sha: str | None = None
    if complete:
        selected = stages.get(FIXED_BENCHMARK_STAGE)
        _require(isinstance(selected, dict), "fixed benchmark result is missing")
        attempts = selected.get("attempts")
        _require(
            selected.get("status") == "passed"
            and isinstance(attempts, list)
            and len(attempts) == 1
            and attempts[0].get("passed") is True
            and attempts[0].get("returncode") == 0,
            "fixed benchmark did not pass exactly one attempt",
        )
        report, report_sha = _json_beneath(resolved_root, report_relative)
        declared = attempts[0].get("declared_artifacts")
        matches = [
            row
            for row in declared if isinstance(row, dict) and row.get("path") == "report.json"
        ] if isinstance(declared, list) else []
        report_size = (resolved_root / report_relative).stat().st_size
        _require(
            len(matches) == 1
            and matches[0].get("sha256") == report_sha
            and matches[0].get("bytes") == report_size,
            "benchmark report declaration differs",
        )
        _require(sums.get(report_relative) == report_sha, "SHA256SUMS report differs")
        jit_relative = Path(plan_block["jit_dir"]).relative_to(
            plan_block["artifact_dir"]
        ).as_posix()
        jit_root = resolved_root / jit_relative
        jit_fd = _open_directory_no_follow(jit_root)
        try:
            _require(bool(os.listdir(jit_fd)), "successful benchmark JIT directory is empty")
        finally:
            os.close(jit_fd)
    else:
        _require(
            not (resolved_root / report_relative).exists(),
            "failed campaign retained a benchmark report",
        )

    finalized_stat = os.stat(resolved_root / "FINALIZED.json", follow_symlinks=False)
    _validate_finalized_is_last(resolved_root, finalized_stat.st_mtime_ns)
    return {
        "campaign_root": resolved_root,
        "artifact_root_stat": root_stat,
        "campaign": campaign,
        "campaign_result_raw_sha256": result_sha,
        "campaign_manifest_raw_sha256": raw_manifest_sha,
        "campaign_finalized_raw_sha256": finalized_sha,
        "benchmark_report_sha256": report_sha,
        "report": report,
        "toolchain": toolchain,
        "source_identity": pre_source,
        "complete": complete,
    }


def _normalized_file(value: object, name: str) -> dict[str, int | str]:
    return _round_candidate_call(round_evaluator._normalize_file_identity, value, name)


def _normalized_sources(
    value: object, name: str
) -> dict[str, dict[str, int | str]]:
    return _round_candidate_call(round_evaluator._normalize_sources, value, name)


def _normalized_toolchain(value: object) -> dict[str, Any]:
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


def _successful_identities(
    *,
    evidence: dict[str, Any],
    plan_block: dict[str, Any],
    source_manifest: dict[str, Any],
    executor_sha256: str,
    coordinator_boot_id: str,
    lock_identity_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = evidence["report"]
    _require(isinstance(report, dict), "successful evidence lacks a report")
    campaign = evidence["campaign"]
    identity = report.get("identity")
    _require(isinstance(identity, dict), "C100 identity is missing")
    pre = identity.get("pre_measurement")
    post = identity.get("post_measurement")
    _require(isinstance(pre, dict) and isinstance(post, dict), "C100 pre/post identity is missing")
    pre_extension = _normalized_file(pre.get("extension"), "pre extension")
    post_extension = _normalized_file(post.get("extension"), "post extension")
    _require(pre_extension == post_extension, "C100 extension changed during measurement")
    pre_sources = _normalized_sources(pre.get("sources"), "pre sources")
    post_sources = _normalized_sources(post.get("sources"), "post sources")
    _require(pre_sources == post_sources, "C100 sources changed during measurement")
    libraries = _round_candidate_call(
        round_evaluator._normalize_libraries,
        pre.get("loaded_libraries"),
        "pre loaded libraries",
    )
    post_libraries = _round_candidate_call(
        round_evaluator._normalize_libraries,
        post.get("loaded_libraries"),
        "post loaded libraries",
    )
    _require(libraries == post_libraries, "C100 loaded libraries changed")
    post_jit = _round_candidate_call(
        round_evaluator._normalize_jit, post.get("jit_cache"), "post JIT"
    )
    cubins = [
        row for row in post_jit if str(row["relative_path"]).endswith("/kernel.cubin")
    ]
    _require(bool(cubins), "C100 post identity contains no kernel cubin")
    source_identity = evidence["source_identity"]
    toolchain_sha = canonical_sha256(_normalized_toolchain(evidence["toolchain"]))
    build_bundle = canonical_sha256(
        {
            "source_commit": plan_block["source_commit"],
            "source_identity_sha256": source_identity[
                "working_tree_identity_sha256"
            ],
            "extension": pre_extension,
            "sources": pre_sources,
            "toolchain_sha256": toolchain_sha,
            "campaign_runner_sha256": campaign["runner_sha256"],
            "campaign_schema_sha256": campaign["schema_sha256"],
        }
    )
    code_identity = canonical_sha256(
        {"build_bundle_sha256": build_bundle, "jit_cubins": cubins}
    )
    worktree = Path(plan_block["worktree"])
    worktree_stat = os.stat(worktree, follow_symlinks=False)
    source_ref = {
        "ref_id": plan_block["variant_id"],
        "commit": plan_block["source_commit"],
        "working_tree_identity_sha256": source_identity[
            "working_tree_identity_sha256"
        ],
        "worktree_realpath": str(worktree),
        "worktree_device": worktree_stat.st_dev,
        "worktree_inode": worktree_stat.st_ino,
        "build_bundle_sha256": build_bundle,
        "code_identity_sha256": code_identity,
    }

    observed = report.get("measurement")
    shape = report.get("shape")
    environment = report.get("environment")
    command = report.get("command")
    _require(
        isinstance(observed, dict)
        and isinstance(shape, dict)
        and isinstance(environment, dict)
        and isinstance(environment.get("variables"), dict)
        and isinstance(command, dict)
        and isinstance(command.get("argv"), list),
        "C100 measurement, shape, environment, or command is missing",
    )
    workload = {field: shape[field] for field in round_evaluator._WORKLOAD_FIELDS}
    algorithm = {field: shape[field] for field in round_evaluator._ALGORITHM_FIELDS}
    timer = {field: observed[field] for field in round_evaluator._TIMER_FIELDS}
    semantic = identity.get("semantic_config")
    semantic_sha = identity.get("semantic_config_sha256")
    _require(
        isinstance(semantic, dict)
        and isinstance(semantic_sha, str)
        and canonical_sha256(semantic) == semantic_sha,
        "C100 semantic config identity differs",
    )
    normalized_environment = dict(environment["variables"])
    normalized_environment.pop("EP_JIT_CACHE_DIR", None)
    normalized_execution = {
        "semantic_config_sha256": semantic_sha,
        "environment": normalized_environment,
        "command": _round_candidate_call(
            round_evaluator._normalize_command, command["argv"]
        ),
    }
    harness_matches = [
        row["sha256"]
        for path, row in pre_sources.items()
        if path.endswith(BENCHMARK_PATH)
    ]
    _require(len(harness_matches) == 1, "C100 benchmark harness identity is ambiguous")
    contract = source_manifest["frozen_execution_contract"]
    expected_executor_sha = source_manifest["source_policy"]["harness_sha256"][
        SOURCE_ROUND_EXECUTOR_PATH
    ]
    _require(expected_executor_sha == executor_sha256, "executor hash changed")
    measurement = {
        "report_schema_version": report.get("schema_version"),
        "claim_scope": report.get("claim_scope"),
        "metric": "stage_truth_ns",
        "warmup_iterations": observed.get("warmup_iterations"),
        "steady_iterations": observed.get("steady_iterations"),
        "world_size": workload["world_size"],
        "order": list(round_evaluator._ABBA_BAAB),
        "timer": timer,
        "workload": workload,
        "algorithm": algorithm,
        "campaign_runner_sha256": campaign["runner_sha256"],
        "campaign_schema_sha256": campaign["schema_sha256"],
        "round_evaluator_sha256": round_evaluator.evaluator_sha256(),
        "benchmark_harness_sha256": harness_matches[0],
        "loaded_libraries_sha256": canonical_sha256(libraries),
        "toolchain_sha256": toolchain_sha,
        "gpu_mapping_sha256": canonical_sha256(contract["gpu_mapping"]),
        "semantic_config_sha256": semantic_sha,
        "normalized_execution_identity_sha256": canonical_sha256(
            normalized_execution
        ),
        "coordinator_boot_id": coordinator_boot_id,
        "coordinator_lock_identity_sha256": lock_identity_sha256,
    }
    expected_timing = {
        "warmup_iterations": measurement["warmup_iterations"],
        "steady_iterations": measurement["steady_iterations"],
        **measurement["timer"],
    }
    _require(
        measurement["algorithm"] == contract["algorithm"]
        and measurement["workload"] == contract["shape"]
        and expected_timing == contract["timing"]
        and measurement["toolchain_sha256"] == canonical_sha256(contract["toolchain"])
        and measurement["gpu_mapping_sha256"]
        == canonical_sha256(contract["gpu_mapping"]),
        "successful report differs from the frozen source-round contract",
    )
    report_relative = Path(plan_block["raw_benchmark_artifact"]).relative_to(
        plan_block["artifact_dir"]
    )
    jit_relative = Path(plan_block["jit_dir"]).relative_to(plan_block["artifact_dir"])
    _round_candidate_call(
        round_evaluator._validate_c100_report,
        report,
        measurement,
        source_ref,
        expected_jit_root=evidence["campaign_root"] / jit_relative,
        expected_report_path=evidence["campaign_root"] / report_relative,
    )
    return source_ref, measurement


def _unavailable_source_ref(
    plan_block: dict[str, Any], source_identity_sha256: str
) -> dict[str, Any]:
    worktree = Path(plan_block["worktree"])
    metadata = os.stat(worktree, follow_symlinks=False)
    sentinel = {
        "status": "unavailable_no_successful_report",
        "ref_id": plan_block["variant_id"],
        "commit": plan_block["source_commit"],
        "working_tree_identity_sha256": source_identity_sha256,
    }
    return {
        "ref_id": plan_block["variant_id"],
        "commit": plan_block["source_commit"],
        "working_tree_identity_sha256": source_identity_sha256,
        "worktree_realpath": str(worktree),
        "worktree_device": metadata.st_dev,
        "worktree_inode": metadata.st_ino,
        "build_bundle_sha256": canonical_sha256({**sentinel, "kind": "build_bundle"}),
        "code_identity_sha256": canonical_sha256({**sentinel, "kind": "code_identity"}),
    }


def _unavailable_measurement_contract(
    *,
    source_manifest: dict[str, Any],
    coordinator_boot_id: str,
    lock_identity_sha256: str,
) -> dict[str, Any]:
    contract = source_manifest["frozen_execution_contract"]
    timing = contract["timing"]
    harness = source_manifest["source_policy"]["harness_sha256"]
    sentinel = {
        "status": "unavailable_no_successful_report",
        "round_id": source_manifest["round_id"],
    }
    return {
        "report_schema_version": 4,
        "claim_scope": "checked_adapter_only",
        "metric": "stage_truth_ns",
        "warmup_iterations": timing["warmup_iterations"],
        "steady_iterations": timing["steady_iterations"],
        "world_size": contract["shape"]["world_size"],
        "order": list(round_evaluator._ABBA_BAAB),
        "timer": {field: timing[field] for field in round_evaluator._TIMER_FIELDS},
        "workload": deepcopy(contract["shape"]),
        "algorithm": deepcopy(contract["algorithm"]),
        "campaign_runner_sha256": harness[RUNNER_PATH],
        "campaign_schema_sha256": harness[SCHEMA_PATH],
        "round_evaluator_sha256": harness[EVALUATOR_PATH],
        "benchmark_harness_sha256": harness[BENCHMARK_PATH],
        "loaded_libraries_sha256": canonical_sha256(
            {**sentinel, "kind": "loaded_libraries"}
        ),
        "toolchain_sha256": canonical_sha256(contract["toolchain"]),
        "gpu_mapping_sha256": canonical_sha256(contract["gpu_mapping"]),
        "semantic_config_sha256": canonical_sha256(
            {**sentinel, "kind": "semantic_config"}
        ),
        "normalized_execution_identity_sha256": canonical_sha256(
            {**sentinel, "kind": "normalized_execution"}
        ),
        "coordinator_boot_id": coordinator_boot_id,
        "coordinator_lock_identity_sha256": lock_identity_sha256,
    }


def _write_coordinator_record(
    *,
    artifact_root: Path,
    source_manifest: dict[str, Any],
    source_manifest_raw_sha256: str,
    plan_raw_sha256: str,
    plan_block: dict[str, Any],
    evidence: dict[str, Any],
    coordinator_sha256: str,
    executor_sha256: str,
    boot_id: str,
    lock_identity_sha256: str,
    gpu_mapping_sha256: str,
    monotonic_start_ns: int,
    monotonic_end_ns: int,
) -> dict[str, Any]:
    _require(
        0 <= monotonic_start_ns < monotonic_end_ns < 2**63,
        "coordinator monotonic interval is invalid",
    )
    worktree = Path(plan_block["worktree"])
    worktree_stat = os.stat(worktree, follow_symlinks=False)
    campaign_root = evidence["campaign_root"]
    root_stat = evidence["artifact_root_stat"]
    record = {
        "schema_version": 1,
        "round_id": source_manifest["round_id"],
        "variant_id": plan_block["variant_id"],
        "source_round_manifest_raw_sha256": source_manifest_raw_sha256,
        "source_round_plan_raw_sha256": plan_raw_sha256,
        "source_round_plan_ordinal": plan_block["ordinal"],
        "source_round_run_id": plan_block["run_id"],
        "source_round_coordinator_sha256": coordinator_sha256,
        "source_round_executor_sha256": executor_sha256,
        "source_round_control_commit": source_manifest["coordinator"]["control_commit"],
        "source_round_parent_commit": source_manifest["parent"]["source_commit"],
        "ordinal": plan_block["repetition"],
        "global_ordinal": plan_block["ordinal"],
        "role": plan_block["role"],
        "boot_id": boot_id,
        "lock_identity_sha256": lock_identity_sha256,
        "monotonic_start_ns": monotonic_start_ns,
        "monotonic_end_ns": monotonic_end_ns,
        "artifact_root_realpath": str(campaign_root),
        "artifact_root_device": root_stat.st_dev,
        "artifact_root_inode": root_stat.st_ino,
        "worktree_realpath": str(worktree),
        "worktree_device": worktree_stat.st_dev,
        "worktree_inode": worktree_stat.st_ino,
        "source_commit": plan_block["source_commit"],
        "source_identity_sha256": evidence["source_identity"][
            "working_tree_identity_sha256"
        ],
        "campaign_result_raw_sha256": evidence["campaign_result_raw_sha256"],
        "campaign_finalized_raw_sha256": evidence[
            "campaign_finalized_raw_sha256"
        ],
        "benchmark_report_sha256": evidence["benchmark_report_sha256"],
        "gpu_mapping_sha256": gpu_mapping_sha256,
        "terminal_evidence_complete": True,
    }
    _require(
        set(record) == round_evaluator._COORDINATOR_FIELDS,
        "coordinator record fields differ from evaluator",
    )
    path = artifact_root / "coordinator" / f"block-{plan_block['ordinal']:02d}.json"
    digest = _atomic_json_no_clobber(path, record)
    return {
        "record": record,
        "path": path,
        "raw_sha256": digest,
        "block": {
            "ordinal": plan_block["repetition"],
            "global_ordinal": plan_block["ordinal"],
            "role": plan_block["role"],
            "artifact_root_realpath": str(campaign_root),
            "artifact_root_device": root_stat.st_dev,
            "artifact_root_inode": root_stat.st_ino,
            "campaign_result_raw_sha256": evidence[
                "campaign_result_raw_sha256"
            ],
            "campaign_manifest_raw_sha256": evidence[
                "campaign_manifest_raw_sha256"
            ],
            "finalized_raw_sha256": evidence["campaign_finalized_raw_sha256"],
            "coordinator_record": str(path),
            "coordinator_record_raw_sha256": digest,
            "stage_id": FIXED_BENCHMARK_STAGE,
            "attempt": 1,
            "artifact": "report.json",
        },
    }


def _build_round_manifest(
    *,
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
    source_manifest_raw_sha256: str,
    source_manifest_canonical_sha256: str,
    plan_path: Path,
    plan: dict[str, Any],
    plan_raw_sha256: str,
    plan_canonical_sha256: str,
    template: dict[str, Any],
    coordinator_sha256: str,
    executor_sha256: str,
    block_rows: list[dict[str, Any]],
    source_refs: dict[str, dict[str, Any]],
    measurement: dict[str, Any],
) -> dict[str, Any]:
    parent_id = source_manifest["parent"]["id"]
    by_global = {
        row["block"]["global_ordinal"]: row["block"] for row in block_rows
    }
    _require(
        set(by_global) == set(range(len(plan["blocks"]))),
        "coordinator block descriptors are incomplete",
    )
    parent_blocks = [
        by_global[block["ordinal"]]
        for block in plan["blocks"]
        if block["role"] == "parent"
    ]
    candidates = []
    for candidate in source_manifest["candidates"]:
        identity = _campaign_candidate(source_manifest, candidate["id"])
        candidates.append(
            {
                **identity,
                "frozen_contract_sha256": canonical_sha256(
                    template["frozen_contract"]
                ),
                "source_ref": source_refs[candidate["id"]],
                "benchmark_blocks": [
                    by_global[block["ordinal"]]
                    for block in plan["blocks"]
                    if block["variant_id"] == candidate["id"]
                ],
            }
        )
    manifest = {
        "schema_version": 1,
        "round_id": source_manifest["round_id"],
        "comparison_axis": "source_version",
        "parent": parent_id,
        "parent_campaign_candidate": _campaign_candidate(
            source_manifest, parent_id
        ),
        "parent_ref": source_refs[parent_id],
        "source_round_binding": {
            "manifest_path": str(source_manifest_path),
            "manifest_raw_sha256": source_manifest_raw_sha256,
            "manifest_canonical_sha256": source_manifest_canonical_sha256,
            "plan_path": str(plan_path),
            "plan_raw_sha256": plan_raw_sha256,
            "plan_canonical_sha256": plan_canonical_sha256,
            "coordinator_code_sha256": coordinator_sha256,
            "source_round_executor_sha256": executor_sha256,
            "control_commit": source_manifest["coordinator"]["control_commit"],
            "parent_commit": source_manifest["parent"]["source_commit"],
        },
        "frozen_contract_sha256": canonical_sha256(template["frozen_contract"]),
        "measurement_contract": measurement,
        "noise_policy": deepcopy(round_evaluator.FORMAL_NOISE_POLICY),
        "parent_blocks": parent_blocks,
        "candidates": candidates,
    }
    try:
        round_evaluator._validate_round_manifest(manifest)
    except round_evaluator.RoundError as error:
        raise ExecutorError(f"generated formal round manifest is invalid: {error}") from error
    return manifest


def _canonical_existing_file(path: Path, name: str) -> Path:
    path = _absolute(path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(f"{name} is unavailable: {error}") from error
    _require(resolved == path, f"{name} must be a canonical realpath")
    _read_regular_no_follow(path, MAX_JSON_BYTES)
    return path


def _prepare_inputs(
    source_manifest_path: Path, plan_path: Path, template_path: Path
) -> dict[str, Any]:
    source_manifest_path = _canonical_existing_file(
        source_manifest_path, "source manifest"
    )
    plan_path = _canonical_existing_file(plan_path, "source-round plan")
    template_path = _canonical_existing_file(template_path, "campaign template")
    try:
        source_manifest, source_raw_sha, source_canonical_sha = (
            source_coordinator.load_manifest(source_manifest_path)
        )
    except source_coordinator.SourceRoundError as error:
        raise ExecutorError(f"invalid source manifest: {error}") from error
    plan, _, plan_raw_sha = _strict_json(plan_path)
    plan_canonical_sha = canonical_sha256(plan)
    _validate_plan(source_manifest, source_raw_sha, source_canonical_sha, plan)
    try:
        template, _, _ = campaign_schema.load_manifest(template_path)
    except campaign_schema.ManifestError as error:
        raise ExecutorError(f"invalid campaign template: {error}") from error
    _validate_live_contract(source_manifest)

    control = Path(source_manifest["coordinator"]["control_worktree"])
    _require(control.resolve(strict=True) == control, "control worktree is not canonical")
    _require(Path.cwd().resolve(strict=True) == control, "executor must run in control worktree")
    _require(
        Path(__file__).resolve(strict=True) == control / SOURCE_ROUND_EXECUTOR_PATH,
        "running executor is not the frozen control-worktree file",
    )
    _require(
        template_path == control / CAMPAIGN_TEMPLATE_PATH,
        "campaign template is not the frozen control-worktree template",
    )
    artifact_root = Path(source_manifest["coordinator"]["artifact_root"])
    artifact_root_fd = _open_directory_no_follow(artifact_root)
    try:
        artifact_root_stat = os.fstat(artifact_root_fd)
        _require(
            stat.S_IMODE(artifact_root_stat.st_mode) == 0o700,
            "source-round artifact root must be mode 0700",
        )
    finally:
        os.close(artifact_root_fd)
    harness = source_manifest["source_policy"]["harness_sha256"]
    for relative, expected in harness.items():
        observed = _sha256_file(control / relative)
        _require(observed == expected, f"current frozen harness differs: {relative}")
    executor_sha = _sha256_file(control / SOURCE_ROUND_EXECUTOR_PATH)
    coordinator_sha = _sha256_file(
        control / "tests/elastic/rail_balance_source_round_coordinator.py"
    )
    _require(
        executor_sha == harness[SOURCE_ROUND_EXECUTOR_PATH],
        "executor is not bound by source manifest harness hashes",
    )
    resources = template["resources"]
    contract = source_manifest["frozen_execution_contract"]
    _require(
        resources["required_gpu_count"] == 8
        and resources["require_all_host_gpus"] is True
        and resources["home_path"] == "/home/chen"
        and resources["home_hard_limit_bytes"] == campaign_schema.HOME_HARD_LIMIT_BYTES
        and resources["lock_path"] == LOCK_PATH
        and resources["expected_gpu_index_uuid_mapping"] == contract["gpu_mapping"],
        "campaign template resources differ from the live source contract",
    )
    _require(
        harness[RUNNER_PATH] == _sha256_file(control / RUNNER_PATH)
        and harness[SCHEMA_PATH] == _sha256_file(control / SCHEMA_PATH)
        and harness[EVALUATOR_PATH] == round_evaluator.evaluator_sha256()
        and harness[BENCHMARK_PATH] == _sha256_file(control / BENCHMARK_PATH),
        "formal runtime harness identity differs",
    )
    return {
        "source_manifest_path": source_manifest_path,
        "source_manifest": source_manifest,
        "source_raw_sha256": source_raw_sha,
        "source_canonical_sha256": source_canonical_sha,
        "plan_path": plan_path,
        "plan": plan,
        "plan_raw_sha256": plan_raw_sha,
        "plan_canonical_sha256": plan_canonical_sha,
        "template_path": template_path,
        "template": template,
        "artifact_root": artifact_root,
        "artifact_root_stat": artifact_root_stat,
        "executor_sha256": executor_sha,
        "coordinator_sha256": coordinator_sha,
        "runner_sha256": harness[RUNNER_PATH],
        "schema_sha256": harness[SCHEMA_PATH],
    }


def _assert_artifact_root_identity(prepared: dict[str, Any]) -> os.stat_result:
    root = prepared["artifact_root"]
    expected = prepared["artifact_root_stat"]
    descriptor = _open_directory_no_follow(root)
    try:
        observed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(
        stat.S_ISDIR(observed.st_mode)
        and observed.st_uid == os.geteuid()
        and stat.S_IMODE(observed.st_mode) == 0o700
        and (observed.st_dev, observed.st_ino) == (expected.st_dev, expected.st_ino),
        "source-round artifact root identity changed",
    )
    return observed


def _terminal_record(
    *,
    prepared: dict[str, Any],
    status: str,
    lock_identity_sha256: str | None,
    disk_preflight: dict[str, Any] | None,
    blocks_completed: int,
    round_manifest_sha256: str | None,
    evaluation: dict[str, Any] | None,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "round_id": prepared["source_manifest"]["round_id"],
        "status": status,
        "source_round_manifest_raw_sha256": prepared["source_raw_sha256"],
        "source_round_plan_raw_sha256": prepared["plan_raw_sha256"],
        "source_round_executor_sha256": prepared["executor_sha256"],
        "source_round_coordinator_sha256": prepared["coordinator_sha256"],
        "round_lock_identity_sha256": lock_identity_sha256,
        "disk_preflight": disk_preflight,
        "planned_block_count": len(prepared["plan"]["blocks"]),
        "completed_block_count": blocks_completed,
        "round_manifest": (
            None
            if round_manifest_sha256 is None
            else {
                "path": str(prepared["artifact_root"] / ROUND_MANIFEST_NAME),
                "raw_sha256": round_manifest_sha256,
            }
        ),
        "evaluation": evaluation,
        "reasons": reasons,
        "limitations": [
            "process closure relies on the frozen runner keeping all stage processes in the executor-created PGID",
            "the formal harness forbids stage setsid; this host exposes no writable cgroup-v2 kill scope",
            "same-UID hostile namespace/path replacement is outside the cooperative experiment threat model",
        ],
    }


def _validate_embedded_evaluation(
    evaluation: dict[str, Any],
    formal_manifest: dict[str, Any],
    round_manifest_sha256: str,
) -> None:
    _require(evaluation.get("round_id") == formal_manifest["round_id"], "evaluation round differs")
    _require(
        evaluation.get("round_manifest_raw_sha256") == round_manifest_sha256,
        "evaluation raw manifest hash differs",
    )
    _require(
        evaluation.get("round_evaluator_sha256")
        == formal_manifest["measurement_contract"]["round_evaluator_sha256"],
        "evaluation code hash differs",
    )
    _require(
        evaluation.get("source_round_binding")
        == formal_manifest["source_round_binding"],
        "evaluation source-round binding differs",
    )
    _require(
        evaluation.get("frozen_contract_sha256")
        == formal_manifest["frozen_contract_sha256"],
        "evaluation frozen contract differs",
    )
    promotion = evaluation.get("promotion")
    promoted = promotion.get("promoted") if isinstance(promotion, dict) else None
    allowed = evaluation.get("source_leader_promotion_allowed")
    _require(
        isinstance(promotion, dict)
        and type(promoted) is bool
        and type(allowed) is bool
        and promoted == allowed,
        "evaluation promotion flags disagree",
    )
    if promoted:
        _require(
            isinstance(promotion.get("candidate_id"), str)
            and promotion["candidate_id"] == evaluation.get("leader_after"),
            "promoted evaluation leader identity differs",
        )
    else:
        _require(
            promotion.get("candidate_id") is None
            and evaluation.get("leader_after") == evaluation.get("leader_before"),
            "non-promoted evaluation changed the leader",
        )
    _require(
        evaluation.get("status") in {"complete", "complete_with_failures"},
        "evaluation terminal status is invalid",
    )
    round_wide = evaluation.get("round_wide_evidence")
    control_audit = (
        round_wide.get("control_plane_audit")
        if isinstance(round_wide, dict)
        else None
    )
    _require(
        isinstance(control_audit, dict) and control_audit.get("passed") is True,
        "evaluation control-plane audit did not pass",
    )
    candidates = evaluation.get("candidates")
    _require(
        isinstance(candidates, list)
        and all(
            isinstance(candidate, dict)
            and isinstance(candidate.get("analysis"), dict)
            and candidate["analysis"].get("status") != "evidence_invalid"
            for candidate in candidates
        ),
        "evaluation contains invalid candidate evidence",
    )


def _run_live(
    prepared: dict[str, Any],
    *,
    timeout_seconds: int,
    gpu_preflight: Callable[[list[dict[str, Any]]], dict[str, Any]] = _gpu_preflight,
    launcher: Callable[..., int] = _launch_runner,
) -> int:
    source_manifest = prepared["source_manifest"]
    plan = prepared["plan"]
    artifact_root = prepared["artifact_root"]
    terminal_path = artifact_root / ROUND_TERMINAL_NAME
    round_manifest_path = artifact_root / ROUND_MANIFEST_NAME
    _require(not terminal_path.exists(), "source-round terminal record already exists")
    _require(not round_manifest_path.exists(), "formal round manifest already exists")
    mapping = source_manifest["frozen_execution_contract"]["gpu_mapping"]
    disk_state: dict[str, Any] | None = None
    blocks_completed = 0
    lock_identity: str | None = None

    with _whole_round_lock(Path(LOCK_PATH)) as (lock_fd, lock_stat):
        try:
            _assert_artifact_root_identity(prepared)
            # The plan's Git proof is immutable, but live execution rechecks it
            # inside the held lease so check-only readiness cannot go stale.
            try:
                checked = source_coordinator.preflight(
                    source_manifest,
                    current_directory=Path(
                        source_manifest["coordinator"]["control_worktree"]
                    ),
                )
            except source_coordinator.SourceRoundError as error:
                raise ExecutorError(f"live source preflight failed: {error}") from error
            _require(checked == plan["preflight"], "live source preflight differs from plan")
            initial_disk_state = _round_disk_preflight(
                len(plan["blocks"]), campaign_schema.HOME_HARD_LIMIT_BYTES
            )
            disk_state = {
                "initial": initial_disk_state,
                "latest": initial_disk_state,
            }
            if (
                initial_disk_state["projected_home_bytes"]
                > initial_disk_state["home_hard_limit_bytes"]
            ):
                raise RoundBlocked("BLOCKED_DISK", ["round_wide_disk_reserve_unavailable"])
            gpu_state = gpu_preflight(mapping)
            if gpu_state.get("idle") is not True:
                reasons = [str(item) for item in gpu_state.get("reasons", [])]
                raise RoundBlocked("BLOCKED_GPU", reasons or ["eight_gpus_not_exclusive"])

            for relative in ("manifests", "runs", "coordinator", "logs"):
                _ensure_directory_chain(artifact_root, relative)
            _assert_artifact_root_identity(prepared)
            for block in plan["blocks"]:
                for path in (
                    artifact_root / block["artifact_dir"],
                    artifact_root / "manifests" / f"{block['run_id']}.json",
                    artifact_root
                    / "coordinator"
                    / f"block-{block['ordinal']:02d}.json",
                ):
                    _require(not path.exists(), f"planned immutable output exists: {path}")

            nonce = secrets.token_hex(32)
            boot_id = _boot_id()
            root_stat = _assert_artifact_root_identity(prepared)
            gpu_mapping_sha = canonical_sha256(mapping)
            block_rows: list[dict[str, Any]] = []
            source_refs: dict[str, dict[str, Any]] = {}
            source_identity_by_variant: dict[str, str] = {}
            measurement: dict[str, Any] | None = None

            for block in plan["blocks"]:
                monotonic_start = time.monotonic_ns()
                _assert_artifact_root_identity(prepared)
                remaining_blocks = len(plan["blocks"]) - block["ordinal"]
                latest_disk_state = _round_disk_preflight(
                    remaining_blocks, campaign_schema.HOME_HARD_LIMIT_BYTES
                )
                assert disk_state is not None
                disk_state["latest"] = latest_disk_state
                if (
                    latest_disk_state["projected_home_bytes"]
                    > latest_disk_state["home_hard_limit_bytes"]
                ):
                    raise RoundBlocked(
                        "BLOCKED_DISK_MID_ROUND",
                        ["remaining_round_disk_reserve_unavailable"],
                    )
                gpu_state = gpu_preflight(mapping)
                if gpu_state.get("idle") is not True:
                    reasons = [str(item) for item in gpu_state.get("reasons", [])]
                    raise RoundBlocked(
                        "BLOCKED_GPU_MID_ROUND",
                        reasons or ["eight_gpus_lost_exclusivity"],
                    )
                manifest = _block_manifest(
                    prepared["template"], source_manifest, block
                )
                manifest_path = (
                    artifact_root / "manifests" / f"{block['run_id']}.json"
                )
                manifest_sha = _atomic_json_no_clobber(manifest_path, manifest)
                campaign_root = artifact_root / block["artifact_dir"]
                shared_offset = 1_000_000 + secrets.randbelow(2**31 - 1_000_000)
                lease_record = {
                    "schema_version": 1,
                    "protocol": LOCK_PROTOCOL,
                    "round_id": source_manifest["round_id"],
                    "plan_raw_sha256": prepared["plan_raw_sha256"],
                    "executor_pid": os.getpid(),
                    "executor_code_sha256": prepared["executor_sha256"],
                    "nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
                    "boot_id": boot_id,
                    "lock_path": LOCK_PATH,
                    "lock_device": lock_stat.st_dev,
                    "lock_inode": lock_stat.st_ino,
                    "shared_offset": shared_offset,
                    "artifact_root": str(artifact_root),
                    "artifact_root_device": root_stat.st_dev,
                    "artifact_root_inode": root_stat.st_ino,
                    "output_dir": str(campaign_root),
                    "run_id": block["run_id"],
                    "campaign_manifest_path": str(manifest_path),
                    "campaign_manifest_raw_sha256": manifest_sha,
                    "round_lock_identity_sha256": "",
                }
                candidate_lock_identity = _round_lock_identity(lease_record)
                lease_record["round_lock_identity_sha256"] = candidate_lock_identity
                if lock_identity is None:
                    lock_identity = candidate_lock_identity
                _require(
                    candidate_lock_identity == lock_identity,
                    "whole-round lock identity changed between blocks",
                )
                lease_identity, observed_offset = _write_lease_record(
                    lock_fd, lease_record
                )
                _require(observed_offset == shared_offset, "lease shared offset differs")
                stdout_path = artifact_root / "logs" / f"block-{block['ordinal']:02d}.stdout.log"
                stderr_path = artifact_root / "logs" / f"block-{block['ordinal']:02d}.stderr.log"
                returncode = launcher(
                    descriptor=lock_fd,
                    worktree=Path(block["worktree"]),
                    manifest_path=manifest_path,
                    plan_block=block,
                    campaign_root=campaign_root,
                    round_id=source_manifest["round_id"],
                    plan_sha=prepared["plan_raw_sha256"],
                    nonce=nonce,
                    lease_identity=lease_identity,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    timeout_seconds=timeout_seconds,
                    expected_runner_sha256=prepared["runner_sha256"],
                )
                _assert_lock_retained(lock_fd, lock_stat)
                evidence = _validate_campaign_evidence(
                    campaign_root=campaign_root,
                    plan_block=block,
                    expected_manifest=manifest,
                    expected_manifest_raw_sha256=manifest_sha,
                    expected_runner_sha256=prepared["runner_sha256"],
                    expected_schema_sha256=prepared["schema_sha256"],
                    expected_gpu_mapping_sha256=gpu_mapping_sha,
                    expected_lease_identity_sha256=lease_identity,
                    child_returncode=returncode,
                )
                observed_source_identity = evidence["source_identity"][
                    "working_tree_identity_sha256"
                ]
                previous_identity = source_identity_by_variant.setdefault(
                    block["variant_id"], observed_source_identity
                )
                _require(
                    previous_identity == observed_source_identity,
                    f"source identity changed across {block['variant_id']} blocks",
                )
                if evidence["complete"]:
                    source_ref, block_measurement = _successful_identities(
                        evidence=evidence,
                        plan_block=block,
                        source_manifest=source_manifest,
                        executor_sha256=prepared["executor_sha256"],
                        coordinator_boot_id=boot_id,
                        lock_identity_sha256=lock_identity,
                    )
                    prior_ref = source_refs.setdefault(block["variant_id"], source_ref)
                    _require(prior_ref == source_ref, "source build/code identity changed")
                    if measurement is None:
                        measurement = block_measurement
                    _require(
                        measurement == block_measurement,
                        "measurement contract changed across successful blocks",
                    )
                monotonic_end = time.monotonic_ns()
                block_rows.append(
                    _write_coordinator_record(
                        artifact_root=artifact_root,
                        source_manifest=source_manifest,
                        source_manifest_raw_sha256=prepared["source_raw_sha256"],
                        plan_raw_sha256=prepared["plan_raw_sha256"],
                        plan_block=block,
                        evidence=evidence,
                        coordinator_sha256=prepared["coordinator_sha256"],
                        executor_sha256=prepared["executor_sha256"],
                        boot_id=boot_id,
                        lock_identity_sha256=lock_identity,
                        gpu_mapping_sha256=gpu_mapping_sha,
                        monotonic_start_ns=monotonic_start,
                        monotonic_end_ns=monotonic_end,
                    )
                )
                blocks_completed += 1
                _assert_lock_retained(lock_fd, lock_stat)
                _assert_artifact_root_identity(prepared)
                try:
                    post_checked = source_coordinator.preflight(
                        source_manifest,
                        current_directory=Path(
                            source_manifest["coordinator"]["control_worktree"]
                        ),
                    )
                except source_coordinator.SourceRoundError as error:
                    raise ExecutorError(
                        f"post-block source preflight failed: {error}"
                    ) from error
                _require(
                    post_checked == plan["preflight"],
                    "source/checkouts drifted after a formal block",
                )

            for block in plan["blocks"]:
                variant_id = block["variant_id"]
                if variant_id not in source_refs:
                    source_refs[variant_id] = _unavailable_source_ref(
                        block, source_identity_by_variant[variant_id]
                    )
            if measurement is None:
                measurement = _unavailable_measurement_contract(
                    source_manifest=source_manifest,
                    coordinator_boot_id=boot_id,
                    lock_identity_sha256=str(lock_identity),
                )
            formal_manifest = _build_round_manifest(
                source_manifest_path=prepared["source_manifest_path"],
                source_manifest=source_manifest,
                source_manifest_raw_sha256=prepared["source_raw_sha256"],
                source_manifest_canonical_sha256=prepared[
                    "source_canonical_sha256"
                ],
                plan_path=prepared["plan_path"],
                plan=plan,
                plan_raw_sha256=prepared["plan_raw_sha256"],
                plan_canonical_sha256=prepared["plan_canonical_sha256"],
                template=prepared["template"],
                coordinator_sha256=prepared["coordinator_sha256"],
                executor_sha256=prepared["executor_sha256"],
                block_rows=block_rows,
                source_refs=source_refs,
                measurement=measurement,
            )
            round_manifest_sha = _atomic_json_no_clobber(
                round_manifest_path, formal_manifest
            )
            _assert_artifact_root_identity(prepared)
            try:
                evaluation = round_evaluator.evaluate_round(round_manifest_path)
            except (round_evaluator.RoundError, round_evaluator.CandidateEvidenceError) as error:
                raise ExecutorError(f"formal evaluator rejected completed round: {error}") from error
            _validate_embedded_evaluation(
                evaluation, formal_manifest, round_manifest_sha
            )
            _assert_artifact_root_identity(prepared)
            _assert_lock_retained(lock_fd, lock_stat)
            terminal = _terminal_record(
                prepared=prepared,
                status="COMPLETE",
                lock_identity_sha256=lock_identity,
                disk_preflight=disk_state,
                blocks_completed=blocks_completed,
                round_manifest_sha256=round_manifest_sha,
                evaluation=evaluation,
                reasons=[],
            )
            _assert_artifact_root_identity(prepared)
            _atomic_json_no_clobber(terminal_path, terminal)
            return 0
        except RoundBlocked as error:
            _assert_lock_retained(lock_fd, lock_stat)
            _assert_artifact_root_identity(prepared)
            terminal = _terminal_record(
                prepared=prepared,
                status=error.status,
                lock_identity_sha256=lock_identity,
                disk_preflight=disk_state,
                blocks_completed=blocks_completed,
                round_manifest_sha256=None,
                evaluation=None,
                reasons=error.reasons,
            )
            _assert_artifact_root_identity(prepared)
            _atomic_json_no_clobber(terminal_path, terminal)
            return 3
        except BaseException as error:
            _assert_lock_retained(lock_fd, lock_stat)
            _assert_artifact_root_identity(prepared)
            terminal = _terminal_record(
                prepared=prepared,
                status="ERROR",
                lock_identity_sha256=lock_identity,
                disk_preflight=disk_state,
                blocks_completed=blocks_completed,
                round_manifest_sha256=(
                    _sha256_file(round_manifest_path)
                    if round_manifest_path.exists()
                    else None
                ),
                evaluation=None,
                reasons=[f"{type(error).__name__}: {error}"],
            )
            _assert_artifact_root_identity(prepared)
            _atomic_json_no_clobber(terminal_path, terminal)
            raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--campaign-template", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--confirm-round-id")
    parser.add_argument("--block-timeout-seconds", type=int, default=14_400)
    args = parser.parse_args(argv)
    if args.block_timeout_seconds <= 0:
        parser.error("block timeout must be positive")
    if args.live:
        if args.confirm_round_id is None:
            parser.error("--live requires --confirm-round-id")
    elif args.confirm_round_id is not None:
        parser.error("--confirm-round-id is only valid with --live")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    prepared = _prepare_inputs(
        args.source_manifest, args.plan, args.campaign_template
    )
    round_id = prepared["source_manifest"]["round_id"]
    if not args.live:
        print(
            f"NOT_RUN source-round executor check-only: round={round_id} "
            f"blocks={len(prepared['plan']['blocks'])} launch=false",
            flush=True,
        )
        return 0
    _require(
        args.confirm_round_id == round_id,
        "--confirm-round-id does not exactly match the frozen round",
    )
    return _run_live(prepared, timeout_seconds=args.block_timeout_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExecutorError, source_coordinator.SourceRoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error
