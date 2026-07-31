"""Strict, stdlib-only schema for hop-aware optimization campaigns.

This module deliberately imports neither PyTorch nor DeepEP.  A campaign must
be validated and resource-gated before any child process can create a CUDA
context.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import stat
import string
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
HOME_HARD_LIMIT_BYTES = 300_000_000_000
_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_GPU_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_STAGE_KINDS = {
    "scaffold",
    "compile",
    "correctness",
    "sanitizer",
    "benchmark",
    "profile_nsys",
    "profile_ncu",
}
_RESOURCE_CLASSES = {"cpu", "exclusive_8gpu", "multinode"}
_PLACEHOLDERS = {"root", "output_dir", "stage_dir", "attempt"}
_PINNED_PYTHON = "/home/chen/.cache/deepep-sjlgpt/bin/python"
_PINNED_NSYS = "/usr/local/cuda/bin/nsys"
_PINNED_NCU = "/usr/local/cuda/bin/ncu"
_PINNED_COMPUTE_SANITIZER = "/usr/local/cuda/bin/compute-sanitizer"
_RUNTIME_LEADERBOARD = ".cache/rail_balance/hop_campaign/leaderboard.jsonl"
_CAMPAIGN_LOCK = "/tmp/deepep-rail-balance-hop-campaign.lock"
_MAX_MANIFEST_BYTES = 1 << 20
_SAFE_ENVIRONMENT_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "EP_DISABLE_GIN",
    "EP_JIT_CACHE_DIR",
    "MAX_JOBS",
    "MASTER_PORT",
    "OMP_NUM_THREADS",
    "PYTHONPATH",
    "TORCH_CUDA_ARCH_LIST",
}
_SAFE_PYTHON_SCRIPTS = {
    "setup.py",
    "tests/elastic/test_rail_balance_hop_plan.py",
    "tests/elastic/test_rail_balance_hop_bench.py",
    "tests/elastic/test_rail_balance_hybrid_api.py",
    "tests/elastic/test_rail_balance_hybrid_dispatch_codegen.py",
    "tests/elastic/test_rail_balance_hybrid_combine_codegen.py",
    "tests/elastic/test_rail_balance_hybrid_layout.py",
    "tests/elastic/test_rail_balance_hybrid_policy.py",
    "tests/elastic/test_rail_balance_hybrid_public_lifecycle.py",
    "tests/elastic/test_rail_balance_hop_one_hop_cuda.py",
    "tests/elastic/bench_rail_balance_hop.py",
    "tests/elastic/bench_rail_balance_hybrid_lsa.py",
    "tests/elastic/run_rail_balance_build_warmup.py",
}
_SCRIPT_STAGE_CONTRACTS = {
    "setup.py": {("compile", "exclusive_8gpu")},
    "tests/elastic/run_rail_balance_build_warmup.py": {
        ("compile", "exclusive_8gpu")
    },
    "tests/elastic/test_rail_balance_hop_one_hop_cuda.py": {
        ("correctness", "exclusive_8gpu"),
        ("sanitizer", "exclusive_8gpu"),
    },
    "tests/elastic/test_rail_balance_hybrid_dispatch_codegen.py": {
        ("correctness", "exclusive_8gpu")
    },
    "tests/elastic/test_rail_balance_hybrid_combine_codegen.py": {
        ("correctness", "exclusive_8gpu")
    },
    "tests/elastic/bench_rail_balance_hop.py": {
        ("correctness", "exclusive_8gpu")
    },
    "tests/elastic/bench_rail_balance_hybrid_lsa.py": {
        ("benchmark", "exclusive_8gpu"),
        ("profile_nsys", "exclusive_8gpu"),
        ("profile_ncu", "exclusive_8gpu"),
    },
    "tests/elastic/test_rail_balance_hop_plan.py": {("correctness", "cpu")},
    "tests/elastic/test_rail_balance_hop_bench.py": {("correctness", "cpu")},
    "tests/elastic/test_rail_balance_hybrid_api.py": {("correctness", "cpu")},
    "tests/elastic/test_rail_balance_hybrid_layout.py": {("correctness", "cpu")},
    "tests/elastic/test_rail_balance_hybrid_policy.py": {("correctness", "cpu")},
    "tests/elastic/test_rail_balance_hybrid_public_lifecycle.py": {
        ("correctness", "cpu")
    },
}
_CONTRACT_FIELDS = {
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
}


class ManifestError(ValueError):
    """Raised when a campaign manifest is ambiguous or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _nonempty_string(value: object, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty")
    _require("\x00" not in value, f"{name} contains NUL")
    return value


def _positive_integer(value: object, name: str) -> int:
    _require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ManifestError(f"non-finite JSON constant is forbidden: {value}")


def read_manifest_bytes(path: Path) -> bytes:
    """Read one bounded, owner-controlled regular file without following links."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise ManifestError(f"cannot safely open manifest {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), "manifest must be a regular file")
        _require(before.st_uid == os.geteuid(), "manifest must be owned by this user")
        _require(before.st_nlink == 1, "manifest must have exactly one hard link")
        _require(
            stat.S_IMODE(before.st_mode) & 0o022 == 0,
            "manifest must not be group/world writable",
        )
        _require(
            0 <= before.st_size <= _MAX_MANIFEST_BYTES,
            f"manifest exceeds the {_MAX_MANIFEST_BYTES}-byte limit",
        )
        chunks: list[bytes] = []
        remaining = _MAX_MANIFEST_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 16, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        _require(
            len(raw) <= _MAX_MANIFEST_BYTES,
            f"manifest exceeds the {_MAX_MANIFEST_BYTES}-byte limit",
        )
        after = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "manifest changed while it was being read",
        )
        _require(len(raw) == after.st_size, "manifest read was incomplete")
        return raw
    finally:
        os.close(descriptor)


def load_manifest(path: Path) -> tuple[dict[str, Any], str, str]:
    raw = read_manifest_bytes(path)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ManifestError(f"invalid JSON: {error}") from error
    _require(isinstance(value, dict), "manifest root must be an object")
    validate_manifest(value)
    try:
        canonical = json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except ValueError as error:
        raise ManifestError(f"manifest is not canonical JSON: {error}") from error
    return value, raw_sha256, hashlib.sha256(canonical).hexdigest()


def _validate_template(value: str, name: str) -> None:
    try:
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(value)
            if field_name is not None
        }
    except ValueError as error:
        raise ManifestError(f"{name} has an invalid placeholder: {error}") from error
    unknown = fields - _PLACEHOLDERS
    _require(not unknown, f"{name} uses unsupported placeholders: {sorted(unknown)}")


def _expanded_for_validation(value: str) -> str:
    return value.format_map(
        {
            "root": "/repo",
            "output_dir": "/repo/.cache/rail_balance/campaigns/example",
            "stage_dir": "/repo/.cache/rail_balance/campaigns/example/stage",
            "attempt": "1",
        }
    )


def _validate_python_command(
    argv: list[object], name: str, offset: int = 0
) -> str:
    _require(len(argv) > offset and argv[offset] == _PINNED_PYTHON, f"{name} must use the pinned Python")
    cursor = offset + 1
    if cursor < len(argv) and argv[cursor] == "-B":
        cursor += 1
    _require(cursor < len(argv), f"{name} is missing its Python script")
    script = _expanded_for_validation(str(argv[cursor]))
    if script.startswith("/repo/"):
        script = script[len("/repo/") :]
    script = posixpath.normpath(script)
    _require(script in _SAFE_PYTHON_SCRIPTS, f"{name} uses an unapproved Python script: {script}")
    _require("-c" not in argv[offset + 1 : cursor + 1], f"{name} cannot execute inline Python")
    return script


def _validate_command(
    argv: list[object], kind: str, resource: str, name: str
) -> None:
    executable = argv[0]
    if executable == _PINNED_PYTHON:
        script = _validate_python_command(argv, name)
    else:
        wrapper = {
            "profile_nsys": _PINNED_NSYS,
            "profile_ncu": _PINNED_NCU,
            "sanitizer": _PINNED_COMPUTE_SANITIZER,
        }.get(kind)
        _require(wrapper is not None and executable == wrapper, f"{name} executable is not approved")
        try:
            python_offset = argv.index(_PINNED_PYTHON)
        except ValueError as error:
            raise ManifestError(f"{name} profiler command lacks the pinned child Python") from error
        script = _validate_python_command(argv, name, python_offset)
    _require(
        (kind, resource) in _SCRIPT_STAGE_CONTRACTS[script],
        f"{name} cannot run {script} as {kind}/{resource}",
    )
    for index, raw in enumerate(argv[1:], start=1):
        expanded = _expanded_for_validation(str(raw))
        candidate = expanded.split("=", 1)[1] if "=" in expanded else expanded
        if candidate.startswith("/"):
            normalized = posixpath.normpath(candidate)
            allowed = candidate == _PINNED_PYTHON or (
                normalized.startswith("/repo/")
                and ".." not in Path(candidate).parts
            )
            _require(allowed, f"{name}.argv[{index}] writes or reads outside the campaign root")
    for index, raw in enumerate(argv):
        value = str(raw)
        if value in {"--output-dir", "--output-json", "--json-out", "--log-file"}:
            _require(index + 1 < len(argv), f"{name}.{value} lacks a path")
            _require(
                str(argv[index + 1]).startswith("{stage_dir}/"),
                f"{name}.{value} must write below the attempt stage directory",
            )
        for prefix in ("--output=", "--log-file="):
            if value.startswith(prefix):
                _require(
                    value[len(prefix) :].startswith("{stage_dir}/"),
                    f"{name}.{prefix[:-1]} must write below the attempt stage directory",
                )


def _validate_environment_value(key: str, value: str, name: str) -> None:
    if key == "PYTHONPATH":
        _require(
            value
            in {
                "{root}",
                "{root}/tests:{root}/tests/elastic:{root}",
                "{root}/tests/elastic:{root}",
            },
            f"{name} has an unapproved PYTHONPATH",
        )
    elif key == "CUDA_VISIBLE_DEVICES":
        _require(value == "0,1,2,3,4,5,6,7", f"{name} must reserve all eight GPUs")
    elif key == "EP_JIT_CACHE_DIR":
        _require(
            value.startswith("{stage_dir}/")
            and ".." not in Path(value).parts,
            f"{name} must use an attempt-isolated stage cache",
        )
    elif key in {"EP_DISABLE_GIN", "OMP_NUM_THREADS"}:
        _require(value == "1", f"{name} must equal 1")
    elif key == "TORCH_CUDA_ARCH_LIST":
        _require(value == "9.0", f"{name} must equal 9.0")
    elif key == "MAX_JOBS":
        _require(value == "8", f"{name} must equal 8")
    elif key == "MASTER_PORT":
        _require(value.isdigit() and 1 <= int(value) <= 65535, f"{name} is invalid")


def _validate_stage(stage: object, index: int, seen: set[str]) -> str:
    name = f"stages[{index}]"
    _require(isinstance(stage, dict), f"{name} must be an object")
    required = {"id", "kind", "resource", "argv", "timeout_seconds", "depends_on"}
    optional = {"env", "repeat", "enabled", "skip_reason", "artifacts"}
    keys = set(stage)
    _require(required <= keys, f"{name} missing keys: {sorted(required - keys)}")
    _require(keys <= required | optional, f"{name} has unknown keys: {sorted(keys - required - optional)}")

    stage_id = _nonempty_string(stage["id"], f"{name}.id")
    _require(bool(_SLUG.fullmatch(stage_id)), f"{name}.id is not a safe slug")
    _require(stage_id not in seen, f"duplicate stage id: {stage_id}")
    kind = _nonempty_string(stage["kind"], f"{name}.kind")
    resource = _nonempty_string(stage["resource"], f"{name}.resource")
    _require(kind in _STAGE_KINDS, f"{name}.kind is unsupported")
    _require(resource in _RESOURCE_CLASSES, f"{name}.resource is unsupported")

    argv = stage["argv"]
    _require(isinstance(argv, list) and argv, f"{name}.argv must be a nonempty list")
    for item_index, item in enumerate(argv):
        item = _nonempty_string(item, f"{name}.argv[{item_index}]")
        _validate_template(item, f"{name}.argv[{item_index}]")
    _validate_command(argv, kind, resource, name)

    dependencies = stage["depends_on"]
    _require(isinstance(dependencies, list), f"{name}.depends_on must be a list")
    _require(len(dependencies) == len(set(dependencies)), f"{name}.depends_on has duplicates")
    for dependency in dependencies:
        dependency = _nonempty_string(dependency, f"{name}.depends_on item")
        _require(dependency in seen, f"{name} dependency must name an earlier stage: {dependency}")

    _positive_integer(stage["timeout_seconds"], f"{name}.timeout_seconds")
    repeat = stage.get("repeat", 1)
    _require(type(repeat) is int and 1 <= repeat <= 10, f"{name}.repeat must be in [1, 10]")
    enabled = stage.get("enabled", True)
    _require(type(enabled) is bool, f"{name}.enabled must be boolean")
    if not enabled:
        _nonempty_string(stage.get("skip_reason"), f"{name}.skip_reason")

    environment = stage.get("env", {})
    _require(isinstance(environment, dict), f"{name}.env must be an object")
    for key, value in environment.items():
        _require(isinstance(key, str) and bool(_ENVIRONMENT_NAME.fullmatch(key)), f"{name}.env has invalid key")
        _require(key in _SAFE_ENVIRONMENT_KEYS, f"{name}.env key is not approved: {key}")
        value = _nonempty_string(value, f"{name}.env[{key}]")
        _validate_template(value, f"{name}.env[{key}]")
        _validate_environment_value(key, value, f"{name}.env[{key}]")

    artifacts = stage.get("artifacts", [])
    _require(isinstance(artifacts, list), f"{name}.artifacts must be a list")
    for artifact_index, artifact in enumerate(artifacts):
        artifact = _nonempty_string(artifact, f"{name}.artifacts[{artifact_index}]")
        _require("{" not in artifact and "}" not in artifact, f"{name}.artifacts cannot use placeholders")
        _require(not Path(artifact).is_absolute(), f"{name}.artifacts must be relative")
        _require(".." not in Path(artifact).parts, f"{name}.artifacts cannot traverse parents")
    seen.add(stage_id)
    return stage_id


def validate_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "candidate",
        "frozen_contract",
        "resources",
        "stages",
        "leaderboard_path",
    }
    optional = {"notes"}
    keys = set(manifest)
    _require(required <= keys, f"manifest missing keys: {sorted(required - keys)}")
    _require(keys <= required | optional, f"manifest has unknown keys: {sorted(keys - required - optional)}")
    _require(manifest["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
    campaign_id = _nonempty_string(manifest["campaign_id"], "campaign_id")
    _require(bool(_SLUG.fullmatch(campaign_id)), "campaign_id is not a safe slug")

    candidate = manifest["candidate"]
    _require(isinstance(candidate, dict), "candidate must be an object")
    candidate_fields = {
        "id", "parent", "primary_change", "hypothesis",
        "expected_profile_metrics", "risks",
    }
    _require(set(candidate) == candidate_fields, "candidate fields are incomplete or unknown")
    for field in ("id", "parent", "primary_change", "hypothesis"):
        value = _nonempty_string(candidate[field], f"candidate.{field}")
        if field == "id":
            _require(bool(_SLUG.fullmatch(value)), "candidate.id is not a safe slug")
    for field in ("expected_profile_metrics", "risks"):
        values = candidate[field]
        _require(isinstance(values, list) and values, f"candidate.{field} must be nonempty")
        for index, value in enumerate(values):
            _nonempty_string(value, f"candidate.{field}[{index}]")

    contract = manifest["frozen_contract"]
    _require(isinstance(contract, dict), "frozen_contract must be an object")
    _require(set(contract) == _CONTRACT_FIELDS, "frozen_contract fields are incomplete or unknown")
    for field, value in contract.items():
        _require(value not in (None, "", [], {}), f"frozen_contract.{field} must be nonempty")
    target_platform = contract["target_platform"]
    target_fields = {
        "gpu_arch",
        "required_gpus",
        "topology",
        "driver",
        "cuda_toolkit",
        "pytorch",
        "python",
        "nsys",
        "ncu",
        "compute_sanitizer",
    }
    _require(
        isinstance(target_platform, dict)
        and set(target_platform) == target_fields,
        "frozen_contract.target_platform fields are incomplete or unknown",
    )
    expected_platform = {
        "gpu_arch": "SM90",
        "required_gpus": 8,
        "driver": "570.172.08",
        "cuda_toolkit": "12.8 build 35404655",
        "pytorch": "2.11.0+cu128",
        "python": "3.11.15",
        "nsys": "2024.6.2.225-246235244400v0",
        "ncu": "2025.1.1.0 build 35528883",
        "compute_sanitizer": "2025.1.0.0 build 35351055",
    }
    for field, expected in expected_platform.items():
        _require(
            target_platform[field] == expected,
            f"frozen_contract.target_platform.{field} differs from the pinned platform",
        )
    _nonempty_string(
        target_platform["topology"], "frozen_contract.target_platform.topology"
    )
    objective = contract["objective_and_budget"]
    _require(
        isinstance(objective, dict),
        "frozen_contract.objective_and_budget must be an object",
    )
    _require(
        objective.get("minimum_candidates_per_round") == 2
        and objective.get("max_candidates_per_round") == 4,
        "formal rounds must contain two to four candidates",
    )

    resources = manifest["resources"]
    _require(isinstance(resources, dict), "resources must be an object")
    resource_fields = {
        "required_gpu_count",
        "require_all_host_gpus",
        "expected_gpu_index_uuid_mapping",
        "home_path",
        "home_hard_limit_bytes",
        "artifact_budget_bytes",
        "lock_path",
    }
    _require(set(resources) == resource_fields, "resources fields are incomplete or unknown")
    _require(_positive_integer(resources["required_gpu_count"], "resources.required_gpu_count") == 8, "local campaign requires exactly eight GPUs")
    _require(resources["require_all_host_gpus"] is True, "all host GPUs must be reserved")
    gpu_mapping = resources["expected_gpu_index_uuid_mapping"]
    _require(
        isinstance(gpu_mapping, list) and len(gpu_mapping) == 8,
        "resources.expected_gpu_index_uuid_mapping must freeze eight GPUs",
    )
    for expected_index, row in enumerate(gpu_mapping):
        name = f"resources.expected_gpu_index_uuid_mapping[{expected_index}]"
        _require(isinstance(row, dict) and set(row) == {"index", "uuid"}, f"{name} is invalid")
        _require(type(row["index"]) is int and row["index"] == expected_index, f"{name}.index must equal {expected_index}")
        uuid = _nonempty_string(row["uuid"], f"{name}.uuid")
        _require(bool(_GPU_UUID.fullmatch(uuid)), f"{name}.uuid is invalid")
    _require(
        len({row["uuid"] for row in gpu_mapping}) == 8,
        "resources.expected_gpu_index_uuid_mapping has duplicate UUIDs",
    )
    home_path = Path(_nonempty_string(resources["home_path"], "resources.home_path"))
    _require(home_path == Path("/home/chen"), "resources.home_path is fixed to /home/chen")
    _require(resources["home_hard_limit_bytes"] == HOME_HARD_LIMIT_BYTES, "home hard limit must remain 300 GB decimal")
    artifact_budget = _positive_integer(resources["artifact_budget_bytes"], "resources.artifact_budget_bytes")
    _require(artifact_budget < HOME_HARD_LIMIT_BYTES, "artifact budget exceeds home hard limit")
    lock_path = Path(_nonempty_string(resources["lock_path"], "resources.lock_path"))
    _require(lock_path == Path(_CAMPAIGN_LOCK), "resources.lock_path is fixed")

    leaderboard_path = _nonempty_string(manifest["leaderboard_path"], "leaderboard_path")
    _require(leaderboard_path == _RUNTIME_LEADERBOARD, "runtime leaderboard path is fixed under ignored .cache")

    stages = manifest["stages"]
    _require(isinstance(stages, list) and stages, "stages must be a nonempty list")
    seen: set[str] = set()
    kinds: dict[str, str] = {}
    stage_resources: dict[str, str] = {}
    dependencies: dict[str, list[str]] = {}
    for index, stage in enumerate(stages):
        stage_id = _validate_stage(stage, index, seen)
        kinds[stage_id] = stage["kind"]
        stage_resources[stage_id] = stage["resource"]
        dependencies[stage_id] = list(stage["depends_on"])

    def ancestor_kinds(stage_id: str) -> set[str]:
        result: set[str] = set()
        pending = list(dependencies[stage_id])
        while pending:
            dependency = pending.pop()
            result.add(kinds[dependency])
            pending.extend(dependencies[dependency])
        return result

    def ancestor_ids(stage_id: str) -> set[str]:
        result: set[str] = set()
        pending = list(dependencies[stage_id])
        while pending:
            dependency = pending.pop()
            if dependency in result:
                continue
            result.add(dependency)
            pending.extend(dependencies[dependency])
        return result

    for stage_id, kind in kinds.items():
        ancestors = ancestor_kinds(stage_id)
        stage_resource = next(
            stage["resource"] for stage in stages if stage["id"] == stage_id
        )
        if kind == "benchmark":
            _require("correctness" in ancestors, f"benchmark {stage_id} lacks a correctness gate")
            _require(
                any(
                    kinds[ancestor] == "correctness"
                    and stage_resources[ancestor] == "exclusive_8gpu"
                    for ancestor in ancestor_ids(stage_id)
                ),
                f"benchmark {stage_id} lacks a GPU correctness gate",
            )
            _require(stage_resource == "exclusive_8gpu", f"benchmark {stage_id} must reserve all eight GPUs")
        if kind == "correctness" and stage_resource != "cpu":
            _require("compile" in ancestors, f"GPU correctness {stage_id} lacks a compile gate")
        if kind == "sanitizer":
            _require("compile" in ancestors, f"sanitizer {stage_id} lacks a compile gate")
            _require(stage_resource == "exclusive_8gpu", f"sanitizer {stage_id} must reserve all eight GPUs")
        if kind == "profile_nsys":
            _require("benchmark" in ancestors, f"Nsys stage {stage_id} lacks benchmark lineage")
            _require(stage_resource == "exclusive_8gpu", f"Nsys stage {stage_id} must reserve all eight GPUs")
        if kind == "profile_ncu":
            _require("profile_nsys" in ancestors, f"NCU stage {stage_id} lacks Nsys lineage")
            _require(stage_resource == "exclusive_8gpu", f"NCU stage {stage_id} must reserve all eight GPUs")
        if kind == "benchmark":
            _require("sanitizer" in ancestors, f"benchmark {stage_id} lacks a sanitizer gate")


def expand_template(value: str, replacements: dict[str, str]) -> str:
    _validate_template(value, "template")
    missing = _PLACEHOLDERS - set(replacements)
    _require(not missing, f"missing template replacements: {sorted(missing)}")
    return value.format_map(replacements)
