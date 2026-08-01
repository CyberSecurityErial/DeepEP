"""Strict CPU-only schema for RailBalance COMMON_FAIR experiments.

The schema freezes comparison semantics before any adapter, build, or GPU
process may run.  It deliberately imports neither PyTorch nor DeepEP.  A
design-only manifest may name explicit blockers; a structurally frozen
manifest may not contain an unfrozen source, build recipe, runtime closure,
adapter, route, payload generator, resource budget, reference, or order
schedule.  Structural freezing is not execution authorization: a separate
closure verifier and executor remain mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 1 << 20

SYSTEM_IDS = (
    "deepep-clean",
    "deepep-off",
    "nccl-ep",
    "railbalance-best",
    "uccl-ep",
)
SYSTEM_ROLES = {
    "deepep-clean": "UPSTREAM_REPRODUCIBILITY_ANCHOR",
    "deepep-off": "SAME_TREE_CAUSAL_BASELINE",
    "nccl-ep": "DIRECT_EP_COMPETITOR",
    "railbalance-best": "TUNING_FROZEN_CANDIDATE",
    "uccl-ep": "DIRECT_EP_AND_NIC_BALANCING_COMPETITOR",
}
SYSTEM_ALGORITHM_MODES = {
    "deepep-clean": "UPSTREAM_DEFAULT",
    "deepep-off": "RAIL_BALANCE_OFF",
    "nccl-ep": "AUTHOR_BACKEND",
    "railbalance-best": "TUNING_FROZEN_RAIL_BALANCE",
    "uccl-ep": "AUTHOR_BACKEND",
}
FIXED_CANDIDATE_IDS = {
    "deepep-clean": "upstream-dd758caf",
    "deepep-off": "rail-balance-off",
    "nccl-ep": "nccl-ep-v0.1.0",
    "uccl-ep": "uccl-ep-61ee4240",
}
PHASE_IDS = (
    "route_prepare_complete",
    "dispatch_complete",
    "combine_complete",
    "roundtrip_complete",
)
ROUTE_PATTERNS = ("balanced", "expert_skew", "rail_hot")
DERIVED_ROUTE_SEMANTIC_SCOPES = {
    "balanced": "LOGICAL_BALANCE_CORRECTNESS_ONLY",
    "expert_skew": "LOGICAL_OWNER_BALANCED_EXPERT_SKEW_CORRECTNESS_ONLY",
    "rail_hot": "LOGICAL_OWNER_HOTSPOT_NOT_PHYSICAL_NIC_EVIDENCE",
}
CONFIRMATORY_ROUTE_SEMANTIC_SCOPES = {
    "balanced": "PHYSICAL_BALANCED_ROUTE_CONFIRMATORY",
    "expert_skew": "PHYSICAL_OWNER_BALANCED_EXPERT_SKEW_CONFIRMATORY",
    "rail_hot": (
        "INTENDED_PHYSICAL_RAIL_HOT_REQUIRES_TOPOLOGY_AND_COUNTER_VERIFICATION"
    ),
}
RESULT_CODES = (
    "BENCHMARK_FAILED",
    "BENCHMARK_INVALID",
    "BLOCKED_DEPENDENCY",
    "BLOCKED_DISK",
    "BLOCKED_GPU",
    "BLOCKED_NETWORK",
    "BLOCKED_TOPOLOGY",
    "BUILD_FAILED",
    "CORRECTNESS_FAILED",
    "EVIDENCE_MISSING",
    "LAUNCH_FAILED",
    "NONDETERMINISTIC",
    "NOT_RUN",
    "NO_COMMON_API",
    "PASSED_BENCHMARK",
    "PASSED_CORRECTNESS",
    "RAW_SAMPLES_COMPLETE_UNVERIFIED",
    "UNSUPPORTED_HARDWARE",
)
RAW_SAMPLE_FIELDS = (
    "block_id",
    "case_id",
    "iteration",
    "order_position",
    "pair_id",
    "phase_ns",
    "process_instance_id",
    "rank",
    "repeat",
    "route_pattern",
    "route_raw_sha256",
    "sample_class",
    "system_id",
)
REQUIRED_RESULT_FIELDS = (
    "evidence_sha256",
    "reasons",
    "result_code",
    "row_id",
    "terminal_stage",
)

_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._+-]{0,254}\Z")
RAW_FILE_SHA256 = "RAW_FILE_BYTES_SHA256"

CASE_SPECS = {
    "confirmatory-ht-n2-w16": ("CONFIRMATORY_COMMON_FAIR", "HT", 2, 8, 16),
    "confirmatory-ht-n4-w32": ("CONFIRMATORY_COMMON_FAIR", "HT", 4, 8, 32),
    "confirmatory-ll-n2-w16": ("CONFIRMATORY_COMMON_FAIR", "LL", 2, 8, 16),
    "confirmatory-ll-n4-w32": ("CONFIRMATORY_COMMON_FAIR", "LL", 4, 8, 32),
    "derived-ht-n1-w2": ("DERIVED_SCALE_DOWN_CORRECTNESS", "HT", 1, 2, 2),
    "derived-ht-n1-w4": ("DERIVED_SCALE_DOWN_CORRECTNESS", "HT", 1, 4, 4),
    "derived-ll-n1-w2": ("DERIVED_SCALE_DOWN_CORRECTNESS", "LL", 1, 2, 2),
    "derived-ll-n1-w4": ("DERIVED_SCALE_DOWN_CORRECTNESS", "LL", 1, 4, 4),
}
PAIR_SPECS = {
    "deepep-clean-vs-railbalance-best": ("deepep-clean", "railbalance-best"),
    "deepep-off-vs-railbalance-best": ("deepep-off", "railbalance-best"),
    "nccl-ep-vs-railbalance-best": ("nccl-ep", "railbalance-best"),
    "uccl-ep-vs-railbalance-best": ("uccl-ep", "railbalance-best"),
}


class CommonFairError(ValueError):
    """Raised when a COMMON_FAIR manifest is ambiguous or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CommonFairError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CommonFairError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CommonFairError(f"non-finite JSON constant is forbidden: {value}")


def _exact_fields(value: object, fields: set[str], name: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{name} must be an object")
    assert isinstance(value, dict)
    observed = set(value)
    _require(
        observed == fields,
        f"{name} fields are incomplete or unknown: "
        f"missing={sorted(fields - observed)}, extra={sorted(observed - fields)}",
    )
    return value


def _string(value: object, name: str) -> str:
    _require(type(value) is str and bool(value), f"{name} must be nonempty")
    assert isinstance(value, str)
    _require(
        all(ord(character) >= 0x20 and ord(character) != 0x7F for character in value),
        f"{name} contains a control character",
    )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CommonFairError(f"{name} is not valid UTF-8 text") from error
    return value


def _slug(value: object, name: str) -> str:
    result = _string(value, name)
    _require(bool(_SLUG.fullmatch(result)), f"{name} is not a safe slug")
    return result


def _integer(
    value: object,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    _require(type(value) is int and value >= minimum, f"{name} is invalid")
    assert isinstance(value, int)
    if maximum is not None:
        _require(value <= maximum, f"{name} exceeds its frozen upper bound")
    return value


def _sha256(value: object, name: str) -> str:
    result = _string(value, name)
    _require(bool(_SHA256.fullmatch(result)), f"{name} is not lowercase SHA256")
    return result


def _git_object(value: object, name: str) -> str:
    result = _string(value, name)
    _require(bool(_GIT_OBJECT.fullmatch(result)), f"{name} is not a Git SHA-1")
    return result


def _relative_path(value: object, name: str) -> str:
    result = _string(value, name)
    path = PurePosixPath(result)
    _require(not path.is_absolute(), f"{name} must be relative")
    _require(result == path.as_posix(), f"{name} is not normalized")
    _require(
        all(part not in {"", ".", ".."} for part in path.parts),
        f"{name} escapes its root",
    )
    _require(
        all(_PATH_COMPONENT.fullmatch(part) for part in path.parts),
        f"{name} contains an unsafe path component",
    )
    return result


def _string_list(
    value: object,
    name: str,
    *,
    nonempty: bool = True,
    sorted_unique: bool = True,
) -> list[str]:
    _require(type(value) is list, f"{name} must be a list")
    assert isinstance(value, list)
    if nonempty:
        _require(bool(value), f"{name} must be nonempty")
    result = [_string(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if sorted_unique:
        _require(result == sorted(set(result)), f"{name} must be sorted and unique")
    return result


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _open_path_without_symlinks(path: Path) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require(absolute.name not in {"", ".", ".."}, "manifest path is invalid")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open("/", directory_flags)
    try:
        for component in absolute.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            absolute.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise CommonFairError(f"cannot safely open manifest {path}: {error}") from error
    finally:
        os.close(directory_fd)


def read_manifest_bytes(path: Path) -> bytes:
    """Read a bounded owner-controlled regular file without following links."""

    descriptor = _open_path_without_symlinks(path)
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
            0 < before.st_size <= MAX_MANIFEST_BYTES,
            f"manifest must contain 1..{MAX_MANIFEST_BYTES} bytes",
        )
        raw = os.read(descriptor, MAX_MANIFEST_BYTES + 1)
        after = os.fstat(descriptor)
        _require(len(raw) <= MAX_MANIFEST_BYTES, "manifest exceeds the byte limit")
        _require(len(raw) == after.st_size, "manifest read was incomplete")
        _require(
            (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_uid,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            "manifest changed while it was being read",
        )
        return raw
    finally:
        os.close(descriptor)


def load_manifest(path: Path) -> tuple[dict[str, Any], bytes, str, str]:
    raw = read_manifest_bytes(path)
    try:
        text = raw.decode("utf-8")
        _require(not text.startswith("\ufeff"), "UTF-8 BOM is forbidden")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except CommonFairError:
        raise
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise CommonFairError(f"invalid JSON: {error}") from error
    _require(type(value) is dict, "manifest root must be an object")
    assert isinstance(value, dict)
    validate_manifest(value)
    return (
        value,
        raw,
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(canonical_bytes(value)).hexdigest(),
    )


def _validate_source(value: object, name: str) -> str:
    source = _exact_fields(
        value,
        {"status", "commit", "tree", "submodules"},
        name,
    )
    status = _string(source["status"], f"{name}.status")
    _require(
        status in {"PINNED", "AWAITING_TUNING_FREEZE"}, f"{name}.status is invalid"
    )
    if status == "PINNED":
        _git_object(source["commit"], f"{name}.commit")
        _git_object(source["tree"], f"{name}.tree")
    else:
        _require(
            source["commit"] is None and source["tree"] is None,
            f"{name} unfrozen source must not carry an identity",
        )
    submodules = source["submodules"]
    _require(type(submodules) is list, f"{name}.submodules must be a list")
    assert isinstance(submodules, list)
    paths: list[str] = []
    for index, item in enumerate(submodules):
        row_name = f"{name}.submodules[{index}]"
        row = _exact_fields(item, {"path", "commit", "tree"}, row_name)
        paths.append(_relative_path(row["path"], f"{row_name}.path"))
        _git_object(row["commit"], f"{row_name}.commit")
        _git_object(row["tree"], f"{row_name}.tree")
    _require(paths == sorted(set(paths)), f"{name}.submodules are not sorted/unique")
    _require(
        status == "PINNED" or not submodules,
        f"{name} unfrozen source cannot pin submodules",
    )
    return status


def _validate_hash_binding(
    value: object,
    name: str,
    *,
    statuses: set[str],
    pinned_status: str,
) -> str:
    row = _exact_fields(
        value,
        {"status", "path", "sha256", "sha256_kind"},
        name,
    )
    _require(
        row["sha256_kind"] == RAW_FILE_SHA256,
        f"{name}.sha256_kind must bind raw file bytes",
    )
    status = _string(row["status"], f"{name}.status")
    _require(status in statuses, f"{name}.status is invalid")
    if status == pinned_status:
        _relative_path(row["path"], f"{name}.path")
        _sha256(row["sha256"], f"{name}.sha256")
    else:
        _require(
            row["path"] is None and row["sha256"] is None,
            f"{name} incomplete binding must not carry a path/SHA256",
        )
    return status


def _validate_api_bindings(value: object, name: str) -> None:
    _require(type(value) is list, f"{name} must be a list")
    assert isinstance(value, list)
    phases: list[str] = []
    for index, item in enumerate(value):
        row_name = f"{name}[{index}]"
        row = _exact_fields(
            item,
            {
                "phase",
                "separability",
                "included_calls",
                "completion_primitive",
                "primary_metric",
            },
            row_name,
        )
        phase = _string(row["phase"], f"{row_name}.phase")
        phases.append(phase)
        separability = _string(row["separability"], f"{row_name}.separability")
        _require(
            separability in {"REQUIRED", "NOT_SEPARABLE"},
            f"{row_name}.separability is invalid",
        )
        if phase != "route_prepare_complete":
            _require(
                separability == "REQUIRED", f"{phase} must remain a required phase"
            )
        calls = _string_list(
            row["included_calls"],
            f"{row_name}.included_calls",
            sorted_unique=False,
        )
        _require(
            len(calls) == len(set(calls)),
            f"{row_name}.included_calls contains duplicates",
        )
        _string(row["completion_primitive"], f"{row_name}.completion_primitive")
        expected_primary = phase in {"dispatch_complete", "combine_complete"}
        _require(
            row["primary_metric"] is expected_primary,
            f"{row_name}.primary_metric changed",
        )
    _require(tuple(phases) == PHASE_IDS, f"{name} must contain the four frozen phases")


def _validate_resources(value: object, name: str) -> tuple[bool, dict[str, object]]:
    row = _exact_fields(
        value,
        {
            "sm_budget_status",
            "sm_budget_per_rank",
            "qp_budget_status",
            "qp_budget_per_rank",
            "nic_set_status",
            "nic_ids",
            "cpu_budget_status",
            "cpu_cores_per_rank",
            "proxy_cpu_budget_status",
            "proxy_cpu_cores_per_rank",
            "device_memory_budget_status",
            "extra_device_memory_bytes_per_rank",
            "resource_track",
        },
        name,
    )
    _require(
        row["resource_track"] == "MATCHED_RESOURCE",
        f"{name}.resource_track must be MATCHED_RESOURCE",
    )
    complete = True
    frozen_values: dict[str, object] = {}
    for prefix, value_field, minimum, maximum in (
        ("sm_budget", "sm_budget_per_rank", 1, 1024),
        ("qp_budget", "qp_budget_per_rank", 1, 65536),
        ("cpu_budget", "cpu_cores_per_rank", 1, 4096),
        ("proxy_cpu_budget", "proxy_cpu_cores_per_rank", 0, 4096),
        (
            "device_memory_budget",
            "extra_device_memory_bytes_per_rank",
            0,
            1 << 40,
        ),
    ):
        status = _string(row[f"{prefix}_status"], f"{name}.{prefix}_status")
        _require(
            status in {"FROZEN", "NOT_FROZEN"}, f"{name}.{prefix}_status is invalid"
        )
        if status == "FROZEN":
            frozen_values[value_field] = _integer(
                row[value_field],
                f"{name}.{value_field}",
                minimum,
                maximum,
            )
        else:
            complete = False
            _require(
                row[value_field] is None,
                f"{name}.{value_field} must be null while unfrozen",
            )
    nic_status = _string(row["nic_set_status"], f"{name}.nic_set_status")
    _require(
        nic_status in {"FROZEN", "NOT_FROZEN"}, f"{name}.nic_set_status is invalid"
    )
    nic_ids = _string_list(
        row["nic_ids"],
        f"{name}.nic_ids",
        nonempty=nic_status == "FROZEN",
    )
    if nic_status == "NOT_FROZEN":
        complete = False
        _require(not nic_ids, f"{name}.nic_ids must be empty while unfrozen")
    else:
        frozen_values["nic_ids"] = tuple(nic_ids)
    cpu = frozen_values.get("cpu_cores_per_rank")
    proxy = frozen_values.get("proxy_cpu_cores_per_rank")
    if cpu is not None and proxy is not None:
        assert isinstance(cpu, int)
        assert isinstance(proxy, int)
        _require(proxy <= cpu, f"{name} proxy CPU budget exceeds total CPU budget")
    return complete, frozen_values


def _validate_systems(value: object, structurally_frozen: bool) -> set[str]:
    _require(type(value) is list, "systems must be a list")
    assert isinstance(value, list)
    ids: list[str] = []
    same_tree_bindings: dict[str, tuple[tuple[object, ...], ...]] = {}
    resource_values: dict[str, dict[str, object]] = {}
    for index, item in enumerate(value):
        name = f"systems[{index}]"
        row = _exact_fields(
            item,
            {
                "id",
                "role",
                "source",
                "build_recipe",
                "runtime_closure",
                "adapter",
                "execution_config",
                "algorithm_mode",
                "candidate_id",
                "backend_modes",
                "api_bindings",
                "resources",
            },
            name,
        )
        system_id = _slug(row["id"], f"{name}.id")
        ids.append(system_id)
        _require(
            row["role"] == SYSTEM_ROLES.get(system_id),
            f"{name}.role differs from the frozen role",
        )
        source_status = _validate_source(row["source"], f"{name}.source")
        build_status = _validate_hash_binding(
            row["build_recipe"],
            f"{name}.build_recipe",
            statuses={"PINNED", "NOT_FROZEN", "BLOCKED_BUILD_CONTRACT_INCOMPLETE"},
            pinned_status="PINNED",
        )
        adapter_status = _validate_hash_binding(
            row["adapter"],
            f"{name}.adapter",
            statuses={"PINNED", "NOT_IMPLEMENTED"},
            pinned_status="PINNED",
        )
        runtime_status = _validate_hash_binding(
            row["runtime_closure"],
            f"{name}.runtime_closure",
            statuses={"PINNED", "NOT_BUILT"},
            pinned_status="PINNED",
        )
        config_status = _validate_hash_binding(
            row["execution_config"],
            f"{name}.execution_config",
            statuses={"PINNED", "NOT_FROZEN"},
            pinned_status="PINNED",
        )
        _require(
            row["algorithm_mode"] == SYSTEM_ALGORITHM_MODES.get(system_id),
            f"{name}.algorithm_mode differs from the frozen role",
        )
        if system_id == "railbalance-best":
            if structurally_frozen:
                _slug(row["candidate_id"], f"{name}.candidate_id")
            else:
                _require(
                    row["candidate_id"] is None,
                    f"{name}.candidate_id must await the tuning freeze",
                )
        else:
            _require(
                row["candidate_id"] == FIXED_CANDIDATE_IDS.get(system_id),
                f"{name}.candidate_id changed",
            )
        modes = _string_list(row["backend_modes"], f"{name}.backend_modes")
        _require(modes == ["HT", "LL"], f"{name}.backend_modes must be HT and LL")
        _validate_api_bindings(row["api_bindings"], f"{name}.api_bindings")
        resources_complete, frozen_resources = _validate_resources(
            row["resources"],
            f"{name}.resources",
        )
        resource_values[system_id] = frozen_resources
        if structurally_frozen:
            _require(
                source_status
                == build_status
                == adapter_status
                == runtime_status
                == config_status
                == "PINNED"
                and resources_complete,
                f"{name} is not fully frozen for STRUCTURALLY_FROZEN_UNVERIFIED",
            )
        if system_id in {"deepep-off", "railbalance-best"}:
            source = row["source"]
            build = row["build_recipe"]
            adapter = row["adapter"]
            runtime = row["runtime_closure"]
            config = row["execution_config"]
            same_tree_bindings[system_id] = (
                (
                    source["commit"],
                    source["tree"],
                    tuple(
                        (
                            item["path"],
                            item["commit"],
                            item["tree"],
                        )
                        for item in source["submodules"]
                    ),
                ),
                (build["status"], build["path"], build["sha256"]),
                (adapter["status"], adapter["path"], adapter["sha256"]),
                (runtime["status"], runtime["path"], runtime["sha256"]),
                (config["status"], config["path"], config["sha256"]),
            )
    _require(
        tuple(ids) == SYSTEM_IDS, "systems must contain the exact sorted system set"
    )
    _require(
        same_tree_bindings["deepep-off"][:-1]
        == same_tree_bindings["railbalance-best"][:-1],
        "same-tree off and RailBalance best must share source/build/adapter/runtime",
    )
    off_config = same_tree_bindings["deepep-off"][-1]
    rail_config = same_tree_bindings["railbalance-best"][-1]
    if structurally_frozen:
        _require(
            off_config[2] != rail_config[2],
            "same-tree off and RailBalance best need distinct config SHA256 values",
        )
    for field in (
        "sm_budget_per_rank",
        "qp_budget_per_rank",
        "nic_ids",
        "cpu_cores_per_rank",
        "proxy_cpu_cores_per_rank",
        "extra_device_memory_bytes_per_rank",
    ):
        observed = {
            resources[field]
            for resources in resource_values.values()
            if field in resources
        }
        _require(len(observed) <= 1, f"systems have unmatched {field}")
    return set(ids)


def _validate_binding_with_path(value: object, name: str) -> str:
    row = _exact_fields(
        value,
        {"status", "path", "sha256", "sha256_kind"},
        name,
    )
    _require(
        row["sha256_kind"] == RAW_FILE_SHA256,
        f"{name}.sha256_kind must bind raw file bytes",
    )
    status = _string(row["status"], f"{name}.status")
    _require(status in {"PINNED", "NOT_FROZEN"}, f"{name}.status is invalid")
    if status == "PINNED":
        _relative_path(row["path"], f"{name}.path")
        _sha256(row["sha256"], f"{name}.sha256")
    else:
        _require(
            row["path"] is None and row["sha256"] is None,
            f"{name} unfrozen binding must not carry path/hash",
        )
    return status


def _validate_route_binding(value: object, name: str) -> str:
    row = _exact_fields(
        value,
        {"status", "path", "raw_sha256", "canonical_sha256"},
        name,
    )
    status = _string(row["status"], f"{name}.status")
    _require(status in {"PINNED", "NOT_FROZEN"}, f"{name}.status is invalid")
    if status == "PINNED":
        _relative_path(row["path"], f"{name}.path")
        _sha256(row["raw_sha256"], f"{name}.raw_sha256")
        _sha256(row["canonical_sha256"], f"{name}.canonical_sha256")
    else:
        _require(
            row["path"] is None
            and row["raw_sha256"] is None
            and row["canonical_sha256"] is None,
            f"{name} unfrozen route must not carry a path/hash",
        )
    return status


def _validate_case(value: object, name: str, structurally_frozen: bool) -> str:
    row = _exact_fields(
        value,
        {
            "id",
            "mode",
            "claim_scope",
            "node_count",
            "ranks_per_node",
            "world_size",
            "tokens_per_rank",
            "num_experts",
            "hidden",
            "top_k",
            "dtypes",
            "expert_owner_policy",
            "routes",
            "payload_generator",
            "stage_limit",
            "future_claim_eligible",
        },
        name,
    )
    case_id = _slug(row["id"], f"{name}.id")
    mode = _string(row["mode"], f"{name}.mode")
    _require(mode in {"HT", "LL"}, f"{name}.mode is invalid")
    scope = _string(row["claim_scope"], f"{name}.claim_scope")
    _require(
        scope in {"DERIVED_SCALE_DOWN_CORRECTNESS", "CONFIRMATORY_COMMON_FAIR"},
        f"{name}.claim_scope is invalid",
    )
    nodes = _integer(row["node_count"], f"{name}.node_count", 1)
    ranks_per_node = _integer(row["ranks_per_node"], f"{name}.ranks_per_node", 1)
    world_size = _integer(row["world_size"], f"{name}.world_size", 1)
    _require(nodes * ranks_per_node == world_size, f"{name}.world_size is inconsistent")
    _require(
        CASE_SPECS.get(case_id) == (scope, mode, nodes, ranks_per_node, world_size),
        f"{name} identity/topology differs from the frozen case",
    )
    expected_tokens = 128 if mode == "LL" else 4096
    _require(
        _integer(row["tokens_per_rank"], f"{name}.tokens_per_rank", 1)
        == expected_tokens,
        f"{name}.tokens_per_rank changed for {mode}",
    )
    _require(
        _integer(row["num_experts"], f"{name}.num_experts", 1) == 256,
        f"{name}.num_experts must be 256",
    )
    _require(
        _integer(row["hidden"], f"{name}.hidden", 1) == 7168,
        f"{name}.hidden must be 7168",
    )
    _require(
        _integer(row["top_k"], f"{name}.top_k", 1) == 8, f"{name}.top_k must remain K8"
    )
    _require(256 % world_size == 0, f"{name} requires E divisible by world_size")
    dtypes = _exact_fields(
        row["dtypes"],
        {"payload", "dispatch", "combine", "topk_idx", "topk_weights"},
        f"{name}.dtypes",
    )
    _require(
        dtypes
        == {
            "payload": "BF16",
            "dispatch": "BF16",
            "combine": "BF16",
            "topk_idx": "INT64",
            "topk_weights": "FP32",
        },
        f"{name}.dtypes changed from the all-BF16 payload contract",
    )
    _require(
        row["expert_owner_policy"] == "CONTIGUOUS_EQUAL",
        f"{name}.expert_owner_policy changed",
    )
    routes = row["routes"]
    _require(type(routes) is list, f"{name}.routes must be a list")
    assert isinstance(routes, list)
    route_statuses: list[str] = []
    observed_patterns: list[str] = []
    for index, item in enumerate(routes):
        route_name = f"{name}.routes[{index}]"
        route = _exact_fields(
            item,
            {
                "pattern",
                "binding",
                "format",
                "per_rank_logical_shape",
                "weight_policy",
                "mask_policy",
                "semantic_scope",
            },
            route_name,
        )
        pattern = _string(route["pattern"], f"{route_name}.pattern")
        _require(
            pattern in ROUTE_PATTERNS,
            f"{route_name}.pattern is not frozen",
        )
        observed_patterns.append(pattern)
        route_statuses.append(
            _validate_route_binding(
                route["binding"],
                f"{route_name}.binding",
            )
        )
        _require(
            route["format"]
            == (
                "rail-balance-logical-route-v1"
                if scope == "DERIVED_SCALE_DOWN_CORRECTNESS"
                else "rail-balance-logical-route-compact-v2"
            ),
            f"{route_name}.format changed",
        )
        _require(
            type(route["per_rank_logical_shape"]) is list,
            f"{route_name}.per_rank_logical_shape must be a list",
        )
        assert isinstance(route["per_rank_logical_shape"], list)
        logical_shape = [
            _integer(
                dimension,
                f"{route_name}.per_rank_logical_shape[{dimension_index}]",
                1,
            )
            for dimension_index, dimension in enumerate(route["per_rank_logical_shape"])
        ]
        _require(
            logical_shape == [expected_tokens, 8],
            f"{route_name}.per_rank_logical_shape changed",
        )
        _require(
            route["weight_policy"] == "PRESERVE_INPUT_FP32",
            f"{route_name}.weight_policy changed",
        )
        _require(
            route["mask_policy"] == "NO_DROPPED_ROUTES",
            f"{route_name}.mask_policy changed",
        )
        _require(
            route["semantic_scope"]
            == (
                DERIVED_ROUTE_SEMANTIC_SCOPES[pattern]
                if scope == "DERIVED_SCALE_DOWN_CORRECTNESS"
                else CONFIRMATORY_ROUTE_SEMANTIC_SCOPES[pattern]
            ),
            f"{route_name}.semantic_scope changed",
        )
    _require(
        tuple(observed_patterns) == ROUTE_PATTERNS,
        f"{name}.routes must contain the exact three synthetic patterns",
    )
    generator = _exact_fields(
        row["payload_generator"],
        {
            "status",
            "id",
            "code_path",
            "code_sha256",
            "code_sha256_kind",
            "seed",
        },
        f"{name}.payload_generator",
    )
    generator_status = _string(generator["status"], f"{name}.payload_generator.status")
    _require(
        generator_status in {"PINNED", "NOT_FROZEN"},
        f"{name}.payload_generator.status is invalid",
    )
    _require(
        generator["id"] == "rail-balance-deterministic-bf16-v1",
        f"{name}.payload_generator.id changed",
    )
    _require(
        generator["code_sha256_kind"] == RAW_FILE_SHA256,
        f"{name}.payload_generator.code_sha256_kind changed",
    )
    _integer(generator["seed"], f"{name}.payload_generator.seed")
    if generator_status == "PINNED":
        _relative_path(
            generator["code_path"],
            f"{name}.payload_generator.code_path",
        )
        _sha256(generator["code_sha256"], f"{name}.payload_generator.code_sha256")
    else:
        _require(
            generator["code_path"] is None and generator["code_sha256"] is None,
            f"{name}.payload_generator code path/SHA256 must be null",
        )
    if scope == "DERIVED_SCALE_DOWN_CORRECTNESS":
        _require(
            nodes == 1 and ranks_per_node in {2, 4},
            f"{name} derived topology must be one node with 2/4 ranks",
        )
        _require(
            row["stage_limit"] == "CORRECTNESS",
            f"{name} derived case cannot enter benchmark",
        )
        _require(
            row["future_claim_eligible"] is False,
            f"{name} derived case cannot allow a performance claim",
        )
    else:
        _require(
            nodes in {2, 4} and ranks_per_node == 8,
            f"{name} confirmatory topology must be 2/4 nodes x 8 ranks",
        )
        _require(
            row["stage_limit"] == "BENCHMARK",
            f"{name} confirmatory case must include benchmark",
        )
        _require(
            row["future_claim_eligible"] is structurally_frozen,
            f"{name}.future_claim_eligible differs from manifest phase",
        )
    if structurally_frozen:
        _require(
            set(route_statuses) == {generator_status} == {"PINNED"},
            f"{name} route/payload generator is not frozen",
        )
    else:
        _require(
            row["future_claim_eligible"] is False,
            f"{name} design-only manifest cannot allow performance claims",
        )
    return case_id


def _validate_cases(value: object, structurally_frozen: bool) -> set[str]:
    _require(type(value) is list and bool(value), "cases must be a nonempty list")
    assert isinstance(value, list)
    ids = [
        _validate_case(item, f"cases[{index}]", structurally_frozen)
        for index, item in enumerate(value)
    ]
    _require(ids == sorted(set(ids)), "cases must be sorted and unique")
    _require(
        ids == sorted(CASE_SPECS),
        "cases must contain the exact 2/4-GPU and 2/4-node matrix",
    )
    return set(ids)


def _validate_matrix(value: object, case_ids: set[str], system_ids: set[str]) -> None:
    _require(type(value) is list, "matrix must be a list")
    assert isinstance(value, list)
    observed: list[tuple[str, str, str, str]] = []
    for index, item in enumerate(value):
        name = f"matrix[{index}]"
        row = _exact_fields(
            item,
            {
                "row_id",
                "case_id",
                "route_pattern",
                "system_id",
                "initial_result_code",
            },
            name,
        )
        case_id = _slug(row["case_id"], f"{name}.case_id")
        system_id = _slug(row["system_id"], f"{name}.system_id")
        route_pattern = _slug(row["route_pattern"], f"{name}.route_pattern")
        _require(case_id in case_ids, f"{name} references an unknown case")
        _require(system_id in system_ids, f"{name} references an unknown system")
        _require(
            route_pattern in ROUTE_PATTERNS,
            f"{name} references an unknown route pattern",
        )
        expected_row_id = f"{case_id}--{route_pattern}--{system_id}"
        _require(row["row_id"] == expected_row_id, f"{name}.row_id changed")
        _require(
            row["initial_result_code"] == "NOT_RUN", f"{name} must start at NOT_RUN"
        )
        observed.append((expected_row_id, case_id, route_pattern, system_id))
    expected = sorted(
        (
            f"{case_id}--{route_pattern}--{system_id}",
            case_id,
            route_pattern,
            system_id,
        )
        for case_id in case_ids
        for route_pattern in ROUTE_PATTERNS
        for system_id in system_ids
    )
    _require(
        observed == expected,
        "matrix must be the exact sorted case x route x system product",
    )


def _validate_measurement(value: object, structurally_frozen: bool) -> None:
    row = _exact_fields(
        value,
        {
            "timing_api",
            "barrier_location",
            "end_event_synchronize_location",
            "warmup_iterations",
            "steady_iterations",
            "independent_process_repeats",
            "minimum_order_blocks",
            "order_schedule",
            "rank_aggregation",
            "raw_rank_samples_required",
            "native_rank_mean_forbidden",
            "raw_sample_fields",
            "summary_statistics",
            "paired_effect",
            "confidence_interval",
            "multiple_testing",
            "aa_noise_gate",
            "outlier_policy",
            "fixed_input_required",
            "fixed_stream_required",
            "fixed_gpu_clocks_or_block_invalid",
            "environment_snapshot_required",
        },
        "measurement",
    )
    _require(row["timing_api"] == "CUDA_EVENT", "measurement timing API changed")
    _require(
        row["barrier_location"] == "OUTSIDE_TIMED_WINDOW",
        "world barrier must remain outside the timed window",
    )
    _require(
        row["end_event_synchronize_location"] == "OUTSIDE_TIMED_WINDOW",
        "end-event synchronize must remain outside the timed window",
    )
    _require(
        _integer(row["warmup_iterations"], "measurement.warmup_iterations") == 10,
        "warmup must remain 10",
    )
    _require(
        _integer(row["steady_iterations"], "measurement.steady_iterations") == 100,
        "steady iterations must remain 100",
    )
    _require(
        _integer(
            row["independent_process_repeats"],
            "measurement.independent_process_repeats",
        )
        == 5,
        "independent process repeats must remain 5",
    )
    minimum_blocks = _integer(
        row["minimum_order_blocks"],
        "measurement.minimum_order_blocks",
    )
    _require(minimum_blocks == 10, "minimum order blocks must remain 10")
    _require(
        row["rank_aggregation"] == "PER_ITERATION_MAX",
        "rank aggregation must remain per-iteration max",
    )
    _require(
        row["raw_rank_samples_required"] is True, "raw rank samples must be required"
    )
    _require(
        row["native_rank_mean_forbidden"] is True,
        "native rank means must remain forbidden",
    )
    fields = _string_list(row["raw_sample_fields"], "measurement.raw_sample_fields")
    _require(
        tuple(fields) == RAW_SAMPLE_FIELDS, "measurement.raw_sample_fields changed"
    )
    statistics_fields = _string_list(
        row["summary_statistics"],
        "measurement.summary_statistics",
    )
    _require(
        statistics_fields == ["CV", "MAD", "P50", "P95", "P99"],
        "measurement summary statistics changed",
    )
    _require(
        row["paired_effect"] == "LOG_LATENCY_RATIO",
        "paired effect must remain the log latency ratio",
    )
    _require(
        row["confidence_interval"] == "BLOCK_BOOTSTRAP_95_PERCENT",
        "confidence interval must remain block bootstrap 95%",
    )
    _require(
        row["multiple_testing"] == "HOLM_PRIMARY_FAMILY",
        "primary multiple testing must remain Holm-corrected",
    )
    _require(
        row["aa_noise_gate"] == "CI_EXCLUDES_1_AND_ABS_LOG_RATIO_EXCEEDS_AA_P95",
        "A/A noise gate changed",
    )
    _require(
        row["outlier_policy"] == "WHOLE_BLOCK_FAIL_NO_SINGLE_SAMPLE_DELETION",
        "outlier policy must remain whole-block failure",
    )
    for field in (
        "fixed_input_required",
        "fixed_stream_required",
        "fixed_gpu_clocks_or_block_invalid",
        "environment_snapshot_required",
    ):
        _require(row[field] is True, f"measurement.{field} must remain true")
    schedule = _exact_fields(
        row["order_schedule"],
        {"status", "pairs", "blocks", "sha256"},
        "measurement.order_schedule",
    )
    status = _string(schedule["status"], "measurement.order_schedule.status")
    _require(status in {"PINNED", "NOT_FROZEN"}, "order schedule status is invalid")
    pairs = schedule["pairs"]
    blocks = schedule["blocks"]
    _require(type(pairs) is list, "measurement.order_schedule.pairs must be a list")
    _require(type(blocks) is list, "measurement.order_schedule.blocks must be a list")
    assert isinstance(pairs, list)
    assert isinstance(blocks, list)
    if status == "PINNED":
        observed_pairs: dict[str, tuple[str, str]] = {}
        for index, item in enumerate(pairs):
            pair_name = f"measurement.order_schedule.pairs[{index}]"
            pair = _exact_fields(
                item,
                {"pair_id", "system_a", "system_b"},
                pair_name,
            )
            pair_id = _slug(pair["pair_id"], f"{pair_name}.pair_id")
            system_a = _slug(pair["system_a"], f"{pair_name}.system_a")
            system_b = _slug(pair["system_b"], f"{pair_name}.system_b")
            _require(
                pair_id not in observed_pairs,
                f"duplicate comparison pair: {pair_id}",
            )
            observed_pairs[pair_id] = (system_a, system_b)
        _require(
            observed_pairs == PAIR_SPECS,
            "order schedule must contain the four frozen RailBalance pairs",
        )
        expected_block_count = len(PAIR_SPECS) * minimum_blocks
        _require(
            len(blocks) == expected_block_count,
            "pinned order schedule has the wrong block count",
        )
        block_ids: list[str] = []
        pair_counts = {pair_id: 0 for pair_id in PAIR_SPECS}
        for index, item in enumerate(blocks):
            block_name = f"measurement.order_schedule.blocks[{index}]"
            block = _exact_fields(
                item,
                {"block_id", "pair_id", "cycle", "order"},
                block_name,
            )
            pair_id = _slug(block["pair_id"], f"{block_name}.pair_id")
            _require(pair_id in observed_pairs, f"{block_name} has an unknown pair")
            ordinal = pair_counts[pair_id]
            expected_id = f"{pair_id}-block-{ordinal:02d}"
            block_id = _slug(block["block_id"], f"{block_name}.block_id")
            _require(block_id == expected_id, f"{block_name}.block_id changed")
            expected_cycle = "ABBA" if ordinal % 2 == 0 else "BAAB"
            _require(
                block["cycle"] == expected_cycle,
                f"{block_name}.cycle does not alternate ABBA/BAAB",
            )
            system_a, system_b = observed_pairs[pair_id]
            expected_order = (
                [system_a, system_b, system_b, system_a]
                if expected_cycle == "ABBA"
                else [system_b, system_a, system_a, system_b]
            )
            _require(
                block["order"] == expected_order,
                f"{block_name}.order differs from its cycle",
            )
            pair_counts[pair_id] += 1
            block_ids.append(block_id)
        _require(
            block_ids == sorted(block_ids),
            "order schedule blocks must be sorted and grouped by pair",
        )
        _require(
            set(pair_counts.values()) == {minimum_blocks},
            "each comparison pair must have the minimum block count",
        )
        expected_sha = hashlib.sha256(
            canonical_bytes({"pairs": pairs, "blocks": blocks})
        ).hexdigest()
        _require(
            schedule["sha256"] == expected_sha,
            "measurement.order_schedule SHA256 changed",
        )
    else:
        _require(
            not pairs and not blocks and schedule["sha256"] is None,
            "unfrozen order schedule must be empty",
        )
    if structurally_frozen:
        _require(
            status == "PINNED",
            "STRUCTURALLY_FROZEN_UNVERIFIED requires a pinned order schedule",
        )


def _validate_correctness(value: object, structurally_frozen: bool) -> None:
    row = _exact_fields(
        value,
        {
            "reference",
            "reference_binding",
            "relative_error_operator",
            "relative_error_threshold",
            "relative_error_formula",
            "relative_error_denominator_floor",
            "nonfinite_output_policy",
            "record_max_absolute_error",
            "record_max_relative_error",
            "record_first_error",
            "integer_metadata_exact",
            "determinism_repetitions",
            "random_seeds",
            "special_inputs",
            "coverage_classes",
            "declared_dtypes_layouts_exhaustive",
        },
        "correctness",
    )
    _require(
        row["reference"] == "CANONICAL_PYTORCH_CPU_EXACT_METADATA",
        "correctness reference changed",
    )
    reference_status = _validate_binding_with_path(
        row["reference_binding"],
        "correctness.reference_binding",
    )
    if structurally_frozen:
        _require(
            reference_status == "PINNED",
            "STRUCTURALLY_FROZEN_UNVERIFIED requires a pinned correctness reference",
        )
    _require(
        row["relative_error_operator"] == "STRICTLY_LESS_THAN",
        "relative error operator must remain strict",
    )
    _require(
        type(row["relative_error_threshold"]) is float,
        "relative error threshold must be a JSON float",
    )
    _require(
        row["relative_error_threshold"] == 1e-3,
        "relative error threshold must remain 1e-3",
    )
    _require(
        row["relative_error_formula"]
        == "ABS(ACTUAL-REFERENCE)/MAX(ABS(REFERENCE),DENOMINATOR_FLOOR)",
        "relative error formula changed",
    )
    _require(
        type(row["relative_error_denominator_floor"]) is float
        and row["relative_error_denominator_floor"] == 1e-12,
        "relative error denominator floor must remain 1e-12",
    )
    _require(
        row["nonfinite_output_policy"] == "REJECT",
        "NaN/Inf outputs must be rejected",
    )
    for field in (
        "record_max_absolute_error",
        "record_max_relative_error",
        "record_first_error",
        "integer_metadata_exact",
    ):
        _require(row[field] is True, f"correctness.{field} must remain true")
    _require(
        _integer(
            row["determinism_repetitions"],
            "correctness.determinism_repetitions",
        )
        == 3,
        "determinism repetitions must remain 3",
    )
    _require(
        type(row["random_seeds"]) is list, "correctness.random_seeds must be a list"
    )
    assert isinstance(row["random_seeds"], list)
    seeds = [
        _integer(seed, f"correctness.random_seeds[{index}]")
        for index, seed in enumerate(row["random_seeds"])
    ]
    _require(seeds == [0, 1, 17, 20260801], "correctness random seeds changed")
    _require(
        row["declared_dtypes_layouts_exhaustive"] is True,
        "all declared dtype/layout combinations must be tested",
    )
    special = _string_list(row["special_inputs"], "correctness.special_inputs")
    _require(
        special
        == [
            "empty-expert",
            "finite-extremes",
            "nonaligned-shape",
            "repeated-expert-ids",
            "zero-route-weights",
            "zeros",
        ],
        "correctness special inputs changed",
    )
    coverage = _string_list(
        row["coverage_classes"],
        "correctness.coverage_classes",
    )
    _require(
        coverage
        == [
            "boundary-shape",
            "common-shape",
            "maximum-declared-shape",
            "minimum-declared-shape",
            "nonaligned-shape",
        ],
        "correctness coverage classes changed",
    )


def _validate_failure_policy(value: object) -> None:
    row = _exact_fields(
        value,
        {
            "result_codes",
            "performance_result_codes",
            "raw_sample_result_codes",
            "required_result_fields",
            "failed_results_must_not_carry_promotable_metrics",
            "evidence_missing_if_matrix_row_absent",
            "results_require_external_verification_before_promotion",
        },
        "failure_policy",
    )
    codes = _string_list(row["result_codes"], "failure_policy.result_codes")
    _require(tuple(codes) == RESULT_CODES, "failure result codes changed")
    performance = _string_list(
        row["performance_result_codes"],
        "failure_policy.performance_result_codes",
    )
    _require(
        performance == ["PASSED_BENCHMARK"],
        "only PASSED_BENCHMARK may carry performance",
    )
    raw_sample_codes = _string_list(
        row["raw_sample_result_codes"],
        "failure_policy.raw_sample_result_codes",
    )
    _require(
        raw_sample_codes == ["RAW_SAMPLES_COMPLETE_UNVERIFIED"],
        "raw sample completion must remain explicitly unverified",
    )
    required = _string_list(
        row["required_result_fields"],
        "failure_policy.required_result_fields",
    )
    _require(
        tuple(required) == REQUIRED_RESULT_FIELDS, "required result fields changed"
    )
    _require(
        row["failed_results_must_not_carry_promotable_metrics"] is True,
        "failed results must not carry promotable metrics",
    )
    _require(
        row["evidence_missing_if_matrix_row_absent"] is True,
        "missing matrix rows must remain evidence failures",
    )
    _require(
        row["results_require_external_verification_before_promotion"] is True,
        "raw results must require external verification before promotion",
    )


def _validate_preregistration(value: object, structurally_frozen: bool) -> None:
    row = _exact_fields(
        value,
        {
            "environment_binding",
            "analysis_plan_binding",
            "tuning_partition_binding",
            "confirmatory_partition_binding",
            "partition_policy",
            "confirmatory_sealed_until_candidate_freeze",
        },
        "preregistration",
    )
    statuses = [
        _validate_binding_with_path(row[field], f"preregistration.{field}")
        for field in (
            "environment_binding",
            "analysis_plan_binding",
            "tuning_partition_binding",
            "confirmatory_partition_binding",
        )
    ]
    _require(
        row["partition_policy"] == "HASHED_DISJOINT_TRACE_SEED_BLOCK_SETS",
        "tuning and confirmatory partitions must remain hash-bound and disjoint",
    )
    _require(
        row["confirmatory_sealed_until_candidate_freeze"] is True,
        "confirmatory inputs must remain sealed until candidate freeze",
    )
    if structurally_frozen:
        _require(
            set(statuses) == {"PINNED"},
            "STRUCTURALLY_FROZEN_UNVERIFIED requires all preregistration bindings",
        )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    root = _exact_fields(
        manifest,
        {
            "schema_version",
            "experiment_id",
            "track",
            "manifest_phase",
            "systems",
            "cases",
            "matrix",
            "measurement",
            "correctness",
            "failure_policy",
            "preregistration",
            "execution_contract",
        },
        "manifest",
    )
    _require(
        _integer(root["schema_version"], "schema_version", 1) == SCHEMA_VERSION,
        "schema_version changed",
    )
    _require(
        root["experiment_id"] == "rail-balance-common-fair-v1", "experiment_id changed"
    )
    _require(root["track"] == "COMMON_FAIR", "track must be COMMON_FAIR")
    phase = _string(root["manifest_phase"], "manifest_phase")
    _require(
        phase in {"DESIGN_ONLY", "STRUCTURALLY_FROZEN_UNVERIFIED"},
        "manifest_phase is invalid",
    )
    structurally_frozen = phase == "STRUCTURALLY_FROZEN_UNVERIFIED"
    contract = _exact_fields(
        root["execution_contract"],
        {
            "gpu_execution_status",
            "build_execution_status",
            "result_status",
            "execution_authorized",
            "performance_results_present",
            "closure_verification_status",
            "executor_status",
        },
        "execution_contract",
    )
    _require(
        contract
        == {
            "gpu_execution_status": "NOT_RUN",
            "build_execution_status": "NOT_RUN",
            "result_status": "NOT_RUN",
            "execution_authorized": False,
            "performance_results_present": False,
            "closure_verification_status": "NOT_RUN",
            "executor_status": "NOT_IMPLEMENTED",
        },
        "manifest execution contract must remain preregistration-only",
    )
    system_ids = _validate_systems(root["systems"], structurally_frozen)
    case_ids = _validate_cases(root["cases"], structurally_frozen)
    _validate_matrix(root["matrix"], case_ids, system_ids)
    _validate_measurement(root["measurement"], structurally_frozen)
    _validate_correctness(root["correctness"], structurally_frozen)
    _validate_failure_policy(root["failure_policy"])
    _validate_preregistration(root["preregistration"], structurally_frozen)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest, _raw, raw_sha256, canonical_sha256 = load_manifest(args.manifest)
    except CommonFairError as error:
        print(f"ERROR: {error}")
        return 2
    print(
        json.dumps(
            {
                "experiment_id": manifest["experiment_id"],
                "manifest_phase": manifest["manifest_phase"],
                "raw_sha256": raw_sha256,
                "canonical_sha256": canonical_sha256,
                "build_execution_status": "NOT_RUN",
                "gpu_execution_status": "NOT_RUN",
                "execution_authorized": False,
                "performance_results_present": False,
                "closure_verification_status": "NOT_RUN",
                "executor_status": "NOT_IMPLEMENTED",
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
