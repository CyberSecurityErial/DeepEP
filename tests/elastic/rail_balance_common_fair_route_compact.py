"""Compact, reproducible logical routes for confirmatory COMMON_FAIR cases.

Version 2 stores an exact algorithm specification and streaming tensor hashes,
not expanded expert-ID or weight arrays.  Validation regenerates every logical
``[source_rank, token, top_k_slot]`` element and checks the pattern invariants.
The ``rail_hot`` route is only a logical endpoint stress pattern: physical rail
placement and hardware counters remain mandatory before making a topology claim.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import stat
import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


ROUTE_FORMAT = "rail-balance-logical-route-compact-v2"
ROUTE_VERSION = 2
MAX_ARTIFACT_BYTES = 64 * 1024
NUM_EXPERTS = 256
HIDDEN = 7168
TOP_K = 8
RANKS_PER_NODE = 8
OWNER_POLICY = "CONTIGUOUS_EQUAL"
LAYOUT = "SOURCE_RANK_TOKEN_TOPK_ROW_MAJOR"
DEFAULT_SEED = 0x5241494C5F5632
MAX_SEED = (1 << 63) - 1
ROUTE_PATTERNS = ("balanced", "expert_skew", "rail_hot")
HOT_EXPERTS_PER_OWNER = 2
CLAIM_SCOPE = "CONFIRMATORY_COMMON_FAIR_LOGICAL_ROUTE_INPUT_ONLY"
RAIL_VERIFICATION_REQUIRED = "REQUIRES_PHYSICAL_TOPOLOGY_AND_COUNTER_VERIFICATION"
WEIGHT_FP32_HEX = (
    "0x3f000000",  # 1/2
    "0x3e800000",  # 1/4
    "0x3e000000",  # 1/8
    "0x3d800000",  # 1/16
    "0x3d000000",  # 1/32
    "0x3c800000",  # 1/64
    "0x3c000000",  # 1/128
    "0x3c000000",  # 1/128; the eight values sum to exactly one
)

CASE_SPECS = {
    "confirmatory-ht-n2-w16": ("HT", 2, 16, 4096),
    "confirmatory-ht-n4-w32": ("HT", 4, 32, 4096),
    "confirmatory-ll-n2-w16": ("LL", 2, 16, 128),
    "confirmatory-ll-n4-w32": ("LL", 4, 32, 128),
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FP32_HEX = re.compile(r"0x[0-9a-f]{8}\Z")
_DECIMAL_SEED = re.compile(r"(?:0|[1-9][0-9]*)\Z")


class CommonFairCompactRouteError(ValueError):
    """Raised when a compact route is unsafe, ambiguous, or non-canonical."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CommonFairCompactRouteError(message)


def _exact_object(value: object, fields: set[str], name: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{name} must be an object")
    assert isinstance(value, dict)
    observed = set(value)
    _require(
        observed == fields,
        f"{name} fields differ: "
        f"missing={sorted(fields - observed)}, extra={sorted(observed - fields)}",
    )
    return value


def _exact_int(value: object, name: str, minimum: int, maximum: int) -> int:
    _require(
        type(value) is int and minimum <= value <= maximum,
        f"{name} is invalid",
    )
    assert isinstance(value, int)
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CommonFairCompactRouteError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CommonFairCompactRouteError(f"non-finite JSON constant is forbidden: {value}")


def canonical_bytes(value: object) -> bytes:
    """Return the one permitted finite canonical JSON encoding."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as error:
        raise CommonFairCompactRouteError(
            f"value is not finite canonical JSON: {error}"
        ) from error


def _identity_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key != "canonical_sha256"}


def canonical_sha256(artifact: dict[str, Any]) -> str:
    """Hash the canonical identity, excluding only the self-hash field."""

    return hashlib.sha256(canonical_bytes(_identity_payload(artifact))).hexdigest()


def _validate_generation_args(
    manifest_case_id: object,
    pattern: object,
    seed: object,
) -> tuple[str, int, int, int, str, int]:
    _require(
        type(manifest_case_id) is str and manifest_case_id in CASE_SPECS,
        "manifest_case_id is not one of the four confirmatory cases",
    )
    _require(
        type(pattern) is str and pattern in ROUTE_PATTERNS,
        f"pattern must be one of {ROUTE_PATTERNS}",
    )
    checked_seed = _exact_int(seed, "seed", 0, MAX_SEED)
    assert isinstance(manifest_case_id, str)
    assert isinstance(pattern, str)
    mode, nodes, world_size, tokens = CASE_SPECS[manifest_case_id]
    return mode, nodes, world_size, tokens, pattern, checked_seed


def _artifact_case_id(manifest_case_id: str, pattern: str) -> str:
    return f"{manifest_case_id}-{pattern}-compact-v2"


def _physical_evidence_status(pattern: str) -> str:
    if pattern == "rail_hot":
        return RAIL_VERIFICATION_REQUIRED
    return "LOGICAL_PATTERN_DOES_NOT_ASSERT_PHYSICAL_PLACEMENT"


def _algorithm_spec(pattern: str, seed: int) -> dict[str, Any]:
    formulas = {
        "balanced": {
            "expert_id_formula": "GLOBAL_LINEAR_INTERLEAVED_OWNER_LOCAL_V2",
            "pattern_invariant": "EXACT_GLOBAL_EXPERT_AND_OWNER_BALANCE",
            "topology_formula": "NONE_LOGICAL_BALANCE_ONLY",
        },
        "expert_skew": {
            "expert_id_formula": "GLOBAL_LINEAR_OWNER_TWO_HOT_LOCALS_V2",
            "pattern_invariant": "EXACT_OWNER_BALANCE_TWO_HOT_EXPERTS_PER_OWNER",
            "topology_formula": "NONE_LOGICAL_SKEW_ONLY",
        },
        "rail_hot": {
            "expert_id_formula": "REMOTE_NODE_FIXED_LOCAL_RANK_RAIL_V2",
            "pattern_invariant": "REMOTE_NODE_ENDPOINTS_ONE_FIXED_LOCAL_RANK",
            "topology_formula": "ROTATE_REMOTE_NODE_FIXED_SEED_MOD_8_LOCAL_RANK",
        },
    }
    return {
        "id": "COMMON_FAIR_CONFIRMATORY_LOGICAL_ROUTE",
        "version": ROUTE_VERSION,
        "seed": seed,
        "iteration_order": LAYOUT,
        "owner_rank_formula": "FLOOR_EXPERT_ID_DIV_EXPERTS_PER_OWNER",
        "weight_formula": "REPEAT_EIGHT_SLOT_FP32_BITS_PER_TOKEN",
        **formulas[pattern],
    }


def _expected_expert(
    *,
    pattern: str,
    nodes: int,
    world_size: int,
    tokens: int,
    seed: int,
    source_rank: int,
    token: int,
    slot: int,
) -> int:
    experts_per_owner = NUM_EXPERTS // world_size
    linear = (source_rank * tokens + token) * TOP_K + slot + seed
    if pattern == "balanced":
        owner = linear % world_size
        local_expert = (linear // world_size) % experts_per_owner
    elif pattern == "expert_skew":
        owner = linear % world_size
        local_expert = (linear // world_size) % HOT_EXPERTS_PER_OWNER
    else:
        assert pattern == "rail_hot"
        source_node = source_rank // RANKS_PER_NODE
        remote_offset = 1 + ((token + seed // RANKS_PER_NODE) % (nodes - 1))
        remote_node = (source_node + remote_offset) % nodes
        rail_local_rank = seed % RANKS_PER_NODE
        owner = remote_node * RANKS_PER_NODE + rail_local_rank
        local_expert = linear % experts_per_owner
    return owner * experts_per_owner + local_expert


def iter_expert_ids(
    *, manifest_case_id: str, pattern: str, seed: int = DEFAULT_SEED
) -> Iterator[int]:
    """Yield the logical INT64 expert tensor without materializing it."""

    _, nodes, world_size, tokens, checked_pattern, checked_seed = (
        _validate_generation_args(manifest_case_id, pattern, seed)
    )
    for source_rank in range(world_size):
        for token in range(tokens):
            for slot in range(TOP_K):
                yield _expected_expert(
                    pattern=checked_pattern,
                    nodes=nodes,
                    world_size=world_size,
                    tokens=tokens,
                    seed=checked_seed,
                    source_rank=source_rank,
                    token=token,
                    slot=slot,
                )


def iter_weight_fp32_bits(
    *, manifest_case_id: str, pattern: str, seed: int = DEFAULT_SEED
) -> Iterator[int]:
    """Yield logical IEEE-754 binary32 bit patterns without materializing them."""

    _, _, world_size, tokens, _, _ = _validate_generation_args(
        manifest_case_id, pattern, seed
    )
    parsed = tuple(int(value[2:], 16) for value in WEIGHT_FP32_HEX)
    for _source_rank in range(world_size):
        for _token in range(tokens):
            yield from parsed


def _stream_summary(
    *, manifest_case_id: str, pattern: str, seed: int
) -> tuple[str, str, int]:
    (
        _,
        nodes,
        world_size,
        tokens,
        checked_pattern,
        checked_seed,
    ) = _validate_generation_args(manifest_case_id, pattern, seed)
    experts_per_owner = NUM_EXPERTS // world_size
    count = world_size * tokens * TOP_K
    expert_digest = hashlib.sha256()
    weight_digest = hashlib.sha256()
    owner_counts = [0] * world_size
    expert_counts = [0] * NUM_EXPERTS
    hot_locals = [set() for _ in range(world_size)]
    rail_local_ranks: set[int] = set()
    remote_nodes_by_source = [set() for _ in range(nodes)]
    weight_values = tuple(int(value[2:], 16) for value in WEIGHT_FP32_HEX)

    for source_rank in range(world_size):
        source_node = source_rank // RANKS_PER_NODE
        for token in range(tokens):
            token_experts: set[int] = set()
            for slot in range(TOP_K):
                expert = _expected_expert(
                    pattern=checked_pattern,
                    nodes=nodes,
                    world_size=world_size,
                    tokens=tokens,
                    seed=checked_seed,
                    source_rank=source_rank,
                    token=token,
                    slot=slot,
                )
                _require(0 <= expert < NUM_EXPERTS, "generated expert is out of range")
                _require(
                    expert not in token_experts,
                    f"source rank {source_rank} token {token} has duplicate experts",
                )
                token_experts.add(expert)
                owner = expert // experts_per_owner
                local_expert = expert % experts_per_owner
                owner_counts[owner] += 1
                expert_counts[expert] += 1
                hot_locals[owner].add(local_expert)
                if checked_pattern == "rail_hot":
                    remote_node = owner // RANKS_PER_NODE
                    _require(
                        remote_node != source_node,
                        "rail_hot generated a same-node endpoint",
                    )
                    rail_local_ranks.add(owner % RANKS_PER_NODE)
                    remote_nodes_by_source[source_node].add(remote_node)
                expert_digest.update(struct.pack("<q", expert))
                weight_digest.update(struct.pack("<I", weight_values[slot]))
            _require(
                len(token_experts) == TOP_K,
                f"source rank {source_rank} token {token} is not top-k unique",
            )

    if checked_pattern == "balanced":
        _require(
            len(set(owner_counts)) == 1,
            "balanced route is not globally owner balanced",
        )
        _require(
            len(set(expert_counts)) == 1,
            "balanced route is not globally expert balanced",
        )
    elif checked_pattern == "expert_skew":
        _require(
            len(set(owner_counts)) == 1,
            "expert_skew route is not globally owner balanced",
        )
        expected_hot = set(range(HOT_EXPERTS_PER_OWNER))
        _require(
            all(locals_seen == expected_hot for locals_seen in hot_locals),
            "expert_skew does not use exactly two hot experts per owner",
        )
        _require(
            all(
                count_value == 0
                for owner in range(world_size)
                for count_value in expert_counts[
                    owner * experts_per_owner + HOT_EXPERTS_PER_OWNER : (owner + 1)
                    * experts_per_owner
                ]
            ),
            "expert_skew used a non-hot local expert",
        )
    else:
        expected_rail = {checked_seed % RANKS_PER_NODE}
        _require(
            rail_local_ranks == expected_rail,
            "rail_hot endpoints are not concentrated on one logical local rank",
        )
        for source_node, observed_remote_nodes in enumerate(remote_nodes_by_source):
            _require(
                observed_remote_nodes == set(range(nodes)) - {source_node},
                "rail_hot did not cover every remote node",
            )

    return expert_digest.hexdigest(), weight_digest.hexdigest(), count


def _tensor_metadata(*, expert_sha: str, weight_sha: str, count: int) -> dict[str, Any]:
    return {
        "expert_ids": {
            "count": count,
            "element_encoding": "LITTLE_ENDIAN_SIGNED_INT64",
            "streaming_sha256": expert_sha,
        },
        "topk_weights": {
            "count": count,
            "element_encoding": "LITTLE_ENDIAN_IEEE754_BINARY32_BITS",
            "streaming_sha256": weight_sha,
        },
    }


def generate_route_artifact(
    *,
    manifest_case_id: str,
    pattern: str,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Generate one compact route descriptor and streaming commitments."""

    mode, nodes, world_size, tokens, checked_pattern, checked_seed = (
        _validate_generation_args(manifest_case_id, pattern, seed)
    )
    expert_sha, weight_sha, count = _stream_summary(
        manifest_case_id=manifest_case_id,
        pattern=checked_pattern,
        seed=checked_seed,
    )
    artifact: dict[str, Any] = {
        "format": ROUTE_FORMAT,
        "version": ROUTE_VERSION,
        "case": {
            "id": _artifact_case_id(manifest_case_id, checked_pattern),
            "manifest_case_id": manifest_case_id,
            "mode": mode,
            "pattern": checked_pattern,
            "claim_scope": CLAIM_SCOPE,
            "route_artifact_claims_performance": False,
            "physical_evidence_status": _physical_evidence_status(checked_pattern),
            "physical_topology_or_counter_evidence_present": False,
            "nodes": nodes,
            "ranks_per_node": RANKS_PER_NODE,
            "world_size": world_size,
            "tokens_per_rank": tokens,
            "num_experts": NUM_EXPERTS,
            "hidden": HIDDEN,
            "top_k": TOP_K,
            "expert_owner_policy": OWNER_POLICY,
        },
        "shape": [world_size, tokens, TOP_K],
        "layout": LAYOUT,
        "dtypes": {
            "expert_ids": "INT64",
            "topk_weights": "IEEE754_BINARY32_BITS",
        },
        "algorithm": _algorithm_spec(checked_pattern, checked_seed),
        "weight_slot_fp32_bits": list(WEIGHT_FP32_HEX),
        "logical_tensors": _tensor_metadata(
            expert_sha=expert_sha,
            weight_sha=weight_sha,
            count=count,
        ),
        "canonical_sha256": "",
    }
    artifact["canonical_sha256"] = canonical_sha256(artifact)
    validate_route_artifact(artifact)
    return artifact


def _validate_case(value: object) -> tuple[str, int, int, int, str]:
    case = _exact_object(
        value,
        {
            "id",
            "manifest_case_id",
            "mode",
            "pattern",
            "claim_scope",
            "route_artifact_claims_performance",
            "physical_evidence_status",
            "physical_topology_or_counter_evidence_present",
            "nodes",
            "ranks_per_node",
            "world_size",
            "tokens_per_rank",
            "num_experts",
            "hidden",
            "top_k",
            "expert_owner_policy",
        },
        "case",
    )
    manifest_case_id = case["manifest_case_id"]
    pattern = case["pattern"]
    seed_placeholder = 0
    mode, nodes, world_size, tokens, checked_pattern, _ = _validate_generation_args(
        manifest_case_id, pattern, seed_placeholder
    )
    assert isinstance(manifest_case_id, str)
    _require(
        type(case["id"]) is str
        and case["id"] == _artifact_case_id(manifest_case_id, checked_pattern),
        "case.id changed",
    )
    expected_scalars = {
        "mode": mode,
        "nodes": nodes,
        "ranks_per_node": RANKS_PER_NODE,
        "world_size": world_size,
        "tokens_per_rank": tokens,
        "num_experts": NUM_EXPERTS,
        "hidden": HIDDEN,
        "top_k": TOP_K,
    }
    for field, expected in expected_scalars.items():
        observed = case[field]
        if type(expected) is int:
            _exact_int(observed, f"case.{field}", 1, MAX_SEED)
        _require(observed == expected, f"case.{field} changed")
    _require(case["claim_scope"] == CLAIM_SCOPE, "case.claim_scope changed")
    _require(
        case["route_artifact_claims_performance"] is False,
        "route artifact cannot claim performance",
    )
    _require(
        case["physical_evidence_status"] == _physical_evidence_status(checked_pattern),
        "case.physical_evidence_status changed",
    )
    _require(
        case["physical_topology_or_counter_evidence_present"] is False,
        "route artifact cannot claim physical topology or counter evidence",
    )
    _require(
        case["expert_owner_policy"] == OWNER_POLICY,
        "case.expert_owner_policy must be CONTIGUOUS_EQUAL",
    )
    return manifest_case_id, nodes, world_size, tokens, checked_pattern


def _validate_weight_bits(value: object) -> None:
    _require(type(value) is list and len(value) == TOP_K, "weight bits changed")
    assert isinstance(value, list)
    decoded: list[float] = []
    for index, item in enumerate(value):
        _require(
            type(item) is str and bool(_FP32_HEX.fullmatch(item)),
            f"weight_slot_fp32_bits[{index}] is not lowercase FP32 bits",
        )
        assert isinstance(item, str)
        number = struct.unpack(">f", bytes.fromhex(item[2:]))[0]
        _require(math.isfinite(number), f"weight_slot_fp32_bits[{index}] is non-finite")
        decoded.append(number)
    _require(tuple(value) == WEIGHT_FP32_HEX, "weight FP32 bit schedule changed")
    _require(sum(decoded) == 1.0, "top-k FP32 weights do not sum exactly to one")


def _validate_tensor_metadata(
    value: object,
    *,
    expected_expert_sha: str,
    expected_weight_sha: str,
    expected_count: int,
) -> None:
    tensors = _exact_object(value, {"expert_ids", "topk_weights"}, "logical_tensors")
    expected = {
        "expert_ids": (
            "LITTLE_ENDIAN_SIGNED_INT64",
            expected_expert_sha,
        ),
        "topk_weights": (
            "LITTLE_ENDIAN_IEEE754_BINARY32_BITS",
            expected_weight_sha,
        ),
    }
    for name, (encoding, digest) in expected.items():
        tensor = _exact_object(
            tensors[name],
            {"count", "element_encoding", "streaming_sha256"},
            f"logical_tensors.{name}",
        )
        _require(
            _exact_int(
                tensor["count"],
                f"logical_tensors.{name}.count",
                1,
                1 << 40,
            )
            == expected_count,
            f"logical_tensors.{name}.count changed",
        )
        _require(
            tensor["element_encoding"] == encoding,
            f"logical_tensors.{name}.element_encoding changed",
        )
        observed_digest = tensor["streaming_sha256"]
        _require(
            type(observed_digest) is str and bool(_SHA256.fullmatch(observed_digest)),
            f"logical_tensors.{name}.streaming_sha256 is invalid",
        )
        _require(
            observed_digest == digest,
            f"logical_tensors.{name}.streaming_sha256 mismatch",
        )


def validate_route_artifact(value: object) -> dict[str, Any]:
    """Strictly validate metadata and stream-regenerate every logical element."""

    artifact = _exact_object(
        value,
        {
            "format",
            "version",
            "case",
            "shape",
            "layout",
            "dtypes",
            "algorithm",
            "weight_slot_fp32_bits",
            "logical_tensors",
            "canonical_sha256",
        },
        "compact route",
    )
    _require(artifact["format"] == ROUTE_FORMAT, "compact route format changed")
    _require(
        _exact_int(artifact["version"], "route.version", 1, MAX_SEED) == ROUTE_VERSION,
        "route.version changed",
    )
    manifest_case_id, _, world_size, tokens, pattern = _validate_case(artifact["case"])
    shape = artifact["shape"]
    _require(type(shape) is list and len(shape) == 3, "route shape changed")
    assert isinstance(shape, list)
    for index, dimension in enumerate(shape):
        _exact_int(dimension, f"shape[{index}]", 1, MAX_SEED)
    _require(shape == [world_size, tokens, TOP_K], "route shape changed")
    _require(artifact["layout"] == LAYOUT, "route layout changed")
    dtypes = _exact_object(artifact["dtypes"], {"expert_ids", "topk_weights"}, "dtypes")
    _require(dtypes["expert_ids"] == "INT64", "expert ID dtype changed")
    _require(
        dtypes["topk_weights"] == "IEEE754_BINARY32_BITS",
        "top-k weight dtype changed",
    )
    algorithm = _exact_object(
        artifact["algorithm"],
        {
            "id",
            "version",
            "seed",
            "iteration_order",
            "owner_rank_formula",
            "weight_formula",
            "expert_id_formula",
            "pattern_invariant",
            "topology_formula",
        },
        "algorithm",
    )
    seed = _exact_int(algorithm["seed"], "algorithm.seed", 0, MAX_SEED)
    _require(
        algorithm == _algorithm_spec(pattern, seed),
        "algorithm specification changed",
    )
    _validate_weight_bits(artifact["weight_slot_fp32_bits"])
    expected_expert_sha, expected_weight_sha, expected_count = _stream_summary(
        manifest_case_id=manifest_case_id,
        pattern=pattern,
        seed=seed,
    )
    _validate_tensor_metadata(
        artifact["logical_tensors"],
        expected_expert_sha=expected_expert_sha,
        expected_weight_sha=expected_weight_sha,
        expected_count=expected_count,
    )
    digest = artifact["canonical_sha256"]
    _require(
        type(digest) is str and bool(_SHA256.fullmatch(digest)),
        "canonical_sha256 is not lowercase SHA256",
    )
    _require(digest == canonical_sha256(artifact), "canonical_sha256 mismatch")
    return artifact


def _directory_identity(metadata: Any) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _file_identity(metadata: Any) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_directory(metadata: Any, name: str, *, final: bool) -> None:
    _require(stat.S_ISDIR(metadata.st_mode), f"{name} must be a directory")
    mode = stat.S_IMODE(metadata.st_mode)
    if final:
        _require(metadata.st_uid == os.geteuid(), f"{name} must be owned by this user")
        _require(mode & 0o022 == 0, f"{name} must not be group/world writable")
        return
    _require(metadata.st_uid in {0, os.geteuid()}, f"{name} has an untrusted owner")
    if mode & 0o022:
        _require(
            metadata.st_uid == 0 and bool(metadata.st_mode & stat.S_ISVTX),
            f"{name} is an unsafe writable ancestor",
        )


def _open_parent(path: Path) -> tuple[int, str, tuple[int, ...]]:
    raw = os.fspath(path)
    _require(type(raw) is str and bool(raw), "route path must be nonempty")
    parts = Path(raw).parts
    _require(bool(parts), "route path must name a file")
    name = parts[-1]
    _require(name not in {"", ".", ".."}, "route path must name a safe file")
    directory_parts = parts[1:-1] if Path(raw).is_absolute() else parts[:-1]
    _require(
        all(part not in {"", ".", ".."} for part in directory_parts),
        "route path contains a non-canonical directory component",
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/" if Path(raw).is_absolute() else ".", flags)
    try:
        try:
            for index, component in enumerate(directory_parts):
                before = os.fstat(descriptor)
                _validate_directory(
                    before, f"route parent component {index}", final=False
                )
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
                try:
                    after = os.fstat(descriptor)
                    _require(
                        _directory_identity(before) == _directory_identity(after),
                        f"route parent component {index} changed during traversal",
                    )
                except BaseException:
                    os.close(next_descriptor)
                    raise
                os.close(descriptor)
                descriptor = next_descriptor
        except OSError as error:
            raise CommonFairCompactRouteError(
                f"cannot safely traverse route parent for {path}: {error}"
            ) from error
        metadata = os.fstat(descriptor)
        _validate_directory(metadata, "route parent", final=True)
        return descriptor, name, _directory_identity(metadata)
    except BaseException:
        os.close(descriptor)
        raise


def read_route_bytes(path: Path) -> bytes:
    """Read one bounded regular file with no link traversal or metadata race."""

    parent, name, parent_identity = _open_parent(path)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent,
            )
            parent_after = os.fstat(parent)
            _validate_directory(parent_after, "route parent", final=True)
            _require(
                _directory_identity(parent_after) == parent_identity,
                "route parent changed while opening the file",
            )
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
                descriptor = None
            raise CommonFairCompactRouteError(
                f"cannot safely open route {path}: {error}"
            ) from error
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
                descriptor = None
            raise
    finally:
        os.close(parent)
    assert descriptor is not None
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), "route must be a regular file")
        _require(before.st_uid == os.geteuid(), "route must be owned by this user")
        _require(before.st_nlink == 1, "route must have exactly one hard link")
        _require(stat.S_IMODE(before.st_mode) == 0o644, "route mode must be 0644")
        _require(
            0 < before.st_size <= MAX_ARTIFACT_BYTES,
            f"route must contain 1..{MAX_ARTIFACT_BYTES} bytes",
        )
        chunks: list[bytes] = []
        remaining = MAX_ARTIFACT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 16, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        _require(len(raw) <= MAX_ARTIFACT_BYTES, "route exceeds the byte limit")
        _require(len(raw) == after.st_size, "route read was incomplete")
        _require(
            _file_identity(before) == _file_identity(after),
            "route metadata, including ctime, changed while reading",
        )
        return raw
    finally:
        os.close(descriptor)


def load_route(path: Path) -> tuple[dict[str, Any], str, str]:
    """Load a strict canonical route and return raw plus canonical hashes."""

    raw = read_route_bytes(path)
    _require(not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM is forbidden")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except CommonFairCompactRouteError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as error:
        raise CommonFairCompactRouteError(f"invalid route JSON: {error}") from error
    artifact = validate_route_artifact(value)
    _require(
        raw == canonical_bytes(artifact) + b"\n",
        "route file does not use canonical UTF-8 bytes",
    )
    return artifact, hashlib.sha256(raw).hexdigest(), canonical_sha256(artifact)


def _same_inode(parent: int, name: str, expected: tuple[int, int]) -> bool:
    try:
        metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except OSError:
        return False
    return (metadata.st_dev, metadata.st_ino) == expected


def write_route(path: Path, artifact: object) -> tuple[str, str, int]:
    """Atomically publish a complete canonical route without replacing a path."""

    route = validate_route_artifact(artifact)
    payload = canonical_bytes(route) + b"\n"
    _require(
        len(payload) <= MAX_ARTIFACT_BYTES,
        "canonical route exceeds the byte limit",
    )
    parent, name, parent_identity = _open_parent(path)
    temp_name = f".{name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor: int | None = None
    temp_exists = False
    published = False
    inode: tuple[int, int] | None = None
    try:
        try:
            descriptor = os.open(
                temp_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | os.O_NOFOLLOW
                | os.O_NONBLOCK,
                0o600,
                dir_fd=parent,
            )
            temp_exists = True
        except OSError as error:
            raise CommonFairCompactRouteError(
                f"cannot safely create temporary route for {path}: {error}"
            ) from error
        os.fchmod(descriptor, 0o644)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "route write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        complete = os.fstat(descriptor)
        _require(stat.S_ISREG(complete.st_mode), "temporary route is not regular")
        _require(complete.st_uid == os.geteuid(), "temporary route owner changed")
        _require(complete.st_nlink == 1, "temporary route link count changed")
        _require(
            stat.S_IMODE(complete.st_mode) == 0o644, "temporary route mode changed"
        )
        _require(complete.st_size == len(payload), "temporary route size changed")
        inode = (complete.st_dev, complete.st_ino)
        parent_before_publish = os.fstat(parent)
        _validate_directory(parent_before_publish, "route parent", final=True)
        _require(
            _directory_identity(parent_before_publish)[:5] == parent_identity[:5],
            "route parent security metadata changed before publication",
        )
        try:
            os.link(
                temp_name,
                name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
            published = True
        except OSError as error:
            raise CommonFairCompactRouteError(
                f"cannot atomically publish route without replacement: {error}"
            ) from error
        os.unlink(temp_name, dir_fd=parent)
        temp_exists = False
        final_open = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent,
        )
        try:
            final_metadata = os.fstat(final_open)
            _require(
                (final_metadata.st_dev, final_metadata.st_ino) == inode,
                "published route inode changed",
            )
            _require(final_metadata.st_nlink == 1, "published route link count changed")
            _require(
                stat.S_IMODE(final_metadata.st_mode) == 0o644,
                "published route mode changed",
            )
            _require(
                final_metadata.st_size == len(payload), "published route size changed"
            )
        finally:
            os.close(final_open)
        os.close(descriptor)
        descriptor = None
        parent_after = os.fstat(parent)
        _validate_directory(parent_after, "route parent", final=True)
        _require(
            _directory_identity(parent_after)[:5] == parent_identity[:5],
            "route parent security metadata changed during publication",
        )
        os.fsync(parent)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if published and inode is not None and _same_inode(parent, name, inode):
            with contextlib.suppress(OSError):
                os.unlink(name, dir_fd=parent)
        if temp_exists:
            with contextlib.suppress(OSError):
                os.unlink(temp_name, dir_fd=parent)
        raise
    finally:
        os.close(parent)
    return hashlib.sha256(payload).hexdigest(), route["canonical_sha256"], len(payload)


def _parse_seed(value: str) -> int:
    if not _DECIMAL_SEED.fullmatch(value):
        raise argparse.ArgumentTypeError("seed must be a canonical nonnegative decimal")
    seed = int(value)
    if seed > MAX_SEED:
        raise argparse.ArgumentTypeError(f"seed must be <= {MAX_SEED}")
    return seed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate one compact route")
    generate.add_argument("--case", choices=tuple(CASE_SPECS), required=True)
    generate.add_argument("--pattern", choices=ROUTE_PATTERNS, required=True)
    generate.add_argument("--seed", type=_parse_seed, default=DEFAULT_SEED)
    generate.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify an existing compact route")
    verify.add_argument("route", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            artifact = generate_route_artifact(
                manifest_case_id=args.case,
                pattern=args.pattern,
                seed=args.seed,
            )
            raw_sha, canonical_sha, size = write_route(args.output, artifact)
            result = {
                "canonical_sha256": canonical_sha,
                "format": ROUTE_FORMAT,
                "manifest_case_id": artifact["case"]["manifest_case_id"],
                "path": os.fspath(args.output),
                "pattern": artifact["case"]["pattern"],
                "physical_evidence_status": artifact["case"][
                    "physical_evidence_status"
                ],
                "raw_sha256": raw_sha,
                "size_bytes": size,
                "status": "GENERATED",
            }
        else:
            artifact, raw_sha, canonical_sha = load_route(args.route)
            result = {
                "canonical_sha256": canonical_sha,
                "case_id": artifact["case"]["id"],
                "format": ROUTE_FORMAT,
                "manifest_case_id": artifact["case"]["manifest_case_id"],
                "path": os.fspath(args.route),
                "pattern": artifact["case"]["pattern"],
                "physical_evidence_status": artifact["case"][
                    "physical_evidence_status"
                ],
                "raw_sha256": raw_sha,
                "status": "VERIFIED",
            }
    except CommonFairCompactRouteError as error:
        print(f"COMMON_FAIR compact route error: {error}", file=sys.stderr)
        return 2
    print(canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
