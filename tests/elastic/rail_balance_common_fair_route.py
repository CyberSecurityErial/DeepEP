"""Canonical CPU-only route artifacts for COMMON_FAIR experiments.

The artifact stores logical ``[rank, token, top-k]`` expert IDs as JSON
integers and the corresponding IEEE-754 binary32 weight bits as lowercase
hexadecimal strings.  It intentionally has no CUDA, PyTorch, or DeepEP
dependency so routes can be frozen before scarce GPUs become available.
Every route is limited to derived scale-down correctness.  In particular,
``rail_hot`` is a logical owner-hotspot pattern, not physical-NIC evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
from pathlib import Path
from typing import Any


ROUTE_FORMAT = "rail-balance-logical-route-v1"
ROUTE_VERSION = 1
MAX_ROUTE_BYTES = 8 * 1024 * 1024
NUM_EXPERTS = 256
TOP_K = 8
OWNER_POLICY = "CONTIGUOUS_EQUAL"
LAYOUT = "RANK_TOKEN_TOPK_ROW_MAJOR"
MODE_TOKENS = {"HT": 4096, "LL": 128}
WORLD_SIZES = (2, 4)
ROUTE_PATTERNS = ("balanced", "expert_skew", "rail_hot")
CLAIM_SCOPE = "DERIVED_SCALE_DOWN_CORRECTNESS"
SEMANTIC_SCOPES = {
    "balanced": "LOGICAL_BALANCE_CORRECTNESS_ONLY",
    "expert_skew": "LOGICAL_OWNER_BALANCED_EXPERT_SKEW_CORRECTNESS_ONLY",
    "rail_hot": "LOGICAL_OWNER_HOTSPOT_NOT_PHYSICAL_NIC_EVIDENCE",
}
WEIGHT_FP32_HEX = (
    "0x3f000000",  # 1/2
    "0x3e800000",  # 1/4
    "0x3e000000",  # 1/8
    "0x3d800000",  # 1/16
    "0x3d000000",  # 1/32
    "0x3c800000",  # 1/64
    "0x3c000000",  # 1/128
    "0x3c000000",  # 1/128; the eight weights sum to exactly one
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FP32_HEX = re.compile(r"0x[0-9a-f]{8}\Z")


class CommonFairRouteError(ValueError):
    """Raised when a route artifact is ambiguous, unsafe, or non-canonical."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CommonFairRouteError(message)


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


def _exact_int(value: object, name: str, minimum: int) -> int:
    _require(type(value) is int and value >= minimum, f"{name} is invalid")
    assert isinstance(value, int)
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CommonFairRouteError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CommonFairRouteError(f"non-finite JSON constant is forbidden: {value}")


def canonical_bytes(value: object) -> bytes:
    """Serialize a finite JSON value with the frozen canonical encoding."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as error:
        raise CommonFairRouteError(
            f"value is not finite canonical JSON: {error}"
        ) from error


def _case_id(mode: str, world_size: int, pattern: str) -> str:
    return f"common-fair-{mode.lower()}-w{world_size}-{pattern}-v1"


def _manifest_case_id(mode: str, world_size: int) -> str:
    return f"derived-{mode.lower()}-n1-w{world_size}"


def _validate_generation_args(
    mode: object, world_size: object, pattern: object
) -> None:
    _require(type(mode) is str and mode in MODE_TOKENS, "mode must be HT or LL")
    _require(
        type(world_size) is int and world_size in WORLD_SIZES,
        "world_size must be exactly 2 or 4",
    )
    _require(
        type(pattern) is str and pattern in ROUTE_PATTERNS,
        f"pattern must be one of {ROUTE_PATTERNS}",
    )


def _expected_expert(
    pattern: str,
    world_size: int,
    source_rank: int,
    token: int,
    slot: int,
) -> int:
    experts_per_rank = NUM_EXPERTS // world_size
    if pattern == "balanced":
        owner = (source_rank + token + slot) % world_size
        occurrence = slot // world_size
        occurrences_per_owner = TOP_K // world_size
        local_expert = (
            token * occurrences_per_owner + occurrence + source_rank
        ) % experts_per_rank
        return owner * experts_per_rank + local_expert
    if pattern == "rail_hot":
        owner = (source_rank + 1) % world_size
        local_expert = (token * TOP_K + slot) % experts_per_rank
        return owner * experts_per_rank + local_expert
    assert pattern == "expert_skew"
    owner = (source_rank + slot) % world_size
    local_expert = slot // world_size
    return owner * experts_per_rank + local_expert


def _identity_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key != "canonical_sha256"}


def canonical_sha256(artifact: dict[str, Any]) -> str:
    """Hash the canonical artifact identity, excluding its self-hash field."""

    return hashlib.sha256(canonical_bytes(_identity_payload(artifact))).hexdigest()


def generate_route_artifact(
    *,
    mode: str,
    world_size: int,
    pattern: str,
) -> dict[str, Any]:
    """Build one deterministic expanded COMMON_FAIR logical route."""

    _validate_generation_args(mode, world_size, pattern)
    tokens = MODE_TOKENS[mode]
    expert_ids: list[int] = []
    weights: list[str] = []
    for rank in range(world_size):
        for token in range(tokens):
            for slot in range(TOP_K):
                expert_ids.append(
                    _expected_expert(pattern, world_size, rank, token, slot)
                )
                weights.append(WEIGHT_FP32_HEX[slot])

    artifact: dict[str, Any] = {
        "format": ROUTE_FORMAT,
        "version": ROUTE_VERSION,
        "case": {
            "id": _case_id(mode, world_size, pattern),
            "manifest_case_id": _manifest_case_id(mode, world_size),
            "mode": mode,
            "pattern": pattern,
            "claim_scope": CLAIM_SCOPE,
            "performance_claim_allowed": False,
            "semantic_scope": SEMANTIC_SCOPES[pattern],
            "world_size": world_size,
            "tokens_per_rank": tokens,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "expert_owner_policy": OWNER_POLICY,
        },
        "shape": [world_size, tokens, TOP_K],
        "layout": LAYOUT,
        "dtypes": {
            "expert_ids": "INT64",
            "topk_weights": "IEEE754_BINARY32_HEX_BITS",
        },
        "expert_ids": expert_ids,
        "topk_weights_fp32_hex": weights,
        "canonical_sha256": "",
    }
    artifact["canonical_sha256"] = canonical_sha256(artifact)
    validate_route_artifact(artifact)
    return artifact


def _validate_case(value: object) -> tuple[str, int, str, int]:
    case = _exact_object(
        value,
        {
            "id",
            "manifest_case_id",
            "mode",
            "pattern",
            "claim_scope",
            "performance_claim_allowed",
            "semantic_scope",
            "world_size",
            "tokens_per_rank",
            "num_experts",
            "top_k",
            "expert_owner_policy",
        },
        "case",
    )
    mode = case["mode"]
    world_size = case["world_size"]
    pattern = case["pattern"]
    _validate_generation_args(mode, world_size, pattern)
    assert isinstance(mode, str)
    assert isinstance(world_size, int)
    assert isinstance(pattern, str)
    tokens = MODE_TOKENS[mode]
    _require(
        type(case["id"]) is str and case["id"] == _case_id(mode, world_size, pattern),
        "case.id changed",
    )
    _require(
        type(case["manifest_case_id"]) is str
        and case["manifest_case_id"] == _manifest_case_id(mode, world_size),
        "case.manifest_case_id changed",
    )
    _require(
        type(case["claim_scope"]) is str and case["claim_scope"] == CLAIM_SCOPE,
        "case.claim_scope must remain derived correctness only",
    )
    _require(
        case["performance_claim_allowed"] is False,
        "case.performance_claim_allowed must remain false",
    )
    _require(
        type(case["semantic_scope"]) is str
        and case["semantic_scope"] == SEMANTIC_SCOPES[pattern],
        "case.semantic_scope changed",
    )
    _require(
        _exact_int(case["tokens_per_rank"], "case.tokens_per_rank", 1) == tokens,
        "case.tokens_per_rank changed",
    )
    _require(
        _exact_int(case["num_experts"], "case.num_experts", 1) == NUM_EXPERTS,
        "case.num_experts must be 256",
    )
    _require(
        _exact_int(case["top_k"], "case.top_k", 1) == TOP_K,
        "case.top_k must be 8",
    )
    _require(
        type(case["expert_owner_policy"]) is str
        and case["expert_owner_policy"] == OWNER_POLICY,
        "case.expert_owner_policy must be CONTIGUOUS_EQUAL",
    )
    return mode, world_size, pattern, tokens


def _validate_fp32_hex(value: object, name: str) -> str:
    _require(
        type(value) is str and bool(_FP32_HEX.fullmatch(value)),
        f"{name} is not lowercase FP32 hexadecimal bits",
    )
    assert isinstance(value, str)
    decoded = struct.unpack(">f", bytes.fromhex(value[2:]))[0]
    _require(math.isfinite(decoded), f"{name} is non-finite")
    return value


def validate_route_artifact(value: object) -> dict[str, Any]:
    """Validate the exact schema, deterministic data, and self-hash."""

    artifact = _exact_object(
        value,
        {
            "format",
            "version",
            "case",
            "shape",
            "layout",
            "dtypes",
            "expert_ids",
            "topk_weights_fp32_hex",
            "canonical_sha256",
        },
        "route",
    )
    _require(
        type(artifact["format"]) is str and artifact["format"] == ROUTE_FORMAT,
        "route format changed",
    )
    _require(
        _exact_int(artifact["version"], "route.version", 1) == ROUTE_VERSION,
        "route version changed",
    )
    _, world_size, pattern, tokens = _validate_case(artifact["case"])
    shape = artifact["shape"]
    _require(type(shape) is list and len(shape) == 3, "route shape changed")
    assert isinstance(shape, list)
    for index, dimension in enumerate(shape):
        _exact_int(dimension, f"route.shape[{index}]", 1)
    _require(
        shape == [world_size, tokens, TOP_K],
        "route shape changed",
    )
    _require(
        type(artifact["layout"]) is str and artifact["layout"] == LAYOUT,
        "route layout changed",
    )
    dtypes = _exact_object(artifact["dtypes"], {"expert_ids", "topk_weights"}, "dtypes")
    _require(
        type(dtypes["expert_ids"]) is str and dtypes["expert_ids"] == "INT64",
        "expert ID dtype changed",
    )
    _require(
        type(dtypes["topk_weights"]) is str
        and dtypes["topk_weights"] == "IEEE754_BINARY32_HEX_BITS",
        "top-k weight encoding changed",
    )

    count = world_size * tokens * TOP_K
    expert_ids = artifact["expert_ids"]
    weights = artifact["topk_weights_fp32_hex"]
    _require(type(expert_ids) is list, "expert_ids must be a list")
    _require(type(weights) is list, "topk_weights_fp32_hex must be a list")
    assert isinstance(expert_ids, list)
    assert isinstance(weights, list)
    _require(len(expert_ids) == count, "expert_ids length differs from shape")
    _require(len(weights) == count, "top-k weights length differs from shape")

    for rank in range(world_size):
        for token in range(tokens):
            offset = (rank * tokens + token) * TOP_K
            token_experts = expert_ids[offset : offset + TOP_K]
            for slot, expert in enumerate(token_experts):
                _exact_int(expert, f"expert_ids[{offset + slot}]", 0)
                _require(
                    expert < NUM_EXPERTS,
                    f"expert_ids[{offset + slot}] is outside [0, 256)",
                )
            _require(
                len(set(token_experts)) == TOP_K,
                f"rank {rank} token {token} contains duplicate experts",
            )
            for slot, expert in enumerate(token_experts):
                expected = _expected_expert(pattern, world_size, rank, token, slot)
                _require(
                    expert == expected,
                    f"expert_ids[{offset + slot}] differs from deterministic route",
                )

    for index, weight in enumerate(weights):
        observed = _validate_fp32_hex(weight, f"topk_weights_fp32_hex[{index}]")
        expected = WEIGHT_FP32_HEX[index % TOP_K]
        _require(observed == expected, f"topk_weights_fp32_hex[{index}] changed")

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


def _validate_directory(metadata: Any, name: str, *, final: bool) -> None:
    _require(stat.S_ISDIR(metadata.st_mode), f"{name} must be a directory")
    mode = stat.S_IMODE(metadata.st_mode)
    if final:
        _require(metadata.st_uid == os.geteuid(), f"{name} must be owned by this user")
        _require(mode & 0o022 == 0, f"{name} must not be group/world writable")
        return
    _require(
        metadata.st_uid in {0, os.geteuid()},
        f"{name} has an untrusted owner",
    )
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
                    before,
                    f"route parent component {index}",
                    final=False,
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
            raise CommonFairRouteError(
                f"cannot safely traverse route parent for {path}: {error}"
            ) from error
        metadata = os.fstat(descriptor)
        _validate_directory(metadata, "route parent", final=True)
        return descriptor, name, _directory_identity(metadata)
    except BaseException:
        os.close(descriptor)
        raise


def read_route_bytes(path: Path) -> bytes:
    """Read a bounded, owner-controlled regular file without following links."""

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
                "route parent changed while the file was being opened",
            )
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
                descriptor = None
            raise CommonFairRouteError(
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
        _require(
            stat.S_IMODE(before.st_mode) & 0o022 == 0,
            "route must not be group/world writable",
        )
        _require(
            0 < before.st_size <= MAX_ROUTE_BYTES,
            f"route must contain 1..{MAX_ROUTE_BYTES} bytes",
        )
        chunks: list[bytes] = []
        remaining = MAX_ROUTE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        _require(len(raw) <= MAX_ROUTE_BYTES, "route exceeds the byte limit")
        _require(len(raw) == after.st_size, "route read was incomplete")
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
            "route changed while it was being read",
        )
        return raw
    finally:
        os.close(descriptor)


def load_route(path: Path) -> tuple[dict[str, Any], str, str]:
    """Safely load a strict route and return it with raw and canonical hashes."""

    raw = read_route_bytes(path)
    _require(not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM is forbidden")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except CommonFairRouteError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as error:
        raise CommonFairRouteError(f"invalid route JSON: {error}") from error
    artifact = validate_route_artifact(value)
    _require(
        raw == canonical_bytes(artifact) + b"\n",
        "route file does not use the canonical byte encoding",
    )
    return (
        artifact,
        hashlib.sha256(raw).hexdigest(),
        canonical_sha256(artifact),
    )


def write_route(path: Path, artifact: object) -> tuple[str, str, int]:
    """Create a canonical route without following or replacing any path."""

    route = validate_route_artifact(artifact)
    payload = canonical_bytes(route) + b"\n"
    _require(len(payload) <= MAX_ROUTE_BYTES, "canonical route exceeds the byte limit")
    parent, name, parent_identity = _open_parent(path)
    descriptor: int | None = None
    created = False
    try:
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o644,
                dir_fd=parent,
            )
            created = True
            parent_after = os.fstat(parent)
            _validate_directory(parent_after, "route parent", final=True)
            _require(
                _directory_identity(parent_after)[:5] == parent_identity[:5],
                "route parent security metadata changed while creating the file",
            )
            created_parent_identity = _directory_identity(parent_after)
        except OSError as error:
            raise CommonFairRouteError(
                f"cannot safely create route {path}: {error}"
            ) from error
        os.fchmod(descriptor, 0o644)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "route write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        output_metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(output_metadata.st_mode), "created route is not regular")
        _require(output_metadata.st_uid == os.geteuid(), "created route owner changed")
        _require(output_metadata.st_nlink == 1, "created route link count changed")
        _require(
            stat.S_IMODE(output_metadata.st_mode) == 0o644,
            "created route mode changed",
        )
        _require(output_metadata.st_size == len(payload), "created route size changed")
        os.close(descriptor)
        descriptor = None
        parent_after_write = os.fstat(parent)
        _validate_directory(parent_after_write, "route parent", final=True)
        _require(
            _directory_identity(parent_after_write) == created_parent_identity,
            "route parent changed while writing the file",
        )
        os.fsync(parent)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            with contextlib.suppress(OSError):
                os.unlink(name, dir_fd=parent)
        raise
    finally:
        os.close(parent)
    return (
        hashlib.sha256(payload).hexdigest(),
        route["canonical_sha256"],
        len(payload),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate one frozen route")
    generate.add_argument("--mode", choices=tuple(MODE_TOKENS), required=True)
    generate.add_argument("--world-size", choices=WORLD_SIZES, type=int, required=True)
    generate.add_argument("--pattern", choices=ROUTE_PATTERNS, required=True)
    generate.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify an existing route")
    verify.add_argument("route", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            artifact = generate_route_artifact(
                mode=args.mode,
                world_size=args.world_size,
                pattern=args.pattern,
            )
            raw_sha, canonical_sha, size = write_route(args.output, artifact)
            result = {
                "canonical_sha256": canonical_sha,
                "claim_scope": artifact["case"]["claim_scope"],
                "manifest_case_id": artifact["case"]["manifest_case_id"],
                "path": os.fspath(args.output),
                "performance_claim_allowed": artifact["case"][
                    "performance_claim_allowed"
                ],
                "raw_sha256": raw_sha,
                "semantic_scope": artifact["case"]["semantic_scope"],
                "size_bytes": size,
                "status": "GENERATED",
            }
        else:
            artifact, raw_sha, canonical_sha = load_route(args.route)
            result = {
                "canonical_sha256": canonical_sha,
                "case_id": artifact["case"]["id"],
                "claim_scope": artifact["case"]["claim_scope"],
                "manifest_case_id": artifact["case"]["manifest_case_id"],
                "path": os.fspath(args.route),
                "performance_claim_allowed": artifact["case"][
                    "performance_claim_allowed"
                ],
                "raw_sha256": raw_sha,
                "semantic_scope": artifact["case"]["semantic_scope"],
                "status": "VERIFIED",
            }
    except CommonFairRouteError as error:
        print(f"COMMON_FAIR route error: {error}", file=sys.stderr)
        return 2
    print(canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
