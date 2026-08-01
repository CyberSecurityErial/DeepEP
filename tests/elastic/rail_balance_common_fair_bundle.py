"""Fail-closed CPU-only E2E verifier for COMMON_FAIR artifact bundles.

The verifier deliberately performs no build, GPU launch, CUDA call, or
benchmark.  A bundle is a directory containing ``bundle.json`` plus the files
named by that self-hashed index.  Successful validation means only that the
design manifest, deterministic routes, payload generator, and optional raw
results form one hash-closed artifact set.  It never authorizes execution and
never promotes a raw result to ``PASSED_BENCHMARK``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import rail_balance_common_fair_payload as payload
import rail_balance_common_fair_results as results
import rail_balance_common_fair_route as expanded_route
import rail_balance_common_fair_route_compact as compact_route
import rail_balance_common_fair_schema as schema


BUNDLE_FORMAT = "rail-balance-common-fair-cpu-bundle-v1"
BUNDLE_VERSION = 1
BUNDLE_INDEX_NAME = "bundle.json"
REPORT_FORMAT = "rail-balance-common-fair-bundle-report-v1"
MAX_INDEX_BYTES = 1 << 20
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 2_000_000
RAW_FILE_SHA256 = "RAW_FILE_BYTES_SHA256"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._+-]{0,254}\Z")


class CommonFairBundleError(ValueError):
    """Raised when a COMMON_FAIR bundle is unsafe, incomplete, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CommonFairBundleError(message)


def _exact(value: object, fields: set[str], name: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{name} must be an object")
    assert isinstance(value, dict)
    observed = set(value)
    _require(
        observed == fields,
        f"{name} fields differ: missing={sorted(fields - observed)}, "
        f"extra={sorted(observed - fields)}",
    )
    return value


def _string(value: object, name: str) -> str:
    _require(type(value) is str and bool(value), f"{name} must be nonempty text")
    assert isinstance(value, str)
    _require(
        all(ord(character) >= 0x20 and ord(character) != 0x7F for character in value),
        f"{name} contains a control character",
    )
    return value


def _sha256(value: object, name: str) -> str:
    checked = _string(value, name)
    _require(bool(_SHA256.fullmatch(checked)), f"{name} is not lowercase SHA256")
    return checked


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    _require(
        type(value) is int and minimum <= value <= maximum,
        f"{name} is invalid",
    )
    assert isinstance(value, int)
    return value


def _relative_path(value: object, name: str) -> str:
    checked = _string(value, name)
    path = PurePosixPath(checked)
    _require(not path.is_absolute(), f"{name} must be relative to the bundle")
    _require(checked == path.as_posix(), f"{name} is not normalized")
    _require(
        bool(path.parts)
        and all(part not in {"", ".", ".."} for part in path.parts)
        and all(_PATH_COMPONENT.fullmatch(part) for part in path.parts),
        f"{name} contains an unsafe path component",
    )
    return checked


def canonical_bytes(value: object) -> bytes:
    """Return the finite, sorted canonical JSON encoding used for bundle IDs."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as error:
        raise CommonFairBundleError(
            f"value is not canonical finite JSON: {error}"
        ) from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CommonFairBundleError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise CommonFairBundleError(f"non-finite JSON constant is forbidden: {value}")


def _parse_integer(value: str) -> int:
    if len(value.lstrip("-")) > 128:
        raise CommonFairBundleError("JSON integer exceeds 128 decimal digits")
    return int(value)


def _parse_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CommonFairBundleError(f"non-finite JSON number is forbidden: {value}")
    return result


def _check_json_shape(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        _require(nodes <= MAX_JSON_NODES, "JSON contains too many values")
        _require(depth <= MAX_JSON_DEPTH, "JSON nesting exceeds the depth limit")
        if type(item) is dict:
            assert isinstance(item, dict)
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            assert isinstance(item, list)
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            assert isinstance(item, float)
            _require(math.isfinite(item), "non-finite JSON number is forbidden")


def _parse_json(raw: bytes, name: str) -> object:
    _require(not raw.startswith(b"\xef\xbb\xbf"), f"{name} UTF-8 BOM is forbidden")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_int=_parse_integer,
            parse_float=_parse_float,
        )
    except CommonFairBundleError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as error:
        raise CommonFairBundleError(f"invalid {name} JSON: {error}") from error
    _check_json_shape(value)
    return value


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


def _open_directory(path: Path) -> tuple[int, tuple[int, ...]]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require(absolute != Path("/"), "bundle root must not be the filesystem root")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for index, component in enumerate(absolute.parts[1:]):
            before = os.fstat(descriptor)
            _validate_directory(before, f"bundle ancestor {index}", final=False)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            try:
                after = os.fstat(descriptor)
                _require(
                    _directory_identity(before) == _directory_identity(after),
                    f"bundle ancestor {index} changed during traversal",
                )
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        _validate_directory(metadata, "bundle root", final=True)
        return descriptor, _directory_identity(metadata)
    except OSError as error:
        os.close(descriptor)
        raise CommonFairBundleError(
            f"cannot safely open bundle directory: {error}"
        ) from error
    except BaseException:
        os.close(descriptor)
        raise


def _read_relative(
    root_descriptor: int,
    relative: str,
    *,
    maximum_bytes: int,
    name: str,
) -> bytes:
    checked = _relative_path(relative, f"{name}.path")
    parts = PurePosixPath(checked).parts
    parent = os.dup(root_descriptor)
    descriptor: int | None = None
    try:
        for index, component in enumerate(parts[:-1]):
            before = os.fstat(parent)
            _validate_directory(before, f"{name} parent {index}", final=True)
            next_parent = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            try:
                after = os.fstat(parent)
                _require(
                    _directory_identity(before) == _directory_identity(after),
                    f"{name} parent changed during traversal",
                )
            except BaseException:
                os.close(next_parent)
                raise
            os.close(parent)
            parent = next_parent
        parent_before = os.fstat(parent)
        _validate_directory(parent_before, f"{name} parent", final=True)
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent,
        )
        parent_after_open = os.fstat(parent)
        _require(
            _directory_identity(parent_before)
            == _directory_identity(parent_after_open),
            f"{name} parent changed while opening the file",
        )
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{name} must be a regular file")
        _require(before.st_uid == os.geteuid(), f"{name} must be owned by this user")
        _require(before.st_nlink == 1, f"{name} must have exactly one hard link")
        _require(stat.S_IMODE(before.st_mode) == 0o644, f"{name} mode must be 0644")
        _require(
            0 < before.st_size <= maximum_bytes,
            f"{name} must contain 1..{maximum_bytes} bytes",
        )
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        _require(len(raw) <= maximum_bytes, f"{name} exceeds its byte limit")
        _require(len(raw) == after.st_size, f"{name} read was incomplete")
        _require(
            _file_identity(before) == _file_identity(after),
            f"{name} changed while reading",
        )
        parent_after_read = os.fstat(parent)
        _require(
            _directory_identity(parent_before)
            == _directory_identity(parent_after_read),
            f"{name} parent changed while reading",
        )
        return raw
    except OSError as error:
        raise CommonFairBundleError(f"cannot safely read {name}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: item for key, item in value.items() if key != field})
    ).hexdigest()


def bundle_sha256(index: Mapping[str, Any]) -> str:
    """Return the index identity excluding only ``bundle_sha256`` itself."""

    return _self_hash(index, "bundle_sha256")


def report_sha256(report: Mapping[str, Any]) -> str:
    """Return the validation-report identity excluding its self-hash field."""

    return _self_hash(report, "report_sha256")


def _binding(value: object, name: str, fields: set[str]) -> dict[str, Any]:
    row = _exact(value, fields, name)
    for field in fields & {
        "raw_sha256",
        "canonical_sha256",
        "source_raw_sha256",
        "specification_sha256",
        "evidence_sha256",
    }:
        _sha256(row[field], f"{name}.{field}")
    return row


def _validate_module_contracts() -> dict[str, object]:
    _require(results.SYSTEM_IDS == schema.SYSTEM_IDS, "result/system constants drifted")
    _require(
        results.FIXED_CANDIDATE_IDS == schema.FIXED_CANDIDATE_IDS,
        "candidate IDs drifted",
    )
    _require(results.PHASE_IDS == schema.PHASE_IDS, "result phase IDs drifted")
    _require(
        results.ROUTE_PATTERNS == schema.ROUTE_PATTERNS, "result route patterns drifted"
    )
    _require(results.RESULT_CODES == schema.RESULT_CODES, "result codes drifted")
    _require(
        results.RAW_SAMPLE_FIELDS == schema.RAW_SAMPLE_FIELDS,
        "raw sample fields drifted",
    )
    _require(results.CASE_SPECS == schema.CASE_SPECS, "result case specs drifted")
    _require(results.PAIR_SPECS == schema.PAIR_SPECS, "result pair specs drifted")
    _require(
        expanded_route.ROUTE_PATTERNS == schema.ROUTE_PATTERNS,
        "expanded route patterns drifted",
    )
    _require(
        compact_route.ROUTE_PATTERNS == schema.ROUTE_PATTERNS,
        "compact route patterns drifted",
    )
    expected_confirmatory = {
        case_id: (mode, nodes, world_size, 128 if mode == "LL" else 4096)
        for case_id, (
            scope,
            mode,
            nodes,
            _ranks,
            world_size,
        ) in schema.CASE_SPECS.items()
        if scope == "CONFIRMATORY_COMMON_FAIR"
    }
    _require(
        expected_confirmatory == compact_route.CASE_SPECS, "compact case specs drifted"
    )
    _require(
        payload.GENERATOR_ID == "rail-balance-deterministic-bf16-v1",
        "payload ID drifted",
    )
    _require(payload.SEED == 20260801, "payload seed drifted")
    identity = payload.specification_identity()
    check = payload.self_check()
    _require(check.get("status") == "PASS", "payload self-check did not pass")
    _require(
        check.get("generator_id") == payload.GENERATOR_ID,
        "payload self-check ID drifted",
    )
    _require(check.get("seed") == payload.SEED, "payload self-check seed drifted")
    _require(
        "PASSED_BENCHMARK" not in results.TERMINAL_STAGE_BY_CODE,
        "raw result can self-promote",
    )
    _require(
        results.TERMINAL_STAGE_BY_CODE.get("RAW_SAMPLES_COMPLETE_UNVERIFIED")
        == "RAW_SAMPLE_CAPTURE",
        "raw-result terminal state drifted",
    )
    return identity


def _load_json_artifact(
    root_descriptor: int,
    path: str,
    *,
    maximum_bytes: int,
    name: str,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_relative(
        root_descriptor,
        path,
        maximum_bytes=maximum_bytes,
        name=name,
    )
    value = _parse_json(raw, name)
    _require(type(value) is dict, f"{name} root must be an object")
    assert isinstance(value, dict)
    return value, raw


def _validate_index(value: object) -> dict[str, Any]:
    index = _exact(
        value,
        {
            "format",
            "version",
            "manifest",
            "payload_generator",
            "routes",
            "results",
            "bundle_sha256",
        },
        "bundle index",
    )
    _require(index["format"] == BUNDLE_FORMAT, "bundle format is unsupported")
    _require(
        _integer(index["version"], "bundle.version", 1, 1) == BUNDLE_VERSION,
        "bundle version changed",
    )
    manifest = _binding(
        index["manifest"],
        "bundle.manifest",
        {"path", "raw_sha256", "canonical_sha256"},
    )
    _relative_path(manifest["path"], "bundle.manifest.path")
    generator = _binding(
        index["payload_generator"],
        "bundle.payload_generator",
        {
            "id",
            "seed",
            "source_path",
            "source_raw_sha256",
            "source_sha256_kind",
            "specification_sha256",
        },
    )
    _string(generator["id"], "bundle.payload_generator.id")
    _integer(generator["seed"], "bundle.payload_generator.seed", 0, (1 << 63) - 1)
    _relative_path(generator["source_path"], "bundle.payload_generator.source_path")
    _require(
        generator["source_sha256_kind"] == RAW_FILE_SHA256,
        "payload source hash must bind raw file bytes",
    )
    routes = index["routes"]
    _require(type(routes) is list, "bundle.routes must be a list")
    assert isinstance(routes, list)
    route_keys: list[tuple[str, str]] = []
    paths = {manifest["path"], generator["source_path"]}
    for position, value in enumerate(routes):
        name = f"bundle.routes[{position}]"
        route = _binding(
            value,
            name,
            {
                "case_id",
                "pattern",
                "format",
                "semantic_scope",
                "path",
                "raw_sha256",
                "canonical_sha256",
            },
        )
        case_id = _string(route["case_id"], f"{name}.case_id")
        pattern = _string(route["pattern"], f"{name}.pattern")
        _string(route["format"], f"{name}.format")
        _string(route["semantic_scope"], f"{name}.semantic_scope")
        path = _relative_path(route["path"], f"{name}.path")
        _require(path not in paths, f"{name}.path is reused")
        paths.add(path)
        route_keys.append((case_id, pattern))
    expected_keys = sorted(
        (case_id, pattern)
        for case_id in schema.CASE_SPECS
        for pattern in schema.ROUTE_PATTERNS
    )
    _require(
        route_keys == expected_keys, "bundle must contain the exact sorted 8 x 3 routes"
    )
    result_entries = index["results"]
    _require(type(result_entries) is list, "bundle.results must be a list")
    assert isinstance(result_entries, list)
    result_keys: list[tuple[str, str]] = []
    for position, value in enumerate(result_entries):
        name = f"bundle.results[{position}]"
        entry = _binding(
            value,
            name,
            {
                "row_id",
                "result_code",
                "path",
                "raw_sha256",
                "canonical_sha256",
                "evidence_sha256",
            },
        )
        row_id = _string(entry["row_id"], f"{name}.row_id")
        _string(entry["result_code"], f"{name}.result_code")
        path = _relative_path(entry["path"], f"{name}.path")
        _require(path not in paths, f"{name}.path is reused")
        paths.add(path)
        result_keys.append((row_id, path))
    _require(
        result_keys == sorted(set(result_keys)),
        "bundle.results must be sorted and uniquely identified by row/path",
    )
    observed_hash = _sha256(index["bundle_sha256"], "bundle.bundle_sha256")
    _require(observed_hash == bundle_sha256(index), "bundle_sha256 mismatch")
    return index


def _manifest_route_map(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]:
    mapped: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    cases = manifest["cases"]
    assert isinstance(cases, list)
    for case_value in cases:
        assert isinstance(case_value, dict)
        route_values = case_value["routes"]
        assert isinstance(route_values, list)
        for route_value in route_values:
            assert isinstance(route_value, dict)
            mapped[(case_value["id"], route_value["pattern"])] = (
                case_value,
                route_value,
            )
    return mapped


def _validate_one_route(
    *,
    artifact: dict[str, Any],
    entry: Mapping[str, Any],
    case: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> None:
    case_id = str(case["id"])
    pattern = str(declaration["pattern"])
    scope = str(case["claim_scope"])
    expected_format = (
        expanded_route.ROUTE_FORMAT
        if scope == "DERIVED_SCALE_DOWN_CORRECTNESS"
        else compact_route.ROUTE_FORMAT
    )
    _require(
        entry["format"] == declaration["format"] == expected_format,
        "route format binding mismatch",
    )
    _require(
        entry["semantic_scope"] == declaration["semantic_scope"],
        "route semantic scope binding mismatch",
    )
    artifact_case = artifact.get("case")
    _require(type(artifact_case) is dict, "route case is missing")
    assert isinstance(artifact_case, dict)
    for field, expected in {
        "manifest_case_id": case_id,
        "mode": case["mode"],
        "world_size": case["world_size"],
        "pattern": pattern,
    }.items():
        _require(
            artifact_case.get(field) == expected, f"route case.{field} binding mismatch"
        )
    if scope == "DERIVED_SCALE_DOWN_CORRECTNESS":
        expanded_route.validate_route_artifact(artifact)
        _require(
            case["stage_limit"] == "CORRECTNESS",
            "derived route escaped correctness stage",
        )
        _require(
            case["future_claim_eligible"] is False,
            "derived route became claim eligible",
        )
        _require(
            artifact_case.get("claim_scope") == expanded_route.CLAIM_SCOPE,
            "expanded claim scope drifted",
        )
        _require(
            artifact_case.get("performance_claim_allowed") is False,
            "expanded route claims performance",
        )
        _require(
            artifact_case.get("semantic_scope") == declaration["semantic_scope"],
            "expanded route semantic scope mismatch",
        )
    else:
        compact_route.validate_route_artifact(artifact)
        _require(
            artifact_case.get("claim_scope") == compact_route.CLAIM_SCOPE,
            "compact claim scope drifted",
        )
        _require(
            artifact_case.get("route_artifact_claims_performance") is False,
            "compact route claims performance",
        )
        _require(
            artifact_case.get("physical_topology_or_counter_evidence_present") is False,
            "compact route improperly carries physical evidence",
        )
        expected_physical = (
            compact_route.RAIL_VERIFICATION_REQUIRED
            if pattern == "rail_hot"
            else "LOGICAL_PATTERN_DOES_NOT_ASSERT_PHYSICAL_PLACEMENT"
        )
        _require(
            artifact_case.get("physical_evidence_status") == expected_physical,
            "compact physical-evidence scope mismatch",
        )
        algorithm = artifact.get("algorithm")
        _require(type(algorithm) is dict, "compact route algorithm is missing")
        assert isinstance(algorithm, dict)
        _require(
            algorithm.get("seed") == compact_route.DEFAULT_SEED,
            "compact route seed is not frozen",
        )


def _current_payload_source_bytes() -> bytes:
    source = Path(payload.__file__)
    descriptor, _identity = _open_directory(source.parent)
    try:
        return _read_relative(
            descriptor,
            source.name,
            maximum_bytes=1 << 20,
            name="installed payload source",
        )
    finally:
        os.close(descriptor)


def validate_bundle(bundle_directory: Path) -> dict[str, Any]:
    """Validate one bundle and return a self-hashed, non-promoting report."""

    identity = _validate_module_contracts()
    root_descriptor, root_identity = _open_directory(bundle_directory)
    try:
        index_value, index_raw = _load_json_artifact(
            root_descriptor,
            BUNDLE_INDEX_NAME,
            maximum_bytes=MAX_INDEX_BYTES,
            name="bundle index",
        )
        index = _validate_index(index_value)
        _require(
            index_raw == canonical_bytes(index) + b"\n",
            "bundle index does not use canonical UTF-8 bytes",
        )

        manifest_binding = index["manifest"]
        assert isinstance(manifest_binding, dict)
        manifest, manifest_raw = _load_json_artifact(
            root_descriptor,
            manifest_binding["path"],
            maximum_bytes=schema.MAX_MANIFEST_BYTES,
            name="design manifest",
        )
        try:
            schema.validate_manifest(manifest)
        except schema.CommonFairError as error:
            raise CommonFairBundleError(f"design manifest rejected: {error}") from error
        manifest_raw_sha = hashlib.sha256(manifest_raw).hexdigest()
        manifest_canonical_sha = hashlib.sha256(
            schema.canonical_bytes(manifest)
        ).hexdigest()
        _require(
            manifest_binding["raw_sha256"] == manifest_raw_sha,
            "manifest raw SHA256 mismatch",
        )
        _require(
            manifest_binding["canonical_sha256"] == manifest_canonical_sha,
            "manifest canonical SHA256 mismatch",
        )

        generator_binding = index["payload_generator"]
        assert isinstance(generator_binding, dict)
        source_raw = _read_relative(
            root_descriptor,
            generator_binding["source_path"],
            maximum_bytes=1 << 20,
            name="payload source",
        )
        source_sha = hashlib.sha256(source_raw).hexdigest()
        _require(
            source_sha == generator_binding["source_raw_sha256"],
            "payload source SHA256 mismatch",
        )
        _require(
            source_raw == _current_payload_source_bytes(),
            "payload source differs from the verified implementation",
        )
        specification_sha = hashlib.sha256(canonical_bytes(identity)).hexdigest()
        _require(
            generator_binding["specification_sha256"] == specification_sha,
            "payload specification SHA256 mismatch",
        )
        _require(
            generator_binding["id"] == payload.GENERATOR_ID,
            "payload generator ID mismatch",
        )
        _require(
            generator_binding["seed"] == payload.SEED, "payload generator seed mismatch"
        )

        cases = manifest["cases"]
        assert isinstance(cases, list)
        for case in cases:
            assert isinstance(case, dict)
            generator = case["payload_generator"]
            assert isinstance(generator, dict)
            _require(
                generator["id"] == generator_binding["id"],
                "manifest payload ID mismatch",
            )
            _require(
                generator["seed"] == generator_binding["seed"],
                "manifest payload seed mismatch",
            )
            if generator["status"] == "PINNED":
                _require(
                    generator["code_path"] == generator_binding["source_path"],
                    "manifest payload path mismatch",
                )
                _require(
                    generator["code_sha256"] == source_sha,
                    "manifest payload SHA256 mismatch",
                )
            else:
                _require(
                    manifest["manifest_phase"] == "DESIGN_ONLY",
                    "unfrozen payload escaped design phase",
                )

        manifest_routes = _manifest_route_map(manifest)
        route_entries = index["routes"]
        assert isinstance(route_entries, list)
        verified_routes: dict[tuple[str, str], Mapping[str, Any]] = {}
        for position, entry_value in enumerate(route_entries):
            assert isinstance(entry_value, dict)
            entry: Mapping[str, Any] = entry_value
            key = (str(entry["case_id"]), str(entry["pattern"]))
            case, declaration = manifest_routes[key]
            maximum = (
                expanded_route.MAX_ROUTE_BYTES
                if declaration["format"] == expanded_route.ROUTE_FORMAT
                else compact_route.MAX_ARTIFACT_BYTES
            )
            artifact, raw = _load_json_artifact(
                root_descriptor,
                str(entry["path"]),
                maximum_bytes=maximum,
                name=f"route {position}",
            )
            _require(
                raw == canonical_bytes(artifact) + b"\n",
                "route file is not canonical JSON",
            )
            try:
                _validate_one_route(
                    artifact=artifact,
                    entry=entry,
                    case=case,
                    declaration=declaration,
                )
            except (
                expanded_route.CommonFairRouteError,
                compact_route.CommonFairCompactRouteError,
            ) as error:
                raise CommonFairBundleError(f"route {key} rejected: {error}") from error
            raw_sha = hashlib.sha256(raw).hexdigest()
            canonical_sha = str(artifact["canonical_sha256"])
            _require(entry["raw_sha256"] == raw_sha, f"route {key} raw SHA256 mismatch")
            _require(
                entry["canonical_sha256"] == canonical_sha,
                f"route {key} canonical SHA256 mismatch",
            )
            binding = declaration["binding"]
            assert isinstance(binding, dict)
            if binding["status"] == "PINNED":
                _require(
                    binding["path"] == entry["path"],
                    f"route {key} manifest path mismatch",
                )
                _require(
                    binding["raw_sha256"] == raw_sha,
                    f"route {key} manifest raw hash mismatch",
                )
                _require(
                    binding["canonical_sha256"] == canonical_sha,
                    f"route {key} manifest canonical hash mismatch",
                )
            else:
                _require(
                    manifest["manifest_phase"] == "DESIGN_ONLY",
                    f"route {key} is not frozen",
                )
            verified_routes[key] = entry

        matrix_rows = {
            row["row_id"] for row in manifest["matrix"] if isinstance(row, dict)
        }
        systems = {
            system["id"]: system
            for system in manifest["systems"]
            if isinstance(system, dict)
        }
        result_codes: Counter[str] = Counter()
        result_entries = index["results"]
        assert isinstance(result_entries, list)
        if result_entries:
            _require(
                manifest["manifest_phase"] == "STRUCTURALLY_FROZEN_UNVERIFIED",
                "result files require a structurally frozen manifest",
            )
        for position, entry_value in enumerate(result_entries):
            assert isinstance(entry_value, dict)
            result_entry: Mapping[str, Any] = entry_value
            document, raw = _load_json_artifact(
                root_descriptor,
                str(result_entry["path"]),
                maximum_bytes=results.MAX_RESULT_BYTES,
                name=f"raw result {position}",
            )
            _require(
                raw == results.canonical_bytes(document) + b"\n",
                "raw result is not canonical JSON",
            )
            try:
                results.validate_raw_result(document)
            except results.RawResultError as error:
                raise CommonFairBundleError(
                    f"raw result {position} rejected: {error}"
                ) from error
            _require(
                document["result_code"] != "PASSED_BENCHMARK",
                "single raw result cannot promote a benchmark",
            )
            _require(
                document["performance_claim_allowed"] is False,
                "raw result claims performance",
            )
            _require(
                document["evidence_verification_status"] == "NOT_RUN",
                "raw result claims external verification",
            )
            _require(
                result_entry["row_id"] == document["row_id"],
                "raw-result row ID mismatch",
            )
            _require(
                result_entry["result_code"] == document["result_code"],
                "raw-result code mismatch",
            )
            _require(
                result_entry["evidence_sha256"] == document["evidence_sha256"],
                "raw-result evidence hash mismatch",
            )
            _require(
                result_entry["raw_sha256"] == hashlib.sha256(raw).hexdigest(),
                "raw-result file hash mismatch",
            )
            _require(
                result_entry["canonical_sha256"]
                == hashlib.sha256(results.canonical_bytes(document)).hexdigest(),
                "raw-result canonical hash mismatch",
            )
            _require(
                document["row_id"] in matrix_rows,
                "raw-result row is outside the manifest matrix",
            )
            _require(
                document["manifest_raw_sha256"] == manifest_raw_sha,
                "raw result binds another manifest raw hash",
            )
            _require(
                document["manifest_canonical_sha256"] == manifest_canonical_sha,
                "raw result binds another manifest canonical hash",
            )
            route_key = (str(document["case_id"]), str(document["route_pattern"]))
            route_entry = verified_routes[route_key]
            _require(
                document["route_raw_sha256"] == route_entry["raw_sha256"],
                "raw result binds another route raw hash",
            )
            _require(
                document["route_canonical_sha256"] == route_entry["canonical_sha256"],
                "raw result binds another route canonical hash",
            )
            _require(
                document["payload_generator_id"] == payload.GENERATOR_ID,
                "raw result payload ID mismatch",
            )
            _require(
                document["payload_generator_seed"] == payload.SEED,
                "raw result payload seed mismatch",
            )
            _require(
                document["payload_generator_code_sha256"] == source_sha,
                "raw result payload source hash mismatch",
            )
            system = systems[str(document["system_id"])]
            for result_field, manifest_field in (
                ("adapter_sha256", "adapter"),
                ("runtime_closure_sha256", "runtime_closure"),
                ("execution_config_sha256", "execution_config"),
            ):
                binding = system[manifest_field]
                assert isinstance(binding, dict)
                _require(
                    binding["status"] == "PINNED",
                    f"raw result {manifest_field} is not frozen",
                )
                _require(
                    document[result_field] == binding["sha256"],
                    f"raw result {manifest_field} hash mismatch",
                )
            _require(
                document["candidate_id"] == system["candidate_id"],
                "raw result candidate ID mismatch",
            )
            schedule = manifest["measurement"]["order_schedule"]
            assert isinstance(schedule, dict)
            _require(
                document["order_schedule_sha256"] == schedule["sha256"],
                "raw result order schedule mismatch",
            )
            environment = manifest["preregistration"]["environment_binding"]
            assert isinstance(environment, dict)
            _require(
                document["environment_sha256"] == environment["sha256"],
                "raw result environment hash mismatch",
            )
            result_codes[str(document["result_code"])] += 1

        root_after = os.fstat(root_descriptor)
        _require(
            _directory_identity(root_after) == root_identity,
            "bundle root changed during validation",
        )
    finally:
        os.close(root_descriptor)

    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "bundle_sha256": index["bundle_sha256"],
        "manifest_phase": manifest["manifest_phase"],
        "case_count": len(schema.CASE_SPECS),
        "route_count": len(schema.CASE_SPECS) * len(schema.ROUTE_PATTERNS),
        "derived_route_count": 12,
        "confirmatory_route_count": 12,
        "payload_generator_id": payload.GENERATOR_ID,
        "payload_generator_seed": payload.SEED,
        "payload_source_raw_sha256": source_sha,
        "result_file_count": sum(result_codes.values()),
        "result_codes": dict(sorted(result_codes.items())),
        "status": "CPU_ARTIFACTS_VALID_UNVERIFIED",
        "build_execution_status": "NOT_RUN",
        "gpu_execution_status": "NOT_RUN",
        "performance_verification_status": "NOT_RUN",
        "execution_authorized": False,
        "performance_claim_allowed": False,
        "report_sha256": "",
    }
    report["report_sha256"] = report_sha256(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="COMMON_FAIR bundle directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        report = validate_bundle(arguments.bundle)
    except CommonFairBundleError as error:
        parser.exit(2, f"COMMON_FAIR bundle rejected: {error}\n")
    print(canonical_bytes(report).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
