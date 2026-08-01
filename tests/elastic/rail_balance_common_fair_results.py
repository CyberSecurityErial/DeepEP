"""Strict CPU-only validation of COMMON_FAIR raw timing evidence.

The wire format is one canonical UTF-8 JSON document followed by one newline.
It records one system at one slot of one preregistered ABBA/BAAB block.  Every
raw sample binds the same case, route, pair, block, slot, process instance,
and source artifacts as the root document.

This module deliberately cannot create or accept ``PASSED_BENCHMARK``.  A
complete timing capture remains ``RAW_SAMPLES_COMPLETE_UNVERIFIED`` with
``performance_claim_allowed=false`` and external verification ``NOT_RUN``.
Aggregation excludes warmup, takes the maximum across ranks for each steady
iteration first, and only then computes statistics.  Its output remains
unverified and is not publication evidence.

Only the Python standard library is imported.  No GPU, build, or network
operation exists in this module.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT_ID = "rail-balance-common-fair-raw-result-v1"
AGGREGATE_FORMAT_ID = "rail-balance-common-fair-aggregate-unverified-v1"
MAX_RESULT_BYTES = 64 << 20
MAX_DURATION_NS = (1 << 63) - 1

WARMUP_ITERATIONS = 10
STEADY_ITERATIONS = 100
INDEPENDENT_REPEATS = 5
TOTAL_ITERATIONS = WARMUP_ITERATIONS + STEADY_ITERATIONS

# Copied exactly from rail_balance_common_fair_schema.py.  Importing that
# module would currently remain standard-library-only, but copying freezes the
# on-wire validator and makes that property locally auditable.
SYSTEM_IDS = (
    "deepep-clean",
    "deepep-off",
    "nccl-ep",
    "railbalance-best",
    "uccl-ep",
)
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

ROOT_FIELDS = frozenset(
    {
        "format",
        "system_id",
        "case_id",
        "route_pattern",
        "row_id",
        "world_size",
        "result_code",
        "performance_claim_allowed",
        "evidence_verification_status",
        "reasons",
        "terminal_stage",
        "manifest_raw_sha256",
        "manifest_canonical_sha256",
        "route_raw_sha256",
        "route_canonical_sha256",
        "payload_generator_id",
        "payload_generator_code_sha256",
        "payload_generator_seed",
        "adapter_sha256",
        "runtime_closure_sha256",
        "execution_config_sha256",
        "candidate_id",
        "environment_sha256",
        "correctness_evidence_status",
        "correctness_evidence_sha256",
        "order_schedule_sha256",
        "order_binding",
        "measurement",
        "clock",
        "phase_contract",
        "samples",
        "evidence_sha256",
    }
)
ORDER_BINDING_FIELDS = frozenset(
    {
        "pair_id",
        "block_id",
        "block_ordinal",
        "cycle",
        "order",
        "order_position",
        "process_instance_ids",
    }
)
CLOCK_CONTRACT = {
    "timing_api": "CUDA_EVENT",
    "unit": "ns",
    "world_barrier_location": "OUTSIDE_TIMED_WINDOW",
    "end_event_synchronize_location": "OUTSIDE_TIMED_WINDOW",
}
MEASUREMENT_CONTRACT = {
    "independent_process_repeats": INDEPENDENT_REPEATS,
    "warmup_iterations": WARMUP_ITERATIONS,
    "steady_iterations": STEADY_ITERATIONS,
    "rank_aggregation": "PER_ITERATION_MAX",
}
TERMINAL_STAGE_BY_CODE = {
    "BENCHMARK_FAILED": "BENCHMARK",
    "BENCHMARK_INVALID": "BENCHMARK",
    "BLOCKED_DEPENDENCY": "PREFLIGHT",
    "BLOCKED_DISK": "PREFLIGHT",
    "BLOCKED_GPU": "PREFLIGHT",
    "BLOCKED_NETWORK": "PREFLIGHT",
    "BLOCKED_TOPOLOGY": "PREFLIGHT",
    "BUILD_FAILED": "BUILD",
    "CORRECTNESS_FAILED": "CORRECTNESS",
    "EVIDENCE_MISSING": "EVIDENCE",
    "LAUNCH_FAILED": "LAUNCH",
    "NONDETERMINISTIC": "CORRECTNESS",
    "NOT_RUN": "NOT_RUN",
    "NO_COMMON_API": "ADAPTER",
    "PASSED_CORRECTNESS": "CORRECTNESS",
    "RAW_SAMPLES_COMPLETE_UNVERIFIED": "RAW_SAMPLE_CAPTURE",
    "UNSUPPORTED_HARDWARE": "PREFLIGHT",
}

_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RawResultError(ValueError):
    """Raised when COMMON_FAIR evidence is incomplete, unsafe, or ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RawResultError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RawResultError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise RawResultError(f"non-finite JSON constant is forbidden: {value}")


def _exact_object(value: object, fields: frozenset[str], name: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{name} must be an object")
    assert isinstance(value, dict)
    observed = set(value)
    _require(
        observed == fields,
        f"{name} fields differ: missing={sorted(fields - observed)}, "
        f"extra={sorted(observed - fields)}",
    )
    return value


def _string(value: object, name: str, *, maximum: int = 2048) -> str:
    _require(type(value) is str and 0 < len(value) <= maximum, f"{name} is invalid")
    assert isinstance(value, str)
    _require(
        all(ord(character) >= 0x20 and ord(character) != 0x7F for character in value),
        f"{name} contains a control character",
    )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise RawResultError(f"{name} is not valid UTF-8 text") from error
    return value


def _slug(value: object, name: str) -> str:
    result = _string(value, name, maximum=128)
    _require(bool(_SLUG.fullmatch(result)), f"{name} is not a safe slug")
    return result


def _sha256(value: object, name: str) -> str:
    result = _string(value, name, maximum=64)
    _require(bool(_SHA256.fullmatch(result)), f"{name} is not lowercase SHA256")
    return result


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    _require(
        type(value) is int and minimum <= value <= maximum,
        f"{name} must be an integer in [{minimum}, {maximum}]",
    )
    assert isinstance(value, int)
    return value


def _duration(value: object, name: str) -> int:
    return _integer(value, name, minimum=0, maximum=MAX_DURATION_NS)


def canonical_bytes(value: object) -> bytes:
    """Serialize finite JSON with the frozen canonical encoding."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as error:
        raise RawResultError(f"value is not finite canonical JSON: {error}") from error


def evidence_sha256(value: object) -> str:
    """Hash every result field except the self-hash field itself."""

    _require(type(value) is dict, "raw result must be an object")
    assert isinstance(value, dict)
    identity = {key: item for key, item in value.items() if key != "evidence_sha256"}
    return hashlib.sha256(canonical_bytes(identity)).hexdigest()


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
    _require(metadata.st_uid in {0, os.geteuid()}, f"{name} has an untrusted owner")
    if mode & 0o022:
        _require(
            metadata.st_uid == 0 and bool(metadata.st_mode & stat.S_ISVTX),
            f"{name} is an unsafe writable ancestor",
        )


def _open_parent(path: Path) -> tuple[int, str, tuple[int, ...]]:
    raw = os.fspath(path)
    _require(type(raw) is str and bool(raw), "raw result path must be nonempty")
    parts = Path(raw).parts
    _require(bool(parts), "raw result path must name a file")
    name = parts[-1]
    _require(name not in {"", ".", ".."}, "raw result path must name a safe file")
    directory_parts = parts[1:-1] if Path(raw).is_absolute() else parts[:-1]
    _require(
        all(part not in {"", ".", ".."} for part in directory_parts),
        "raw result path contains a non-canonical directory component",
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/" if Path(raw).is_absolute() else ".", flags)
    try:
        try:
            for index, component in enumerate(directory_parts):
                before = os.fstat(descriptor)
                _validate_directory(
                    before, f"raw result parent component {index}", final=False
                )
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
                try:
                    after = os.fstat(descriptor)
                    _require(
                        _directory_identity(before) == _directory_identity(after),
                        f"raw result parent component {index} changed during traversal",
                    )
                except BaseException:
                    os.close(next_descriptor)
                    raise
                os.close(descriptor)
                descriptor = next_descriptor
        except OSError as error:
            raise RawResultError(
                f"cannot safely traverse raw result parent for {path}: {error}"
            ) from error
        metadata = os.fstat(descriptor)
        _validate_directory(metadata, "raw result parent", final=True)
        return descriptor, name, _directory_identity(metadata)
    except BaseException:
        os.close(descriptor)
        raise


def read_raw_result_bytes(path: Path) -> bytes:
    """Safely read one bounded owner-controlled regular result file."""

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
            _validate_directory(parent_after, "raw result parent", final=True)
            _require(
                _directory_identity(parent_after) == parent_identity,
                "raw result parent changed while the file was being opened",
            )
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
                descriptor = None
            raise RawResultError(
                f"cannot safely open raw result {path}: {error}"
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
        _require(stat.S_ISREG(before.st_mode), "raw result must be a regular file")
        _require(before.st_uid == os.geteuid(), "raw result must be owned by this user")
        _require(before.st_nlink == 1, "raw result must have exactly one hard link")
        _require(
            stat.S_IMODE(before.st_mode) & 0o022 == 0,
            "raw result must not be group/world writable",
        )
        _require(
            0 < before.st_size <= MAX_RESULT_BYTES,
            f"raw result must contain 1..{MAX_RESULT_BYTES} bytes",
        )
        chunks: list[bytes] = []
        remaining = MAX_RESULT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        _require(len(raw) <= MAX_RESULT_BYTES, "raw result exceeds the byte limit")
        _require(len(raw) == after.st_size, "raw result read was incomplete")
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
            "raw result changed while it was being read",
        )
        return raw
    finally:
        os.close(descriptor)


def load_raw_result(path: Path) -> tuple[dict[str, Any], str, str]:
    """Load canonical strict JSON and return document/raw/canonical SHA256."""

    raw = read_raw_result_bytes(path)
    _require(not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM is forbidden")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except RawResultError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as error:
        raise RawResultError(f"invalid raw result JSON: {error}") from error
    validate_raw_result(value)
    assert isinstance(value, dict)
    _require(
        raw == canonical_bytes(value) + b"\n",
        "raw result file does not use the canonical byte encoding",
    )
    return (
        value,
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(canonical_bytes(value)).hexdigest(),
    )


def _validate_reasons(value: object) -> list[str]:
    _require(type(value) is list, "reasons must be a list")
    assert isinstance(value, list)
    reasons = [_string(item, f"reasons[{index}]") for index, item in enumerate(value)]
    _require(reasons == sorted(set(reasons)), "reasons must be sorted and unique")
    return reasons


def _validate_clock(value: object) -> None:
    row = _exact_object(value, frozenset(CLOCK_CONTRACT), "clock")
    _require(row == CLOCK_CONTRACT, "clock differs from the frozen CUDA-event contract")


def _validate_measurement(value: object) -> None:
    row = _exact_object(value, frozenset(MEASUREMENT_CONTRACT), "measurement")
    for field, expected in (
        ("independent_process_repeats", INDEPENDENT_REPEATS),
        ("warmup_iterations", WARMUP_ITERATIONS),
        ("steady_iterations", STEADY_ITERATIONS),
    ):
        _integer(row[field], f"measurement.{field}", minimum=expected, maximum=expected)
    _require(
        row["rank_aggregation"] == "PER_ITERATION_MAX",
        "measurement.rank_aggregation must be PER_ITERATION_MAX",
    )


def _validate_phase_contract(value: object) -> dict[str, str]:
    row = _exact_object(value, frozenset(PHASE_IDS), "phase_contract")
    route_contract = _string(
        row["route_prepare_complete"],
        "phase_contract.route_prepare_complete",
        maximum=64,
    )
    _require(
        route_contract in {"MEASURED", "NOT_SEPARABLE"},
        "route_prepare_complete contract must be MEASURED or NOT_SEPARABLE",
    )
    for phase in PHASE_IDS[1:]:
        contract = _string(row[phase], f"phase_contract.{phase}", maximum=64)
        _require(contract == "MEASURED", f"{phase} contract must be MEASURED")
    return row


def _expected_order(pair_id: str, block_ordinal: int) -> tuple[str, list[str]]:
    system_a, system_b = PAIR_SPECS[pair_id]
    if block_ordinal % 2 == 0:
        return "ABBA", [system_a, system_b, system_b, system_a]
    return "BAAB", [system_b, system_a, system_a, system_b]


def _validate_order_binding(value: object, system_id: str) -> dict[str, Any]:
    row = _exact_object(value, ORDER_BINDING_FIELDS, "order_binding")
    pair_id = _slug(row["pair_id"], "order_binding.pair_id")
    _require(
        pair_id in PAIR_SPECS, "order_binding.pair_id is not a frozen comparison pair"
    )
    ordinal = _integer(
        row["block_ordinal"],
        "order_binding.block_ordinal",
        minimum=0,
        maximum=9,
    )
    expected_block_id = f"{pair_id}-block-{ordinal:02d}"
    _require(
        row["block_id"] == expected_block_id,
        "order_binding.block_id differs from its 00..09 ordinal",
    )
    cycle, order = _expected_order(pair_id, ordinal)
    _require(row["cycle"] == cycle, "order_binding.cycle does not alternate ABBA/BAAB")
    _require(
        row["order"] == order, "order_binding.order differs from its ABBA/BAAB cycle"
    )
    position = _integer(
        row["order_position"],
        "order_binding.order_position",
        minimum=0,
        maximum=3,
    )
    _require(
        order[position] == system_id, "order_binding slot does not contain system_id"
    )
    instances = row["process_instance_ids"]
    _require(
        type(instances) is list, "order_binding.process_instance_ids must be a list"
    )
    assert isinstance(instances, list)
    process_ids = [
        _slug(item, f"order_binding.process_instance_ids[{index}]")
        for index, item in enumerate(instances)
    ]
    _require(
        len(process_ids) == INDEPENDENT_REPEATS,
        "order binding must name five process instances",
    )
    _require(
        len(set(process_ids)) == len(process_ids),
        "process instance IDs must be distinct across repeats",
    )
    return row


def _validate_phase_ns(
    value: object,
    phase_contract: Mapping[str, str],
    name: str,
) -> None:
    row = _exact_object(value, frozenset(PHASE_IDS), name)
    for phase in PHASE_IDS:
        duration = row[phase]
        if phase_contract[phase] == "NOT_SEPARABLE":
            _require(
                duration is None, f"{name}.{phase} must be null when NOT_SEPARABLE"
            )
        else:
            _duration(duration, f"{name}.{phase}")


def _validate_samples(
    value: object,
    *,
    root: Mapping[str, Any],
    world_size: int,
    phase_contract: Mapping[str, str],
    order_binding: Mapping[str, Any],
) -> None:
    _require(type(value) is list, "samples must be a list")
    assert isinstance(value, list)
    instances = order_binding["process_instance_ids"]
    assert isinstance(instances, list)
    seen: set[tuple[int, int, int]] = set()
    for index, item in enumerate(value):
        name = f"samples[{index}]"
        row = _exact_object(item, frozenset(RAW_SAMPLE_FIELDS), name)
        for field in ("system_id", "case_id", "route_pattern"):
            _require(
                row[field] == root[field], f"{name}.{field} breaks the root binding"
            )
        _require(
            row["route_raw_sha256"] == root["route_raw_sha256"],
            f"{name}.route_raw_sha256 breaks the root binding",
        )
        for field in ("block_id", "pair_id", "order_position"):
            _require(
                row[field] == order_binding[field],
                f"{name}.{field} breaks the order binding",
            )
        rank = _integer(row["rank"], f"{name}.rank", minimum=0, maximum=world_size - 1)
        repeat = _integer(
            row["repeat"],
            f"{name}.repeat",
            minimum=0,
            maximum=INDEPENDENT_REPEATS - 1,
        )
        _require(
            row["process_instance_id"] == instances[repeat],
            f"{name}.process_instance_id breaks its repeat binding",
        )
        iteration = _integer(
            row["iteration"],
            f"{name}.iteration",
            minimum=0,
            maximum=TOTAL_ITERATIONS - 1,
        )
        expected_class = "warmup" if iteration < WARMUP_ITERATIONS else "steady"
        _require(
            row["sample_class"] == expected_class,
            f"{name}.sample_class disagrees with its global iteration",
        )
        _validate_phase_ns(row["phase_ns"], phase_contract, f"{name}.phase_ns")
        key = (rank, repeat, iteration)
        _require(key not in seen, f"duplicate rank/repeat/iteration sample: {key}")
        seen.add(key)

    expected = {
        (rank, repeat, iteration)
        for rank in range(world_size)
        for repeat in range(INDEPENDENT_REPEATS)
        for iteration in range(TOTAL_ITERATIONS)
    }
    missing = expected - seen
    extra = seen - expected
    _require(
        not missing and not extra,
        "rank/repeat/iteration coverage is incomplete: "
        f"expected={len(expected)}, observed={len(seen)}, "
        f"first_missing={min(missing) if missing else None}, "
        f"first_extra={min(extra) if extra else None}",
    )


def _validate_correctness_binding(root: Mapping[str, Any], result_code: str) -> None:
    status = _string(
        root["correctness_evidence_status"],
        "correctness_evidence_status",
        maximum=64,
    )
    _require(
        status in {"PINNED_PASSED", "SELF", "NOT_AVAILABLE"},
        "correctness_evidence_status is invalid",
    )
    digest = root["correctness_evidence_sha256"]
    if status == "PINNED_PASSED":
        _sha256(digest, "correctness_evidence_sha256")
    else:
        _require(
            digest is None, "unpinned correctness evidence must have a null SHA256"
        )
    if result_code == "RAW_SAMPLES_COMPLETE_UNVERIFIED":
        _require(
            status == "PINNED_PASSED",
            "raw samples require pinned passed correctness evidence",
        )
    if result_code == "PASSED_CORRECTNESS":
        _require(
            status == "SELF",
            "PASSED_CORRECTNESS must identify itself as the correctness evidence",
        )


def validate_raw_result(value: object) -> None:
    """Validate one parsed COMMON_FAIR result; verify its self-hash last."""

    root = _exact_object(value, ROOT_FIELDS, "raw result")
    _require(root["format"] == FORMAT_ID, "raw result format is unsupported")
    system_id = _slug(root["system_id"], "system_id")
    _require(system_id in SYSTEM_IDS, "system_id is outside the frozen system set")
    case_id = _slug(root["case_id"], "case_id")
    _require(case_id in CASE_SPECS, "case_id is outside the exact frozen case set")
    scope, _mode, _nodes, _ranks_per_node, expected_world = CASE_SPECS[case_id]
    world_size = _integer(root["world_size"], "world_size", minimum=1, maximum=32)
    _require(
        world_size == expected_world,
        "case_id/world_size binding differs from CASE_SPECS",
    )
    route_pattern = _slug(root["route_pattern"], "route_pattern")
    _require(
        route_pattern in ROUTE_PATTERNS, "route_pattern is outside the frozen route set"
    )
    expected_row = f"{case_id}--{route_pattern}--{system_id}"
    _require(
        root["row_id"] == expected_row, "row_id breaks the case/route/system binding"
    )

    result_code = _string(root["result_code"], "result_code", maximum=64)
    _require(result_code in RESULT_CODES, "result_code is unknown")
    _require(
        result_code != "PASSED_BENCHMARK",
        "this unverified module forbids PASSED_BENCHMARK",
    )
    _require(
        root["performance_claim_allowed"] is False,
        "performance_claim_allowed must remain false",
    )
    _require(
        root["evidence_verification_status"] == "NOT_RUN",
        "evidence_verification_status must remain NOT_RUN",
    )
    reasons = _validate_reasons(root["reasons"])
    expected_stage = TERMINAL_STAGE_BY_CODE[result_code]
    _require(
        root["terminal_stage"] == expected_stage,
        "terminal_stage differs from result_code",
    )

    for field in (
        "manifest_raw_sha256",
        "manifest_canonical_sha256",
        "route_raw_sha256",
        "route_canonical_sha256",
        "payload_generator_code_sha256",
        "adapter_sha256",
        "runtime_closure_sha256",
        "execution_config_sha256",
        "environment_sha256",
        "order_schedule_sha256",
    ):
        _sha256(root[field], field)
    _require(
        root["payload_generator_id"] == "rail-balance-deterministic-bf16-v1",
        "payload_generator_id differs from the manifest contract",
    )
    _integer(
        root["payload_generator_seed"],
        "payload_generator_seed",
        minimum=0,
        maximum=(1 << 63) - 1,
    )
    candidate_id = _slug(root["candidate_id"], "candidate_id")
    fixed_candidate = FIXED_CANDIDATE_IDS.get(system_id)
    if fixed_candidate is not None:
        _require(
            candidate_id == fixed_candidate,
            "candidate_id differs from the system's frozen candidate",
        )
    _validate_correctness_binding(root, result_code)
    _validate_measurement(root["measurement"])
    _validate_clock(root["clock"])
    phase_contract = _validate_phase_contract(root["phase_contract"])
    order_binding = _validate_order_binding(root["order_binding"], system_id)

    if result_code == "RAW_SAMPLES_COMPLETE_UNVERIFIED":
        _require(
            scope == "CONFIRMATORY_COMMON_FAIR",
            "derived cases cannot carry benchmark timing samples",
        )
        _require(
            expected_world in {16, 32},
            "only confirmatory w16/w32 may carry timing samples",
        )
        _require(
            not reasons, "complete raw samples must not carry terminal failure reasons"
        )
        _validate_samples(
            root["samples"],
            root=root,
            world_size=world_size,
            phase_contract=phase_contract,
            order_binding=order_binding,
        )
    else:
        _require(root["samples"] == [], "non-raw result must not carry timing samples")
        if result_code == "PASSED_CORRECTNESS":
            _require(not reasons, "PASSED_CORRECTNESS must not carry failure reasons")
        else:
            _require(
                bool(reasons), "terminal result must explain why timing is unavailable"
            )

    observed_evidence_sha = _sha256(root["evidence_sha256"], "evidence_sha256")
    _require(
        observed_evidence_sha == evidence_sha256(root),
        "evidence_sha256 mismatch",
    )


def _percentile(values: Sequence[int | float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot summarize empty samples")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _statistics(values: Sequence[int | float]) -> dict[str, int | float]:
    _require(bool(values), "cannot summarize empty samples")
    numeric = [float(value) for value in values]
    _require(
        all(math.isfinite(value) and value >= 0 for value in numeric),
        "statistics require finite nonnegative samples",
    )
    center = float(statistics.median(numeric))
    scaled_mad = 1.4826 * float(
        statistics.median(abs(value - center) for value in numeric)
    )
    mean = statistics.fmean(numeric)
    stddev = statistics.pstdev(numeric)
    return {
        "count": len(numeric),
        "median_ns": center,
        "p95_ns": _percentile(numeric, 95.0),
        "p99_ns": _percentile(numeric, 99.0),
        "mean_ns": mean,
        "stddev_population_ns": stddev,
        "cv_population": 0.0 if mean == 0.0 else stddev / mean,
        "min_ns": min(numeric),
        "max_ns": max(numeric),
        "scaled_mad_ns": scaled_mad,
        "relative_scaled_mad": 0.0 if center == 0.0 else scaled_mad / center,
    }


def aggregate_raw_result(value: object) -> dict[str, Any]:
    """Aggregate complete unverified samples without promoting their status."""

    validate_raw_result(value)
    assert isinstance(value, dict)
    _require(
        value["result_code"] == "RAW_SAMPLES_COMPLETE_UNVERIFIED",
        "only RAW_SAMPLES_COMPLETE_UNVERIFIED can be aggregated",
    )
    samples = value["samples"]
    assert isinstance(samples, list)
    by_key: dict[tuple[int, int], dict[int, Mapping[str, int | None]]] = {}
    for sample in samples:
        assert isinstance(sample, dict)
        iteration = int(sample["iteration"])
        if iteration < WARMUP_ITERATIONS:
            continue
        repeat = int(sample["repeat"])
        rank = int(sample["rank"])
        phases = sample["phase_ns"]
        assert isinstance(phases, dict)
        by_key.setdefault((repeat, iteration), {})[rank] = phases

    phase_contract = value["phase_contract"]
    assert isinstance(phase_contract, dict)
    phase_results: dict[str, Any] = {}
    for phase in PHASE_IDS:
        if phase_contract[phase] == "NOT_SEPARABLE":
            phase_results[phase] = {
                "contract": "NOT_SEPARABLE",
                "per_repeat": None,
                "repeat_medians_ns": None,
                "primary_statistics": None,
                "pooled_diagnostic_statistics": None,
            }
            continue
        per_repeat: list[dict[str, Any]] = []
        repeat_medians: list[float] = []
        pooled: list[int] = []
        for repeat in range(INDEPENDENT_REPEATS):
            rank_max_ns: list[int] = []
            for iteration in range(WARMUP_ITERATIONS, TOTAL_ITERATIONS):
                rank_phases = by_key[(repeat, iteration)]
                durations = [rank_phases[rank][phase] for rank in sorted(rank_phases)]
                assert all(type(duration) is int for duration in durations)
                maximum = max(
                    int(duration) for duration in durations if duration is not None
                )
                rank_max_ns.append(maximum)
            statistics_row = _statistics(rank_max_ns)
            repeat_medians.append(float(statistics_row["median_ns"]))
            pooled.extend(rank_max_ns)
            per_repeat.append(
                {
                    "repeat": repeat,
                    "process_instance_id": value["order_binding"][
                        "process_instance_ids"
                    ][repeat],
                    "steady_iteration_start": WARMUP_ITERATIONS,
                    "rank_max_ns": rank_max_ns,
                    "statistics": statistics_row,
                }
            )
        phase_results[phase] = {
            "contract": "MEASURED",
            "per_repeat": per_repeat,
            "repeat_medians_ns": repeat_medians,
            "primary_statistics": _statistics(repeat_medians),
            "pooled_diagnostic_statistics": _statistics(pooled),
        }

    return {
        "format": AGGREGATE_FORMAT_ID,
        "source_evidence_sha256": value["evidence_sha256"],
        "source_result_canonical_sha256": hashlib.sha256(
            canonical_bytes(value)
        ).hexdigest(),
        "system_id": value["system_id"],
        "candidate_id": value["candidate_id"],
        "execution_config_sha256": value["execution_config_sha256"],
        "case_id": value["case_id"],
        "route_pattern": value["route_pattern"],
        "route_raw_sha256": value["route_raw_sha256"],
        "route_canonical_sha256": value["route_canonical_sha256"],
        "row_id": value["row_id"],
        "world_size": value["world_size"],
        "order_binding": copy.deepcopy(value["order_binding"]),
        "result_code": "RAW_SAMPLES_COMPLETE_UNVERIFIED",
        "performance_claim_allowed": False,
        "evidence_verification_status": "NOT_RUN",
        "publication_claim_allowed": False,
        "external_verification_required": True,
        "warmup_samples_excluded": True,
        "reduction_order": "PER_STEADY_ITERATION_RANK_MAX_THEN_STATISTICS",
        "primary_estimator": "MEDIAN_OF_FIVE_REPEAT_MEDIANS",
        "phases": phase_results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result", type=Path, help="canonical COMMON_FAIR raw-result JSON"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="validate without aggregation"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        value, raw_sha256, canonical_sha256 = load_raw_result(args.result)
        if args.validate_only:
            output: dict[str, Any] = {
                "valid": True,
                "raw_sha256": raw_sha256,
                "canonical_sha256": canonical_sha256,
                "result_code": value["result_code"],
                "performance_claim_allowed": False,
                "evidence_verification_status": "NOT_RUN",
            }
        else:
            output = aggregate_raw_result(value)
            output["source_result_raw_sha256"] = raw_sha256
    except RawResultError as error:
        parser.exit(2, f"COMMON_FAIR raw result rejected: {error}\n")
    print(json.dumps(output, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
