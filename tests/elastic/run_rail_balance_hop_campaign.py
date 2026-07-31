#!/usr/bin/env python3
"""Fail-closed orchestrator for hop-aware CUDA optimization campaigns.

The parent process is stdlib-only.  It validates and hashes the frozen
manifest, acquires a campaign lock, checks the 300 GB home limit, and samples
all host GPUs before it may launch a command that imports PyTorch or DeepEP.
When exclusive eight-GPU execution is unavailable, only CPU stages run and the
result is explicitly labelled ``scaffold_only``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from rail_balance_campaign_schema import (
    HOME_HARD_LIMIT_BYTES,
    ManifestError,
    expand_template,
    load_manifest,
    read_manifest_bytes,
)


_RUNNER_PATH = Path(__file__).resolve()
_ROOT = _RUNNER_PATH.parents[2]
_SCHEMA_PATH = _RUNNER_PATH.with_name("rail_balance_campaign_schema.py")
_EXECUTOR_PATH = _RUNNER_PATH.with_name("rail_balance_source_round_executor.py")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_ROUND_LEASE_PROTOCOL = "deepep-source-round-flock-lease-v1"
_SOURCE_ROUND_LEASE_ENV = {
    "DEEP_EP_SOURCE_ROUND_LEASE_FD",
    "DEEP_EP_SOURCE_ROUND_LEASE_NONCE",
    "DEEP_EP_SOURCE_ROUND_LEASE_PARENT_PID",
    "DEEP_EP_SOURCE_ROUND_LEASE_PLAN_SHA256",
    "DEEP_EP_SOURCE_ROUND_LEASE_ROUND_ID",
    "DEEP_EP_SOURCE_ROUND_LEASE_IDENTITY_SHA256",
}
_SOURCE_ROUND_LEASE_FIELDS = {
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
_GPU_IDLE_PROBES = 3
_GPU_IDLE_PROBE_INTERVAL_SECONDS = 0.5
_HOME_EMERGENCY_RESERVE_BYTES = 5_000_000_000
_FINALIZATION_RESERVE_BYTES = 10_000_000
_RESOURCE_POLL_SECONDS = 2.0
_HOME_POLL_SECONDS = 10.0
_CHILD_PATH = (
    "/home/chen/.cache/deepep-sjlgpt/bin:/usr/local/cuda/bin:"
    "/opt/nvidia/nsight-compute/2025.1.1:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin"
)
_CHILD_BASE_ENVIRONMENT = {
    "CUDA_HOME": "/usr/local/cuda",
    "HOME": "/home/chen",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
    "LOGNAME": "chen",
    "PATH": _CHILD_PATH,
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "SHELL": "/bin/bash",
    "TMPDIR": "/tmp",
    "TZ": "UTC",
    "USER": "chen",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"duplicate source-round lease key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise RuntimeError(f"non-finite source-round lease constant: {value}")


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot read boot ID for source-round lease: {error}") from error
    if not value:
        raise RuntimeError("empty boot ID for source-round lease")
    return value


def _leased_artifact_root_identity(record: dict[str, object]) -> os.stat_result:
    root = Path(str(record["artifact_root"]))
    if not root.is_absolute() or root != Path(os.path.abspath(root)):
        raise RuntimeError("source-round artifact root is not canonical")
    cursor = Path(root.anchor)
    try:
        for component in root.parts[1:]:
            cursor /= component
            metadata = os.lstat(cursor)
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(
                    f"source-round artifact root has a symlink ancestor: {cursor}"
                )
    except OSError as error:
        raise RuntimeError(f"cannot inspect source-round artifact root: {error}") from error
    metadata = os.stat(root, follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_dev != record.get("artifact_root_device")
        or metadata.st_ino != record.get("artifact_root_inode")
    ):
        raise RuntimeError(
            "source-round artifact root identity/owner/mode differs from lease"
        )
    return metadata


def _source_round_lock_identity(record: dict[str, object]) -> str:
    return _canonical_sha256(
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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
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


def _atomic_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, payload)


def _capture(argv: Sequence[str], timeout: int = 15) -> dict[str, object]:
    try:
        result = subprocess.run(
            list(argv),
            cwd=_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "argv": list(argv),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "argv": list(argv),
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": repr(error),
        }


def _home_usage(path: Path) -> dict[str, object]:
    result = _capture(
        ("/usr/bin/du", "-sx", "--block-size=1", "--", str(path)),
        timeout=60,
    )
    usage = None
    if result["returncode"] == 0:
        with suppress(IndexError, ValueError):
            usage = int(str(result["stdout"]).split()[0])
    return {"path": str(path), "bytes": usage, "command": result}


def _toolchain_snapshot(target_platform: dict[str, object]) -> dict[str, object]:
    specs = (
        (
            "python",
            Path("/home/chen/.cache/deepep-sjlgpt/bin/python"),
            ("--version",),
            (f"Python {target_platform['python']}",),
        ),
        (
            "nvcc",
            Path("/usr/local/cuda/bin/nvcc"),
            ("--version",),
            (
                f"release {str(target_platform['cuda_toolkit']).split()[0]}",
                str(target_platform["cuda_toolkit"]).split()[-1],
            ),
        ),
        (
            "nsys",
            Path("/usr/local/cuda/bin/nsys"),
            ("--version",),
            (str(target_platform["nsys"]),),
        ),
        (
            "ncu",
            Path("/usr/local/cuda/bin/ncu"),
            ("--version",),
            tuple(str(target_platform["ncu"]).split()),
        ),
        (
            "compute_sanitizer",
            Path("/usr/local/cuda/bin/compute-sanitizer"),
            ("--version",),
            tuple(str(target_platform["compute_sanitizer"]).split()),
        ),
        (
            "git",
            Path("/usr/bin/git"),
            ("--version",),
            ("git version",),
        ),
        (
            "du",
            Path("/usr/bin/du"),
            ("--version",),
            ("GNU coreutils",),
        ),
    )
    rows = []
    reasons = []
    for name, path, arguments, required_fragments in specs:
        resolved = path.resolve()
        command = _capture((str(path), *arguments))
        output = f"{command.get('stdout', '')}\n{command.get('stderr', '')}"
        row = {
            "name": name,
            "path": str(path),
            "realpath": str(resolved),
            "sha256": _sha256(resolved) if resolved.is_file() else None,
            "command": command,
            "required_fragments": list(required_fragments),
        }
        rows.append(row)
        if command.get("returncode") != 0 or not resolved.is_file():
            reasons.append(f"tool_unavailable:{name}")
        for fragment in required_fragments:
            if fragment not in output:
                reasons.append(f"tool_version_mismatch:{name}:{fragment}")
    driver = _capture(
        (
            "/usr/bin/nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        )
    )
    driver_rows = [
        row.strip() for row in str(driver.get("stdout", "")).splitlines() if row
    ]
    if (
        driver.get("returncode") != 0
        or len(driver_rows) != 8
        or set(driver_rows) != {str(target_platform["driver"])}
    ):
        reasons.append("driver_version_mismatch")
    return {
        "target_platform": target_platform,
        "tools": rows,
        "driver": driver,
        "driver_rows": driver_rows,
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
    }


def _directory_bytes(path: Path) -> int:
    total = 0
    for root, dirnames, filenames in os.walk(path, followlinks=False):
        for dirname in dirnames:
            entry = Path(root) / dirname
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"artifact tree contains a directory symlink: {entry}")
            if not stat.S_ISDIR(mode):
                raise RuntimeError(f"artifact tree contains a special directory entry: {entry}")
        for filename in filenames:
            file_path = Path(root) / filename
            try:
                metadata = file_path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"artifact tree contains a file symlink: {file_path}")
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"artifact tree contains a special file: {file_path}")
            if metadata.st_nlink != 1:
                raise RuntimeError(f"artifact tree contains a hard-linked file: {file_path}")
            total += metadata.st_size
    return total


def _git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(
        ("/usr/bin/git", *args), cwd=_ROOT, text=text, stderr=subprocess.STDOUT
    )


def _git_identity() -> dict[str, object]:
    commit = str(_git("rev-parse", "HEAD")).strip()
    top = Path(str(_git("rev-parse", "--show-toplevel")).strip()).resolve()
    if top != _ROOT:
        raise RuntimeError(f"Git root mismatch: {top} != {_ROOT}")
    status = str(
        _git("status", "--porcelain=v1", "--untracked-files=all")
    ).splitlines()
    tracked_diff = bytes(
        _git("diff", "--binary", "--no-ext-diff", "HEAD", text=False)
    )
    untracked_raw = bytes(
        _git("ls-files", "--others", "--exclude-standard", "-z", text=False)
    )
    untracked = [item.decode("utf-8") for item in untracked_raw.split(b"\0") if item]
    untracked_digest = hashlib.sha256()
    untracked_files = []
    for relative in sorted(untracked):
        path = (_ROOT / relative).resolve()
        try:
            path.relative_to(_ROOT)
        except ValueError as error:
            raise RuntimeError(f"untracked path escapes repository: {relative}") from error
        digest = _sha256(path) if path.is_file() else None
        untracked_files.append({"path": relative, "sha256": digest})
        untracked_digest.update(relative.encode("utf-8") + b"\0")
        untracked_digest.update((digest or "non-file").encode("ascii") + b"\n")
    combined = hashlib.sha256()
    combined.update(commit.encode("ascii") + b"\n")
    combined.update(hashlib.sha256(tracked_diff).digest())
    combined.update(untracked_digest.digest())
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_tree_sha256": untracked_digest.hexdigest(),
        "untracked_files": untracked_files,
        "working_tree_identity_sha256": combined.hexdigest(),
    }


def _process_identity(pid: int) -> dict[str, object]:
    result: dict[str, object] = {"pid": pid}
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        suffix = stat_text[stat_text.rfind(")") + 2 :].split()
        result["process_group"] = int(suffix[2])
        result["starttime_ticks"] = int(suffix[19])
    except (FileNotFoundError, IndexError, ValueError, OSError) as error:
        result["stat_error"] = repr(error)
    try:
        result["exe"] = os.readlink(f"/proc/{pid}/exe")
    except OSError as error:
        result["exe_error"] = repr(error)
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        result["argv"] = [item.decode("utf-8", "replace") for item in raw.split(b"\0") if item]
    except OSError as error:
        result["cmdline_error"] = repr(error)
    return result


def _gpu_snapshot() -> dict[str, object]:
    gpu_state = _capture(
        (
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,compute_mode,mig.mode.current",
            "--format=csv,noheader,nounits",
        )
    )
    compute_apps = _capture(
        (
            "/usr/bin/nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    )
    process_table = _capture(
        ("/usr/bin/ps", "-eo", "pid=,ppid=,pgid=,etimes=,comm=,args=")
    )
    app_rows = []
    if compute_apps["returncode"] == 0:
        for line in str(compute_apps["stdout"]).splitlines():
            fields = [field.strip() for field in line.split(",", 3)]
            if len(fields) != 4:
                app_rows.append({"raw": line, "parse_error": True})
                continue
            try:
                pid: int | None = int(fields[1])
                memory_mib: int | None = int(fields[3].split()[0])
            except (IndexError, ValueError):
                pid = None
                memory_mib = None
            app_rows.append(
                {
                    "gpu_uuid": fields[0],
                    "pid": pid,
                    "process_name": fields[2],
                    "used_memory_mib": memory_mib,
                    "raw": line,
                    "process_identity": (
                        None if pid is None else _process_identity(pid)
                    ),
                }
            )
    process_lines = str(process_table["stdout"]).splitlines()
    parsed_processes = []
    for line in process_lines:
        fields = line.strip().split(None, 5)
        if len(fields) >= 5:
            parsed_processes.append(
                {
                    "pid": int(fields[0]),
                    "ppid": int(fields[1]),
                    "pgid": int(fields[2]),
                    "elapsed_seconds": int(fields[3]),
                    "comm": fields[4],
                    "raw": line,
                    "args": fields[5] if len(fields) == 6 else "",
                }
            )
    return {
        "created_utc": _utc_now(),
        "gpu_state": gpu_state,
        "compute_apps": compute_apps,
        "app_rows": app_rows,
        "process_table": process_table,
        "mps_process_rows": [
            row
            for row in parsed_processes
            if str(row["comm"]).startswith("nvidia-cuda-mps")
        ],
        "profiler_process_rows": [
            row
            for row in parsed_processes
            if row["comm"] in {"nsys", "ncu", "nvprof"}
            or str(row["comm"]).startswith("compute-sanit")
        ],
    }


def _gpu_snapshot_reasons(
    snapshot: dict[str, object],
    required_gpu_count: int,
    expected_gpu_mapping: Sequence[dict[str, object]] | None = None,
) -> list[str]:
    reasons = []
    for key in ("gpu_state", "compute_apps", "process_table"):
        command = snapshot[key]
        assert isinstance(command, dict)
        if command.get("returncode") != 0:
            reasons.append(f"{key}_unavailable")
    gpu_state = snapshot["gpu_state"]
    assert isinstance(gpu_state, dict)
    gpu_rows = [line for line in str(gpu_state.get("stdout", "")).splitlines() if line]
    if len(gpu_rows) != required_gpu_count:
        reasons.append(f"expected_{required_gpu_count}_host_gpus_got_{len(gpu_rows)}")
    parsed_gpus = []
    for line in gpu_rows:
        fields = [field.strip() for field in line.split(",", 5)]
        if len(fields) != 6:
            reasons.append("invalid_gpu_state_row")
            continue
        parsed_gpus.append(fields)
        if fields[4] != "Default":
            reasons.append(f"gpu_{fields[0]}_compute_mode_{fields[4]}")
        if fields[5] not in {"Disabled", "[N/A]", "N/A"}:
            reasons.append(f"gpu_{fields[0]}_mig_mode_{fields[5]}")
    if len({fields[0] for fields in parsed_gpus}) != len(parsed_gpus):
        reasons.append("duplicate_gpu_index")
    if len({fields[1] for fields in parsed_gpus}) != len(parsed_gpus):
        reasons.append("duplicate_gpu_uuid")
    if expected_gpu_mapping is not None:
        actual_mapping = [
            {"index": int(fields[0]), "uuid": fields[1]}
            for fields in parsed_gpus
            if fields[0].isdigit()
        ]
        expected_mapping = [
            {"index": int(row["index"]), "uuid": str(row["uuid"])}
            for row in expected_gpu_mapping
        ]
        if actual_mapping != expected_mapping:
            reasons.append("gpu_index_uuid_mapping_mismatch")
    app_rows = snapshot["app_rows"]
    assert isinstance(app_rows, list)
    if app_rows:
        pids = sorted(
            {row.get("pid") for row in app_rows if isinstance(row, dict) and row.get("pid")}
        )
        reasons.append(f"gpu_compute_processes_present:{pids}")
    if snapshot["mps_process_rows"]:
        reasons.append("cuda_mps_active")
    if snapshot["profiler_process_rows"]:
        reasons.append("external_profiler_active")
    return reasons


def _stable_gpu_preflight(
    required_gpu_count: int,
    expected_gpu_mapping: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    samples = []
    reasons: list[str] = []
    for ordinal in range(_GPU_IDLE_PROBES):
        snapshot = _gpu_snapshot()
        sample_reasons = _gpu_snapshot_reasons(
            snapshot, required_gpu_count, expected_gpu_mapping
        )
        samples.append({"ordinal": ordinal, "snapshot": snapshot, "reasons": sample_reasons})
        reasons.extend(sample_reasons)
        if sample_reasons:
            break
        if ordinal + 1 < _GPU_IDLE_PROBES:
            time.sleep(_GPU_IDLE_PROBE_INTERVAL_SECONDS)
    return {
        "idle": not reasons and len(samples) == _GPU_IDLE_PROBES,
        "required_consecutive_idle_samples": _GPU_IDLE_PROBES,
        "sample_interval_seconds": _GPU_IDLE_PROBE_INTERVAL_SECONDS,
        "samples": samples,
        "reasons": sorted(set(reasons)),
        "limitation": (
            "The lock coordinates this campaign runner only. An unrelated scheduler "
            "that does not honor the same lock can still race after the final sample; "
            "the gate is repeated before every GPU attempt."
        ),
    }


def _foreign_profiler_rows(
    rows: Sequence[object], stage_kind: str, owned_process_group: int
) -> list[object]:
    profiler_stage = stage_kind in {
        "profile_nsys",
        "profile_ncu",
        "sanitizer",
    }
    return [
        row
        for row in rows
        if not profiler_stage
        or not isinstance(row, dict)
        or row.get("pgid") != owned_process_group
    ]


def _validated_source_round_lease(
    path: Path, lease: dict[str, object]
) -> tuple[int, dict[str, object]]:
    """Validate an explicitly inherited whole-round flock lease.

    The inherited descriptor shares the executor's locked open-file
    description.  This function must never unlock it: ``LOCK_UN`` on either
    duplicate would release the executor's whole-round lease.
    """

    expected_fields = {
        "fd",
        "round_id",
        "plan_raw_sha256",
        "nonce",
        "parent_pid",
        "identity_sha256",
    }
    if set(lease) != expected_fields:
        raise RuntimeError("source-round lease arguments are incomplete or unknown")
    descriptor = lease["fd"]
    parent_pid = lease["parent_pid"]
    round_id = lease["round_id"]
    plan_sha = lease["plan_raw_sha256"]
    nonce = lease["nonce"]
    identity_sha = lease["identity_sha256"]
    if type(descriptor) is not int or descriptor < 3:
        raise RuntimeError("source-round lease FD is invalid")
    if type(parent_pid) is not int or parent_pid <= 1 or parent_pid != os.getppid():
        raise RuntimeError("source-round lease parent PID is not the live direct parent")
    if not isinstance(round_id, str) or not _RUN_ID.fullmatch(round_id):
        raise RuntimeError("source-round lease round ID is invalid")
    for value, name in ((plan_sha, "plan SHA"), (identity_sha, "identity SHA")):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise RuntimeError(f"source-round lease {name} is invalid")
    if not isinstance(nonce, str) or not _SHA256.fullmatch(nonce):
        raise RuntimeError("source-round lease nonce is invalid")

    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise RuntimeError(f"cannot inspect inherited source-round lease FD: {error}") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise RuntimeError(
            "inherited source-round lease must be a 0600 regular file owned "
            "by this user with one link"
        )
    try:
        path_metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(f"cannot inspect source-round lease path: {error}") from error
    if stat.S_ISLNK(path_metadata.st_mode) or (
        path_metadata.st_dev,
        path_metadata.st_ino,
    ) != (metadata.st_dev, metadata.st_ino):
        raise RuntimeError("source-round lease FD/path identity differs")
    try:
        parent_metadata = os.stat(
            f"/proc/{parent_pid}/fd/{descriptor}", follow_symlinks=True
        )
    except OSError as error:
        raise RuntimeError(
            f"cannot prove source-round lease ownership in parent: {error}"
        ) from error
    if (parent_metadata.st_dev, parent_metadata.st_ino) != (
        metadata.st_dev,
        metadata.st_ino,
    ):
        raise RuntimeError("source-round parent does not retain the inherited lease FD")
    if metadata.st_size <= 0 or metadata.st_size > 16 * 1024:
        raise RuntimeError("source-round lease record size is invalid")
    try:
        raw = os.pread(descriptor, metadata.st_size, 0)
        record = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise RuntimeError(f"cannot read strict source-round lease record: {error}") from error
    if not isinstance(record, dict) or set(record) != _SOURCE_ROUND_LEASE_FIELDS:
        raise RuntimeError("source-round lease record fields differ")
    expected = {
        "schema_version": 1,
        "protocol": _SOURCE_ROUND_LEASE_PROTOCOL,
        "round_id": round_id,
        "plan_raw_sha256": plan_sha,
        "executor_pid": parent_pid,
        "nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        "boot_id": _boot_id(),
        "lock_path": str(path),
        "lock_device": metadata.st_dev,
        "lock_inode": metadata.st_ino,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise RuntimeError(f"source-round lease record {field} differs")
    executor_sha = record.get("executor_code_sha256")
    shared_offset = record.get("shared_offset")
    if not isinstance(executor_sha, str) or not _SHA256.fullmatch(executor_sha):
        raise RuntimeError("source-round executor code identity is invalid")
    if _sha256(_EXECUTOR_PATH) != executor_sha:
        raise RuntimeError("source-round executor code differs from the leased identity")
    if type(shared_offset) is not int or not 1_000_000 <= shared_offset < 2**63:
        raise RuntimeError("source-round shared file offset is invalid")
    for field in ("artifact_root", "output_dir", "campaign_manifest_path"):
        value = record.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise RuntimeError(f"source-round lease record {field} is invalid")
    if record.get("run_id") is None or not _RUN_ID.fullmatch(str(record["run_id"])):
        raise RuntimeError("source-round lease record run_id is invalid")
    for field in ("artifact_root_device", "artifact_root_inode"):
        if type(record.get(field)) is not int or int(record[field]) <= 0:
            raise RuntimeError(f"source-round lease record {field} is invalid")
    _leased_artifact_root_identity(record)
    expected_output = Path(str(record["artifact_root"])) / "runs" / str(
        record["run_id"]
    )
    if Path(str(record["output_dir"])) != expected_output:
        raise RuntimeError(
            "source-round leased output must be artifact_root/runs/run_id"
        )
    campaign_manifest_sha = record.get("campaign_manifest_raw_sha256")
    if not isinstance(campaign_manifest_sha, str) or not _SHA256.fullmatch(
        campaign_manifest_sha
    ):
        raise RuntimeError("source-round campaign manifest identity is invalid")
    round_lock_identity = record.get("round_lock_identity_sha256")
    if not isinstance(round_lock_identity, str) or not _SHA256.fullmatch(
        round_lock_identity
    ):
        raise RuntimeError("source-round whole-lock identity is invalid")
    if _source_round_lock_identity(record) != round_lock_identity:
        raise RuntimeError("source-round whole-lock identity cannot be reproduced")
    if os.lseek(descriptor, 0, os.SEEK_CUR) != shared_offset:
        raise RuntimeError("source-round lease FD does not share the executor offset")
    if _canonical_sha256(record) != identity_sha:
        raise RuntimeError("source-round lease identity hash differs")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("inherited source-round lease FD does not own the flock") from error

    probe: int | None = None
    try:
        probe = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        probe_metadata = os.fstat(probe)
        if (probe_metadata.st_dev, probe_metadata.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise RuntimeError("source-round lease path changed during validation")
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise RuntimeError("source-round lease is not exclusively held")
    finally:
        if probe is not None:
            os.close(probe)
    return descriptor, record


@contextmanager
def _campaign_lock(
    path: Path, inherited_lease: dict[str, object] | None = None
) -> Iterator[dict[str, object] | None]:
    if inherited_lease is not None:
        descriptor, record = _validated_source_round_lease(path, inherited_lease)
        try:
            if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
                raise RuntimeError(
                    "source-round runner must be its own process-group/session leader"
                )
            # The executor deliberately passes this duplicate through one exec.
            # Make it non-inheritable immediately so stage descendants cannot
            # retain the round lease through any nonstandard subprocess path.
            os.set_inheritable(descriptor, False)
            yield record
        finally:
            # Closing this process's duplicate is safe.  Never issue LOCK_UN:
            # the executor retains another duplicate of the same locked OFD.
            os.close(descriptor)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as error:
        raise RuntimeError(f"cannot open campaign lock {path}: {error}") from error
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise RuntimeError(
            "campaign lock must be a 0600 regular file owned by this user "
            "with one link"
        )
    with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"campaign lock is already held: {path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={_utc_now()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield None
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _safe_output_dir(
    path: Path, source_round_lease: dict[str, object] | None = None
) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        raise RuntimeError(f"output directory already exists: {resolved}")
    if resolved == _ROOT or resolved == Path("/home/chen") or resolved == Path("/"):
        raise RuntimeError(f"unsafe output directory: {resolved}")
    if source_round_lease is not None:
        artifact_root = Path(str(source_round_lease["artifact_root"]))
        _leased_artifact_root_identity(source_round_lease)
        canonical_root = artifact_root
        expected_output = artifact_root / "runs" / str(source_round_lease["run_id"])
        if Path(str(source_round_lease["output_dir"])) != expected_output:
            raise RuntimeError("source-round output formula differs from its lease")
        runs = artifact_root / "runs"
        runs_metadata = os.stat(runs, follow_symlinks=False)
        if (
            not stat.S_ISDIR(runs_metadata.st_mode)
            or runs_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(runs_metadata.st_mode) != 0o700
        ):
            raise RuntimeError("source-round runs parent must be an owned 0700 directory")
        if resolved != expected_output or expected_output != path:
            raise RuntimeError("campaign output differs from the leased source-round path")
        try:
            resolved.relative_to(canonical_root)
        except ValueError as error:
            raise RuntimeError("source-round campaign output escapes artifact root") from error
        if resolved == canonical_root:
            raise RuntimeError("source-round campaign cannot replace the artifact root")
        return resolved
    campaign_root = (_ROOT / ".cache" / "rail_balance" / "campaigns").resolve()
    try:
        resolved.relative_to(campaign_root)
    except ValueError as error:
        raise RuntimeError(
            f"campaign output must stay below {campaign_root}"
        ) from error
    ignored = subprocess.run(
        (
            "/usr/bin/git",
            "check-ignore",
            "--quiet",
            "--no-index",
            str(resolved),
        ),
        cwd=_ROOT,
        check=False,
    )
    if ignored.returncode != 0:
        raise RuntimeError("campaign output inside the repository must be Git-ignored")
    return resolved


def _resolve_leaderboard(path_text: str) -> Path:
    path = Path(path_text)
    path = path if path.is_absolute() else _ROOT / path
    path = path.resolve()
    expected = (
        _ROOT / ".cache" / "rail_balance" / "hop_campaign" / "leaderboard.jsonl"
    ).resolve()
    if path != expected:
        raise RuntimeError(f"runtime leaderboard path must be {expected}")
    ignored = subprocess.run(
        ("/usr/bin/git", "check-ignore", "--quiet", "--no-index", str(path)),
        cwd=_ROOT,
        check=False,
    )
    if ignored.returncode != 0:
        raise RuntimeError("runtime leaderboard inside the repository must be Git-ignored")
    return path


def _process_group_members(process_group_id: int) -> list[int]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            suffix = stat_text[stat_text.rfind(")") + 2 :].split()
            if int(suffix[2]) == process_group_id:
                members.append(int(entry.name))
        except (FileNotFoundError, IndexError, ValueError, OSError):
            continue
    return sorted(members)


def _terminate_process_group(process: subprocess.Popen[object]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while _process_group_members(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.1)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while _process_group_members(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.1)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1)


def _stop_attempt_process(
    process: subprocess.Popen[object], *, inherited_round_lease: bool
) -> None:
    if not inherited_round_lease:
        _terminate_process_group(process)
        return
    # In a formal round the runner is the process-group leader.  Killing that
    # group here would kill the runner before it can report why the attempt
    # failed.  Stop only the direct attempt leader; the executor owns the
    # surrounding PGID and proves it empty (TERM then KILL) after runner exit.
    with suppress(ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    with suppress(ProcessLookupError):
        process.kill()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=2)


def _expanded_stage(
    stage: dict[str, object], output_dir: Path, stage_dir: Path, attempt: int
) -> tuple[list[str], dict[str, str]]:
    replacements = {
        "root": str(_ROOT),
        "output_dir": str(output_dir),
        "stage_dir": str(stage_dir),
        "attempt": str(attempt),
    }
    argv = [expand_template(str(item), replacements) for item in stage["argv"]]
    environment = {
        str(key): expand_template(str(value), replacements)
        for key, value in dict(stage.get("env", {})).items()
    }
    return argv, environment


def _declared_artifacts(
    stage: dict[str, object], stage_dir: Path
) -> tuple[list[dict[str, object]], list[str]]:
    rows = []
    errors = []
    try:
        _directory_bytes(stage_dir)
    except RuntimeError as error:
        return rows, [str(error)]
    for relative_text in stage.get("artifacts", []):
        relative = Path(str(relative_text))
        resolved = (stage_dir / relative).resolve()
        try:
            resolved.relative_to(stage_dir.resolve())
        except ValueError:
            errors.append(f"declared artifact escapes stage directory: {relative}")
            continue
        if not resolved.is_file():
            errors.append(f"declared artifact is missing: {relative}")
            continue
        rows.append(
            {
                "path": str(relative),
                "bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
    return rows, errors


def _child_environment(
    stage: dict[str, object],
    environment_updates: dict[str, str],
    required_gpu_count: int,
) -> dict[str, str]:
    environment = dict(_CHILD_BASE_ENVIRONMENT)
    environment.update(environment_updates)
    environment["DEEP_EP_CAMPAIGN_SUPERVISED"] = "1"
    # CPU-only gates must be incapable of stealing a visible GPU from another
    # job even if a test imports torch/deep_ep.  Conversely, an exclusive
    # attempt always sees the exact eight host indices that the parent gated;
    # it cannot inherit a caller's narrowed or reordered mapping.
    if stage["resource"] == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif stage["resource"] == "exclusive_8gpu":
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in range(required_gpu_count)
        )
    return environment


def _run_attempt(
    *,
    stage: dict[str, object],
    attempt: int,
    output_dir: Path,
    artifact_budget_bytes: int,
    home_path: Path,
    home_hard_limit_bytes: int,
    monitor_exclusive_gpus: bool,
    required_gpu_count: int,
    expected_gpu_mapping: Sequence[dict[str, object]],
    inherited_round_lease: bool = False,
) -> dict[str, object]:
    stage_dir = output_dir / "stages" / str(stage["id"]) / f"attempt-{attempt:02d}"
    stage_dir.mkdir(parents=True)
    argv, environment_updates = _expanded_stage(stage, output_dir, stage_dir, attempt)
    environment = _child_environment(
        stage, environment_updates, required_gpu_count
    )
    command = {
        "argv": argv,
        "cwd": str(_ROOT),
        "environment_overrides": environment_updates,
        "supervisor_environment": {
            "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES"),
            "DEEP_EP_CAMPAIGN_SUPERVISED": environment[
                "DEEP_EP_CAMPAIGN_SUPERVISED"
            ],
        },
        "shell": False,
    }
    _atomic_json(stage_dir / "command.json", command)
    started_utc = _utc_now()
    started = time.monotonic()
    timed_out = False
    budget_exceeded = False
    disk_limit_reached = False
    foreign_gpu_process_detected = False
    owned_process_group_leaked = False
    requires_executor_cleanup = False
    process_started = False
    launch_gpu_snapshot = None
    launch_gpu_reasons: list[str] = []
    gpu_monitor_events = []
    return_code: int | None = None
    error = None
    next_home_poll = started
    with (stage_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
        stage_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        try:
            artifact_bytes = _directory_bytes(output_dir)
            home = _home_usage(home_path)
            remaining_artifact_budget = max(
                0, artifact_budget_bytes - artifact_bytes
            )
            if artifact_bytes >= artifact_budget_bytes:
                budget_exceeded = True
                raise RuntimeError(
                    "pre-launch artifact budget gate rejected the attempt"
                )
            if (
                home["bytes"] is None
                or int(home["bytes"])
                + remaining_artifact_budget
                + _HOME_EMERGENCY_RESERVE_BYTES
                > home_hard_limit_bytes
            ):
                disk_limit_reached = True
                raise RuntimeError(
                    "pre-launch disk headroom gate rejected the attempt"
                )
            if monitor_exclusive_gpus:
                launch_gpu_snapshot = _gpu_snapshot()
                launch_gpu_reasons = _gpu_snapshot_reasons(
                    launch_gpu_snapshot,
                    required_gpu_count,
                    expected_gpu_mapping,
                )
                if launch_gpu_reasons:
                    foreign_gpu_process_detected = any(
                        reason.startswith("gpu_compute_processes_present:")
                        for reason in launch_gpu_reasons
                    )
                    raise RuntimeError(
                        "final pre-launch GPU gate rejected the attempt: "
                        + ", ".join(launch_gpu_reasons)
                    )
            process = subprocess.Popen(
                argv,
                cwd=_ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=not inherited_round_lease,
            )
            process_started = True
            owned_group = os.getpgrp() if inherited_round_lease else process.pid
            while return_code is None:
                return_code = process.poll()
                now = time.monotonic()
                if return_code is not None:
                    group_members = _process_group_members(owned_group)
                    unexpected_members = (
                        [pid for pid in group_members if pid != os.getpid()]
                        if inherited_round_lease
                        else group_members
                    )
                    if unexpected_members:
                        owned_process_group_leaked = True
                        requires_executor_cleanup = inherited_round_lease
                        if not inherited_round_lease:
                            _terminate_process_group(process)
                    break
                if now - started > int(stage["timeout_seconds"]):
                    timed_out = True
                    _stop_attempt_process(
                        process, inherited_round_lease=inherited_round_lease
                    )
                    requires_executor_cleanup = inherited_round_lease
                    return_code = process.returncode
                    break
                if _directory_bytes(output_dir) > artifact_budget_bytes:
                    budget_exceeded = True
                    _stop_attempt_process(
                        process, inherited_round_lease=inherited_round_lease
                    )
                    requires_executor_cleanup = inherited_round_lease
                    return_code = process.returncode
                    break
                if now >= next_home_poll:
                    home = _home_usage(home_path)
                    artifact_bytes = _directory_bytes(output_dir)
                    remaining_artifact_budget = max(
                        0, artifact_budget_bytes - artifact_bytes
                    )
                    if (
                        home["bytes"] is None
                        or int(home["bytes"])
                        + remaining_artifact_budget
                        + _HOME_EMERGENCY_RESERVE_BYTES
                        > home_hard_limit_bytes
                    ):
                        disk_limit_reached = True
                        _stop_attempt_process(
                            process, inherited_round_lease=inherited_round_lease
                        )
                        requires_executor_cleanup = inherited_round_lease
                        return_code = process.returncode
                        break
                    next_home_poll = now + _HOME_POLL_SECONDS
                if monitor_exclusive_gpus:
                    snapshot = _gpu_snapshot()
                    observed_reasons = _gpu_snapshot_reasons(
                        snapshot,
                        required_gpu_count,
                        expected_gpu_mapping,
                    )
                    base_reasons = [
                        reason
                        for reason in observed_reasons
                        if not reason.startswith("gpu_compute_processes_present:")
                        and reason != "external_profiler_active"
                    ]
                    profiler_rows = snapshot["profiler_process_rows"]
                    assert isinstance(profiler_rows, list)
                    foreign_profiler_rows = _foreign_profiler_rows(
                        profiler_rows, str(stage["kind"]), owned_group
                    )
                    if foreign_profiler_rows:
                        base_reasons.append("external_profiler_active")
                    foreign_pids = []
                    for row in snapshot["app_rows"]:
                        if not isinstance(row, dict) or row.get("pid") is None:
                            foreign_pids.append(row.get("pid") if isinstance(row, dict) else None)
                            continue
                        identity = row.get("process_identity")
                        process_group = (
                            identity.get("process_group")
                            if isinstance(identity, dict)
                            else None
                        )
                        if process_group != owned_group:
                            foreign_pids.append(row["pid"])
                    if base_reasons or foreign_pids:
                        foreign_gpu_process_detected = bool(foreign_pids)
                        gpu_monitor_events.append(
                            {
                                "created_utc": _utc_now(),
                                "base_reasons": base_reasons,
                                "foreign_profiler_rows": foreign_profiler_rows,
                                "foreign_gpu_pids": sorted(
                                    pid for pid in foreign_pids if isinstance(pid, int)
                                ),
                            }
                        )
                        _stop_attempt_process(
                            process, inherited_round_lease=inherited_round_lease
                        )
                        requires_executor_cleanup = inherited_round_lease
                        return_code = process.returncode
                        break
                time.sleep(_RESOURCE_POLL_SECONDS)
        except BaseException:
            error = traceback.format_exc()
            if "process" in locals() and process.poll() is None:
                _stop_attempt_process(
                    process, inherited_round_lease=inherited_round_lease
                )
                requires_executor_cleanup = inherited_round_lease
            return_code = None if "process" not in locals() else process.returncode
    artifacts, artifact_errors = _declared_artifacts(stage, stage_dir)
    passed = (
        return_code == 0
        and not timed_out
        and not budget_exceeded
        and not disk_limit_reached
        and not foreign_gpu_process_detected
        and not owned_process_group_leaked
        and error is None
        and not artifact_errors
    )
    result = {
        "attempt": attempt,
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "duration_seconds": time.monotonic() - started,
        "returncode": return_code,
        "timed_out": timed_out,
        "artifact_budget_exceeded": budget_exceeded,
        "home_hard_limit_reached": disk_limit_reached,
        "foreign_gpu_process_detected": foreign_gpu_process_detected,
        "owned_process_group_leaked": owned_process_group_leaked,
        "requires_executor_cleanup": requires_executor_cleanup,
        "process_started": process_started,
        "launch_gpu_snapshot": launch_gpu_snapshot,
        "launch_gpu_reasons": launch_gpu_reasons,
        "gpu_monitor_events": gpu_monitor_events,
        "error": error,
        "declared_artifacts": artifacts,
        "artifact_errors": artifact_errors,
        "passed": passed,
    }
    _atomic_json(stage_dir / "result.json", result)
    return result


def _stage_skip(stage: dict[str, object], reason: str) -> dict[str, object]:
    return {
        "id": stage["id"],
        "kind": stage["kind"],
        "resource": stage["resource"],
        "status": "skipped",
        "reason": reason,
        "attempts": [],
    }


def _run_stages(
    *,
    manifest: dict[str, object],
    output_dir: Path,
    gpu_ready: bool,
    no_execute: bool,
    expected_source_identity_sha256: str,
    inherited_round_lease: bool,
) -> tuple[list[dict[str, object]], bool]:
    resources = dict(manifest["resources"])
    results: list[dict[str, object]] = []
    by_id: dict[str, dict[str, object]] = {}
    environment_lost = False
    for raw_stage in manifest["stages"]:
        stage = dict(raw_stage)
        stage_id = str(stage["id"])
        if not stage.get("enabled", True):
            result = _stage_skip(stage, f"disabled: {stage['skip_reason']}")
        elif no_execute:
            result = _stage_skip(stage, "--no-execute command rendering only")
        elif any(by_id[str(dependency)]["status"] != "passed" for dependency in stage["depends_on"]):
            result = _stage_skip(stage, "dependency gate did not pass")
        elif stage["resource"] == "multinode":
            result = _stage_skip(stage, "multinode execution requires coordinated node launch")
        elif stage["resource"] == "exclusive_8gpu" and not gpu_ready:
            result = _stage_skip(stage, "exclusive eight-GPU preflight unavailable")
        else:
            if stage["resource"] == "exclusive_8gpu":
                boundary_source = _git_identity()
                _atomic_json(
                    output_dir / "environment" / f"source-pre-{stage_id}.json",
                    boundary_source,
                )
                if (
                    boundary_source["working_tree_identity_sha256"]
                    != expected_source_identity_sha256
                ):
                    environment_lost = True
                    gpu_ready = False
                    result = _stage_skip(stage, "source drift before GPU stage")
                    results.append(result)
                    by_id[stage_id] = result
                    continue
                boundary = _stable_gpu_preflight(
                    int(resources["required_gpu_count"]),
                    list(resources["expected_gpu_index_uuid_mapping"]),
                )
                _atomic_json(
                    output_dir / "environment" / f"pre-{stage_id}.json", boundary
                )
                if not boundary["idle"]:
                    environment_lost = True
                    gpu_ready = False
                    result = _stage_skip(stage, "GPU boundary preflight failed")
                    results.append(result)
                    by_id[stage_id] = result
                    continue
            attempts = []
            for attempt in range(1, int(stage.get("repeat", 1)) + 1):
                attempt_result = _run_attempt(
                    stage=stage,
                    attempt=attempt,
                    output_dir=output_dir,
                    artifact_budget_bytes=int(resources["artifact_budget_bytes"]),
                    home_path=Path(str(resources["home_path"])),
                    home_hard_limit_bytes=int(resources["home_hard_limit_bytes"]),
                    monitor_exclusive_gpus=(
                        stage["resource"] == "exclusive_8gpu"
                    ),
                    required_gpu_count=int(resources["required_gpu_count"]),
                    expected_gpu_mapping=list(
                        resources["expected_gpu_index_uuid_mapping"]
                    ),
                    inherited_round_lease=inherited_round_lease,
                )
                attempts.append(attempt_result)
                if attempt_result.get("requires_executor_cleanup") is True:
                    raise RuntimeError(
                        "formal source-round attempt left processes for executor cleanup"
                    )
            result = {
                "id": stage_id,
                "kind": stage["kind"],
                "resource": stage["resource"],
                "status": "passed" if all(item["passed"] for item in attempts) else "failed",
                "reason": None,
                "attempts": attempts,
            }
            if stage["resource"] == "exclusive_8gpu":
                source_post = _git_identity()
                _atomic_json(
                    output_dir / "environment" / f"source-post-{stage_id}.json",
                    source_post,
                )
                if (
                    source_post["working_tree_identity_sha256"]
                    != expected_source_identity_sha256
                ):
                    environment_lost = True
                    gpu_ready = False
                    result["status"] = "failed"
                    result["reason"] = "source drift during GPU stage"
        results.append(result)
        by_id[stage_id] = result
    return results, environment_lost


def _write_sha256sums(output_dir: Path) -> Path:
    # Refuse links and special files before opening anything for hashing.  The
    # campaign artifact must be a self-contained regular-file tree.
    _directory_bytes(output_dir)
    path = output_dir / "SHA256SUMS"
    rows = []
    for candidate in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if candidate == path:
            continue
        rows.append(f"{_sha256(candidate)}  {candidate.relative_to(output_dir)}")
    _atomic_bytes(path, ("\n".join(rows) + "\n").encode("utf-8"))
    return path


def _encode_leaderboard_row(row: dict[str, object]) -> tuple[str, bytes]:
    identity = {
        key: row[key]
        for key in (
            "campaign_id",
            "run_id",
            "candidate",
            "source_identity_sha256",
            "manifest_canonical_sha256",
            "status",
            "artifact_dir",
            "sha256sums_sha256",
        )
    }
    canonical = json.dumps(
        identity, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    entry_id = hashlib.sha256(canonical).hexdigest()
    encoded = json.dumps(
        {**row, "entry_id": entry_id}, allow_nan=False, sort_keys=True
    ).encode("utf-8") + b"\n"
    return entry_id, encoded


def _append_jsonl(path: Path, row: dict[str, object]) -> tuple[str, bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry_id, encoded = _encode_leaderboard_row(row)
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    appended = False
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise RuntimeError(
                "runtime leaderboard must be a 0600 regular file owned by "
                "this user with one link"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        for line in b"".join(chunks).splitlines():
            try:
                existing = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RuntimeError(
                    f"runtime leaderboard contains invalid JSON: {path}"
                ) from error
            if existing.get("entry_id") == entry_id:
                break
        else:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise RuntimeError("short write while appending runtime leaderboard")
                offset += written
            appended = True
        os.fsync(descriptor)
    finally:
        with suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    if appended:
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return entry_id, appended


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scaffold-only", action="store_true")
    parser.add_argument("--no-execute", action="store_true")
    parser.add_argument(
        "--source-round-lease",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--source-round-lease-fd",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--source-round-lease-round-id",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--source-round-lease-plan-sha256",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if not _RUN_ID.fullmatch(args.run_id):
        parser.error("run-id must be a safe lowercase slug")
    environment_present = {
        name for name in _SOURCE_ROUND_LEASE_ENV if name in os.environ
    }
    option_present = any(
        value is not None
        for value in (
            args.source_round_lease_fd,
            args.source_round_lease_round_id,
            args.source_round_lease_plan_sha256,
        )
    )
    if not args.source_round_lease:
        if environment_present or option_present:
            parser.error(
                "source-round lease material is forbidden without the explicit "
                "--source-round-lease capability"
            )
        args.inherited_lease = None
        return args
    if args.scaffold_only or args.no_execute:
        parser.error("source-round lease cannot be used for a non-live run")
    if environment_present != _SOURCE_ROUND_LEASE_ENV:
        parser.error("source-round lease environment is incomplete")
    if (
        args.source_round_lease_fd is None
        or args.source_round_lease_round_id is None
        or args.source_round_lease_plan_sha256 is None
    ):
        parser.error("source-round lease argv is incomplete")
    try:
        environment_fd = int(os.environ["DEEP_EP_SOURCE_ROUND_LEASE_FD"])
        parent_pid = int(os.environ["DEEP_EP_SOURCE_ROUND_LEASE_PARENT_PID"])
    except ValueError as error:
        parser.error(f"source-round lease integer is invalid: {error}")
    if environment_fd != args.source_round_lease_fd:
        parser.error("source-round lease FD differs between argv and environment")
    if (
        os.environ["DEEP_EP_SOURCE_ROUND_LEASE_ROUND_ID"]
        != args.source_round_lease_round_id
        or os.environ["DEEP_EP_SOURCE_ROUND_LEASE_PLAN_SHA256"]
        != args.source_round_lease_plan_sha256
    ):
        parser.error("source-round lease identity differs between argv and environment")
    args.inherited_lease = {
        "fd": args.source_round_lease_fd,
        "round_id": args.source_round_lease_round_id,
        "plan_raw_sha256": args.source_round_lease_plan_sha256,
        "nonce": os.environ["DEEP_EP_SOURCE_ROUND_LEASE_NONCE"],
        "parent_pid": parent_pid,
        "identity_sha256": os.environ[
            "DEEP_EP_SOURCE_ROUND_LEASE_IDENTITY_SHA256"
        ],
    }
    return args


def _terminal_exit(overall_status: str, explicit_non_gpu_run: bool) -> tuple[str, int]:
    if overall_status == "complete":
        return "PASS", 0
    if overall_status in {"scaffold_only", "partial"} and explicit_non_gpu_run:
        return "SCAFFOLD", 0
    return "FAIL", 3


def _finalization_reasons(
    *,
    artifact_bytes: int,
    artifact_budget_bytes: int,
    home_bytes: int | None,
    home_hard_limit_bytes: int,
) -> list[str]:
    reasons = []
    if artifact_bytes + _FINALIZATION_RESERVE_BYTES > artifact_budget_bytes:
        reasons.append("insufficient_artifact_finalization_reserve")
    if home_bytes is None:
        reasons.append("home_usage_unavailable_before_finalization")
    elif (
        home_bytes
        + _HOME_EMERGENCY_RESERVE_BYTES
        + _FINALIZATION_RESERVE_BYTES
        > home_hard_limit_bytes
    ):
        reasons.append("insufficient_home_finalization_reserve")
    return reasons


def _tmp_failure_journal(run_id: str, value: object) -> Path:
    path = Path("/tmp") / (
        f"deepep-rail-balance-hop-{run_id}-{os.getpid()}-failure.json"
    )
    _atomic_json(path, value)
    return path


def main() -> int:
    args = _parse_args()
    manifest_path = Path(os.path.abspath(args.manifest))
    manifest, raw_manifest_sha256, canonical_manifest_sha256 = load_manifest(
        manifest_path
    )
    manifest_raw = read_manifest_bytes(manifest_path)
    if hashlib.sha256(manifest_raw).hexdigest() != raw_manifest_sha256:
        raise RuntimeError("manifest changed while it was being validated")
    resources = dict(manifest["resources"])
    default_output = (
        _ROOT
        / ".cache"
        / "rail_balance"
        / "campaigns"
        / str(manifest["campaign_id"])
        / args.run_id
    )
    requested_output_dir = args.output_dir or default_output
    leaderboard_path = _resolve_leaderboard(str(manifest["leaderboard_path"]))
    lock_path = Path(str(resources["lock_path"]))

    with _campaign_lock(lock_path, args.inherited_lease) as lease_record:
        if lease_record is not None:
            if lease_record["run_id"] != args.run_id:
                raise RuntimeError("campaign run ID differs from source-round lease")
            if Path(str(lease_record["campaign_manifest_path"])) != manifest_path:
                raise RuntimeError("campaign manifest path differs from source-round lease")
            if lease_record["campaign_manifest_raw_sha256"] != raw_manifest_sha256:
                raise RuntimeError("campaign manifest hash differs from source-round lease")
        output_dir = _safe_output_dir(requested_output_dir, lease_record)
        home_pre = _home_usage(Path(str(resources["home_path"])))
        if home_pre["bytes"] is None:
            raise RuntimeError("cannot measure /home/chen usage; refusing to write")
        projected = (
            int(home_pre["bytes"])
            + int(resources["artifact_budget_bytes"])
            + _HOME_EMERGENCY_RESERVE_BYTES
        )
        if projected > int(resources["home_hard_limit_bytes"]):
            raise RuntimeError(
                "campaign budget plus emergency reserve would cross the 300 GB home limit"
            )
        if int(resources["home_hard_limit_bytes"]) != HOME_HARD_LIMIT_BYTES:
            raise RuntimeError("manifest weakened the fixed home hard limit")

        source_identity = _git_identity()
        target_platform = dict(manifest["frozen_contract"])["target_platform"]
        if not isinstance(target_platform, dict):
            raise RuntimeError("frozen target_platform must be an object")
        toolchain = _toolchain_snapshot(target_platform)
        gpu_preflight = _stable_gpu_preflight(
            int(resources["required_gpu_count"]),
            list(resources["expected_gpu_index_uuid_mapping"]),
        )
        campaign_preflight_reasons = list(gpu_preflight["reasons"])
        campaign_preflight_reasons.extend(toolchain["reasons"])
        if source_identity["dirty"]:
            campaign_preflight_reasons.append("source_worktree_dirty")
        if args.scaffold_only:
            campaign_preflight_reasons.append("scaffold_only_requested")
        gpu_ready = not campaign_preflight_reasons
        output_dir.mkdir(parents=True)
        (output_dir / "environment").mkdir()
        if lease_record is not None:
            _atomic_json(
                output_dir / "environment" / "source-round-lease.json",
                {
                    **lease_record,
                    "identity_sha256": _canonical_sha256(lease_record),
                    "inherited_flock_validated": True,
                },
            )
        _atomic_bytes(output_dir / "manifest.raw.json", manifest_raw)
        _atomic_json(output_dir / "manifest.json", manifest)
        _atomic_json(output_dir / "environment" / "home-pre.json", home_pre)
        _atomic_json(output_dir / "environment" / "gpu-preflight.json", gpu_preflight)
        _atomic_json(output_dir / "environment" / "toolchain.json", toolchain)
        _atomic_json(output_dir / "environment" / "source-identity.json", source_identity)

        stage_results, environment_lost = _run_stages(
            manifest=manifest,
            output_dir=output_dir,
            gpu_ready=gpu_ready,
            no_execute=args.no_execute,
            expected_source_identity_sha256=str(
                source_identity["working_tree_identity_sha256"]
            ),
            inherited_round_lease=lease_record is not None,
        )
        source_identity_post = _git_identity()
        source_drift = (
            source_identity_post["working_tree_identity_sha256"]
            != source_identity["working_tree_identity_sha256"]
        )
        environment_lost = environment_lost or source_drift
        _atomic_json(
            output_dir / "environment" / "source-identity-post.json",
            source_identity_post,
        )
        home_post = _home_usage(Path(str(resources["home_path"])))
        _atomic_json(output_dir / "environment" / "home-post.json", home_post)
        artifact_bytes_before_result = _directory_bytes(output_dir)
        pre_finalization_reasons = _finalization_reasons(
            artifact_bytes=artifact_bytes_before_result,
            artifact_budget_bytes=int(resources["artifact_budget_bytes"]),
            home_bytes=(
                None if home_post["bytes"] is None else int(home_post["bytes"])
            ),
            home_hard_limit_bytes=int(resources["home_hard_limit_bytes"]),
        )
        if pre_finalization_reasons:
            journal = _tmp_failure_journal(
                args.run_id,
                {
                    "created_utc": _utc_now(),
                    "run_id": args.run_id,
                    "status": "blocked_disk",
                    "reasons": pre_finalization_reasons,
                    "artifact_dir": str(output_dir),
                    "artifact_bytes": artifact_bytes_before_result,
                    "home_usage_post_bytes": home_post["bytes"],
                },
            )
            print(
                "FAIL campaign finalization refused to write under /home; "
                f"journal={journal}",
                file=sys.stderr,
                flush=True,
            )
            return 3
        if home_post["bytes"] is None or int(home_post["bytes"]) >= HOME_HARD_LIMIT_BYTES:
            overall_status = "blocked_disk"
        elif any(stage["status"] == "failed" for stage in stage_results):
            overall_status = "failed"
        elif environment_lost:
            overall_status = "environment_lost"
        elif not gpu_ready:
            overall_status = "scaffold_only"
        elif any(
            stage["status"] == "skipped"
            and stage["resource"] != "multinode"
            and "disabled" not in str(stage["reason"])
            for stage in stage_results
        ):
            overall_status = "partial"
        else:
            overall_status = "complete"

        result = {
            "schema_version": 1,
            "run_id": args.run_id,
            "campaign_id": manifest["campaign_id"],
            "candidate": manifest["candidate"],
            "created_utc": _utc_now(),
            "status": overall_status,
            "claim_scope": (
                "scaffold_only"
                if overall_status in {"scaffold_only", "partial", "environment_lost"}
                else "failed_candidate"
                if overall_status in {"failed", "blocked_disk"}
                else "local_candidate_gate"
            ),
            "performance_claim_allowed": False,
            "manifest_raw_sha256": raw_manifest_sha256,
            "manifest_canonical_sha256": canonical_manifest_sha256,
            "runner_sha256": _sha256(_RUNNER_PATH),
            "schema_sha256": _sha256(_SCHEMA_PATH),
            "source_identity": source_identity,
            "source_identity_post": source_identity_post,
            "source_drift": source_drift,
            "gpu_preflight_reasons": gpu_preflight["reasons"],
            "campaign_preflight_reasons": campaign_preflight_reasons,
            "stages": stage_results,
            "home_usage_pre_bytes": home_pre["bytes"],
            "home_usage_post_bytes": home_post["bytes"],
            "artifact_bytes_before_result": artifact_bytes_before_result,
            "artifact_budget_bytes": resources["artifact_budget_bytes"],
            "limitations": [
                "No performance claim is automatic; benchmark noise and profile attribution require review.",
                "A scaffold-only result contains no GPU or network performance evidence.",
                "Multinode stages require coordinated per-node launch and physical NIC/QP/runtime counters.",
            ],
        }
        explicit_non_gpu_run = args.scaffold_only or args.no_execute
        label, exit_code = _terminal_exit(
            overall_status, explicit_non_gpu_run
        )
        _atomic_json(output_dir / "result.json", result)
        sums = _write_sha256sums(output_dir)
        artifact_bytes_final = _directory_bytes(output_dir)
        home_final = _home_usage(Path(str(resources["home_path"])))
        finalization_reasons = []
        if artifact_bytes_final > int(resources["artifact_budget_bytes"]):
            finalization_reasons.append("artifact_budget_exceeded_after_finalization")
        if (
            home_final["bytes"] is None
            or int(home_final["bytes"]) + _HOME_EMERGENCY_RESERVE_BYTES
            > HOME_HARD_LIMIT_BYTES
        ):
            finalization_reasons.append("home_headroom_lost_after_finalization")
        if finalization_reasons:
            journal = _tmp_failure_journal(
                args.run_id,
                {
                    "created_utc": _utc_now(),
                    "run_id": args.run_id,
                    "status": "blocked_disk",
                    "reasons": finalization_reasons,
                    "artifact_dir": str(output_dir),
                    "artifact_bytes": artifact_bytes_final,
                    "home_usage_final_bytes": home_final["bytes"],
                },
            )
            print(
                "FAIL campaign crossed its finalization safety threshold; "
                f"journal={journal}",
                file=sys.stderr,
                flush=True,
            )
            return 3
        leaderboard_row = {
            "schema_version": 1,
            "created_utc": _utc_now(),
            "run_id": args.run_id,
            "campaign_id": manifest["campaign_id"],
            "candidate": manifest["candidate"],
            "source_commit": source_identity["commit"],
            "source_dirty": source_identity["dirty"],
            "source_identity_sha256": source_identity["working_tree_identity_sha256"],
            "manifest_canonical_sha256": canonical_manifest_sha256,
            "runner_sha256": result["runner_sha256"],
            "schema_sha256": result["schema_sha256"],
            "status": overall_status,
            "compile": [stage for stage in stage_results if stage["kind"] == "compile"],
            "correctness": [stage for stage in stage_results if stage["kind"] == "correctness"],
            "sanitizer": [
                stage for stage in stage_results if stage["kind"] == "sanitizer"
            ],
            "benchmark": [stage for stage in stage_results if stage["kind"] == "benchmark"],
            "profile": [stage for stage in stage_results if str(stage["kind"]).startswith("profile_")],
            "latency": None,
            "speedup": None,
            "noise_verdict": "not_measured",
            "conclusion": "scaffold only" if overall_status == "scaffold_only" else overall_status,
            "failure_reason": (
                campaign_preflight_reasons
                if not gpu_ready
                else ["source_drift"]
                if source_drift
                else [
                    f"{stage['id']}:{stage['reason'] or stage['status']}"
                    for stage in stage_results
                    if stage["status"] == "failed"
                ]
                or None
            ),
            "artifact_dir": str(output_dir),
            "finalized_record": str(output_dir / "FINALIZED.json"),
            "round_evaluation_requires_finalized_record": True,
            "sha256sums_sha256": _sha256(sums),
            "artifact_bytes_final": artifact_bytes_final,
            "home_usage_final_bytes": home_final["bytes"],
        }
        expected_entry_id, leaderboard_payload = _encode_leaderboard_row(
            leaderboard_row
        )
        terminal_record = {
            "schema_version": 1,
            "created_utc": _utc_now(),
            "campaign_id": manifest["campaign_id"],
            "run_id": args.run_id,
            "campaign_status": overall_status,
            "claim_scope": result["claim_scope"],
            "terminal_label": label,
            "terminal_exit_code": exit_code,
            "round_evaluation_allowed": (
                overall_status == "complete" and exit_code == 0
            ),
            "result_sha256": _sha256(output_dir / "result.json"),
            "sha256sums_sha256": _sha256(sums),
            "source_identity_sha256": source_identity[
                "working_tree_identity_sha256"
            ],
            "manifest_raw_sha256": raw_manifest_sha256,
            "leaderboard_entry_id": expected_entry_id,
        }
        terminal_payload = (
            json.dumps(
                terminal_record, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n"
        ).encode("utf-8")
        commit_reasons = []
        if (
            artifact_bytes_final + len(terminal_payload)
            > int(resources["artifact_budget_bytes"])
        ):
            commit_reasons.append("terminal_record_would_exceed_artifact_budget")
        if (
            home_final["bytes"] is None
            or int(home_final["bytes"])
            + len(leaderboard_payload)
            + len(terminal_payload)
            + _HOME_EMERGENCY_RESERVE_BYTES
            > HOME_HARD_LIMIT_BYTES
        ):
            commit_reasons.append("terminal_commit_would_cross_home_reserve")
        if commit_reasons:
            journal = _tmp_failure_journal(
                args.run_id,
                {
                    "created_utc": _utc_now(),
                    "run_id": args.run_id,
                    "status": "blocked_disk",
                    "reasons": commit_reasons,
                    "artifact_dir": str(output_dir),
                    "artifact_bytes": artifact_bytes_final,
                    "home_usage_final_bytes": home_final["bytes"],
                },
            )
            print(
                "FAIL campaign cannot publish its terminal commit record; "
                f"journal={journal}",
                file=sys.stderr,
                flush=True,
            )
            return 3
        entry_id, _ = _append_jsonl(leaderboard_path, leaderboard_row)
        if entry_id != expected_entry_id:
            raise RuntimeError("leaderboard entry identity changed during append")
        if lease_record is not None:
            _leased_artifact_root_identity(lease_record)
            final_output = os.stat(output_dir, follow_symlinks=False)
            if (
                not stat.S_ISDIR(final_output.st_mode)
                or final_output.st_uid != os.geteuid()
            ):
                raise RuntimeError(
                    "source-round campaign output identity is unsafe at finalization"
                )
        # This is the last write below the campaign directory.  Formal round
        # evaluators must require this commit record; a provisional result left
        # behind by any earlier failure is not eligible evidence.
        _atomic_bytes(output_dir / "FINALIZED.json", terminal_payload)
        stream = sys.stdout if exit_code == 0 else sys.stderr
        print(
            f"{label} campaign orchestration status={overall_status} "
            f"claim_scope={result['claim_scope']} -> {output_dir / 'result.json'}",
            file=stream,
            flush=True,
        )
        return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error
