#!/usr/bin/false
"""Immutable, CPU-only preflight for frozen local EP competitors.

The only supported mode is ``check_only``.  Build commands in the manifest
are data: this program never executes them, imports a Python extension, or
starts torchrun/mpirun/CUDA.  It records enough static evidence to decide
whether a later, separately authorized build may start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


sys.dont_write_bytecode = True

SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
HOME_HARD_LIMIT_BYTES = 300_000_000_000
ARTIFACT_BUDGET_BYTES = 8_000_000_000
EMERGENCY_RESERVE_BYTES = 1_000_000_000
MAX_JSON_BYTES = 1 << 20
MAX_TOOL_BYTES = 64 << 20
MAX_TRACKED_FILE_BYTES = 512 << 20
MAX_TRACKED_SUBTREE_BYTES = 2 << 30
COMMAND_TIMEOUT_SECONDS = 180
OUTPUT_ROOT = Path(
    "/home/chen/workspace/source_code/DeepEP/.cache/rail_balance/competitors"
)
PROGRAM_PATH = Path(
    "/home/chen/workspace/source_code/DeepEP/"
    "tests/elastic/rail_balance_competitor_preflight.py"
)
MANIFEST_PATH = Path(
    "/home/chen/workspace/source_code/DeepEP/"
    "tests/elastic/experiments/rail_balance_competitor_local_v1.json"
)
_PINNED_PREFLIGHT_INTERPRETER = (
    "/usr/bin/python3",
    "/usr/bin/python3.12",
    "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118",
    0,
)
_EXPECTED_PROCESS_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_GPU_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")
_PCI_BDF = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\Z")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

_PINNED_COMPETITORS: Mapping[
    str, tuple[str, str, str, str, str, str]
] = {
    "nccl-ep": (
        "/home/chen/workspace/infra/nccl-ep-v0.1.0",
        "63cf786b015b2b6bff6cf263461621acf584bd18",
        "22f0c4ad56f505c4b47dd7a8f0a3c4f93efed4ce",
        "contrib/nccl_ep",
        "a417b4e0b427bd8b2a815796d19be0be34eba48c",
        "0bfef077ebd863d7d4e01833779082bd4217bf2bf9571bf8b975afe3202b5fac",
    ),
    "uccl-ep": (
        "/home/chen/workspace/infra/uccl-ep-61ee4240",
        "61ee42402819cabba3ac2a56dd4addec3363976c",
        "0a9acaafb8fd24cde14b76ee01f0ab5a1c4c4161",
        "ep",
        "a1c4179b18024bf8d5c47c818e9844d45b43f409",
        "d248f24f4c259f27a0dabe1bd8a8317c1682ea4c35adfe61ea83b2a1447ae8bb",
    ),
}
_EXPECTED_SOURCE_MATERIALIZATIONS: Mapping[
    str, tuple[tuple[str, ...], str, str]
] = {
    "nccl-ep": (
        (".",),
        "{stage_root}/nccl-ep/source-copy",
        "d5786299223487c99506997c8caf6e093ffc9fd1659e41a7601c4279164d705b",
    ),
    "uccl-ep": (
        ("ep", "include"),
        "{stage_root}/uccl/source-copy",
        "0c4f219e052c49b16b93c6b1cc3b083fabc5517331eaacc53d1f63701531f368",
    ),
}
_PINNED_TOOL_IDENTITIES: Mapping[str, tuple[str, str, str, int]] = {
    "du": (
        "/usr/bin/du",
        "/usr/bin/du",
        "860235f0294d9c2185140f4ebab8301d5db8357fb195f7b79ad3b94b495763c7",
        0,
    ),
    "git": (
        "/usr/bin/git",
        "/usr/bin/git",
        "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668",
        0,
    ),
    "gxx": (
        "/usr/bin/g++",
        "/usr/bin/x86_64-linux-gnu-g++-13",
        "1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769",
        0,
    ),
    "ibv_devices": (
        "/usr/bin/ibv_devices",
        "/usr/bin/ibv_devices",
        "41732736a50a5a11c408ad295b9b6cf5a98e5431853a3ea6da9fd33b82127b9f",
        0,
    ),
    "make": (
        "/usr/bin/make",
        "/usr/bin/make",
        "d78b8f1d099fbcfb6f2f49ab87223b9b68fb3956642f92d6ec6de812e8afa965",
        0,
    ),
    "mpicc": (
        "/usr/bin/mpicc",
        "/usr/bin/opal_wrapper",
        "4f59e0ba9ae1fb3c3d3e8ae494eb8f0cd8ca563fb9f9b88f523e0ded9ee291af",
        0,
    ),
    "mpirun": (
        "/usr/bin/mpirun",
        "/usr/bin/orterun",
        "e3a009cd4ab8b41ef23019df3a1388944294b330b7c51ad17be18a0120b36daa",
        0,
    ),
    "nvcc": (
        "/usr/local/cuda/bin/nvcc",
        "/public/lianghong/shenjie/cuda128_full/bin/nvcc",
        "3aadf006f53dd288a372b2c4fabd46df1caa25aacd2e60964d7a26af25e91517",
        0,
    ),
    "nvidia_smi": (
        "/usr/bin/nvidia-smi",
        "/usr/bin/nvidia-smi",
        "b6ea95ba54db6dd177bc463bb441853ba0f0d08bd978a1f169090c9b0505ce18",
        0,
    ),
    "python": (
        "/home/chen/.cache/deepep-sjlgpt/bin/python",
        "/home/chen/.cache/deepep-sjlgpt/bin/python3.11",
        "8180848c5183d004dd3478f9522dca9076e77a09b1a4c9f3ef9a430749751ac6",
        1004,
    ),
    "readelf": (
        "/usr/bin/readelf",
        "/usr/bin/x86_64-linux-gnu-readelf",
        "64c58e15274bbbb5153f31078e455e9e77ee5f51489e709bba5bb788ce9df2b0",
        0,
    ),
}
_PINNED_TOOL_PATHS: Mapping[str, str] = {
    tool_id: identity[0]
    for tool_id, identity in _PINNED_TOOL_IDENTITIES.items()
}
_EXPECTED_BUILD_CWD: Mapping[str, str] = {
    "nccl-ep-core": "{stage_root}/nccl-ep",
    "nccl-ep-correctness": "{stage_root}/nccl-ep",
    "uccl-ep-extension": "{stage_root}/uccl/source-copy/ep",
}
_EXPECTED_BUILD_ARGV: Mapping[str, tuple[str, ...]] = {
    "nccl-ep-core": (
        "/usr/bin/make",
        "-C",
        "{stage_root}/nccl-ep/source-copy",
        "-j8",
        "src.build",
        "BUILDDIR={stage_root}/nccl-ep/nccl-core",
        "CUDA_HOME=/usr/local/cuda",
        "CXX=/usr/bin/g++",
        "VERBOSE=1",
        "NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90",
    ),
    "nccl-ep-correctness": (
        "/usr/bin/make",
        "-C",
        "{stage_root}/nccl-ep/source-copy/contrib/nccl_ep",
        "-j8",
        "ep_test",
        "ep_bench",
        "BUILDDIR={stage_root}/nccl-ep/nccl-ep",
        "NCCL_HOME={stage_root}/nccl-ep/nccl-core",
        "NCCL_EP_BUILDDIR={stage_root}/nccl-ep/nccl-ep",
        "MPI=1",
        "MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi",
        "CUDA_HOME=/usr/local/cuda",
        "CXX=/usr/bin/g++",
        "VERBOSE=1",
        "NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90",
    ),
    "uccl-ep-extension": (
        "/usr/bin/make",
        "-C",
        "{stage_root}/uccl/source-copy/ep",
        "-j8",
        "all",
        "PYTHON=/home/chen/.cache/deepep-sjlgpt/bin/python",
        "CUDA_PATH=/usr/local/cuda",
        "DETECTED_SM=90",
        "SM=90",
        "GPU_NAME=NVIDIA L20X",
        "EFA_HOME=/nonexistent",
        "USE_DMABUF=0",
        "USE_LIBFABRIC_CXI=0",
        "USE_INTEL_RDMA_NIC=0",
        "PER_EXPERT_BATCHING=0",
        "MAX_NUM_GPUS=8",
        "NUM_MAX_NVL_PEERS=8",
    ),
}
_EXPECTED_BUILD_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "nccl-ep-core": (),
    "nccl-ep-correctness": ("nccl-ep-core",),
    "uccl-ep-extension": (),
}
_EXPECTED_BUILD_OUTPUTS: Mapping[str, tuple[str, ...]] = {
    "nccl-ep-core": (
        "{stage_root}/nccl-ep/nccl-core/include/nccl.h",
        "{stage_root}/nccl-ep/nccl-core/lib/libnccl.so.2.30.7",
    ),
    "nccl-ep-correctness": (
        "{stage_root}/nccl-ep/nccl-ep/include/nccl_ep.h",
        "{stage_root}/nccl-ep/nccl-ep/lib/libnccl_ep.a",
        "{stage_root}/nccl-ep/nccl-ep/lib/libnccl_ep.so",
        "{stage_root}/nccl-ep/nccl-ep/lib/libnccl_ep.so.0",
        "{stage_root}/nccl-ep/nccl-ep/lib/libnccl_ep.so.0.1.0",
        "{stage_root}/nccl-ep/nccl-ep/test/nccl_ep/ep_bench",
        "{stage_root}/nccl-ep/nccl-ep/test/nccl_ep/ep_test",
    ),
    "uccl-ep-extension": (
        "{stage_root}/uccl/source-copy/ep/ep.cpython-311-x86_64-linux-gnu.so",
    ),
}
_EXPECTED_BUILD_ENV: Mapping[str, Mapping[str, str]] = {
    "nccl-ep-core": {
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": "{stage_root}/nccl-ep/home",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "TMPDIR": "{stage_root}/nccl-ep/tmp",
    },
    "nccl-ep-correctness": {
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": "{stage_root}/nccl-ep/home",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "TMPDIR": "{stage_root}/nccl-ep/tmp",
    },
    "uccl-ep-extension": {
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": "{stage_root}/uccl/home",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": (
            "{stage_root}/uccl/toolshim:/usr/local/cuda/bin:/usr/bin:/bin"
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "{stage_root}/uccl/python_deps",
        "TMPDIR": "{stage_root}/uccl/tmp",
    },
}
_EXPECTED_DEPENDENCIES: Mapping[
    str, tuple[str, str, tuple[str, ...], str]
] = {
    "common-cuda-header": (
        "common",
        "file",
        ("/usr/local/cuda/include/cuda.h",),
        "BLOCKED_DEPENDENCY",
    ),
    "common-nvcc": (
        "common",
        "executable",
        ("/usr/local/cuda/bin/nvcc",),
        "BLOCKED_DEPENDENCY",
    ),
    "common-python": (
        "common",
        "executable",
        ("/home/chen/.cache/deepep-sjlgpt/bin/python",),
        "BLOCKED_DEPENDENCY",
    ),
    "nccl-mpi-header": (
        "nccl-ep",
        "file",
        ("/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h",),
        "BLOCKED_DEPENDENCY",
    ),
    "nccl-mpi-library": (
        "nccl-ep",
        "file",
        ("/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so",),
        "BLOCKED_DEPENDENCY",
    ),
    "uccl-ibverbs-header": (
        "uccl-ep",
        "file",
        ("/usr/include/infiniband/verbs.h",),
        "BLOCKED_DEPENDENCY",
    ),
    "uccl-ibverbs-library": (
        "uccl-ep",
        "file",
        ("/usr/lib/x86_64-linux-gnu/libibverbs.so",),
        "BLOCKED_DEPENDENCY",
    ),
    "uccl-libnl-header": (
        "uccl-ep",
        "file",
        ("/usr/include/libnl3/netlink/netlink.h",),
        "BLOCKED_DEPENDENCY",
    ),
    "uccl-libnl-library": (
        "uccl-ep",
        "file",
        ("/usr/lib/x86_64-linux-gnu/libnl-3.so",),
        "BLOCKED_DEPENDENCY",
    ),
    "uccl-libnl-route-library": (
        "uccl-ep",
        "file",
        ("/usr/lib/x86_64-linux-gnu/libnl-route-3.so",),
        "BLOCKED_DEPENDENCY",
    ),
    "uccl-nanobind": (
        "uccl-ep",
        "file",
        (
            "/home/chen/.cache/deepep-sjlgpt/lib/python3.11/"
            "site-packages/nanobind/__init__.py",
        ),
        "BLOCKED_DEPENDENCY_NANOBIND",
    ),
    "uccl-numa-header": (
        "uccl-ep",
        "file",
        ("/usr/include/numa.h",),
        "BLOCKED_DEPENDENCY",
    ),
    "uccl-numa-library": (
        "uccl-ep",
        "file",
        ("/usr/lib/x86_64-linux-gnu/libnuma.so",),
        "BLOCKED_DEPENDENCY",
    ),
    "uccl-torch-headers": (
        "uccl-ep",
        "directory",
        (
            "/home/chen/.cache/deepep-sjlgpt/lib/python3.11/"
            "site-packages/torch/include",
        ),
        "BLOCKED_DEPENDENCY",
    ),
    "uccl-torch-package": (
        "uccl-ep",
        "directory",
        (
            "/home/chen/.cache/deepep-sjlgpt/lib/python3.11/"
            "site-packages/torch",
        ),
        "BLOCKED_DEPENDENCY",
    ),
}
_EXPECTED_GPU_MAPPING = (
    (0, "GPU-34d06e04-4c63-02ac-12c5-cb4fbe14e78f"),
    (1, "GPU-7c2d67e2-baa9-e52c-b14b-d323d689db49"),
    (2, "GPU-43dd5b59-3ad6-d72e-4b30-97f70795d0d9"),
    (3, "GPU-b2d439c0-5ad1-b1f4-b639-28da8cd5bf6a"),
    (4, "GPU-60b40059-df23-0d8d-8583-9e3f8d746874"),
    (5, "GPU-30413ecd-d9b3-ae05-d2d7-aa5bbe208cc2"),
    (6, "GPU-100c4b75-c009-44ff-0a6d-79097e0ce112"),
    (7, "GPU-b018d7a0-2075-e96a-33f9-d970485abb1f"),
)
_EXPECTED_GPU_STATIC_FACTS = tuple(
    (index, uuid, "NVIDIA L20X", 143771, "Default", "Disabled")
    for index, uuid in _EXPECTED_GPU_MAPPING
)
_EXPECTED_NICS = (
    ("mlx5_bond_0", "NIC0", 1, "4: ACTIVE", "5: LinkUp", "200 Gb/sec (2X NDR)", "Ethernet", "0000:0d:00.0", 0, "0x15b3", "0x1021"),
    ("mlx5_bond_1", "NIC1", 1, "4: ACTIVE", "5: LinkUp", "200 Gb/sec (2X NDR)", "Ethernet", "0000:5a:00.0", 0, "0x15b3", "0x1021"),
    ("mlx5_bond_2", "NIC2", 1, "4: ACTIVE", "5: LinkUp", "200 Gb/sec (2X NDR)", "Ethernet", "0000:60:00.0", 0, "0x15b3", "0x1021"),
    ("mlx5_bond_3", "NIC3", 1, "4: ACTIVE", "5: LinkUp", "200 Gb/sec (2X NDR)", "Ethernet", "0000:67:00.0", 0, "0x15b3", "0x1021"),
    ("mlx5_bond_4", "NIC4", 1, "4: ACTIVE", "5: LinkUp", "200 Gb/sec (2X NDR)", "Ethernet", "0000:99:00.0", 1, "0x15b3", "0x1021"),
    ("mlx5_bond_5", "NIC5", 1, "4: ACTIVE", "5: LinkUp", "200 Gb/sec (2X NDR)", "Ethernet", "0000:b9:00.0", 1, "0x15b3", "0x1021"),
    ("mlx5_bond_6", "NIC6", 1, "4: ACTIVE", "5: LinkUp", "200 Gb/sec (2X NDR)", "Ethernet", "0000:c9:00.0", 1, "0x15b3", "0x1021"),
    ("mlx5_bond_7", "NIC7", 1, "4: ACTIVE", "5: LinkUp", "200 Gb/sec (2X NDR)", "Ethernet", "0000:da:00.0", 1, "0x15b3", "0x1021"),
)
_GPU_QUERY_GUARD_PAYLOAD = b"#!/bin/sh\nexit 97\n"
_GPU_QUERY_GUARD_PATH = "{stage_root}/uccl/toolshim/nvidia-smi"
_TERMINAL_STATUSES = {
    "STATIC_CHECKS_PASSED_NO_BUILD_AUTHORITY",
    "WAITING_GPU",
    "BLOCKED_DEPENDENCY_NANOBIND",
    "BLOCKED_DEPENDENCY",
    "BLOCKED_DISK",
    "BLOCKED_GPU_MAPPING",
    "BLOCKED_TOPOLOGY",
    "BLOCKED_NETWORK",
    "BLOCKED_RUNTIME_PROBE",
}


class PreflightError(ValueError):
    """The manifest or observed local state is unsafe or ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PreflightError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise PreflightError(f"non-finite JSON constant is forbidden: {value}")


def _validate_json_tree(value: object, name: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        _require(math.isfinite(value), f"{name} contains a non-finite number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{name}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            _require(type(key) is str, f"{name} contains a non-string key")
            _validate_json_tree(item, f"{name}.{key}")
        return
    raise PreflightError(f"{name} contains a non-JSON value")


def _exact_fields(value: object, fields: set[str], name: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{name} must be an object")
    assert isinstance(value, dict)
    _require(set(value) == fields, f"{name} fields are incomplete or unknown")
    return value


def _nonempty_string(value: object, name: str) -> str:
    _require(type(value) is str and bool(value.strip()), f"{name} must be nonempty")
    assert isinstance(value, str)
    _require("\x00" not in value and "\n" not in value and "\r" not in value,
             f"{name} contains a control character")
    return value


def _slug(value: object, name: str) -> str:
    text = _nonempty_string(value, name)
    _require(bool(_SLUG.fullmatch(text)), f"{name} is not a safe slug")
    return text


def _sha256_text(value: object, name: str) -> str:
    text = _nonempty_string(value, name)
    _require(bool(_SHA256.fullmatch(text)), f"{name} is not a lowercase SHA256")
    return text


def _git_object(value: object, name: str) -> str:
    text = _nonempty_string(value, name)
    _require(bool(_GIT_OBJECT.fullmatch(text)), f"{name} is not a full Git object ID")
    return text


def _exact_int(value: object, name: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum,
             f"{name} must be an exact integer >= {minimum}")
    assert isinstance(value, int)
    return value


def _absolute_path(value: object, name: str) -> Path:
    text = _nonempty_string(value, name)
    path = Path(text)
    _require(path.is_absolute(), f"{name} must be absolute")
    _require(str(path) == os.path.abspath(text), f"{name} must be normalized")
    return path


def _safe_relative(value: object, name: str) -> str:
    text = _nonempty_string(value, name)
    path = Path(text)
    _require(not path.is_absolute(), f"{name} must be relative")
    _require(text == path.as_posix(), f"{name} must use normalized POSIX separators")
    _require(path.parts and all(part not in {"", ".", ".."} for part in path.parts),
             f"{name} traverses or is ambiguous")
    return text


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PreflightError(f"value is not strict JSON: {error}") from error


def _read_regular_no_follow(
    path: Path,
    maximum_bytes: int,
    *,
    owner_uid: int | None = None,
    single_link: bool = False,
) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        if owner_uid is not None:
            _require(before.st_uid == owner_uid, f"foreign-owned file: {path}")
        if single_link:
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
        raise PreflightError(f"cannot read {path}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _assert_no_symlink_components(path: Path, name: str) -> os.stat_result:
    path = Path(os.path.abspath(path))
    cursor = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            cursor /= component
            metadata = os.lstat(cursor)
            _require(not stat.S_ISLNK(metadata.st_mode),
                     f"{name} contains a symlink: {cursor}")
        return os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise PreflightError(f"cannot inspect {name} {path}: {error}") from error


def load_manifest(path: Path) -> tuple[dict[str, Any], bytes, str, str]:
    """Load one bounded, owner-controlled, strict JSON manifest."""

    path = Path(os.path.abspath(path))
    metadata = _assert_no_symlink_components(path, "manifest")
    _require(metadata.st_mode & 0o022 == 0,
             f"manifest is group/other writable: {path}")
    raw = _read_regular_no_follow(
        path,
        MAX_JSON_BYTES,
        owner_uid=os.geteuid(),
        single_link=True,
    )
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise PreflightError(f"invalid strict JSON manifest {path}: {error}") from error
    _require(type(value) is dict, "manifest root must be an object")
    assert isinstance(value, dict)
    validate_manifest(value)
    return (
        value,
        raw,
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(_canonical_bytes(value)).hexdigest(),
    )


def _validate_gpu_mapping(value: object) -> None:
    _require(type(value) is list and len(value) == 8,
             "resources.expected_gpu_mapping must contain eight GPUs")
    assert isinstance(value, list)
    observed: list[tuple[int, str]] = []
    for index, item in enumerate(value):
        row = _exact_fields(item, {"index", "uuid"}, f"expected_gpu_mapping[{index}]")
        gpu_index = _exact_int(row["index"], f"expected_gpu_mapping[{index}].index")
        uuid = _nonempty_string(row["uuid"], f"expected_gpu_mapping[{index}].uuid")
        _require(bool(_GPU_UUID.fullmatch(uuid)), f"invalid GPU UUID at index {index}")
        observed.append((gpu_index, uuid))
    _require(tuple(observed) == _EXPECTED_GPU_MAPPING,
             "resources.expected_gpu_mapping differs from the frozen host mapping")


def _validate_tools(value: object) -> None:
    _require(type(value) is list, "tools must be a list")
    assert isinstance(value, list)
    ids: list[str] = []
    for index, item in enumerate(value):
        row = _exact_fields(
            item,
            {"id", "path", "realpath", "sha256", "owner_uid"},
            f"tools[{index}]",
        )
        tool_id = _slug(row["id"], f"tools[{index}].id")
        ids.append(tool_id)
        path = _absolute_path(row["path"], f"tools[{index}].path")
        realpath = _absolute_path(row["realpath"], f"tools[{index}].realpath")
        digest = _sha256_text(row["sha256"], f"tools[{index}].sha256")
        owner_uid = _exact_int(row["owner_uid"], f"tools[{index}].owner_uid")
        _require(
            (str(path), str(realpath), digest, owner_uid)
            == _PINNED_TOOL_IDENTITIES.get(tool_id),
            f"tools[{index}] differs from the pinned {tool_id} identity",
        )
        _require(realpath != Path("/"), f"tools[{index}].realpath is unsafe")
    _require(ids == sorted(_PINNED_TOOL_PATHS),
             "tools must contain the exact sorted pinned tool set")


def _validate_dependency(value: object, index: int, competitor_ids: set[str]) -> str:
    name = f"dependencies[{index}]"
    row = _exact_fields(
        value,
        {
            "id",
            "competitor_id",
            "kind",
            "required",
            "visibility_paths",
            "missing_status",
        },
        name,
    )
    dependency_id = _slug(row["id"], f"{name}.id")
    competitor_id = _slug(row["competitor_id"], f"{name}.competitor_id")
    _require(competitor_id in competitor_ids | {"common"},
             f"{name}.competitor_id is unknown")
    _require(row["kind"] in {"file", "directory", "executable"},
             f"{name}.kind is invalid")
    _require(type(row["required"]) is bool, f"{name}.required must be bool")
    paths = row["visibility_paths"]
    _require(type(paths) is list and bool(paths), f"{name}.visibility_paths is empty")
    assert isinstance(paths, list)
    normalized = [str(_absolute_path(path, f"{name}.visibility_paths[{i}]"))
                  for i, path in enumerate(paths)]
    _require(normalized == sorted(set(normalized)),
             f"{name}.visibility_paths must be sorted and unique")
    _require(row["missing_status"] in {
        "BLOCKED_DEPENDENCY", "BLOCKED_DEPENDENCY_NANOBIND"
    }, f"{name}.missing_status is invalid")
    if dependency_id == "uccl-nanobind":
        _require(row["missing_status"] == "BLOCKED_DEPENDENCY_NANOBIND",
                 "uccl-nanobind must retain its distinct blocker")
    expected = _EXPECTED_DEPENDENCIES.get(dependency_id)
    _require(expected is not None, f"{name}.id is not a frozen dependency")
    assert expected is not None
    _require(
        (
            competitor_id,
            row["kind"],
            tuple(normalized),
            row["missing_status"],
        )
        == expected,
        f"{name} differs from its frozen dependency contract",
    )
    _require(row["required"] is True, f"{name}.required must remain true")
    return dependency_id


def _validate_build_stage(value: object, name: str) -> str:
    stage = _exact_fields(
        value,
        {
            "id",
            "cwd",
            "argv",
            "env",
            "depends_on",
            "expected_outputs",
            "gpu_query_guard",
        },
        name,
    )
    stage_id = _slug(stage["id"], f"{name}.id")
    _require(stage_id in _EXPECTED_BUILD_ARGV, f"{name}.id is not a frozen build stage")
    cwd = _nonempty_string(stage["cwd"], f"{name}.cwd")
    _require(cwd.startswith("{stage_root}/"), f"{name}.cwd must remain below stage_root")
    _require(cwd == _EXPECTED_BUILD_CWD[stage_id], f"{name}.cwd changed")
    argv = stage["argv"]
    _require(type(argv) is list and len(argv) >= 2, f"{name}.argv is invalid")
    assert isinstance(argv, list)
    for index, argument in enumerate(argv):
        _nonempty_string(argument, f"{name}.argv[{index}]")
    _require(argv[0] == _PINNED_TOOL_PATHS["make"],
             f"{name}.argv must use pinned make")
    _require(tuple(argv) == _EXPECTED_BUILD_ARGV[stage_id],
             f"{name}.argv changed from the frozen future build command")
    env = stage["env"]
    _require(type(env) is dict, f"{name}.env must be an object")
    assert isinstance(env, dict)
    for key, item in env.items():
        _require(type(key) is str and bool(_ENVIRONMENT_NAME.fullmatch(key)),
                 f"{name}.env has an unsafe key")
        _nonempty_string(item, f"{name}.env.{key}") if key != "CUDA_VISIBLE_DEVICES" else _require(
            item == "", f"{name}.env.CUDA_VISIBLE_DEVICES must be empty"
        )
    _require(
        env == _EXPECTED_BUILD_ENV[stage_id],
        f"{name}.env changed from the frozen future build environment",
    )
    for field in ("depends_on", "expected_outputs"):
        values = stage[field]
        _require(type(values) is list, f"{name}.{field} must be a list")
        assert isinstance(values, list)
        checked: list[str] = []
        for index, item in enumerate(values):
            if field == "depends_on":
                checked.append(_slug(item, f"{name}.{field}[{index}]"))
            else:
                path = _nonempty_string(item, f"{name}.{field}[{index}]")
                _require(path.startswith("{stage_root}/"),
                         f"{name}.{field}[{index}] escapes stage_root")
                _safe_relative(
                    path.removeprefix("{stage_root}/"),
                    f"{name}.{field}[{index}]",
                )
                checked.append(path)
        _require(checked == sorted(set(checked)),
                 f"{name}.{field} must be sorted and unique")
        if field == "depends_on":
            _require(tuple(checked) == _EXPECTED_BUILD_DEPENDENCIES[stage_id],
                     f"{name}.depends_on changed")
        else:
            _require(
                tuple(checked) == _EXPECTED_BUILD_OUTPUTS[stage_id],
                f"{name}.expected_outputs changed",
            )
    guard = stage["gpu_query_guard"]
    if guard is not None:
        guard = _exact_fields(
            guard,
            {
                "path",
                "content_hex",
                "sha256",
                "mode",
                "expected_exit_code",
            },
            f"{name}.gpu_query_guard",
        )
        guard_path = _nonempty_string(guard["path"], f"{name}.gpu_query_guard.path")
        _require(guard_path.startswith("{stage_root}/"),
                 f"{name}.gpu_query_guard.path escapes stage_root")
        content_hex = _nonempty_string(
            guard["content_hex"], f"{name}.gpu_query_guard.content_hex"
        )
        try:
            content = bytes.fromhex(content_hex)
        except ValueError as error:
            raise PreflightError(f"{name}.gpu_query_guard.content_hex is invalid") from error
        _require(content.hex() == content_hex,
                 f"{name}.gpu_query_guard.content_hex is not canonical lowercase hex")
        expected_sha256 = _sha256_text(
            guard["sha256"], f"{name}.gpu_query_guard.sha256"
        )
        _require(hashlib.sha256(content).hexdigest() == expected_sha256,
                 f"{name}.gpu_query_guard content hash differs")
        _require(
            guard_path == _GPU_QUERY_GUARD_PATH
            and content == _GPU_QUERY_GUARD_PAYLOAD,
            f"{name}.gpu_query_guard differs from the frozen deny shim",
        )
        _require(guard["mode"] == 0o555,
                 f"{name}.gpu_query_guard.mode must be 0555")
        _require(guard["expected_exit_code"] == 97,
                 f"{name}.gpu_query_guard.expected_exit_code must be 97")
    return stage_id


def _validate_competitors(value: object) -> set[str]:
    _require(type(value) is list, "competitors must be a list")
    assert isinstance(value, list)
    ids: list[str] = []
    stage_ids: list[str] = []
    for index, item in enumerate(value):
        name = f"competitors[{index}]"
        competitor = _exact_fields(
            item,
            {
                "id",
                "checkout",
                "commit",
                "source_tree",
                "subtree",
                "subtree_tree",
                "subtree_content_sha256",
                "build_source_policy",
                "source_materialization",
                "future_build_stages",
            },
            name,
        )
        competitor_id = _slug(competitor["id"], f"{name}.id")
        ids.append(competitor_id)
        checkout = _absolute_path(competitor["checkout"], f"{name}.checkout")
        commit = _git_object(competitor["commit"], f"{name}.commit")
        subtree = _safe_relative(competitor["subtree"], f"{name}.subtree")
        source_tree = _git_object(competitor["source_tree"], f"{name}.source_tree")
        subtree_tree = _git_object(competitor["subtree_tree"], f"{name}.subtree_tree")
        subtree_content_sha256 = _sha256_text(
            competitor["subtree_content_sha256"],
            f"{name}.subtree_content_sha256",
        )
        _require(
            _PINNED_COMPETITORS.get(competitor_id)
            == (
                str(checkout),
                commit,
                source_tree,
                subtree,
                subtree_tree,
                subtree_content_sha256,
            ),
            f"{name} differs from its frozen source identity",
        )
        _require(
            competitor["build_source_policy"]
            == "PINNED_GIT_OBJECTS_TO_FRESH_STAGE",
                 f"{name}.build_source_policy is unsafe")
        materialization = _exact_fields(
            competitor["source_materialization"],
            {"method", "paths", "destination", "expected_content_sha256"},
            f"{name}.source_materialization",
        )
        _require(
            materialization["method"] == "PINNED_GIT_OBJECTS_ONLY",
            f"{name}.source_materialization.method is unsafe",
        )
        paths = materialization["paths"]
        _require(type(paths) is list and bool(paths),
                 f"{name}.source_materialization.paths is empty")
        assert isinstance(paths, list)
        normalized_paths: list[str] = []
        for path_index, path in enumerate(paths):
            value = _nonempty_string(
                path, f"{name}.source_materialization.paths[{path_index}]"
            )
            if value != ".":
                value = _safe_relative(
                    value, f"{name}.source_materialization.paths[{path_index}]"
                )
            normalized_paths.append(value)
        _require(
            normalized_paths == sorted(set(normalized_paths)),
            f"{name}.source_materialization.paths must be sorted and unique",
        )
        destination = _nonempty_string(
            materialization["destination"],
            f"{name}.source_materialization.destination",
        )
        _require(destination.startswith("{stage_root}/"),
                 f"{name}.source_materialization.destination escapes stage_root")
        _safe_relative(
            destination.removeprefix("{stage_root}/"),
            f"{name}.source_materialization.destination",
        )
        materialized_sha = _sha256_text(
            materialization["expected_content_sha256"],
            f"{name}.source_materialization.expected_content_sha256",
        )
        _require(
            (tuple(normalized_paths), destination, materialized_sha)
            == _EXPECTED_SOURCE_MATERIALIZATIONS[competitor_id],
            f"{name}.source_materialization differs from the frozen closure",
        )
        stages = competitor["future_build_stages"]
        _require(type(stages) is list and bool(stages),
                 f"{name}.future_build_stages must be nonempty")
        assert isinstance(stages, list)
        local_ids = [
            _validate_build_stage(stage, f"{name}.future_build_stages[{stage_index}]")
            for stage_index, stage in enumerate(stages)
        ]
        _require(local_ids == sorted(set(local_ids)),
                 f"{name}.future_build_stages must be sorted and unique")
        expected_stage_ids = {
            "nccl-ep": ["nccl-ep-core", "nccl-ep-correctness"],
            "uccl-ep": ["uccl-ep-extension"],
        }[competitor_id]
        _require(local_ids == expected_stage_ids,
                 f"{name}.future_build_stages differs from the frozen stage set")
        guards = [stage["gpu_query_guard"] for stage in stages]
        if competitor_id == "uccl-ep":
            _require(len(stages) == 1 and guards[0] is not None,
                     "UCCL future build must freeze a deny-by-default nvidia-smi shim")
            guard_path = str(guards[0]["path"])
            path_prefix = str(stages[0]["env"]["PATH"]).split(":", 1)[0]
            _require(str(Path(guard_path).parent) == path_prefix,
                     "UCCL nvidia-smi shim directory must lead future build PATH")
        else:
            _require(all(guard is None for guard in guards),
                     f"{name} declares an unexpected GPU query guard")
        stage_ids.extend(local_ids)
    _require(ids == sorted(_PINNED_COMPETITORS),
             "competitors must contain the exact sorted frozen competitor set")
    _require(len(stage_ids) == len(set(stage_ids)), "future build stage IDs collide")
    return set(ids)


def _validate_network(value: object) -> None:
    network = _exact_fields(
        value,
        {"nics", "ibv_expected_devices", "ibv_devinfo_execution_status"},
        "network",
    )
    _require(network["ibv_devinfo_execution_status"] == "NOT_RUN",
             "ibv_devinfo_execution_status must remain NOT_RUN")
    expected_devices = network["ibv_expected_devices"]
    _require(type(expected_devices) is list, "ibv_expected_devices must be a list")
    _require(expected_devices == sorted({row[0] for row in _EXPECTED_NICS}),
             "ibv_expected_devices differs from the frozen bond set")
    nics = network["nics"]
    _require(type(nics) is list and len(nics) == 8,
             "network.nics must contain eight entries")
    assert isinstance(nics, list)
    identities: list[tuple[object, ...]] = []
    for index, item in enumerate(nics):
        name = f"network.nics[{index}]"
        row = _exact_fields(
            item,
            {
                "name",
                "topology_label",
                "port",
                "state",
                "physical_state",
                "rate",
                "link_layer",
                "pci_bdf",
                "numa_node",
                "vendor",
                "device",
            },
            name,
        )
        nic_name = _nonempty_string(row["name"], f"{name}.name")
        topology_label = _nonempty_string(
            row["topology_label"], f"{name}.topology_label"
        )
        port = _exact_int(row["port"], f"{name}.port", minimum=1)
        _require(port == 1, f"{name}.port must be 1")
        pci_bdf = _nonempty_string(row["pci_bdf"], f"{name}.pci_bdf")
        _require(bool(_PCI_BDF.fullmatch(pci_bdf)), f"{name}.pci_bdf is invalid")
        numa_node = _exact_int(row["numa_node"], f"{name}.numa_node")
        text_fields: dict[str, str] = {}
        for field in (
            "state", "physical_state", "rate", "link_layer", "vendor", "device"
        ):
            text_fields[field] = _nonempty_string(row[field], f"{name}.{field}")
        identities.append((
            nic_name,
            topology_label,
            port,
            text_fields["state"],
            text_fields["physical_state"],
            text_fields["rate"],
            text_fields["link_layer"],
            pci_bdf,
            numa_node,
            text_fields["vendor"],
            text_fields["device"],
        ))
    _require(tuple(identities) == _EXPECTED_NICS,
             "network NIC contract differs from the frozen host")


def _validate_probes(value: object, home_path: str) -> None:
    probes = _exact_fields(
        value,
        {
            "du_argv",
            "gpu_mapping_argv",
            "gpu_process_argv",
            "topology_argv",
            "ibv_devices_argv",
        },
        "probes",
    )
    expected = {
        "du_argv": [
            _PINNED_TOOL_PATHS["du"], "-sx", "--block-size=1", "--", home_path
        ],
        "gpu_mapping_argv": [
            _PINNED_TOOL_PATHS["nvidia_smi"],
            "--query-gpu=index,uuid,name,memory.total,compute_mode,mig.mode.current",
            "--format=csv,noheader,nounits",
        ],
        "gpu_process_argv": [
            _PINNED_TOOL_PATHS["nvidia_smi"],
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        "topology_argv": [_PINNED_TOOL_PATHS["nvidia_smi"], "topo", "-m"],
        "ibv_devices_argv": [_PINNED_TOOL_PATHS["ibv_devices"]],
    }
    _require(probes == expected, "probe argv differs from the frozen check-only commands")


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the exact semantic schema without touching the filesystem."""

    _validate_json_tree(manifest, "manifest")
    _exact_fields(
        manifest,
        {
            "schema_version",
            "preflight_id",
            "mode",
            "execution_contract",
            "resources",
            "tools",
            "probes",
            "network",
            "competitors",
            "dependencies",
        },
        "manifest",
    )
    _require(manifest["schema_version"] == SCHEMA_VERSION,
             "unsupported schema_version")
    _slug(manifest["preflight_id"], "preflight_id")
    _require(manifest["mode"] == "check_only", "mode must be check_only")
    contract = _exact_fields(
        manifest["execution_contract"],
        {
            "gpu_launch_policy",
            "build_execution_status",
            "future_build_executor_status",
            "performance_claim_allowed",
            "checkout_write_policy",
            "stage_root_policy",
            "source_materialization_policy",
        },
        "execution_contract",
    )
    _require(contract == {
        "gpu_launch_policy": "FORBIDDEN",
        "build_execution_status": "NOT_RUN",
        "future_build_executor_status": "NOT_IMPLEMENTED",
        "performance_claim_allowed": False,
        "checkout_write_policy": "FORBIDDEN",
        "stage_root_policy": "FRESH_EMPTY_OWNER_0700_NO_SYMLINKS_NO_REUSE",
        "source_materialization_policy": "PINNED_GIT_OBJECTS_ONLY",
    }, "execution_contract must remain fail-closed and check-only")

    resources = _exact_fields(
        manifest["resources"],
        {
            "home_path",
            "home_hard_limit_bytes",
            "artifact_budget_bytes",
            "emergency_reserve_bytes",
            "required_gpu_count",
            "expected_gpu_mapping",
        },
        "resources",
    )
    _require(str(_absolute_path(resources["home_path"], "resources.home_path"))
             == "/home/chen", "resources.home_path must remain /home/chen")
    _require(resources["home_hard_limit_bytes"] == HOME_HARD_LIMIT_BYTES,
             "home hard limit must remain 300 GB decimal")
    _require(resources["artifact_budget_bytes"] == ARTIFACT_BUDGET_BYTES,
             "artifact budget must remain 8 GB decimal")
    _require(resources["emergency_reserve_bytes"] == EMERGENCY_RESERVE_BYTES,
             "emergency reserve must remain 1 GB decimal")
    _require(resources["required_gpu_count"] == 8,
             "required_gpu_count must remain eight")
    _validate_gpu_mapping(resources["expected_gpu_mapping"])
    _validate_tools(manifest["tools"])
    competitor_ids = _validate_competitors(manifest["competitors"])
    dependencies = manifest["dependencies"]
    _require(type(dependencies) is list and bool(dependencies),
             "dependencies must be a nonempty list")
    assert isinstance(dependencies, list)
    dependency_ids = [
        _validate_dependency(item, index, competitor_ids)
        for index, item in enumerate(dependencies)
    ]
    _require(
        dependency_ids == sorted(_EXPECTED_DEPENDENCIES),
        "dependencies must contain the exact sorted frozen dependency set",
    )
    _validate_network(manifest["network"])
    _validate_probes(manifest["probes"], resources["home_path"])


def _capture(argv: Sequence[str], timeout: int = COMMAND_TIMEOUT_SECONDS) -> dict[str, Any]:
    command = tuple(argv)
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    started_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            cwd="/",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        returncode: int | None = completed.returncode
        stdout = completed.stdout.decode("utf-8", "replace")
        stderr = completed.stderr.decode("utf-8", "replace")
        error = None
    except (OSError, subprocess.TimeoutExpired) as caught:
        returncode = None
        stdout = ""
        stderr = ""
        error = repr(caught)
        if isinstance(caught, subprocess.TimeoutExpired):
            if isinstance(caught.stdout, bytes):
                stdout = caught.stdout.decode("utf-8", "replace")
            if isinstance(caught.stderr, bytes):
                stderr = caught.stderr.decode("utf-8", "replace")
    return {
        "argv": list(command),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "duration_ns": time.monotonic_ns() - started_ns,
    }


def _inspect_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(tool["path"]))
    try:
        requested = os.lstat(path)
        realpath = path.resolve(strict=True)
        metadata = os.stat(realpath, follow_symlinks=False)
    except OSError as error:
        raise PreflightError(f"cannot inspect pinned tool {tool['id']}: {error}") from error
    _require(str(realpath) == tool["realpath"],
             f"pinned tool {tool['id']} escaped or changed realpath")
    _require(stat.S_ISREG(metadata.st_mode), f"pinned tool {tool['id']} is not regular")
    _require(metadata.st_mode & 0o111 != 0, f"pinned tool {tool['id']} is not executable")
    _require(metadata.st_uid == tool["owner_uid"],
             f"pinned tool {tool['id']} owner changed")
    _require(metadata.st_mode & 0o022 == 0,
             f"pinned tool {tool['id']} is group/other writable")
    payload = _read_regular_no_follow(realpath, MAX_TOOL_BYTES)
    digest = hashlib.sha256(payload).hexdigest()
    _require(digest == tool["sha256"], f"pinned tool {tool['id']} SHA256 changed")
    return {
        "id": tool["id"],
        "path": str(path),
        "path_is_symlink": stat.S_ISLNK(requested.st_mode),
        "realpath": str(realpath),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "bytes": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
        "owner_uid": metadata.st_uid,
        "sha256": digest,
        "status": "PASSED",
    }


def _inspect_preflight_program() -> dict[str, Any]:
    path = Path(os.path.abspath(__file__))
    _require(path == PROGRAM_PATH,
             f"preflight program path must be {PROGRAM_PATH}")
    metadata = _assert_no_symlink_components(path, "preflight program")
    _require(stat.S_ISREG(metadata.st_mode), "preflight program is not regular")
    _require(metadata.st_uid == os.geteuid(), "preflight program is foreign-owned")
    _require(metadata.st_mode & 0o022 == 0,
             "preflight program is group/other writable")
    payload = _read_regular_no_follow(
        path,
        MAX_TOOL_BYTES,
        owner_uid=os.geteuid(),
    )
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "bytes": len(payload),
        "mode": stat.S_IMODE(metadata.st_mode),
        "owner_uid": metadata.st_uid,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": "PASSED",
    }


def _inspect_proc_executable(expected_realpath: str) -> dict[str, Any]:
    try:
        raw_target = os.readlink("/proc/self/exe")
        realpath = Path(raw_target).resolve(strict=True)
        metadata = os.stat(realpath, follow_symlinks=False)
    except OSError as error:
        raise PreflightError(f"cannot inspect /proc/self/exe: {error}") from error
    _require(str(realpath) == expected_realpath,
             "/proc/self/exe differs from the pinned preflight interpreter")
    return {
        "readlink": raw_target,
        "realpath": str(realpath),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "owner_uid": metadata.st_uid,
        "mode": stat.S_IMODE(metadata.st_mode),
    }


def _inspect_interpreter(*, require_isolated_startup: bool) -> dict[str, Any]:
    requested = os.path.abspath(sys.executable)
    if require_isolated_startup:
        expected_path, expected_realpath, expected_sha256, expected_owner = (
            _PINNED_PREFLIGHT_INTERPRETER
        )
        _require(requested == expected_path,
                 f"preflight interpreter must be {expected_path}")
        result = _inspect_tool({
            "id": "preflight-python",
            "path": expected_path,
            "realpath": expected_realpath,
            "sha256": expected_sha256,
            "owner_uid": expected_owner,
        })
        result["proc_self_exe"] = _inspect_proc_executable(expected_realpath)
    else:
        try:
            realpath = Path(requested).resolve(strict=True)
            metadata = os.stat(realpath, follow_symlinks=False)
        except OSError as error:
            raise PreflightError(
                f"cannot resolve test interpreter {requested}: {error}"
            ) from error
        payload = _read_regular_no_follow(realpath, MAX_TOOL_BYTES)
        result = {
            "id": "test-python",
            "path": requested,
            "realpath": str(realpath),
            "owner_uid": metadata.st_uid,
            "mode": stat.S_IMODE(metadata.st_mode),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "status": "TEST_ONLY",
        }
    result["sys_executable"] = requested
    result["python_version"] = sys.version
    startup_flags = {
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
        "ignore_environment": sys.flags.ignore_environment,
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": sys.flags.safe_path,
    }
    expected_flags = {
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "safe_path": True,
    }
    if require_isolated_startup:
        _require(
            startup_flags == expected_flags,
            "production preflight requires pinned Python flags -I -S -B",
        )
    result["startup_flags"] = startup_flags
    result["startup_contract_status"] = (
        "PINNED_ISOLATED_NO_SITE"
        if require_isolated_startup and startup_flags == expected_flags
        else "TEST_NONATTESTED"
    )
    return result


def _read_proc_nul_fields(path: Path, name: str) -> tuple[bytes, list[str]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PreflightError(f"cannot read {name}: {error}") from error
    _require(0 < len(raw) <= (1 << 20), f"{name} has an invalid size")
    _require(raw.endswith(b"\0"), f"{name} is not NUL terminated")
    try:
        fields = [item.decode("utf-8", "strict") for item in raw[:-1].split(b"\0")]
    except UnicodeDecodeError as error:
        raise PreflightError(f"{name} is not strict UTF-8") from error
    _require(all("\0" not in item for item in fields), f"{name} contains NUL")
    return raw, fields


def _inspect_process_launch(
    *, manifest_path: Path, output_path: Path
) -> dict[str, Any]:
    manifest_path = Path(os.path.abspath(manifest_path))
    output_path = Path(os.path.abspath(output_path))
    _require(manifest_path == MANIFEST_PATH,
             f"production manifest path must be {MANIFEST_PATH}")
    command_raw, command = _read_proc_nul_fields(
        Path("/proc/self/cmdline"), "process cmdline"
    )
    expected_command = [
        _PINNED_PREFLIGHT_INTERPRETER[0],
        "-I",
        "-S",
        "-B",
        str(PROGRAM_PATH),
        "--manifest",
        str(MANIFEST_PATH),
        "--output",
        str(output_path),
    ]
    _require(command == expected_command,
             "process cmdline differs from the frozen production argv")

    environment_raw, environment_fields = _read_proc_nul_fields(
        Path("/proc/self/environ"), "process environment"
    )
    environment: dict[str, str] = {}
    for field in environment_fields:
        _require("=" in field, "process environment contains an invalid entry")
        key, value = field.split("=", 1)
        _require(key not in environment, f"duplicate process environment key: {key}")
        environment[key] = value
    _require(
        environment == _EXPECTED_PROCESS_ENVIRONMENT,
        "process environment differs from the frozen empty-environment allowlist",
    )
    _require(
        dict(os.environ) == _EXPECTED_PROCESS_ENVIRONMENT,
        "Python environment view differs from the frozen process environment",
    )
    return {
        "argv": command,
        "cmdline_sha256": hashlib.sha256(command_raw).hexdigest(),
        "environment": environment,
        "environ_sha256": hashlib.sha256(environment_raw).hexdigest(),
        "status": "PINNED_ARGV_EMPTY_ENV",
    }


def _inspect_runtime_closure() -> dict[str, Any]:
    sources: dict[Path, set[str]] = {}

    def add_path(raw_path: str, source: str) -> None:
        if not raw_path.startswith("/"):
            return
        path = Path(raw_path.removesuffix(" (deleted)"))
        _require(not raw_path.endswith(" (deleted)"),
                 f"runtime closure contains a deleted file: {raw_path}")
        sources.setdefault(path, set()).add(source)

    for module_name, module in sorted(sys.modules.items()):
        module_path = getattr(module, "__file__", None)
        if isinstance(module_path, str):
            add_path(os.path.abspath(module_path), f"module:{module_name}")
        cached_path = getattr(module, "__cached__", None)
        if isinstance(cached_path, str) and Path(cached_path).exists():
            add_path(os.path.abspath(cached_path), f"cached:{module_name}")

    try:
        maps_lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PreflightError(f"cannot read process maps: {error}") from error
    for line in maps_lines:
        fields = line.split(maxsplit=5)
        if len(fields) == 6:
            add_path(fields[5], "process-map")

    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256(b"rail-balance-runtime-closure-v1\0")
    for requested_path in sorted(sources, key=str):
        try:
            realpath = requested_path.resolve(strict=True)
            metadata = os.stat(realpath, follow_symlinks=False)
        except OSError as error:
            raise PreflightError(
                f"cannot resolve runtime closure file {requested_path}: {error}"
            ) from error
        _require(stat.S_ISREG(metadata.st_mode),
                 f"runtime closure path is not regular: {realpath}")
        is_program = realpath == PROGRAM_PATH
        expected_owner = os.geteuid() if is_program else 0
        _require(metadata.st_uid == expected_owner,
                 f"runtime closure file has unexpected owner: {realpath}")
        _require(metadata.st_mode & 0o022 == 0,
                 f"runtime closure file is group/other writable: {realpath}")
        payload = _read_regular_no_follow(
            realpath,
            MAX_TRACKED_FILE_BYTES,
            owner_uid=expected_owner,
        )
        file_sha256 = hashlib.sha256(payload).hexdigest()
        path_bytes = str(realpath).encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "big") + path_bytes)
        digest.update(len(payload).to_bytes(8, "big") + payload)
        rows.append({
            "requested_path": str(requested_path),
            "realpath": str(realpath),
            "sources": sorted(sources[requested_path]),
            "owner_uid": metadata.st_uid,
            "mode": stat.S_IMODE(metadata.st_mode),
            "bytes": len(payload),
            "sha256": file_sha256,
        })
    _require(bool(rows), "runtime closure is empty")
    return {
        "algorithm": "rail-balance-runtime-closure-v1",
        "file_count": len(rows),
        "sha256": digest.hexdigest(),
        "files": rows,
        "status": "ROOT_OWNED_EXCEPT_PINNED_PROGRAM",
    }


def _git(worktree: Path, *arguments: str, text: bool = True) -> str | bytes:
    command = (
        _PINNED_TOOL_PATHS["git"],
        "--no-pager",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.allow=never",
        "-C",
        str(worktree),
        *arguments,
    )
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    try:
        result = subprocess.run(
            command,
            cwd="/",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise PreflightError(
            f"pinned Git timed out after {COMMAND_TIMEOUT_SECONDS} seconds"
        ) from error
    except OSError as error:
        raise PreflightError(f"cannot execute pinned Git: {error}") from error
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace")
        raise PreflightError(f"Git preflight failed: {stderr.strip()}")
    if text:
        return result.stdout.decode("utf-8", "strict")
    return result.stdout


def _tracked_paths_content(
    worktree: Path,
    roots: Sequence[str],
    *,
    algorithm: str,
    label: str,
) -> dict[str, Any]:
    normalized_roots = tuple(roots)
    _require(bool(normalized_roots), f"tracked {label} roots are empty")
    _require(
        normalized_roots == tuple(sorted(set(normalized_roots))),
        f"tracked {label} roots must be sorted and unique",
    )
    for index, root in enumerate(normalized_roots):
        if root != ".":
            _safe_relative(root, f"tracked {label} root[{index}]")
    raw = _git(
        worktree,
        "ls-files",
        "-s",
        "-z",
        "--",
        *normalized_roots,
        text=False,
    )
    assert isinstance(raw, bytes)
    entries: list[tuple[str, str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            prefix, raw_path = record.split(b"\t", 1)
            mode, blob, stage = prefix.decode("ascii").split(" ")
            relative = raw_path.decode("utf-8", "strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise PreflightError(
                f"Git returned an invalid tracked {label} record"
            ) from error
        _require(stage == "0", f"tracked {label} contains an unmerged index entry")
        _require(mode in {"100644", "100755", "120000"},
                 f"tracked {label} contains a submodule/special mode: {relative}")
        _safe_relative(relative, f"tracked {label} path")
        _require(
            any(
                root == "."
                or relative == root
                or relative.startswith(f"{root}/")
                for root in normalized_roots
            ),
            f"tracked path escaped {label}: {relative}",
        )
        entries.append((relative, mode, blob))
    _require(bool(entries), f"tracked {label} is empty: {normalized_roots}")
    _require(entries == sorted(entries), f"tracked {label} paths are not sorted")

    digest = hashlib.sha256(algorithm.encode("ascii") + b"\0")
    if algorithm != "rail-balance-subtree-content-v1":
        digest.update(_canonical_bytes(list(normalized_roots)) + b"\0")
    total_bytes = 0
    rows: list[dict[str, Any]] = []
    for relative, mode, blob in entries:
        path = worktree / relative
        if mode == "120000":
            # UCCL intentionally tracks three wrapper links.  Hash the link
            # bytes exactly as Git does, never follow it for content, and
            # require the resolved target to remain inside this checkout.
            parent_metadata = _assert_no_symlink_components(
                path.parent, f"tracked symlink parent {relative}"
            )
            _require(parent_metadata.st_uid == os.geteuid(),
                     f"tracked symlink parent is foreign-owned: {relative}")
            metadata = os.lstat(path)
            _require(stat.S_ISLNK(metadata.st_mode),
                     f"tracked symlink is not a symlink: {relative}")
            _require(metadata.st_uid == os.geteuid(),
                     f"tracked symlink is foreign-owned: {relative}")
            raw_target = os.readlink(os.fsencode(path))
            assert isinstance(raw_target, bytes)
            payload = raw_target
            try:
                resolved_target = path.resolve(strict=True)
                resolved_target.relative_to(worktree)
            except (OSError, ValueError) as error:
                raise PreflightError(
                    f"tracked symlink escapes or is dangling: {relative}"
                ) from error
        else:
            metadata = _assert_no_symlink_components(path, f"tracked file {relative}")
            _require(stat.S_ISREG(metadata.st_mode),
                     f"tracked path is not regular: {relative}")
            _require(metadata.st_uid == os.geteuid(),
                     f"tracked path is foreign-owned: {relative}")
            _require(metadata.st_mode & 0o022 == 0,
                     f"tracked path is group/other writable: {relative}")
            payload = _read_regular_no_follow(
                path,
                MAX_TRACKED_FILE_BYTES,
                owner_uid=os.geteuid(),
            )
        total_bytes += len(payload)
        _require(total_bytes <= MAX_TRACKED_SUBTREE_BYTES,
                 f"tracked {label} exceeds the static hash budget")
        path_bytes = relative.encode("utf-8")
        digest.update(mode.encode("ascii") + b"\0")
        digest.update(len(path_bytes).to_bytes(8, "big") + path_bytes)
        digest.update(len(payload).to_bytes(8, "big") + payload)
        rows.append({
            "path": relative,
            "mode": mode,
            "git_blob": blob,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    return {
        "algorithm": algorithm,
        "roots": list(normalized_roots),
        "file_count": len(rows),
        "tracked_symlink_count": sum(row["mode"] == "120000" for row in rows),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "files": rows,
    }


def _tracked_subtree_content(worktree: Path, subtree: str) -> dict[str, Any]:
    return _tracked_paths_content(
        worktree,
        (subtree,),
        algorithm="rail-balance-subtree-content-v1",
        label="subtree",
    )


def _tracked_build_input_content(
    worktree: Path, roots: Sequence[str]
) -> dict[str, Any]:
    return _tracked_paths_content(
        worktree,
        roots,
        algorithm="rail-balance-build-input-content-v1",
        label="build input closure",
    )


def _inspect_checkout(competitor: Mapping[str, Any]) -> dict[str, Any]:
    checkout = Path(str(competitor["checkout"]))
    metadata = _assert_no_symlink_components(checkout, f"{competitor['id']} checkout")
    _require(stat.S_ISDIR(metadata.st_mode), f"{competitor['id']} checkout is not a directory")
    _require(metadata.st_uid == os.geteuid(), f"{competitor['id']} checkout is foreign-owned")
    _require(metadata.st_mode & 0o022 == 0,
             f"{competitor['id']} checkout is group/other writable")
    top = Path(str(_git(checkout, "rev-parse", "--show-toplevel")).strip()).resolve(
        strict=True
    )
    _require(top == checkout, f"{competitor['id']} checkout is not its Git top level")
    head = str(_git(checkout, "rev-parse", "--verify", "HEAD^{commit}")).strip()
    _require(head == competitor["commit"], f"{competitor['id']} HEAD is stale")
    source_tree = str(_git(checkout, "rev-parse", "--verify", "HEAD^{tree}")).strip()
    _require(source_tree == competitor["source_tree"],
             f"{competitor['id']} source tree changed")
    subtree_tree = str(
        _git(
            checkout,
            "rev-parse",
            "--verify",
            f"HEAD:{competitor['subtree']}",
        )
    ).strip()
    _require(subtree_tree == competitor["subtree_tree"],
             f"{competitor['id']} subtree tree changed")
    content = _tracked_subtree_content(checkout, str(competitor["subtree"]))
    _require(content["sha256"] == competitor["subtree_content_sha256"],
             f"{competitor['id']} subtree content SHA256 changed")
    materialization = competitor["source_materialization"]
    build_inputs = _tracked_build_input_content(
        checkout,
        tuple(str(path) for path in materialization["paths"]),
    )
    _require(
        build_inputs["sha256"] == materialization["expected_content_sha256"],
        f"{competitor['id']} build input closure SHA256 changed",
    )
    return {
        "id": competitor["id"],
        "checkout": str(checkout),
        "checkout_device": metadata.st_dev,
        "checkout_inode": metadata.st_ino,
        "checkout_owner_uid": metadata.st_uid,
        "commit": head,
        "source_tree": source_tree,
        "subtree": competitor["subtree"],
        "subtree_tree": subtree_tree,
        "subtree_content": content,
        "source_materialization": materialization,
        "build_input_closure": build_inputs,
        "git_status_execution_status": "NOT_RUN",
        "worktree_untracked_policy": "IGNORED_NOT_BUILD_INPUT",
        "future_build_execution_status": "NOT_RUN",
        "build_source_policy": competitor["build_source_policy"],
        "future_build_stages": competitor["future_build_stages"],
        "status": "PASSED",
    }


def _recheck_checkouts(
    before: Sequence[Mapping[str, Any]],
    competitors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Require checkout identity and content to be stable across all probes."""

    _require(len(before) == len(competitors), "checkout preflight cardinality changed")
    rows: list[dict[str, Any]] = []
    for index, competitor in enumerate(competitors):
        after = _inspect_checkout(competitor)
        before_bytes = _canonical_bytes(before[index])
        after_bytes = _canonical_bytes(after)
        _require(
            before_bytes == after_bytes,
            f"{competitor['id']} checkout changed during static preflight",
        )
        rows.append({
            "id": competitor["id"],
            "pre": before[index],
            "post": after,
            "stable_identity_sha256": hashlib.sha256(before_bytes).hexdigest(),
            "status": "PASSED",
        })
    return rows


def _probe_dependencies(dependencies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blocker_statuses: list[str] = []
    for dependency in dependencies:
        observations: list[dict[str, Any]] = []
        visible = False
        for raw_path in dependency["visibility_paths"]:
            path = Path(str(raw_path))
            try:
                metadata = path.stat()
                realpath = path.resolve(strict=True)
                exists = True
                kind_matches = {
                    "file": stat.S_ISREG(metadata.st_mode),
                    "directory": stat.S_ISDIR(metadata.st_mode),
                    "executable": (
                        stat.S_ISREG(metadata.st_mode) and metadata.st_mode & 0o111 != 0
                    ),
                }[str(dependency["kind"])]
                visible = visible or kind_matches
                error = None
                observation = {
                    "path": str(path),
                    "exists": exists,
                    "kind_matches": bool(kind_matches),
                    "realpath": str(realpath),
                    "owner_uid": metadata.st_uid,
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "error": error,
                }
            except OSError as caught:
                observation = {
                    "path": str(path),
                    "exists": False,
                    "kind_matches": False,
                    "realpath": None,
                    "owner_uid": None,
                    "mode": None,
                    "error": repr(caught),
                }
            observations.append(observation)
        blocked = bool(dependency["required"]) and not visible
        if blocked:
            blocker_statuses.append(str(dependency["missing_status"]))
        rows.append({
            "id": dependency["id"],
            "competitor_id": dependency["competitor_id"],
            "kind": dependency["kind"],
            "required": dependency["required"],
            "visible": visible,
            "missing_status": dependency["missing_status"],
            "observations": observations,
            "status": "BLOCKED" if blocked else "PASSED",
        })
    return {
        "status": "PASSED" if not blocker_statuses else "BLOCKED",
        "blocker_statuses": sorted(set(blocker_statuses)),
        "checks": rows,
    }


def _disk_probe(command: Mapping[str, Any], resources: Mapping[str, Any]) -> dict[str, Any]:
    used_bytes: int | None = None
    output_path: str | None = None
    lines = str(command["stdout"]).splitlines()
    if (
        command["returncode"] == 0
        and command.get("error") is None
        and str(command["stderr"]) == ""
        and len(lines) == 1
    ):
        fields = lines[0].split(maxsplit=1)
        if len(fields) == 2:
            with suppress(ValueError):
                candidate = int(fields[0])
                if candidate >= 0 and fields[1] == resources["home_path"]:
                    used_bytes = candidate
                    output_path = fields[1]
    projected: int | None = None
    if used_bytes is not None:
        projected = (
            used_bytes
            + int(resources["artifact_budget_bytes"])
            + int(resources["emergency_reserve_bytes"])
        )
    ready = projected is not None and projected <= int(resources["home_hard_limit_bytes"])
    return {
        "home_path": resources["home_path"],
        "parsed_output_path": output_path,
        "home_usage_bytes": used_bytes,
        "artifact_budget_bytes": resources["artifact_budget_bytes"],
        "emergency_reserve_bytes": resources["emergency_reserve_bytes"],
        "projected_home_bytes": projected,
        "home_hard_limit_bytes": resources["home_hard_limit_bytes"],
        "command": command,
        "status": "PASSED" if ready else "BLOCKED_DISK",
    }


def _process_identity(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": pid}
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        result["argv"] = [
            item.decode("utf-8", "replace") for item in raw.split(b"\0") if item
        ]
    except OSError as error:
        result["argv_error"] = repr(error)
    try:
        result["exe"] = os.readlink(f"/proc/{pid}/exe")
    except OSError as error:
        result["exe_error"] = repr(error)
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        suffix = stat_text[stat_text.rfind(")") + 2:].split()
        result["process_group"] = int(suffix[2])
        result["starttime_ticks"] = int(suffix[19])
    except (OSError, IndexError, ValueError) as error:
        result["stat_error"] = repr(error)
    return result


def _gpu_probe(
    mapping_command: Mapping[str, Any],
    process_command: Mapping[str, Any],
    expected_mapping: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    rows: list[dict[str, Any]] = []
    if (
        mapping_command["returncode"] != 0
        or mapping_command.get("error") is not None
        or str(mapping_command["stderr"]) != ""
    ):
        reasons.append("gpu_mapping_command_failed")
    else:
        for line in str(mapping_command["stdout"]).splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",", 5)]
            if len(fields) != 6:
                reasons.append("invalid_gpu_mapping_row")
                continue
            try:
                index = int(fields[0])
                memory_total_mib: int | None = int(fields[3])
            except ValueError:
                reasons.append("invalid_gpu_mapping_row")
                continue
            rows.append({
                "index": index,
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": memory_total_mib,
                "compute_mode": fields[4],
                "mig_mode": fields[5],
                "raw": line,
            })
    actual_mapping = [{"index": row["index"], "uuid": row["uuid"]} for row in rows]
    if actual_mapping != list(expected_mapping):
        reasons.append("gpu_index_uuid_mapping_mismatch")
    actual_static_facts = tuple(
        (
            row["index"],
            row["uuid"],
            row["name"],
            row["memory_total_mib"],
            row["compute_mode"],
            row["mig_mode"],
        )
        for row in rows
    )
    if actual_static_facts != _EXPECTED_GPU_STATIC_FACTS:
        reasons.append("gpu_static_facts_mismatch")
    if any(row["compute_mode"] != "Default" for row in rows):
        reasons.append("gpu_compute_mode_mismatch")
    if any(row["mig_mode"] not in {"Disabled", "N/A", "[N/A]"} for row in rows):
        reasons.append("gpu_mig_mode_mismatch")

    process_rows: list[dict[str, Any]] = []
    process_snapshot_valid = (
        process_command["returncode"] == 0
        and process_command.get("error") is None
        and str(process_command["stderr"]) == ""
    )
    if not process_snapshot_valid:
        reasons.append("gpu_process_command_failed")
    else:
        for line in str(process_command["stdout"]).splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",", 3)]
            if len(fields) != 4:
                reasons.append("invalid_gpu_process_row")
                process_snapshot_valid = False
                process_rows.append({"raw": line, "parse_error": True})
                continue
            try:
                pid: int | None = int(fields[1])
            except ValueError:
                pid = None
                reasons.append("invalid_gpu_process_row")
                process_snapshot_valid = False
            try:
                used_memory_mib: int | None = int(fields[3].split()[0])
            except (IndexError, ValueError):
                used_memory_mib = None
                reasons.append("invalid_gpu_process_row")
                process_snapshot_valid = False
            if not _GPU_UUID.fullmatch(fields[0]) or fields[0] not in {
                str(row["uuid"]) for row in expected_mapping
            }:
                reasons.append("invalid_gpu_process_row")
                process_snapshot_valid = False
            process_rows.append({
                "gpu_uuid": fields[0],
                "pid": pid,
                "process_name": fields[2],
                "used_memory_mib": used_memory_mib,
                "raw": line,
                "process_identity": None if pid is None else _process_identity(pid),
            })
    occupied_pids = sorted({
        row["pid"] for row in process_rows
        if isinstance(row.get("pid"), int)
    })
    mapping_reasons = {
        reason
        for reason in reasons
        if reason != "gpu_process_command_failed"
        and reason != "invalid_gpu_process_row"
    }
    mapping_ready = not mapping_reasons
    availability = (
        "UNKNOWN"
        if not process_snapshot_valid
        else "WAITING_GPU" if occupied_pids else "IDLE"
    )
    return {
        "mapping_command": mapping_command,
        "process_command": process_command,
        "gpu_rows": rows,
        "process_rows": process_rows,
        "occupied_pids": occupied_pids,
        "mapping_status": "PASSED" if mapping_ready else "BLOCKED_GPU_MAPPING",
        "process_status": "PASSED" if process_snapshot_valid else "BLOCKED",
        "availability_status": availability,
        "reasons": sorted(set(reasons)),
    }


def _topology_probe(command: Mapping[str, Any], nic_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    matrix: dict[str, dict[str, str]] = {}
    expected_columns = [f"GPU{index}" for index in range(8)] + [
        str(row["topology_label"]) for row in nic_rows
    ]
    if command["returncode"] != 0:
        reasons.append("topology_command_failed")
    else:
        # Newer nvidia-smi versions underline the header even when stdout is
        # redirected.  Retain raw stdout in ``command`` but parse a copy with
        # ANSI control sequences removed.
        lines = [
            _ANSI_ESCAPE.sub("", line)
            for line in str(command["stdout"]).splitlines()
        ]
        header_index = next(
            (
                index for index, line in enumerate(lines)
                if line.split()[:16] == expected_columns
            ),
            None,
        )
        if header_index is None:
            reasons.append("topology_header_mismatch")
        else:
            for line in lines[header_index + 1:]:
                fields = line.split()
                if not fields or fields[0] not in expected_columns:
                    continue
                if len(fields) < 17:
                    reasons.append(f"topology_short_row:{fields[0]}")
                    continue
                matrix[fields[0]] = dict(zip(expected_columns, fields[1:17]))
            for gpu_index in range(8):
                label = f"GPU{gpu_index}"
                row = matrix.get(label)
                if row is None:
                    reasons.append(f"topology_missing_row:{label}")
                    continue
                for peer_index in range(8):
                    expected = "X" if peer_index == gpu_index else "NV18"
                    if row.get(f"GPU{peer_index}") != expected:
                        reasons.append(
                            f"topology_gpu_link_mismatch:{label}:GPU{peer_index}"
                        )
                nic_label = str(nic_rows[gpu_index]["topology_label"])
                if row.get(nic_label) != "PIX":
                    reasons.append(f"topology_gpu_nic_mismatch:{label}:{nic_label}")
    return {
        "command": command,
        "expected_gpu_peer_link": "NV18",
        "expected_matching_gpu_nic_link": "PIX",
        "matrix": matrix,
        "reasons": sorted(set(reasons)),
        "status": "PASSED" if not reasons else "BLOCKED_TOPOLOGY",
    }


def _read_sysfs_text(path: Path) -> tuple[str | None, str | None]:
    try:
        payload = path.read_bytes()
        _require(len(payload) <= 4096, f"sysfs value is too large: {path}")
        return payload.decode("utf-8", "strict").strip(), None
    except (OSError, UnicodeDecodeError, PreflightError) as error:
        return None, repr(error)


def _network_probe(
    nic_specs: Sequence[Mapping[str, Any]],
    ibv_command: Mapping[str, Any],
    expected_devices: Sequence[str],
) -> dict[str, Any]:
    reasons: list[str] = []
    rows: list[dict[str, Any]] = []
    for spec in nic_specs:
        base = Path("/sys/class/infiniband") / str(spec["name"])
        values: dict[str, str | None] = {}
        errors: dict[str, str] = {}
        paths = {
            "state": base / "ports" / str(spec["port"]) / "state",
            "physical_state": base / "ports" / str(spec["port"]) / "phys_state",
            "rate": base / "ports" / str(spec["port"]) / "rate",
            "link_layer": base / "ports" / str(spec["port"]) / "link_layer",
            "numa_node": base / "device" / "numa_node",
            "vendor": base / "device" / "vendor",
            "device": base / "device" / "device",
        }
        for field, path in paths.items():
            observed, error = _read_sysfs_text(path)
            values[field] = observed
            if error is not None:
                errors[field] = error
        try:
            device_realpath = (base / "device").resolve(strict=True)
            bdf: str | None = device_realpath.name
            _require(str(device_realpath).startswith("/sys/devices/"),
                     f"NIC device escaped sysfs: {spec['name']}")
        except (OSError, PreflightError) as error:
            bdf = None
            errors["pci_bdf"] = repr(error)
        expected_values = {
            "state": spec["state"],
            "physical_state": spec["physical_state"],
            "rate": spec["rate"],
            "link_layer": spec["link_layer"],
            "numa_node": str(spec["numa_node"]),
            "vendor": spec["vendor"],
            "device": spec["device"],
        }
        mismatches = sorted(
            field for field, expected in expected_values.items()
            if values.get(field) != expected
        )
        if bdf != spec["pci_bdf"]:
            mismatches.append("pci_bdf")
        if errors or mismatches:
            reasons.append(f"nic_mismatch:{spec['name']}")
        rows.append({
            "name": spec["name"],
            "topology_label": spec["topology_label"],
            "port": spec["port"],
            "expected": {**expected_values, "pci_bdf": spec["pci_bdf"]},
            "observed": {**values, "pci_bdf": bdf},
            "errors": errors,
            "mismatches": sorted(set(mismatches)),
            "status": "PASSED" if not errors and not mismatches else "BLOCKED",
        })

    parsed_devices: list[str] = []
    if ibv_command["returncode"] != 0:
        reasons.append("ibv_devices_command_failed")
    else:
        for line in str(ibv_command["stdout"]).splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0].startswith("mlx5_"):
                parsed_devices.append(fields[0])
        parsed_devices = sorted(set(parsed_devices))
        if parsed_devices != list(expected_devices):
            reasons.append("ibv_devices_mismatch")
    return {
        "nics": rows,
        "ibv_devices": {
            "command": ibv_command,
            "parsed_devices": parsed_devices,
            "expected_devices": list(expected_devices),
            "status": "PASSED" if (
                ibv_command["returncode"] == 0
                and parsed_devices == list(expected_devices)
            ) else "BLOCKED_RUNTIME_PROBE",
        },
        "ibv_devinfo": {
            "execution_status": "NOT_RUN",
            "required_for_static_ready": False,
            "limitation": (
                "sysfs ACTIVE/LinkUp and ibv_devices enumeration do not establish "
                "that a verbs context, device/net directory, RDMA path, or EP runtime opens"
            ),
        },
        "reasons": sorted(set(reasons)),
        "status": "PASSED" if not reasons else "BLOCKED_NETWORK",
    }


def _select_status(statuses: Sequence[str]) -> str:
    priorities = (
        "BLOCKED_RUNTIME_PROBE",
        "BLOCKED_DISK",
        "BLOCKED_GPU_MAPPING",
        "BLOCKED_TOPOLOGY",
        "BLOCKED_NETWORK",
        "BLOCKED_DEPENDENCY_NANOBIND",
        "BLOCKED_DEPENDENCY",
        "WAITING_GPU",
    )
    observed = set(statuses)
    for status in priorities:
        if status in observed:
            return status
    return "STATIC_CHECKS_PASSED_NO_BUILD_AUTHORITY"


def run_preflight(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    manifest_raw_sha256: str,
    manifest_canonical_sha256: str,
    capture: Callable[[Sequence[str], int], dict[str, Any]] = _capture,
) -> dict[str, Any]:
    """Run only the declared static checks and return a strict JSON result."""

    loaded_manifest, _raw, observed_raw_sha256, observed_canonical_sha256 = (
        load_manifest(manifest_path)
    )
    _require(
        _canonical_bytes(loaded_manifest) == _canonical_bytes(manifest),
        "in-memory manifest differs from the immutable manifest path",
    )
    _require(
        manifest_raw_sha256 == observed_raw_sha256,
        "caller-supplied raw manifest SHA256 is stale or forged",
    )
    _require(
        manifest_canonical_sha256 == observed_canonical_sha256,
        "caller-supplied canonical manifest SHA256 is stale or forged",
    )
    capture_identity = (
        "PINNED_SUBPROCESS_CAPTURE" if capture is _capture else "TEST_DOUBLE_CAPTURE"
    )
    created_utc = datetime.now(timezone.utc).isoformat()
    started_ns = time.monotonic_ns()
    interpreter = _inspect_interpreter(
        require_isolated_startup=capture is _capture
    )
    program = _inspect_preflight_program()
    runtime_closure = (
        _inspect_runtime_closure()
        if capture is _capture
        else {"status": "TEST_NOT_ATTESTED"}
    )
    tools = [_inspect_tool(tool) for tool in manifest["tools"]]
    checkout_pre = [_inspect_checkout(item) for item in manifest["competitors"]]
    dependencies = _probe_dependencies(manifest["dependencies"])
    probes = manifest["probes"]
    disk_command = capture(tuple(probes["du_argv"]), COMMAND_TIMEOUT_SECONDS)
    mapping_command = capture(tuple(probes["gpu_mapping_argv"]), COMMAND_TIMEOUT_SECONDS)
    process_command = capture(tuple(probes["gpu_process_argv"]), COMMAND_TIMEOUT_SECONDS)
    topology_command = capture(tuple(probes["topology_argv"]), COMMAND_TIMEOUT_SECONDS)
    ibv_command = capture(tuple(probes["ibv_devices_argv"]), COMMAND_TIMEOUT_SECONDS)

    disk = _disk_probe(disk_command, manifest["resources"])
    gpu = _gpu_probe(
        mapping_command,
        process_command,
        manifest["resources"]["expected_gpu_mapping"],
    )
    topology = _topology_probe(topology_command, manifest["network"]["nics"])
    network = _network_probe(
        manifest["network"]["nics"],
        ibv_command,
        manifest["network"]["ibv_expected_devices"],
    )
    # This is deliberately the final filesystem observation before result
    # materialization.  A clean initial checkout is insufficient if it drifts
    # while disk/GPU/NIC probes are running.
    checkouts = _recheck_checkouts(checkout_pre, manifest["competitors"])

    observations: list[str] = list(dependencies["blocker_statuses"])
    if disk["status"] != "PASSED":
        observations.append("BLOCKED_DISK")
    if gpu["mapping_status"] != "PASSED":
        observations.append("BLOCKED_GPU_MAPPING")
    if gpu["availability_status"] == "WAITING_GPU":
        observations.append("WAITING_GPU")
    elif gpu["availability_status"] == "UNKNOWN":
        observations.append("BLOCKED_GPU_MAPPING")
    if topology["status"] != "PASSED":
        observations.append("BLOCKED_TOPOLOGY")
    if network["status"] != "PASSED":
        observations.append("BLOCKED_NETWORK")
    command_results = (
        disk_command,
        mapping_command,
        process_command,
        topology_command,
        ibv_command,
    )
    if any(command["returncode"] is None for command in command_results):
        observations.append("BLOCKED_RUNTIME_PROBE")
    observations = sorted(set(observations))
    overall_status = _select_status(observations)
    _require(overall_status in _TERMINAL_STATUSES, "internal invalid terminal status")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "preflight_id": manifest["preflight_id"],
        "created_utc": created_utc,
        "duration_ns": time.monotonic_ns() - started_ns,
        "mode": "check_only",
        "manifest": {
            "path": str(Path(os.path.abspath(manifest_path))),
            "raw_sha256": manifest_raw_sha256,
            "canonical_sha256": manifest_canonical_sha256,
        },
        "preflight_program": program,
        "interpreter": interpreter,
        "runtime_closure": runtime_closure,
        "probe_capture_identity": capture_identity,
        "execution_contract": {
            "gpu_launch_policy": "FORBIDDEN",
            "build_execution_status": "NOT_RUN",
            "future_build_executor_status": "NOT_IMPLEMENTED",
            "performance_claim_allowed": False,
            "checkout_write_policy": "FORBIDDEN",
            "stage_root_policy": "FRESH_EMPTY_OWNER_0700_NO_SYMLINKS_NO_REUSE",
            "source_materialization_policy": "PINNED_GIT_OBJECTS_ONLY",
        },
        "overall_status": overall_status,
        "status_observations": observations,
        "tools": tools,
        "competitors": checkouts,
        "dependencies": dependencies,
        "disk": disk,
        "gpu": gpu,
        "topology": topology,
        "network": network,
        "limitations": [
            "No build command recorded in the manifest was executed.",
            "No torch, deep_ep, torchrun, mpirun, CUDA API, or competitor extension was imported or launched.",
            "nvidia-smi and sysfs observations are a point-in-time snapshot, not an exclusive GPU lease.",
            (
                "The argv/environment/runtime-closure record is in-process evidence "
                "under the root-owned OS trust base, not remote attestation against "
                "a hostile parent, ptrace peer, kernel, or same-UID process."
            ),
            (
                "Dependency probes establish only frozen-path presence and kind. "
                "A future build executor must independently stage or hash every "
                "Python/header/library input and recheck disk, source, and environment."
            ),
            (
                "STATIC_CHECKS_PASSED_NO_BUILD_AUTHORITY is only a static "
                "snapshot and makes no build, correctness, or performance claim."
            ),
        ],
    }


def _open_directory_no_follow(path: Path) -> int:
    path = Path(os.path.abspath(path))
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
        _require(metadata.st_uid == os.geteuid(), f"foreign-owned output directory: {path}")
        _require(
            stat.S_IMODE(metadata.st_mode) == 0o700,
            f"output directory mode must be 0700: {path}",
        )
        return descriptor
    except (OSError, PreflightError) as error:
        os.close(descriptor)
        if isinstance(error, PreflightError):
            raise
        raise PreflightError(f"cannot open output directory safely {path}: {error}") from error


def _validate_cli_output_path(path: Path) -> Path:
    normalized = Path(os.path.abspath(path))
    _require(normalized.name == "result.json",
             "CLI output filename must be result.json")
    try:
        relative = normalized.relative_to(OUTPUT_ROOT)
    except ValueError as error:
        raise PreflightError(
            f"CLI output must remain below {OUTPUT_ROOT}"
        ) from error
    _require(
        len(relative.parts) == 2 and relative.parts[0] not in {"", ".", ".."},
        "CLI output must be one run directory below the competitor artifact root",
    )
    return normalized


def _atomic_json_no_clobber(path: Path, value: object) -> str:
    payload = _canonical_bytes(value) + b"\n"
    path = Path(os.path.abspath(path))
    _require(path.name not in {"", ".", ".."}, "output filename is invalid")
    parent_fd = _open_directory_no_follow(path.parent)
    temporary = f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor: int | None = None
    try:
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise PreflightError(f"immutable output already exists: {path}")
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
            os.fchmod(output.fileno(), 0o444)
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
            raise PreflightError(f"output appeared concurrently: {path}") from error
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return hashlib.sha256(payload).hexdigest()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=parent_fd)
        os.close(parent_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        output_path = _validate_cli_output_path(arguments.output)
        process_launch = _inspect_process_launch(
            manifest_path=arguments.manifest,
            output_path=output_path,
        )
        manifest, _raw, raw_sha256, canonical_sha256 = load_manifest(arguments.manifest)
        result = run_preflight(
            manifest,
            manifest_path=arguments.manifest,
            manifest_raw_sha256=raw_sha256,
            manifest_canonical_sha256=canonical_sha256,
        )
        _require(
            result["probe_capture_identity"] == "PINNED_SUBPROCESS_CAPTURE",
            "CLI result did not use the pinned subprocess capture",
        )
        _require(
            result["interpreter"]["startup_contract_status"]
            == "PINNED_ISOLATED_NO_SITE",
            "CLI result did not use the pinned isolated startup contract",
        )
        _require(
            result["runtime_closure"]["status"]
            == "ROOT_OWNED_EXCEPT_PINNED_PROGRAM",
            "CLI runtime closure is not trusted",
        )
        result["process_launch"] = process_launch
        output_sha256 = _atomic_json_no_clobber(output_path, result)
    except PreflightError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(output_path),
        "output_sha256": output_sha256,
        "overall_status": result["overall_status"],
        "build_execution_status": "NOT_RUN",
        "performance_claim_allowed": False,
    }, allow_nan=False, sort_keys=True))
    return (
        0
        if result["overall_status"]
        == "STATIC_CHECKS_PASSED_NO_BUILD_AUTHORITY"
        else 3
    )


if __name__ == "__main__":
    raise SystemExit(main())
