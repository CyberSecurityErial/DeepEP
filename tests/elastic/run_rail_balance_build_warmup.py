#!/usr/bin/env python3
"""Rebuild DeepEP and preflight one production-shaped Hybrid codegen pair.

This is compile-only evidence.  It does not initialize a real Rail team and it
does not warm the full production force runtime cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import sysconfig
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[2]
_LABEL = "HYBRID_CODEGEN_WARMUP_ONLY"
_DISPATCH_CASE = "8x2_h7168_k8"
_COMBINE_CASE = "8x2_h7168_k8_rank_tt"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=_ROOT, text=True).strip()


def _require_clean_tree() -> None:
    _require(not _git("status", "--porcelain"), "Git worktree must be clean")


def _require_safe_output(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    _require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    try:
        output_dir.relative_to(_ROOT)
    except ValueError:
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", str(output_dir)],
        cwd=_ROOT,
        check=False,
    )
    _require(
        ignored.returncode == 0,
        "output inside the repository must be Git-ignored (use .cache/...)",
    )


def _terminate_process_group(
    process: subprocess.Popen, grace_seconds: int = 10
) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=grace_seconds)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def _run_stage(
    name: str,
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    timeout: int,
) -> dict[str, Any]:
    print(f"C105 prepare: {name}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _terminate_process_group(process)
            raise RuntimeError(
                f"{name} timed out after {timeout}s; see {log_path}"
            ) from error
        except BaseException:
            _terminate_process_group(process)
            raise
    if return_code:
        _terminate_process_group(process, grace_seconds=0)
        raise RuntimeError(f"{name} exited {return_code}; see {log_path}")
    return {"name": name, "command": command, "log": log_path.name}


def _read_probe(log_path: Path) -> dict[str, Any]:
    prefix = "C105_PROBE="
    lines = [
        line for line in log_path.read_text().splitlines() if line.startswith(prefix)
    ]
    _require(len(lines) == 1, "import probe did not emit exactly one record")
    return json.loads(lines[0][len(prefix) :])


def _summarize_cache(cache_root: Path) -> dict[str, Any]:
    cache_dir = cache_root / "cache"
    kernel_dirs = sorted(path for path in cache_dir.glob("kernel.*") if path.is_dir())
    _require(len(kernel_dirs) == 4, "representative pair must emit four kernels")

    tree_digest = hashlib.sha256()
    kernels = []
    for directory in kernel_dirs:
        files = {}
        for filename in ("kernel.cu", "kernel.cubin", "kernel.ptx", "kernel.sass"):
            path = directory / filename
            _require(path.is_file() and path.stat().st_size > 0, f"missing {path}")
            digest = _sha256(path)
            files[filename] = {"bytes": path.stat().st_size, "sha256": digest}
            tree_digest.update(f"{directory.name}/{filename}\0{digest}\n".encode())
        include_hash = (directory / "kernel.cu").read_text().splitlines()[0]
        _require(
            include_hash.startswith("// Includes' hash value: "),
            f"missing recursive include hash in {directory}",
        )
        kernels.append(
            {
                "cache_key": directory.name,
                "recursive_include_hash": include_hash.split(": ", 1)[1],
                "files": files,
            }
        )
    return {
        "path": str(cache_root.resolve()),
        "kernel_count": len(kernels),
        "tree_sha256": tree_digest.hexdigest(),
        "kernels": kernels,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if not args.run_id.strip():
        parser.error("run-id must be nonempty")
    if args.cuda_device < 0 or args.jobs <= 0:
        parser.error("cuda-device must be nonnegative and jobs positive")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    return args


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    _require_clean_tree()
    _require_safe_output(output_dir)
    output_dir.mkdir(parents=True)
    cache_root = output_dir / "jit"
    cache_root.mkdir()

    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.cuda_device),
        EP_DISABLE_GIN="1",
        EP_JIT_CACHE_DIR=str(cache_root),
        EP_JIT_PRINT_COMPILER_COMMAND="1",
        EP_JIT_PTXAS_VERBOSE="1",
        EP_JIT_PTXAS_CHECK="1",
        EP_JIT_DUMP_PTX="1",
        EP_JIT_DUMP_SASS="1",
        MAX_JOBS=str(args.jobs),
        PYTHONPATH=str(_ROOT),
        TORCH_CUDA_ARCH_LIST="9.0",
    )
    python = str(Path(sys.executable).resolve())
    commands = []
    try:
        nvcc_path = shutil.which("nvcc")
        _require(nvcc_path is not None, "nvcc is unavailable")
        toolchain = {
            "nvcc_path": str(Path(nvcc_path).resolve()),
            "nvcc_version": subprocess.check_output(
                [nvcc_path, "--version"], text=True
            ).strip(),
            "driver_versions": sorted(
                set(
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=driver_version",
                            "--format=csv,noheader",
                        ],
                        text=True,
                    ).splitlines()
                )
            ),
        }
        commands.append(
            _run_stage(
                "extension build",
                [
                    python,
                    "setup.py",
                    "build_ext",
                    "--inplace",
                    "--force",
                    "--build-temp",
                    str(output_dir / "build-temp"),
                    "--build-lib",
                    str(output_dir / "build-lib"),
                    "--parallel",
                    str(args.jobs),
                ],
                environment,
                output_dir / "build.log",
                args.timeout,
            )
        )

        extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        _require(extension_suffix, "Python EXT_SUFFIX is unavailable")
        extension_path = (_ROOT / "deep_ep" / f"_C{extension_suffix}").resolve()
        _require(extension_path.is_file(), f"built extension missing: {extension_path}")
        extension_sha = _sha256(extension_path)

        probe = """
import json
import torch
import deep_ep
from deep_ep import _C
from deep_ep.buffers import elastic
torch.cuda.set_device(0)
torch.empty(1, device="cuda")
print("C105_PROBE=" + json.dumps({
    "package_path": deep_ep.__file__,
    "extension_path": _C.__file__,
    "python_version": __import__("sys").version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "nccl": str(torch.cuda.nccl.version()),
    "gpu": torch.cuda.get_device_name(),
    "compute_capability": list(torch.cuda.get_device_capability()),
    "host_force_available": elastic._RAIL_BALANCE_FORCE_HOST_AVAILABLE,
    "compiled_force_available": _C._rail_balance_force_available(),
}, sort_keys=True))
"""
        commands.append(
            _run_stage(
                "extension import",
                [python, "-B", "-c", probe],
                environment,
                output_dir / "import.log",
                args.timeout,
            )
        )
        runtime = _read_probe(output_dir / "import.log")
        _require(
            Path(runtime["package_path"]).resolve() == (_ROOT / "deep_ep/__init__.py"),
            "imported DeepEP package is not this checkout",
        )
        _require(
            Path(runtime["extension_path"]).resolve() == extension_path,
            "imported extension is not the freshly built file",
        )
        _require(runtime["compute_capability"] == [9, 0], "C105 requires SM90")
        _require(runtime["host_force_available"] is False, "host capability changed")
        _require(
            runtime["compiled_force_available"] is False,
            "compiled capability changed",
        )

        commands.append(
            _run_stage(
                "representative dispatch codegen",
                [
                    python,
                    "-B",
                    "tests/elastic/test_rail_balance_hybrid_dispatch_codegen.py",
                    "--case",
                    _DISPATCH_CASE,
                ],
                environment,
                output_dir / "dispatch-codegen.log",
                args.timeout,
            )
        )
        commands.append(
            _run_stage(
                "representative combine codegen",
                [
                    python,
                    "-B",
                    "tests/elastic/test_rail_balance_hybrid_combine_codegen.py",
                    "--case",
                    _COMBINE_CASE,
                ],
                environment,
                output_dir / "combine-codegen.log",
                args.timeout,
            )
        )

        _require(_sha256(extension_path) == extension_sha, "codegen changed extension")
        _require_clean_tree()
        manifest = {
            "schema_version": 1,
            "evidence_label": _LABEL,
            "claim_scope": "build_and_main_kernel_codegen_only",
            "real_gin_runtime": False,
            "full_force_runtime_cache": False,
            "run_id": args.run_id,
            "git_commit": _git("rev-parse", "HEAD"),
            "python": python,
            "toolchain": toolchain,
            "runtime": runtime,
            "extension": {
                "path": str(extension_path),
                "bytes": extension_path.stat().st_size,
                "sha256": extension_sha,
            },
            "environment": {
                key: environment[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "EP_DISABLE_GIN",
                    "EP_JIT_CACHE_DIR",
                    "EP_JIT_PRINT_COMPILER_COMMAND",
                    "EP_JIT_PTXAS_VERBOSE",
                    "EP_JIT_PTXAS_CHECK",
                    "EP_JIT_DUMP_PTX",
                    "EP_JIT_DUMP_SASS",
                    "MAX_JOBS",
                    "PYTHONPATH",
                    "TORCH_CUDA_ARCH_LIST",
                )
            },
            "commands": commands,
            "jit_cache": _summarize_cache(cache_root),
            "limitations": [
                "No real Rail/Gin team was initialized or launched.",
                "Only one production-shaped main dispatch/combine force/legacy pair is cached.",
                "Probe keys use fixed SM64/M8192/Q9/timeout geometry.",
                "Run the D>1 balanced public round trip first to warm the exact full runtime.",
            ],
        }
        _write_json_atomic(output_dir / "manifest.json", manifest)
        print(
            f"PASS C105 build/codegen: {_LABEL} -> {output_dir / 'manifest.json'}",
            flush=True,
        )
    except BaseException:
        (output_dir / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
