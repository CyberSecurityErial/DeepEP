"""CPU-only contracts for the formal source-version campaign round."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import rail_balance_campaign_round as round_eval
from rail_balance_campaign_schema import validate_manifest
import rail_balance_source_round_coordinator as source_coordinator


_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT = _ROOT / "tests/elastic/experiments/hop_local8_sm90_v1.json"
_TEMPLATE = json.loads(_EXPERIMENT.read_text(encoding="utf-8"))
_CONTRACT = deepcopy(_TEMPLATE["frozen_contract"])
_CONTRACT_SHA = round_eval.canonical_sha256(_CONTRACT)
_GPU_MAPPING = deepcopy(_TEMPLATE["resources"]["expected_gpu_index_uuid_mapping"])
_GPU_MAPPING_SHA = round_eval.canonical_sha256(_GPU_MAPPING)
_RUNNER_SHA = hashlib.sha256(
    (_ROOT / "tests/elastic/run_rail_balance_hop_campaign.py").read_bytes()
).hexdigest()
_SCHEMA_SHA = hashlib.sha256(
    (_ROOT / "tests/elastic/rail_balance_campaign_schema.py").read_bytes()
).hexdigest()
_HARNESS_SHA = hashlib.sha256(
    (_ROOT / "tests/elastic/bench_rail_balance_hybrid_lsa.py").read_bytes()
).hexdigest()
_TOOLCHAIN = {
    "schema_version": 1,
    "valid": True,
    "target": "SM90/CUDA-12.8/PyTorch-2.11.0+cu128",
}
_TOOLCHAIN_SHA = round_eval.canonical_sha256(_TOOLCHAIN)
_BOOT_ID = "synthetic-boot-id"
_LOCK_SHA = hashlib.sha256(b"formal-round-global-lock").hexdigest()
_LOADED_LIBRARIES = [
    _file
    for _file in (
        {
            "path": "/usr/local/cuda/lib64/libcudart.so",
            "size_bytes": 1_000_000,
            "sha256": hashlib.sha256(b"libcudart").hexdigest(),
        },
        {
            "path": "/opt/nccl/lib/libnccl.so",
            "size_bytes": 2_000_000,
            "sha256": hashlib.sha256(b"libnccl").hexdigest(),
        },
    )
]
_LOADED_LIBRARIES_SHA = round_eval.canonical_sha256(
    sorted(
        (
            {"sha256": item["sha256"], "size_bytes": item["size_bytes"]}
            for item in _LOADED_LIBRARIES
        ),
        key=lambda row: (row["sha256"], row["size_bytes"]),
    )
)
_PARENT = "leader-v1"
_PARENT_COMMIT = "1" * 40
_WORKLOAD = {
    "stage": "source",
    "case_name": "c100_volume_h256",
    "world_size": 8,
    "hidden": 256,
    "input_iteration": 100,
    "init_dist_seed": 100,
    "remainder_seed": 100,
    "tokens_per_rank": [1024] * 8,
    "num_topk": 4,
    "num_channels": 256,
    "num_experts": 72,
    "num_destinations": 9,
    "local_destination": 0,
    "proxy_capacity_per_egress": 896,
    "dtype": "bfloat16",
    "interference_mode": "none",
    "interference_compute_shape": [2048, 2048, 2048],
}
_ALGORITHM = {
    "hop_mode": "adaptive",
    "two_hop_threshold_percent": 0,
    "max_two_hop_percent": 25,
    "hop_penalty_percent": 50,
}
_TIMER = {
    "clock": (
        "time.perf_counter_ns common monotonic host clock; all ranks "
        "are processes on this single node"
    ),
    "stage_truth": "max(stage_end)-min(stage_start) across 8 ranks",
    "percentile_method": "linear interpolation over sorted samples",
    "std_method": "population",
    "logical_token_bytes": 576,
    "logical_bytes_per_iteration": 0,
    "logical_bandwidth_denominator": "same-node 8-rank global target-stage span",
}
_SEMANTIC = {
    "shape": {**_WORKLOAD, **_ALGORITHM},
    "warmup_iterations": 10,
    "steady_iterations": 100,
    "devices": [f"GPU-{index}" for index in range(8)],
    "environment_class": "isolated-eight-gpu",
}
_SEMANTIC_SHA = round_eval.canonical_sha256(_SEMANTIC)
_BASE_COMMAND = [
    "/home/chen/.cache/deepep-sjlgpt/bin/python",
    "-B",
    "tests/elastic/bench_rail_balance_hybrid_lsa.py",
    "--stage",
    "source",
    "--case-name",
    "c100_volume_h256",
    "--hop-mode",
    "adaptive",
    "--two-hop-threshold-percent",
    "0",
    "--max-two-hop-percent",
    "25",
    "--hop-penalty-percent",
    "50",
    "--warmup-iters",
    "10",
    "--steady-iters",
    "100",
]
_BASE_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "EP_DISABLE_GIN": "1",
    "OMP_NUM_THREADS": "1",
}
_NORMALIZED_EXECUTION_SHA = round_eval.canonical_sha256(
    {
        "semantic_config_sha256": _SEMANTIC_SHA,
        "environment": _BASE_ENVIRONMENT,
        "command": _BASE_COMMAND,
    }
)


def _write_json(path: Path, value: object, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_identity(label: str, size: int) -> dict[str, Any]:
    return {
        "path": f"/synthetic/{label}",
        "size_bytes": size,
        "sha256": hashlib.sha256(label.encode()).hexdigest(),
    }


def _candidate_identity(candidate_id: str, *, parent: str = _PARENT) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "parent": parent,
        "primary_change": f"one CUDA source change for {candidate_id}",
        "hypothesis": f"{candidate_id} lowers source-stage latency",
        "expected_profile_metrics": ["lower target-kernel duration"],
        "risks": ["the changed kernel may no longer be exposed"],
    }


def _source_artifacts(label: str) -> dict[str, Any]:
    extension = _file_identity(f"{label}/deep_ep.so", 4096)
    sources = {
        "tests/elastic/bench_rail_balance_hybrid_lsa.py": {
            "path": str(_ROOT / "tests/elastic/bench_rail_balance_hybrid_lsa.py"),
            "size_bytes": (
                _ROOT / "tests/elastic/bench_rail_balance_hybrid_lsa.py"
            ).stat().st_size,
            "sha256": _HARNESS_SHA,
        },
        "csrc/kernels/internode_ll.cu": _file_identity(
            f"{label}/internode_ll.cu", 8192
        ),
    }
    cubin = {
        **_file_identity(f"{label}/kernel.cubin", 16384),
        "relative_path": "rail_balance_source/kernel.cubin",
    }
    return {"extension": extension, "sources": sources, "cubin": cubin}


def _normalized_file(value: dict[str, Any]) -> dict[str, Any]:
    return {"sha256": value["sha256"], "size_bytes": value["size_bytes"]}


def _normalized_sources(value: dict[str, Any]) -> dict[str, Any]:
    return {path: _normalized_file(identity) for path, identity in value.items()}


def _source_ref(
    worktree: Path,
    *,
    ref_id: str,
    commit: str,
    source_tree_sha: str,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    metadata = worktree.stat()
    build = round_eval.canonical_sha256(
        {
            "source_commit": commit,
            "source_identity_sha256": source_tree_sha,
            "extension": _normalized_file(artifacts["extension"]),
            "sources": _normalized_sources(artifacts["sources"]),
            "toolchain_sha256": _TOOLCHAIN_SHA,
            "campaign_runner_sha256": _RUNNER_SHA,
            "campaign_schema_sha256": _SCHEMA_SHA,
        }
    )
    normalized_cubin = {
        "relative_path": artifacts["cubin"]["relative_path"],
        **_normalized_file(artifacts["cubin"]),
    }
    code = round_eval.canonical_sha256(
        {"build_bundle_sha256": build, "jit_cubins": [normalized_cubin]}
    )
    return {
        "ref_id": ref_id,
        "commit": commit,
        "working_tree_identity_sha256": source_tree_sha,
        "worktree_realpath": str(worktree.resolve()),
        "worktree_device": metadata.st_dev,
        "worktree_inode": metadata.st_ino,
        "build_bundle_sha256": build,
        "code_identity_sha256": code,
    }


def _measurement_contract() -> dict[str, Any]:
    return {
        "report_schema_version": 4,
        "claim_scope": "checked_adapter_only",
        "metric": "stage_truth_ns",
        "warmup_iterations": 10,
        "steady_iterations": 100,
        "world_size": 8,
        "order": list(round_eval._ABBA_BAAB),
        "timer": deepcopy(_TIMER),
        "workload": deepcopy(_WORKLOAD),
        "algorithm": deepcopy(_ALGORITHM),
        "campaign_runner_sha256": _RUNNER_SHA,
        "campaign_schema_sha256": _SCHEMA_SHA,
        "round_evaluator_sha256": round_eval.evaluator_sha256(),
        "benchmark_harness_sha256": _HARNESS_SHA,
        "loaded_libraries_sha256": _LOADED_LIBRARIES_SHA,
        "toolchain_sha256": _TOOLCHAIN_SHA,
        "gpu_mapping_sha256": _GPU_MAPPING_SHA,
        "semantic_config_sha256": _SEMANTIC_SHA,
        "normalized_execution_identity_sha256": _NORMALIZED_EXECUTION_SHA,
        "coordinator_boot_id": _BOOT_ID,
        "coordinator_lock_identity_sha256": _LOCK_SHA,
    }


def _sample(
    category: str,
    index: int,
    duration: int,
    *,
    epoch_base_ns: int,
) -> dict[str, Any]:
    category_offset = 0 if category == "warm" else 100
    epoch = epoch_base_ns + (category_offset + index) * 10_000_000
    rows = [
        {
            "rank": rank,
            "stage_start_ns": epoch,
            "stage_end_ns": epoch + duration,
            "timings_ns": {"stage": duration},
        }
        for rank in range(8)
    ]
    return {
        "category": category,
        "category_index": index,
        "rank_raw": rows,
        "stage_truth_ns": duration,
        "stage_global_span_ns": duration,
        "stage_rank_max_duration_ns": duration,
        "rank_max_ns": {"stage": duration},
    }


def _values(center: int, count: int) -> list[int]:
    return [center + offset for offset in range(-(count // 2), count - count // 2)]


def _report(
    *,
    run_id: str,
    center: int,
    source_ref: dict[str, Any],
    artifacts: dict[str, Any],
    jit_root: str,
    master_port: int,
    json_out: str,
    sample_epoch_base_ns: int,
) -> dict[str, Any]:
    command = [*_BASE_COMMAND, "--master-port", str(master_port), "--json-out", json_out]
    environment = {**_BASE_ENVIRONMENT, "EP_JIT_CACHE_DIR": jit_root}
    pre_jit = {"root": jit_root, "cache_existed": False, "artifacts": []}
    post_jit = {
        "root": jit_root,
        "cache_existed": True,
        "artifacts": [deepcopy(artifacts["cubin"])],
    }
    pre = {
        "git": {"commit": source_ref["commit"], "dirty": False, "status": []},
        "extension": deepcopy(artifacts["extension"]),
        "loaded_libraries": deepcopy(_LOADED_LIBRARIES),
        "sources": deepcopy(artifacts["sources"]),
        "jit_cache": pre_jit,
    }
    post = {
        "git": {"commit": source_ref["commit"], "dirty": False, "status": []},
        "extension": deepcopy(artifacts["extension"]),
        "loaded_libraries": deepcopy(_LOADED_LIBRARIES),
        "sources": deepcopy(artifacts["sources"]),
        "jit_cache": post_jit,
    }
    execution = {
        "semantic_config_sha256": _SEMANTIC_SHA,
        "environment": environment,
        "command": command,
        "jit_cache_root": jit_root,
    }
    warm_values = _values(center, 10)
    steady_values = _values(center, 100)
    return {
        "schema_version": 4,
        "run_id": run_id,
        "claim_scope": "checked_adapter_only",
        "git": {"commit": source_ref["commit"], "dirty": False, "status": []},
        "identity": {
            "semantic_config": deepcopy(_SEMANTIC),
            "semantic_config_sha256": _SEMANTIC_SHA,
            "execution_config_sha256": round_eval.canonical_sha256(execution),
            "code_identity_stable_during_run": True,
            "jit_cache_empty_before_run": True,
            "jit_identity_stable_after_cold": True,
            "pre_measurement": pre,
            "after_cold_transaction_jit": deepcopy(post_jit),
            "post_measurement": post,
        },
        "command": {"argv": command, "cwd": source_ref["worktree_realpath"]},
        "environment": {
            "variables": environment,
            "runtime_state": {
                "system_commands_ok": True,
                "mps_processes_absent": True,
                "profiler_processes_absent": True,
                "unexpected_compute_app_pids": [],
                "gpu_state_validation": {"valid": True, "reasons": []},
            },
        },
        "shape": {**_WORKLOAD, **_ALGORITHM},
        "measurement": {
            "baseline_collection_eligible": True,
            "measurement_depth_ok": True,
            "persistent_report_requested": True,
            "profiler_mode_declared": "disabled",
            "extra_synchronize_inside_stage": False,
            "warmup_iterations": 10,
            "steady_iterations": 100,
            **_TIMER,
        },
        "warm": [
            _sample(
                "warm",
                index,
                value,
                epoch_base_ns=sample_epoch_base_ns,
            )
            for index, value in enumerate(warm_values)
        ],
        "steady": [
            _sample(
                "steady",
                index,
                value,
                epoch_base_ns=sample_epoch_base_ns,
            )
            for index, value in enumerate(steady_values)
        ],
        "steady_summary": {"stage_truth_ns": round_eval._stats(steady_values)},
    }


def _set_option(argv: list[str], option: str, value: str) -> None:
    if option in argv:
        index = argv.index(option)
        argv[index + 1] = value
    else:
        argv.extend((option, value))


def _attempt(
    passed: bool,
    *,
    global_ordinal: int,
    stage_index: int,
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    second = global_ordinal * 1000 + stage_index * 2
    return {
        "attempt": 1,
        "started_utc": f"2026-08-01T00:{second // 60 % 60:02d}:{second % 60:02d}+00:00",
        "finished_utc": f"2026-08-01T00:{second // 60 % 60:02d}:{second % 60 + 1:02d}+00:00",
        "duration_seconds": 1.0,
        "returncode": 0 if passed else 1,
        "declared_artifacts": artifacts or [],
        "passed": passed,
    }


def _source_identity(source_ref: dict[str, Any]) -> dict[str, Any]:
    empty = hashlib.sha256(b"").hexdigest()
    return {
        "commit": source_ref["commit"],
        "dirty": False,
        "status_porcelain": [],
        "tracked_diff_sha256": empty,
        "untracked_tree_sha256": empty,
        "untracked_files": [],
        "working_tree_identity_sha256": source_ref[
            "working_tree_identity_sha256"
        ],
    }


def _write_sha256sums(root: Path) -> Path:
    path = root / "SHA256SUMS"
    rows = []
    for candidate in sorted(item for item in root.rglob("*") if item.is_file()):
        if candidate == path:
            continue
        rows.append(f"{_sha(candidate)}  {candidate.relative_to(root)}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _reseal_block(block: dict[str, Any]) -> None:
    """Rebuild the synthetic terminal chain after an intentional attack mutation."""

    root = Path(block["artifact_root_realpath"])
    raw_manifest_path = root / "manifest.raw.json"
    saved_manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    finalized_path = root / "FINALIZED.json"
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    _write_json(saved_manifest_path, raw_manifest)
    campaign = json.loads(result_path.read_text(encoding="utf-8"))
    campaign["manifest_raw_sha256"] = _sha(raw_manifest_path)
    campaign["manifest_canonical_sha256"] = round_eval.canonical_sha256(raw_manifest)
    report_path = (
        root
        / "stages"
        / block["stage_id"]
        / "attempt-01"
        / block["artifact"]
    )
    report_sha: str | None = None
    if report_path.exists():
        report_sha = _sha(report_path)
        for stage in campaign["stages"]:
            if stage["id"] != block["stage_id"]:
                continue
            declared = stage["attempts"][0]["declared_artifacts"]
            declared[:] = [
                {
                    "path": block["artifact"],
                    "bytes": report_path.stat().st_size,
                    "sha256": report_sha,
                }
            ]
    _write_json(result_path, campaign)
    finalized = json.loads(finalized_path.read_text(encoding="utf-8"))
    finalized_path.unlink()
    sums = _write_sha256sums(root)
    sums_sha = _sha(sums)
    finalized.update(
        {
            "result_sha256": _sha(result_path),
            "sha256sums_sha256": sums_sha,
            "manifest_raw_sha256": _sha(raw_manifest_path),
            "leaderboard_entry_id": round_eval._leaderboard_entry_id(
                campaign=campaign,
                raw_manifest=raw_manifest,
                source={
                    "working_tree_identity_sha256": campaign["source_identity"][
                        "working_tree_identity_sha256"
                    ]
                },
                artifact_root=root,
                sha256sums_sha256=sums_sha,
            ),
        }
    )
    _write_json(finalized_path, finalized)
    coordinator_path = Path(block["coordinator_record"])
    coordinator = json.loads(coordinator_path.read_text(encoding="utf-8"))
    coordinator["campaign_result_raw_sha256"] = _sha(result_path)
    coordinator["campaign_finalized_raw_sha256"] = _sha(finalized_path)
    coordinator["benchmark_report_sha256"] = report_sha
    _write_json(coordinator_path, coordinator)
    block.update(
        {
            "campaign_result_raw_sha256": _sha(result_path),
            "campaign_manifest_raw_sha256": _sha(raw_manifest_path),
            "finalized_raw_sha256": _sha(finalized_path),
            "coordinator_record_raw_sha256": _sha(coordinator_path),
        }
    )


def _block_campaign(
    source_artifact_root: Path,
    *,
    round_variant: dict[str, Any],
    campaign_candidate: dict[str, Any],
    source_ref: dict[str, Any],
    source_artifacts: dict[str, Any],
    role: str,
    ordinal: int,
    global_ordinal: int,
    center: int,
    gate_failure: str | None,
) -> dict[str, Any]:
    source_round_run_id = (
        f"round-1-{global_ordinal:02d}-{round_variant['id']}-{ordinal}"
    )
    root = source_artifact_root / "runs" / source_round_run_id
    root.mkdir(parents=True)
    manifest = deepcopy(_TEMPLATE)
    manifest["campaign_id"] = f"round-{round_variant['id']}-b{ordinal}"
    manifest["candidate"] = deepcopy(campaign_candidate)
    manifest["frozen_contract"] = deepcopy(_CONTRACT)
    selected_id = "bench-abba-a1"
    formal_dependencies = [
        stage["id"]
        for stage in manifest["stages"]
        if stage["kind"] in {"correctness", "sanitizer"}
        and stage.get("enabled", True)
        and stage["resource"] != "multinode"
    ]
    for stage in manifest["stages"]:
        if stage["kind"] == "benchmark":
            _set_option(stage["argv"], "--hop-mode", _ALGORITHM["hop_mode"])
            _set_option(
                stage["argv"],
                "--two-hop-threshold-percent",
                str(_ALGORITHM["two_hop_threshold_percent"]),
            )
            _set_option(
                stage["argv"],
                "--max-two-hop-percent",
                str(_ALGORITHM["max_two_hop_percent"]),
            )
            _set_option(
                stage["argv"],
                "--hop-penalty-percent",
                str(_ALGORITHM["hop_penalty_percent"]),
            )
            _set_option(stage["argv"], "--warmup-iters", "10")
            _set_option(stage["argv"], "--steady-iters", "100")
            if stage["id"] != selected_id:
                stage["enabled"] = False
                stage["skip_reason"] = "single-block formal coordinator plan"
            else:
                stage.pop("enabled", None)
                stage.pop("skip_reason", None)
                stage["depends_on"] = formal_dependencies
        elif str(stage["kind"]).startswith("profile_"):
            stage["enabled"] = False
            stage["skip_reason"] = "profile follows round promotion"
    validate_manifest(manifest)
    _write_json(root / "manifest.raw.json", manifest, sort_keys=False)
    _write_json(root / "manifest.json", manifest, sort_keys=True)
    raw_manifest_sha = _sha(root / "manifest.raw.json")

    target_failure_id = None
    if gate_failure == "compile":
        target_failure_id = next(stage["id"] for stage in manifest["stages"] if stage["kind"] == "compile")
    elif gate_failure == "correctness":
        target_failure_id = "gpu-planner"
    elif gate_failure == "sanitizer":
        target_failure_id = next(stage["id"] for stage in manifest["stages"] if stage["kind"] == "sanitizer")

    stage_results = []
    stage_statuses: dict[str, str] = {}
    report_sha: str | None = None
    for stage_index, plan in enumerate(manifest["stages"]):
        if not plan.get("enabled", True):
            result = {
                "id": plan["id"],
                "kind": plan["kind"],
                "resource": plan["resource"],
                "status": "skipped",
                "reason": f"disabled: {plan['skip_reason']}",
                "attempts": [],
            }
            stage_results.append(result)
            stage_statuses[plan["id"]] = "skipped"
            continue
        if any(
            stage_statuses[dependency] != "passed"
            for dependency in plan["depends_on"]
        ):
            result = {
                "id": plan["id"],
                "kind": plan["kind"],
                "resource": plan["resource"],
                "status": "skipped",
                "reason": "dependency gate did not pass",
                "attempts": [],
            }
            stage_results.append(result)
            stage_statuses[plan["id"]] = "skipped"
            continue
        passed = plan["id"] != target_failure_id
        declared: list[dict[str, Any]] = []
        if plan["id"] == selected_id and gate_failure is None:
            report_path = root / "stages" / selected_id / "attempt-01" / "report.json"
            report = _report(
                run_id=f"c100-{round_variant['id']}-{ordinal}",
                center=center,
                source_ref=source_ref,
                artifacts=source_artifacts,
                jit_root=f"{root}/stages/{selected_id}/attempt-01/jit",
                master_port=31000 + global_ordinal,
                json_out=str(report_path),
                sample_epoch_base_ns=(
                    global_ordinal * 10_000_000_000 + 1_000_000_000
                ),
            )
            _write_json(report_path, report)
            report_sha = _sha(report_path)
            declared = [
                {
                    "path": "report.json",
                    "bytes": report_path.stat().st_size,
                    "sha256": report_sha,
                }
            ]
        result = {
            "id": plan["id"],
            "kind": plan["kind"],
            "resource": plan["resource"],
            "status": "passed" if passed else "failed",
            "reason": None if passed else "synthetic gate failure",
            "attempts": [
                _attempt(
                    passed,
                    global_ordinal=global_ordinal,
                    stage_index=stage_index,
                    artifacts=declared,
                )
            ],
        }
        stage_results.append(result)
        stage_statuses[plan["id"]] = result["status"]
    complete = gate_failure is None
    source_identity = _source_identity(source_ref)
    campaign = {
        "schema_version": 1,
        "run_id": source_round_run_id,
        "campaign_id": manifest["campaign_id"],
        "candidate": deepcopy(campaign_candidate),
        "status": "complete" if complete else "failed",
        "claim_scope": "local_candidate_gate" if complete else "failed_candidate",
        "performance_claim_allowed": False,
        "manifest_raw_sha256": raw_manifest_sha,
        "manifest_canonical_sha256": round_eval.canonical_sha256(manifest),
        "runner_sha256": _RUNNER_SHA,
        "schema_sha256": _SCHEMA_SHA,
        "source_identity": source_identity,
        "source_identity_post": deepcopy(source_identity),
        "source_drift": False,
        "gpu_preflight_reasons": [],
        "campaign_preflight_reasons": [],
        "stages": stage_results,
    }
    _write_json(root / "result.json", campaign)
    _write_json(
        root / "environment/toolchain.json",
        {
            **deepcopy(_TOOLCHAIN),
            "created_utc": f"2026-08-01T00:00:{global_ordinal:02d}+00:00",
        },
    )
    (root / "campaign.log").write_text(
        f"synthetic terminal log for {source_round_run_id}\n", encoding="utf-8"
    )
    sums = _write_sha256sums(root)
    result_sha = _sha(root / "result.json")
    finalized = {
        "schema_version": 1,
        "created_utc": "2026-08-01T00:00:00+00:00",
        "campaign_id": campaign["campaign_id"],
        "run_id": campaign["run_id"],
        "campaign_status": campaign["status"],
        "claim_scope": campaign["claim_scope"],
        "terminal_label": "PASS" if complete else "FAIL",
        "terminal_exit_code": 0 if complete else 1,
        "round_evaluation_allowed": complete,
        "result_sha256": result_sha,
        "sha256sums_sha256": _sha(sums),
        "source_identity_sha256": source_ref["working_tree_identity_sha256"],
        "manifest_raw_sha256": raw_manifest_sha,
        "leaderboard_entry_id": round_eval._leaderboard_entry_id(
            campaign=campaign,
            raw_manifest=manifest,
            source=source_ref,
            artifact_root=root.resolve(),
            sha256sums_sha256=_sha(sums),
        ),
    }
    _write_json(root / "FINALIZED.json", finalized)
    root_stat = root.stat()
    coordinator = {
        "schema_version": 1,
        "round_id": "round-1",
        "variant_id": round_variant["id"],
        "ordinal": ordinal,
        "global_ordinal": global_ordinal,
        "role": role,
        "boot_id": _BOOT_ID,
        "lock_identity_sha256": _LOCK_SHA,
        "monotonic_start_ns": global_ordinal * 10_000_000_000 + 1,
        "monotonic_end_ns": global_ordinal * 10_000_000_000 + 9_000_000_000,
        "artifact_root_realpath": str(root.resolve()),
        "artifact_root_device": root_stat.st_dev,
        "artifact_root_inode": root_stat.st_ino,
        "worktree_realpath": source_ref["worktree_realpath"],
        "worktree_device": source_ref["worktree_device"],
        "worktree_inode": source_ref["worktree_inode"],
        "source_commit": source_ref["commit"],
        "source_identity_sha256": source_ref["working_tree_identity_sha256"],
        "campaign_result_raw_sha256": result_sha,
        "campaign_finalized_raw_sha256": _sha(root / "FINALIZED.json"),
        "benchmark_report_sha256": report_sha,
        "gpu_mapping_sha256": _GPU_MAPPING_SHA,
        "terminal_evidence_complete": True,
    }
    coordinator_path = (
        source_artifact_root / "coordinator" / f"block-{global_ordinal:02d}.json"
    )
    _write_json(coordinator_path, coordinator)
    return {
        "ordinal": ordinal,
        "global_ordinal": global_ordinal,
        "role": role,
        "artifact_root_realpath": str(root.resolve()),
        "artifact_root_device": root_stat.st_dev,
        "artifact_root_inode": root_stat.st_ino,
        "campaign_result_raw_sha256": result_sha,
        "campaign_manifest_raw_sha256": raw_manifest_sha,
        "finalized_raw_sha256": _sha(root / "FINALIZED.json"),
        "coordinator_record": str(coordinator_path.resolve()),
        "coordinator_record_raw_sha256": _sha(coordinator_path),
        "stage_id": selected_id,
        "attempt": 1,
        "artifact": "report.json",
    }


class Fixture:
    def __init__(self, root: Path, *, candidate_count: int = 2) -> None:
        if not 2 <= candidate_count <= 4:
            raise ValueError("fixture candidate_count must be 2-4")
        self.root = root
        self.candidate_count = candidate_count
        self.source_control_worktree = root / "worktrees" / "control"
        self.source_control_worktree.mkdir(parents=True)
        self.git_common = root / "git-common"
        self.git_common.mkdir()
        self.source_artifact_root = root / "source-round-artifacts"
        self.source_artifact_root.mkdir()
        self.parent_worktree = root / "worktrees" / "parent"
        self.parent_worktree.mkdir(parents=True)
        self.parent_artifacts = _source_artifacts("parent")
        self.parent_ref = _source_ref(
            self.parent_worktree,
            ref_id=_PARENT,
            commit=_PARENT_COMMIT,
            source_tree_sha=hashlib.sha256(b"parent-source-tree").hexdigest(),
            artifacts=self.parent_artifacts,
        )
        self.parent_identity = _candidate_identity(_PARENT, parent="leader-v0")
        self.candidate_ordinal = 0
        self.parent_blocks: list[dict[str, Any]] | None = None
        self.parent_centers: list[int] | None = None

    def _ensure_parent_blocks(self, centers: list[int]) -> None:
        if len(centers) != 4:
            raise ValueError("parent needs exactly four centers")
        if self.parent_blocks is not None:
            if centers != self.parent_centers:
                raise ValueError("all candidates must reference the shared parent blocks")
            return
        parent_global_ordinals = (
            0,
            2 * self.candidate_count + 1,
            3 * self.candidate_count + 2,
            3 * self.candidate_count + 3,
        )
        self.parent_centers = list(centers)
        self.parent_blocks = [
            _block_campaign(
                self.source_artifact_root,
                round_variant=self.parent_identity,
                campaign_candidate=self.parent_identity,
                source_ref=self.parent_ref,
                source_artifacts=self.parent_artifacts,
                role="parent",
                ordinal=ordinal,
                global_ordinal=global_ordinal,
                center=center,
                gate_failure=None,
            )
            for ordinal, (global_ordinal, center) in enumerate(
                zip(parent_global_ordinals, centers)
            )
        ]

    def candidate(
        self,
        candidate_id: str,
        *,
        parent_centers: list[int],
        candidate_centers: list[int],
        gate_failure: str | None = None,
        commit: str | None = None,
    ) -> dict[str, Any]:
        if self.candidate_ordinal >= self.candidate_count:
            raise ValueError("more candidates created than declared")
        if len(candidate_centers) != 4:
            raise ValueError("candidate needs exactly four centers")
        self._ensure_parent_blocks(parent_centers)
        candidate_index = self.candidate_ordinal
        label = f"candidate-{candidate_index}"
        self.candidate_ordinal += 1
        worktree = self.root / "worktrees" / candidate_id
        worktree.mkdir(parents=True)
        artifacts = _source_artifacts(label)
        source = _source_ref(
            worktree,
            ref_id=candidate_id,
            commit=commit or f"{candidate_index + 2:x}" * 40,
            source_tree_sha=hashlib.sha256(f"{label}-tree".encode()).hexdigest(),
            artifacts=artifacts,
        )
        identity = _candidate_identity(candidate_id)
        candidate_global_ordinals = (
            1 + candidate_index,
            1 + self.candidate_count + candidate_index,
            2 + 2 * self.candidate_count + candidate_index,
            4 + 3 * self.candidate_count + candidate_index,
        )
        blocks = [
            _block_campaign(
                self.source_artifact_root,
                round_variant=identity,
                campaign_candidate=identity,
                source_ref=source,
                source_artifacts=artifacts,
                role="candidate",
                ordinal=ordinal,
                global_ordinal=global_ordinal,
                center=center,
                gate_failure=gate_failure,
            )
            for ordinal, (global_ordinal, center) in enumerate(
                zip(candidate_global_ordinals, candidate_centers)
            )
        ]
        return {
            **identity,
            "frozen_contract_sha256": _CONTRACT_SHA,
            "source_ref": source,
            "benchmark_blocks": blocks,
        }

    def _checked_worktree(
        self,
        worktree: Path,
        *,
        commit: str,
        tree_label: str,
    ) -> dict[str, Any]:
        worktree_stat = worktree.stat()
        common_stat = self.git_common.stat()
        return {
            "worktree": str(worktree.resolve()),
            "worktree_device": worktree_stat.st_dev,
            "worktree_inode": worktree_stat.st_ino,
            "source_commit": commit,
            "source_tree": hashlib.sha1(tree_label.encode()).hexdigest(),
            "git_common_dir": str(self.git_common.resolve()),
            "git_common_device": common_stat.st_dev,
            "git_common_inode": common_stat.st_ino,
        }

    def _materialize_source_round_binding(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        candidate_ids = [candidate["id"] for candidate in candidates]
        if candidate_ids != sorted(candidate_ids):
            raise ValueError("source-round candidates must be sorted")
        source_contract = {
            "algorithm": deepcopy(_ALGORITHM),
            "shape": deepcopy(_WORKLOAD),
            "timing": {
                "warmup_iterations": 10,
                "steady_iterations": 100,
                **deepcopy(_TIMER),
            },
            "toolchain": deepcopy(_TOOLCHAIN),
            "gpu_mapping": deepcopy(_GPU_MAPPING),
        }
        source_contract_sha = source_coordinator.canonical_sha256(source_contract)

        def variant(
            variant_id: str, source_ref: dict[str, Any]
        ) -> dict[str, Any]:
            return {
                "id": variant_id,
                "worktree": source_ref["worktree_realpath"],
                "source_commit": source_ref["commit"],
                "frozen_execution_contract_sha256": source_contract_sha,
                "build_bundle_ref": f"bundles/{variant_id}/build",
                "gate_bundle_ref": f"bundles/{variant_id}/gates",
            }

        parent_variant = variant(_PARENT, self.parent_ref)
        source_candidates = []
        patch_checks = []
        checked_candidates = []
        changed_files = ["csrc/kernels/internode_ll.cu"]
        for candidate in candidates:
            source_candidate = variant(candidate["id"], candidate["source_ref"])
            diff_sha = hashlib.sha256(
                f"{_PARENT_COMMIT}..{candidate['source_ref']['commit']}".encode()
            ).hexdigest()
            source_candidate.update(
                {
                    "declared_patch": {
                        "base_commit": _PARENT_COMMIT,
                        "diff_sha256": diff_sha,
                        "changed_files": changed_files,
                    },
                    "primary_change": candidate["primary_change"],
                    "hypothesis": candidate["hypothesis"],
                    "expected_profile_metrics": candidate[
                        "expected_profile_metrics"
                    ],
                    "risks": candidate["risks"],
                }
            )
            source_candidates.append(source_candidate)
            checked = self._checked_worktree(
                Path(candidate["source_ref"]["worktree_realpath"]),
                commit=candidate["source_ref"]["commit"],
                tree_label=f"tree-{candidate['id']}",
            )
            checked_candidates.append({"id": candidate["id"], **checked})
            patch_checks.append(
                {
                    "candidate_id": candidate["id"],
                    "base_commit": _PARENT_COMMIT,
                    "candidate_commit": candidate["source_ref"]["commit"],
                    "commit_distance": 1,
                    "direct_single_parent": True,
                    "changed_files": changed_files,
                    "diff_sha256": diff_sha,
                }
            )
        harness_sha = {
            path: _sha(_ROOT / path)
            for path in sorted(source_coordinator.REQUIRED_HARNESS_PATHS)
        }
        source_manifest = {
            "schema_version": 1,
            "round_id": "round-1",
            "mode": "check_only",
            "execution_status": "NOT_RUN",
            "coordinator": {
                "control_worktree": str(self.source_control_worktree.resolve()),
                "control_commit": _PARENT_COMMIT,
                "artifact_root": str(self.source_artifact_root.resolve()),
                "lock_path": source_coordinator.SOURCE_ROUND_LOCK,
            },
            "frozen_execution_contract": source_contract,
            "source_policy": {
                "parent_id": _PARENT,
                "allowed_candidate_files": changed_files,
                "harness_sha256": harness_sha,
            },
            "parent": parent_variant,
            "candidates": source_candidates,
        }
        source_manifest_path = self.root / "source-round-manifest.json"
        _write_json(source_manifest_path, source_manifest, sort_keys=False)
        source_manifest_raw_sha = _sha(source_manifest_path)
        source_manifest_canonical_sha = source_coordinator.canonical_sha256(
            source_manifest
        )
        common_stat = self.git_common.stat()
        checked_control = self._checked_worktree(
            self.source_control_worktree,
            commit=_PARENT_COMMIT,
            tree_label="tree-control",
        )
        checked_parent = self._checked_worktree(
            self.parent_worktree,
            commit=_PARENT_COMMIT,
            tree_label="tree-parent",
        )
        preflight = {
            "status": "passed",
            "scope": "git_and_plan_only_no_cuda_execution",
            "control": checked_control,
            "parent": checked_parent,
            "candidates": checked_candidates,
            "git_common_identity": {
                "realpath": str(self.git_common.resolve()),
                "device": common_stat.st_dev,
                "inode": common_stat.st_ino,
            },
            "coordinator_git": {
                "path": "/usr/bin/git",
                "realpath": "/usr/bin/git",
                "device": 1,
                "inode": 1,
                "sha256": hashlib.sha256(b"synthetic-pinned-git").hexdigest(),
                "version": "git version synthetic",
            },
            "artifact_root": str(self.source_artifact_root.resolve()),
            "harness_sha256": harness_sha,
            "harness_checkout_count": 2 + len(candidates),
            "patches": patch_checks,
        }
        plan = source_coordinator.materialize_plan(
            source_manifest,
            source_manifest_raw_sha,
            source_manifest_canonical_sha,
            preflight,
        )
        plan_path = self.root / "source-round-plan.json"
        _write_json(plan_path, plan)
        plan_raw_sha = _sha(plan_path)
        coordinator_sha = _sha(
            _ROOT / "tests/elastic/rail_balance_source_round_coordinator.py"
        )
        executor_sha = _sha(
            _ROOT / "tests/elastic/rail_balance_source_round_executor.py"
        )
        binding = {
            "manifest_path": str(source_manifest_path.resolve()),
            "manifest_raw_sha256": source_manifest_raw_sha,
            "manifest_canonical_sha256": source_manifest_canonical_sha,
            "plan_path": str(plan_path.resolve()),
            "plan_raw_sha256": plan_raw_sha,
            "plan_canonical_sha256": source_coordinator.canonical_sha256(plan),
            "coordinator_code_sha256": coordinator_sha,
            "source_round_executor_sha256": executor_sha,
            "control_commit": _PARENT_COMMIT,
            "parent_commit": _PARENT_COMMIT,
        }
        all_blocks = [*self.parent_blocks]
        all_blocks.extend(
            block for candidate in candidates for block in candidate["benchmark_blocks"]
        )
        for block in all_blocks:
            coordinator_path = Path(block["coordinator_record"])
            coordinator = json.loads(coordinator_path.read_text(encoding="utf-8"))
            plan_block = plan["blocks"][block["global_ordinal"]]
            coordinator.update(
                {
                    "source_round_manifest_raw_sha256": source_manifest_raw_sha,
                    "source_round_plan_raw_sha256": plan_raw_sha,
                    "source_round_plan_ordinal": block["global_ordinal"],
                    "source_round_run_id": plan_block["run_id"],
                    "source_round_coordinator_sha256": coordinator_sha,
                    "source_round_executor_sha256": executor_sha,
                    "source_round_control_commit": _PARENT_COMMIT,
                    "source_round_parent_commit": _PARENT_COMMIT,
                }
            )
            _write_json(coordinator_path, coordinator)
            block["coordinator_record_raw_sha256"] = _sha(coordinator_path)
        return binding

    def manifest(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if self.parent_blocks is None:
            raise ValueError("parent blocks have not been created")
        binding = self._materialize_source_round_binding(candidates)
        return {
            "schema_version": 1,
            "round_id": "round-1",
            "comparison_axis": "source_version",
            "parent": _PARENT,
            "parent_campaign_candidate": deepcopy(self.parent_identity),
            "parent_ref": deepcopy(self.parent_ref),
            "source_round_binding": binding,
            "frozen_contract_sha256": _CONTRACT_SHA,
            "measurement_contract": _measurement_contract(),
            "noise_policy": deepcopy(round_eval.FORMAL_NOISE_POLICY),
            "parent_blocks": deepcopy(self.parent_blocks),
            "candidates": candidates,
        }

    def evaluate(self, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], Path]:
        path = self.root / "round.json"
        _write_json(path, self.manifest(candidates))
        return round_eval.evaluate_round(path), path


def _verdicts(result: dict[str, Any]) -> dict[str, Any]:
    return {row["id"]: row for row in result["candidates"]}


def _expect_round_error(manifest: dict[str, Any]) -> str:
    try:
        round_eval._validate_round_manifest(manifest)
    except round_eval.RoundError as error:
        return str(error)
    raise AssertionError("unsafe formal round manifest was accepted")


def test_three_candidate_global_4_plus_4n_schedule_and_attacks() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory), candidate_count=3)
        candidates = [
            fixture.candidate(
                "candidate-a",
                parent_centers=[1_000_000] * 4,
                candidate_centers=[700_000] * 4,
            ),
            fixture.candidate(
                "candidate-b",
                parent_centers=[1_000_000] * 4,
                candidate_centers=[800_000] * 4,
            ),
            fixture.candidate(
                "candidate-c",
                parent_centers=[1_000_000] * 4,
                candidate_centers=[900_000] * 4,
            ),
        ]
        manifest = fixture.manifest(candidates)
        labeled = [
            (block["global_ordinal"], "parent", block["ordinal"])
            for block in manifest["parent_blocks"]
        ]
        labeled.extend(
            (block["global_ordinal"], candidate["id"], block["ordinal"])
            for candidate in candidates
            for block in candidate["benchmark_blocks"]
        )
        assert [(variant, repetition) for _, variant, repetition in sorted(labeled)] == [
            ("parent", 0),
            ("candidate-a", 0),
            ("candidate-b", 0),
            ("candidate-c", 0),
            ("candidate-a", 1),
            ("candidate-b", 1),
            ("candidate-c", 1),
            ("parent", 1),
            ("candidate-a", 2),
            ("candidate-b", 2),
            ("candidate-c", 2),
            ("parent", 2),
            ("parent", 3),
            ("candidate-a", 3),
            ("candidate-b", 3),
            ("candidate-c", 3),
        ]
        path = fixture.root / "round-three.json"
        _write_json(path, manifest)
        result = round_eval.evaluate_round(path)
        assert result["leader_after"] == "candidate-a"
        assert result["round_wide_evidence"]["control_plane_audit"][
            "block_count"
        ] == 16
        assert all(
            tuple(block["role"] for block in row["analysis"]["blocks"])
            == round_eval._ABBA_BAAB
            for row in result["candidates"]
        )

        duplicated = deepcopy(manifest)
        duplicated["candidates"][1]["benchmark_blocks"][0]["global_ordinal"] = (
            duplicated["candidates"][0]["benchmark_blocks"][0]["global_ordinal"]
        )
        assert "global_ordinal" in _expect_round_error(duplicated)
        wrong_order = deepcopy(manifest)
        wrong_order["candidates"][2]["benchmark_blocks"][0:2] = reversed(
            wrong_order["candidates"][2]["benchmark_blocks"][0:2]
        )
        assert "ordinal" in _expect_round_error(wrong_order)


def test_source_round_binding_rejects_unrelated_candidate_and_plan_tamper() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        first = fixture.candidate(
            "candidate-one",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[800_000] * 4,
        )
        second = fixture.candidate(
            "candidate-two",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[850_000] * 4,
        )
        manifest = fixture.manifest([first, second])
        missing_binding = deepcopy(manifest)
        del missing_binding["source_round_binding"]
        assert "fields" in _expect_round_error(missing_binding)

        wrong_executor = deepcopy(manifest)
        wrong_executor["source_round_binding"]["source_round_executor_sha256"] = (
            "0" * 64
        )
        assert "executor code hash" in _expect_round_error(wrong_executor)

        source_manifest_path = Path(
            manifest["source_round_binding"]["manifest_path"]
        )
        source_plan_path = Path(manifest["source_round_binding"]["plan_path"])
        original_source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        original_source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
        mismatched_source_manifest = deepcopy(original_source_manifest)
        mismatched_source_manifest["frozen_execution_contract"]["algorithm"][
            "max_two_hop_percent"
        ] = 24
        _write_json(source_manifest_path, mismatched_source_manifest, sort_keys=False)
        mismatched_source_plan = deepcopy(original_source_plan)
        mismatched_source_plan["manifest_raw_sha256"] = _sha(source_manifest_path)
        mismatched_source_plan["manifest_canonical_sha256"] = (
            source_coordinator.canonical_sha256(mismatched_source_manifest)
        )
        mismatched_contract_sha = source_coordinator.canonical_sha256(
            mismatched_source_manifest["frozen_execution_contract"]
        )
        mismatched_source_manifest["parent"][
            "frozen_execution_contract_sha256"
        ] = mismatched_contract_sha
        for source_candidate in mismatched_source_manifest["candidates"]:
            source_candidate[
                "frozen_execution_contract_sha256"
            ] = mismatched_contract_sha
        _write_json(source_manifest_path, mismatched_source_manifest, sort_keys=False)
        mismatched_source_plan["manifest_raw_sha256"] = _sha(source_manifest_path)
        mismatched_source_plan["manifest_canonical_sha256"] = (
            source_coordinator.canonical_sha256(mismatched_source_manifest)
        )
        mismatched_source_plan["frozen_execution_contract_sha256"] = (
            mismatched_contract_sha
        )
        for plan_block in mismatched_source_plan["blocks"]:
            plan_block["frozen_execution_contract_sha256"] = mismatched_contract_sha
        _write_json(source_plan_path, mismatched_source_plan)
        mismatched_contract = deepcopy(manifest)
        mismatched_contract["source_round_binding"].update(
            {
                "manifest_raw_sha256": _sha(source_manifest_path),
                "manifest_canonical_sha256": source_coordinator.canonical_sha256(
                    mismatched_source_manifest
                ),
                "plan_raw_sha256": _sha(source_plan_path),
                "plan_canonical_sha256": source_coordinator.canonical_sha256(
                    mismatched_source_plan
                ),
            }
        )
        assert "measurement contract" in _expect_round_error(mismatched_contract)
        _write_json(source_manifest_path, original_source_manifest, sort_keys=False)
        _write_json(source_plan_path, original_source_plan)

        unrelated = deepcopy(manifest)
        unrelated["candidates"][0]["source_ref"]["commit"] = "f" * 40
        assert "candidate 0 commit" in _expect_round_error(unrelated)

        block = manifest["candidates"][0]["benchmark_blocks"][0]
        coordinator_path = Path(block["coordinator_record"])
        coordinator = json.loads(coordinator_path.read_text(encoding="utf-8"))
        forged = deepcopy(coordinator)
        forged["source_round_plan_raw_sha256"] = "0" * 64
        _write_json(coordinator_path, forged)
        block["coordinator_record_raw_sha256"] = _sha(coordinator_path)
        round_path = fixture.root / "forged-block-round.json"
        _write_json(round_path, manifest)
        result = round_eval.evaluate_round(round_path)
        assert result["leader_after"] == _PARENT
        assert result["round_wide_evidence"]["failure_reason"] == "round_control_plane_invalid"
        _write_json(coordinator_path, coordinator)
        block["coordinator_record_raw_sha256"] = _sha(coordinator_path)

        plan_path = Path(manifest["source_round_binding"]["plan_path"])
        original_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        forged_plan = deepcopy(original_plan)
        forged_plan["blocks"][1]["jit_dir"] = forged_plan["blocks"][0][
            "jit_dir"
        ]
        _write_json(plan_path, forged_plan)
        manifest["source_round_binding"]["plan_raw_sha256"] = _sha(plan_path)
        manifest["source_round_binding"]["plan_canonical_sha256"] = (
            round_eval.canonical_sha256(forged_plan)
        )
        assert "plan block 1" in _expect_round_error(manifest)

        _write_json(plan_path, original_plan)
        manifest["source_round_binding"]["plan_raw_sha256"] = _sha(plan_path)
        manifest["source_round_binding"]["plan_canonical_sha256"] = (
            round_eval.canonical_sha256(original_plan)
        )
        plan_path.write_text(
            plan_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
        assert "plan raw hash" in _expect_round_error(manifest)


def test_distinct_source_same_algorithm_promotes_absolute_fastest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        fast = fixture.candidate(
            "candidate-fast",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[800_000] * 4,
        )
        modest = fixture.candidate(
            "candidate-modest",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[900_000] * 4,
        )
        result, _ = fixture.evaluate([fast, modest])
        assert result["comparison_axis"] == "source_version"
        assert result["leader_after"] == "candidate-fast"
        assert result["promotion"]["promoted"] is True
        assert result["source_leader_promotion_allowed"] is True
        rows = _verdicts(result)
        assert rows["candidate-fast"]["analysis"]["verdict"] == "promoted_absolute_fastest"
        assert rows["candidate-modest"]["analysis"]["verdict"] == "significant_not_absolute_fastest"
        analysis = rows["candidate-fast"]["analysis"]
        assert analysis["pooled"]["parent"]["count"] == 400
        assert analysis["pooled"]["candidate"]["count"] == 400
        assert len(analysis["blocks"]) == 8
        assert all(len(block["samples_ns"]) == 100 for block in analysis["blocks"])
        assert all(rows["candidate-fast"]["gates"][kind]["passed"] for kind in ("compile", "correctness", "sanitizer"))


def test_same_commit_algorithm_mode_cannot_enter_source_round() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        first = fixture.candidate(
            "candidate-mode-only",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[800_000] * 4,
            commit=_PARENT_COMMIT,
        )
        second = fixture.candidate(
            "candidate-real-source",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[850_000] * 4,
        )
        try:
            round_eval._validate_round_manifest(fixture.manifest([first, second]))
        except (round_eval.RoundError, source_coordinator.SourceRoundError) as error:
            assert "commit" in str(error)
        else:
            raise AssertionError("same-commit mode comparison entered source round")


def test_relative_gain_larger_but_absolute_slower_does_not_win() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        absolute_fast = fixture.candidate(
            "absolute-fast",
            parent_centers=[991_000, 991_000, 1_009_000, 1_009_000],
            candidate_centers=[808_000, 906_000, 808_000, 931_000],
        )
        relative_large = fixture.candidate(
            "relative-large",
            parent_centers=[991_000, 991_000, 1_009_000, 1_009_000],
            candidate_centers=[928_000, 805_000, 822_000, 915_000],
        )
        result, _ = fixture.evaluate([absolute_fast, relative_large])
        assert result["round_wide_evidence"]["parent_baseline_drift_fraction"] < 0.02
        rows = _verdicts(result)
        assert result["leader_after"] != "relative-large"
        assert rows["relative-large"]["analysis"]["pooled"]["candidate"][
            "median"
        ] > rows["absolute-fast"]["analysis"]["pooled"]["candidate"]["median"]
        assert rows["relative-large"]["analysis"]["noise"][
            "median_paired_improvement_fraction"
        ] > rows["absolute-fast"]["analysis"]["noise"][
            "median_paired_improvement_fraction"
        ]


def test_parent_baseline_drift_blocks_all_promotion() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        first = fixture.candidate(
            "candidate-a",
            parent_centers=[1_000_000, 1_100_000, 1_000_000, 1_100_000],
            candidate_centers=[800_000] * 4,
        )
        second = fixture.candidate(
            "candidate-b",
            parent_centers=[1_000_000, 1_100_000, 1_000_000, 1_100_000],
            candidate_centers=[700_000] * 4,
        )
        result, _ = fixture.evaluate([first, second])
        assert result["leader_after"] == _PARENT
        assert result["round_wide_evidence"]["failure_reason"] == "parent_baseline_drift_exceeded"
        assert all("parent_baseline_drift" in row["analysis"]["verdict"] for row in result["candidates"])


def test_top_two_absolute_latency_within_noise_does_not_choose_arbitrarily() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        first = fixture.candidate(
            "candidate-850",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[850_000] * 4,
        )
        second = fixture.candidate(
            "candidate-851",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[851_000] * 4,
        )
        result, _ = fixture.evaluate([first, second])
        assert result["leader_after"] == _PARENT
        assert result["source_leader_promotion_allowed"] is False
        assert result["leader_separation"]["evaluated"] is True
        assert result["leader_separation"]["passed"] is False
        assert all("within_noise" in row["analysis"]["verdict"] for row in result["candidates"])


def test_compile_correctness_and_sanitizer_failures_are_independent_verdicts() -> None:
    for gate in ("compile", "correctness", "sanitizer"):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            failed = fixture.candidate(
                f"failed-{gate}",
                parent_centers=[1_000_000] * 4,
                candidate_centers=[100_000] * 4,
                gate_failure=gate,
            )
            good = fixture.candidate(
                f"good-{gate}",
                parent_centers=[1_000_000] * 4,
                candidate_centers=[800_000] * 4,
            )
            result, _ = fixture.evaluate([failed, good])
            rows = _verdicts(result)
            assert rows[f"failed-{gate}"]["analysis"]["status"] == f"{gate}_failed"
            assert result["candidate_failure_count"] == 1
            assert result["status"] == "complete_with_failures"
            assert result["leader_after"] == f"good-{gate}"


def test_report_or_final_commit_tamper_blocks_round() -> None:
    for tamper in ("report", "finalized"):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            bad = fixture.candidate(
                f"bad-{tamper}",
                parent_centers=[1_000_000] * 4,
                candidate_centers=[700_000] * 4,
            )
            good = fixture.candidate(
                f"good-{tamper}",
                parent_centers=[1_000_000] * 4,
                candidate_centers=[800_000] * 4,
            )
            block = bad["benchmark_blocks"][0]
            root = Path(block["artifact_root_realpath"])
            if tamper == "report":
                path = root / "stages" / block["stage_id"] / "attempt-01/report.json"
            else:
                path = root / "FINALIZED.json"
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            result, _ = fixture.evaluate([bad, good])
            rows = _verdicts(result)
            assert rows[f"bad-{tamper}"]["analysis"]["status"] == "evidence_invalid"
            assert result["leader_after"] == _PARENT
            assert result["round_wide_evidence"]["failure_reason"] == (
                "round_control_plane_invalid"
            )


def test_failed_candidate_coordinator_overlap_blocks_round_and_paths_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        bad = fixture.candidate(
            "bad-coordinator",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[700_000] * 4,
            gate_failure="compile",
        )
        good = fixture.candidate(
            "good-coordinator",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[800_000] * 4,
        )
        block = bad["benchmark_blocks"][1]
        coordinator_path = Path(block["coordinator_record"])
        record = json.loads(coordinator_path.read_text(encoding="utf-8"))
        record["monotonic_start_ns"] = 2
        _write_json(coordinator_path, record)
        block["coordinator_record_raw_sha256"] = _sha(coordinator_path)
        result, _ = fixture.evaluate([bad, good])
        assert _verdicts(result)["bad-coordinator"]["analysis"]["status"] == "compile_failed"
        assert result["leader_after"] == _PARENT
        assert result["round_wide_evidence"]["failure_reason"] == "round_control_plane_invalid"
        assert "overlap" in result["round_wide_evidence"]["control_plane_audit"][
            "failure_reason"
        ]

        broken = deepcopy(bad)
        broken["id"] = "bad-missing-root"
        broken["primary_change"] = "one CUDA source change for bad-missing-root"
        broken["hypothesis"] = "bad-missing-root lowers source-stage latency"
        broken["source_ref"] = deepcopy(bad["source_ref"])
        broken["source_ref"]["ref_id"] = "bad-missing-root"
        broken["benchmark_blocks"][0]["artifact_root_realpath"] = "/no/such/formal-round-root"
        # Manifest-level duplicate source refs are intentionally rejected, so
        # this path isolation check calls the per-candidate analyzer directly.
        try:
            analysis_manifest = fixture.manifest([bad, good])
            round_eval._analyze_candidate(
                analysis_manifest,
                round_eval._load_source_round_binding(analysis_manifest),
                broken,
            )
        except round_eval.CandidateEvidenceError:
            pass
        else:
            raise AssertionError("missing candidate root did not fail closed")


def test_resealed_repeated_samples_and_impossible_dependencies_fail_closed() -> None:
    for attack in ("repeated-samples", "skipped-dependency"):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            attacked = fixture.candidate(
                f"attacked-{attack}",
                parent_centers=[1_000_000] * 4,
                candidate_centers=[700_000] * 4,
            )
            good = fixture.candidate(
                f"good-{attack}",
                parent_centers=[1_000_000] * 4,
                candidate_centers=[800_000] * 4,
            )
            manifest = fixture.manifest([attacked, good])
            block = attacked["benchmark_blocks"][0]
            root = Path(block["artifact_root_realpath"])
            if attack == "repeated-samples":
                report_path = (
                    root
                    / "stages"
                    / block["stage_id"]
                    / "attempt-01"
                    / block["artifact"]
                )
                report = json.loads(report_path.read_text(encoding="utf-8"))
                repeated = deepcopy(report["steady"][0])
                report["steady"] = [
                    {**deepcopy(repeated), "category_index": index}
                    for index in range(100)
                ]
                report["steady_summary"]["stage_truth_ns"] = round_eval._stats(
                    [repeated["stage_truth_ns"]] * 100
                )
                _write_json(report_path, report)
            else:
                raw_path = root / "manifest.raw.json"
                raw_manifest = json.loads(raw_path.read_text(encoding="utf-8"))
                benchmark = next(
                    stage
                    for stage in raw_manifest["stages"]
                    if stage["id"] == block["stage_id"]
                )
                dependency = benchmark["depends_on"][0]
                dependency_plan = next(
                    stage
                    for stage in raw_manifest["stages"]
                    if stage["id"] == dependency
                )
                dependency_plan["enabled"] = False
                dependency_plan["skip_reason"] = "synthetic impossible dependency attack"
                _write_json(raw_path, raw_manifest, sort_keys=False)
                campaign_path = root / "result.json"
                campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
                dependency_result = next(
                    stage for stage in campaign["stages"] if stage["id"] == dependency
                )
                dependency_result.update(
                    {
                        "status": "skipped",
                        "reason": "disabled: synthetic impossible dependency attack",
                        "attempts": [],
                    }
                )
                _write_json(campaign_path, campaign)
            _reseal_block(block)
            path = fixture.root / f"round-{attack}.json"
            _write_json(path, manifest)
            result = round_eval.evaluate_round(path)
            assert result["leader_after"] == _PARENT
            assert result["source_leader_promotion_allowed"] is False
            assert result["round_wide_evidence"]["failure_reason"] == (
                "round_control_plane_invalid"
            )


def test_resealed_cross_block_report_and_raw_sample_copy_fail_closed() -> None:
    for attack in ("whole-report", "raw-samples"):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            attacked = fixture.candidate(
                f"attacked-{attack}",
                parent_centers=[1_000_000] * 4,
                candidate_centers=[700_000] * 4,
            )
            good = fixture.candidate(
                f"good-{attack}",
                parent_centers=[1_000_000] * 4,
                candidate_centers=[800_000] * 4,
            )
            manifest = fixture.manifest([attacked, good])
            source_block, target_block = attacked["benchmark_blocks"][:2]

            def report_path(block: dict[str, Any]) -> Path:
                return (
                    Path(block["artifact_root_realpath"])
                    / "stages"
                    / block["stage_id"]
                    / "attempt-01"
                    / block["artifact"]
                )

            source_report = json.loads(
                report_path(source_block).read_text(encoding="utf-8")
            )
            target_path = report_path(target_block)
            if attack == "whole-report":
                attacked_report = source_report
            else:
                attacked_report = json.loads(target_path.read_text(encoding="utf-8"))
                for field in ("warm", "steady", "steady_summary"):
                    attacked_report[field] = deepcopy(source_report[field])
            _write_json(target_path, attacked_report)
            _reseal_block(target_block)

            path = fixture.root / f"round-cross-block-{attack}.json"
            _write_json(path, manifest)
            result = round_eval.evaluate_round(path)
            assert result["leader_after"] == _PARENT
            assert result["source_leader_promotion_allowed"] is False
            assert result["round_wide_evidence"]["failure_reason"] == (
                "round_control_plane_invalid"
            )


def test_failed_candidate_later_block_requires_complete_sha256_tree() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        failed = fixture.candidate(
            "failed-tree",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[700_000] * 4,
            gate_failure="compile",
        )
        good = fixture.candidate(
            "good-tree",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[800_000] * 4,
        )
        manifest = fixture.manifest([failed, good])
        later_block = failed["benchmark_blocks"][1]
        (Path(later_block["artifact_root_realpath"]) / "campaign.log").unlink()
        path = fixture.root / "round-missing-later-file.json"
        _write_json(path, manifest)
        result = round_eval.evaluate_round(path)
        assert _verdicts(result)["failed-tree"]["analysis"]["status"] == (
            "compile_failed"
        )
        assert result["leader_after"] == _PARENT
        assert result["round_wide_evidence"]["failure_reason"] == (
            "round_control_plane_invalid"
        )
        assert "SHA256SUMS" in result["round_wide_evidence"][
            "control_plane_audit"
        ]["failure_reason"]


def test_self_consistent_but_wrong_leaderboard_derivation_blocks_round() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        first = fixture.candidate(
            "candidate-entry-a",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[700_000] * 4,
        )
        second = fixture.candidate(
            "candidate-entry-b",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[800_000] * 4,
        )
        manifest = fixture.manifest([first, second])
        block = first["benchmark_blocks"][0]
        finalized_path = Path(block["artifact_root_realpath"]) / "FINALIZED.json"
        finalized = json.loads(finalized_path.read_text(encoding="utf-8"))
        finalized["leaderboard_entry_id"] = "0" * 64
        _write_json(finalized_path, finalized)
        block["finalized_raw_sha256"] = _sha(finalized_path)
        coordinator_path = Path(block["coordinator_record"])
        coordinator = json.loads(coordinator_path.read_text(encoding="utf-8"))
        coordinator["campaign_finalized_raw_sha256"] = _sha(finalized_path)
        _write_json(coordinator_path, coordinator)
        block["coordinator_record_raw_sha256"] = _sha(coordinator_path)
        path = fixture.root / "round-wrong-entry-id.json"
        _write_json(path, manifest)
        result = round_eval.evaluate_round(path)
        assert result["leader_after"] == _PARENT
        assert result["round_wide_evidence"]["failure_reason"] == (
            "round_control_plane_invalid"
        )
        assert "leaderboard entry id" in result["round_wide_evidence"][
            "control_plane_audit"
        ]["failure_reason"]


def test_fixed_noise_policy_and_real_production_campaign_schema_are_enforced() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        first = fixture.candidate(
            "candidate-one",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[900_000] * 4,
        )
        second = fixture.candidate(
            "candidate-two",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[850_000] * 4,
        )
        manifest = fixture.manifest([first, second])
        manifest["noise_policy"]["minimum_winning_pairs"] = 1
        try:
            round_eval._validate_round_manifest(manifest)
        except round_eval.RoundError as error:
            assert "noise_policy" in str(error)
        else:
            raise AssertionError("post-hoc weak noise policy was accepted")
        campaign_root = Path(first["benchmark_blocks"][0]["artifact_root_realpath"])
        production_manifest = json.loads(
            (campaign_root / "manifest.raw.json").read_text(encoding="utf-8")
        )
        validate_manifest(production_manifest)
        assert len(production_manifest["frozen_contract"]) == 10
        assert len(
            [
                stage
                for stage in production_manifest["stages"]
                if stage["kind"] == "sanitizer" and stage.get("enabled", True)
            ]
        ) == 4


def test_toolchain_timestamp_is_normalized_but_static_drift_fails_closed() -> None:
    first_snapshot = {**deepcopy(_TOOLCHAIN), "created_utc": "first"}
    second_snapshot = {**deepcopy(_TOOLCHAIN), "created_utc": "second"}
    assert round_eval.canonical_sha256(
        round_eval._normalized_toolchain(first_snapshot)
    ) == round_eval.canonical_sha256(
        round_eval._normalized_toolchain(second_snapshot)
    )
    static_drift = deepcopy(second_snapshot)
    static_drift["target"] = "different-static-toolchain"
    assert round_eval.canonical_sha256(
        round_eval._normalized_toolchain(static_drift)
    ) != _TOOLCHAIN_SHA

    with tempfile.TemporaryDirectory() as directory:
        fixture = Fixture(Path(directory))
        attacked = fixture.candidate(
            "toolchain-drift",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[700_000] * 4,
        )
        good = fixture.candidate(
            "toolchain-good",
            parent_centers=[1_000_000] * 4,
            candidate_centers=[800_000] * 4,
        )
        manifest = fixture.manifest([attacked, good])
        block = attacked["benchmark_blocks"][0]
        toolchain_path = (
            Path(block["artifact_root_realpath"])
            / "environment"
            / "toolchain.json"
        )
        toolchain = json.loads(toolchain_path.read_text(encoding="utf-8"))
        toolchain["target"] = "different-static-toolchain"
        _write_json(toolchain_path, toolchain)
        _reseal_block(block)
        path = fixture.root / "round-toolchain-static-drift.json"
        _write_json(path, manifest)
        result = round_eval.evaluate_round(path)
        assert result["leader_after"] == _PARENT
        assert result["source_leader_promotion_allowed"] is False
        assert result["round_wide_evidence"]["failure_reason"] == (
            "round_control_plane_invalid"
        )


def test_strict_json_manifest_symlink_and_atomic_no_clobber() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
        nonfinite = root / "nonfinite.json"
        nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
        for path in (duplicate, nonfinite):
            try:
                round_eval._read_strict_json(path)
            except round_eval.RoundError:
                pass
            else:
                raise AssertionError("unsafe JSON was accepted")
        target = root / "target.json"
        _write_json(target, {"schema_version": 1})
        link = root / "round-link.json"
        link.symlink_to(target)
        try:
            round_eval.evaluate_round(link)
        except round_eval.RoundError:
            pass
        else:
            raise AssertionError("symlink round manifest was followed")
        output = root / "result.json"
        _write_json(output, {"old": True})
        try:
            round_eval._atomic_json(output, {"new": True}, overwrite=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("atomic no-clobber overwrote an existing verdict")
        assert json.loads(output.read_text(encoding="utf-8")) == {"old": True}
        try:
            round_eval._atomic_json(root / "nan.json", {"bad": math.nan})
        except round_eval.RoundError:
            pass
        else:
            raise AssertionError("non-finite output was accepted")


def test_cli_no_clobber_survives_create_between_parse_and_publish() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "manifest.json"
        output = root / "output.json"
        _write_json(manifest, {"placeholder": True})
        original = round_eval.evaluate_round

        def racing_evaluator(_: Path) -> dict[str, Any]:
            _write_json(output, {"racer": True})
            return {"status": "complete", "leader_before": "a", "leader_after": "b"}

        round_eval.evaluate_round = racing_evaluator
        try:
            try:
                round_eval.main(
                    ("--manifest", str(manifest), "--output", str(output))
                )
            except FileExistsError:
                pass
            else:
                raise AssertionError("CLI clobbered a concurrently created verdict")
        finally:
            round_eval.evaluate_round = original
        assert json.loads(output.read_text(encoding="utf-8")) == {"racer": True}


def test_module_import_is_stdlib_only() -> None:
    code = (
        "import sys; import rail_balance_campaign_round; "
        "assert 'torch' not in sys.modules; assert 'deep_ep' not in sys.modules"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parent)
    subprocess.run(
        (sys.executable, "-B", "-c", code),
        cwd=_ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    tests = sorted(
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} formal source-round CPU contracts")
