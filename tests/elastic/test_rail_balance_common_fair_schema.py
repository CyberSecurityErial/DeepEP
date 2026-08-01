"""CPU-only adversarial tests for the COMMON_FAIR manifest contract."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import rail_balance_common_fair_schema as schema


_THIS_FILE = Path(__file__)
_PROGRAM = _THIS_FILE.with_name("rail_balance_common_fair_schema.py")
_MANIFEST = (
    _THIS_FILE.with_name("experiments") / "rail_balance_common_fair_design_v1.json"
)


def _expect_error(function: Callable[..., Any], *args: Any) -> str:
    try:
        function(*args)
    except schema.CommonFairError as error:
        return str(error)
    raise AssertionError("unsafe COMMON_FAIR input was accepted")


def _binding(status: str) -> dict[str, object]:
    return {
        "status": status,
        "path": None,
        "sha256": None,
        "sha256_kind": schema.RAW_FILE_SHA256,
    }


def _route_binding(status: str) -> dict[str, object]:
    return {
        "status": status,
        "path": None,
        "raw_sha256": None,
        "canonical_sha256": None,
    }


def _source(
    status: str,
    commit: str | None = None,
    tree: str | None = None,
    submodules: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "commit": commit,
        "tree": tree,
        "submodules": [] if submodules is None else submodules,
    }


def _api_bindings(system_id: str) -> list[dict[str, object]]:
    return [
        {
            "phase": "route_prepare_complete",
            "separability": (
                "NOT_SEPARABLE" if system_id in {"nccl-ep", "uccl-ep"} else "REQUIRED"
            ),
            "included_calls": ["prepare_route"],
            "completion_primitive": "host_route_ready",
            "primary_metric": False,
        },
        {
            "phase": "dispatch_complete",
            "separability": "REQUIRED",
            "included_calls": ["dispatch", "dispatch_completion_wait"],
            "completion_primitive": "cuda_event_after_public_completion",
            "primary_metric": True,
        },
        {
            "phase": "combine_complete",
            "separability": "REQUIRED",
            "included_calls": ["combine", "combine_completion_wait"],
            "completion_primitive": "cuda_event_after_public_completion",
            "primary_metric": True,
        },
        {
            "phase": "roundtrip_complete",
            "separability": "REQUIRED",
            "included_calls": ["dispatch", "identity_expert", "combine"],
            "completion_primitive": "cuda_event_after_public_completion",
            "primary_metric": False,
        },
    ]


def _resources(frozen: bool = False) -> dict[str, object]:
    status = "FROZEN" if frozen else "NOT_FROZEN"
    return {
        "resource_track": "MATCHED_RESOURCE",
        "sm_budget_status": status,
        "sm_budget_per_rank": 114 if frozen else None,
        "qp_budget_status": status,
        "qp_budget_per_rank": 8 if frozen else None,
        "nic_set_status": status,
        "nic_ids": ["rail0", "rail1"] if frozen else [],
        "cpu_budget_status": status,
        "cpu_cores_per_rank": 4 if frozen else None,
        "proxy_cpu_budget_status": status,
        "proxy_cpu_cores_per_rank": 2 if frozen else None,
        "device_memory_budget_status": status,
        "extra_device_memory_bytes_per_rank": 1 << 30 if frozen else None,
    }


def _system(system_id: str) -> dict[str, object]:
    sources: dict[str, dict[str, object]] = {
        "deepep-clean": _source(
            "PINNED",
            "dd758caf451848bd150e1046af3d0a73e5fff38d",
            "b385fecc3105afac74e0f7c0bf6a90b15bf9b336",
            [
                {
                    "path": "third-party/fmt",
                    "commit": "a4c7e17133ee9cb6a2f45545f6e974dd3c393efa",
                    "tree": "21478d0a3998e3dfef6accc73ba289be868bf835",
                }
            ],
        ),
        "deepep-off": _source("AWAITING_TUNING_FREEZE"),
        "nccl-ep": _source(
            "PINNED",
            "63cf786b015b2b6bff6cf263461621acf584bd18",
            "22f0c4ad56f505c4b47dd7a8f0a3c4f93efed4ce",
        ),
        "railbalance-best": _source("AWAITING_TUNING_FREEZE"),
        "uccl-ep": _source(
            "PINNED",
            "61ee42402819cabba3ac2a56dd4addec3363976c",
            "0a9acaafb8fd24cde14b76ee01f0ab5a1c4c4161",
        ),
    }
    build_status = (
        "BLOCKED_BUILD_CONTRACT_INCOMPLETE"
        if system_id in {"nccl-ep", "uccl-ep"}
        else "NOT_FROZEN"
    )
    return {
        "id": system_id,
        "role": schema.SYSTEM_ROLES[system_id],
        "source": sources[system_id],
        "build_recipe": _binding(build_status),
        "runtime_closure": _binding("NOT_BUILT"),
        "adapter": _binding("NOT_IMPLEMENTED"),
        "execution_config": _binding("NOT_FROZEN"),
        "algorithm_mode": schema.SYSTEM_ALGORITHM_MODES[system_id],
        "candidate_id": (
            None
            if system_id == "railbalance-best"
            else schema.FIXED_CANDIDATE_IDS[system_id]
        ),
        "backend_modes": ["HT", "LL"],
        "api_bindings": _api_bindings(system_id),
        "resources": _resources(),
    }


def _case(case_id: str) -> dict[str, object]:
    scope, mode, nodes, ranks_per_node, world_size = schema.CASE_SPECS[case_id]
    tokens = 128 if mode == "LL" else 4096
    return {
        "id": case_id,
        "mode": mode,
        "claim_scope": scope,
        "node_count": nodes,
        "ranks_per_node": ranks_per_node,
        "world_size": world_size,
        "tokens_per_rank": tokens,
        "num_experts": 256,
        "hidden": 7168,
        "top_k": 8,
        "dtypes": {
            "payload": "BF16",
            "dispatch": "BF16",
            "combine": "BF16",
            "topk_idx": "INT64",
            "topk_weights": "FP32",
        },
        "expert_owner_policy": "CONTIGUOUS_EQUAL",
        "routes": [
            {
                "pattern": pattern,
                "binding": _route_binding("NOT_FROZEN"),
                "format": (
                    "rail-balance-logical-route-v1"
                    if scope == "DERIVED_SCALE_DOWN_CORRECTNESS"
                    else "rail-balance-logical-route-compact-v2"
                ),
                "per_rank_logical_shape": [tokens, 8],
                "weight_policy": "PRESERVE_INPUT_FP32",
                "mask_policy": "NO_DROPPED_ROUTES",
                "semantic_scope": (
                    schema.DERIVED_ROUTE_SEMANTIC_SCOPES[pattern]
                    if scope == "DERIVED_SCALE_DOWN_CORRECTNESS"
                    else schema.CONFIRMATORY_ROUTE_SEMANTIC_SCOPES[pattern]
                ),
            }
            for pattern in schema.ROUTE_PATTERNS
        ],
        "payload_generator": {
            "status": "PINNED",
            "id": "rail-balance-deterministic-bf16-v1",
            "code_path": "tests/elastic/rail_balance_common_fair_payload.py",
            "code_sha256": (
                "07db5033a3861c7d5416aca5081d310d" "fdbc685d7d85d56d2b00f011ea67b781"
            ),
            "code_sha256_kind": schema.RAW_FILE_SHA256,
            "seed": 20260801,
        },
        "stage_limit": (
            "CORRECTNESS" if scope == "DERIVED_SCALE_DOWN_CORRECTNESS" else "BENCHMARK"
        ),
        "future_claim_eligible": False,
    }


def design_manifest() -> dict[str, Any]:
    case_ids = sorted(schema.CASE_SPECS)
    system_ids = list(schema.SYSTEM_IDS)
    return {
        "schema_version": 1,
        "experiment_id": "rail-balance-common-fair-v1",
        "track": "COMMON_FAIR",
        "manifest_phase": "DESIGN_ONLY",
        "systems": [_system(system_id) for system_id in system_ids],
        "cases": [_case(case_id) for case_id in case_ids],
        "matrix": [
            {
                "row_id": f"{case_id}--{route_pattern}--{system_id}",
                "case_id": case_id,
                "route_pattern": route_pattern,
                "system_id": system_id,
                "initial_result_code": "NOT_RUN",
            }
            for case_id in case_ids
            for route_pattern in schema.ROUTE_PATTERNS
            for system_id in system_ids
        ],
        "measurement": {
            "timing_api": "CUDA_EVENT",
            "barrier_location": "OUTSIDE_TIMED_WINDOW",
            "end_event_synchronize_location": "OUTSIDE_TIMED_WINDOW",
            "warmup_iterations": 10,
            "steady_iterations": 100,
            "independent_process_repeats": 5,
            "minimum_order_blocks": 10,
            "order_schedule": {
                "status": "NOT_FROZEN",
                "pairs": [],
                "blocks": [],
                "sha256": None,
            },
            "rank_aggregation": "PER_ITERATION_MAX",
            "raw_rank_samples_required": True,
            "native_rank_mean_forbidden": True,
            "raw_sample_fields": list(schema.RAW_SAMPLE_FIELDS),
            "summary_statistics": ["CV", "MAD", "P50", "P95", "P99"],
            "paired_effect": "LOG_LATENCY_RATIO",
            "confidence_interval": "BLOCK_BOOTSTRAP_95_PERCENT",
            "multiple_testing": "HOLM_PRIMARY_FAMILY",
            "aa_noise_gate": "CI_EXCLUDES_1_AND_ABS_LOG_RATIO_EXCEEDS_AA_P95",
            "outlier_policy": "WHOLE_BLOCK_FAIL_NO_SINGLE_SAMPLE_DELETION",
            "fixed_input_required": True,
            "fixed_stream_required": True,
            "fixed_gpu_clocks_or_block_invalid": True,
            "environment_snapshot_required": True,
        },
        "correctness": {
            "reference": "CANONICAL_PYTORCH_CPU_EXACT_METADATA",
            "reference_binding": _binding("NOT_FROZEN"),
            "relative_error_operator": "STRICTLY_LESS_THAN",
            "relative_error_threshold": 1e-3,
            "relative_error_formula": (
                "ABS(ACTUAL-REFERENCE)/MAX(ABS(REFERENCE),DENOMINATOR_FLOOR)"
            ),
            "relative_error_denominator_floor": 1e-12,
            "nonfinite_output_policy": "REJECT",
            "record_max_absolute_error": True,
            "record_max_relative_error": True,
            "record_first_error": True,
            "integer_metadata_exact": True,
            "determinism_repetitions": 3,
            "random_seeds": [0, 1, 17, 20260801],
            "special_inputs": [
                "empty-expert",
                "finite-extremes",
                "nonaligned-shape",
                "repeated-expert-ids",
                "zero-route-weights",
                "zeros",
            ],
            "coverage_classes": [
                "boundary-shape",
                "common-shape",
                "maximum-declared-shape",
                "minimum-declared-shape",
                "nonaligned-shape",
            ],
            "declared_dtypes_layouts_exhaustive": True,
        },
        "failure_policy": {
            "result_codes": list(schema.RESULT_CODES),
            "performance_result_codes": ["PASSED_BENCHMARK"],
            "raw_sample_result_codes": ["RAW_SAMPLES_COMPLETE_UNVERIFIED"],
            "required_result_fields": list(schema.REQUIRED_RESULT_FIELDS),
            "failed_results_must_not_carry_promotable_metrics": True,
            "evidence_missing_if_matrix_row_absent": True,
            "results_require_external_verification_before_promotion": True,
        },
        "preregistration": {
            "environment_binding": _binding("NOT_FROZEN"),
            "analysis_plan_binding": _binding("NOT_FROZEN"),
            "tuning_partition_binding": _binding("NOT_FROZEN"),
            "confirmatory_partition_binding": _binding("NOT_FROZEN"),
            "partition_policy": "HASHED_DISJOINT_TRACE_SEED_BLOCK_SETS",
            "confirmatory_sealed_until_candidate_freeze": True,
        },
        "execution_contract": {
            "gpu_execution_status": "NOT_RUN",
            "build_execution_status": "NOT_RUN",
            "result_status": "NOT_RUN",
            "execution_authorized": False,
            "performance_results_present": False,
            "closure_verification_status": "NOT_RUN",
            "executor_status": "NOT_IMPLEMENTED",
        },
    }


def _pin(binding: dict[str, object], identity: str, digest: str) -> None:
    binding.update(
        {
            "status": "PINNED",
            "path": f"bindings/{identity}.json",
            "sha256": digest,
        }
    )


def _pin_route(binding: dict[str, object], identity: str, digest: str) -> None:
    binding.update(
        {
            "status": "PINNED",
            "path": f"routes/{identity}.json",
            "raw_sha256": digest,
            "canonical_sha256": "f" * 64,
        }
    )


def structurally_frozen_manifest() -> dict[str, Any]:
    manifest = design_manifest()
    manifest["manifest_phase"] = "STRUCTURALLY_FROZEN_UNVERIFIED"
    same_source = _source(
        "PINNED",
        "1111111111111111111111111111111111111111",
        "2222222222222222222222222222222222222222",
        [
            {
                "path": "third-party/fmt",
                "commit": "a4c7e17133ee9cb6a2f45545f6e974dd3c393efa",
                "tree": "21478d0a3998e3dfef6accc73ba289be868bf835",
            }
        ],
    )
    for index, system in enumerate(manifest["systems"]):
        assert isinstance(system, dict)
        system_id = str(system["id"])
        if system_id in {"deepep-off", "railbalance-best"}:
            system["source"] = copy.deepcopy(same_source)
            suffix = "same-tree"
            build_digest = "9" * 64
        else:
            suffix = system_id
            build_digest = f"{index + 1}" * 64
        _pin(system["build_recipe"], f"{suffix}-build", build_digest)
        _pin(system["adapter"], f"{suffix}-adapter", "a" * 64)
        _pin(system["runtime_closure"], f"{suffix}-runtime", "b" * 64)
        _pin(
            system["execution_config"],
            f"{system_id}-execution-config",
            f"{index + 5:x}" * 64,
        )
        if system_id == "railbalance-best":
            system["candidate_id"] = "candidate-round-07-c3"
        system["resources"] = _resources(frozen=True)
    for case in manifest["cases"]:
        assert isinstance(case, dict)
        case_id = str(case["id"])
        for route in case["routes"]:
            assert isinstance(route, dict)
            _pin_route(
                route["binding"],
                f"{case_id}-{route['pattern']}-route",
                "c" * 64,
            )
        generator = case["payload_generator"]
        assert isinstance(generator, dict)
        generator["status"] = "PINNED"
        generator["code_path"] = "tests/elastic/rail_balance_common_fair_payload.py"
        generator["code_sha256"] = "d" * 64
        if case["claim_scope"] == "CONFIRMATORY_COMMON_FAIR":
            case["future_claim_eligible"] = True

    reference = manifest["correctness"]["reference_binding"]
    _pin(reference, "correctness-reference", "e" * 64)
    for index, field in enumerate(
        (
            "environment_binding",
            "analysis_plan_binding",
            "tuning_partition_binding",
            "confirmatory_partition_binding",
        )
    ):
        _pin(
            manifest["preregistration"][field],
            field.removesuffix("_binding").replace("_", "-"),
            f"{index + 5:x}" * 64,
        )
    pairs = [
        {"pair_id": pair_id, "system_a": systems[0], "system_b": systems[1]}
        for pair_id, systems in schema.PAIR_SPECS.items()
    ]
    blocks = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        system_a = pair["system_a"]
        system_b = pair["system_b"]
        for ordinal in range(10):
            cycle = "ABBA" if ordinal % 2 == 0 else "BAAB"
            order = (
                [system_a, system_b, system_b, system_a]
                if cycle == "ABBA"
                else [system_b, system_a, system_a, system_b]
            )
            blocks.append(
                {
                    "block_id": f"{pair_id}-block-{ordinal:02d}",
                    "pair_id": pair_id,
                    "cycle": cycle,
                    "order": order,
                }
            )
    schedule = manifest["measurement"]["order_schedule"]
    schedule.update(
        {
            "status": "PINNED",
            "pairs": pairs,
            "blocks": blocks,
            "sha256": hashlib.sha256(
                schema.canonical_bytes({"pairs": pairs, "blocks": blocks})
            ).hexdigest(),
        }
    )
    return manifest


def test_design_and_structurally_frozen_contracts_are_valid_but_not_authorized() -> (
    None
):
    design = design_manifest()
    schema.validate_manifest(design)
    assert len(design["cases"]) == 8
    assert len(design["matrix"]) == 120
    assert all(not case["future_claim_eligible"] for case in design["cases"])
    loaded, raw, raw_sha256, canonical_sha256 = schema.load_manifest(_MANIFEST)
    assert loaded == design
    assert raw == _MANIFEST.read_bytes()
    assert raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert (
        canonical_sha256 == hashlib.sha256(schema.canonical_bytes(design)).hexdigest()
    )

    frozen = structurally_frozen_manifest()
    schema.validate_manifest(frozen)
    assert frozen["execution_contract"] == {
        "gpu_execution_status": "NOT_RUN",
        "build_execution_status": "NOT_RUN",
        "result_status": "NOT_RUN",
        "execution_authorized": False,
        "performance_results_present": False,
        "closure_verification_status": "NOT_RUN",
        "executor_status": "NOT_IMPLEMENTED",
    }
    assert frozen["manifest_phase"] == "STRUCTURALLY_FROZEN_UNVERIFIED"


def test_numeric_bool_and_float_smuggling_is_rejected() -> None:
    attacks: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("schema bool", lambda value: value.update(schema_version=True)),
        (
            "warmup float",
            lambda value: value["measurement"].update(warmup_iterations=10.0),
        ),
        (
            "tokens float",
            lambda value: value["cases"][0].update(tokens_per_rank=4096.0),
        ),
        ("top-k float", lambda value: value["cases"][0].update(top_k=8.0)),
        (
            "logical shape float",
            lambda value: value["cases"][0]["routes"][0].update(
                per_rank_logical_shape=[4096.0, 8]
            ),
        ),
        (
            "determinism float",
            lambda value: value["correctness"].update(determinism_repetitions=3.0),
        ),
        (
            "seed bool",
            lambda value: value["correctness"].update(
                random_seeds=[False, 1, 17, 20260801]
            ),
        ),
    ]
    for label, mutate in attacks:
        manifest = design_manifest()
        mutate(manifest)
        assert _expect_error(schema.validate_manifest, manifest), label


def test_case_matrix_dtype_and_scope_drift_is_rejected() -> None:
    manifest = design_manifest()
    duplicate = copy.deepcopy(manifest["cases"][-1])
    duplicate["id"] = "derived-ll-n1-w4-duplicate"
    manifest["cases"].append(duplicate)
    assert "frozen case" in _expect_error(schema.validate_manifest, manifest)

    manifest = design_manifest()
    manifest["matrix"].pop()
    assert "exact sorted case x route x system product" in _expect_error(
        schema.validate_manifest, manifest
    )

    manifest = design_manifest()
    manifest["cases"][0]["dtypes"]["payload"] = "FP8"
    assert "dtypes changed" in _expect_error(schema.validate_manifest, manifest)

    manifest = design_manifest()
    derived = next(
        case
        for case in manifest["cases"]
        if case["claim_scope"] == "DERIVED_SCALE_DOWN_CORRECTNESS"
    )
    derived["stage_limit"] = "BENCHMARK"
    assert "cannot enter benchmark" in _expect_error(schema.validate_manifest, manifest)

    manifest = design_manifest()
    confirmatory = next(
        case
        for case in manifest["cases"]
        if case["claim_scope"] == "CONFIRMATORY_COMMON_FAIR"
    )
    confirmatory["future_claim_eligible"] = True
    assert "differs from manifest phase" in _expect_error(
        schema.validate_manifest, manifest
    )


def test_matched_resources_and_same_tree_bindings_are_enforced() -> None:
    manifest = structurally_frozen_manifest()
    manifest["systems"][0]["resources"]["sm_budget_per_rank"] = 113
    assert "unmatched sm_budget_per_rank" in _expect_error(
        schema.validate_manifest, manifest
    )

    for field, value in (
        ("sm_budget_per_rank", 0),
        ("qp_budget_per_rank", 0),
        ("cpu_cores_per_rank", 0),
        ("sm_budget_per_rank", 1025),
    ):
        manifest = structurally_frozen_manifest()
        manifest["systems"][0]["resources"][field] = value
        assert _expect_error(schema.validate_manifest, manifest)

    manifest = structurally_frozen_manifest()
    manifest["systems"][0]["resources"]["proxy_cpu_cores_per_rank"] = 5
    assert "exceeds total CPU" in _expect_error(schema.validate_manifest, manifest)

    manifest = structurally_frozen_manifest()
    off = next(system for system in manifest["systems"] if system["id"] == "deepep-off")
    off["build_recipe"]["sha256"] = "f" * 64
    assert "same-tree" in _expect_error(schema.validate_manifest, manifest)

    manifest = structurally_frozen_manifest()
    rail = next(
        system for system in manifest["systems"] if system["id"] == "railbalance-best"
    )
    rail["source"]["tree"] = "f" * 40
    assert "same-tree" in _expect_error(schema.validate_manifest, manifest)

    manifest = structurally_frozen_manifest()
    off = next(system for system in manifest["systems"] if system["id"] == "deepep-off")
    rail = next(
        system for system in manifest["systems"] if system["id"] == "railbalance-best"
    )
    rail["execution_config"]["sha256"] = off["execution_config"]["sha256"]
    assert "distinct config SHA256" in _expect_error(schema.validate_manifest, manifest)


def test_structural_freeze_requires_every_closure_without_authorizing_execution() -> (
    None
):
    manifest = structurally_frozen_manifest()
    manifest["systems"][2]["adapter"] = _binding("NOT_IMPLEMENTED")
    assert "not fully frozen" in _expect_error(schema.validate_manifest, manifest)

    manifest = structurally_frozen_manifest()
    manifest["cases"][0]["routes"][0]["binding"] = _route_binding("NOT_FROZEN")
    assert "route/payload generator" in _expect_error(
        schema.validate_manifest, manifest
    )

    manifest = structurally_frozen_manifest()
    manifest["correctness"]["reference_binding"] = _binding("NOT_FROZEN")
    assert "pinned correctness reference" in _expect_error(
        schema.validate_manifest, manifest
    )

    manifest = structurally_frozen_manifest()
    manifest["preregistration"]["environment_binding"] = _binding("NOT_FROZEN")
    assert "all preregistration bindings" in _expect_error(
        schema.validate_manifest, manifest
    )

    manifest = structurally_frozen_manifest()
    manifest["execution_contract"]["execution_authorized"] = True
    assert "preregistration-only" in _expect_error(schema.validate_manifest, manifest)


def test_order_schedule_is_exact_balanced_abba_baab_and_hash_bound() -> None:
    valid = structurally_frozen_manifest()
    schedule = valid["measurement"]["order_schedule"]
    assert len(schedule["pairs"]) == 4
    assert len(schedule["blocks"]) == 40
    schema.validate_manifest(valid)

    manifest = structurally_frozen_manifest()
    manifest["measurement"]["order_schedule"]["blocks"][1]["cycle"] = "ABBA"
    assert "alternate ABBA/BAAB" in _expect_error(schema.validate_manifest, manifest)

    manifest = structurally_frozen_manifest()
    block = manifest["measurement"]["order_schedule"]["blocks"][0]
    block["order"] = sorted(block["order"])
    assert "differs from its cycle" in _expect_error(schema.validate_manifest, manifest)

    manifest = structurally_frozen_manifest()
    manifest["measurement"]["order_schedule"]["blocks"].pop()
    assert "wrong block count" in _expect_error(schema.validate_manifest, manifest)

    manifest = structurally_frozen_manifest()
    manifest["measurement"]["order_schedule"]["sha256"] = "0" * 64
    assert "SHA256 changed" in _expect_error(schema.validate_manifest, manifest)


def test_binding_paths_and_text_are_restricted_before_a_future_executor() -> None:
    for path in ("../escape.json", "/absolute.json", "-option.json", "bad\nname.json"):
        manifest = structurally_frozen_manifest()
        manifest["systems"][0]["build_recipe"]["path"] = path
        assert _expect_error(schema.validate_manifest, manifest), path

    manifest = structurally_frozen_manifest()
    manifest["systems"][0]["build_recipe"]["path"] = "bad\ud800.json"
    assert _expect_error(schema.validate_manifest, manifest)

    manifest = design_manifest()
    manifest["cases"][0]["routes"][0]["pattern"] = "unknown"
    assert "pattern is not frozen" in _expect_error(schema.validate_manifest, manifest)


def _write(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.write_bytes(payload)
    path.chmod(mode)


def test_loader_rejects_duplicate_nonfinite_encoding_links_and_fifo() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        duplicate = root / "duplicate.json"
        _write(duplicate, b'{"schema_version":1,"schema_version":1}\n')
        assert "duplicate JSON key" in _expect_error(schema.load_manifest, duplicate)

        nonfinite = root / "nonfinite.json"
        _write(nonfinite, b'{"value":NaN}\n')
        assert "non-finite" in _expect_error(schema.load_manifest, nonfinite)

        utf16 = root / "utf16.json"
        _write(utf16, '{"x":1}'.encode("utf-16"))
        assert "invalid JSON" in _expect_error(schema.load_manifest, utf16)

        too_deep = root / "deep.json"
        _write(too_deep, ("[" * 2000 + "0" + "]" * 2000).encode())
        deep_error = _expect_error(schema.load_manifest, too_deep)
        assert "invalid JSON" in deep_error or "manifest root" in deep_error

        long_integer = root / "long-int.json"
        _write(long_integer, ('{"x":' + "1" * 10000 + "}").encode())
        assert "invalid JSON" in _expect_error(schema.load_manifest, long_integer)

        valid = root / "valid.json"
        _write(valid, schema.canonical_bytes(design_manifest()) + b"\n")
        linked = root / "linked.json"
        linked.symlink_to(valid.name)
        assert "safely open" in _expect_error(schema.load_manifest, linked)

        real_parent = root / "real"
        real_parent.mkdir()
        nested = real_parent / "manifest.json"
        _write(nested, schema.canonical_bytes(design_manifest()) + b"\n")
        linked_parent = root / "alias"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        assert "safely open" in _expect_error(
            schema.load_manifest, linked_parent / "manifest.json"
        )

        valid.chmod(0o666)
        assert "group/world writable" in _expect_error(schema.load_manifest, valid)

        hardlink_source = root / "hardlink-source.json"
        _write(hardlink_source, schema.canonical_bytes(design_manifest()) + b"\n")
        hardlink_alias = root / "hardlink-alias.json"
        os.link(hardlink_source, hardlink_alias)
        assert "exactly one hard link" in _expect_error(
            schema.load_manifest, hardlink_source
        )

        fifo = root / "manifest.fifo"
        os.mkfifo(fifo, 0o600)
        assert "regular file" in _expect_error(schema.load_manifest, fifo)


def test_cli_is_check_only_and_module_stays_stdlib_only() -> None:
    imported = {
        node.names[0].name.split(".", 1)[0]
        for node in ast.walk(ast.parse(_PROGRAM.read_text(encoding="utf-8")))
        if isinstance(node, ast.Import)
    } | {
        node.module.split(".", 1)[0]
        for node in ast.walk(ast.parse(_PROGRAM.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module != "__future__"
    }
    assert imported <= {
        "argparse",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "re",
        "stat",
        "typing",
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        _write(path, schema.canonical_bytes(design_manifest()) + b"\n")
        completed = subprocess.run(
            (sys.executable, "-I", "-S", "-B", str(_PROGRAM), "--manifest", str(path)),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["execution_authorized"] is False
        assert result["performance_results_present"] is False
        assert result["build_execution_status"] == "NOT_RUN"
        assert result["gpu_execution_status"] == "NOT_RUN"
        assert result["executor_status"] == "NOT_IMPLEMENTED"
