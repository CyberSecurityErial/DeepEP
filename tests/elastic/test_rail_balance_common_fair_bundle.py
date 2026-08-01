"""CPU-only E2E and fail-closed tests for COMMON_FAIR bundles."""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

import rail_balance_common_fair_bundle as bundle
import rail_balance_common_fair_payload as payload
import rail_balance_common_fair_results as results
import rail_balance_common_fair_route as expanded_route
import rail_balance_common_fair_route_compact as compact_route
import rail_balance_common_fair_schema as schema
import test_rail_balance_common_fair_results as result_fixtures
import test_rail_balance_common_fair_schema as schema_fixtures


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o644)


def _expect_bundle_error(
    function: Callable[..., Any], *args: Any, **kwargs: Any
) -> str:
    try:
        function(*args, **kwargs)
    except bundle.CommonFairBundleError as error:
        return str(error)
    raise AssertionError("invalid bundle input was accepted")


def _rehash_index(index: dict[str, Any]) -> None:
    index["bundle_sha256"] = bundle.bundle_sha256(index)


class CommonFairBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        manifest = schema_fixtures.design_manifest()
        manifest_raw = bundle.canonical_bytes(manifest) + b"\n"
        _write(cls.root / "manifest.json", manifest_raw)

        payload_raw = Path(payload.__file__).read_bytes()
        payload_relative = "tests/elastic/rail_balance_common_fair_payload.py"
        _write(cls.root / payload_relative, payload_raw)
        specification_sha = hashlib.sha256(
            bundle.canonical_bytes(payload.specification_identity())
        ).hexdigest()

        declarations = {
            (case["id"], route["pattern"]): route
            for case in manifest["cases"]
            for route in case["routes"]
        }
        route_entries: list[dict[str, Any]] = []
        for case_id, pattern in sorted(declarations):
            scope, mode, _nodes, _ranks, world_size = schema.CASE_SPECS[case_id]
            if scope == "DERIVED_SCALE_DOWN_CORRECTNESS":
                artifact = expanded_route.generate_route_artifact(
                    mode=mode,
                    world_size=world_size,
                    pattern=pattern,
                )
                route_format = expanded_route.ROUTE_FORMAT
            else:
                artifact = compact_route.generate_route_artifact(
                    manifest_case_id=case_id,
                    pattern=pattern,
                )
                route_format = compact_route.ROUTE_FORMAT
            relative = f"routes/{case_id}--{pattern}.json"
            raw = bundle.canonical_bytes(artifact) + b"\n"
            _write(cls.root / relative, raw)
            route_entries.append(
                {
                    "case_id": case_id,
                    "pattern": pattern,
                    "format": route_format,
                    "semantic_scope": declarations[(case_id, pattern)][
                        "semantic_scope"
                    ],
                    "path": relative,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "canonical_sha256": artifact["canonical_sha256"],
                }
            )

        cls.index: dict[str, Any] = {
            "format": bundle.BUNDLE_FORMAT,
            "version": bundle.BUNDLE_VERSION,
            "manifest": {
                "path": "manifest.json",
                "raw_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "canonical_sha256": hashlib.sha256(
                    schema.canonical_bytes(manifest)
                ).hexdigest(),
            },
            "payload_generator": {
                "id": payload.GENERATOR_ID,
                "seed": payload.SEED,
                "source_path": payload_relative,
                "source_raw_sha256": hashlib.sha256(payload_raw).hexdigest(),
                "source_sha256_kind": bundle.RAW_FILE_SHA256,
                "specification_sha256": specification_sha,
            },
            "routes": route_entries,
            "results": [],
            "bundle_sha256": "",
        }
        _rehash_index(cls.index)
        cls._write_index(cls.index)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    @classmethod
    def _write_index(cls, index: dict[str, Any]) -> None:
        _write(
            cls.root / bundle.BUNDLE_INDEX_NAME, bundle.canonical_bytes(index) + b"\n"
        )

    def _mutated_validation(self, mutate: Callable[[dict[str, Any]], None]) -> str:
        changed = copy.deepcopy(self.index)
        mutate(changed)
        _rehash_index(changed)
        self._write_index(changed)
        try:
            return _expect_bundle_error(bundle.validate_bundle, self.root)
        finally:
            self._write_index(self.index)

    def test_exact_8_by_3_e2e_is_hash_closed_but_never_promoted(self) -> None:
        report = bundle.validate_bundle(self.root)
        self.assertEqual(report["case_count"], 8)
        self.assertEqual(report["route_count"], 24)
        self.assertEqual(report["derived_route_count"], 12)
        self.assertEqual(report["confirmatory_route_count"], 12)
        self.assertEqual(report["result_file_count"], 0)
        self.assertEqual(report["status"], "CPU_ARTIFACTS_VALID_UNVERIFIED")
        self.assertEqual(report["gpu_execution_status"], "NOT_RUN")
        self.assertEqual(report["performance_verification_status"], "NOT_RUN")
        self.assertFalse(report["execution_authorized"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertEqual(report["report_sha256"], bundle.report_sha256(report))

    def test_missing_route_and_case_pattern_mismatch_fail_before_execution(
        self,
    ) -> None:
        missing = self._mutated_validation(lambda value: value["routes"].pop())
        self.assertIn("exact sorted 8 x 3 routes", missing)

        def mismatch(value: dict[str, Any]) -> None:
            value["routes"][0]["pattern"] = "rail_hot"

        wrong = self._mutated_validation(mismatch)
        self.assertIn("exact sorted 8 x 3 routes", wrong)

    def test_route_and_payload_hash_mismatch_are_rejected(self) -> None:
        def route_hash(value: dict[str, Any]) -> None:
            value["routes"][0]["raw_sha256"] = "0" * 64

        self.assertIn("route", self._mutated_validation(route_hash))

        def payload_hash(value: dict[str, Any]) -> None:
            value["payload_generator"]["source_raw_sha256"] = "0" * 64

        self.assertIn(
            "payload source SHA256 mismatch", self._mutated_validation(payload_hash)
        )

    def test_strict_json_and_safe_reader_reject_ambiguous_inputs(self) -> None:
        for raw in (
            b'{"a":1,"a":2}',
            b'{"a":NaN}',
            '{"a":1}'.encode("utf-16"),
        ):
            with self.subTest(raw=raw[:16]):
                _expect_bundle_error(bundle._parse_json, raw, "attack")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target").write_bytes(b"x")
            (root / "target").chmod(0o644)
            (root / "link").symlink_to("target")
            os.mkfifo(root / "fifo", 0o644)
            (root / "unsafe").write_bytes(b"x")
            (root / "unsafe").chmod(0o666)
            descriptor, _identity = bundle._open_directory(root)
            try:
                for name in ("link", "fifo", "unsafe"):
                    with self.subTest(name=name):
                        _expect_bundle_error(
                            bundle._read_relative,
                            descriptor,
                            name,
                            maximum_bytes=16,
                            name=name,
                        )
            finally:
                os.close(descriptor)

    def test_raw_result_contract_forbids_single_file_benchmark_promotion(self) -> None:
        self.assertEqual(results.CASE_SPECS, schema.CASE_SPECS)
        self.assertEqual(results.RAW_SAMPLE_FIELDS, schema.RAW_SAMPLE_FIELDS)
        document = result_fixtures._document(result_code="PASSED_CORRECTNESS")
        document["result_code"] = "PASSED_BENCHMARK"
        document["terminal_stage"] = "BENCHMARK"
        document["evidence_sha256"] = results.evidence_sha256(document)
        with self.assertRaisesRegex(results.RawResultError, "forbids PASSED_BENCHMARK"):
            results.validate_raw_result(document)


if __name__ == "__main__":
    unittest.main()
