"""CPU-only attacks for COMMON_FAIR unverified raw timing evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import statistics
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import rail_balance_common_fair_results as results
import rail_balance_common_fair_schema as schema


def _hash(character: str) -> str:
    return character * 64


def _expected_order(pair_id: str, ordinal: int) -> tuple[str, list[str]]:
    system_a, system_b = results.PAIR_SPECS[pair_id]
    if ordinal % 2 == 0:
        return "ABBA", [system_a, system_b, system_b, system_a]
    return "BAAB", [system_b, system_a, system_a, system_b]


def _rehash(document: dict[str, object]) -> None:
    document["evidence_sha256"] = results.evidence_sha256(document)


def _phase_values(rank: int, repeat: int, iteration: int) -> dict[str, int]:
    base = 1_000 + rank * 100 + repeat * 10 + iteration
    return {
        "route_prepare_complete": base,
        "dispatch_complete": base + 1_000,
        "combine_complete": base + 2_000,
        "roundtrip_complete": base + 3_000,
    }


def _document(
    *,
    case_id: str = "confirmatory-ll-n2-w16",
    route_pattern: str = "balanced",
    pair_id: str = "deepep-off-vs-railbalance-best",
    block_ordinal: int = 0,
    order_position: int = 1,
    result_code: str = "RAW_SAMPLES_COMPLETE_UNVERIFIED",
    route_not_separable: bool = False,
) -> dict[str, object]:
    scope, _mode, _nodes, _ranks_per_node, world_size = results.CASE_SPECS[case_id]
    cycle, order = _expected_order(pair_id, block_ordinal)
    system_id = order[order_position]
    block_id = f"{pair_id}-block-{block_ordinal:02d}"
    process_ids = [
        f"{block_id}-position-{order_position}-repeat-{repeat}"
        for repeat in range(results.INDEPENDENT_REPEATS)
    ]
    raw_samples = result_code == "RAW_SAMPLES_COMPLETE_UNVERIFIED"
    samples: list[dict[str, object]] = []
    if raw_samples:
        for rank in range(world_size):
            for repeat in range(results.INDEPENDENT_REPEATS):
                for iteration in range(results.TOTAL_ITERATIONS):
                    phases: dict[str, int | None] = _phase_values(
                        rank, repeat, iteration
                    )
                    if route_not_separable:
                        phases["route_prepare_complete"] = None
                    samples.append(
                        {
                            "block_id": block_id,
                            "case_id": case_id,
                            "iteration": iteration,
                            "order_position": order_position,
                            "pair_id": pair_id,
                            "phase_ns": phases,
                            "process_instance_id": process_ids[repeat],
                            "rank": rank,
                            "repeat": repeat,
                            "route_pattern": route_pattern,
                            "route_raw_sha256": _hash("3"),
                            "sample_class": (
                                "warmup"
                                if iteration < results.WARMUP_ITERATIONS
                                else "steady"
                            ),
                            "system_id": system_id,
                        }
                    )
    terminal_stage = results.TERMINAL_STAGE_BY_CODE[result_code]
    reasons = (
        []
        if result_code
        in {
            "RAW_SAMPLES_COMPLETE_UNVERIFIED",
            "PASSED_CORRECTNESS",
        }
        else ["terminal fixture reason"]
    )
    if result_code == "RAW_SAMPLES_COMPLETE_UNVERIFIED":
        correctness_status = "PINNED_PASSED"
        correctness_sha: str | None = _hash("a")
    elif result_code == "PASSED_CORRECTNESS":
        correctness_status = "SELF"
        correctness_sha = None
    else:
        correctness_status = "NOT_AVAILABLE"
        correctness_sha = None
    document: dict[str, object] = {
        "format": results.FORMAT_ID,
        "system_id": system_id,
        "case_id": case_id,
        "route_pattern": route_pattern,
        "row_id": f"{case_id}--{route_pattern}--{system_id}",
        "world_size": world_size,
        "result_code": result_code,
        "performance_claim_allowed": False,
        "evidence_verification_status": "NOT_RUN",
        "reasons": reasons,
        "terminal_stage": terminal_stage,
        "manifest_raw_sha256": _hash("1"),
        "manifest_canonical_sha256": _hash("2"),
        "route_raw_sha256": _hash("3"),
        "route_canonical_sha256": _hash("4"),
        "payload_generator_id": "rail-balance-deterministic-bf16-v1",
        "payload_generator_code_sha256": _hash("5"),
        "payload_generator_seed": 20260801,
        "adapter_sha256": _hash("6"),
        "runtime_closure_sha256": _hash("7"),
        "execution_config_sha256": _hash("b"),
        "candidate_id": results.FIXED_CANDIDATE_IDS.get(
            system_id,
            "candidate-round-07-c3",
        ),
        "environment_sha256": _hash("8"),
        "correctness_evidence_status": correctness_status,
        "correctness_evidence_sha256": correctness_sha,
        "order_schedule_sha256": _hash("9"),
        "order_binding": {
            "pair_id": pair_id,
            "block_id": block_id,
            "block_ordinal": block_ordinal,
            "cycle": cycle,
            "order": order,
            "order_position": order_position,
            "process_instance_ids": process_ids,
        },
        "measurement": dict(results.MEASUREMENT_CONTRACT),
        "clock": dict(results.CLOCK_CONTRACT),
        "phase_contract": {
            "route_prepare_complete": (
                "NOT_SEPARABLE" if route_not_separable else "MEASURED"
            ),
            "dispatch_complete": "MEASURED",
            "combine_complete": "MEASURED",
            "roundtrip_complete": "MEASURED",
        },
        "samples": samples,
        "evidence_sha256": "",
    }
    assert scope in {"CONFIRMATORY_COMMON_FAIR", "DERIVED_SCALE_DOWN_CORRECTNESS"}
    _rehash(document)
    return document


def _sample(
    document: dict[str, object], rank: int, repeat: int, iteration: int
) -> dict[str, object]:
    samples = document["samples"]
    assert isinstance(samples, list)
    for value in samples:
        assert isinstance(value, dict)
        if (
            value["rank"] == rank
            and value["repeat"] == repeat
            and value["iteration"] == iteration
        ):
            return value
    raise AssertionError("fixture sample is missing")


def _write_canonical(path: Path, document: dict[str, object]) -> None:
    path.write_bytes(results.canonical_bytes(document) + b"\n")


class CommonFairRawResultTests(unittest.TestCase):
    def test_copied_wire_constants_match_the_current_schema_exactly(self) -> None:
        self.assertEqual(results.SYSTEM_IDS, schema.SYSTEM_IDS)
        self.assertEqual(results.FIXED_CANDIDATE_IDS, schema.FIXED_CANDIDATE_IDS)
        self.assertEqual(results.PHASE_IDS, schema.PHASE_IDS)
        self.assertEqual(results.ROUTE_PATTERNS, schema.ROUTE_PATTERNS)
        self.assertEqual(results.RESULT_CODES, schema.RESULT_CODES)
        self.assertEqual(results.RAW_SAMPLE_FIELDS, schema.RAW_SAMPLE_FIELDS)
        self.assertEqual(results.CASE_SPECS, schema.CASE_SPECS)
        self.assertEqual(results.PAIR_SPECS, schema.PAIR_SPECS)

    def test_formal_confirmatory_case_uses_exact_current_schema_fields(self) -> None:
        document = _document()
        results.validate_raw_result(document)
        self.assertEqual(document["case_id"], "confirmatory-ll-n2-w16")
        self.assertEqual(document["world_size"], 16)
        samples = document["samples"]
        assert isinstance(samples, list)
        self.assertEqual(len(samples), 16 * 5 * 110)
        self.assertEqual(set(samples[0]), set(results.RAW_SAMPLE_FIELDS))
        self.assertFalse(document["performance_claim_allowed"])
        self.assertEqual(document["evidence_verification_status"], "NOT_RUN")

    def test_per_iteration_rank_max_precedes_all_statistics(self) -> None:
        document = _document()
        first_two = ([1, 100], [100, 1])
        for rank in range(16):
            values = first_two[rank] if rank < 2 else (0, 0)
            for offset, duration in enumerate(values):
                phases = _sample(
                    document,
                    rank,
                    0,
                    results.WARMUP_ITERATIONS + offset,
                )["phase_ns"]
                assert isinstance(phases, dict)
                phases["roundtrip_complete"] = duration
            warmup = _sample(document, rank, 0, 0)["phase_ns"]
            assert isinstance(warmup, dict)
            warmup["roundtrip_complete"] = 10**12
        _rehash(document)

        aggregate = results.aggregate_raw_result(document)
        roundtrip = aggregate["phases"]["roundtrip_complete"]
        reduced = roundtrip["per_repeat"][0]["rank_max_ns"]
        self.assertEqual(reduced[:2], [100, 100])
        self.assertEqual(
            max(statistics.fmean([1, 100]), statistics.fmean([100, 1])), 50.5
        )
        self.assertEqual(roundtrip["pooled_diagnostic_statistics"]["count"], 500)
        self.assertLess(roundtrip["pooled_diagnostic_statistics"]["max_ns"], 10**12)
        for field in ("mean_ns", "stddev_population_ns", "cv_population"):
            self.assertIn(field, roundtrip["primary_statistics"])
        self.assertEqual(aggregate["result_code"], "RAW_SAMPLES_COMPLETE_UNVERIFIED")
        self.assertEqual(aggregate["candidate_id"], document["candidate_id"])
        self.assertEqual(
            aggregate["execution_config_sha256"],
            document["execution_config_sha256"],
        )
        self.assertEqual(aggregate["route_raw_sha256"], document["route_raw_sha256"])
        self.assertFalse(aggregate["performance_claim_allowed"])
        self.assertFalse(aggregate["publication_claim_allowed"])
        self.assertTrue(aggregate["external_verification_required"])

    def test_route_prepare_may_be_diagnostic_not_separable(self) -> None:
        aggregate = results.aggregate_raw_result(_document(route_not_separable=True))
        route = aggregate["phases"]["route_prepare_complete"]
        self.assertEqual(route["contract"], "NOT_SEPARABLE")
        self.assertIsNone(route["per_repeat"])
        self.assertEqual(
            aggregate["phases"]["dispatch_complete"]["contract"], "MEASURED"
        )

    def test_all_four_pairs_and_00_09_alternating_cycles_are_accepted(self) -> None:
        for pair_id in results.PAIR_SPECS:
            for ordinal in (0, 1, 8, 9):
                cycle, order = _expected_order(pair_id, ordinal)
                for position in range(4):
                    with self.subTest(pair=pair_id, ordinal=ordinal, position=position):
                        document = _document(
                            pair_id=pair_id,
                            block_ordinal=ordinal,
                            order_position=position,
                            result_code="PASSED_CORRECTNESS",
                        )
                        results.validate_raw_result(document)
                        binding = document["order_binding"]
                        assert isinstance(binding, dict)
                        self.assertEqual(binding["cycle"], cycle)
                        self.assertEqual(binding["order"], order)
                        self.assertEqual(document["system_id"], order[position])

    def test_derived_case_cannot_be_promoted_or_carry_raw_timing(self) -> None:
        document = _document()
        document["case_id"] = "derived-ll-n1-w2"
        document["world_size"] = 2
        document["row_id"] = f"derived-ll-n1-w2--balanced--{document['system_id']}"
        with self.assertRaisesRegex(results.RawResultError, "derived cases cannot"):
            results.validate_raw_result(document)

        correctness = _document(
            case_id="derived-ll-n1-w2",
            result_code="PASSED_CORRECTNESS",
        )
        results.validate_raw_result(correctness)
        self.assertEqual(correctness["samples"], [])

        terminal = _document(
            case_id="derived-ll-n1-w2",
            result_code="CORRECTNESS_FAILED",
        )
        results.validate_raw_result(terminal)
        self.assertEqual(terminal["samples"], [])

        forbidden = _document(result_code="PASSED_CORRECTNESS")
        forbidden["result_code"] = "PASSED_BENCHMARK"
        forbidden["terminal_stage"] = "BENCHMARK"
        with self.assertRaisesRegex(results.RawResultError, "forbids PASSED_BENCHMARK"):
            results.validate_raw_result(forbidden)

    def test_case_world_row_and_route_pattern_bindings_are_exact(self) -> None:
        document = _document()
        document["world_size"] = 32
        with self.assertRaisesRegex(results.RawResultError, "case_id/world_size"):
            results.validate_raw_result(document)

        document = _document()
        document["row_id"] = "confirmatory-ll-n2-w16--railbalance-best"
        with self.assertRaisesRegex(results.RawResultError, "row_id"):
            results.validate_raw_result(document)

        document = _document()
        document["route_pattern"] = "unknown"
        with self.assertRaisesRegex(results.RawResultError, "route_pattern"):
            results.validate_raw_result(document)

        document = _document()
        _sample(document, 0, 0, 0)["route_pattern"] = "rail_hot"
        with self.assertRaisesRegex(results.RawResultError, "route_pattern breaks"):
            results.validate_raw_result(document)

    def test_abba_slot_block_ordinal_and_process_binding_attacks_fail(self) -> None:
        document = _document()
        binding = document["order_binding"]
        assert isinstance(binding, dict)
        binding["order_position"] = 0
        with self.assertRaisesRegex(results.RawResultError, "slot does not contain"):
            results.validate_raw_result(document)

        document = _document(result_code="PASSED_CORRECTNESS")
        binding = document["order_binding"]
        assert isinstance(binding, dict)
        binding["block_ordinal"] = 10
        with self.assertRaisesRegex(results.RawResultError, "block_ordinal"):
            results.validate_raw_result(document)

        document = _document(result_code="PASSED_CORRECTNESS")
        binding = document["order_binding"]
        assert isinstance(binding, dict)
        binding["cycle"] = "BAAB"
        with self.assertRaisesRegex(results.RawResultError, "alternate ABBA/BAAB"):
            results.validate_raw_result(document)

        document = _document()
        _sample(document, 0, 0, 0)["process_instance_id"] = "different-process"
        with self.assertRaisesRegex(results.RawResultError, "process_instance_id"):
            results.validate_raw_result(document)

        for field, invalid in (
            ("block_id", "wrong-block"),
            ("pair_id", "wrong-pair"),
            ("order_position", 0),
        ):
            with self.subTest(sample_order_field=field):
                document = _document()
                _sample(document, 0, 0, 0)[field] = invalid
                with self.assertRaisesRegex(
                    results.RawResultError, "breaks the order binding"
                ):
                    results.validate_raw_result(document)

    def test_manifest_route_runtime_environment_and_evidence_bindings_fail_closed(
        self,
    ) -> None:
        document = _document(result_code="PASSED_CORRECTNESS")
        document["manifest_raw_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(results.RawResultError, "manifest_raw_sha256"):
            results.validate_raw_result(document)

        document = _document()
        _sample(document, 0, 0, 0)["route_raw_sha256"] = _hash("f")
        with self.assertRaisesRegex(results.RawResultError, "route_raw_sha256 breaks"):
            results.validate_raw_result(document)

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
            "correctness_evidence_sha256",
            "order_schedule_sha256",
        ):
            with self.subTest(field=field):
                document = _document()
                document[field] = "bad"
                with self.assertRaisesRegex(results.RawResultError, field):
                    results.validate_raw_result(document)

        document = _document(result_code="PASSED_CORRECTNESS")
        document["payload_generator_id"] = "unknown-generator"
        with self.assertRaisesRegex(results.RawResultError, "payload_generator_id"):
            results.validate_raw_result(document)

        document = _document(result_code="PASSED_CORRECTNESS")
        document["payload_generator_seed"] = 1.0
        with self.assertRaisesRegex(results.RawResultError, "payload_generator_seed"):
            results.validate_raw_result(document)

        document = _document(
            pair_id="deepep-off-vs-railbalance-best",
            order_position=0,
            result_code="PASSED_CORRECTNESS",
        )
        document["candidate_id"] = "different-candidate"
        with self.assertRaisesRegex(results.RawResultError, "candidate_id differs"):
            results.validate_raw_result(document)

        document = _document(result_code="PASSED_CORRECTNESS")
        document["candidate_id"] = None
        with self.assertRaisesRegex(results.RawResultError, "candidate_id"):
            results.validate_raw_result(document)

        document = _document(result_code="PASSED_CORRECTNESS")
        document["adapter_sha256"] = _hash("f")
        with self.assertRaisesRegex(results.RawResultError, "evidence_sha256 mismatch"):
            results.validate_raw_result(document)

        document = _document(result_code="PASSED_CORRECTNESS")
        expected = document["evidence_sha256"]
        document["evidence_sha256"] = _hash("f")
        self.assertEqual(results.evidence_sha256(document), expected)
        with self.assertRaisesRegex(results.RawResultError, "evidence_sha256 mismatch"):
            results.validate_raw_result(document)

    def test_failure_or_unverified_state_cannot_smuggle_promotable_timing(self) -> None:
        document = _document()
        document["result_code"] = "BENCHMARK_FAILED"
        document["terminal_stage"] = "BENCHMARK"
        document["reasons"] = ["benchmark failed"]
        document["correctness_evidence_status"] = "PINNED_PASSED"
        with self.assertRaisesRegex(results.RawResultError, "must not carry timing"):
            results.validate_raw_result(document)

        document = _document()
        document["performance_claim_allowed"] = True
        with self.assertRaisesRegex(results.RawResultError, "must remain false"):
            results.validate_raw_result(document)

        document = _document()
        document["evidence_verification_status"] = "PASSED"
        with self.assertRaisesRegex(results.RawResultError, "must remain NOT_RUN"):
            results.validate_raw_result(document)

        terminal = _document(result_code="BENCHMARK_FAILED")
        results.validate_raw_result(terminal)
        with self.assertRaisesRegex(results.RawResultError, "only RAW_SAMPLES"):
            results.aggregate_raw_result(terminal)

    def test_missing_rank_duplicate_and_illegal_nanoseconds_are_rejected(self) -> None:
        document = _document()
        samples = document["samples"]
        assert isinstance(samples, list)
        samples.remove(_sample(document, 15, 4, results.TOTAL_ITERATIONS - 1))
        with self.assertRaisesRegex(results.RawResultError, "coverage is incomplete"):
            results.validate_raw_result(document)

        document = _document()
        samples = document["samples"]
        assert isinstance(samples, list)
        samples.append(copy.deepcopy(_sample(document, 0, 0, 0)))
        with self.assertRaisesRegex(
            results.RawResultError, "duplicate rank/repeat/iteration"
        ):
            results.validate_raw_result(document)

        for invalid in (-1, 1.5, True):
            with self.subTest(invalid=invalid):
                document = _document()
                phases = _sample(document, 0, 0, 0)["phase_ns"]
                assert isinstance(phases, dict)
                phases["dispatch_complete"] = invalid
                with self.assertRaisesRegex(
                    results.RawResultError, "must be an integer"
                ):
                    results.validate_raw_result(document)

    def test_clock_measurement_and_phase_contracts_are_exact(self) -> None:
        document = _document()
        clock = document["clock"]
        assert isinstance(clock, dict)
        clock["timing_api"] = "HOST_MONOTONIC"
        with self.assertRaisesRegex(results.RawResultError, "clock differs"):
            results.validate_raw_result(document)

        document = _document()
        measurement = document["measurement"]
        assert isinstance(measurement, dict)
        measurement["steady_iterations"] = 99
        with self.assertRaisesRegex(results.RawResultError, "measurement"):
            results.validate_raw_result(document)

        document = _document()
        phases = _sample(document, 0, 0, 0)["phase_ns"]
        assert isinstance(phases, dict)
        del phases["combine_complete"]
        with self.assertRaisesRegex(results.RawResultError, "phase_ns fields differ"):
            results.validate_raw_result(document)

        document = _document(result_code="PASSED_CORRECTNESS")
        phase_contract = document["phase_contract"]
        assert isinstance(phase_contract, dict)
        phase_contract["route_prepare_complete"] = []
        with self.assertRaisesRegex(results.RawResultError, "phase_contract"):
            results.validate_raw_result(document)

        document = _document(result_code="PASSED_CORRECTNESS")
        document["correctness_evidence_status"] = []
        with self.assertRaisesRegex(
            results.RawResultError, "correctness_evidence_status"
        ):
            results.validate_raw_result(document)

    def test_canonical_loader_round_trip_and_strict_json(self) -> None:
        document = _document(
            case_id="derived-ll-n1-w2",
            result_code="PASSED_CORRECTNESS",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "result.json"
            _write_canonical(path, document)
            loaded, raw_sha256, canonical_sha256 = results.load_raw_result(path)
            self.assertEqual(loaded, document)
            self.assertEqual(raw_sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(
                canonical_sha256,
                hashlib.sha256(results.canonical_bytes(document)).hexdigest(),
            )

            duplicate = root / "duplicate.json"
            duplicate.write_text('{"format":"x","format":"y"}\n', encoding="utf-8")
            with self.assertRaisesRegex(results.RawResultError, "duplicate JSON key"):
                results.load_raw_result(duplicate)

            bom = root / "bom.json"
            bom.write_bytes(b"\xef\xbb\xbf" + results.canonical_bytes(document) + b"\n")
            with self.assertRaisesRegex(results.RawResultError, "BOM"):
                results.load_raw_result(bom)

            invalid_utf8 = root / "invalid-utf8.json"
            invalid_utf8.write_bytes(b"\xff\n")
            with self.assertRaisesRegex(
                results.RawResultError, "invalid raw result JSON"
            ):
                results.load_raw_result(invalid_utf8)

            pretty = root / "pretty.json"
            pretty.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                results.RawResultError, "canonical byte encoding"
            ):
                results.load_raw_result(pretty)

    def test_loader_rejects_fifo_links_and_unsafe_modes_without_blocking(self) -> None:
        document = _document(
            case_id="derived-ll-n1-w2",
            result_code="PASSED_CORRECTNESS",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            _write_canonical(target, document)

            fifo = root / "result.fifo"
            os.mkfifo(fifo, 0o600)
            with self.assertRaisesRegex(results.RawResultError, "regular file"):
                results.load_raw_result(fifo)

            symlink = root / "symlink.json"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(results.RawResultError, "cannot safely open"):
                results.load_raw_result(symlink)

            hardlink = root / "hardlink.json"
            os.link(target, hardlink)
            with self.assertRaisesRegex(
                results.RawResultError, "exactly one hard link"
            ):
                results.load_raw_result(hardlink)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "writable.json"
            _write_canonical(path, document)
            path.chmod(0o666)
            with self.assertRaisesRegex(results.RawResultError, "group/world writable"):
                results.load_raw_result(path)

    def test_loader_rejects_symlinked_parent_and_ctime_race(self) -> None:
        document = _document(
            case_id="derived-ll-n1-w2",
            result_code="PASSED_CORRECTNESS",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_parent = root / "real"
            real_parent.mkdir()
            path = real_parent / "result.json"
            _write_canonical(path, document)
            linked = root / "linked"
            linked.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(results.RawResultError, "safely traverse"):
                results.load_raw_result(linked / "result.json")

            real_fstat = os.fstat
            regular_calls = 0

            def racing_fstat(descriptor: int) -> object:
                nonlocal regular_calls
                metadata = real_fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    return metadata
                regular_calls += 1
                if regular_calls != 2:
                    return metadata
                values = {
                    name: getattr(metadata, name)
                    for name in (
                        "st_dev",
                        "st_ino",
                        "st_mode",
                        "st_uid",
                        "st_nlink",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                }
                values["st_ctime_ns"] += 1
                return types.SimpleNamespace(**values)

            with (
                mock.patch.object(results.os, "fstat", side_effect=racing_fstat),
                self.assertRaisesRegex(results.RawResultError, "changed while"),
            ):
                results.load_raw_result(path)


if __name__ == "__main__":
    unittest.main()
