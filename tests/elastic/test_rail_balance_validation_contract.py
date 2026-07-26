#!/usr/bin/env python3

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch

import run_rail_balance_hybrid_multinode as multinode_runner

from rail_balance_validation_common import (
    CANONICAL_CASES,
    RESULT_SCHEMA_VERSION,
    build_validation_bundle,
    build_deterministic_topk,
    count_distinct_remote_destinations,
    new_result_record,
)


class DeterministicRouteTest(unittest.TestCase):
    D = 3
    G = 4
    N = 3
    K = 4
    E = 48

    def build(self, case):
        return build_deterministic_topk(
            case,
            num_scaleout_ranks=self.D,
            num_scaleup_ranks=self.G,
            num_tokens_per_rank=self.N,
            num_topk=self.K,
            num_experts=self.E,
        )

    def test_routes_are_deterministic_valid_and_distinct(self):
        expected_active_rails = {
            "balanced": self.G,
            "two_hot": 2,
            "one_hot": 1,
            "capacity": 1,
        }
        experts_per_server = self.E // self.D
        for case in CANONICAL_CASES:
            with self.subTest(case=case):
                routes = self.build(case)
                self.assertEqual(routes, self.build(case))
                self.assertEqual(len(routes), self.D * self.G)
                for rank, rank_routes in enumerate(routes):
                    self.assertEqual(len(rank_routes), self.N)
                    source_server = rank // self.G
                    local_rank = rank % self.G
                    expected_server = (
                        (source_server + 1) % self.D
                        if local_rank < expected_active_rails[case]
                        else source_server
                    )
                    for token_route in rank_routes:
                        self.assertEqual(len(token_route), self.K)
                        self.assertEqual(len(set(token_route)), self.K)
                        self.assertTrue(all(0 <= expert < self.E for expert in token_route))
                        self.assertEqual(
                            {expert // experts_per_server for expert in token_route},
                            {expected_server},
                        )

    def test_canonical_remote_counts_and_server_deduplication(self):
        expected_active_rails = {
            "balanced": 4,
            "two_hot": 2,
            "one_hot": 1,
            "capacity": 1,
        }
        for case, active_rails in expected_active_rails.items():
            with self.subTest(case=case):
                counts = count_distinct_remote_destinations(
                    self.build(case),
                    num_scaleout_ranks=self.D,
                    num_scaleup_ranks=self.G,
                    num_experts=self.E,
                )
                for source_server in range(self.D):
                    remote_server = (source_server + 1) % self.D
                    for local_rank in range(self.G):
                        expected = self.N if local_rank < active_rails else 0
                        self.assertEqual(counts[source_server][local_rank][remote_server], expected)
                        self.assertEqual(sum(counts[source_server][local_rank]), expected)

    def test_canonical_cases_intentionally_have_different_remote_totals(self):
        # Future traffic records must report these different payload volumes;
        # the fixtures must not be padded to make their Gin traffic equal.
        expected_totals = {
            "balanced": 36,
            "two_hot": 18,
            "one_hot": 9,
            "capacity": 9,
        }
        actual_totals = {}
        for case in CANONICAL_CASES:
            counts = count_distinct_remote_destinations(
                self.build(case),
                num_scaleout_ranks=self.D,
                num_scaleup_ranks=self.G,
                num_experts=self.E,
            )
            actual_totals[case] = sum(
                count
                for server_counts in counts
                for rail_counts in server_counts
                for count in rail_counts
            )
        self.assertEqual(actual_totals, expected_totals)

    def test_force_v1_domain_boundaries_and_exact_integer_types(self):
        minimum = build_deterministic_topk(
            "balanced",
            num_scaleout_ranks=2,
            num_scaleup_ranks=2,
            num_tokens_per_rank=0,
            num_topk=1,
            num_experts=4,
        )
        maximum = build_deterministic_topk(
            "balanced",
            num_scaleout_ranks=32,
            num_scaleup_ranks=32,
            num_tokens_per_rank=0,
            num_topk=32,
            num_experts=1024,
        )
        expert_maximum = build_deterministic_topk(
            "balanced",
            num_scaleout_ranks=2,
            num_scaleup_ranks=4,
            num_tokens_per_rank=0,
            num_topk=32,
            num_experts=2048,
        )
        self.assertEqual(len(minimum), 4)
        self.assertEqual(len(maximum), 1024)
        self.assertEqual(len(expert_maximum), 8)

        defaults = {
            "case": "balanced",
            "num_scaleout_ranks": self.D,
            "num_scaleup_ranks": self.G,
            "num_tokens_per_rank": 0,
            "num_topk": self.K,
            "num_experts": self.E,
        }
        invalid_boundaries = (
            {"num_scaleout_ranks": 1},
            {"num_scaleout_ranks": 33},
            {"num_scaleout_ranks": True},
            {"num_scaleup_ranks": 1},
            {"num_scaleup_ranks": 33},
            {"num_scaleup_ranks": False},
            {"num_topk": 0},
            {"num_topk": 33},
            {"num_topk": True},
            {"num_tokens_per_rank": True},
            {"num_tokens_per_rank": -1},
            {"num_tokens_per_rank": 1 << 31},
            {"num_tokens_per_rank": (1 << 31) // (self.D * self.G) + 1},
            {"num_experts": True},
            {"num_experts": 0},
            {"num_experts": 4096},
        )
        for override in invalid_boundaries:
            with self.subTest(override=override):
                args = dict(defaults)
                args.update(override)
                with self.assertRaises(ValueError):
                    build_deterministic_topk(**args)

        with self.assertRaises(ValueError):
            build_deterministic_topk(
                "balanced",
                num_scaleout_ranks=2,
                num_scaleup_ranks=2,
                num_tokens_per_rank=0,
                num_topk=1,
                num_experts=2048,
            )
        with self.assertRaises(ValueError):
            build_deterministic_topk(
                "balanced",
                num_scaleout_ranks=2,
                num_scaleup_ranks=8,
                num_tokens_per_rank=0,
                num_topk=1,
                num_experts=4096,
            )
        with self.assertRaises(ValueError):
            build_deterministic_topk(
                "balanced",
                num_scaleout_ranks=2,
                num_scaleup_ranks=2,
                num_tokens_per_rank=(1 << 31) // (4 * 32) + 1,
                num_topk=32,
                num_experts=1024,
            )

    def test_two_hot_canonical_fixture_requires_an_inactive_rail(self):
        # G=2 is a valid force-v1 dimension.  This restriction only keeps the
        # two_hot fixture observably different from balanced.
        with self.assertRaises(ValueError):
            build_deterministic_topk(
                "two_hot",
                num_scaleout_ranks=2,
                num_scaleup_ranks=2,
                num_tokens_per_rank=0,
                num_topk=2,
                num_experts=8,
            )

    def test_invalid_route_inputs_are_rejected(self):
        invalid_calls = (
            {"case": "unknown"},
            {"num_tokens_per_rank": -1},
            {"num_topk": 0},
            {"num_topk": 17},
            {"num_experts": 47},
        )
        defaults = {
            "case": "balanced",
            "num_scaleout_ranks": self.D,
            "num_scaleup_ranks": self.G,
            "num_tokens_per_rank": self.N,
            "num_topk": self.K,
            "num_experts": self.E,
        }
        for override in invalid_calls:
            with self.subTest(override=override):
                args = dict(defaults)
                args.update(override)
                with self.assertRaises(ValueError):
                    build_deterministic_topk(**args)


class CountOracleTest(unittest.TestCase):
    def test_non_square_three_server_handwritten_oracle(self):
        # This fixture is deliberately handwritten rather than produced by
        # build_deterministic_topk.  D=3 and G=2 expose swapped dimensions.
        routes = [
            [[12, 13], [24, 25]],
            [[12, 24], [14, 15]],
            [[0, 1], [2, 3]],
            [[0, 24], [2, 3]],
            [[0, 1], [12, 13]],
            [[0, 12], [14, 15]],
        ]
        expected = [
            [[0, 1, 1], [0, 2, 1]],
            [[2, 0, 0], [2, 0, 1]],
            [[1, 1, 0], [1, 2, 0]],
        ]
        self.assertEqual(
            count_distinct_remote_destinations(
                routes,
                num_scaleout_ranks=3,
                num_scaleup_ranks=2,
                num_experts=36,
            ),
            expected,
        )

    def test_invalid_and_ragged_routes_raise_value_error(self):
        valid = [
            [[0, 1]],
            [[0, 1]],
            [[4, 5]],
            [[4, 5]],
        ]
        invalid_routes = []
        invalid_routes.append(tuple(valid))
        invalid_routes.append(valid[:-1])

        rank_tuple = copy.deepcopy(valid)
        rank_tuple[0] = tuple(rank_tuple[0])
        invalid_routes.append(rank_tuple)

        token_count_ragged = copy.deepcopy(valid)
        token_count_ragged[0].append([0, 1])
        invalid_routes.append(token_count_ragged)

        token_tuple = copy.deepcopy(valid)
        token_tuple[0][0] = tuple(token_tuple[0][0])
        invalid_routes.append(token_tuple)

        topk_ragged = copy.deepcopy(valid)
        topk_ragged[0][0].append(2)
        invalid_routes.append(topk_ragged)

        topk_zero = [[[]] for _ in range(4)]
        invalid_routes.append(topk_zero)

        bool_expert = copy.deepcopy(valid)
        bool_expert[0][0][0] = True
        invalid_routes.append(bool_expert)

        out_of_range = copy.deepcopy(valid)
        out_of_range[0][0][0] = 8
        invalid_routes.append(out_of_range)

        duplicate_expert = copy.deepcopy(valid)
        duplicate_expert[0][0] = [0, 0]
        invalid_routes.append(duplicate_expert)

        for routes in invalid_routes:
            with self.subTest(routes=routes):
                with self.assertRaises(ValueError):
                    count_distinct_remote_destinations(
                        routes,
                        num_scaleout_ranks=2,
                        num_scaleup_ranks=2,
                        num_experts=8,
                    )

        for invalid_dimension in (True, 1, 33):
            with self.subTest(invalid_dimension=invalid_dimension):
                with self.assertRaises(ValueError):
                    count_distinct_remote_destinations(
                        valid,
                        num_scaleout_ranks=invalid_dimension,
                        num_scaleup_ranks=2,
                        num_experts=8,
                    )


class ResultContractTest(unittest.TestCase):
    def test_result_record_has_stable_json_contract(self):
        topology = {"world_size": 8, "num_scaleout_ranks": 2, "num_scaleup_ranks": 4}
        config = {"num_tokens_per_rank": 3, "num_topk": 4, "num_experts": 32}
        record = new_result_record(
            run_id="cpu-contract",
            case="one_hot",
            mode="force",
            topology=topology,
            config=config,
        )

        self.assertEqual(record["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(record["evidence_label"], "REAL_HYBRID_RUNTIME_UNTESTED")
        self.assertEqual(
            set(record),
            {
                "schema_version",
                "run_id",
                "evidence_label",
                "case",
                "mode",
                "topology",
                "config",
                "correctness",
                "plan",
                "traffic",
                "runtime",
                "error",
                "claim_scope",
            },
        )
        self.assertEqual(record["claim_scope"], "validation_only")
        self.assertEqual(record["traffic"]["scope"], "payload_only")
        self.assertEqual(record["runtime"]["availability"], "not_instrumented")
        self.assertIsNone(record["runtime"]["wait_cycles"])
        self.assertIsNone(record["runtime"]["qp_utilization"])
        self.assertIsNone(record["runtime"]["nic_bytes"])
        self.assertIsNone(record["error"])
        json.dumps(record, sort_keys=True)

        topology["world_size"] = 16
        config["num_topk"] = 8
        self.assertEqual(record["topology"]["world_size"], 8)
        self.assertEqual(record["config"]["num_topk"], 4)

    def test_result_metadata_is_strict(self):
        defaults = {
            "run_id": "cpu-contract",
            "case": "one_hot",
            "mode": "force",
            "topology": {},
            "config": {},
        }
        invalid_metadata = (
            {"run_id": 1},
            {"run_id": True},
            {"run_id": "   "},
            {"case": True},
            {"case": "unknown"},
            {"mode": True},
            {"mode": "auto"},
        )
        for override in invalid_metadata:
            with self.subTest(override=override):
                args = dict(defaults)
                args.update(override)
                with self.assertRaises(ValueError):
                    new_result_record(**args)

    def test_nested_inputs_are_json_roundtrip_copied(self):
        topology = {"nodes": [{"ranks": [0, 1]}]}
        config = {"shape": {"topk": [2, 4]}}
        record = new_result_record(
            run_id="cpu-contract",
            case="balanced",
            mode="off",
            topology=topology,
            config=config,
        )
        topology["nodes"][0]["ranks"].append(2)
        config["shape"]["topk"].append(8)
        self.assertEqual(record["topology"], {"nodes": [{"ranks": [0, 1]}]})
        self.assertEqual(record["config"], {"shape": {"topk": [2, 4]}})

    def test_non_dict_or_non_json_inputs_are_rejected(self):
        class ListSubclass(list):
            pass

        recursive = {}
        recursive["self"] = recursive
        invalid_pairs = (
            ([], {}),
            ({}, []),
            ({"bad": {1, 2}}, {}),
            ({}, {"bad": object()}),
            ({1: "bad-key"}, {}),
            ({"nested": {1: "bad-key"}}, {}),
            ({"nested": ListSubclass([{1: "bad-key"}])}, {}),
            (recursive, {}),
            ({"bad": float("nan")}, {}),
            ({}, {"bad": float("inf")}),
        )
        for topology, config in invalid_pairs:
            with self.subTest(topology=topology, config=config):
                with self.assertRaises(ValueError):
                    new_result_record(
                        run_id="cpu-contract",
                        case="balanced",
                        mode="off",
                        topology=topology,
                        config=config,
                    )


class ValidationBundleTest(unittest.TestCase):
    def test_bundle_uses_hybrid_oracle_and_expected_payload_traffic(self):
        bundle = build_validation_bundle(
            run_id="bundle",
            case="two_hot",
            modes=("off", "force"),
            num_scaleout_ranks=3,
            num_scaleup_ranks=4,
            num_tokens_per_rank=8,
            num_topk=4,
            num_experts=48,
            hidden=256,
            num_channels=2,
        )
        self.assertEqual(bundle["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(bundle["evidence_label"], "REAL_HYBRID_RUNTIME_UNTESTED")
        self.assertEqual(bundle["modes"], ["off", "force"])
        self.assertFalse(bundle["capacity_failed"])
        self.assertEqual(bundle["total_remote_copies"], 3 * 2 * 8)
        self.assertEqual(set(bundle["records"]), {"off", "force"})

        force = bundle["records"]["force"]
        self.assertEqual(force["claim_scope"], "validation_only")
        self.assertEqual(force["plan"]["source"], "cpu_oracle")
        self.assertEqual(force["traffic"]["source"], "cpu_oracle_expected")
        self.assertEqual(force["traffic"]["expected_gin_puts"], 48)
        self.assertEqual(force["traffic"]["expected_gin_bytes"], 48 * 256 * 2)
        self.assertGreater(force["plan"]["moved_copies"], 0)
        json.dumps(bundle, sort_keys=True)

    def test_capacity_bundle_fails_closed_before_runtime(self):
        bundle = build_validation_bundle(
            run_id="capacity",
            case="capacity",
            modes=("force",),
            num_scaleout_ranks=3,
            num_scaleup_ranks=4,
            num_tokens_per_rank=8,
            num_topk=4,
            num_experts=48,
            hidden=256,
            num_channels=2,
        )
        self.assertTrue(bundle["capacity_failed"])
        self.assertLess(
            bundle["config"]["proxy_slots_per_rank"],
            bundle["max_proxy_required"],
        )
        force = bundle["records"]["force"]
        self.assertFalse(force["runtime"]["completed"])
        self.assertEqual(force["evidence_label"], "REAL_HYBRID_RUNTIME_UNTESTED")

    def test_proxy_demand_is_independent_of_channel_striping(self):
        proxy_required = []
        for num_channels in (1, 2, 8):
            bundle = build_validation_bundle(
                run_id=f"channel-{num_channels}",
                case="one_hot",
                modes=("force",),
                num_scaleout_ranks=3,
                num_scaleup_ranks=4,
                num_tokens_per_rank=8,
                num_topk=4,
                num_experts=48,
                hidden=256,
                num_channels=num_channels,
                proxy_slots_per_rank=64,
            )
            proxy_required.append(bundle["records"]["force"]["plan"]["proxy_required"])
        self.assertEqual(proxy_required[0], proxy_required[1])
        self.assertEqual(proxy_required[0], proxy_required[2])

    def test_bundle_metadata_is_strict(self):
        defaults = {
            "run_id": "bundle",
            "case": "balanced",
            "modes": ("force",),
            "num_scaleout_ranks": 3,
            "num_scaleup_ranks": 4,
            "num_tokens_per_rank": 8,
            "num_topk": 4,
            "num_experts": 48,
            "hidden": 256,
            "num_channels": 2,
        }
        invalid = (
            {"modes": "force"},
            {"modes": ()},
            {"modes": ("force", "force")},
            {"modes": ("auto",)},
            {"hidden": 0},
            {"hidden": True},
            {"hidden": 255},
            {"hidden": 257},
            {"payload_dtype": "fp8"},
            {"num_channels": 0},
            {"num_channels": 1025},
            {"num_tokens_per_rank": 0},
            {"case": "capacity", "proxy_slots_per_rank": 64},
        )
        for override in invalid:
            with self.subTest(override=override):
                args = dict(defaults)
                args.update(override)
                with self.assertRaises(ValueError):
                    build_validation_bundle(**args)

    def test_multinode_runner_preserves_the_truthful_runtime_boundary(self):
        source = Path(__file__).with_name(
            "run_rail_balance_hybrid_multinode.py"
        ).read_text()
        self.assertIn('num_scaleout_ranks > 1', source)
        self.assertIn('"off/force A/B"', source)
        self.assertIn('"dispatch-plan rejected rank 0"', source)
        self.assertIn('_C._rail_balance_force_available = original_capability', source)
        self.assertNotIn('os.environ["EP_DISABLE_GIN"]', source)

    def test_multinode_launch_environment_fails_closed(self):
        valid = {
            "WORLD_SIZE": "2",
            "RANK": "1",
            "MASTER_ADDR": "10.0.0.1",
            "MASTER_PORT": "29500",
        }
        with mock.patch.dict(os.environ, valid, clear=True):
            multinode_runner._validate_launch_environment()
        invalid = (
            {},
            {**valid, "WORLD_SIZE": "1"},
            {**valid, "RANK": "2"},
            {**valid, "MASTER_ADDR": "127.0.0.1"},
            {**valid, "MASTER_PORT": "0"},
            {**valid, "EP_DISABLE_GIN": "1"},
        )
        for environment in invalid:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(ValueError):
                        multinode_runner._validate_launch_environment()

    def test_watchdog_kills_group_even_when_spawn_leader_exits(self):
        process = mock.Mock(pid=12345)
        process.wait.return_value = 1
        with mock.patch.object(os, "killpg") as killpg:
            multinode_runner._terminate_process_group(process)
        self.assertEqual(
            killpg.call_args_list,
            [mock.call(12345, signal.SIGTERM), mock.call(12345, signal.SIGKILL)],
        )
        self.assertEqual(process.wait.call_count, 2)

    def test_multinode_runner_uses_unique_exact_weights(self):
        routes = [
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
        ]
        x, topk_idx, topk_weights = multinode_runner._make_input(
            rank=1,
            routes=routes,
            num_tokens=2,
            hidden=256,
            device=torch.device("cpu"),
            topk_dtype=torch.int64,
        )
        self.assertEqual(tuple(x.shape), (2, 256))
        self.assertTrue(torch.equal(topk_idx, torch.tensor(routes[1])))
        self.assertTrue(torch.equal(
            topk_weights,
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ))
        self.assertEqual(torch.unique(topk_weights).numel(), 4)

    def test_runtime_record_does_not_claim_unavailable_counters(self):
        record = new_result_record(
            run_id="runtime-record",
            case="one_hot",
            mode="force",
            topology={"world_size": 8},
            config={"hidden": 256},
        )
        multinode_runner._mark_completed(record, {
            "global_digest": "abc",
            "rank_digests": ["a", "b"],
        })
        self.assertEqual(
            record["evidence_label"],
            "REAL_HYBRID_D_GT_1_CORRECTNESS_VALIDATED_COUNTERS_UNAVAILABLE",
        )
        self.assertTrue(record["runtime"]["completed"])
        self.assertIsNone(record["runtime"]["wait_cycles"])
        self.assertIsNone(record["runtime"]["qp_utilization"])
        self.assertIsNone(record["runtime"]["nic_bytes"])

    def test_cli_emits_stable_json_bundle(self):
        script = Path(__file__).with_name("run_rail_balance_validation_bundle.py")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bundle.json"
            subprocess.check_call([
                sys.executable,
                str(script),
                "--run-id",
                "cli",
                "--case",
                "two_hot",
                "--mode",
                "both",
                "--num-scaleout-ranks",
                "3",
                "--num-scaleup-ranks",
                "4",
                "--num-tokens-per-rank",
                "8",
                "--num-topk",
                "4",
                "--num-experts",
                "48",
                "--hidden",
                "256",
                "--num-channels",
                "2",
                "--output",
                str(output),
            ])
            payload = json.loads(output.read_text())
        self.assertEqual(payload["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(len(payload["bundles"]), 1)
        self.assertEqual(payload["bundles"][0]["case"], "two_hot")


if __name__ == "__main__":
    unittest.main()
