"""CPU-only tests for the dependency-free FAST-style experiment planner."""

import unittest

from fast_style_planner import (
    plan_cyclic_source_local_waves,
    plan_equal_chunk_interleaved_waves,
    plan_fast_style_waves,
    reconstruct_demand,
)


class FastStylePlannerTest(unittest.TestCase):

    def test_weighted_four_node_cycles(self):
        weights = (5, 3, 2)
        demand = [[0] * 4 for _ in range(4)]
        for source in range(4):
            for offset, weight in enumerate(weights, start=1):
                demand[source][(source + offset) % 4] = weight

        waves = plan_fast_style_waves(demand)
        self.assertEqual(len(waves), 3)
        self.assertEqual([sum(item[3] for item in wave) for wave in waves],
                         [20, 12, 8])
        self.assertEqual(reconstruct_demand(4, waves), demand)
        for wave in waves:
            self.assertEqual(len({item[0] for item in wave}), 4)
            self.assertEqual(len({item[1] for item in wave}), 4)

        local_waves = plan_cyclic_source_local_waves(demand)
        self.assertEqual(local_waves, waves)
        self.assertEqual(reconstruct_demand(4, local_waves), demand)

    def test_unbalanced_matrix_padding_is_not_transmitted(self):
        demand = [
            [0, 5, 0],
            [0, 0, 2],
            [1, 0, 0],
        ]
        waves = plan_fast_style_waves(demand)
        self.assertEqual(reconstruct_demand(3, waves), demand)
        self.assertEqual(sum(item[3] for wave in waves for item in wave), 8)

    def test_equal_chunks_are_interleaved_before_tails(self):
        # Each row/column contains the same hot/mid/cold multiset, but the
        # weights are deliberately not aligned with cyclic destination shifts.
        demand = [
            [0, 10, 6, 3],
            [10, 0, 3, 6],
            [6, 3, 0, 10],
            [3, 6, 10, 0],
        ]
        waves = plan_equal_chunk_interleaved_waves(demand, 4)
        self.assertEqual(reconstruct_demand(4, waves), demand)
        for wave in waves:
            self.assertEqual(len({item[0] for item in wave}), len(wave))
            self.assertEqual(len({item[1] for item in wave}), len(wave))
        counts = [item[3] for wave in waves for item in wave]
        first_tail = next(index for index, count in enumerate(counts)
                          if count < 4)
        self.assertTrue(all(count == 4 for count in counts[:first_tail]))
        self.assertTrue(all(count < 4 for count in counts[first_tail:]))

    def test_equal_chunk_argument_validation(self):
        demand = [[0, 1], [1, 0]]
        with self.assertRaises(ValueError):
            plan_equal_chunk_interleaved_waves(demand, 0)
        with self.assertRaises(TypeError):
            plan_equal_chunk_interleaved_waves(demand, 1.5)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            plan_fast_style_waves([[1, 2]])
        with self.assertRaises(ValueError):
            plan_fast_style_waves([[0, -1], [1, 0]])
        with self.assertRaises(TypeError):
            plan_fast_style_waves([[0, 1.5], [1, 0]])


if __name__ == '__main__':
    unittest.main()
