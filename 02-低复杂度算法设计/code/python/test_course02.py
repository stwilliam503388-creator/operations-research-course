"""Unit tests for course 02 low-complexity algorithm examples.

These tests cover sorting, knapsack DP, binary-search answers, graph shortest
paths, and monotonic data-structure utilities.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from case03_merge import merge, merge_sort, solve_bruteforce as merge_bruteforce, solve_optimized as merge_optimized
from case04_knapsack import solve_bruteforce as knapsack_bruteforce, solve_optimized as knapsack_optimized, space_optimized
from case05_search import can_chop, can_cut, solve_book, solve_choco, solve_wood
from case06_graph import solve_floyd, solve_optimized as dijkstra_solve
from case07_opt import (
    next_greater_brute,
    next_greater_monotonic,
    sliding_max_brute,
    sliding_max_deque,
)


class MergeSortTests(unittest.TestCase):
    """Tests for merge and merge sort routines."""

    def test_merge_two_sorted_lists(self) -> None:
        self.assertEqual(merge([1, 3, 5], [2, 4, 6]), [1, 2, 3, 4, 5, 6])

    def test_merge_sort_various_inputs(self) -> None:
        cases = [
            [],
            [7],
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [3, 1, 2, 3, 2, 1],
        ]
        for arr in cases:
            with self.subTest(arr=arr):
                self.assertEqual(merge_sort(arr), sorted(arr))

    def test_solve_optimized_matches_bruteforce_on_random_data(self) -> None:
        random.seed(2024)
        for _ in range(10):
            size = random.randint(0, 20)
            arr = [random.randint(-20, 20) for _ in range(size)]
            self.assertEqual(merge_optimized(arr), merge_bruteforce(arr))


class KnapsackTests(unittest.TestCase):
    """Tests for 0/1 knapsack implementations."""

    def test_known_instance(self) -> None:
        weights = [4, 3, 3]
        values = [12, 8, 7]
        self.assertEqual(knapsack_optimized(weights, values, 6), 15)

    def test_edge_cases(self) -> None:
        self.assertEqual(knapsack_optimized([1, 2], [5, 6], 0), 0)
        self.assertEqual(knapsack_optimized([3], [9], 3), 9)

    def test_bruteforce_and_dp_versions_agree(self) -> None:
        random.seed(2025)
        for _ in range(10):
            n = random.randint(1, 6)
            weights = [random.randint(1, 6) for _ in range(n)]
            values = [random.randint(1, 12) for _ in range(n)]
            capacity = random.randint(0, 12)
            expected = knapsack_bruteforce(weights, values, capacity)
            self.assertEqual(knapsack_optimized(weights, values, capacity), expected)
            self.assertEqual(space_optimized(weights, values, capacity), expected)


class BinarySearchAnswerTests(unittest.TestCase):
    """Tests for binary-search-on-answer examples."""

    def test_can_cut_known_inputs(self) -> None:
        woods = [10, 24, 15]
        self.assertTrue(can_cut(6, woods, 7))
        self.assertFalse(can_cut(7, woods, 7))

    def test_solve_wood_known_input(self) -> None:
        self.assertEqual(solve_wood([10, 24, 15], 7), 6)

    def test_solve_book_known_input(self) -> None:
        self.assertEqual(solve_book([12, 34, 67, 90], 2), 113)

    def test_can_chop_known_inputs(self) -> None:
        chocolates = [(5, 6), (6, 7)]
        self.assertTrue(can_chop(3, chocolates, 6))
        self.assertFalse(can_chop(4, chocolates, 6))

    def test_solve_choco_known_input(self) -> None:
        self.assertEqual(solve_choco([(5, 6), (6, 7)], 6), 3)


class GraphAlgorithmTests(unittest.TestCase):
    """Tests for Dijkstra and Floyd-Warshall."""

    def setUp(self) -> None:
        self.graph = [
            [(1, 2), (2, 5)],
            [(2, 1), (3, 3)],
            [(3, 1)],
            [],
        ]

    def test_dijkstra_on_small_graph(self) -> None:
        self.assertEqual(dijkstra_solve(self.graph, 0), [0, 2, 3, 4])

    def test_floyd_matches_dijkstra_for_each_source(self) -> None:
        all_pairs = solve_floyd(self.graph)
        for src in range(len(self.graph)):
            with self.subTest(src=src):
                self.assertEqual(all_pairs[src], dijkstra_solve(self.graph, src))

    def test_triangle_inequality_holds_for_all_pairs_distances(self) -> None:
        all_pairs = solve_floyd(self.graph)
        n = len(all_pairs)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if math.isinf(all_pairs[i][k]) or math.isinf(all_pairs[k][j]):
                        continue
                    self.assertLessEqual(all_pairs[i][j], all_pairs[i][k] + all_pairs[k][j] + 1e-9)


class MonotonicStructureTests(unittest.TestCase):
    """Tests for sliding window maxima and next greater elements."""

    def test_sliding_window_known_example(self) -> None:
        self.assertEqual(sliding_max_deque([1, 3, 2, 4], 2), [3, 3, 4])

    def test_next_greater_known_example(self) -> None:
        self.assertEqual(next_greater_monotonic([4, 2, 1, 3]), [-1, 3, 3, -1])

    def test_optimized_versions_match_bruteforce_on_various_inputs(self) -> None:
        arrays = [
            [],
            [5],
            [1, 3, 2, 4],
            [4, 2, 1, 3],
            [2, 2, 2, 2],
            [5, 1, 5, 1, 5],
        ]
        for arr in arrays:
            max_k = max(1, len(arr)) if arr else 1
            for k in range(1, max_k + 1):
                with self.subTest(arr=arr, k=k):
                    self.assertEqual(sliding_max_deque(arr, k), sliding_max_brute(arr, k))
            with self.subTest(arr=arr, problem="next_greater"):
                self.assertEqual(next_greater_monotonic(arr), next_greater_brute(arr))


if __name__ == "__main__":
    random.seed(0)
    unittest.main()
