"""Unit tests for course 05 operations research examples.

These tests cover deterministic helper logic and small optimization instances.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np

COURSE_DIR = Path(__file__).resolve().parent
if str(COURSE_DIR) not in sys.path:
    sys.path.insert(0, str(COURSE_DIR))

import case03_logistics
import case04_inventory
import case05_product_mix
import case_metaheuristic_tsp


class InventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(2024)
        np.random.seed(2024)

    def test_newsvendor_returns_expected_critical_ratio_and_quantile(self) -> None:
        demand_probs = {0: 0.2, 1: 0.3, 2: 0.5}
        q_opt, expected_cost, critical_ratio = case04_inventory.newsvendor_analytical(
            h=1.0, p=3.0, demand_probs=demand_probs
        )
        self.assertAlmostEqual(critical_ratio, 3.0 / 4.0, places=12)
        self.assertEqual(q_opt, 2)
        self.assertGreaterEqual(expected_cost, 0.0)

    def test_single_period_dp_matches_newsvendor_when_fixed_cost_is_zero(self) -> None:
        demand_dist = [(0, 0.2), (1, 0.3), (2, 0.5)]
        q_opt, expected_cost, _ = case04_inventory.newsvendor_analytical(1.0, 3.0, demand_dist)
        stats = case04_inventory.dp_inventory(
            T=1, K=0.0, h=1.0, p=3.0, C=5, demand_dist=demand_dist, init_inventory=0
        )
        self.assertEqual(stats["policy"][0][0], q_opt)
        self.assertAlmostEqual(stats["V"][0][0], expected_cost, places=12)

    def test_compute_s_s_policy_returns_lists_with_expected_lengths(self) -> None:
        policy = [
            {0: 5, 1: 4, 2: 0},
            {0: 2, 1: 0, 2: 0},
            {0: 0, 1: 0, 2: 0},
        ]
        s_policy, S_policy = case04_inventory.compute_s_s_policy(policy, T=3)
        self.assertEqual(len(s_policy), 3)
        self.assertEqual(len(S_policy), 3)
        self.assertEqual(s_policy[:2], [0, 0])
        self.assertEqual(S_policy[:2], [5, 2])
        self.assertEqual(s_policy[2], 101)


class LogisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(123)

    def test_generate_data_returns_balanced_dimensions(self) -> None:
        data = case03_logistics.generate_data()
        self.assertEqual(len(data["supplies"]), data["n_factories"])
        self.assertEqual(len(data["warehouse_capacities"]), data["n_warehouses"])
        self.assertEqual(len(data["customer_demands"]), data["n_customers"])
        self.assertEqual(len(data["f2w_cost"]), data["n_factories"])
        self.assertEqual(len(data["f2w_cost"][0]), data["n_warehouses"])
        self.assertEqual(len(data["w2c_cost"]), data["n_warehouses"])
        self.assertEqual(len(data["w2c_cost"][0]), data["n_customers"])
        self.assertLessEqual(sum(data["customer_demands"]), sum(data["supplies"]))
        self.assertGreaterEqual(sum(data["warehouse_capacities"]), sum(data["supplies"]))

    def test_northwest_corner_preserves_supply_and_demand_totals(self) -> None:
        supply = [20, 30]
        demand = [10, 15, 25]
        alloc = case03_logistics.northwest_corner(supply, demand)
        self.assertEqual(len(alloc), 2)
        self.assertEqual(len(alloc[0]), 3)
        np.testing.assert_allclose(np.sum(alloc, axis=1), supply)
        np.testing.assert_allclose(np.sum(alloc, axis=0), demand)

    def test_solve_transportation_finds_known_small_optimum(self) -> None:
        supply = [20, 30]
        demand = [30, 20]
        cost = [[2, 3], [4, 1]]
        result = case03_logistics.solve_transportation(supply, demand, cost)
        alloc = np.array(result["allocation"])
        np.testing.assert_allclose(alloc.sum(axis=1), supply)
        np.testing.assert_allclose(alloc.sum(axis=0), demand)
        self.assertAlmostEqual(result["total_cost"], 100.0, places=8)

    def test_assign_customers_to_warehouses_uses_feasible_low_cost_choices(self) -> None:
        assignment, total_cost, remaining = case03_logistics.assign_customers_to_warehouses(
            warehouse_supplies=[6, 5],
            customer_demands=[2, 3, 4],
            w2c_cost=[[1, 10, 2], [5, 1, 1]],
        )
        self.assertEqual(assignment, [0, 1, 0])
        self.assertAlmostEqual(total_cost, 13.0, places=12)
        self.assertEqual(remaining, [0, 2])


class ProductMixTests(unittest.TestCase):
    def test_build_and_solve_returns_feasible_known_optimum_for_default_data(self) -> None:
        _, quantities, total_profit = case05_product_mix.build_and_solve(
            case05_product_mix.products,
            case05_product_mix.profit,
            case05_product_mix.consumption,
            case05_product_mix.resources_data,
        )
        self.assertAlmostEqual(total_profit, 120000.0, places=6)
        self.assertAlmostEqual(quantities["A"], 1000.0, places=6)
        self.assertAlmostEqual(quantities["B"], 0.0, places=6)
        self.assertAlmostEqual(quantities["C"], 0.0, places=6)
        for resource, limit in case05_product_mix.resources_data.items():
            used = sum(
                case05_product_mix.consumption[p][resource] * quantities[p]
                for p in case05_product_mix.products
            )
            self.assertLessEqual(used, limit + 1e-8)


class TspTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(42)
        np.random.seed(42)

    def test_generate_tsp_data_structure_via_city_generator(self) -> None:
        cities = case_metaheuristic_tsp.gen_cities(8, seed=42)
        self.assertEqual(cities.shape, (8, 2))
        self.assertTrue(np.all(cities >= 0.0))
        self.assertTrue(np.all(cities <= 100.0))

    def test_distance_matrix_is_symmetric_and_greedy_tour_is_valid(self) -> None:
        cities = case_metaheuristic_tsp.gen_cities(8, seed=7)
        distances = case_metaheuristic_tsp.dist_matrix(cities)
        np.testing.assert_allclose(distances, distances.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-12)
        tour = case_metaheuristic_tsp.greedy_tsp(distances)
        self.assertEqual(len(tour), 8)
        self.assertEqual(set(tour), set(range(8)))
        self.assertGreater(case_metaheuristic_tsp.tour_length(tour, distances), 0.0)


if __name__ == "__main__":
    unittest.main()
