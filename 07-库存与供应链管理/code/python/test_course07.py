"""Unit tests for course 07 inventory and supply chain examples.

These tests use only the standard library in the test code and focus on
deterministic helper logic.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parent
if str(COURSE_DIR) not in sys.path:
    sys.path.insert(0, str(COURSE_DIR))

import case03_eoq
import case04_newsvendor
import case05_bullwhip
import case06_contract
import case07_network


class EoqTests(unittest.TestCase):
    def test_eoq_matches_closed_form_and_integer_enumeration(self) -> None:
        q_star = case03_eoq.eoq(1000, 50, 1)
        best_q, best_cost = case03_eoq.enumerate_best(1000, 50, 1, 10)
        self.assertAlmostEqual(q_star, math.sqrt(100000.0), places=12)
        self.assertEqual(best_q, 316)
        self.assertAlmostEqual(best_cost, case03_eoq.total_cost(best_q, 1000, 50, 1, 10), places=12)
        self.assertLessEqual(best_cost, case03_eoq.total_cost(best_q - 1, 1000, 50, 1, 10))
        self.assertLessEqual(best_cost, case03_eoq.total_cost(best_q + 1, 1000, 50, 1, 10))


class NewsvendorTests(unittest.TestCase):
    def test_symmetric_costs_produce_mean_order_quantity_and_balanced_service_level(self) -> None:
        q_star = case04_newsvendor.newsvendor_optimal_q(1, 1, 100, 10)
        summary = case04_newsvendor.expected_profit(q_star, 1, 1, 100, 10, 5, 3)
        self.assertAlmostEqual(q_star, 100.0, places=9)
        self.assertAlmostEqual(summary["service_level"], 0.5, places=6)
        self.assertAlmostEqual(summary["stockout_rate"], 0.5, places=6)
        self.assertAlmostEqual(summary["service_level"] + summary["stockout_rate"], 1.0, places=9)
        self.assertAlmostEqual(summary["expected_leftover"], q_star - summary["expected_sales"], places=9)


class BullwhipTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(123)

    def test_simulation_returns_expected_structure_and_upstream_amplification(self) -> None:
        result = case05_bullwhip.simulate_bullwhip(
            num_periods=40,
            base_demand=100,
            demand_noise=8,
            smoothing_factor=0.35,
            safety_factor=1.2,
        )
        self.assertEqual(sorted(result.keys()), ["demand_series", "node_names", "nodes", "order_series", "stats"])
        self.assertEqual(result["node_names"], ["零售", "批发", "分销", "工厂"])
        self.assertEqual(len(result["order_series"]["零售"]), 40)
        self.assertEqual(result["order_series"]["工厂"][:4], [0, 0, 0, 0])
        amplification = result["stats"]["amplification"]
        self.assertGreater(amplification["批发"], amplification["零售"])
        self.assertGreater(amplification["分销"], amplification["零售"])
        self.assertGreater(amplification["工厂"], amplification["零售"])


class ContractTests(unittest.TestCase):
    def test_expected_sales_and_leftover_are_consistent(self) -> None:
        q_value = 100.0
        expected_sales = case06_contract.expected_sales(q_value, 80.0, 10.0)
        expected_leftover = case06_contract.expected_leftover(q_value, 80.0, 10.0)
        self.assertLessEqual(expected_sales, q_value)
        self.assertLessEqual(expected_sales, 80.0)
        self.assertAlmostEqual(expected_leftover, q_value - expected_sales, places=9)

    def test_decentralized_decision_reports_internal_consistency(self) -> None:
        decision = case06_contract.decentralized_decision(20.0, 12.0, 3.0, 500.0, 80.0)
        self.assertAlmostEqual(decision["service_level"], decision["critical_ratio"], places=9)
        self.assertAlmostEqual(decision["leftover"], decision["Q"] - decision["sales"], places=9)
        self.assertGreater(decision["profit_total"], decision["profit_retailer"])
        self.assertGreater(decision["profit_supplier"], 0.0)


class NetworkTests(unittest.TestCase):
    def test_distance_and_solution_evaluation_match_small_hand_computed_example(self) -> None:
        customers = [
            case07_network.Customer(1, 0.0, 0.0, 10.0),
            case07_network.Customer(2, 10.0, 0.0, 5.0),
        ]
        warehouses = [
            case07_network.Warehouse(1, 0.0, 0.0, 100.0, 20.0),
            case07_network.Warehouse(2, 10.0, 0.0, 120.0, 20.0),
        ]
        result = case07_network.evaluate_solution([0], warehouses, customers, transport_rate=1.0)
        self.assertAlmostEqual(case07_network.euclidean_dist(0.0, 0.0, 3.0, 4.0), 5.0, places=12)
        self.assertEqual(customers[0].id, 1)
        self.assertEqual(warehouses[1].id, 2)
        self.assertAlmostEqual(result["fixed_cost"], 100.0, places=12)
        self.assertAlmostEqual(result["transport_cost"], 50.0, places=12)
        self.assertAlmostEqual(result["total_cost"], 150.0, places=12)
        self.assertEqual(result["assignment"], {1: 1, 2: 1})

    def test_greedy_best_matches_bruteforce_on_tiny_instance(self) -> None:
        customers = [
            case07_network.Customer(1, 0.0, 0.0, 10.0),
            case07_network.Customer(2, 10.0, 0.0, 5.0),
        ]
        warehouses = [
            case07_network.Warehouse(1, 0.0, 0.0, 100.0, 20.0),
            case07_network.Warehouse(2, 10.0, 0.0, 120.0, 20.0),
        ]
        brute_force = case07_network.brute_force_best(warehouses, customers, transport_rate=1.0)
        greedy = case07_network.greedy_best(warehouses, customers, transport_rate=1.0)
        self.assertEqual(brute_force["open_whs"], [0])
        self.assertEqual(greedy["open_whs"], brute_force["open_whs"])
        self.assertAlmostEqual(greedy["total_cost"], brute_force["total_cost"], places=12)


if __name__ == "__main__":
    unittest.main()
