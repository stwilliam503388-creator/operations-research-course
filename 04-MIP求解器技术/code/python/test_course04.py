"""Unit tests for course 04 MIP examples.

The tests stay solver-light and focus on deterministic data generation,
geometry, and helper computations.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

COURSE_DIR = Path(__file__).resolve().parent
if str(COURSE_DIR) not in sys.path:
    sys.path.insert(0, str(COURSE_DIR))

import case01_vrp
import case02_scheduling
import case03_portfolio


class VrpDataTests(unittest.TestCase):
    def test_generate_data_returns_expected_shapes(self) -> None:
        depo, orders, rider_capacity = case01_vrp.generate_data(6, 2, seed=7)
        self.assertEqual(depo.shape, (2,))
        self.assertEqual(orders.shape, (6, 2))
        self.assertTrue(np.issubdtype(orders.dtype, np.floating))
        self.assertIsInstance(rider_capacity, (int, np.integer))
        self.assertGreater(int(rider_capacity), 0)

    def test_distance_matrix_is_symmetric_with_zero_diagonal_and_metric(self) -> None:
        depo, orders, _ = case01_vrp.generate_data(5, 2, seed=11)
        dist = case01_vrp.build_distance_matrix(depo, orders)
        np.testing.assert_allclose(dist, dist.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-12)
        n = dist.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    self.assertLessEqual(dist[i, k], dist[i, j] + dist[j, k] + 1e-10)


class SchedulingTests(unittest.TestCase):
    def test_generate_instance_returns_consistent_routes_processing_and_setup(self) -> None:
        routes, p, setup, product_types, n_machines = case02_scheduling.generate_instance(
            n_jobs=8, n_machines=5, seed=123
        )
        expected_routes = {
            0: [0, 2, 4],
            1: [1, 0, 3, 4],
            2: [0, 1, 2, 3, 4],
        }
        self.assertEqual(len(routes), 8)
        self.assertEqual(len(product_types), 8)
        self.assertEqual(n_machines, 5)
        for j, route in enumerate(routes):
            self.assertEqual(route, expected_routes[product_types[j]])
            for m in route:
                self.assertIn((j, m), p)
                self.assertGreater(p[(j, m)], 0)
        for (j, k, m), value in setup.items():
            self.assertNotEqual(j, k)
            self.assertIn(m, routes[j])
            self.assertIn(m, routes[k])
            if product_types[j] == product_types[k]:
                self.assertGreaterEqual(value, 3)
                self.assertLessEqual(value, 7)
            else:
                self.assertGreaterEqual(value, 20)
                self.assertLessEqual(value, 44)

    def test_estimate_manual_makespan_accounts_for_order_and_setup_times(self) -> None:
        routes = [[0], [0]]
        p = {(0, 0): 3, (1, 0): 4}
        setup = {(0, 1, 0): 2, (1, 0, 0): 1}
        makespan = case02_scheduling.estimate_manual_makespan(routes, p, setup, n_jobs=2)
        self.assertAlmostEqual(makespan, 9.0, places=10)


class PortfolioTests(unittest.TestCase):
    def test_generate_market_data_has_expected_shapes_and_covariance_properties(self) -> None:
        returns, cov_matrix, rf = case03_portfolio.generate_market_data(12, seed=5, rf=0.03)
        self.assertEqual(returns.shape, (12,))
        self.assertEqual(cov_matrix.shape, (12, 12))
        np.testing.assert_allclose(cov_matrix, cov_matrix.T, atol=1e-12)
        self.assertTrue(np.all(np.diag(cov_matrix) > 0.0))
        self.assertAlmostEqual(rf, 0.03, places=12)

    def test_compute_portfolio_stats_matches_manual_calculation(self) -> None:
        weights = np.array([0.25, 0.75])
        returns = np.array([0.08, 0.12])
        cov_matrix = np.array([[0.04, 0.01], [0.01, 0.09]])
        exp_return, volatility, sharpe = case03_portfolio.compute_portfolio_stats(
            weights, returns, cov_matrix, rf=0.02
        )
        expected_return = float(weights @ returns)
        expected_volatility = float(np.sqrt(weights @ cov_matrix @ weights))
        self.assertAlmostEqual(exp_return, expected_return, places=12)
        self.assertAlmostEqual(volatility, expected_volatility, places=12)
        self.assertAlmostEqual(sharpe, (expected_return - 0.02) / expected_volatility, places=12)

    def test_build_comparison_returns_normalized_baseline_and_random_portfolios(self) -> None:
        returns, cov_matrix, rf = case03_portfolio.generate_market_data(10, seed=19)
        comparison = case03_portfolio.build_comparison(returns, cov_matrix, rf, k=3, seed=19)
        self.assertEqual(len(comparison), 2)
        self.assertEqual(comparison[0]["name"], "等权-全选")
        self.assertEqual(comparison[1]["name"], "等权-随机")
        self.assertAlmostEqual(np.sum(comparison[0]["weights"]), 1.0, places=12)
        self.assertAlmostEqual(np.sum(comparison[1]["weights"]), 1.0, places=12)
        self.assertEqual(np.count_nonzero(comparison[1]["weights"]), 3)


if __name__ == "__main__":
    unittest.main()
