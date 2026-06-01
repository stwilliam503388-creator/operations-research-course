"""Unit tests for course 08 stochastic and robust optimization examples.

These tests cover deterministic helper logic and small reproducible model
properties.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

COURSE_DIR = Path(__file__).resolve().parent
if str(COURSE_DIR) not in sys.path:
    sys.path.insert(0, str(COURSE_DIR))

import case01_newsvendor
import case02_two_stage
import case03_robust_portfolio


class NewsvendorTests(unittest.TestCase):
    def test_profit_and_analytical_solution_match_known_values(self) -> None:
        q_star, critical_ratio = case01_newsvendor.analytical_solution()
        self.assertAlmostEqual(case01_newsvendor.profit(100, 80), 32.0, places=12)
        self.assertAlmostEqual(critical_ratio, 5.0 / 9.0, places=12)
        self.assertGreater(q_star, case01_newsvendor.mu)

    def test_grid_search_best_q_is_close_to_analytical_solution(self) -> None:
        best_q, best_profit, _, _ = case01_newsvendor.search_best_q(
            q_range=(95, 105),
            step=1,
            n_scenarios=2000,
            seed=42,
        )
        q_star, _ = case01_newsvendor.analytical_solution()
        self.assertEqual(best_q, 102)
        self.assertAlmostEqual(best_q, q_star, delta=1.0)
        self.assertGreater(best_profit, case01_newsvendor.simulate_expected_profit(95, n_scenarios=2000, seed=42))


class TwoStageTests(unittest.TestCase):
    def test_two_stage_solution_is_feasible_for_each_scenario(self) -> None:
        x_value, y_values, objective = case02_two_stage.solve_two_stage(case02_two_stage.scenarios)
        self.assertLessEqual(x_value, case02_two_stage.max_raw)
        self.assertEqual(y_values.shape, (len(case02_two_stage.scenarios), 2))
        for i, scenario in enumerate(case02_two_stage.scenarios):
            self.assertLessEqual(y_values[i][0] + y_values[i][1], x_value + 1e-9)
            self.assertLessEqual(y_values[i][0], scenario["demand_A"] + 1e-9)
            self.assertLessEqual(y_values[i][1], scenario["demand_B"] + 1e-9)
        self.assertGreater(objective, 0.0)

    def test_generated_scenarios_are_deterministic_and_probabilities_sum_to_one(self) -> None:
        scenarios = case02_two_stage.generate_n_scenarios(3, seed=42)
        self.assertEqual(len(scenarios), 3)
        self.assertAlmostEqual(sum(s["prob"] for s in scenarios), 1.0, places=12)
        self.assertEqual(scenarios[0]["demand_A"], 179)
        self.assertEqual(scenarios[0]["demand_B"], 161)
        for scenario in scenarios:
            self.assertGreaterEqual(scenario["demand_A"], 50)
            self.assertGreaterEqual(scenario["demand_B"], 50)


class RobustPortfolioTests(unittest.TestCase):
    def test_mean_and_box_models_return_valid_weight_vectors(self) -> None:
        w_mean, ret_mean = case03_robust_portfolio.solve_mean_model()
        w_box, ret_box = case03_robust_portfolio.solve_box_robust()
        self.assertAlmostEqual(np.sum(w_mean), 1.0, places=9)
        self.assertAlmostEqual(np.sum(w_box), 1.0, places=9)
        self.assertTrue(np.all(w_mean >= -1e-12))
        self.assertTrue(np.all(w_box >= -1e-12))
        self.assertGreaterEqual(ret_mean, ret_box)

    def test_budget_robust_value_is_monotone_and_matches_worst_case_return(self) -> None:
        gamma_values = [0, 1, 2.5, case03_robust_portfolio.N]
        objectives = []
        for gamma in gamma_values:
            weights, objective = case03_robust_portfolio.solve_budget_robust(gamma)
            objectives.append(objective)
            self.assertAlmostEqual(np.sum(weights), 1.0, places=9)
            self.assertAlmostEqual(
                objective,
                case03_robust_portfolio.worst_case_return(weights, gamma),
                places=9,
            )
        self.assertGreaterEqual(objectives[0], objectives[1])
        self.assertGreaterEqual(objectives[1], objectives[2])
        self.assertAlmostEqual(objectives[2], objectives[3], places=9)


if __name__ == "__main__":
    unittest.main()
