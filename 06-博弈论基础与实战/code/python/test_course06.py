"""Unit tests for course 06 game theory examples.

These tests cover deterministic helper logic in pricing, auctions,
bargaining, cooperation, and signaling modules.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

COURSE_DIR = Path(__file__).resolve().parent
if str(COURSE_DIR) not in sys.path:
    sys.path.insert(0, str(COURSE_DIR))

import case03_pricing
import case04_auction
import case05_bargaining
import case06_cooperation
import case07_signaling


class PricingTests(unittest.TestCase):
    def test_best_responses_identify_low_price_as_dominant_strategy(self) -> None:
        a_best, b_best = case03_pricing.find_best_responses()
        self.assertEqual(a_best, [1, 1])
        self.assertEqual(b_best, [1, 1])
        self.assertEqual(case03_pricing.PAYOFFS[1][1], (3, 3))

    def test_find_nash_equilibria_returns_unique_low_price_profile(self) -> None:
        equilibria = case03_pricing.find_nash_equilibria()
        self.assertEqual(equilibria, [(1, 1)])
        i, j = equilibria[0]
        self.assertEqual(case03_pricing.A_STRATEGIES[i], "低价")
        self.assertEqual(case03_pricing.B_STRATEGIES[j], "低价")


class AuctionTests(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(2024)

    def test_first_price_auction_uses_highest_bid_and_computes_payoff(self) -> None:
        valuations = np.array([0.2, 0.5, 0.9])
        winner, winner_value, winner_bid, winner_payoff = case04_auction.first_price_auction(
            n_bidders=3,
            valuations=valuations,
            bid_function=lambda v: 0.75 * v,
        )
        self.assertEqual(winner, 2)
        self.assertAlmostEqual(winner_value, 0.9, places=12)
        self.assertAlmostEqual(winner_bid, 0.675, places=12)
        self.assertAlmostEqual(winner_payoff, 0.225, places=12)

    def test_simulate_auctions_average_revenue_tracks_theory(self) -> None:
        avg_revenue, theoretical_revenue, sample_details = case04_auction.simulate_auctions(
            n_bidders=4,
            n_simulations=2000,
            seed=7,
        )
        self.assertAlmostEqual(theoretical_revenue, 3.0 / 5.0, places=12)
        self.assertAlmostEqual(avg_revenue, theoretical_revenue, delta=0.03)
        self.assertEqual(len(sample_details), 6)


class BargainingTests(unittest.TestCase):
    def test_two_round_solution_matches_backward_induction_formula(self) -> None:
        a_share, b_share = case05_bargaining.finite_round_bargaining(delta=0.8, total_rounds=2, total=1.0)
        self.assertAlmostEqual(a_share, 0.2, places=12)
        self.assertAlmostEqual(b_share, 0.8, places=12)
        self.assertAlmostEqual(a_share + b_share, 1.0, places=12)

    def test_many_rounds_converge_to_rubinstein_solution(self) -> None:
        a_finite, b_finite = case05_bargaining.finite_round_bargaining(delta=0.5, total_rounds=100)
        a_infinite, b_infinite = case05_bargaining.rubinstein_solution(delta=0.5)
        self.assertAlmostEqual(a_finite, a_infinite, places=10)
        self.assertAlmostEqual(b_finite, b_infinite, places=10)


class CooperationTests(unittest.TestCase):
    def test_shapley_values_are_efficient_and_in_core_for_default_game(self) -> None:
        values = case06_cooperation.shapley_values(
            case06_cooperation.PLAYERS,
            case06_cooperation.coalition_value,
        )
        self.assertAlmostEqual(sum(values.values()), case06_cooperation.coalition_value(0b111), places=10)
        self.assertAlmostEqual(values["A"], 26.666666666666668, places=10)
        self.assertAlmostEqual(values["B"], 39.166666666666664, places=10)
        self.assertAlmostEqual(values["C"], 34.166666666666664, places=10)
        self.assertEqual(
            case06_cooperation.check_core(
                values,
                case06_cooperation.PLAYERS,
                case06_cooperation.coalition_value,
            ),
            [],
        )


class SignalingTests(unittest.TestCase):
    def test_separating_equilibrium_satisfies_incentive_constraints(self) -> None:
        model = case07_signaling.SpenceModel(theta_L=50, theta_H=100, c_L=8, c_H=4, p=0.5)
        equilibrium = model.separating_equilibrium()
        self.assertTrue(model.separating_feasible)
        self.assertIsNotNone(equilibrium)
        self.assertAlmostEqual(equilibrium["e_H"], 9.375, places=12)
        self.assertAlmostEqual(equilibrium["e_L"], 0.0, places=12)
        self.assertTrue(equilibrium["ic_L"])
        self.assertTrue(equilibrium["ic_H"])
        self.assertGreaterEqual(equilibrium["e_H"], equilibrium["e_min"])
        self.assertLessEqual(equilibrium["e_H"], equilibrium["e_max"])

    def test_pooling_and_infeasible_separating_cases_behave_as_expected(self) -> None:
        pooling_model = case07_signaling.SpenceModel(theta_L=50, theta_H=100, c_L=8, c_H=4, p=0.5)
        pooling = pooling_model.pooling_equilibrium(e_pool=5)
        self.assertAlmostEqual(pooling["w_pool"], 75.0, places=12)
        self.assertTrue(pooling["no_deviation_H"])
        self.assertFalse(pooling["no_deviation_L"])

        infeasible_model = case07_signaling.SpenceModel(theta_L=50, theta_H=100, c_L=4, c_H=8, p=0.5)
        self.assertFalse(infeasible_model.separating_feasible)
        self.assertIsNone(infeasible_model.separating_equilibrium())


if __name__ == "__main__":
    unittest.main()
