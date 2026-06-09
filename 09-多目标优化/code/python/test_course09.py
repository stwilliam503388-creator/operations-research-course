"""Unit tests for Course 09 multi-objective optimization examples.

Covered modules:
- case03_portfolio_multi: portfolio stats, dominance, nondominated sort
- case04_product_design: simulate_design
- case05_scheduling_multi: simulate_schedule, normalize_objectives
- case06_supply_chain_multi: dominates (three-objective)
- case07_nsga2: ZDT1, dominates, Individual, non_dominated_sort
"""

import math
import os
import random
import sys
import unittest

MODULE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, MODULE_DIR)

import case03_portfolio_multi as portfolio_multi
import case04_product_design as product_design
import case05_scheduling_multi as scheduling_multi
import case06_supply_chain_multi as supply_chain_multi
import case07_nsga2 as nsga2


class PortfolioMultiTests(unittest.TestCase):
    def test_random_returns_are_deterministic_and_bounded(self):
        returns_a = portfolio_multi.random_returns(n=6, seed=42)
        returns_b = portfolio_multi.random_returns(n=6, seed=42)
        self.assertEqual(len(returns_a), 6)
        self.assertEqual(returns_a, returns_b)
        for value in returns_a:
            self.assertGreaterEqual(value, 0.05)
            self.assertLessEqual(value, 0.15)

    def test_random_covariance_matrix_is_symmetric_with_positive_diagonal(self):
        cov = portfolio_multi.random_covariance_matrix(n=4, seed=7)
        self.assertEqual(len(cov), 4)
        for i in range(4):
            self.assertGreater(cov[i][i], 0.0)
            for j in range(4):
                self.assertAlmostEqual(cov[i][j], cov[j][i])

    def test_portfolio_stats_manual(self):
        weights = [0.5, 0.5]
        returns = [0.10, 0.20]
        cov = [[0.04, 0.01], [0.01, 0.09]]
        exp_ret, var = portfolio_multi.portfolio_stats(weights, returns, cov)
        # Expected return = 0.5*0.1 + 0.5*0.2 = 0.15
        self.assertAlmostEqual(exp_ret, 0.15)
        # Variance = 0.25*0.04 + 2*0.25*0.01 + 0.25*0.09 = 0.0375
        self.assertAlmostEqual(var, 0.0375)

    def test_is_dominated(self):
        # is_dominated(a, b) checks if b dominates a
        # Maximize return (index 0), minimize risk/variance (index 1)
        a = (0.1, 0.5, 0)
        b = (0.2, 0.3, 1)  # higher return, lower risk → b dominates a
        self.assertTrue(portfolio_multi.is_dominated(a, b))
        self.assertFalse(portfolio_multi.is_dominated(b, a))

    def test_nondominated_sort_returns_only_nondominated(self):
        points = [
            (0.10, 0.50, 0),
            (0.20, 0.30, 1),  # not dominated
            (0.18, 0.45, 2),  # dominated by (0.20, 0.30)
            (0.16, 0.25, 3),  # not dominated
            (0.12, 0.60, 4),  # dominated
        ]
        pareto = portfolio_multi.nondominated_sort(points)
        pareto_ids = {p[2] for p in pareto}
        self.assertIn(1, pareto_ids)
        self.assertIn(3, pareto_ids)
        self.assertNotIn(4, pareto_ids)


class Nsga2Tests(unittest.TestCase):
    def test_zdt1_boundary_values(self):
        # x = [0, 0, ..., 0]: f1=0, g=1, f2 = 1*(1-sqrt(0/1)) = 1
        f1, f2 = nsga2.zdt1([0.0] * 10)
        self.assertAlmostEqual(f1, 0.0)
        self.assertAlmostEqual(f2, 1.0)
        # x = [1, 0, ..., 0]: f1=1, g=1, f2 = 1*(1-sqrt(1/1)) = 0
        f1, f2 = nsga2.zdt1([1.0] + [0.0] * 9)
        self.assertAlmostEqual(f1, 1.0)
        self.assertAlmostEqual(f2, 0.0)

    def test_dominates(self):
        # (0.2, 0.8) dominates (0.3, 0.9) — better in both objectives (minimize)
        self.assertTrue(nsga2.dominates((0.2, 0.8), (0.3, 0.9)))
        # (0.2, 0.8) does NOT dominate (0.1, 0.9) — worse in f1
        self.assertFalse(nsga2.dominates((0.2, 0.8), (0.1, 0.9)))
        # Equal → not dominated
        self.assertFalse(nsga2.dominates((0.5, 0.5), (0.5, 0.5)))

    def test_individual_creation(self):
        random.seed(11)
        ind = nsga2.Individual(n_vars=5)
        self.assertEqual(len(ind.x), 5)
        for v in ind.x:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        ind.evaluate()
        self.assertIsNotNone(ind.fitness)
        self.assertEqual(len(ind.fitness), 2)


class ProductDesignTests(unittest.TestCase):
    def test_simulate_design_positive_and_monotone(self):
        perf_low_cost = product_design.simulate_design(cost=1.0, weight=2.0)
        perf_high_cost = product_design.simulate_design(cost=9.0, weight=2.0)
        self.assertGreater(perf_low_cost, 0.0)
        self.assertGreater(perf_high_cost, perf_low_cost)

    def test_simulate_design_lower_weight_gives_higher_performance(self):
        perf_heavy = product_design.simulate_design(cost=5.0, weight=3.0)
        perf_light = product_design.simulate_design(cost=5.0, weight=0.5)
        self.assertGreater(perf_light, perf_heavy)


class SchedulingMultiTests(unittest.TestCase):
    def test_simulate_schedule_returns_expected_keys(self):
        result = scheduling_multi.simulate_schedule(speed=2.0, batch=50, maintenance=False)
        self.assertIn("makespan", result)
        self.assertIn("cost", result)
        self.assertIn("energy", result)

    def test_maintenance_reduces_energy(self):
        no_maint = scheduling_multi.simulate_schedule(speed=3.0, batch=40, maintenance=False)
        with_maint = scheduling_multi.simulate_schedule(speed=3.0, batch=40, maintenance=True)
        self.assertLess(with_maint["energy"], no_maint["energy"])

    def test_higher_speed_reduces_makespan(self):
        slow = scheduling_multi.simulate_schedule(speed=1.0, batch=50, maintenance=False)
        fast = scheduling_multi.simulate_schedule(speed=5.0, batch=50, maintenance=False)
        self.assertLess(fast["makespan"], slow["makespan"])


class SupplyChainDominatesTests(unittest.TestCase):
    def test_three_objective_dominance(self):
        # dominates(a, b): a dominates b if a is <= in all and < in at least one (minimization)
        self.assertTrue(supply_chain_multi.dominates((1, 2, 3), (2, 3, 4)))
        self.assertFalse(supply_chain_multi.dominates((1, 2, 3), (1, 2, 3)))
        self.assertFalse(supply_chain_multi.dominates((1, 2, 3), (0, 3, 4)))


if __name__ == "__main__":
    unittest.main()
