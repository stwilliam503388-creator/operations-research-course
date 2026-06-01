"""Unit tests for Course 10 convex and nonlinear optimization examples.

Covered modules:
- case03_least_squares: generate_data, loss, closed_form_solution, gradient_descent
- case04_portfolio_qp: portfolio_data, solve_portfolio
- case05_logistic_regression: sigmoid, generate_data, loss_and_grad, train
- case06_constrained_design: objective, material_budget_residual, solve_design
- case07_nonconvex_pitfalls: f_scalar, solve_from
"""

import os
import sys
import unittest

import numpy as np

MODULE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, MODULE_DIR)

import case03_least_squares as least_squares
import case04_portfolio_qp as portfolio_qp
import case05_logistic_regression as logistic_regression
import case06_constrained_design as constrained_design
import case07_nonconvex_pitfalls as nonconvex_pitfalls


class LeastSquaresTests(unittest.TestCase):
    def test_generate_data_shapes_and_determinism(self):
        x1, y1 = least_squares.generate_data(seed=42)
        x2, y2 = least_squares.generate_data(seed=42)
        self.assertEqual(x1.shape, (80,))
        self.assertEqual(y1.shape, (80,))
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_loss_zero_for_perfect_fit(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([5.0, 7.0, 9.0])  # y = 2*x + 3
        self.assertAlmostEqual(least_squares.loss(x, y, 2.0, 3.0), 0.0, places=10)

    def test_loss_positive_for_bad_fit(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([5.0, 7.0, 9.0])
        self.assertGreater(least_squares.loss(x, y, 0.0, 0.0), 0.0)

    def test_closed_form_solution_recovers_true_params(self):
        x, y = least_squares.generate_data(seed=42)
        w, b = least_squares.closed_form_solution(x, y)
        # True relationship: y ≈ 3*x + 5
        self.assertAlmostEqual(w, 3.0, delta=0.1)
        self.assertAlmostEqual(b, 5.0, delta=0.5)

    def test_gradient_descent_converges(self):
        x, y = least_squares.generate_data(seed=42)
        w, b, history = least_squares.gradient_descent(x, y, lr=0.01, steps=4000)
        # Loss should decrease
        self.assertLess(history[-1], history[0])
        # Should be close to closed form
        cf_w, cf_b = least_squares.closed_form_solution(x, y)
        self.assertAlmostEqual(w, cf_w, delta=0.05)
        self.assertAlmostEqual(b, cf_b, delta=0.25)


class PortfolioQpTests(unittest.TestCase):
    def test_portfolio_data_shapes(self):
        mu, cov = portfolio_qp.portfolio_data()
        self.assertEqual(mu.shape, (4,))
        self.assertEqual(cov.shape, (4, 4))
        # Covariance should be symmetric
        np.testing.assert_allclose(cov, cov.T)
        # Diagonal should be positive
        for i in range(4):
            self.assertGreater(cov[i, i], 0.0)

    def test_solve_portfolio_feasible(self):
        weights, ret, risk = portfolio_qp.solve_portfolio(gamma=0.5)
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertTrue(np.all(weights >= -1e-8))
        self.assertTrue(np.all(weights <= 0.650001))
        self.assertGreater(ret, 0.0)
        self.assertGreater(risk, 0.0)


class LogisticRegressionTests(unittest.TestCase):
    def test_sigmoid_known_values(self):
        self.assertAlmostEqual(logistic_regression.sigmoid(np.array(0.0)).item(), 0.5)
        self.assertGreater(logistic_regression.sigmoid(np.array(5.0)).item(), 0.99)
        self.assertLess(logistic_regression.sigmoid(np.array(-5.0)).item(), 0.01)

    def test_generate_data_shapes_and_determinism(self):
        X1, y1 = logistic_regression.generate_data(seed=7)
        X2, y2 = logistic_regression.generate_data(seed=7)
        self.assertEqual(X1.shape[0], 240)
        self.assertEqual(X1.shape[1], 2)
        self.assertEqual(y1.shape, (240,))
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_training_reduces_loss(self):
        X, y = logistic_regression.generate_data(seed=7)
        w, b, initial_loss, final_loss = logistic_regression.train(X, y)
        self.assertLess(final_loss, initial_loss)


class ConstrainedDesignTests(unittest.TestCase):
    def test_objective_returns_scalar(self):
        val = constrained_design.objective(np.array([2.0, 2.0, 2.0]))
        self.assertIsInstance(val, float)

    def test_material_budget_residual(self):
        # For x=[2,2,2]: surface = 2*(2*2 + 2*2 + 2*2) = 24. Residual = 42 - 24 = 18
        residual = constrained_design.material_budget_residual(np.array([2.0, 2.0, 2.0]))
        self.assertAlmostEqual(residual, 18.0)

    def test_solve_design_feasible(self):
        x, obj, residual = constrained_design.solve_design()
        self.assertEqual(len(x), 3)
        self.assertGreaterEqual(residual, -1e-6)  # feasible
        self.assertTrue(np.all(x >= 0.5 - 1e-8))
        self.assertTrue(np.all(x <= 5.0 + 1e-8))


class NonconvexPitfallsTests(unittest.TestCase):
    def test_f_scalar_at_zero(self):
        # f(0) = sin(0) + 0.08*0 = 0
        self.assertAlmostEqual(nonconvex_pitfalls.f_scalar(0.0), 0.0)

    def test_solve_from_different_starts_give_different_solutions(self):
        x1, v1, s1 = nonconvex_pitfalls.solve_from(-4.0)
        x2, v2, s2 = nonconvex_pitfalls.solve_from(2.0)
        self.assertTrue(s1)
        self.assertTrue(s2)
        # Different starts should find different local minima
        self.assertNotAlmostEqual(x1, x2, places=1)


if __name__ == "__main__":
    unittest.main()
