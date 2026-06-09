"""Unit tests for course 01 probability and statistics examples.

These tests cover deterministic statistical helpers in case03_distributions,
case05_bayesian, case06_hypothesis, and case07_monte_carlo.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from case03_distributions import exponential_cdf, pearson_correlation, sample_quantile
from case05_bayesian import bayes_update, bayesian_sequential_update, format_percent
from case06_hypothesis import confidence_interval_prop_diff, two_proportion_z_test
from case07_monte_carlo import cholesky_decomposition


class DistributionFunctionTests(unittest.TestCase):
    """Tests for introductory distribution utilities."""

    def test_exponential_cdf_known_values_and_monotonicity(self) -> None:
        self.assertAlmostEqual(exponential_cdf(0.0, 2.0), 0.0, places=12)
        self.assertAlmostEqual(exponential_cdf(1.0, 1.0), 1.0 - 2.718281828459045 ** (-1.0), places=12)
        self.assertAlmostEqual(exponential_cdf(2.0, 0.5), 1.0 - 2.718281828459045 ** (-1.0), places=12)

        xs = [0.0, 0.5, 1.0, 2.0, 4.0]
        values = [exponential_cdf(x, 1.2) for x in xs]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later)

    def test_sample_quantile_for_quartiles_and_median(self) -> None:
        data = [1, 2, 3, 4, 5]
        self.assertEqual(sample_quantile(data, 0.5), 3)
        self.assertEqual(sample_quantile(data, 0.25), 2)
        self.assertEqual(sample_quantile(data, 0.75), 4)

    def test_pearson_correlation_for_positive_negative_and_zero_cases(self) -> None:
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0, places=12)
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [6, 4, 2]), -1.0, places=12)
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [1, 0, 1]), 0.0, places=12)


class BayesianUpdateTests(unittest.TestCase):
    """Tests for Bayesian updating helpers."""

    def test_bayes_update_disease_screening_example(self) -> None:
        posterior, evidence = bayes_update(0.001, 0.99, 0.01)
        self.assertAlmostEqual(evidence, 0.01098, places=8)
        self.assertAlmostEqual(posterior, 0.0901639344, places=8)

    def test_bayesian_sequential_update_is_monotonic_for_repeated_positives(self) -> None:
        posteriors = bayesian_sequential_update(0.001, 0.99, 0.01, 4)
        for earlier, later in zip(posteriors, posteriors[1:]):
            self.assertLess(earlier, later)

    def test_format_percent(self) -> None:
        self.assertEqual(format_percent(0.5), "50.00%")


class HypothesisTestingTests(unittest.TestCase):
    """Tests for two-proportion inference helpers."""

    def test_two_proportion_z_test_equal_proportions_is_not_significant(self) -> None:
        z_stat, p_value, *_ = two_proportion_z_test(100, 50, 120, 60)
        self.assertAlmostEqual(z_stat, 0.0, places=12)
        self.assertGreater(p_value, 0.05)

    def test_two_proportion_z_test_sign_matches_difference(self) -> None:
        z_stat, p_value, *_ = two_proportion_z_test(100, 40, 100, 60)
        self.assertLess(z_stat, 0.0)
        self.assertLess(p_value, 0.05)

    def test_confidence_interval_contains_true_difference(self) -> None:
        low, high, diff = confidence_interval_prop_diff(1000, 600, 1000, 500)
        self.assertAlmostEqual(diff, 0.1, places=12)
        self.assertLessEqual(low, 0.1)
        self.assertGreaterEqual(high, 0.1)


class MonteCarloLinearAlgebraTests(unittest.TestCase):
    """Tests for the Cholesky decomposition helper."""

    @staticmethod
    def _reconstruct(lower: list[float]) -> list[float]:
        n = int(len(lower) ** 0.5)
        product = [0.0] * (n * n)
        for i in range(n):
            for j in range(n):
                total = 0.0
                for k in range(n):
                    total += lower[i * n + k] * lower[j * n + k]
                product[i * n + j] = total
        return product

    def test_cholesky_identity_matrix(self) -> None:
        lower = cholesky_decomposition([1.0, 0.0, 0.0, 1.0])
        self.assertEqual(lower, [1.0, 0.0, 0.0, 1.0])

    def test_cholesky_reconstructs_original_matrix(self) -> None:
        matrix = [4.0, 2.0, 2.0, 3.0]
        lower = cholesky_decomposition(matrix)
        reconstructed = self._reconstruct(lower)
        for expected, actual in zip(matrix, reconstructed):
            self.assertAlmostEqual(actual, expected, places=9)


if __name__ == "__main__":
    random.seed(0)
    unittest.main()
