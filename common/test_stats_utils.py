"""Unit tests for :mod:`common.stats_utils`.

These deterministic tests exercise the shared statistical helpers that are
reused across several course examples.  They use only the standard library so
they run under the repository smoke checks without extra dependencies.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.stats_utils import (
    generate_normal,
    normal_cdf,
    normal_pdf,
    normal_ppf,
)


class NormalPdfTests(unittest.TestCase):
    def test_peak_at_mean(self) -> None:
        self.assertAlmostEqual(normal_pdf(0.0), 1.0 / math.sqrt(2.0 * math.pi), places=12)

    def test_known_value_with_mu_sigma(self) -> None:
        # Density at the mean of a N(2, 3) distribution.
        expected = 1.0 / (3.0 * math.sqrt(2.0 * math.pi))
        self.assertAlmostEqual(normal_pdf(2.0, mu=2.0, sigma=3.0), expected, places=12)

    def test_symmetry_about_mean(self) -> None:
        self.assertAlmostEqual(normal_pdf(-1.5), normal_pdf(1.5), places=12)
        self.assertAlmostEqual(
            normal_pdf(2.0, mu=5.0, sigma=2.0),
            normal_pdf(8.0, mu=5.0, sigma=2.0),
            places=12,
        )

    def test_density_is_positive_and_below_peak(self) -> None:
        peak = normal_pdf(0.0)
        for x in (-3.0, -1.0, 0.5, 2.5):
            value = normal_pdf(x)
            self.assertGreater(value, 0.0)
            self.assertLessEqual(value, peak)

    def test_non_positive_sigma_returns_zero(self) -> None:
        self.assertEqual(normal_pdf(1.0, sigma=0.0), 0.0)
        self.assertEqual(normal_pdf(1.0, sigma=-2.0), 0.0)

    def test_integrates_to_one(self) -> None:
        # Coarse midpoint rule over [-8, 8] should be close to 1.
        step = 0.001
        x = -8.0
        total = 0.0
        while x < 8.0:
            total += normal_pdf(x + step / 2.0) * step
            x += step
        self.assertAlmostEqual(total, 1.0, places=4)


class NormalCdfTests(unittest.TestCase):
    def test_value_at_mean_is_half(self) -> None:
        self.assertAlmostEqual(normal_cdf(0.0), 0.5, places=7)
        self.assertAlmostEqual(normal_cdf(4.0, mu=4.0, sigma=2.5), 0.5, places=7)

    def test_known_quantiles(self) -> None:
        self.assertAlmostEqual(normal_cdf(1.96), 0.975, places=4)
        self.assertAlmostEqual(normal_cdf(-1.96), 0.025, places=4)
        self.assertAlmostEqual(normal_cdf(1.0), 0.8413, places=3)

    def test_reflection_symmetry(self) -> None:
        for z in (0.3, 1.0, 2.2, 3.5):
            self.assertAlmostEqual(normal_cdf(-z), 1.0 - normal_cdf(z), places=6)

    def test_monotonic_increasing(self) -> None:
        xs = [-3.0, -1.0, 0.0, 0.5, 2.0, 4.0]
        values = [normal_cdf(x) for x in xs]
        for earlier, later in zip(values, values[1:]):
            self.assertLess(earlier, later)

    def test_bounded_between_zero_and_one(self) -> None:
        for x in (-50.0, -5.0, 0.0, 5.0, 50.0):
            value = normal_cdf(x)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_extreme_tails_saturate(self) -> None:
        self.assertEqual(normal_cdf(100.0), 1.0)
        self.assertEqual(normal_cdf(-100.0), 0.0)

    def test_scaling_with_sigma(self) -> None:
        # One standard deviation above the mean is invariant to scaling.
        self.assertAlmostEqual(
            normal_cdf(7.0, mu=4.0, sigma=3.0),
            normal_cdf(1.0),
            places=7,
        )

    def test_degenerate_sigma_is_step_function(self) -> None:
        self.assertEqual(normal_cdf(0.0, mu=1.0, sigma=0.0), 0.0)
        self.assertEqual(normal_cdf(1.0, mu=1.0, sigma=0.0), 0.5)
        self.assertEqual(normal_cdf(2.0, mu=1.0, sigma=0.0), 1.0)
        self.assertEqual(normal_cdf(0.0, mu=1.0, sigma=-1.0), 0.0)


class NormalPpfTests(unittest.TestCase):
    def test_median_is_mean(self) -> None:
        self.assertAlmostEqual(normal_ppf(0.5), 0.0, places=6)
        self.assertAlmostEqual(normal_ppf(0.5, mu=3.0, sigma=2.0), 3.0, places=6)

    def test_known_quantile(self) -> None:
        self.assertAlmostEqual(normal_ppf(0.975), 1.96, places=2)

    def test_round_trip_with_cdf(self) -> None:
        for p in (0.05, 0.25, 0.5, 0.75, 0.95):
            self.assertAlmostEqual(normal_cdf(normal_ppf(p)), p, places=4)

    def test_round_trip_with_mu_sigma(self) -> None:
        for p in (0.1, 0.4, 0.9):
            x = normal_ppf(p, mu=10.0, sigma=4.0)
            self.assertAlmostEqual(normal_cdf(x, mu=10.0, sigma=4.0), p, places=4)

    def test_monotonic_increasing(self) -> None:
        ps = [0.05, 0.2, 0.5, 0.8, 0.95]
        values = [normal_ppf(p) for p in ps]
        for earlier, later in zip(values, values[1:]):
            self.assertLess(earlier, later)

    def test_boundary_probabilities(self) -> None:
        self.assertEqual(normal_ppf(0.0), -float("inf"))
        self.assertEqual(normal_ppf(-0.5), -float("inf"))
        self.assertEqual(normal_ppf(1.0), float("inf"))
        self.assertEqual(normal_ppf(1.5), float("inf"))

    def test_degenerate_sigma_returns_mean(self) -> None:
        self.assertEqual(normal_ppf(0.3, mu=2.0, sigma=0.0), 2.0)
        self.assertEqual(normal_ppf(0.7, mu=-1.0, sigma=-3.0), -1.0)


class GenerateNormalTests(unittest.TestCase):
    def test_length_even_odd_and_zero(self) -> None:
        self.assertEqual(len(generate_normal(10)), 10)
        self.assertEqual(len(generate_normal(7)), 7)
        self.assertEqual(len(generate_normal(1)), 1)
        self.assertEqual(generate_normal(0), [])

    def test_returns_floats(self) -> None:
        for value in generate_normal(6):
            self.assertIsInstance(value, float)

    def test_is_deterministic_under_seed(self) -> None:
        random.seed(1234)
        first = generate_normal(8)
        random.seed(1234)
        second = generate_normal(8)
        self.assertEqual(first, second)

    def test_sample_statistics_are_reasonable(self) -> None:
        random.seed(2024)
        samples = generate_normal(20000, mu=5.0, sigma=2.0)
        self.assertAlmostEqual(statistics.fmean(samples), 5.0, delta=0.1)
        self.assertAlmostEqual(statistics.pstdev(samples), 2.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
