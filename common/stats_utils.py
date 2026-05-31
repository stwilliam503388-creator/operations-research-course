"""Small statistical utilities shared by teaching examples."""

from __future__ import annotations

import math
import random


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Return the normal probability density at x."""
    if sigma <= 0:
        return 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Return an Abramowitz-Stegun approximation of the normal CDF."""
    if sigma <= 0:
        return 0.5 if x == mu else (1.0 if x > mu else 0.0)
    z = (x - mu) / sigma
    if z < -8.0:
        return 0.0
    if z > 8.0:
        return 1.0

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1.0 if z >= 0 else -1.0
    z_abs = abs(z) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs)
    return 0.5 * (1.0 + sign * y)


def normal_ppf(
    p: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Return the normal quantile by bisection against normal_cdf()."""
    if sigma <= 0:
        return mu
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")

    lo, hi = mu - 10.0 * sigma, mu + 10.0 * sigma
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        cdf_mid = normal_cdf(mid, mu, sigma)
        if abs(cdf_mid - p) < tol:
            return mid
        if cdf_mid < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def generate_normal(n: int, mu: float = 0.0, sigma: float = 1.0) -> list[float]:
    """Generate n normal samples with the Box-Muller transform."""
    samples: list[float] = []
    for _ in range(n // 2 + 1):
        u1 = max(random.random(), 1e-12)
        u2 = random.random()
        radius = math.sqrt(-2.0 * math.log(u1))
        samples.append(mu + sigma * radius * math.cos(2.0 * math.pi * u2))
        samples.append(mu + sigma * radius * math.sin(2.0 * math.pi * u2))
    return samples[:n]
