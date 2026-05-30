"""Capstone: continuous optimization for marketing budget allocation."""
# 教学注释：把梯度、凸性、约束和步长选择对应到优化迭代过程。
# 收敛指标和残差帮助判断算法是否稳定接近最优解。


from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def channel_params() -> tuple[np.ndarray, np.ndarray]:
    # a controls scale, b controls how fast marginal returns saturate.
    return np.array([120.0, 95.0, 80.0]), np.array([0.08, 0.12, 0.18])


def revenue(x: np.ndarray) -> float:
    a, b = channel_params()
    return float(np.sum(a * np.log1p(b * x)))


def objective(x: np.ndarray) -> float:
    return -revenue(x)


def solve_with_slsqp(budget: float) -> tuple[np.ndarray, float]:
    n = 3
    bounds = [(0.0, 0.75 * budget)] * n
    constraints = [{"type": "ineq", "fun": lambda x: budget - float(np.sum(x))}]
    x0 = np.full(n, budget / n)
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x, revenue(result.x)


def project_to_budget_box(x: np.ndarray, budget: float) -> np.ndarray:
    x = np.clip(x, 0.0, 0.75 * budget)
    total = float(np.sum(x))
    if total <= budget:
        return x
    return x * (budget / total)


def solve_with_projected_gradient(budget: float, steps: int = 1500, lr: float = 0.8) -> tuple[np.ndarray, float]:
    a, b = channel_params()
    x = np.full(3, budget / 3)
    for _ in range(steps):
        grad_revenue = a * b / (1.0 + b * x)
        x = project_to_budget_box(x + lr * grad_revenue, budget)
    return x, revenue(x)


def run_budget_case(budget: float) -> dict[str, float]:
    slsqp_x, slsqp_rev = solve_with_slsqp(budget)
    pg_x, pg_rev = solve_with_projected_gradient(budget)
    baseline_x = np.full(3, budget / 3)
    baseline_rev = revenue(baseline_x)

    print(f"budget={budget:.1f}")
    print(f"  slsqp_x={np.round(slsqp_x, 3)}, revenue={slsqp_rev:.6f}, sum={slsqp_x.sum():.6f}")
    print(f"  pg_x={np.round(pg_x, 3)}, revenue={pg_rev:.6f}, sum={pg_x.sum():.6f}")
    print(f"  baseline_revenue={baseline_rev:.6f}")

    assert float(np.sum(slsqp_x)) <= budget + 1e-6
    assert float(np.sum(pg_x)) <= budget + 1e-6
    assert slsqp_rev >= baseline_rev - 1e-6
    assert pg_rev >= baseline_rev - 1e-4
    return {"budget": budget, "slsqp": slsqp_rev, "pg": pg_rev, "baseline": baseline_rev}


def main() -> None:
    print("Marketing Budget Capstone")
    rows = [run_budget_case(b) for b in [60.0, 100.0, 140.0]]
    print("sensitivity_revenues=", [round(row["slsqp"], 4) for row in rows])
    assert rows[2]["slsqp"] >= rows[0]["slsqp"]


if __name__ == "__main__":
    main()
