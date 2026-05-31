"""案例 04：用小型约束二次规划演示均值-方差投资组合。"""


# 教学注释：把梯度、凸性、约束和步长选择对应到优化迭代过程。
# 收敛指标和残差帮助判断算法是否稳定接近最优解。



from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def portfolio_data() -> tuple[np.ndarray, np.ndarray]:
    mu = np.array([0.12, 0.09, 0.07, 0.05])
    std = np.array([0.22, 0.16, 0.10, 0.06])
    corr = np.array(
        [
            [1.0, 0.35, 0.20, 0.10],
            [0.35, 1.0, 0.25, 0.15],
            [0.20, 0.25, 1.0, 0.30],
            [0.10, 0.15, 0.30, 1.0],
        ]
    )
    cov = np.outer(std, std) * corr
    return mu, cov


def solve_portfolio(gamma: float, upper_bound: float = 0.65) -> tuple[np.ndarray, float, float]:
    mu, cov = portfolio_data()
    n = len(mu)

    def objective(w: np.ndarray) -> float:
        risk = float(w @ cov @ w)
        ret = float(mu @ w)
        return risk - gamma * ret

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, upper_bound)] * n
    x0 = np.full(n, 1.0 / n)
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(result.message)
    weights = result.x
    return weights, float(mu @ weights), float(weights @ cov @ weights)


def main() -> None:
    conservative = solve_portfolio(gamma=0.0)
    aggressive = solve_portfolio(gamma=1.2)

    for name, (weights, ret, risk) in [
        ("conservative", conservative),
        ("aggressive", aggressive),
    ]:
        print(f"{name}_weights={np.round(weights, 4)}")
        print(f"{name}_return={ret:.6f}")
        print(f"{name}_risk={risk:.6f}")
        print(f"{name}_sum={weights.sum():.6f}")
        assert abs(weights.sum() - 1.0) < 1e-6
        assert np.all(weights >= -1e-8)
        assert np.all(weights <= 0.650001)
        assert risk >= 0.0

    print(f"return_increased={aggressive[1] >= conservative[1]}")
    assert aggressive[1] >= conservative[1]


if __name__ == "__main__":
    main()
