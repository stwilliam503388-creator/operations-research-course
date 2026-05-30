"""Case 07: nonconvex optimization can depend on the initial point."""


# 教学注释：把梯度、凸性、约束和步长选择对应到优化迭代过程。
# 收敛指标和残差帮助判断算法是否稳定接近最优解。



from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def f_scalar(x: float) -> float:
    return float(np.sin(3.0 * x) + 0.08 * x * x)


def solve_from(start: float) -> tuple[float, float, bool]:
    result = minimize(lambda z: f_scalar(float(z[0])), np.array([start]), method="BFGS")
    return float(result.x[0]), float(result.fun), bool(result.success)


def main() -> None:
    starts = [-4.0, -2.0, 0.0, 2.0, 4.0]
    results = [solve_from(s) for s in starts]

    print("Nonconvex Pitfalls Case")
    for start, (x_star, value, success) in zip(starts, results):
        print(f"start={start:+.1f} -> x={x_star:+.6f}, f={value:+.6f}, success={success}")

    rounded_locations = {round(x_star, 2) for x_star, _, _ in results}
    best = min(results, key=lambda item: item[1])
    print(f"distinct_solutions={len(rounded_locations)}")
    print(f"best_found_x={best[0]:.6f}, best_found_value={best[1]:.6f}")

    assert all(success for _, _, success in results)
    assert len(rounded_locations) >= 2


if __name__ == "__main__":
    main()
