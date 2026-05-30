"""Case 06: constrained nonlinear product design."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def objective(x: np.ndarray) -> float:
    length, width, height = x
    volume = length * width * height
    surface = length * width + length * height + width * height
    return 0.02 * surface - 3.0 * np.log1p(volume)


def material_budget_residual(x: np.ndarray) -> float:
    length, width, height = x
    material = 2.0 * (length * width + length * height + width * height)
    return 42.0 - material


def solve_design() -> tuple[np.ndarray, float, float]:
    x0 = np.array([2.0, 2.0, 2.0])
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.5, 5.0), (0.5, 5.0), (0.5, 5.0)],
        constraints=[{"type": "ineq", "fun": material_budget_residual}],
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x, float(objective(result.x)), float(material_budget_residual(result.x))


def main() -> None:
    x0 = np.array([2.0, 2.0, 2.0])
    initial_obj = float(objective(x0))
    solution, final_obj, residual = solve_design()

    print("Constrained Design Case")
    print(f"initial_objective={initial_obj:.6f}")
    print(f"final_objective={final_obj:.6f}")
    print(f"solution={np.round(solution, 4)}")
    print(f"budget_residual={residual:.6f}")
    print(f"improved={final_obj < initial_obj}")
    print(f"feasible={residual >= -1e-6}")
    print(f"budget_active={abs(residual) < 1e-4}")

    assert final_obj < initial_obj
    assert residual >= -1e-6
    assert abs(residual) < 1e-4
    assert np.all(solution >= 0.5 - 1e-8)
    assert np.all(solution <= 5.0 + 1e-8)


if __name__ == "__main__":
    main()
