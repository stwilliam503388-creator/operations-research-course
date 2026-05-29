"""Case 03: least squares via closed form and gradient descent."""

from __future__ import annotations

import numpy as np


def generate_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, 80)
    noise = rng.normal(0.0, 1.2, size=x.shape)
    y = 3.0 * x + 5.0 + noise
    return x, y


def loss(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    pred = w * x + b
    return float(np.mean((pred - y) ** 2))


def closed_form_solution(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    w, b = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(w), float(b)


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    steps: int = 4000,
) -> tuple[float, float, list[float]]:
    w, b = 0.0, 0.0
    history: list[float] = []
    n = len(x)
    for step in range(steps):
        pred = w * x + b
        err = pred - y
        grad_w = 2.0 * float(np.dot(err, x)) / n
        grad_b = 2.0 * float(np.sum(err)) / n
        w -= lr * grad_w
        b -= lr * grad_b
        if step % 100 == 0 or step == steps - 1:
            history.append(loss(x, y, w, b))
    return w, b, history


def main() -> None:
    x, y = generate_data()
    initial_loss = loss(x, y, 0.0, 0.0)
    cf_w, cf_b = closed_form_solution(x, y)
    gd_w, gd_b, history = gradient_descent(x, y)
    final_loss = loss(x, y, gd_w, gd_b)
    closed_loss = loss(x, y, cf_w, cf_b)

    print("Least Squares Case")
    print(f"initial_loss={initial_loss:.6f}")
    print(f"closed_form_w={cf_w:.6f}, closed_form_b={cf_b:.6f}, loss={closed_loss:.6f}")
    print(f"gradient_w={gd_w:.6f}, gradient_b={gd_b:.6f}, loss={final_loss:.6f}")
    print(f"loss_decreased={final_loss < initial_loss}")
    print(f"close_to_closed_form={abs(gd_w - cf_w) < 0.05 and abs(gd_b - cf_b) < 0.25}")

    assert final_loss < initial_loss
    assert abs(gd_w - cf_w) < 0.05
    assert abs(gd_b - cf_b) < 0.25
    assert history[-1] <= history[0]


if __name__ == "__main__":
    main()
