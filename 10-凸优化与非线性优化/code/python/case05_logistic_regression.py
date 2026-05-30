"""Case 05: logistic regression with gradient descent and L2 regularization."""


# 教学注释：把梯度、凸性、约束和步长选择对应到优化迭代过程。
# 收敛指标和残差帮助判断算法是否稳定接近最优解。



from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def generate_data(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 240
    logins = rng.normal(0.0, 1.0, n)
    complaints = rng.normal(0.0, 1.0, n)
    X = np.column_stack([logins, complaints])
    true_w = np.array([-1.5, 2.0])
    logits = X @ true_w - 0.2
    p = sigmoid(logits)
    y = rng.binomial(1, p)
    return X, y


def loss_and_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, reg: float) -> tuple[float, np.ndarray, float]:
    n = len(y)
    p = sigmoid(X @ w + b)
    eps = 1e-12
    loss = -float(np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    loss += reg * float(w @ w)
    err = p - y
    grad_w = X.T @ err / n + 2.0 * reg * w
    grad_b = float(np.mean(err))
    return loss, grad_w, grad_b


def train(X: np.ndarray, y: np.ndarray, reg: float = 0.05, lr: float = 0.4, steps: int = 800):
    w = np.zeros(X.shape[1])
    b = 0.0
    initial_loss, _, _ = loss_and_grad(X, y, w, b, reg)
    for _ in range(steps):
        _, grad_w, grad_b = loss_and_grad(X, y, w, b, reg)
        w -= lr * grad_w
        b -= lr * grad_b
    final_loss, _, _ = loss_and_grad(X, y, w, b, reg)
    return w, b, initial_loss, final_loss


def main() -> None:
    X, y = generate_data()
    w, b, initial_loss, final_loss = train(X, y)
    probs = sigmoid(X @ w + b)
    pred = (probs >= 0.5).astype(int)
    accuracy = float(np.mean(pred == y))
    baseline = max(float(np.mean(y)), 1.0 - float(np.mean(y)))

    print("Logistic Regression Case")
    print(f"initial_loss={initial_loss:.6f}")
    print(f"final_loss={final_loss:.6f}")
    print(f"weights={np.round(w, 4)}, bias={b:.4f}")
    print(f"accuracy={accuracy:.4f}, baseline={baseline:.4f}")
    print(f"loss_decreased={final_loss < initial_loss}")
    print(f"beats_baseline={accuracy > baseline}")

    assert final_loss < initial_loss
    assert accuracy > baseline
    assert np.linalg.norm(w) < 5.0


if __name__ == "__main__":
    main()
