"""
投资组合双目标优化 — 随机权重枚举法求帕累托前沿

问题描述：
  在 N=10 个资产中分配资金，同时优化两个冲突目标：
    目标1：最大化期望收益 E(r)
    目标2：最小化组合风险（方差） σ²

方法：
  随机生成 10000 组权重向量（Dirichlet 分布），计算每个组合的期望收益和方差，
  再通过 O(N²) 非支配筛选提取帕累托前沿。

输出：
  - 帕累托前沿上的点数
  - 每个点的 (风险, 收益) 对
  - 前沿是否单调递增（风险越高收益越高）
  - 验证：前沿上任意两点满足支配关系
"""


# 教学注释：关注多个目标之间的冲突、Pareto 支配关系和权衡参数。
# 输出的解集用于筛选可解释、可落地的折中方案。



import random
import math


def random_covariance_matrix(n: int, seed: int = 42) -> list[list[float]]:
    """生成一个随机半正定协方差矩阵（通过随机相关矩阵 + 随机标准差）。"""
    rng = random.Random(seed)
    # 随机标准差（5%~15%）
    stds = [0.05 + 0.10 * rng.random() for _ in range(n)]
    # 随机相关矩阵（保证对称且对角为1）
    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        corr[i][i] = 1.0
        for j in range(i + 1, n):
            val = rng.uniform(-0.3, 0.8)  # 控制相关性范围
            corr[i][j] = val
            corr[j][i] = val
    # Cov = diag(std) * Corr * diag(std)
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cov[i][j] = stds[i] * stds[j] * corr[i][j]
    return cov


def random_returns(n: int, seed: int = 7) -> list[float]:
    """生成随机期望收益率（5%~15%）。"""
    rng = random.Random(seed)
    return [0.05 + 0.10 * rng.random() for _ in range(n)]


def portfolio_stats(weights: list[float],
                    returns: list[float],
                    cov: list[list[float]]) -> tuple[float, float]:
    """计算组合的期望收益和方差。"""
    n = len(weights)
    # 期望收益
    exp_return = sum(w * r for w, r in zip(weights, returns))
    # 方差 = w^T * Cov * w
    var = 0.0
    for i in range(n):
        for j in range(n):
            var += weights[i] * weights[j] * cov[i][j]
    return exp_return, var


def random_weights(n: int, rng: random.Random) -> list[float]:
    """生成一组随机权重（Dirichlet 分布，和为1）。"""
    xs = [rng.random() for _ in range(n)]
    total = sum(xs)
    return [x / total for x in xs]


def is_dominated(a: tuple[float, float, int],
                 b: tuple[float, float, int]) -> bool:
    """
    判断 a 是否被 b 支配。
    目标：最大化收益（第0维），最小化风险/方差（第1维）。
    返回 True 如果 b 支配 a（即 b 在所有目标上不劣于 a，且至少一个严格更优）。
    """
    # b 收益 >= a 收益 且 b 风险 <= a 风险
    b_not_worse = b[0] >= a[0] and b[1] <= a[1]
    b_strict = b[0] > a[0] or b[1] < a[1]
    return b_not_worse and b_strict


def nondominated_sort(points: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    """
    O(N²) 非支配筛选。
    points: 每个元素为 (expected_return, variance, index)
    返回帕累托前沿上的点。
    """
    n = len(points)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if is_dominated(points[j], points[i]):
                dominated[j] = True
            elif is_dominated(points[i], points[j]):
                dominated[i] = True
                break
    return [p for i, p in enumerate(points) if not dominated[i]]


def is_monotonic_increasing(frontier: list[tuple[float, float]]) -> bool:
    """检查前沿是否单调递增（风险升序时收益不降）。"""
    if len(frontier) <= 1:
        return True
    sorted_frontier = sorted(frontier, key=lambda x: x[1])  # 按风险排序
    for k in range(1, len(sorted_frontier)):
        if sorted_frontier[k][0] < sorted_frontier[k - 1][0] - 1e-10:
            return False
    return True


def verify_frontier(frontier: list[tuple[float, float, int]]) -> bool:
    """验证前沿上任意两点互不支配。"""
    for i in range(len(frontier)):
        for j in range(i + 1, len(frontier)):
            if is_dominated(frontier[i], frontier[j]) or is_dominated(frontier[j], frontier[i]):
                return False
    return True


def main():
    print("=" * 60)
    print("投资组合双目标优化 — 随机权重枚举法")
    print("=" * 60)

    # 1. 生成随机数据
    N = 10
    print(f"\n资产数量: N = {N}")
    returns = random_returns(N, seed=7)
    cov = random_covariance_matrix(N, seed=42)
    print("期望收益率: ", [f"{r*100:.1f}%" for r in returns])

    # 2. 随机枚举权重
    NUM_SAMPLES = 10000
    rng = random.Random(12345)
    population: list[tuple[float, float, int]] = []

    for idx in range(NUM_SAMPLES):
        w = random_weights(N, rng)
        exp_ret, var = portfolio_stats(w, returns, cov)
        population.append((exp_ret, var, idx))

    # 3. O(N²) 非支配筛选
    print(f"\n枚举组合数: {NUM_SAMPLES}")
    print("正在筛选帕累托前沿...")
    frontier = nondominated_sort(population)
    frontier_sorted = sorted(frontier, key=lambda x: x[1])  # 按风险升序

    # 4. 输出结果
    print(f"\n帕累托前沿上的点数: {len(frontier)}")
    print("\n帕累托前沿 (风险(方差), 期望收益):")
    print(f"{'风险(σ²)':>12} | {'期望收益':>10} | {'索引':>6}")
    print("-" * 32)
    for exp_ret, var, idx in frontier_sorted:
        print(f"{var:>12.6f} | {exp_ret:>10.6f} | {idx:>6}")

    # 5. 单调性检查
    monotonic = is_monotonic_increasing([(r, v) for r, v, _ in frontier_sorted])
    print(f"\n前沿单调递增（风险越高收益越高）: {monotonic}")

    # 6. 验证
    valid = verify_frontier(frontier)
    print(f"前沿任意两点互不支配（验证通过）: {valid}")

    # 7. 展示两个极端点
    if frontier_sorted:
        min_risk = frontier_sorted[0]
        max_return = frontier_sorted[-1]
        print(f"\n最小风险点: 风险={min_risk[1]:.6f}, 收益={min_risk[0]:.6f}")
        print(f"最大收益点: 风险={max_return[1]:.6f}, 收益={max_return[0]:.6f}")

    print("\n完成。")


if __name__ == "__main__":
    main()
