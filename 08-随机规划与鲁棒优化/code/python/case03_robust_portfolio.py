"""
案例3：鲁棒投资组合
难度：★★★☆☆
方法：鲁棒优化
依赖：pip install numpy scipy
运行：python code/python/case03_robust_portfolio.py

简述：用鲁棒优化构建投资组合——在收益率不确定时最大化最坏情况收益。
"""
# 教学注释：通过场景、概率权重和坏情况边界刻画不确定需求或收益。
# 对比不同假设下的解，理解随机性和鲁棒性如何改变决策。


import numpy as np
from scipy.optimize import linprog


# ============================================================
# 数据准备
# ============================================================
N = 10  # 股票数量

# 为了让结果可复现，固定随机种子
rng = np.random.default_rng(42)

# 随机生成期望收益率和波动率
r = rng.uniform(0.05, 0.20, N)   # 期望收益率 5%~20%
sigma = rng.uniform(0.02, 0.08, N)  # 波动率 2%~8%


# ============================================================
# 模型构建
# ============================================================
def solve_mean_model():
    """
    确定性均值模型: 只最大化期望收益率，没有不确定性考虑。
    max Σ r_i w_i,  s.t. Σ w_i = 1, w_i >= 0
    结果：全部投在收益率最高的股票上。
    """
    c_obj = [-x for x in r]  # linprog 最小化
    A_eq = [np.ones(N)]
    b_eq = [1.0]
    bounds = [(0, 1)] * N

    result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if result.success:
        return result.x, -result.fun
    raise RuntimeError(f"均值模型求解失败: {result.message}")


def solve_box_robust():
    """
    箱式鲁棒模型 (Γ = N):
    假设所有股票的收益率同时降到最差值。
    max Σ (r_i - σ_i) w_i
    """
    c_obj = [-(r[i] - sigma[i]) for i in range(N)]
    A_eq = [np.ones(N)]
    b_eq = [1.0]
    bounds = [(0, 1)] * N

    result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if result.success:
        return result.x, -result.fun
    raise RuntimeError(f"箱式鲁棒求解失败: {result.message}")


def solve_budget_robust(Gamma):
    """
    预算式鲁棒模型 (Γ 可调):
    max Σ r_i w_i - max_{|S|<=Γ} Σ_{j∈S} σ_j w_j

    等价线性化: 引入辅助变量 z, r_j
    max Σ r_i w_i - Γ·z - Σ r_j
    s.t. z + r_j >= σ_j w_j, z >= 0, r_j >= 0
         Σ w_i = 1, w_i >= 0
    """
    # 变量: [w_1...w_N, z, r_1...r_N]
    n_vars = N + 1 + N

    # 目标: -[Σ r_i w_i - Γ·z - Σ r_j] = -Σ r_i w_i + Γ·z + Σ r_j
    c_obj = [-r[i] for i in range(N)] + [Gamma] + [1.0] * N

    # 约束
    A_ub = []
    b_ub = []

    # z + r_j >= σ_j w_j → -σ_j w_j + z + r_j >= 0 → σ_j w_j - z - r_j <= 0
    for j in range(N):
        row = [0.0] * n_vars
        row[j] = sigma[j]
        row[N] = -1.0       # -z
        row[N + 1 + j] = -1.0  # -r_j
        A_ub.append(row)
        b_ub.append(0)

    # Σ w_i = 1 (等式约束)
    A_eq = [[1.0 if i < N else 0.0 for i in range(n_vars)]]
    b_eq = [1.0]

    # 边界
    bounds = [(0, 1)] * N + [(0, None)] + [(0, None)] * N

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
    if result.success:
        w = result.x[:N]
        obj_val = -result.fun
        return w, obj_val
    raise RuntimeError(f"预算式鲁棒求解失败: {result.message}")


def worst_case_return(w, Gamma=None):
    """
    计算组合在最坏情况下的收益率。
    如果 Gamma=None 用箱式（所有股票最差）；
    否则最多 Gamma 只股票同时最差。
    """
    if Gamma is None:
        # 箱式: 所有股票同时最差
        return np.sum((r - sigma) * w)
    else:
        # 预算式: 选择使组合收益最差的 Gamma 只股票
        # 对每只股票，最差情况偏离 = -sigma_j * w_j
        deviations = sigma * w
        sorted_idx = np.argsort(-deviations)  # 从大到小排序
        total = np.sum(r * w)
        # Γ 取整
        k = int(np.floor(Gamma))
        fractional = Gamma - k
        total -= np.sum(deviations[sorted_idx[:k]])
        if k < N and fractional > 0:
            total -= fractional * deviations[sorted_idx[k]]
        return total


# ============================================================
# 结果输出
# ============================================================
def print_summary():
    print("=" * 60)
    print("案例3：鲁棒投资组合")
    print("=" * 60)

    # 1. 均值模型
    w_mean, ret_mean = solve_mean_model()
    print(f"\n▶ 确定性均值模型:")
    print(f"  期望收益 = {ret_mean:.4f} ({ret_mean * 100:.2f}%)")
    print(f"  集中度: {', '.join([f'股票{i + 1}={w:.2f}' for i, w in enumerate(w_mean) if w > 0.01])}")

    # 2. 箱式鲁棒
    w_box, ret_box = solve_box_robust()
    print(f"\n▶ 箱式鲁棒模型 (Γ={N}):")
    print(f"  最坏情况收益 = {ret_box:.4f} ({ret_box * 100:.2f}%)")

    # 3. 预算式鲁棒
    print(f"\n▶ 预算式鲁棒模型 (Γ 从 0 到 {N}):")
    print(f"{'Γ':>4} | {'期望收益':>10} | {'最坏收益':>10} | {'持仓熵':>8}")
    print("-" * 45)
    for Gamma in range(0, N + 1):
        w, obj = solve_budget_robust(Gamma)
        worst_ret = worst_case_return(w, Gamma)
        # 持仓熵（衡量分散度）：越高越分散
        nonzero_w = w[w > 0.001]
        if len(nonzero_w) > 0:
            entropy = -np.sum(nonzero_w * np.log(nonzero_w))
        else:
            entropy = 0
        print(f"{Gamma:>4} | {obj:>10.4f} | {worst_ret:>10.4f} | {entropy:>8.2f}")

    # 4. 验证: Γ=0 = 均值模型? (应该基本一致，但因为有多个最优解可能不同)
    w_g0, _ = solve_budget_robust(0)
    print(f"\n▶ Γ=0 vs 均值模型:")
    print(f"  Γ=0 方案集中度: {np.sum(w_g0 > 0.01)} 只股票")
    print(f"  均值方案集中度: {np.sum(w_mean > 0.01)} 只股票")


def solve_small():
    """小规模测试：N=5"""
    print("\n=== 小规模测试: N=5 ===")
    w, obj = solve_budget_robust(2)
    print(f"Γ=2, 最坏收益={obj:.4f}")


def solve_medium():
    """中等规模测试：N=20"""
    print("\n=== 中等规模测试: N=20 ===")
    global N, r, sigma
    old_N, old_r, old_sigma = N, r.copy(), sigma.copy()
    N = 20
    local_rng = np.random.default_rng(123)
    r = local_rng.uniform(0.05, 0.20, N)
    sigma = local_rng.uniform(0.02, 0.08, N)
    try:
        for Gamma in [0, 5, 10, 20]:
            w, obj = solve_budget_robust(Gamma)
            worst = worst_case_return(w, Gamma)
            print(f"Γ={Gamma:>2}: 期望收益={obj:.4f}, 最坏收益={worst:.4f}")
    finally:
        N, r, sigma = old_N, old_r, old_sigma


if __name__ == "__main__":
    solve_small()
    solve_medium()
    print("\n")
    print_summary()
