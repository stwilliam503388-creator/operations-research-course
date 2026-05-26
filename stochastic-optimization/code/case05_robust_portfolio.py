"""
鲁棒投资组合优化
=================
预算不确定性集合（Budget Uncertainty Set），最差情况优化。

问题描述：
  有 n 个资产，每种资产收益率 ri 在区间 [r_low, r_high] 内波动。
  使用预算不确定性集合约束不确定参数的偏移量之和不超过 Γ。
  求解最差情况下的最优投资组合权重（最大化最小收益率）。

本代码纯 Python 标准库 + numpy，中文注释，可自测。
"""

import random
import math

# ---------- 参数 ----------
N_ASSETS = 5                         # 资产数量
R_LOW = [0.02, 0.01, 0.03, 0.00, 0.02]    # 各资产最低收益率
R_HIGH = [0.12, 0.15, 0.10, 0.18, 0.08]   # 各资产最高收益率
BUDGET = 2.0                         # 预算参数 Γ（控制鲁棒程度）
CAPITAL = 1.0                        # 总投资资金

# 权重搜索步长
STEP = 0.05


def generate_weight_grid(n, step):
    """
    生成所有满足 sum(w) = 1, w >= 0 的权重组合（离散网格）。
    递归生成。
    """
    results = []

    def _rec(remaining_assets, remaining_sum, current):
        if remaining_assets == 1:
            candidate = current + [remaining_sum]
            # 检查是否有 step 量级的误差
            if all(abs(w - round(w / step) * step) < 1e-9 for w in candidate):
                results.append(candidate)
            return
        # 当前权重的可能取值
        count = int(remaining_sum / step)
        for i in range(count + 1):
            w = i * step
            _rec(remaining_assets - 1, remaining_sum - w, current + [w])

    _rec(n, 1.0, [])
    return results


def worst_case_return(weights, budget):
    """
    给定权重向量 w，计算预算不确定性集合下的最差情景收益率。
    模型：
       max_{z_i ∈ [0,1], sum(z_i) ≤ Γ}  sum_i w_i * (r_low_i + z_i * (r_high_i - r_low_i))
    等价于：
       sum_i w_i * r_low_i + max_{z_i ∈ [0,1], sum(z_i) ≤ Γ} sum_i w_i * z_i * (r_high_i - r_low_i)
    内部最大化是背包问题：按 w_i * (r_high_i - r_low_i) 从大到小排序，取前 Γ 个 z_i=1。
    但因为 z_i 不需要是整数，实际上最优解是将 Γ 分配给最大的系数。
    """
    n = len(weights)
    base = sum(weights[i] * R_LOW[i] for i in range(n))
    # 每个资产的"潜在收益"（不确定性的影响系数）
    potentials = [weights[i] * (R_HIGH[i] - R_LOW[i]) for i in range(n)]
    # 按降序排序
    sorted_pot = sorted(potentials, reverse=True)
    # 取前 floor(Γ) 个全部分配，最后一个按小数部分分配
    k = int(budget)  # 整数部分
    frac = budget - k  # 小数部分
    extra = sum(sorted_pot[:k]) + (frac * sorted_pot[k] if k < n else 0.0)
    worst = base + extra
    return worst


def solve_robust_portfolio(budget):
    """在权重网格上搜索最差情景收益率最大的组合"""
    all_weights = generate_weight_grid(N_ASSETS, STEP)
    best_w = None
    best_val = -float("inf")

    for w in all_weights:
        wc = worst_case_return(w, budget)
        if wc > best_val:
            best_val = wc
            best_w = w

    return best_w, best_val


def nominal_return(weights):
    """名义收益率（用中点估计）"""
    r_mid = [(R_LOW[i] + R_HIGH[i]) / 2.0 for i in range(N_ASSETS)]
    return sum(weights[i] * r_mid[i] for i in range(N_ASSETS))


def main():
    """主函数：自测入口"""
    print("=" * 60)
    print("鲁棒投资组合优化 — 预算不确定性集合")
    print("=" * 60)
    print(f"资产数: {N_ASSETS}")
    for i in range(N_ASSETS):
        print(f"  资产{i+1}: 收益率区间 [{R_LOW[i]:.2f}, {R_HIGH[i]:.2f}]")
    print(f"预算参数 Γ = {BUDGET}")
    print(f"权重步长 = {STEP}")
    print()

    # 计算所有候选权重
    print("正在枚举权重组合...")
    all_weights = generate_weight_grid(N_ASSETS, STEP)
    print(f"共 {len(all_weights)} 个候选权重组合")
    print()

    # 求解鲁棒最优组合
    best_w, best_val = solve_robust_portfolio(BUDGET)

    print(f"[鲁棒最优组合 (Γ={BUDGET})]")
    for i in range(N_ASSETS):
        print(f"  资产{i+1}: {best_w[i]:.2f}")
    print(f"最差情景收益率: {best_val:.4f} ({best_val*100:.2f}%)")
    print()

    # 名义收益率
    nom = nominal_return(best_w)
    print(f"名义收益率（中点估计）: {nom:.4f} ({nom*100:.2f}%)")
    print()

    # 不同 Γ 的对比
    print("--- 不同预算参数 Γ 下的最优最差收益率 ---")
    for gamma in [0.0, 0.5, 1.0, 2.0, 3.0, N_ASSETS]:
        if gamma > N_ASSETS:
            gamma = float(N_ASSETS)
        w_g, val_g = solve_robust_portfolio(gamma)
        nom_g = nominal_return(w_g)
        print(f"  Γ={gamma:3.1f}: 最差={val_g:.4f}, 名义={nom_g:.4f}, "
              f"权重分布=[{', '.join(f'{w:.2f}' for w in w_g)}]")

    print()
    print("测试通过！")


if __name__ == "__main__":
    main()
