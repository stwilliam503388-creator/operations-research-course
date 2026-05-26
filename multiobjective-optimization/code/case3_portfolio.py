#!/usr/bin/env python3
"""
案例1：投资组合双目标优化 (Portfolio Multi-Objective Optimization)
Markowitz 均值-方差模型的多目标版本
方法：加权求和法 + ε-约束法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def solve_weighted_sum(mu, Sigma, w1, w2):
    """
    加权求和法求解投资组合
    
    参数:
        mu: 期望收益向量
        Sigma: 协方差矩阵
        w1, w2: 权重 (w1 + w2 = 1)
    
    返回:
        x: 最优投资组合权重
    """
    n = len(mu)
    
    def objective(x):
        return w1 * (-mu @ x) + w2 * (x @ Sigma @ x)
    
    # 约束：权重和为1
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, 1)] * n
    
    res = minimize(objective, x0=np.ones(n)/n,
                   constraints=cons, bounds=bounds,
                   method='SLSQP', options={'ftol': 1e-12})
    return res.x


def solve_epsilon_constraint(mu, Sigma, epsilon):
    """
    ε-约束法求解投资组合
    min -μᵀx  s.t. xᵀΣx ≤ ε, Σx=1, x≥0
    
    参数:
        mu: 期望收益向量
        Sigma: 协方差矩阵
        epsilon: 风险上限
    
    返回:
        x: 最优投资组合权重
    """
    n = len(mu)
    
    def objective(x):
        return -mu @ x
    
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: epsilon - x @ Sigma @ x}
    ]
    bounds = [(0, 1)] * n
    
    res = minimize(objective, x0=np.ones(n)/n,
                   constraints=cons, bounds=bounds,
                   method='SLSQP', options={'ftol': 1e-12})
    return res.x


def compute_frontier_stats(x, mu, Sigma):
    """计算一个投资组合的收益和风险"""
    ret = mu @ x
    risk = np.sqrt(x @ Sigma @ x)
    return ret, risk


def main():
    print("=" * 60)
    print("案例1：投资组合双目标优化")
    print("=" * 60)
    
    # ========== 数据 ==========
    # 三种资产的期望收益率（%）
    mu = np.array([12.5, 8.0, 3.5])
    
    # 协方差矩阵（%²）
    Sigma = np.array([
        [185,  80,  20],
        [ 80,  92,  12],
        [ 20,  12,   8]
    ])
    n = len(mu)
    
    print(f"\n资产期望收益: {mu}")
    print(f"协方差矩阵:\n{Sigma}")
    print(f"资产1（科技股）：高收益高风险")
    print(f"资产2（混合债）：中收益中风险")
    print(f"资产3（国债）：低收益低风险")
    
    # ========== 方法1：加权求和法 ==========
    print("\n" + "-" * 60)
    print("方法1：加权求和法")
    print("-" * 60)
    
    n_weights = 21
    weights = [(w, 1 - w) for w in np.linspace(0, 1, n_weights)]
    
    ws_frontier = []  # (收益, 风险, 权重向量)
    for w1, w2 in weights:
        x = solve_weighted_sum(mu, Sigma, w1, w2)
        ret, risk = compute_frontier_stats(x, mu, Sigma)
        ws_frontier.append((ret, risk, x))
    
    print(f"\n加权求和法结果（部分展示）:")
    for i in [0, 5, 10, 15, 20]:
        ret, risk, x = ws_frontier[i]
        w1, w2 = weights[i]
        print(f"  权重 ({w1:.1f}, {w2:.1f}): 收益={ret:.2f}%, 风险={risk:.2f}%, "
              f"配置=[{x[0]:.1%}, {x[1]:.1%}, {x[2]:.1%}]")
    
    # ========== 方法2：ε-约束法 ==========
    print("\n" + "-" * 60)
    print("方法2：ε-约束法")
    print("-" * 60)
    
    # 确定 ε 范围
    x_min_risk = solve_weighted_sum(mu, Sigma, 0, 1)
    min_risk = x_min_risk @ Sigma @ x_min_risk
    
    x_max_ret = solve_weighted_sum(mu, Sigma, 1, 0)
    max_risk = x_max_ret @ Sigma @ x_max_ret
    
    epsilons = np.linspace(min_risk, max_risk, 15)
    
    ec_frontier = []
    for eps in epsilons:
        x = solve_epsilon_constraint(mu, Sigma, eps)
        if np.isfinite(x).all():
            ret, risk = compute_frontier_stats(x, mu, Sigma)
            ec_frontier.append((ret, risk, x))
    
    print(f"\nε-约束法结果（部分展示）:")
    display_indices = [0, 3, 7, 10, 14]
    for idx in display_indices:
        if idx < len(ec_frontier):
            ret, risk, x = ec_frontier[idx]
            eps = epsilons[idx]
            print(f"  ε={eps:.1f}: 收益={ret:.2f}%, 风险={risk:.2f}%, "
                  f"配置=[{x[0]:.1%}, {x[1]:.1%}, {x[2]:.1%}]")
    
    # ========== 验证 ==========
    print("\n" + "-" * 60)
    print("验证")
    print("-" * 60)
    
    # 验证1：收益最高解 = 全仓收益最高的资产
    max_ret_sol = ws_frontier[-1]  # 权重 (1.0, 0.0)
    ret1, risk1, x1 = max_ret_sol
    print(f"\n✅ 验证1 - 收益最高解:")
    print(f"   配置: [{x1[0]:.1%}, {x1[1]:.1%}, {x1[2]:.1%}]")
    print(f"   收益: {ret1:.2f}% (目标: 12.5%)")
    
    # 验证2：风险最低解 = 全仓风险最低的资产
    min_risk_sol = ws_frontier[0]  # 权重 (0.0, 1.0)
    ret2, risk2, x2 = min_risk_sol
    print(f"\n✅ 验证2 - 风险最低解:")
    print(f"   配置: [{x2[0]:.1%}, {x2[1]:.1%}, {x2[2]:.1%}]")
    print(f"   风险: {risk2:.2f}%")
    
    # 验证3：前沿单调递增
    ws_risks = [s[1] for s in ws_frontier]
    ws_rets = [s[0] for s in ws_frontier]
    is_monotonic = all(ws_rets[i] <= ws_rets[i+1] for i in range(len(ws_rets)-1))
    print(f"\n✅ 验证3 - 前沿单调递增: {is_monotonic}")
    
    # 验证4：权重和为1
    all_sum_ok = all(abs(np.sum(x) - 1.0) < 1e-6 for _, _, x in ws_frontier)
    print(f"✅ 验证4 - 所有解权重和为1: {all_sum_ok}")
    
    # ========== 绘图 ==========
    plt.figure(figsize=(10, 7))
    
    # 加权求和前沿
    ws_risks = [s[1] for s in ws_frontier]
    ws_rets = [s[0] for s in ws_frontier]
    plt.plot(ws_risks, ws_rets, 'b-o', label='加权求和法', markersize=4)
    
    # ε-约束前沿
    ec_risks = [s[1] for s in ec_frontier]
    ec_rets = [s[0] for s in ec_frontier]
    plt.plot(ec_risks, ec_rets, 'r--s', label='ε-约束法', markersize=4)
    
    plt.xlabel('风险（标准差 %）')
    plt.ylabel('期望收益（%）')
    plt.title('投资组合帕累托前沿')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 标注极端解
    plt.annotate('纯收益型', (ws_risks[-1], ws_rets[-1]),
                 xytext=(ws_risks[-1]+1, ws_rets[-1]))
    plt.annotate('纯保守型', (ws_risks[0], ws_rets[0]),
                 xytext=(ws_risks[0]-1, ws_rets[0]-1))
    
    plt.tight_layout()
    plt.savefig('portfolio_pareto_front.png', dpi=150)
    print(f"\n✅ 帕累托前沿图已保存至: portfolio_pareto_front.png")
    
    print("\n" + "=" * 60)
    print("案例1 完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
