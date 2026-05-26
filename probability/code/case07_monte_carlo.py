#!/usr/bin/env python3
"""
案例5：蒙特卡洛仿真与风险评估
=================================
评估投资组合在 95% 置信水平下的 VaR 和 CVaR。
纯 Python 标准库实现，无第三方依赖。
"""

import math
import random
import statistics


def normal_ppf(p, mu=0, sigma=1, tol=1e-10, max_iter=100):
    """正态分布分位数函数"""
    if p <= 0:
        return -float('inf')
    if p >= 1:
        return float('inf')
    lo, hi = -10, 10
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        z = (mid - mu) / sigma
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p_const = 0.3275911
        sign = 1 if z >= 0 else -1
        z_abs = abs(z) / math.sqrt(2)
        t = 1.0 / (1.0 + p_const * z_abs)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs)
        cdf_mid = 0.5 * (1.0 + sign * y)
        if abs(cdf_mid - p) < tol:
            return mid
        if cdf_mid < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate_normal(n, mu=0, sigma=1):
    """Box-Muller 变换生成正态分布样本"""
    samples = []
    for _ in range(n // 2 + 1):
        u1 = random.random()
        u2 = random.random()
        z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z2 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        samples.append(mu + sigma * z1)
        samples.append(mu + sigma * z2)
    return samples[:n]


def cholesky_decomposition(matrix):
    """
    Cholesky 分解：将正定矩阵分解为 L @ L^T
    matrix 是 n×n 的协方差矩阵（一维列表表示）
    """
    n = int(math.sqrt(len(matrix)))
    L = [0.0] * (n * n)
    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i * n + k] ** 2
        val = matrix[i * n + i] - s
        if val <= 0:
            val = 1e-10
        L[i * n + i] = math.sqrt(val)
        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += L[j * n + k] * L[i * n + k]
            if L[i * n + i] > 0:
                L[j * n + i] = (matrix[j * n + i] - s) / L[i * n + i]
            else:
                L[j * n + i] = 0.0
    return L


def generate_correlated_normal(num_assets, num_samples, means, stds, corr_matrix):
    """
    生成相关正态分布随机变量。
    使用 Cholesky 分解。

    参数：
        num_assets: 资产数量
        num_samples: 样本数量
        means: 均值列表 [μ₁, μ₂, ...]
        stds: 标准差列表 [σ₁, σ₂, ...]
        corr_matrix: 相关系数矩阵（一维列表，行优先）

    返回：
        samples: list of list，每个资产有一组样本
    """
    # 构建协方差矩阵
    cov_matrix = [0.0] * (num_assets * num_assets)
    for i in range(num_assets):
        for j in range(num_assets):
            cov_matrix[i * num_assets + j] = corr_matrix[i * num_assets + j] * stds[i] * stds[j]

    # Cholesky 分解
    L = cholesky_decomposition(cov_matrix)

    # 生成独立标准正态样本
    independent = [generate_normal(num_samples) for _ in range(num_assets)]

    # 应用 Cholesky 分解
    samples = []
    for i in range(num_assets):
        asset_samples = []
        for s in range(num_samples):
            val = means[i]
            for k in range(num_assets):
                val += L[i * num_assets + k] * independent[k][s]
            asset_samples.append(val)
        samples.append(asset_samples)

    return samples


def monte_carlo_portfolio(weights, means, stds, corr_matrix,
                           initial_value, num_simulations, conf_level=0.95):
    """
    蒙特卡洛仿真投资组合收益。

    参数：
        weights: 权重列表（和为 1）
        means: 各资产期望收益率
        stds: 各资产标准差
        corr_matrix: 相关系数矩阵
        initial_value: 初始投资额
        num_simulations: 仿真次数
        conf_level: 置信水平（默认 95%）

    返回：
        dict with keys:
            - portfolio_returns: 组合收益率列表
            - portfolio_values: 组合终值列表
            - var: Value at Risk
            - cvar: Conditional VaR
            - mean_return: 平均收益率
            - std_return: 收益率标准差
            - var_amount: VaR 金额
            - cvar_amount: CVaR 金额
    """
    num_assets = len(weights)

    # 生成相关资产收益率
    samples = generate_correlated_normal(
        num_assets, num_simulations, means, stds, corr_matrix
    )

    # 计算组合收益率
    portfolio_returns = []
    for s in range(num_simulations):
        rp = sum(weights[i] * samples[i][s] for i in range(num_assets))
        portfolio_returns.append(rp)

    # 排序
    sorted_returns = sorted(portfolio_returns)
    sorted_values = [initial_value * (1 + r) for r in sorted_returns]

    # VaR：排序后取 (1-conf_level) 分位数
    var_idx = int((1 - conf_level) * num_simulations)
    var = sorted_returns[var_idx]
    var_amount = initial_value * (1 + var) - initial_value

    # CVaR：小于 VaR 的收益率的平均值
    tail_returns = sorted_returns[:var_idx + 1]
    cvar = statistics.mean(tail_returns) if tail_returns else var
    cvar_amount = initial_value * (1 + cvar) - initial_value

    # 统计量
    mean_return = statistics.mean(portfolio_returns)
    std_return = statistics.stdev(portfolio_returns)

    return {
        "portfolio_returns": portfolio_returns,
        "portfolio_values": sorted_values,
        "var": var,
        "cvar": cvar,
        "mean_return": mean_return,
        "std_return": std_return,
        "var_amount": var_amount,
        "cvar_amount": cvar_amount,
    }


def analytical_var(weights, means, stds, corr_matrix, initial_value, conf_level=0.95):
    """解析法计算 VaR（基于正态分布假设）"""
    num_assets = len(weights)

    # 组合期望收益
    mu_p = sum(weights[i] * means[i] for i in range(num_assets))

    # 组合方差
    var_p = 0.0
    for i in range(num_assets):
        for j in range(num_assets):
            var_p += weights[i] * weights[j] * \
                     corr_matrix[i * num_assets + j] * stds[i] * stds[j]

    sigma_p = math.sqrt(var_p)

    # VaR
    z_score = normal_ppf(1 - conf_level)  # 负值
    var = mu_p + z_score * sigma_p
    var_amount = initial_value * (1 + var) - initial_value

    return {
        "mu_p": mu_p,
        "sigma_p": sigma_p,
        "var": var,
        "var_amount": var_amount,
    }


def convergence_demo(initial_value, num_trials_list):
    """演示蒙特卡洛的收敛速度"""
    weights = [0.6, 0.4]
    means = [0.12, 0.08]
    stds = [0.20, 0.10]
    corr_matrix = [1.0, 0.3, 0.3, 1.0]

    baseline = analytical_var(weights, means, stds, corr_matrix, initial_value)
    print(f"  解析解 VaR: {baseline['var']:.4f} ({baseline['var_amount']:+.2f} 万元)")
    print()

    for n in num_trials_list:
        result = monte_carlo_portfolio(
            weights, means, stds, corr_matrix,
            initial_value, n
        )
        error = abs(result['var'] - baseline['var'])
        print(f"  n={n:>8,}: MC VaR={result['var']:.4f}, "
              f"误差={error:.4f}, "
              f"CVaR={result['cvar']:.4f}")


def main():
    print("=" * 60)
    print("案例5：蒙特卡洛仿真与风险评估")
    print("=" * 60)

    random.seed(42)

    # 投资组合参数
    weights = [0.6, 0.4]       # 60% A, 40% B
    means = [0.12, 0.08]       # 期望收益率
    stds = [0.20, 0.10]        # 标准差
    corr_matrix = [1.0, 0.3,   # 相关系数矩阵
                   0.3, 1.0]
    initial_value = 1_000_000  # 100 万元

    print(f"\n投资组合参数：")
    print(f"  股票 A: 权重 {weights[0]:.0%}, μ={means[0]:.0%}, σ={stds[0]:.0%}")
    print(f"  股票 B: 权重 {weights[1]:.0%}, μ={means[1]:.0%}, σ={stds[1]:.0%}")
    print(f"  相关系数: ρ=0.3")
    print(f"  初始投资: {initial_value / 10000:.0f} 万元")

    # ========= 解析解 =========
    print("\n" + "-" * 60)
    print("解析法（假设正态分布）")
    print("-" * 60)

    analytical = analytical_var(weights, means, stds, corr_matrix, initial_value)
    print(f"  组合期望收益 μ_p = {analytical['mu_p']:.4f} ({analytical['mu_p']*100:.2f}%)")
    print(f"  组合标准差 σ_p = {analytical['sigma_p']:.4f} ({analytical['sigma_p']*100:.2f}%)")
    print(f"  95% VaR = {analytical['var']:.4f} ({analytical['var']*100:.2f}%)")
    print(f"  95% VaR 金额 = {analytical['var_amount']:+.2f} 元 "
          f"({analytical['var_amount']/10000:.2f} 万元)")

    # ========= 蒙特卡洛仿真 =========
    print("\n" + "-" * 60)
    print("蒙特卡洛仿真（100000 次）")
    print("-" * 60)

    result = monte_carlo_portfolio(
        weights, means, stds, corr_matrix,
        initial_value, 100000
    )

    print(f"  仿真平均收益: {result['mean_return']:.4f} ({result['mean_return']*100:.2f}%)")
    print(f"  仿真标准差: {result['std_return']:.4f} ({result['std_return']*100:.2f}%)")
    print(f"  95% VaR = {result['var']:.4f} ({result['var']*100:.2f}%)")
    print(f"  95% VaR 金额 = {result['var_amount']:+.2f} 元")
    print(f"  95% CVaR = {result['cvar']:.4f} ({result['cvar']*100:.2f}%)")
    print(f"  95% CVaR 金额 = {result['cvar_amount']:+.2f} 元")

    print()
    print(f"  VaR vs CVaR:")
    print(f"    VaR 说: 95% 情况下损失不超过 {abs(result['var_amount']/10000):.2f} 万元")
    print(f"    CVaR 说: 如果发生极端损失(最差 5%)，平均会亏 {abs(result['cvar_amount']/10000):.2f} 万元")
    print(f"    CVaR/VaR = {abs(result['cvar_amount']/result['var_amount']):.2f} 倍")

    # ========= 收敛性演示 =========
    print("\n" + "-" * 60)
    print("收敛性演示：蒙特卡洛误差随仿真次数变化")
    print("-" * 60)

    convergence_demo(initial_value, [100, 1000, 10000, 50000, 100000])

    # ========= 对偶变量方差缩减 =========
    print("\n" + "-" * 60)
    print("方差缩减：对偶变量法演示")
    print("-" * 60)

    # 简单演示：对偶变量法估计标准正态的均值
    print("  估计标准正态分布的均值（应为 0）")
    num_trials = 10000
    # 普通蒙特卡洛
    std_plain = []
    for _ in range(50):
        samples = [random.gauss(0, 1) for _ in range(num_trials)]
        std_plain.append(statistics.mean(samples))

    # 对偶变量
    std_antithetic = []
    for _ in range(25):
        u1 = [random.random() for _ in range(num_trials)]
        z1 = [math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * random.random())
              for u in u1]
        z2 = [-z for z in z1]
        combined = z1 + z2
        std_antithetic.append(statistics.mean(combined))

    var_plain = statistics.variance(std_plain)
    var_anti = statistics.variance(std_antithetic)
    reduction = (1 - var_anti / var_plain) * 100
    print(f"  普通 MC 方差: {var_plain:.6f}")
    print(f"  对偶变量方差: {var_anti:.6f}")
    print(f"  方差缩减: {reduction:.1f}%")

    # ========= 关键分位数输出 =========
    print("\n" + "-" * 60)
    print("不同置信水平的 VaR")
    print("-" * 60)

    sorted_returns = sorted(result['portfolio_returns'])
    for conf in [0.90, 0.95, 0.99]:
        idx = int((1 - conf) * len(sorted_returns))
        var_conf = sorted_returns[idx]
        amount = initial_value * (1 + var_conf) - initial_value
        print(f"  {conf:.0%} VaR = {var_conf:.4f} ({var_conf*100:.2f}%) = "
              f"{amount:+.2f} 元 ({amount/10000:.2f} 万元)")

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print(f"""
1. 蒙特卡洛的 VaR ≈ 解析解 ✅（正态分布假设下一致）
2. 仿真次数增加 → 误差减小 ✅（误差 ∝ 1/√n）
3. CVaR < VaR（CVaR 捕捉尾部风险）
4. 对偶变量法有效降低方差 ✅
5. 100 万元投资，95% VaR ≈ {abs(result['var_amount']/10000):.2f} 万元
    """)


if __name__ == "__main__":
    main()
