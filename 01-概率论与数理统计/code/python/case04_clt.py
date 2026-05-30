#!/usr/bin/env python3
"""
案例2：中心极限定理演示
=========================
展示从不同分布中抽样时，样本均值的分布如何随样本量增加趋近正态。
纯 Python 标准库实现，无第三方依赖。
"""


# 教学注释：围绕随机分布、样本统计量和蒙特卡洛估计观察不确定性。
# 运行时重点比较理论量与模拟结果如何支撑决策判断。



import math
import random
import statistics


def normal_pdf(x, mu=0, sigma=1):
    """正态分布概率密度函数"""
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * \
           math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generate_uniform(n, lo=0, hi=1):
    """生成均匀分布样本"""
    return [random.uniform(lo, hi) for _ in range(n)]


def generate_exponential(n, lam=1.0):
    """生成指数分布样本（逆变换法）"""
    return [-math.log(random.random()) / lam for _ in range(n)]


def generate_binomial(n_trials, p=0.5):
    """生成单个二项分布值（n次伯努利试验）"""
    return sum(1 for _ in range(n_trials) if random.random() < p)


def generate_binomial_samples(n, n_trials=10, p=0.5):
    """生成 n 个二项分布样本"""
    return [generate_binomial(n_trials, p) for _ in range(n)]


def clt_demo(dist_name, dist_generator, dist_params, pop_mean, pop_var,
             sample_sizes, num_repeats=10000):
    """
    演示 CLT：对给定分布，在不同样本量下抽样并观察均值分布。

    参数：
        dist_name: 分布名称（字符串）
        dist_generator: 生成 n 个样本的函数
        dist_params: 传给生成函数的参数（字典）
        pop_mean: 总体均值
        pop_var: 总体方差
        sample_sizes: 要尝试的样本量列表
        num_repeats: 每种样本量重复多少次（默认 10000）
    """
    print(f"\n{'=' * 60}")
    print(f"分布: {dist_name}")
    print(f"总体均值 μ = {pop_mean:.4f}, 总体方差 σ² = {pop_var:.4f}, σ = {math.sqrt(pop_var):.4f}")
    print(f"{'=' * 60}")

    for n in sample_sizes:
        # 重复抽样 num_repeats 次，每次计算样本均值
        sample_means = []
        for _ in range(num_repeats):
            sample = dist_generator(n, **dist_params)
            sample_means.append(statistics.mean(sample))

        # 计算样本均值的统计量
        mean_of_means = statistics.mean(sample_means)
        var_of_means = statistics.variance(sample_means)  # 样本方差
        std_of_means = math.sqrt(var_of_means)

        # CLT 理论预测
        theoretical_mean = pop_mean
        theoretical_std = math.sqrt(pop_var / n)

        # 计算正态性评估：分位数相关性
        sorted_means = sorted(sample_means)
        # 与理论正态分位数比较
        mu_est = mean_of_means
        sigma_est = std_of_means
        r = compute_qq_corr(sorted_means, mu_est, sigma_est, num_repeats)

        print(f"\n  n = {n:4d}（重复 {num_repeats} 次）:")
        print(f"    样本均值的均值: {mean_of_means:.4f}  (理论: {theoretical_mean:.4f})")
        print(f"    样本均值的标准差: {std_of_means:.4f}  (理论 σ/√n = {theoretical_std:.4f})")
        print(f"    CLT 方差公式验证: Var(mean) = {var_of_means:.6f} ≈ σ²/n = {pop_var / n:.6f}")
        print(f"    QQ 相关性 (正态性): {r:.4f}")

        # 评估趋近正态的程度
        if r > 0.998:
            status = "✅ 接近完美正态"
        elif r > 0.99:
            status = "✅ 近似正态"
        elif r > 0.97:
            status = "⚠️ 大致正态"
        else:
            status = "❌ 仍偏离正态"
        print(f"    状态: {status}")

        # 计算一个简单正态性检验的统计量
        # 使用标准化的偏度
        m3 = sum((x - mean_of_means) ** 3 for x in sample_means) / num_repeats
        skewness = m3 / (std_of_means ** 3) if std_of_means > 0 else 0
        print(f"    偏度: {skewness:.4f}  (正态≈0)")


def normal_ppf(p, mu=0, sigma=1, tol=1e-10, max_iter=100):
    """正态分布分位数函数（二分法）"""
    if p <= 0:
        return -float('inf')
    if p >= 1:
        return float('inf')
    lo, hi = -10, 10
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        # normal_cdf 近似
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


def compute_qq_corr(sorted_data, mu, sigma, n):
    """计算数据与对应理论正态分位数的相关系数"""
    x_vals = []
    y_vals = []
    data_mean = statistics.mean(sorted_data)
    data_std = statistics.stdev(sorted_data)
    if data_std == 0:
        return 1.0
    for i in range(1, n):
        p = i / (n + 1)
        theoretical = normal_ppf(p, mu, sigma)
        x_vals.append(theoretical)
        y_vals.append((sorted_data[i - 1] - data_mean) / data_std)
    # pearson correlation
    n_pts = len(x_vals)
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    cov = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n_pts))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_vals))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_vals))
    if std_x * std_y == 0:
        return 0
    return cov / (std_x * std_y)


def main():
    print("=" * 60)
    print("案例2：中心极限定理（CLT）演示")
    print("演示从非正态分布中抽样时，样本均值的分布趋近正态")
    print("=" * 60)

    random.seed(42)

    sample_sizes = [1, 5, 30, 100]
    num_repeats = 10000

    # 测试1：均匀分布 U(0, 1)
    # μ = 0.5, σ² = 1/12 ≈ 0.0833
    clt_demo(
        dist_name="均匀分布 U(0, 1)",
        dist_generator=generate_uniform,
        dist_params={"lo": 0, "hi": 1},
        pop_mean=0.5,
        pop_var=1.0 / 12.0,
        sample_sizes=sample_sizes,
        num_repeats=num_repeats,
    )

    # 测试2：指数分布 Exp(λ=1)
    # μ = 1/λ = 1, σ² = 1/λ² = 1
    clt_demo(
        dist_name="指数分布 Exp(λ=1)",
        dist_generator=generate_exponential,
        dist_params={"lam": 1.0},
        pop_mean=1.0,
        pop_var=1.0,
        sample_sizes=sample_sizes,
        num_repeats=num_repeats,
    )

    # 测试3：二项分布 Bin(10, 0.3)
    # μ = np = 3, σ² = np(1-p) = 2.1
    clt_demo(
        dist_name="二项分布 Bin(10, 0.3)",
        dist_generator=generate_binomial_samples,
        dist_params={"n_trials": 10, "p": 0.3},
        pop_mean=3.0,
        pop_var=2.1,
        sample_sizes=sample_sizes,
        num_repeats=num_repeats,
    )

    # 额外的演示：指数分布需要更大的 n 才能收敛
    print("\n" + "=" * 60)
    print("额外测试：指数分布需要更大 n 才能完全收敛")
    print("=" * 60)
    clt_demo(
        dist_name="指数分布 Exp(λ=1) 大样本",
        dist_generator=generate_exponential,
        dist_params={"lam": 1.0},
        pop_mean=1.0,
        pop_var=1.0,
        sample_sizes=[5, 30, 100, 500],
        num_repeats=num_repeats,
    )

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print("""
1. n=1: 样本均值的分布 = 原始分布（完全不正态）
2. n=5: 开始趋近正态，但对偏态分布仍明显偏斜
3. n=30: 经验法则——大多数分布下均值接近正态
4. n=100: 即使是偏态分布，均值也基本正态了
5. 方差公式验证: Var(mean) ≈ σ²/n ✅
    """)


if __name__ == "__main__":
    main()
