#!/usr/bin/env python3
"""
案例4：假设检验与A/B测试
===========================
用双比例 z 检验判断新算法是否显著提高了点击率。
纯 Python 标准库实现，无第三方依赖。
"""


# 教学注释：围绕随机分布、样本统计量和蒙特卡洛估计观察不确定性。
# 运行时重点比较理论量与模拟结果如何支撑决策判断。



import math
import random
import statistics


def normal_cdf(x, mu=0, sigma=1):
    """正态分布累积分布函数"""
    z = (x - mu) / sigma
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1 if z >= 0 else -1
    z_abs = abs(z) / math.sqrt(2)
    t = 1.0 / (1.0 + p * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs)
    return 0.5 * (1.0 + sign * y)


def normal_ppf(p, mu=0, sigma=1, tol=1e-10, max_iter=100):
    """正态分布分位数函数"""
    if p <= 0:
        return -float('inf')
    if p >= 1:
        return float('inf')
    lo, hi = -10, 10
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        cdf_mid = normal_cdf(mid, mu, sigma)
        if abs(cdf_mid - p) < tol:
            return mid
        if cdf_mid < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def two_proportion_z_test(n1, x1, n2, x2, alternative='two-sided'):
    """
    双比例 z 检验。

    参数：
        n1, n2: 两组样本量
        x1, x2: 两组成功次数
        alternative: 'two-sided', 'greater', 'less'

    返回：
        z_stat: z 统计量
        p_value: p 值
        p1, p2: 两组比例
        pooled_p: 合并比例
        se: 标准误
    """
    p1 = x1 / n1
    p2 = x2 / n2
    pooled_p = (x1 + x2) / (n1 + n2)

    # 标准误
    se = math.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float('inf'), 0, p1, p2, pooled_p, se

    # z 统计量
    z_stat = (p1 - p2) / se

    # p 值（基于标准正态分布）
    if alternative == 'two-sided':
        p_value = 2 * (1 - normal_cdf(abs(z_stat)))
    elif alternative == 'greater':
        p_value = 1 - normal_cdf(z_stat)
    elif alternative == 'less':
        p_value = normal_cdf(z_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return z_stat, p_value, p1, p2, pooled_p, se


def confidence_interval_prop_diff(n1, x1, n2, x2, conf_level=0.95):
    """两个比例之差的置信区间"""
    p1 = x1 / n1
    p2 = x2 / n2
    diff = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = normal_ppf(1 - (1 - conf_level) / 2)
    margin = z_crit * se
    return diff - margin, diff + margin, diff


def simulate_ab_test(p_control, p_treatment, n_per_group, num_simulations=10000):
    """
    模拟 A/B 测试，统计拒绝 H₀ 的比例。

    参数：
        p_control: 对照组真实 CTR
        p_treatment: 实验组真实 CTR
        n_per_group: 每组样本量
        num_simulations: 模拟次数

    返回：
        reject_rate: 拒绝 H₀ 的比例（检验效力或第一类错误率）
        avg_p_value: 平均 p 值
    """
    reject_count = 0
    p_values = []
    for _ in range(num_simulations):
        x1 = sum(1 for _ in range(n_per_group) if random.random() < p_control)
        x2 = sum(1 for _ in range(n_per_group) if random.random() < p_treatment)
        # 测试：处理组 > 对照组
        _, p_val, _, _, _, _ = two_proportion_z_test(
            n_per_group, x2, n_per_group, x1, alternative='greater'
        )
        p_values.append(p_val)
        if p_val < 0.05:
            reject_count += 1

    reject_rate = reject_count / num_simulations
    avg_p = statistics.mean(p_values)
    return reject_rate, avg_p


def power_analysis(p_control, p_treatment, sample_sizes, alpha=0.05,
                   num_simulations=5000):
    """
    检验效力分析：对不同的样本量计算 power。

    返回：
        list of (sample_size, power)
    """
    results = []
    for n in sample_sizes:
        power, _ = simulate_ab_test(p_control, p_treatment, n, num_simulations)
        results.append((n, power))
        print(f"    n={n:6d}: power={power:.4f}")
    return results


def main():
    print("=" * 60)
    print("案例4：假设检验与 A/B 测试")
    print("=" * 60)

    random.seed(42)

    # ========= 场景1：无差异（验证第一类错误） =========
    print("\n" + "-" * 60)
    print("场景1：新旧算法无差异（验证第一类错误率）")
    print("-" * 60)

    p_ctrl = 0.05  # 对照组 CTR = 5%
    n_per_group = 5000

    print(f"  对照组 CTR: {p_ctrl:.1%}")
    print(f"  实验组 CTR: {p_ctrl:.1%}（相同）")
    print(f"  每组样本量: {n_per_group}")
    print(f"  模拟次数: 10000")
    print()

    reject_rate, avg_p = simulate_ab_test(p_ctrl, p_ctrl, n_per_group, 10000)
    print(f"  拒绝 H₀ 的比例: {reject_rate:.4f}")
    print(f"  平均 p 值: {avg_p:.4f}")
    if reject_rate <= 0.07:
        print(f"  ✅ 第一类错误率 ≈ {reject_rate:.4f}，接近理论值 0.05")
    else:
        print(f"  ⚠️ 第一类错误率偏高: {reject_rate:.4f}")

    # ========= 场景2：有差异（验证检验效力） =========
    print("\n" + "-" * 60)
    print("场景2：新算法确实更好（验证检验效力）")
    print("-" * 60)

    p_ctrl2 = 0.05   # 5%
    p_trtm2 = 0.06   # 6%（提升 20%）
    n_per_group2 = 5000

    print(f"  对照组 CTR: {p_ctrl2:.1%}")
    print(f"  实验组 CTR: {p_trtm2:.1%}（提升 {(p_trtm2-p_ctrl2)/p_ctrl2*100:.0f}%）")
    print(f"  每组样本量: {n_per_group2}")
    print(f"  模拟次数: 10000")
    print()

    power, avg_p2 = simulate_ab_test(p_ctrl2, p_trtm2, n_per_group2, 10000)
    print(f"  拒绝 H₀ 的比例（power）: {power:.4f}")
    print(f"  平均 p 值: {avg_p2:.4f}")
    if power > 0.7:
        print(f"  ✅ 能够检测出差异 (power={power:.4f})")
    else:
        print(f"  ⚠️ 检验效力偏低 (power={power:.4f})")

    # 展示一个具体的结果
    print()
    print(f"  举例：一次 A/B 测试结果")
    x1 = sum(1 for _ in range(n_per_group2) if random.random() < p_ctrl2)
    x2 = sum(1 for _ in range(n_per_group2) if random.random() < p_trtm2)
    z, p_val, p1, p2, pooled, se = two_proportion_z_test(
        n_per_group2, x1, n_per_group2, x2, alternative='greater'
    )
    ci_lo, ci_hi, diff = confidence_interval_prop_diff(
        n_per_group2, x1, n_per_group2, x2
    )
    print(f"    对照组: {x1}/{n_per_group2} = {p1:.4f}")
    print(f"    实验组: {x2}/{n_per_group2} = {p2:.4f}")
    print(f"    差异: {diff:.4f} ({diff*100:.2f}%)")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    z = {z:.4f}, p = {p_val:.4f}")

    # ========= 场景3：样本量对检验效力的影响 =========
    print("\n" + "-" * 60)
    print("场景3：样本量对检验效力的影响")
    print("-" * 60)

    print(f"  对照组 CTR: 5%, 实验组 CTR: 6%")
    print(f"  α = 0.05（单侧）")
    print()

    sample_sizes = [500, 1000, 2000, 5000, 10000, 20000]
    power_results = power_analysis(0.05, 0.06, sample_sizes, num_simulations=3000)

    print()
    print(f"  所需样本量估算（80% power）：")
    for n, pow_val in power_results:
        if pow_val >= 0.8:
            print(f"    ✅ n ≈ {n} 达到 80% power")
            break

    # ========= 场景4：效应量 vs 样本量 =========
    print("\n" + "-" * 60)
    print("场景4：不同效应量下所需样本量")
    print("-" * 60)

    effects = [0.001, 0.003, 0.005, 0.01, 0.02]
    base_p = 0.05
    target_power = 0.8

    print(f"  基准 CTR: {base_p:.1%}")
    print(f"  目标 power: {target_power:.0%}")
    print()
    print(f"  {'效应量':>8} {'提升%':>8} {'所需 n':>10}")
    print(f"  {'-'*28}")
    for delta in effects:
        # 粗略估算：用公式 n = (z_α + z_β)² × 2p(1-p) / δ²
        z_alpha = normal_ppf(0.95)  # 单侧 α=0.05
        z_beta = normal_ppf(target_power)
        p_avg = base_p + delta / 2
        n_est = int((z_alpha + z_beta) ** 2 * 2 * p_avg * (1 - p_avg) / (delta ** 2)) + 1
        print(f"  {delta:>8.3f} {delta/base_p*100:>7.1f}% {n_est:>10,}")

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print("""
1. 无差异时: 拒绝率 ≈ 5%（理论 α），不误报 ✅
2. 有差异时: 能检测出（power 随样本量增大而提高）✅
3. 效应量越小 → 所需样本量越大
4. 实战建议: A/B 测试前先算所需的样本量
    """)


if __name__ == "__main__":
    main()
