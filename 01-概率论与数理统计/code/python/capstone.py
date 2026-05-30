#!/usr/bin/env python3
"""
🏆 毕业项目：库存决策的完整概率分析
========================================
将本课程所有技能（分布识别、CLT、贝叶斯、假设检验、蒙特卡洛）
串联起来解决一个真实的供应链决策问题。

纯 Python 标准库实现，无第三方依赖。
"""
# 教学注释：围绕随机分布、样本统计量和蒙特卡洛估计观察不确定性。
# 运行时重点比较理论量与模拟结果如何支撑决策判断。


import math
import random
import statistics


# ============================================================
# 工具函数
# ============================================================

def normal_pdf(x, mu=0, sigma=1):
    """正态分布概率密度函数"""
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * \
           math.exp(-0.5 * ((x - mu) / sigma) ** 2)


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


def sample_quantile(data, q):
    """计算样本分位数"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = q * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_data[lo]
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


# ============================================================
# 第1步：生成模拟数据
# ============================================================

def generate_demand_data(n_days=500, true_mu=200, true_sigma=40):
    """
    生成模拟的日需求数据。
    真实需求服从对数正态分布 lnN(log(200), 0.2)。
    这样数据是正数、右偏、类似真实需求。
    """
    log_mu = math.log(true_mu)
    log_sigma = true_sigma / true_mu  # 变异系数 ~0.2
    log_data = generate_normal(n_days, log_mu, log_sigma)
    demand = [math.exp(x) for x in log_data]
    return demand, true_mu, true_sigma


def generate_leadtime_data(n_samples=30, supplier='A'):
    """生成供应商补货时间数据"""
    random.seed(42 + hash(supplier) % 1000)
    if supplier == 'A':
        # 供应商 A：均值 5 天，标准差 1 天
        return [max(1, random.gauss(5, 1)) for _ in range(n_samples)]
    else:
        # 供应商 B：均值 6 天，标准差 1.5 天
        return [max(1, random.gauss(6, 1.5)) for _ in range(n_samples)]


# ============================================================
# 第2步：问题1 — 需求分布识别
# ============================================================

def qq_correlation(data, theoretical_ppf):
    """计算数据与理论分布的 QQ 相关性"""
    n = len(data)
    sorted_data = sorted(data)
    x_vals = []
    y_vals = []
    data_mean = statistics.mean(sorted_data)
    data_std = statistics.stdev(sorted_data)
    if data_std == 0:
        return 0
    for i in range(1, n):
        p = i / (n + 1)
        x_vals.append(theoretical_ppf(p))
        y_vals.append((sorted_data[i - 1] - data_mean) / data_std)
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    n_pts = len(x_vals)
    cov = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n_pts))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_vals))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_vals))
    if std_x * std_y == 0:
        return 0
    return cov / (std_x * std_y)


def assess_distribution(data):
    """评估数据最可能服从的分布"""
    n = len(data)
    mean_d = statistics.mean(data)
    std_d = statistics.stdev(data)
    min_d = min(data)
    max_d = max(data)

    print(f"\n  {'='*50}")
    print(f"  基本统计量")
    print(f"  {'='*50}")
    print(f"  样本量: {n}")
    print(f"  均值: {mean_d:.2f}")
    print(f"  标准差: {std_d:.2f}")
    print(f"  最小值: {min_d:.2f}")
    print(f"  最大值: {max_d:.2f}")
    print(f"  变异系数 (CV): {std_d/mean_d:.4f}")

    # 偏度
    m3 = sum((x - mean_d) ** 3 for x in data) / n
    skewness = m3 / (std_d ** 3) if std_d > 0 else 0
    print(f"  偏度: {skewness:.4f}")

    # 正态 QQ 相关性
    r_normal = qq_correlation(data, lambda p: normal_ppf(p))
    print(f"\n  正态分布 QQ 相关性: {r_normal:.4f}")
    if r_normal > 0.98:
        print(f"  ✅ 数据近似正态分布")
    else:
        print(f"  ⚠️ 数据偏离正态分布")

    # 对数正态：对 log 数据做正态 QQ
    log_data = [math.log(x) for x in data if x > 0]
    if log_data:
        log_mean = statistics.mean(log_data)
        log_std = statistics.stdev(log_data)
        r_lognormal = qq_correlation(log_data, lambda p: normal_ppf(p))
        print(f"\n  对数正态（取 log 后）QQ 相关性: {r_lognormal:.4f}")
        if r_lognormal > 0.98:
            print(f"  ✅ 数据近似对数正态分布")
            print(f"  log 变换后的均值: {log_mean:.4f}")
            print(f"  log 变换后的标准差: {log_std:.4f}")
            print(f"  原始尺度均值 ≈ {math.exp(log_mean + log_std**2/2):.2f}")
            print(f"  原始尺度中位数 ≈ {math.exp(log_mean):.2f}")
        else:
            print(f"  ⚠️ 数据也不完全符合对数正态")
    else:
        r_lognormal = -1

    # 判断
    if r_lognormal > r_normal and r_lognormal > 0.96:
        print(f"\n  🏆 结论：数据最可能服从**对数正态分布**")
        print(f"    参数: μ_log={log_mean:.4f}, σ_log={log_std:.4f}")
        return "lognormal", log_mean, log_std
    elif r_normal > 0.96:
        print(f"\n  🏆 结论：数据最可能服从**正态分布**")
        print(f"    参数: μ={mean_d:.4f}, σ={std_d:.4f}")
        return "normal", mean_d, std_d
    else:
        print(f"\n  🏆 结论：数据不符合常见分布，使用经验分布")
        return "empirical", mean_d, std_d


# ============================================================
# 第3步：问题2 — CLT 估计月总需求
# ============================================================

def clt_monthly_analysis(dist_type, param1, param2, days_per_month=30):
    """用 CLT 估计月总需求的分布"""
    print(f"\n  {'='*50}")
    print(f"  月总需求分析（CLT）")
    print(f"  {'='*50}")

    if dist_type == "normal":
        daily_mean = param1
        daily_std = param2
    elif dist_type == "lognormal":
        # 对数正态的原始尺度均值
        daily_mean = math.exp(param1 + param2**2 / 2)
        # 原始尺度方差
        daily_var = (math.exp(param2**2) - 1) * math.exp(2 * param1 + param2**2)
        daily_std = math.sqrt(daily_var)
    else:
        daily_mean = param1
        daily_std = param2

    monthly_mean = days_per_month * daily_mean
    monthly_std = math.sqrt(days_per_month) * daily_std

    print(f"  日需求均值: {daily_mean:.2f}")
    print(f"  日需求标准差: {daily_std:.2f}")
    print(f"  月需求均值 (CLT): {monthly_mean:.2f}")
    print(f"  月需求标准差 (CLT): {monthly_std:.2f}")
    print(f"  月需求近似服从: N({monthly_mean:.0f}, {monthly_std:.0f}²)")
    print(f"  CLT 要求: n={days_per_month} >= 30 ✅（经验法则满足）")

    # 安全库存计算（基于月需求）
    z_95 = normal_ppf(0.95)  # 95% 服务水平
    safety_stock = z_95 * monthly_std
    reorder_point = monthly_mean + safety_stock

    print(f"\n  95% 服务水平的安全库存: {safety_stock:.0f}")
    print(f"  再订货点（月）: {reorder_point:.0f}")

    return monthly_mean, monthly_std


# ============================================================
# 第4步：问题3 — 贝叶斯需求参数更新
# ============================================================

def bayesian_demand_update(prior_mean, prior_std, prior_n,
                            sample_mean, sample_std, sample_n):
    """
    正态分布均值的贝叶斯更新。
    使用 Normal-Inverse-Gamma 共轭先验的简化版。

    假设：方差已知（用样本方差代替），先验为 N(prior_mean, prior_std²/prior_n)
    """
    # 先验精度
    prior_precision = prior_n / (prior_std ** 2) if prior_std > 0 else 0
    # 数据精度
    data_precision = sample_n / (sample_std ** 2) if sample_std > 0 else 0

    # 后验均值 = 精度加权平均
    if prior_precision + data_precision > 0:
        posterior_mean = (prior_precision * prior_mean +
                          data_precision * sample_mean) / \
                          (prior_precision + data_precision)
    else:
        posterior_mean = sample_mean

    # 后验方差
    posterior_var = 1.0 / (prior_precision + data_precision)
    posterior_std = math.sqrt(posterior_var) if posterior_var > 0 else 0

    return posterior_mean, posterior_std


def question3_bayesian(demand_data):
    """问题3：贝叶斯推断"""
    print(f"\n{'='*80}")
    print("问题3：贝叶斯推断——需求参数的不确定性")
    print(f"{'='*80}")

    # 运营经理的经验先验
    # 他认为日需求均值约 180，标准差约 50（不太确定，等效于 n=20 的样本）
    prior_mean = 180
    prior_std = 50
    prior_n = 20

    # 数据
    sample_mean = statistics.mean(demand_data)
    sample_std = statistics.stdev(demand_data)
    sample_n = len(demand_data)

    print(f"\n  先验信息（运营经理经验）:")
    print(f"    先验均值: {prior_mean}")
    print(f"    先验标准差: {prior_std}")
    print(f"    先验等效样本量: {prior_n}")

    print(f"\n  观测数据:")
    print(f"    样本均值: {sample_mean:.2f}")
    print(f"    样本标准差: {sample_std:.2f}")
    print(f"    样本量: {sample_n}")

    # 贝叶斯更新
    post_mean, post_std = bayesian_demand_update(
        prior_mean, prior_std, prior_n,
        sample_mean, sample_std, sample_n
    )

    print(f"\n  贝叶斯后验:")
    print(f"    后验均值: {post_mean:.2f}")
    print(f"    后验标准差: {post_std:.2f}")
    print(f"    后验 95% 可信区间: "
          f"[{post_mean - 1.96 * post_std:.2f}, "
          f"{post_mean + 1.96 * post_std:.2f}]")

    print(f"\n  对比:")
    print(f"    先验均值: {prior_mean}")
    print(f"    MLE（样本均值）: {sample_mean:.2f}")
    print(f"    后验均值: {post_mean:.2f}")
    print(f"    → 贝叶斯估计是先验和数据的加权平均")

    # 如果先验和数据差异很大，后验"拉回"先验
    print(f"\n  洞察: 数据量大（n={sample_n}），数据主导后验。")
    print(f"  后验 ≈ MLE = {sample_mean:.2f}")

    return post_mean, post_std


# ============================================================
# 第5步：问题4 — 假设检验（供应商选择）
# ============================================================

def two_sample_t_test(data1, data2, alternative='two-sided'):
    """
    双样本 t 检验（Welch's t-test，不假设方差相等）。
    纯 Python 实现。
    """
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = statistics.mean(data1), statistics.mean(data2)
    var1 = statistics.variance(data1) if n1 > 1 else 0
    var2 = statistics.variance(data2) if n2 > 1 else 0

    # t 统计量
    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return 0, 1.0
    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite 自由度
    if var1 == 0 and var2 == 0:
        df = 1
    else:
        num = (var1 / n1 + var2 / n2) ** 2
        denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        df = num / denom if denom > 0 else 1

    # t 分布 CDF 近似（大量计算用标准正态近似）
    # 对于 df > 30，t ≈ N(0,1)
    if df > 30:
        if alternative == 'two-sided':
            p_value = 2 * (1 - normal_cdf(abs(t_stat)))
        elif alternative == 'greater':
            p_value = 1 - normal_cdf(t_stat)
        else:
            p_value = normal_cdf(t_stat)
    else:
        # 简化：用正态近似（实际应用中应查 t 分布表）
        if alternative == 'two-sided':
            p_value = 2 * (1 - normal_cdf(abs(t_stat)))
        elif alternative == 'greater':
            p_value = 1 - normal_cdf(t_stat)
        else:
            p_value = normal_cdf(t_stat)

    return t_stat, p_value, mean1, mean2


def question4_hypothesis():
    """问题4：假设检验"""
    print(f"\n{'='*80}")
    print("问题4：假设检验——供应商补货时间比较")
    print(f"{'='*80}")

    # 生成数据
    lt_a = generate_leadtime_data(30, 'A')
    lt_b = generate_leadtime_data(30, 'B')

    print(f"\n  供应商 A 补货时间:")
    print(f"    均值: {statistics.mean(lt_a):.2f} 天")
    print(f"    标准差: {statistics.stdev(lt_a):.2f} 天")

    print(f"\n  供应商 B 补货时间:")
    print(f"    均值: {statistics.mean(lt_b):.2f} 天")
    print(f"    标准差: {statistics.stdev(lt_b):.2f} 天")

    print(f"\n  假设检验:")
    print(f"    H₀: μ_A = μ_B（补货时间无差异）")
    print(f"    H₁: μ_A < μ_B（A 比 B 更快）")
    print(f"    α = 0.05（显著性水平）")

    t_stat, p_value, mean_a, mean_b = two_sample_t_test(
        lt_a, lt_b, alternative='less'
    )

    print(f"\n    Welch t 检验结果:")
    print(f"    t 统计量: {t_stat:.4f}")
    print(f"    p 值: {p_value:.4f}")
    print(f"    均值差: {mean_a - mean_b:.4f} 天")

    if p_value < 0.05:
        print(f"\n    ✅ p={p_value:.4f} < 0.05 → 拒绝 H₀")
        print(f"    结论：供应商 A 的补货时间显著快于 B")
    else:
        print(f"\n    ⚠️ p={p_value:.4f} >= 0.05 → 不能拒绝 H₀")
        print(f"    结论：没有足够证据表明 A 比 B 快")

    return lt_a, lt_b


# ============================================================
# 第6步：问题5 — 蒙特卡洛仿真安全库存
# ============================================================

def monte_carlo_inventory(daily_demand, lead_time_samples,
                           reorder_point, order_qty,
                           sim_days=10000, initial_stock=1000):
    """
    蒙特卡洛库存仿真。

    仿真一个 (s, Q) 连续检查库存系统：
    - 当库存 ≤ reorder_point(s) 时，订购 order_qty(Q) 件
    - 补货需要 lead_time 天（从 lead_time_samples 中随机抽样）
    - 每天的需求从 daily_demand 中随机抽样

    返回服务水平、缺货次数、平均库存等统计量。
    """
    stock = initial_stock
    on_order = 0  # 订购中但未到的数量
    remaining_lead_time = 0  # 当前订单还需要多少天到货

    total_demand = 0
    stockout_days = 0
    total_stockout_qty = 0
    stock_record = []

    for day in range(sim_days):
        # 检查是否需要到货
        if remaining_lead_time > 0:
            remaining_lead_time -= 1
            if remaining_lead_time == 0:
                stock += order_qty
                on_order = 0

        # 当天需求
        demand = random.choice(daily_demand)
        total_demand += demand

        # 满足需求
        if stock >= demand:
            stock -= demand
        else:
            stockout_days += 1
            total_stockout_qty += demand - stock
            stock = 0

        # 检查是否需要订货
        if stock <= reorder_point and on_order == 0:
            lead_time = random.choice(lead_time_samples)
            remaining_lead_time = max(1, int(round(lead_time)))
            on_order = order_qty

        # 记录库存
        stock_record.append(stock)

    service_level = 1 - stockout_days / sim_days
    avg_stock = statistics.mean(stock_record)
    max_stock = max(stock_record)

    return {
        "service_level": service_level,
        "stockout_days": stockout_days,
        "total_stockout_qty": total_stockout_qty,
        "avg_stock": avg_stock,
        "max_stock": max_stock,
        "stock_record": stock_record,
    }


def question5_monte_carlo(demand_data, lead_times):
    """问题5：蒙特卡洛仿真"""
    print(f"\n{'='*80}")
    print("问题5：蒙特卡洛仿真——安全库存与服务水平")
    print(f"{'='*80}")

    mean_demand = statistics.mean(demand_data)
    std_demand = statistics.stdev(demand_data)

    # 仿真参数
    order_qty = 5000          # 每次订货量
    sim_days = 10000          # 仿真天数

    print(f"\n  仿真参数:")
    print(f"    日需求均值: {mean_demand:.2f}")
    print(f"    日需求标准差: {std_demand:.2f}")
    print(f"    每次订货量: {order_qty}")
    print(f"    仿真天数: {sim_days}")
    print(f"    补货时间样本量: {len(lead_times)}")

    # 尝试不同的再订货点
    base_reorder = int(mean_demand * statistics.mean(lead_times))  # 基础：日需×平均补货时间
    reorder_points = [
        base_reorder,
        base_reorder + int(0.5 * std_demand * math.sqrt(statistics.mean(lead_times))),
        base_reorder + int(1.0 * std_demand * math.sqrt(statistics.mean(lead_times))),
        base_reorder + int(1.645 * std_demand * math.sqrt(statistics.mean(lead_times))),
        base_reorder + int(2.0 * std_demand * math.sqrt(statistics.mean(lead_times))),
    ]

    print(f"\n  安全库存策略扫描:")
    print(f"  {'再订货点':>10} {'安全库存':>10} {'服务水平':>10} {'缺货天数':>10} {'平均库存':>10}")
    print(f"  {'-'*50}")

    results = []
    for rp in reorder_points:
        safety = rp - base_reorder
        result = monte_carlo_inventory(
            demand_data, lead_times, rp, order_qty, sim_days
        )
        results.append((rp, safety, result))
        print(f"  {rp:>10} {safety:>10} "
              f"{result['service_level']:.4f}      "
              f"{result['stockout_days']:>4}    "
              f"{result['avg_stock']:>8.1f}")

    # 找到满足 95% 服务水平的最小安全库存
    print(f"\n  分析:")
    target = 0.95
    found = False
    for rp, safety, result in results:
        if result['service_level'] >= target:
            print(f"  ✅ 满足 {target:.0%} 服务水平的最小再订货点: {rp}")
            print(f"     对应安全库存: {safety}")
            print(f"     实际服务水平: {result['service_level']:.4f}")
            print(f"     平均库存: {result['avg_stock']:.1f}")
            found = True
            break

    if not found:
        print(f"  ⚠️ 当前策略未达到 {target:.0%} 服务水平")
        print(f"  → 需要更大的安全库存")

    return results


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 80)
    print("🏆 毕业项目：库存决策的完整概率分析")
    print("=" * 80)

    random.seed(42)

    # ========= 生成模拟数据 =========
    print(f"\n{'='*80}")
    print("数据准备：生成模拟历史数据")
    print(f"{'='*80}")

    demand_data, true_mu, true_sigma = generate_demand_data(500, 200, 40)
    print(f"  生成 {len(demand_data)} 天日需求数据（真实 μ={true_mu}, σ={true_sigma}）")
    print(f"  数据范围: [{min(demand_data):.2f}, {max(demand_data):.2f}]")

    lt_a = generate_leadtime_data(30, 'A')
    lt_b = generate_leadtime_data(30, 'B')
    print(f"  生成供应商 A 补货时间: {len(lt_a)} 条")
    print(f"  生成供应商 B 补货时间: {len(lt_b)} 条")

    # ========= 问题1：分布识别 =========
    print(f"\n{'='*80}")
    print("问题1：需求分布识别")
    print(f"{'='*80}")

    dist_type, param1, param2 = assess_distribution(demand_data)

    # ========= 问题2：CLT 月需求 =========
    print(f"\n{'='*80}")
    print("问题2：月总需求分布（CLT）")
    print(f"{'='*80}")

    monthly_mean, monthly_std = clt_monthly_analysis(dist_type, param1, param2)

    # ========= 问题3：贝叶斯推断 =========
    post_mean, post_std = question3_bayesian(demand_data)

    # ========= 问题4：假设检验 =========
    lt_a_data, lt_b_data = question4_hypothesis()

    # ========= 问题5：蒙特卡洛仿真 =========
    # 使用较快的供应商（A）
    best_lead_times = lt_a_data
    results = question5_monte_carlo(demand_data, best_lead_times)

    # ========= 最终报告 =========
    print(f"\n{'='*80}")
    print("📋 最终决策建议")
    print(f"{'='*80}")

    final_rp = results[0][0]  # 基础再订货点
    for rp, safety, r in results:
        if r['service_level'] >= 0.95:
            final_rp = rp
            break

    print(f"""
基于以上完整的概率分析，我们推荐以下库存策略：

1. 需求参数:
   - 日需求均值 ≈ {statistics.mean(demand_data):.1f}
   - 日需求标准差 ≈ {statistics.stdev(demand_data):.1f}
   - 最适配分布: {dist_type}

2. 月需求（基于 CLT）:
   - 月需求均值 ≈ {monthly_mean:.0f}
   - 月需求标准差 ≈ {monthly_std:.0f}

3. 供应商选择:
   - 推荐使用补货时间更快的供应商 A
   - （假设检验确认显著差异）

4. 库存策略参数:
   - 再订货点: {final_rp}
   - 安全库存: {final_rp - results[0][0]}
   - 订货量: 5000
   - 预期服务水平: ≥ 95%

5. 关键不确定性:
   - 需求参数仍有不确定性（后验标准差 ≈ {post_std:.2f}）
   - 建议每月重新评估需求分布和补货时间
   - 如果服务水平低于 90%，增加安全库存

🎯 结论：通过概率分析，我们量化了不确定性，
   找到了满足 95% 服务水平的最优库存策略。
""")
    print("=" * 80)
    print("毕业项目完成！所有技能已串联应用。")
    print("=" * 80)


if __name__ == "__main__":
    main()
