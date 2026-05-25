"""
案例5：分布鲁棒优化 (Wasserstein DRO)
难度：★★★★☆
方法：分布鲁棒优化
依赖：pip install numpy scipy
运行：python code/case05_distributionally_robust.py

简述：用 Wasserstein DRO 求解新闻摊贩问题——当真实分布未知时，
      在最坏的「可能分布」下做最优订货决策。
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


# ============================================================
# 数据准备
# ============================================================
c = 0.5     # 进货价
p = 1.0     # 零售价
v = 0.1     # 退货价
n_samples = 100  # 历史数据量

rng = np.random.default_rng(42)

# 真实分布（学生不知道）：偏态分布（Gamma 分布）
# 形状参数调成均值≈100，方差≈400
true_shape = 25  # Gamma shape
true_scale = 4   # Gamma scale
# Gamma(25, 4): 均值=100, 方差=100*4=400, 标准差=20

# 生成历史数据（从真实分布抽样）
historical_data = rng.gamma(true_shape, true_scale, n_samples)
data_mean = np.mean(historical_data)
data_std = np.std(historical_data)
data_min = np.min(historical_data)
data_max = np.max(historical_data)


# ============================================================
# 利润函数
# ============================================================
def profit(q, d):
    """给定进货量 q 和需求 d 的利润。"""
    sold = min(d, q)
    leftover = max(q - d, 0)
    return sold * p + leftover * v - q * c


# ============================================================
# 方法1: 随机规划（正态假设）
# ============================================================
def solve_sp_normal():
    """
    假设需求服从 Normal(data_mean, data_std) 的随机规划。
    用临界分位数公式。
    """
    critical_ratio = (p - c) / (p - v)
    q_star = stats.norm.ppf(critical_ratio, loc=data_mean, scale=data_std)
    return q_star


# ============================================================
# 方法2: 随机规划（经验分布）
# ============================================================
def expected_profit_empirical(q, data):
    """用经验分布（等权重样本）计算期望利润。"""
    profits = np.array([profit(q, d) for d in data])
    return np.mean(profits)


def solve_sp_empirical(data):
    """用经验分布搜索最优订货量。"""
    best_q = None
    best_profit = -np.inf
    for q in range(int(data_min * 0.5), int(data_max * 1.5) + 1):
        ep = expected_profit_empirical(q, data)
        if ep > best_profit:
            best_profit = ep
            best_q = q
    return best_q, best_profit


# ============================================================
# 方法3: Wasserstein DRO
# ============================================================
def wasserstein_worst_case_profit(q, data, epsilon):
    """
    对给定的 q，求解 Wasserstein 模糊球内的最坏情况期望利润。
    
    使用 Wasserstein DRO 对偶形式（简化版）：
    在 1-Wasserstein 距离下，单变量问题的对偶形式可以通过
    对样本点进行某种变换得到。
    
    简化实现：对于单变量新闻摊贩问题，最坏分布等价于
    将样本点向「使利润最小化」的方向移动最多 epsilon 距离。
    
    注意：这是一个简化实现，完整 Wasserstein DRO 需要求解 LP。
    """
    n = len(data)
    # 对每个样本点，计算利润对需求的导数（粗略估计移动方向）
    # 对 newsvendor：d(profit)/dD = p if D <= q, 0 otherwise
    # 最坏分布会移动样本点使利润降低：
    # - 对 D <= q 的样本，降低需求（减少销售）
    # - 对 D > q 的样本，提高需求（不增加利润但可能...）
    # 实际上简化：把样本向对利润最不利的方向移动
    
    worst_profits = []
    for d in data:
        if d <= q:
            # 需求低于进货量：利润 = d*p + (q-d)*v - q*c
            # 降低需求会减少利润，最多降低 epsilon
            # 但需求不能小于 0
            d_worst = max(0, d - epsilon)
        else:
            # 需求高于进货量：利润 = q*p - q*c
            # 提高需求不改变利润（因为销量上限是 q）
            # 但降低需求会使利润减少
            d_worst = max(0, d - epsilon)
        worst_profits.append(profit(q, d_worst))
    
    return np.mean(worst_profits)


def solve_dro(data, epsilon):
    """
    求解 Wasserstein DRO: 找到最大化最坏期望利润的 q。
    """
    best_q = None
    best_worst_profit = -np.inf
    for q in range(int(data_min * 0.5), int(data_max * 1.5) + 1):
        wp = wasserstein_worst_case_profit(q, data, epsilon)
        if wp > best_worst_profit:
            best_worst_profit = wp
            best_q = q
    return best_q, best_worst_profit


# ============================================================
# 方法4: 鲁棒优化（箱式）
# ============================================================
def solve_robust_box():
    """
    箱式鲁棒优化：假设需求在 [data_min, data_max] 之间。
    最坏情况利润最大化。
    """
    # 鲁棒目标：max_q min_{D in [min, max]} profit(q, D)
    # 最坏情况是需求最小的时候：d = data_min
    # profit(q, data_min) 在 q = data_min 时取最大值
    # 但这太保守了...
    # 更合理的：用最坏情况约束
    
    best_q = None
    best_profit = -np.inf
    for q in range(int(data_min * 0.5), int(data_max * 1.5) + 1):
        # 最坏情况：需求取最小值
        worst = profit(q, data_min)
        if worst > best_profit:
            best_profit = worst
            best_q = q
    return best_q, best_profit


# ============================================================
# Out-of-sample 评估
# ============================================================
def evaluate_out_of_sample(q, n_test=10000):
    """在真实分布（Gamma）下评估方案表现。"""
    test_demands = rng.gamma(true_shape, true_scale, n_test)
    profits = np.array([profit(q, d) for d in test_demands])
    return np.mean(profits), np.percentile(profits, 5)


# ============================================================
# 结果输出
# ============================================================
def print_summary():
    print("=" * 60)
    print("案例5：分布鲁棒优化 (Wasserstein DRO)")
    print("=" * 60)

    print(f"\n▶ 历史数据: {n_samples} 天")
    print(f"  均值 = {data_mean:.1f}, 标准差 = {data_std:.1f}")
    print(f"  最小值 = {data_min:.1f}, 最大值 = {data_max:.1f}")
    print(f"  真实分布: Gamma(shape={true_shape}, scale={true_scale})")

    # 1. 随机规划（正态假设）
    q_sp_norm = solve_sp_normal()
    print(f"\n▶ 随机规划 (正态假设):")
    print(f"  订货量 = {q_sp_norm:.1f}")

    # 2. 随机规划（经验分布）
    q_sp_emp, r_sp_emp = solve_sp_empirical(historical_data)
    print(f"\n▶ 随机规划 (经验分布):")
    print(f"  订货量 = {q_sp_emp}, 期望利润 = {r_sp_emp:.2f}")

    # 3. DRO (不同 ε)
    print(f"\n▶ 分布鲁棒优化 (不同 ε 值):")
    epsilons = [0, 0.5, 1.0, 2.0, 5.0]
    dro_results = []
    for eps in epsilons:
        q_dro, wp_dro = solve_dro(historical_data, eps)
        dro_results.append((eps, q_dro, wp_dro))
        print(f"  ε={eps:.1f}: 订货量={q_dro}, 最坏期望利润={wp_dro:.2f}")

    # 4. 鲁棒优化
    q_rob, r_rob = solve_robust_box()
    print(f"\n▶ 鲁棒优化 (箱式):")
    print(f"  订货量 = {q_rob}, 最坏利润 = {r_rob:.2f}")

    # 5. Out-of-sample 对比
    print(f"\n▶ Out-of-sample 对比 (在真实 Gamma 分布下测试 10000 样本):")
    print(f"{'方法':>25} | {'订货量':>8} | {'期望利润':>10} | {'最差5%':>10}")
    print("-" * 65)
    
    # SP 正态
    mean_sp, p5_sp = evaluate_out_of_sample(round(q_sp_norm))
    print(f"{'随机规划 (正态)':>25} | {q_sp_norm:>8.1f} | {mean_sp:>10.2f} | {p5_sp:>10.2f}")
    
    # SP 经验
    mean_se, p5_se = evaluate_out_of_sample(q_sp_emp)
    print(f"{'随机规划 (经验)':>25} | {q_sp_emp:>8} | {mean_se:>10.2f} | {p5_se:>10.2f}")
    
    # DRO
    for eps, q_dro, wp in dro_results:
        mean_dro, p5_dro = evaluate_out_of_sample(q_dro)
        print(f"{f'DRO (ε={eps:.1f})':>25} | {q_dro:>8} | {mean_dro:>10.2f} | {p5_dro:>10.2f}")
    
    # 鲁棒
    mean_r, p5_r = evaluate_out_of_sample(q_rob)
    print(f"{'鲁棒优化 (箱式)':>25} | {q_rob:>8} | {mean_r:>10.2f} | {p5_r:>10.2f}")


def solve_small():
    """小规模测试"""
    print("\n=== 小规模测试 ===")
    q, wp = solve_dro(historical_data[:20], 1.0)
    print(f"DRO (ε=1.0, 20样本): q={q}, 最坏期望利润={wp:.2f}")


def solve_medium():
    """中等规模测试：不同半径对比"""
    print("\n=== 中等规模测试: ε 灵敏度分析 ===")
    for eps in [0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        q, wp = solve_dro(historical_data, eps)
        print(f"ε={eps:>5.1f}: q={q:>4}, 最坏期望利润={wp:.2f}")


if __name__ == "__main__":
    solve_small()
    solve_medium()
    print("\n")
    print_summary()
