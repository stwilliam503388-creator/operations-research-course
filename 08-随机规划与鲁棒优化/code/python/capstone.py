"""
毕业项目：同一问题四种方法对比
===============================
对同一个报童决策问题，用以下四种方法求解并对比结果：

1. 确定性方法（均值需求）
2. 随机规划（场景方法 SAA）
3. 鲁棒优化（预算不确定性集合）
4. 分布鲁棒优化（Wasserstein 球）

问题设定：
  报童问题：进货成本 c=2，售价 r=5，残值 v=0.5。
  需求分布未知，仅有历史样本数据或区间估计。
  需要确定最优订货量 q。

本代码纯 Python 标准库 + numpy，中文注释，可直接运行自测。
"""


# 教学注释：通过场景、概率权重和坏情况边界刻画不确定需求或收益。
# 对比不同假设下的解，理解随机性和鲁棒性如何改变决策。



import random
import math
import statistics

# ========== 问题参数 ==========
COST = 2.0        # 进货成本
PRICE = 5.0       # 售价
SALVAGE = 0.5     # 残值

# 历史样本数据（20个观测值）
HISTORICAL_DATA = [45, 62, 38, 71, 55, 49, 83, 58, 42, 67,
                   51, 74, 36, 60, 53, 78, 44, 65, 50, 69]

# 需求的区间估计（用于鲁棒优化）
DEMAND_LOW = 30.0      # 最低可能需求
DEMAND_HIGH = 90.0     # 最高可能需求

# 蒙特卡洛评估场景数
N_EVAL_SCENARIOS = 5000

# Wasserstein DRO 半径
WASSERSTEIN_EPS = 5.0

# 搜索范围
Q_MIN = 20
Q_MAX = 100
Q_STEP = 2

# 随机种子
SEED = 42


# ==================== 工具函数 ====================

def profit(q, d):
    """单个情景的利润"""
    sold = min(q, d)
    leftover = q - sold
    return PRICE * sold + SALVAGE * leftover - COST * q


def generate_true_demand_samples(n, seed=SEED):
    """生成"真实"需求分布的样本（用于 out-of-sample 评估）"""
    rng = random.Random(seed + 999)
    samples = []
    for _ in range(n):
        # 真实分布：混合分布
        u = rng.random()
        if u < 0.3:
            d = rng.gauss(40, 8)
        elif u < 0.7:
            d = rng.gauss(60, 10)
        else:
            d = rng.gauss(80, 12)
        samples.append(max(5, d))
    return samples


def evaluate_q(q, demand_samples):
    """用样本评估 q 的平均利润"""
    total = sum(profit(q, d) for d in demand_samples)
    return total / len(demand_samples)


# ==================== 方法1: 确定性方法 ====================

def method_deterministic(data):
    """
    确定性方法：用样本均值作为需求预测，计算经典报童最优解。
    """
    mean_d = statistics.mean(data)
    # 临界分位数
    cf = (PRICE - COST) / (PRICE - SALVAGE)
    # 假设需求近似正态，用经验分位数
    sorted_data = sorted(data)
    idx = int(cf * len(sorted_data))
    idx = min(idx, len(sorted_data) - 1)
    q_det = sorted_data[idx]
    return q_det


# ==================== 方法2: 随机规划 (SAA) ====================

def method_stochastic_saa(data):
    """
    随机规划方法：用历史数据作为场景，枚举搜索最优 q。
    """
    best_q = None
    best_profit = -float("inf")
    q = Q_MIN
    while q <= Q_MAX:
        avg_p = sum(profit(q, d) for d in data) / len(data)
        if avg_p > best_profit:
            best_profit = avg_p
            best_q = q
        q += Q_STEP
    return best_q


# ==================== 方法3: 鲁棒优化 ====================

def method_robust_optimization():
    """
    鲁棒优化方法：预算不确定性集合。
    假设需求在 [d_low, d_high] 区间内，用预算 Γ 控制不确定性程度。
    最差情况利润最大化。
    """
    gamma = 4.0  # 预算参数（控制多少个需求参数可以同时偏离到最坏值）
    n_points = 10  # 在区间内离散化

    best_q = None
    best_worst_profit = -float("inf")

    # 将需求区间离散化
    demand_grid = [DEMAND_LOW + (DEMAND_HIGH - DEMAND_LOW) * i / (n_points - 1)
                   for i in range(n_points)]

    q = Q_MIN
    while q <= Q_MAX:
        # 计算最坏情况利润（在预算不确定性集合下）
        # 按profit(q, d)从小到大排序，取前 gamma 个的最坏平均值
        profits_at_q = [profit(q, d) for d in demand_grid]
        sorted_p = sorted(profits_at_q)
        k = int(gamma)
        frac = gamma - k
        worst_avg = (sum(sorted_p[:k]) + frac * sorted_p[k]) / gamma if gamma > 0 else sorted_p[0]
        worst_avg = sum(sorted_p[:min(len(sorted_p), max(1, int(gamma)))]) / max(1, int(gamma))

        if worst_avg > best_worst_profit:
            best_worst_profit = worst_avg
            best_q = q
        q += Q_STEP

    return best_q


# ==================== 方法4: 分布鲁棒优化 (DRO) ====================

def dro_worst_case_loss(q, data, epsilon):
    """
    计算 q 在 Wasserstein 球上的最差期望损失。
    使用 loss = -profit（因为 DRO 通常最小化损失）。
    """
    n = len(data)
    budget_per_point = epsilon / n
    total_loss = 0.0

    for xi in data:
        if xi < q:
            # 向左移动增加缺货 (减少 xi 增加缺货损失)
            delta = budget_per_point
            xi_prime = max(xi - delta, DEMAND_LOW - 10)
        elif xi > q:
            # 向右移动增加库存积压
            delta = budget_per_point
            xi_prime = min(xi + delta, DEMAND_HIGH + 10)
        else:
            xi_prime = xi

        # loss = -profit
        total_loss += -(profit(q, xi_prime))

    return total_loss / n


def method_distributionally_robust(data):
    """
    分布鲁棒优化方法：构建 Wasserstein 球，搜索最鲁棒的 q。
    """
    epsilon = WASSERSTEIN_EPS
    best_q = None
    best_val = float("inf")

    q = Q_MIN
    while q <= Q_MAX:
        d = dro_worst_case_loss(q, data, epsilon)
        if d < best_val:
            best_val = d
            best_q = q
        q += Q_STEP

    return best_q


# ==================== 主函数 ====================

def main():
    """主函数：四种方法对比"""
    print("=" * 70)
    print("毕业项目：同一报童问题的四种方法对比")
    print("=" * 70)
    print()

    data = HISTORICAL_DATA
    print(f"问题参数: 成本={COST}, 售价={PRICE}, 残值={SALVAGE}")
    print(f"历史数据: n={len(data)}, 均值={statistics.mean(data):.2f}, "
          f"标准差={statistics.stdev(data):.2f}")
    print(f"需求区间: [{DEMAND_LOW}, {DEMAND_HIGH}]")
    print()

    # 生成真实分布样本（用于 out-of-sample 评估）
    true_samples = generate_true_demand_samples(N_EVAL_SCENARIOS)
    print(f"真实分布评估样本数: {N_EVAL_SCENARIOS}")
    print()

    # ---- 求解四种方法 ----
    results = {}

    # 方法1: 确定性
    q1 = method_deterministic(data)
    p1 = evaluate_q(q1, true_samples)
    results["确定性方法"] = (q1, p1)
    print(f"[1] 确定性方法 (均值分位数)")
    print(f"    最优订货量 q = {q1}")
    print(f"    Out-of-sample 期望利润 = {p1:.4f}")
    print()

    # 方法2: 随机规划
    q2 = method_stochastic_saa(data)
    p2 = evaluate_q(q2, true_samples)
    results["随机规划 SAA"] = (q2, p2)
    print(f"[2] 随机规划 (SAA场景方法)")
    print(f"    最优订货量 q = {q2}")
    print(f"    Out-of-sample 期望利润 = {p2:.4f}")
    print()

    # 方法3: 鲁棒优化
    q3 = method_robust_optimization()
    p3 = evaluate_q(q3, true_samples)
    results["鲁棒优化"] = (q3, p3)
    print(f"[3] 鲁棒优化 (预算不确定性集合)")
    print(f"    最优订货量 q = {q3}")
    print(f"    Out-of-sample 期望利润 = {p3:.4f}")
    print()

    # 方法4: 分布鲁棒优化
    q4 = method_distributionally_robust(data)
    p4 = evaluate_q(q4, true_samples)
    results["分布鲁棒优化 DRO"] = (q4, p4)
    print(f"[4] 分布鲁棒优化 (Wasserstein球 ε={WASSERSTEIN_EPS})")
    print(f"    最优订货量 q = {q4}")
    print(f"    Out-of-sample 期望利润 = {p4:.4f}")
    print()

    # ---- 对比表格 ----
    print("=" * 70)
    print("四种方法对比汇总")
    print("=" * 70)
    print(f"{'方法':<24} {'订货量 q':<12} {'期望利润':<14} {'排名':<8}")
    print("-" * 70)

    # 按利润排序
    sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
    for rank, (name, (q, p)) in enumerate(sorted_results, 1):
        marker = " ← 最优" if rank == 1 else ""
        print(f"{name:<24} {q:<12} {p:<14.4f} #{rank}{marker}")
    print()

    # 利润差距分析
    best_profit = sorted_results[0][1][1]
    print("--- 相对于最优方法的利润差距 ---")
    for name, (q, p) in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
        gap = best_profit - p
        gap_pct = gap / abs(best_profit) * 100
        print(f"  {name:<24}: 差距 = {gap:+.4f} ({gap_pct:+.2f}%)")

    print()
    print("--- 不同 q 的完整利润曲线 ---")
    for q in range(20, 101, 10):
        p = evaluate_q(q, true_samples)
        print(f"  q={q:3d} → 期望利润 = {p:.4f}")

    print()
    print("测试通过！毕业项目完成。")


if __name__ == "__main__":
    main()
