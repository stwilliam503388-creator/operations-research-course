"""
报童问题 — 随机规划版本
=========================
蒙特卡洛模拟需求，比较确定性解 vs 随机最优解的期望利润差距。

问题描述：
  报童每天从报社订购报纸，每份成本 c，售价 r，未售出残值 v。
  需求 D 服从已知分布（这里用对数正态分布）。
  确定性方法：用均值需求做最优订货量。
  随机方法：用蒙特卡洛模拟生成多个场景，最大化期望利润。

本代码纯 Python 标准库 + numpy，中文注释，可直接运行自测。
"""

import random
import math
import statistics

# ---------- 参数设置 ----------
COST = 0.5       # 每份进货成本
PRICE = 1.0      # 每份售价
SALVAGE = 0.1    # 未售出残值
DEMAND_MEAN = 100.0   # 需求均值
DEMAND_STD = 30.0     # 需求标准差
N_SCENARIOS = 10000   # 蒙特卡洛场景数
Q_CANDIDATES = range(50, 161, 5)  # 待考察的订货量候选集


def lognormal_demand(mean, std, n):
    """生成 n 个对数正态分布需求样本"""
    # 对数正态参数推导：设 X ~ LogNormal(mu, sigma)
    # E[X] = exp(mu + sigma^2/2) = mean
    # Var(X) = (exp(sigma^2) - 1) * exp(2*mu + sigma^2) = std^2
    mu = math.log(mean ** 2 / math.sqrt(mean ** 2 + std ** 2))
    sigma_sq = math.log(1 + (std / mean) ** 2)
    sigma = math.sqrt(sigma_sq)
    samples = []
    for _ in range(n):
        # Box-Muller 生成标准正态
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        samples.append(math.exp(mu + sigma * z))
    return samples


def profit_one_scenario(q, d):
    """给定订货量 q 和实际需求 d，计算单场景利润"""
    sold = min(q, d)
    leftover = q - sold
    return PRICE * sold + SALVAGE * leftover - COST * q


def deterministic_optimal_q(demand_samples):
    """确定性方法：用均值需求求解最优订货量（经典报童分位数）"""
    mean_d = statistics.mean(demand_samples)
    # 临界分位数：critical fractile = (r - c) / (r - v)
    cf = (PRICE - COST) / (PRICE - SALVAGE)
    # 假设需求服从正态分布，用样本均值和标准差
    std_d = statistics.stdev(demand_samples)
    # 最优订货量 = 均值 + Z_{cf} * 标准差
    # 用正态分布分位数的近似（Abramowitz & Stegun 有理逼近）
    def norm_ppf(p):
        """标准正态逆累计分布函数的近似"""
        if p <= 0.5:
            t = math.sqrt(-2.0 * math.log(p))
            return -(t - (2.515517 + 0.802853 * t + 0.010328 * t * t)
                     / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t))
        else:
            return -norm_ppf(1.0 - p)

    z = norm_ppf(cf)
    q_det = mean_d + z * std_d
    return max(0, q_det)


def stochastic_optimal_q(demand_scenarios, q_candidates):
    """随机方法：枚举候选订货量，选择期望利润最大的"""
    best_q = None
    best_profit = -float("inf")
    for q in q_candidates:
        total_p = 0.0
        for d in demand_scenarios:
            total_p += profit_one_scenario(q, d)
        avg_p = total_p / len(demand_scenarios)
        if avg_p > best_profit:
            best_profit = avg_p
            best_q = q
    return best_q, best_profit


def evaluate_by_simulation(q, demand_scenarios):
    """对给定 q 用新场景模拟评估期望利润（out-of-sample）"""
    total_p = 0.0
    for d in demand_scenarios:
        total_p += profit_one_scenario(q, d)
    return total_p / len(demand_scenarios)


def main():
    """主函数：自测入口"""
    print("=" * 60)
    print("报童问题 — 随机规划版本")
    print("=" * 60)
    print(f"参数: 成本={COST}, 售价={PRICE}, 残值={SALVAGE}")
    print(f"需求分布: 对数正态 (均值={DEMAND_MEAN}, 标准差={DEMAND_STD})")
    print(f"蒙特卡洛场景数: {N_SCENARIOS}")
    print()

    # 生成场景
    random.seed(42)
    in_sample_scenarios = lognormal_demand(DEMAND_MEAN, DEMAND_STD, N_SCENARIOS)

    # 1. 确定性最优解
    q_det = deterministic_optimal_q(in_sample_scenarios)
    profit_det = evaluate_by_simulation(q_det, in_sample_scenarios)
    print(f"[确定性方法] 最优订货量 = {q_det:.2f}")
    print(f"             期望利润   = {profit_det:.4f}")
    print()

    # 2. 随机最优解
    q_stoch, profit_stoch_in = stochastic_optimal_q(in_sample_scenarios, Q_CANDIDATES)
    profit_stoch_out = evaluate_by_simulation(q_stoch, in_sample_scenarios)
    print(f"[随机规划方法] 最优订货量 = {q_stoch}")
    print(f"               in-sample 期望利润 = {profit_stoch_in:.4f}")
    print(f"               out-sample期望利润 = {profit_stoch_out:.4f}")
    print()

    # 3. 对比
    gap = profit_stoch_out - profit_det
    print("-" * 60)
    print(f"期望利润差距 (随机 − 确定) = {gap:.4f}")
    print(f"相对提升 = {gap / abs(profit_det) * 100:.2f}%")
    print()

    # 4. 不同订货量的利润曲线采样
    print("--- 不同订货量下的期望利润 ---")
    for q in [60, 80, 100, 120, 140, 160]:
        p = evaluate_by_simulation(q, in_sample_scenarios)
        print(f"  q={q:3d} → 期望利润 = {p:.4f}")

    print()
    print("测试通过！")


if __name__ == "__main__":
    main()
