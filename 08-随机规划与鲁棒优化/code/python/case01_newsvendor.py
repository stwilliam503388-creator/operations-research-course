"""
案例1：新闻摊贩的随机版
难度：★★☆☆☆
方法：随机规划 / 单阶段
依赖：pip install numpy scipy
运行：python code/python/case01_newsvendor.py

简述：求解新闻摊贩问题的最优订货量——对比解析解与蒙特卡洛仿真解。
"""
# 教学注释：通过场景、概率权重和坏情况边界刻画不确定需求或收益。
# 对比不同假设下的解，理解随机性和鲁棒性如何改变决策。


import numpy as np
from scipy import stats


# ============================================================
# 数据准备
# ============================================================
c = 0.5    # 进货价 (元/份)
p = 1.0    # 零售价 (元/份)
v = 0.1    # 退货价 (元/份)
mu = 100   # 需求均值
sigma = 20 # 需求标准差


# ============================================================
# 利润函数
# ============================================================
def profit(q, d):
    """
    给定进货量 q 和实际需求 d，计算实际利润。
    """
    sold = min(d, q)          # 卖出的份数
    leftover = max(q - d, 0)  # 没卖完的
    return sold * p + leftover * v - q * c


# ============================================================
# 方法1：解析解
# ============================================================
def analytical_solution():
    """
    用临界分位数公式 q* = F^{-1}((p-c)/(p-v))
    """
    critical_ratio = (p - c) / (p - v)
    # 正态分布的分位数
    q_star = stats.norm.ppf(critical_ratio, loc=mu, scale=sigma)
    return q_star, critical_ratio


# ============================================================
# 方法2：蒙特卡洛仿真搜索
# ============================================================
def simulate_expected_profit(q, n_scenarios=10000, seed=42):
    """
    对给定 q，用蒙特卡洛仿真估计期望利润。
    """
    rng = np.random.default_rng(seed)
    demands = rng.normal(loc=mu, scale=sigma, size=n_scenarios)
    profits = np.array([profit(q, d) for d in demands])
    return profits.mean()


def search_best_q(q_range=(50, 150), step=1, n_scenarios=10000, seed=42):
    """
    网格搜索最优 q。
    """
    candidates = np.arange(q_range[0], q_range[1] + 1, step)
    best_q = None
    best_profit = -np.inf
    results = []
    for q in candidates:
        avg_profit = simulate_expected_profit(q, n_scenarios, seed)
        results.append((q, avg_profit))
        if avg_profit > best_profit:
            best_profit = avg_profit
            best_q = q
    return best_q, best_profit, candidates, [r[1] for r in results]


# ============================================================
# 灵敏度分析：σ 变化的影响
# ============================================================
def sensitivity_sigma(sigmas, n_scenarios=10000, seed=42):
    """
    在不同标准差下计算最优 q 和期望利润。
    """
    results = []
    for s in sigmas:
        # 重新定义全局 sigma 的效果
        old_sigma = globals()['sigma']
        globals()['sigma'] = s
        # 解析解
        q_star, _ = analytical_solution()
        # 仿真验证
        best_q_sim, best_profit_sim, _, _ = search_best_q(
            q_range=(50, 150), step=1, n_scenarios=n_scenarios, seed=seed
        )
        results.append((s, q_star, best_q_sim, best_profit_sim))
        globals()['sigma'] = old_sigma
    return results


# ============================================================
# 结果输出
# ============================================================
def print_summary():
    print("=" * 60)
    print("案例1：新闻摊贩的随机版")
    print("=" * 60)

    # 1. 解析解
    q_star, critical_ratio = analytical_solution()
    print(f"\n▶ 数据: c={c}, p={p}, v={v}, μ={mu}, σ={sigma}")
    print(f"▶ 临界比率 = (p-c)/(p-v) = {critical_ratio:.4f}")
    print(f"▶ 解析最优 q* = {q_star:.2f}")

    # 2. 仿真解
    best_q_sim, best_profit_sim, qs, avg_profits = search_best_q(
        q_range=(50, 150), step=1, n_scenarios=10000, seed=42
    )
    print(f"\n▶ 仿真最优 q* = {best_q_sim}, 期望利润 = {best_profit_sim:.2f} 元")

    # 3. 验证一致性
    diff_pct = abs(best_q_sim - q_star) / q_star * 100
    print(f"\n▶ 解析与仿真偏差: {diff_pct:.2f}%")
    if diff_pct < 1:
        print("✅ 验证通过：解析解与仿真解一致 (< 1%)")
    else:
        print("⚠️ 验证未完全通过：偏差较大")

    # 4. 凹函数验证
    # 检查利润曲线是否单峰
    max_idx = np.argmax(avg_profits)
    is_concave = True
    for i in range(1, max_idx):
        if avg_profits[i] < avg_profits[i - 1]:
            is_concave = False
            break
    for i in range(max_idx + 1, len(avg_profits) - 1):
        if avg_profits[i] < avg_profits[i + 1]:
            is_concave = False
            break
    print(f"▶ 利润曲线单峰? {'✅ 是' if is_concave else '⚠️ 否'}")


def solve_small():
    """小规模测试：50 场景"""
    print("\n=== 小规模测试: 50 场景 ===")
    q_star, cr = analytical_solution()
    print(f"解析 q*={q_star:.2f}")
    best_q, best_p, _, _ = search_best_q(
        q_range=(80, 120), step=1, n_scenarios=50, seed=42
    )
    print(f"仿真 (50场景) q*={best_q}, 期望利润={best_p:.2f}")


def solve_medium():
    """中等规模测试：1000 场景 + 灵敏度分析"""
    print("\n=== 中等规模测试: 1000 场景 ===")
    best_q, best_p, _, _ = search_best_q(
        q_range=(80, 120), step=1, n_scenarios=1000, seed=42
    )
    print(f"仿真 (1000场景) q*={best_q}, 期望利润={best_p:.2f}")

    print("\n▶ 灵敏度分析: σ 变化")
    sigmas = [10, 20, 30, 50]
    res = sensitivity_sigma(sigmas, n_scenarios=5000, seed=42)
    print(f"{'σ':>6} | {'解析 q*':>8} | {'仿真 q*':>8} | {'期望利润':>10}")
    print("-" * 40)
    for s, q_ana, q_sim, profit_sim in res:
        print(f"{s:>6} | {q_ana:>8.2f} | {q_sim:>8} | {profit_sim:>10.2f}")


if __name__ == "__main__":
    solve_small()
    solve_medium()
    print("\n")
    print_summary()
