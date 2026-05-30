#!/usr/bin/env python3
"""
案例2：拍卖出价策略 ★★★☆☆

包含：
1. 第一价格密封拍卖仿真（N个竞拍者，私人估价）
2. 均衡出价策略验证：b(v) = v × (N-1)/N
3. 出价 shading 分析（N增加时出价趋近估价）
4. 赢者诅咒演示（共同价值拍卖）
5. 卖家期望收入验证（收入等价定理）
"""
# 教学注释：从参与者、策略和收益矩阵出发理解交互决策结构。
# 计算结果用于验证均衡、分配规则或机制设计是否符合预期。


import numpy as np
from typing import Callable, List, Tuple


# ============================================================================
# 1. 第一价格密封拍卖仿真
# ============================================================================

def first_price_auction(
    n_bidders: int,
    valuations: np.ndarray,
    bid_function: Callable[[float], float],
) -> Tuple[int, float, float, float]:
    """
    模拟一场第一价格密封拍卖。

    Args:
        n_bidders: 竞拍者数量
        valuations: 每个竞拍者的估价 (形状: n_bidders,)
        bid_function: 出价函数 b(v)

    Returns:
        (赢家索引, 赢家估价, 赢家出价, 赢家收益)
    """
    # 每个竞拍者根据估价出价
    bids = bid_function(valuations)

    # 找最高出价（若平局，随机选一个）
    max_bid = np.max(bids)
    winners = np.where(bids == max_bid)[0]
    winner = np.random.choice(winners)

    winner_valuation = valuations[winner]
    winner_bid = bids[winner]
    winner_payoff = winner_valuation - winner_bid

    return winner, winner_valuation, winner_bid, winner_payoff


def simulate_auctions(
    n_bidders: int,
    n_simulations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, List[dict]]:
    """
    运行多场拍卖仿真，统计结果。

    Args:
        n_bidders: 竞拍者数量
        n_simulations: 仿真场次
        seed: 随机种子

    Returns:
        (平均卖家收入, 理论期望收入, 部分场次详请)
    """
    rng = np.random.default_rng(seed)

    # 出价函数：b(v) = v × (N-1)/N
    shading_factor = (n_bidders - 1) / n_bidders

    def bid_func(v):
        return v * shading_factor

    total_revenue = 0.0
    sample_details = []

    for sim in range(n_simulations):
        # 生成估价：U[0, 1]
        valuations = rng.uniform(0, 1, size=n_bidders)

        winner, winner_v, winner_b, winner_p = first_price_auction(
            n_bidders, valuations, bid_func
        )

        total_revenue += winner_b

        # 记录前5场的详请 + 最后1场
        if sim < 5 or sim == n_simulations - 1:
            sample_details.append({
                "sim": sim,
                "valuations": valuations.copy(),
                "winner": winner,
                "winner_v": winner_v,
                "winner_b": winner_b,
                "winner_p": winner_p,
            })

    avg_revenue = total_revenue / n_simulations
    theoretical_revenue = (n_bidders - 1) / (n_bidders + 1)

    return avg_revenue, theoretical_revenue, sample_details


# ============================================================================
# 2. 显示仿真结果
# ============================================================================

def print_simulation_results(
    n_bidders: int,
    n_simulations: int,
    avg_revenue: float,
    theoretical_revenue: float,
    sample_details: List[dict],
):
    """格式化输出拍卖仿真结果"""
    shading_factor = (n_bidders - 1) / n_bidders

    print(f"\n--- 场景：N={n_bidders} 个竞拍者，估价 U[0,1] ---")
    print(f"理论出价函数：b(v) = v × (N-1)/N = v × {shading_factor:.3f}")

    for detail in sample_details[:5]:
        print(
            f"  模拟竞拍者{detail['winner']+1}: "
            f"估价={detail['winner_v']:.3f}, "
            f"出价={detail['winner_b']:.3f}, "
            f"收益={detail['winner_p']:.3f} "
            f"{'(未赢)' if detail['winner_p'] <= 0 else '(赢)'}"
        )

    last = sample_details[-1]
    print(
        f"  胜者: 竞拍者#{last['winner']+1}, "
        f"估价={last['winner_v']:.3f}, "
        f"出价={last['winner_b']:.3f}, "
        f"支付={last['winner_b']:.3f}, "
        f"实际收益={last['winner_p']:.3f}"
    )

    print(f"  卖家收入平均值（{n_simulations}场）: {avg_revenue:.3f}")
    print(f"  理论期望收入: (N-1)/(N+1) = {theoretical_revenue:.3f}")


# ============================================================================
# 3. 出价 Shading 分析
# ============================================================================

def shading_analysis():
    """分析不同人数下的出价比例"""
    print("\n" + "=" * 60)
    print("出价 shading 分析")
    print("=" * 60)
    print(f"{'竞拍者数量 N':<16} {'出价比例 (N-1)/N':<20} {'含义':<20}")
    print("-" * 56)

    for n in [2, 3, 5, 10, 100]:
        ratio = (n - 1) / n
        meaning = f"出价 = 估价 × {ratio*100:.0f}%"
        print(f"N={n:<14} {ratio:<20.1%} {meaning}")

    print("\n结论：竞拍者越多，出价越接近真实估价（竞争越激烈）")
    print("      N → ∞ 时 b(v) → v")


# ============================================================================
# 4. 赢者诅咒演示
# ============================================================================

def winner_curse_demo(seed: int = 42):
    """
    演示赢者诅咒现象。
    
    场景：共同价值拍卖
    - 物品的真实价值对所有竞拍者相同，但未知
    - 每个竞拍者得到有噪声的估测信号
    - 最高估者中标，但往往高估了价值
    """
    print("\n" + "=" * 60)
    print("赢者诅咒演示")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    # 真实价值
    true_value = 50.0
    n_bidders = 5

    print(f"共同价值拍卖（真实价值={true_value}，各竞拍者估测有噪声）：")

    # 每个竞拍者得到有噪声的估测
    # 噪声服从 N(0, σ²)，σ = 10
    noise_std = 10.0
    estimates = true_value + rng.normal(0, noise_std, size=n_bidders)

    # 竞拍者用自己的估测作为估价，按第一价格密封拍卖出价
    shading = (n_bidders - 1) / n_bidders
    bids = estimates * shading

    # 找出价最高的
    max_bid_idx = np.argmax(bids)
    max_bid = bids[max_bid_idx]
    max_est = estimates[max_bid_idx]

    for i in range(n_bidders):
        est = estimates[i]
        bid = bids[i]
        est_bias = est - true_value
        status = "中标" if i == max_bid_idx else "未赢"

        print(
            f"  竞拍者#{i+1}: "
            f"估测={est:.1f}, "
            f"偏差={est_bias:+.1f}, "
            f"出价={bid:.1f}, "
            f"状态={status}"
        )

    print(f"\n分析：")
    print(f"  真实价值 = {true_value}")
    print(f"  平均估测 = {estimates.mean():.1f}")
    print(f"  最高估测 = {max_est:.1f}（偏差 = {max_est - true_value:+.1f}）")
    print(f"  中标者出价 = {max_bid:.1f}")
    print(f"  如果支付 = 出价，实际利润 = {true_value - max_bid:.1f}")

    if max_est > true_value:
        print(f"\n  ⚠ 中标者是估测最高的那个——他高估了 {max_est - true_value:.1f}")
        print(f"  这就是赢者诅咒：赢了拍卖，但付出了过高的代价")

    # 理性调整
    print(f"\n  理性竞拍者应该考虑 '条件期望'：")
    print(f"  '如果我赢了，说明我的估测是所有竞拍者中最高的'")
    print(f"  → 需要进一步下调出价以补偿这个条件偏差")
    print(f"  调整后的出价 ≈ estimates × shading × 调整因子")


# ============================================================================
# 5. 收入等价定理验证
# ============================================================================

def revenue_equivalence_demo(n_simulations: int = 50000, seed: int = 42):
    """
    验证收入等价定理：第一价格和第二价格拍卖的卖家期望收入相同。
    """
    print("\n" + "=" * 60)
    print("收入等价定理验证")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    n_bidders = 3

    first_price_revenues = []
    second_price_revenues = []

    shading = (n_bidders - 1) / n_bidders

    for _ in range(n_simulations):
        valuations = rng.uniform(0, 1, size=n_bidders)

        # 第一价格拍卖
        bids_fp = valuations * shading
        winner_fp = np.argmax(bids_fp)
        revenue_fp = bids_fp[winner_fp]
        first_price_revenues.append(revenue_fp)

        # 第二价格拍卖（Vickrey）
        # 在第二价格拍卖中，按真实估价出价是占优策略
        bids_sp = valuations.copy()
        sorted_bids = np.sort(bids_sp)
        revenue_sp = sorted_bids[-2]  # 第二高价
        second_price_revenues.append(revenue_sp)

    avg_fp = np.mean(first_price_revenues)
    avg_sp = np.mean(second_price_revenues)
    theoretical = (n_bidders - 1) / (n_bidders + 1)

    print(f"竞拍者数量 N = {n_bidders}")
    print(f"仿真场次 = {n_simulations}")
    print(f"\n  第一价格拍卖平均收入: {avg_fp:.4f}")
    print(f"  第二价格拍卖平均收入: {avg_sp:.4f}")
    print(f"  理论期望收入 (N-1)/(N+1): {theoretical:.4f}")
    print(f"  差异: {abs(avg_fp - avg_sp):.4f}")

    if abs(avg_fp - avg_sp) < 0.01:
        print(f"\n  ✅ 收入等价定理验证通过！")
        print(f"     两种拍卖机制的期望收入基本相同")
    else:
        print(f"\n  ⚠ 有差异（可能因为仿真误差）")


# ============================================================================
# 6. 主程序
# ============================================================================

def main():
    print("=" * 60)
    print("博弈论案例2：拍卖出价策略")
    print("-" * 60)
    print("第一价格密封拍卖仿真与分析")
    print("=" * 60)

    # 场景 1: N=2
    avg_1, theo_1, details_1 = simulate_auctions(n_bidders=2, n_simulations=10000)
    print_simulation_results(2, 10000, avg_1, theo_1, details_1)

    # 场景 2: N=5
    avg_2, theo_2, details_2 = simulate_auctions(n_bidders=5, n_simulations=10000)
    print_simulation_results(5, 10000, avg_2, theo_2, details_2)

    # 场景 3: N=10
    avg_3, theo_3, details_3 = simulate_auctions(n_bidders=10, n_simulations=10000)
    print_simulation_results(10, 10000, avg_3, theo_3, details_3)

    # 出价 shading 分析
    shading_analysis()

    # 赢者诅咒演示
    winner_curse_demo(seed=42)

    # 收入等价定理验证
    revenue_equivalence_demo(n_simulations=50000, seed=42)


if __name__ == "__main__":
    main()
