"""
case04_newsvendor.py — 报童模型与服务水平的决策
=================================================
演示内容：
1. 给定正态分布需求，最优订货量 = 分位数计算
2. 服务水平（CSL）vs 订货量曲线
3. 缺货率与服务水平互补关系验证
4. 缺货成本与过期成本对决策的影响

仅使用 Python 标准库（math, random）
"""


# 教学注释：围绕订货、库存、契约和网络配置观察供应链决策变量。
# 重点比较成本、缺货风险与服务水平之间的权衡。



import math
import random
import sys
from pathlib import Path


# ============================================================
# 1. 正态分布工具函数（纯标准库实现）
# ============================================================

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.stats_utils import normal_cdf, normal_pdf, normal_ppf


# ============================================================
# 2. 报童模型核心函数
# ============================================================

def newsvendor_optimal_q(cu: float, co: float, mu: float, sigma: float) -> float:
    """
    报童模型最优订货量

    参数:
        cu — 缺货成本（underage cost，少订一个的损失）
        co — 过期成本（overage cost，多订一个的损失）
        mu — 需求均值
        sigma — 需求标准差

    最优条件 F(Q*) = cu / (cu + co)
    即临界分位率（Critical Ratio）

    返回:
        最优订货量 Q*
    """
    critical_ratio = cu / (cu + co) if (cu + co) > 0 else 0.5
    return normal_ppf(critical_ratio, mu, sigma)


def expected_profit(Q: float, cu: float, co: float, mu: float,
                    sigma: float, unit_revenue: float,
                    unit_cost: float) -> dict:
    """
    计算给定订货量 Q 的预期利润及相关指标

    返回字典:
        - profit: 预期利润
        - expected_sales: 预期销售量
        - expected_leftover: 预期剩余量
        - stockout_rate: 缺货率（P(需求 > Q)）
        - service_level: 服务水平（P(需求 <= Q)）
    """
    service_level = normal_cdf(Q, mu, sigma)
    stockout_rate = 1.0 - service_level

    # 预期销售 = 积分(0~Q) x*f(x)dx + Q*P(X>Q)
    # 使用近似：expected_sales = mu - sigma * L(z), z=(Q-mu)/sigma
    z = (Q - mu) / sigma if sigma > 0 else 0.0
    # 标准正态损失函数 L(z) = phi(z) - z*(1-Phi(z))
    phi_z = normal_pdf(z, 0, 1)
    Phi_z = normal_cdf(z, 0, 1)
    loss_z = phi_z - z * (1.0 - Phi_z)
    expected_sales = mu - sigma * loss_z
    expected_leftover = Q - expected_sales

    # 利润 = 收入 - 成本 - 缺货损失 + 残值
    revenue = unit_revenue * expected_sales
    cost = unit_cost * Q
    # 过期损失 = co * expected_leftover
    # 缺货损失 = cu * (mu - expected_sales)  [近似]
    profit = revenue - cost

    return {
        "profit": profit,
        "expected_sales": expected_sales,
        "expected_leftover": expected_leftover,
        "stockout_rate": stockout_rate,
        "service_level": service_level,
        "z": z,
        "critical_ratio": cu / (cu + co) if (cu + co) > 0 else 0.5,
    }


# ============================================================
# 3. 服务水平 vs 订货量曲线（文本）
# ============================================================

def service_level_curve(mu: float, sigma: float, cu: float, co: float,
                        num_points: int = 30) -> str:
    """
    生成服务水平 vs 订货量的文本曲线

    显示：
    - 各订货量下的服务水平
    - 最优订货量标注
    """
    q_opt = newsvendor_optimal_q(cu, co, mu, sigma)
    q_min = max(0, int(mu - 4 * sigma))
    q_max = int(mu + 4 * sigma)
    step = max(1, (q_max - q_min) // num_points)

    lines = []
    lines.append("=" * 70)
    lines.append(f"服务水平 vs 订货量曲线")
    lines.append(f"需求: N(μ={mu}, σ={sigma})")
    lines.append(f"缺货成本 cu={cu}, 过期成本 co={co}")
    lines.append(f"最优订货量 Q* = {q_opt:.2f}  临界比 = {cu/(cu+co):.4f}")
    lines.append("=" * 70)
    lines.append(f"{'订货量':>8} | {'服务水平':>8} | {'缺货率':>8} | {'预期利润':>12} | 图")
    lines.append("-" * 70)

    # 找到利润范围用于条形
    profits = []
    qs = []
    for q in range(q_min, q_max + 1, step):
        r = expected_profit(q, cu, co, mu, sigma, unit_revenue=cu + co, unit_cost=co)
        profits.append(r["profit"])
        qs.append(q)

    min_p = min(profits)
    max_p = max(profits)
    span_p = max_p - min_p if max_p != min_p else 1.0

    for q, p in zip(qs, profits):
        r = expected_profit(q, cu, co, mu, sigma, unit_revenue=cu + co, unit_cost=co)
        sl = r["service_level"]
        sr = r["stockout_rate"]
        bar_len = int(((p - min_p) / span_p) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        marker = " ◀ Q*" if abs(q - q_opt) < step * 0.9 else ""
        lines.append(f"{q:>8} | {sl:>7.2%} | {sr:>7.2%} | {p:>12.2f} | {bar}{marker}")

    lines.append("=" * 70)
    return "\n".join(lines)


# ============================================================
# 4. 主程序：自测与演示
# ============================================================

def main():
    """主函数：演示报童模型的各项功能"""
    print("\n" + "★" * 40)
    print("  报童模型与服务水平决策演示")
    print("★" * 40)

    # ---- 参数设定 ----
    mu = 1000           # 需求均值 1000 件
    sigma = 200         # 需求标准差 200
    unit_cost = 30      # 单位采购成本 30 元
    unit_price = 50     # 单位售价 50 元
    salvage_value = 10  # 残值 10 元

    # 缺货成本 = 售价 - 成本（少订一件损失毛利）
    cu = unit_price - unit_cost   # = 20 (少卖一件损失)
    # 过期成本 = 成本 - 残值（多订一件损失）
    co = unit_cost - salvage_value  # = 20 (多订一件浪费)

    print(f"\n▶ 基础参数")
    print(f"   需求分布: 正态分布 μ={mu}, σ={sigma}")
    print(f"   单位售价: {unit_price}元")
    print(f"   单位成本: {unit_cost}元")
    print(f"   残值:     {salvage_value}元")
    print(f"   缺货成本(cu) = {cu}")
    print(f"   过期成本(co) = {co}")

    # ---- (1) 最优订货量计算 ----
    print("\n▶ 1. 报童最优订货量计算")
    print("-" * 50)
    q_opt = newsvendor_optimal_q(cu, co, mu, sigma)
    critical_ratio = cu / (cu + co)
    print(f"   临界分位率 = cu/(cu+co) = {cu}/{cu+co} = {critical_ratio:.4f}")
    print(f"   最优订货量 Q* = {q_opt:.2f} 件")

    # ---- (2) 验证：服务水平与缺货率互补 ----
    print("\n▶ 2. 服务水平与缺货率验证")
    print("-" * 50)
    result = expected_profit(q_opt, cu, co, mu, sigma, unit_price, unit_cost)
    print(f"   最优 Q* = {q_opt:.2f}")
    print(f"   服务水平 F(Q*) = {result['service_level']:.6f} ({result['service_level']*100:.2f}%)")
    print(f"   缺货率 1-F(Q*) = {result['stockout_rate']:.6f} ({result['stockout_rate']*100:.2f}%)")
    print(f"   互补验证: SL + SR = {result['service_level'] + result['stockout_rate']:.10f} (应为 1.0)")
    assert abs(result['service_level'] + result['stockout_rate'] - 1.0) < 1e-9, \
        "❌ 服务水平与缺货率之和不等于 1！"
    print(f"   ✅ 验证通过！")
    print(f"   预期利润 = {result['profit']:.2f} 元")
    print(f"   预期销售 = {result['expected_sales']:.2f} 件")
    print(f"   预期剩余 = {result['expected_leftover']:.2f} 件")

    # ---- (3) 服务水平 vs 订货量曲线 ----
    print("\n▶ 3. 服务水平 vs 订货量曲线")
    print(service_level_curve(mu, sigma, cu, co))

    # ---- (4) 不同服务水平下的决策对比 ----
    print("\n▶ 4. 不同目标服务水平下的订货量")
    print("-" * 50)
    for target_sl in [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]:
        q_target = normal_ppf(target_sl, mu, sigma)
        r = expected_profit(q_target, cu, co, mu, sigma, unit_price, unit_cost)
        print(f"   SL={target_sl:.0%} → Q={q_target:>8.2f}  预期利润={r['profit']:>10.2f}  预期剩余={r['expected_leftover']:>8.2f}")

    # ---- (5) 成本比例变化对最优量的影响 ----
    print("\n▶ 5. 成本比例(cu/co)变化对最优订货量的影响")
    print("-" * 60)
    print(f"{'cu':>6} {'co':>6} {'cu/co':>8} {'临界比':>8} {'Q*':>10} {'服务水平':>10}")
    print("-" * 60)
    for ratio in [0.25, 0.5, 1.0, 2.0, 4.0]:
        cu2 = 20 * ratio
        co2 = 20
        q2 = newsvendor_optimal_q(cu2, co2, mu, sigma)
        cr2 = cu2 / (cu2 + co2)
        sl2 = normal_cdf(q2, mu, sigma)
        print(f"{cu2:>6.1f} {co2:>6.1f} {cu2/co2:>8.2f} {cr2:>8.4f} {q2:>10.2f} {sl2:>9.2%}")

    print("\n" + "★" * 40)
    print("  报童模型演示完毕")
    print("★" * 40 + "\n")


if __name__ == "__main__":
    main()
