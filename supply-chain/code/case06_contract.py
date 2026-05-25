"""
case06_contract.py — 供应链契约模型：回购契约
=================================================
演示内容：
1. 报童模型下的分散决策（零售商独立决策）
2. 集中决策（供应链整体最优）
3. 回购契约：供应商以回购价回收未售商品
4. 验证：适当回购价格可使分散决策利润 = 集中决策

仅使用 Python 标准库（math, random）
"""

import math
import random


# ============================================================
# 1. 正态分布函数（复用 case04 逻辑）
# ============================================================

def _phi(x: float) -> float:
    """标准正态分布 PDF"""
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
    """标准正态分布 CDF（近似）"""
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421405429, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    t = 1.0 / (1.0 + p * abs(x))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2.0)
    return 0.5 * (1.0 + sign * y)


def _Phi_inv(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """正态分布分位函数"""
    if sigma <= 0:
        return mu
    if p <= 0.0:
        return mu - 8 * sigma
    if p >= 1.0:
        return mu + 8 * sigma
    x = mu + sigma * (p - 0.5) * 2.5066
    for _ in range(50):
        cdf_val = _Phi((x - mu) / sigma)
        diff = cdf_val - p
        if abs(diff) < 1e-12:
            break
        pdf_val = _phi((x - mu) / sigma) / sigma
        if pdf_val < 1e-300:
            break
        x -= diff / pdf_val
    return x


def _loss_std(z: float) -> float:
    """标准正态损失函数 L(z) = phi(z) - z*(1-Phi(z))"""
    return _phi(z) - z * (1.0 - _Phi(z))


# ============================================================
# 2. 报童模型基础
# ============================================================

def expected_sales(Q: float, mu: float, sigma: float) -> float:
    """给定订货量 Q，预期销售量"""
    if sigma <= 0:
        return min(Q, mu)
    z = (Q - mu) / sigma
    return mu - sigma * _loss_std(z)


def expected_leftover(Q: float, mu: float, sigma: float) -> float:
    """给定订货量 Q，预期剩余量"""
    return Q - expected_sales(Q, mu, sigma)


# ============================================================
# 3. 分散决策 vs 集中决策 vs 回购契约
# ============================================================

def decentralized_decision(p: float, c: float, s: float,
                           mu: float, sigma: float) -> dict:
    """
    分散决策：零售商独立决定订货量

    参数:
        p — 零售价格
        c — 批发价格（零售商进货成本）
        s — 残值
        mu — 需求均值
        sigma — 需求标准差

    返回:
        包含最优订货量、利润等信息的字典
    """
    # 零售商的缺货成本 = 售价 - 进价
    # 零售商的过期成本 = 进价 - 残值
    cu_r = p - c       # 少订一件损失
    co_r = c - s       # 多订一件损失
    cr_r = cu_r / (cu_r + co_r) if (cu_r + co_r) > 0 else 0.0

    Q_r = _Phi_inv(cr_r, mu, sigma)
    sales = expected_sales(Q_r, mu, sigma)
    leftover = expected_leftover(Q_r, mu, sigma)

    # 零售商利润 = p * 销售 + s * 剩余 - c * 订货量
    profit_r = p * sales + s * leftover - c * Q_r

    # 供应商利润 = (c - c_mfg) * Q_r   (假设供应商生产成本为 0，简化)
    # 这里让供应商利润 = c * Q_r - 0 (假设生产成本为 0，即供应商收入全部为利润)
    # 为公平比较，供应商生产成本设为 c_mfg
    c_mfg = 0.0  # 简化为 0
    profit_s = (c - c_mfg) * Q_r

    total_profit = profit_r + profit_s

    return {
        "Q": Q_r,
        "critical_ratio": cr_r,
        "service_level": _Phi((Q_r - mu) / sigma) if sigma > 0 else 0.0,
        "sales": sales,
        "leftover": leftover,
        "profit_retailer": profit_r,
        "profit_supplier": profit_s,
        "profit_total": total_profit,
    }


def centralized_decision(p: float, c_mfg: float, s: float,
                         mu: float, sigma: float) -> dict:
    """
    集中决策：供应链整体最优（垂直整合）

    参数:
        p — 零售价格
        c_mfg — 生产成本
        s — 残值
        mu — 需求均值
        sigma — 需求标准差

    返回:
        包含最优订货量、总利润等信息的字典
    """
    # 缺货成本 = 售价 - 生产成本
    # 过期成本 = 生产成本 - 残值
    cu_c = p - c_mfg
    co_c = c_mfg - s
    cr_c = cu_c / (cu_c + co_c) if (cu_c + co_c) > 0 else 0.0

    Q_c = _Phi_inv(cr_c, mu, sigma)
    sales = expected_sales(Q_c, mu, sigma)
    leftover = expected_leftover(Q_c, mu, sigma)

    # 供应链总利润 = p * 销售 + s * 剩余 - c_mfg * 订货量
    profit = p * sales + s * leftover - c_mfg * Q_c

    return {
        "Q": Q_c,
        "critical_ratio": cr_c,
        "service_level": _Phi((Q_c - mu) / sigma) if sigma > 0 else 0.0,
        "sales": sales,
        "leftover": leftover,
        "profit_total": profit,
    }


def buyback_contract(p: float, c: float, b: float, c_mfg: float,
                     s: float, mu: float, sigma: float) -> dict:
    """
    回购契约模型

    参数:
        p — 零售价格
        c — 批发价格
        b — 回购价格（供应商承诺以 b 回购未售商品）
        c_mfg — 供应商生产成本
        s — 残值（供应商处理剩余的外部残值）
        mu — 需求均值
        sigma — 需求标准差

    在回购契约下：
        - 零售商的过期成本 = c - b（因为未售商品可以 b 退回）
        - 零售商的缺货成本 = p - c（不变）
        - 供应商的净收入 = c - c_mfg - (b - s)*剩余
    """
    # ---- 零售商决策 ----
    cu_r = p - c
    co_r = c - b          # 关键：回购降低了零售商过期成本
    cr_r = cu_r / (cu_r + co_r) if (cu_r + co_r) > 0 else 0.0

    Q_r = _Phi_inv(cr_r, mu, sigma)
    sales = expected_sales(Q_r, mu, sigma)
    leftover = expected_leftover(Q_r, mu, sigma)

    # 零售商利润
    profit_r = p * sales + b * leftover - c * Q_r

    # 供应商利润
    # 供应商收入 = c * Q_r（批发收入）
    # 供应商成本 = c_mfg * Q_r（生产成本） + (b - s) * leftover（回购净成本）
    profit_s = c * Q_r - c_mfg * Q_r - (b - s) * leftover

    total_profit = profit_r + profit_s

    return {
        "Q": Q_r,
        "critical_ratio": cr_r,
        "service_level": _Phi((Q_r - mu) / sigma) if sigma > 0 else 0.0,
        "sales": sales,
        "leftover": leftover,
        "profit_retailer": profit_r,
        "profit_supplier": profit_s,
        "profit_total": total_profit,
        "coordinated": abs(Q_r - centralized_decision(p, c_mfg, s, mu, sigma)["Q"]) < 0.5,
    }


# ============================================================
# 4. 文本曲线：回购价格对利润的影响
# ============================================================

def buyback_impact_curve(p: float, c: float, c_mfg: float, s: float,
                         mu: float, sigma: float, num_points: int = 15) -> str:
    """
    生成回购价格对各项利润的影响曲线（文本）
    """
    b_min = s                     # 最低回购价 = 残值
    b_max = c                     # 最高回购价 = 批发价
    step = (b_max - b_min) / num_points if num_points > 0 else 1

    # 集中决策利润（基准）
    central = centralized_decision(p, c_mfg, s, mu, sigma)
    central_profit = central["profit_total"]

    lines = []
    lines.append("=" * 80)
    lines.append("  回购价格对供应链利润的影响分析")
    lines.append(f"  参数: p={p}, c={c}, c_mfg={c_mfg}, s={s}, μ={mu}, σ={sigma}")
    lines.append(f"  集中决策总利润（基准）= {central_profit:.2f}")
    lines.append("=" * 80)
    lines.append(f"{'回购价':>8} | {'订货量':>10} | {'零售商利润':>12} | "
                 f"{'供应商利润':>12} | {'总利润':>12} | {'协调?':>6}")
    lines.append("-" * 80)

    for i in range(num_points + 1):
        b_val = b_min + i * step
        result = buyback_contract(p, c, b_val, c_mfg, s, mu, sigma)
        coord = "✅" if result["coordinated"] else "❌"
        lines.append(f"{b_val:>8.2f} | {result['Q']:>10.2f} | "
                     f"{result['profit_retailer']:>12.2f} | "
                     f"{result['profit_supplier']:>12.2f} | "
                     f"{result['profit_total']:>12.2f} | {coord:>6}")

    lines.append("=" * 80)
    return "\n".join(lines)


# ============================================================
# 5. 寻找协调回购价格
# ============================================================

def find_coordinating_buyback(p: float, c: float, c_mfg: float, s: float,
                              mu: float, sigma: float, tolerance: float = 0.5) -> float:
    """
    通过二分法寻找使分散决策等价于集中决策的回购价格 b

    协调条件：Q_buyback(b) == Q_centralized
    """
    central = centralized_decision(p, c_mfg, s, mu, sigma)
    Q_target = central["Q"]

    lo, hi = s, c
    for _ in range(100):
        mid = (lo + hi) / 2.0
        result = buyback_contract(p, c, mid, c_mfg, s, mu, sigma)
        Q_mid = result["Q"]
        if abs(Q_mid - Q_target) < tolerance:
            return mid
        if Q_mid < Q_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ============================================================
# 6. 主程序
# ============================================================

def main():
    """主函数：演示回购契约对供应链协调的作用"""
    print("\n" + "★" * 45)
    print("  供应链契约模型：回购契约演示")
    print("★" * 45)

    # ---- 参数设定 ----
    p = 50.0       # 零售价
    c = 30.0       # 批发价
    c_mfg = 10.0   # 供应商生产成本
    s = 5.0        # 残值
    mu = 1000.0    # 需求均值
    sigma = 200.0  # 需求标准差

    print(f"\n▶ 基础参数")
    print(f"   零售价(p)     = {p}")
    print(f"   批发价(c)     = {c}")
    print(f"   生产成本(c_mfg) = {c_mfg}")
    print(f"   残值(s)       = {s}")
    print(f"   需求: N(μ={mu}, σ={sigma})")

    # ---- (1) 分散决策 ----
    print("\n▶ 1. 分散决策（零售商独立决策）")
    print("-" * 60)
    dec = decentralized_decision(p, c, s, mu, sigma)
    print(f"   零售商临界比 = {dec['critical_ratio']:.4f}")
    print(f"   零售商订货量 = {dec['Q']:.2f}")
    print(f"   服务水平    = {dec['service_level']:.2%}")
    print(f"   预期销售    = {dec['sales']:.2f}")
    print(f"   预期剩余    = {dec['leftover']:.2f}")
    print(f"   零售商利润  = {dec['profit_retailer']:.2f}")
    print(f"   供应商利润  = {dec['profit_supplier']:.2f}")
    print(f"   总利润      = {dec['profit_total']:.2f}")

    # ---- (2) 集中决策 ----
    print("\n▶ 2. 集中决策（供应链整体最优）")
    print("-" * 60)
    cen = centralized_decision(p, c_mfg, s, mu, sigma)
    print(f"   集中临界比  = {cen['critical_ratio']:.4f}")
    print(f"   集中订货量  = {cen['Q']:.2f}")
    print(f"   服务水平    = {cen['service_level']:.2%}")
    print(f"   预期销售    = {cen['sales']:.2f}")
    print(f"   预期剩余    = {cen['leftover']:.2f}")
    print(f"   供应链总利润 = {cen['profit_total']:.2f}")

    # 对比
    profit_gap = cen["profit_total"] - dec["profit_total"]
    print(f"\n   分散 vs 集中:")
    print(f"   订货量差异: {dec['Q'] - cen['Q']:+.2f} (分散 {'偏低' if dec['Q'] < cen['Q'] else '偏高'})")
    print(f"   利润差距:   {profit_gap:+.2f} (集中更高)" if profit_gap > 0.01 else f"   利润差距:   {profit_gap:+.2f} (基本一致)")

    # ---- (3) 回购契约影响 ----
    print("\n▶ 3. 回购价格对供应链的影响")
    print("-" * 60)
    print(buyback_impact_curve(p, c, c_mfg, s, mu, sigma))

    # ---- (4) 寻找协调回购价格 ----
    print("\n▶ 4. 寻找协调回购价格")
    print("-" * 60)
    b_coord = find_coordinating_buyback(p, c, c_mfg, s, mu, sigma)
    coord_result = buyback_contract(p, c, b_coord, c_mfg, s, mu, sigma)
    print(f"   协调回购价格 b* = {b_coord:.4f}")
    print(f"   协调后订货量 Q  = {coord_result['Q']:.2f} (目标={cen['Q']:.2f})")
    print(f"   零售商利润      = {coord_result['profit_retailer']:.2f}")
    print(f"   供应商利润      = {coord_result['profit_supplier']:.2f}")
    print(f"   总利润          = {coord_result['profit_total']:.2f} "
          f"(目标={cen['profit_total']:.2f})")
    print(f"   供应链协调?     {'✅ 是' if coord_result['coordinated'] else '❌ 否'}")

    # ---- (5) 验证：适当回购价可使分散=集中 ----
    print("\n▶ 5. 验证：适当回购价使分散决策利润 = 集中决策总利润")
    print("-" * 60)
    diff = abs(coord_result["profit_total"] - cen["profit_total"])
    if diff < 1.0:
        print(f"   ✅ 验证通过！协调回购价 b={b_coord:.2f} 下，")
        print(f"      分散决策总利润 ({coord_result['profit_total']:.2f})")
        print(f"      ≈ 集中决策总利润 ({cen['profit_total']:.2f})")
        print(f"      差距 = {diff:.6f}")
    else:
        print(f"   ⚠️  差距较大: {diff:.2f}（可能需要更精确搜索）")

    # ---- (6) 不同批发价下的协调回购价 ----
    print("\n▶ 6. 不同批发价下的协调回购价")
    print("-" * 70)
    print(f"{'批发价(c)':>10} {'协调回购价(b*)':>16} {'零售商利润':>12} {'供应商利润':>12} {'总利润':>12}")
    print("-" * 70)
    for c_test in [20, 25, 30, 35, 40]:
        b_test = find_coordinating_buyback(p, c_test, c_mfg, s, mu, sigma)
        r = buyback_contract(p, c_test, b_test, c_mfg, s, mu, sigma)
        print(f"{c_test:>10.2f} {b_test:>16.4f} {r['profit_retailer']:>12.2f} "
              f"{r['profit_supplier']:>12.2f} {r['profit_total']:>12.2f}")

    print("\n" + "★" * 45)
    print("  回购契约演示完毕")
    print("★" * 45 + "\n")


if __name__ == "__main__":
    main()
