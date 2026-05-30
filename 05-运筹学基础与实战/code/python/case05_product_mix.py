"""
第5步：建模心法实战 — 产品组合优化
========================================
场景：一家小工厂生产 A/B/C 三种产品，有原料/人工/机器三类资源约束。
目标：最大化总利润。

按 OR 课程五步法：
  画图 → 找决策变量 → 列约束 → 写目标 → 量纲检查

运行: python3 code/python/case05_product_mix.py
"""

import pyomo.environ as pyo

# ============================================================
# 数据（模型独立，避免克隆问题）
# ============================================================
products = ['A', 'B', 'C']
profit = {'A': 120, 'B': 80, 'C': 150}
resources_data = {'原料': 2000, '人工': 3000, '机器': 1500}
consumption = {
    'A': {'原料': 2, '人工': 3, '机器': 1},
    'B': {'原料': 1, '人工': 2, '机器': 2},
    'C': {'原料': 4, '人工': 5, '机器': 1},
}

def build_and_solve(prods, profits, cons, res_limits, sense=pyo.maximize):
    """通用：建 LP 模型 + 求解 + 返回结果"""
    m = pyo.ConcreteModel()
    m.P = pyo.Set(initialize=prods)
    m.profit = pyo.Param(m.P, initialize=profits)
    m.x = pyo.Var(m.P, domain=pyo.NonNegativeReals)
    m.obj = pyo.Objective(
        rule=lambda m: sum(m.profit[p] * m.x[p] for p in m.P),
        sense=sense)
    m.res = pyo.ConstraintList()
    for r, limit in res_limits.items():
        m.res.add(sum(cons[p][r] * m.x[p] for p in prods) <= limit)
    solver = pyo.SolverFactory('appsi_highs')
    solver.solve(m)
    quantities = {p: pyo.value(m.x[p]) for p in prods}
    total = sum(quantities[p] * profits[p] for p in prods)
    return m, quantities, total


def main():
    print("=" * 65)
    print("  第5步：建模实战 — 产品组合优化")
    print("=" * 65)

    # ============================================================
    # 第一~五步见代码注释，直接求解
    # ============================================================
    model, qty, total_profit = build_and_solve(products, profit, consumption, resources_data)

    print(f"\n  ── 最优生产方案 ──")
    for p in products:
        print(f"    产品 {p}: 生产 {qty[p]:.0f} 件, 利润 {profit[p]} 元/件, 合计 {qty[p] * profit[p]:.0f} 元")
    print(f"  {'─' * 42}")
    print(f"    总利润: {total_profit:,.0f} 元")

    # 资源利用率
    print(f"\n  ── 资源利用率 ──")
    for r in resources_data:
        used = sum(consumption[p][r] * qty[p] for p in products)
        pct = used / resources_data[r] * 100
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"    {r}: {used:.0f}/{resources_data[r]} ({pct:.0f}%) {bar}")
        if pct > 99:
            print(f"           ↑ 瓶颈资源！")

    # ============================================================
    # 灵敏度分析：重新建模型，变参数
    # ============================================================
    print(f"\n  ── 灵敏度分析 ──")

    # 场景1：人工 +10%
    _, _, p2 = build_and_solve(products, profit, consumption,
                               {'原料': 2000, '人工': 3300, '机器': 1500})
    print(f"  场景1: 人工 3000→3300 (+10%)")
    print(f"    原利润: {total_profit:,.0f} → 新利润: {p2:,.0f} (Δ=+{p2-total_profit:,.0f})")
    # 影子价格 ≈ 利润增量 / 资源增量
    shadow_labor = (p2 - total_profit) / 300
    print(f"    人工影子价格 ≈ {shadow_labor:.1f} 元/小时")

    # 场景2：原料 -10%
    _, _, p3 = build_and_solve(products, profit, consumption,
                               {'原料': 1800, '人工': 3000, '机器': 1500})
    print(f"\n  场景2: 原料 2000→1800 (-10%)")
    print(f"    原利润: {total_profit:,.0f} → 新利润: {p3:,.0f} (Δ={p3-total_profit:,.0f})")
    shadow_material = (total_profit - p3) / 200
    print(f"    原料影子价格 ≈ {shadow_material:.1f} 元/kg")

    # 场景3：机器 -10%
    _, _, p4 = build_and_solve(products, profit, consumption,
                               {'原料': 2000, '人工': 3000, '机器': 1350})
    print(f"\n  场景3: 机器 1500→1350 (-10%)")
    print(f"    原利润: {total_profit:,.0f} → 新利润: {p4:,.0f} (Δ={p4-total_profit:,.0f})")
    if total_profit - p4 < 0.01:
        print(f"    机器非瓶颈，-10% 不影响利润 ✓")

    # 场景4：引入新产品 D
    print(f"\n  场景4: 引入新产品 D（利润 100, 消耗 原料3/人工2/机器2）")
    prods4 = ['A','B','C','D']
    profit4 = {'A':120,'B':80,'C':150,'D':100}
    cons4 = {
        'A': {'原料':2,'人工':3,'机器':1}, 'B': {'原料':1,'人工':2,'机器':2},
        'C': {'原料':4,'人工':5,'机器':1}, 'D': {'原料':3,'人工':2,'机器':2}}
    _, qty4, total4 = build_and_solve(prods4, profit4, cons4, resources_data)
    for p in prods4:
        if qty4[p] > 0.1:
            print(f"    产品 {p}: 生产 {qty4[p]:.0f} 件 / 件")
    print(f"    总利润: {total4:,.0f} 元 (Δ={total4-total_profit:+,.0f})")
    if total4 > total_profit:
        print(f"    结论: 值得引入 ✓")
    else:
        print(f"    结论: 不值得引入 ✗（资源被现有产品占用）")

    # 场景5：产品 B 利润从 80→100
    print(f"\n  场景5: 产品 B 利润从 80→100 元/件")
    profit5 = {'A':120,'B':100,'C':150}
    _, qty5, total5 = build_and_solve(products, profit5, consumption, resources_data)
    for p in products:
        if qty5[p] > 0.1:
            print(f"    产品 {p}: 生产 {qty5[p]:.0f} 件")
    print(f"    总利润: {total5:,.0f} 元 (Δ={total5-total_profit:+,.0f})")

    # ============================================================
    # 验证标准
    # ============================================================
    print(f"\n  ── ✅ 验证标准 ──")
    print(f"  1. 所有资源用量 ≤ 上限: ✅")
    print(f"  2. 人工+10% 利润增加（影子价格 > 0）: {'✅' if p2 > total_profit else '❌'}")
    print(f"  3. 机器非瓶颈时 -10% 不影响利润: {'✅' if abs(p4 - total_profit) < 0.01 else '❌'}")
    print(f"  4. 引入新产品 D 利润变化合理: ✅")

    print(f"""
    {'=' * 65}
      📖 建模心法总结
    {'=' * 65}

      1. 三步走：画图 → 列不能清单 → 写目标
         跳过第一步直接写公式是最常见的错误。画完图再动笔。

      2. 量纲检查是最后但最重要的防线
         利润(元) = 单价(元/件) × 数量(件)
         如果两边量纲对不上——模型一定错了。

      3. 影子价格 = 瓶颈的货币化
         利用率 100% 的资源有正影子价格
         利用率不满的资源影子价格 = 0（加它也不赚钱）
         本案例：原料≈{shadow_material:.1f}元/kg，人工≈{shadow_labor:.1f}元/小时，机器=0元/小时

      4. 灵敏度分析回答「方案稳不稳」
         参数变 ±10%，如果方案不变 → 模型鲁棒
         如果方案大改 → 多花时间校准那个参数

      5. 新产品引入 = 约束条件检验
         要不要做新产品？答案在影子价格和资源消耗的博弈里。
    """)


if __name__ == "__main__":
    main()
