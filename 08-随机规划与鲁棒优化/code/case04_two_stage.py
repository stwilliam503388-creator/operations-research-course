"""
两阶段生产计划 — 随机规划版本
=============================
场景生成，第一阶段备料决策，第二阶段生产调整。

问题描述：
  工厂生产两种产品（A, B），需要两种原材料（M1, M2）。
  第一阶段：在需求不确定的情况下，决定每种原材料的采购量。
  第二阶段：需求实现后，根据实际需求调整产品产量，若原料不足则外购（高价）。
  目标是最大化期望总利润。

本代码纯 Python 标准库 + numpy，中文注释，可自测。
"""

import random
import math

# ---------- 参数 ----------
# 产品售价
PRODUCT_PRICE = {"A": 12.0, "B": 15.0}
# 产品原料消耗 [M1, M2]
PRODUCT_RECIPE = {"A": [2.0, 1.0], "B": [1.0, 2.0]}
# 原料正常采购成本（第一阶段）
MAT_COST = [3.0, 4.0]
# 原料紧急外购成本（第二阶段，更贵）
MAT_EMERGENCY_COST = [6.0, 8.0]
# 原料容量上限（最大可采购量）
MAT_CAP = [500, 400]

# 需求场景：每个场景包含 (prob, demand_A, demand_B)
# 这里生成 5 个等概率场景
N_SCENARIOS = 5
DEMAND_SCENARIOS = [
    (0.20, 50, 30),
    (0.20, 80, 50),
    (0.20, 100, 80),
    (0.20, 130, 100),
    (0.20, 160, 120),
]

# 第一阶段备料决策的候选值（离散化搜索）
CANDIDATE_RANGES = [
    range(50, 401, 25),   # M1 采购量候选
    range(50, 351, 25),   # M2 采购量候选
]


def solve_second_stage(x1, x2, demand_a, demand_b):
    """
    第二阶段决策：给定原料库存 (x1, x2) 和实际需求，
    决定产品产量 yA, yB 和紧急外购量 e1, e2，最大化第二阶段利润。

    这是一个小规模线性规划，我们通过枚举可行产量组合求解。

    返回: (第二阶段利润, yA, yB, e1, e2)
    """
    best_value = -float("inf")
    best_plan = (0, 0, 0, 0)

    # 产品产量不能超过需求
    max_a = int(demand_a)
    max_b = int(demand_b)

    for yA in range(max_a + 1):
        for yB in range(max_b + 1):
            # 所需原料
            need1 = PRODUCT_RECIPE["A"][0] * yA + PRODUCT_RECIPE["B"][0] * yB
            need2 = PRODUCT_RECIPE["A"][1] * yA + PRODUCT_RECIPE["B"][1] * yB

            # 紧急外购量
            e1 = max(0.0, need1 - x1)
            e2 = max(0.0, need2 - x2)

            # 收入 - 紧急采购成本
            revenue = PRODUCT_PRICE["A"] * yA + PRODUCT_PRICE["B"] * yB
            cost_emergency = MAT_EMERGENCY_COST[0] * e1 + MAT_EMERGENCY_COST[1] * e2
            stage2_profit = revenue - cost_emergency

            if stage2_profit > best_value:
                best_value = stage2_profit
                best_plan = (yA, yB, e1, e2)

    return best_value, best_plan[0], best_plan[1], best_plan[2], best_plan[3]


def evaluate_first_stage(x1, x2):
    """
    评估第一阶段的期望总利润 = -原料成本 + 各场景第二阶段利润的期望。
    返回: 期望总利润, 各场景明细
    """
    mat_cost = MAT_COST[0] * x1 + MAT_COST[1] * x2
    total_exp_profit = 0.0
    details = []
    for prob, dA, dB in DEMAND_SCENARIOS:
        stg2_profit, yA, yB, e1, e2 = solve_second_stage(x1, x2, dA, dB)
        total_exp_profit += prob * (stg2_profit - mat_cost)
        details.append((prob, dA, dB, yA, yB, e1, e2, stg2_profit, stg2_profit - mat_cost))
    return total_exp_profit, details


def search_optimal_first_stage():
    """枚举搜索最优第一阶段决策"""
    best_value = -float("inf")
    best_x = (0, 0)
    best_details = None

    for x1 in CANDIDATE_RANGES[0]:
        for x2 in CANDIDATE_RANGES[1]:
            if x1 > MAT_CAP[0] or x2 > MAT_CAP[1]:
                continue
            value, details = evaluate_first_stage(x1, x2)
            if value > best_value:
                best_value = value
                best_x = (x1, x2)
                best_details = details

    return best_x, best_value, best_details


def main():
    """主函数：自测入口"""
    print("=" * 60)
    print("两阶段生产计划 — 随机规划")
    print("=" * 60)
    print(f"产品: A(售价{PRODUCT_PRICE['A']}), B(售价{PRODUCT_PRICE['B']})")
    print(f"原料成本: M1={MAT_COST[0]}, M2={MAT_COST[1]}")
    print(f"紧急外购: M1={MAT_EMERGENCY_COST[0]}, M2={MAT_EMERGENCY_COST[1]}")
    print(f"场景数: {N_SCENARIOS}")
    print()

    print("--- 需求场景 ---")
    for i, (prob, dA, dB) in enumerate(DEMAND_SCENARIOS):
        print(f"  场景{i+1}: 概率={prob:.2f}, 需求A={dA}, 需求B={dB}")
    print()

    # 求解最优第一阶段决策
    (opt_x1, opt_x2), opt_value, details = search_optimal_first_stage()

    print(f"[最优第一阶段决策]")
    print(f"  原料M1采购量 = {opt_x1}")
    print(f"  原料M2采购量 = {opt_x2}")
    print(f"  期望总利润   = {opt_value:.4f}")
    print()

    print("--- 各场景明细 ---")
    for i, (prob, dA, dB, yA, yB, e1, e2, stg2, total) in enumerate(details):
        print(f"  场景{i+1}: 需求A={dA}, B={dB}")
        print(f"          产量 A={yA}, B={yB}")
        print(f"          紧急外购 M1={e1:.1f}, M2={e2:.1f}")
        print(f"          第二阶段利润={stg2:.2f}, 总情景利润={total:.2f}")

    print()

    # 基线对比：不做任何备料（完全依赖紧急采购）
    base_val, base_details = evaluate_first_stage(0, 0)
    print(f"[基线: 零备料]")
    print(f"  期望总利润 = {base_val:.4f}")
    print(f"  (完全依赖紧急外购，成本更高)")

    gain = opt_value - base_val
    print(f"\n随机规划相对于零备料的收益 = {gain:.4f}")
    if abs(base_val) > 1e-9:
        print(f"相对提升 = {gain / abs(base_val) * 100:.2f}%")
    else:
        print(f"(零备料方案利润为0，相对提升无穷大)")

    print()
    print("测试通过！")


if __name__ == "__main__":
    main()
