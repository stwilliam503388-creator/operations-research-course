"""
案例2：两阶段生产计划
难度：★★★☆☆
方法：随机规划 / 两阶段
依赖：pip install numpy scipy
运行：python code/python/case02_two_stage.py

简述：两阶段生产计划——先采购原材料，观察需求后决定产量。
"""


# 教学注释：通过场景、概率权重和坏情况边界刻画不确定需求或收益。
# 对比不同假设下的解，理解随机性和鲁棒性如何改变决策。



import numpy as np
from scipy.optimize import linprog
import time


# ============================================================
# 数据准备
# ============================================================
c = 10      # 原材料成本 (元/单位)
profit_A = 25  # 产品A利润 (元/件)
profit_B = 30  # 产品B利润 (元/件)
max_raw = 500  # 原材料采购上限

# 两个基础场景
scenarios = [
    {"prob": 0.5, "demand_A": 200, "demand_B": 150, "name": "高需求"},
    {"prob": 0.5, "demand_A": 120, "demand_B": 100, "name": "低需求"},
]


# ============================================================
# 模型求解
# ============================================================
def solve_two_stage(scenarios_list):
    """
    求解两阶段随机规划的对等模型。
    变量顺序: [x, yA_1, yB_1, yA_2, yB_2, ...]
    """
    n = len(scenarios_list)

    # 目标系数
    c_obj = [-c]  # 第一阶段：负的采购成本
    for s in scenarios_list:
        c_obj.extend([-profit_A * s["prob"], -profit_B * s["prob"]])

    # 不等式约束: A_ub @ x <= b_ub
    A_ub = []
    b_ub = []

    # 1. 原材料上限约束
    row = [1] + [0] * (2 * n)
    A_ub.append(row)
    b_ub.append(max_raw)

    # 2. 每个场景的原材料约束: yA_s + yB_s <= x
    for i in range(n):
        row = [-1] + [0] * (2 * n)  # -x + ...
        row[1 + 2 * i] = 1      # yA_s
        row[1 + 2 * i + 1] = 1  # yB_s
        A_ub.append(row)
        b_ub.append(0)

    # 3. 每个场景的需求上限
    for i, s in enumerate(scenarios_list):
        # yA_s <= demand_A_s
        row = [0] * (1 + 2 * n)
        row[1 + 2 * i] = 1
        A_ub.append(row)
        b_ub.append(s["demand_A"])
        # yB_s <= demand_B_s
        row = [0] * (1 + 2 * n)
        row[1 + 2 * i + 1] = 1
        A_ub.append(row)
        b_ub.append(s["demand_B"])

    # 变量边界: 所有变量 >= 0
    bounds = [(0, None)] * (1 + 2 * n)

    # 求解
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        x_val = result.x[0]
        y_vals = result.x[1:].reshape(n, 2)
        # 计算期望利润（注意 linprog 最小化，目标值取负）
        obj_val = -result.fun
        return x_val, y_vals, obj_val
    else:
        raise RuntimeError(f"求解失败: {result.message}")


def solve_deterministic():
    """
    确定性方案：用平均需求。
    """
    avg_A = np.mean([s["demand_A"] for s in scenarios])
    avg_B = np.mean([s["demand_B"] for s in scenarios])

    # 模型: max 25yA + 30yB - 10x
    # 等价于 min -25yA -30yB + 10x
    c_obj = [10, -profit_A, -profit_B]  # x, yA, yB

    A_ub = [
        [1, 0, 0],       # x <= 500
        [-1, 1, 1],       # yA + yB <= x
        [0, 1, 0],        # yA <= avg_A
        [0, 0, 1],        # yB <= avg_B
    ]
    b_ub = [max_raw, 0, avg_A, avg_B]
    bounds = [(0, None)] * 3

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        x_val = result.x[0]
        yA, yB = result.x[1], result.x[2]
        obj_val = -result.fun
        return x_val, yA, yB, obj_val
    else:
        raise RuntimeError(f"确定性求解失败: {result.message}")


def evaluate_deterministic_on_scenarios(x_val):
    """
    将确定性方案的 x 放到各场景下评估实际利润。
    """
    total_profit = 0
    for s in scenarios:
        # 在给定 x 和场景需求下，最优第二阶段决策
        # 问题: max 25yA + 30yB st yA+yB<=x, yA<=dA, yB<=dB
        dA, dB = s["demand_A"], s["demand_B"]
        # 直观分配：优先高利润的 B
        yB = min(dB, x_val)
        yA = min(dA, x_val - yB)
        stage2_profit = profit_A * yA + profit_B * yB
        total_profit += s["prob"] * (-c * x_val + stage2_profit)
    return total_profit


def generate_n_scenarios(n, seed=42):
    """从基础场景分布生成 N 个随机场景。"""
    rng = np.random.default_rng(seed)
    scenarios_n = []
    for i in range(n):
        prob = 1.0 / n
        # 从两个基础场景中随机选一个作为基础，加一点噪声
        base = rng.choice([0, 1])
        if base == 0:  # 高需求
            dA = max(50, int(200 + rng.normal(0, 20)))
            dB = max(50, int(150 + rng.normal(0, 15)))
        else:
            dA = max(50, int(120 + rng.normal(0, 15)))
            dB = max(50, int(100 + rng.normal(0, 10)))
        scenarios_n.append({"prob": prob, "demand_A": dA, "demand_B": dB, "name": f"S{i + 1}"})
    return scenarios_n


# ============================================================
# 结果输出
# ============================================================
def print_summary():
    print("=" * 60)
    print("案例2：两阶段生产计划")
    print("=" * 60)

    # 1. 两阶段解
    x_2s, y_2s, obj_2s = solve_two_stage(scenarios)
    print(f"\n▶ 两阶段随机规划解:")
    print(f"  采购量 x* = {x_2s:.1f}")
    for i, s in enumerate(scenarios):
        print(f"  [{s['name']}] yA={y_2s[i][0]:.1f}, yB={y_2s[i][1]:.1f}")
    print(f"  期望利润 = {obj_2s:.2f} 元")

    # 2. 确定性解
    x_det, yA_det, yB_det, obj_det = solve_deterministic()
    print(f"\n▶ 确定性方案 (用平均需求 A={np.mean([s['demand_A'] for s in scenarios]):.0f}, B={np.mean([s['demand_B'] for s in scenarios]):.0f}):")
    print(f"  采购量 x = {x_det:.1f}, yA = {yA_det:.1f}, yB = {yB_det:.1f}")
    print(f"  模型利润 = {obj_det:.2f} 元")

    # 3. 确定性方案在随机场景下的实际表现
    det_profit = evaluate_deterministic_on_scenarios(x_det)
    print(f"\n▶ 确定性方案在各场景下的实际期望利润 = {det_profit:.2f} 元")

    # 4. 对比
    print(f"\n▶ 对比:")
    print(f"  两阶段期望利润: {obj_2s:.2f}")
    print(f"  确定性期望利润: {det_profit:.2f}")
    gain = (obj_2s - det_profit) / det_profit * 100
    print(f"  两阶段多赚: {gain:.2f}%")
    if obj_2s > det_profit:
        print("✅ 验证通过：两阶段 > 确定性")
    else:
        print("⚠️ 验证未通过")


def solve_small():
    """小规模：2 场景"""
    print("\n=== 小规模测试: 2 场景 ===")
    x, y, obj = solve_two_stage(scenarios)
    print(f"采购量 x={x:.1f}, 期望利润={obj:.2f}")


def solve_medium():
    """中等规模：10, 50, 100 场景"""
    print("\n=== 中等规模测试: 扩展场景 ===")
    for n in [10, 50, 100]:
        scens = generate_n_scenarios(n, seed=42)
        start = time.time()
        x, y, obj = solve_two_stage(scens)
        elapsed = time.time() - start
        print(f"N={n:>4}: x={x:.1f}, 期望利润={obj:.2f}, 耗时={elapsed:.3f}s")


if __name__ == "__main__":
    solve_small()
    solve_medium()
    print("\n")
    print_summary()
