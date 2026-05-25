"""
案例4：随机供应链网络设计
难度：★★★★☆
方法：随机规划 / 两阶段 + 设施选址
依赖：pip install numpy scipy
运行：python code/case04_supply_chain.py

简述：电商仓库选址——先决定建哪些仓库，再根据需求情景决定配送方案。
"""

import numpy as np
from scipy.optimize import linprog
import itertools


# ============================================================
# 数据准备
# ============================================================
rng = np.random.default_rng(42)

n_warehouses = 8  # 候选仓库数
n_customers = 20  # 客户数
n_scenarios = 4   # 情景数

# 建设成本 (500~1500万)
build_cost = rng.uniform(500, 1500, n_warehouses)
# 仓库容量 (每个仓库最大配送量)
capacity = rng.uniform(80, 200, n_warehouses)

# 客户位置 (随机的二维坐标)
cust_pos = rng.uniform(0, 100, (n_customers, 2))
ware_pos = rng.uniform(0, 100, (n_warehouses, 2))

# 计算运输成本 (欧氏距离 × 单位运输成本 = 2 元/单位/距离)
unit_transport_cost = 2.0
transport_cost = np.zeros((n_warehouses, n_customers))
for j in range(n_warehouses):
    for i in range(n_customers):
        dist = np.sqrt(np.sum((cust_pos[i] - ware_pos[j]) ** 2))
        transport_cost[j, i] = dist * unit_transport_cost

# 需求情景
scenario_names = ["高速增长", "中速增长", "低速增长", "衰退"]
scenario_probs = [0.15, 0.35, 0.35, 0.15]
# 每个情景下每个客户的需求 (基础需求 × 增长率)
base_demand = rng.uniform(10, 40, n_customers)
growth_rates = [1.4, 1.1, 0.9, 0.6]

demands = np.zeros((n_scenarios, n_customers))
for s in range(n_scenarios):
    demands[s] = base_demand * growth_rates[s]


# ============================================================
# 模型求解 — 枚举法 (小规模快且精确)
# ============================================================
def solve_scenario_given_warehouses(open_warehouses, s):
    """
    给定已开仓库集合和情景 s，求解最优配送方案 (运输成本最小化)。
    这是一个标准的运输问题。
    """
    open_list = [j for j in range(n_warehouses) if open_warehouses[j] == 1]
    if not open_list:
        return np.inf, None

    n_open = len(open_list)
    # 变量: x[j][i] 从仓库 j 运到客户 i
    n_vars = n_open * n_customers

    # 目标: 最小化运输成本
    c_obj = []
    for j_idx, j in enumerate(open_list):
        for i in range(n_customers):
            c_obj.append(transport_cost[j, i])

    # 约束
    A_ub = []
    b_ub = []

    # 仓库容量约束: Σ_i x_ji <= capacity_j
    for j_idx, j in enumerate(open_list):
        row = [0] * n_vars
        for i in range(n_customers):
            row[j_idx * n_customers + i] = 1
        A_ub.append(row)
        b_ub.append(capacity[j])

    # 客户需求约束 (等式): Σ_j x_ji = demand_i_s
    A_eq = []
    b_eq = []
    for i in range(n_customers):
        row = [0] * n_vars
        for j_idx in range(n_open):
            row[j_idx * n_customers + i] = 1
        A_eq.append(row)
        b_eq.append(demands[s, i])

    bounds = [(0, None)] * n_vars
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
    if result.success:
        return result.fun, result.x
    return np.inf, None


def evaluate_solution(open_warehouses):
    """
    评估一个仓库开放方案在所有情景下的期望总成本。
    """
    total_build = np.sum(build_cost * open_warehouses)
    total_transport = 0
    for s in range(n_scenarios):
        cost, _ = solve_scenario_given_warehouses(open_warehouses, s)
        if np.isinf(cost):
            return np.inf
        total_transport += scenario_probs[s] * cost
    return total_build + total_transport


def solve_two_stage_supply_chain():
    """
    枚举所有可能的仓库开放组合（2^8=256种），选择期望总成本最小的。
    """
    best_cost = np.inf
    best_solution = None
    best_details = None

    for bits in itertools.product([0, 1], repeat=n_warehouses):
        # 至少开一个仓库
        if sum(bits) == 0:
            continue
        total_cost = evaluate_solution(np.array(bits))
        if total_cost < best_cost:
            best_cost = total_cost
            best_solution = np.array(bits)
            # 计算每个情景的详细成本
            details = {}
            details["build_cost"] = np.sum(build_cost * best_solution)
            details["transport_costs"] = []
            for s in range(n_scenarios):
                tc, _ = solve_scenario_given_warehouses(best_solution, s)
                details["transport_costs"].append(tc)
            details["expected_transport"] = sum(
                scenario_probs[s] * details["transport_costs"][s]
                for s in range(n_scenarios)
            )
            best_details = details

    return best_solution, best_cost, best_details


def solve_deterministic_supply_chain():
    """
    确定性方案：用平均需求做选址。
    """
    avg_demand = np.mean(demands, axis=0)
    best_cost = np.inf
    best_solution = None
    best_details = None

    for bits in itertools.product([0, 1], repeat=n_warehouses):
        if sum(bits) == 0:
            continue
        open_list = [j for j in range(n_warehouses) if bits[j] == 1]
        n_open = len(open_list)
        n_vars = n_open * n_customers
        c_obj = []
        for j_idx, j in enumerate(open_list):
            for i in range(n_customers):
                c_obj.append(transport_cost[j, i])

        A_ub = []
        b_ub = []
        for j_idx, j in enumerate(open_list):
            row = [0] * n_vars
            for i in range(n_customers):
                row[j_idx * n_customers + i] = 1
            A_ub.append(row)
            b_ub.append(capacity[j])

        A_eq = []
        b_eq = []
        for i in range(n_customers):
            row = [0] * n_vars
            for j_idx in range(n_open):
                row[j_idx * n_customers + i] = 1
            A_eq.append(row)
            b_eq.append(avg_demand[i])

        bounds = [(0, None)] * n_vars
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
        if result.success:
            tc = result.fun
            build = np.sum(build_cost * np.array(bits))
            total = build + tc
            if total < best_cost:
                best_cost = total
                best_solution = np.array(bits)
                # 在各情景下评估
                details = {}
                details["build_cost"] = build
                details["transport_costs"] = []
                for s in range(n_scenarios):
                    stc, _ = solve_scenario_given_warehouses(best_solution, s)
                    details["transport_costs"].append(stc)
                details["expected_transport"] = sum(
                    scenario_probs[s] * details["transport_costs"][s]
                    for s in range(n_scenarios)
                )
                best_details = details
    return best_solution, best_cost, best_details


# ============================================================
# 结果输出
# ============================================================
def print_summary():
    print("=" * 60)
    print("案例4：随机供应链网络设计")
    print("=" * 60)

    # 1. 随机解
    rand_sol, rand_cost, rand_det = solve_two_stage_supply_chain()
    rand_n = int(np.sum(rand_sol))
    print(f"\n▶ 两阶段随机方案:")
    print(f"  开放仓库数: {rand_n}/8")
    print(f"  开放列表: {[j for j in range(n_warehouses) if rand_sol[j] == 1]}")
    print(f"  建设成本: {rand_det['build_cost']:.0f} 万")
    print(f"  期望年运输成本: {rand_det['expected_transport']:.0f} 万")
    for s in range(n_scenarios):
        print(f"    [{scenario_names[s]}] 运输成本: {rand_det['transport_costs'][s]:.0f} 万")
    print(f"  总成本 (4年): {rand_det['build_cost'] + 4 * rand_det['expected_transport']:.0f} 万")

    # 2. 确定性解
    det_sol, det_cost, det_det = solve_deterministic_supply_chain()
    det_n = int(np.sum(det_sol))
    print(f"\n▶ 确定性方案 (平均需求):")
    print(f"  开放仓库数: {det_n}/8")
    print(f"  开放列表: {[j for j in range(n_warehouses) if det_sol[j] == 1]}")
    print(f"  建设成本: {det_det['build_cost']:.0f} 万")
    print(f"  期望年运输成本: {det_det['expected_transport']:.0f} 万")
    for s in range(n_scenarios):
        print(f"    [{scenario_names[s]}] 运输成本: {det_det['transport_costs'][s]:.0f} 万")
    print(f"  总成本 (4年): {det_det['build_cost'] + 4 * det_det['expected_transport']:.0f} 万")

    # 3. 对比
    rand_4y = rand_det['build_cost'] + 4 * rand_det['expected_transport']
    det_4y = det_det['build_cost'] + 4 * det_det['expected_transport']
    print(f"\n▶ 对比 (4年期总成本):")
    print(f"  随机方案: {rand_4y:.0f} 万")
    print(f"  确定性方案: {det_4y:.0f} 万")
    if rand_4y < det_4y:
        print(f"  ✅ 随机方案节省: {det_4y - rand_4y:.0f} 万 ({((det_4y - rand_4y) / det_4y * 100):.1f}%)")
    else:
        print(f"  ⚠️ 随机方案更贵: {rand_4y - det_4y:.0f} 万")

    # 4. 各情景对比
    print(f"\n▶ 各情景运输成本对比:")
    print(f"{'情景':>10} | {'随机方案':>10} | {'确定性方案':>10} | {'差异':>10}")
    print("-" * 50)
    for s in range(n_scenarios):
        diff = rand_det['transport_costs'][s] - det_det['transport_costs'][s]
        print(f"{scenario_names[s]:>10} | {rand_det['transport_costs'][s]:>10.0f} | {det_det['transport_costs'][s]:>10.0f} | {diff:>+10.0f}")


def solve_small():
    """小规模测试：4仓库, 10客户"""
    print("\n=== 小规模测试: 4仓库, 10客户 ===")
    # 临时缩小规模
    global n_warehouses, n_customers, build_cost, capacity, transport_cost, demands
    old_w, old_c = n_warehouses, n_customers

    n_warehouses = 4
    n_customers = 10
    build_cost = rng.uniform(500, 1500, n_warehouses)
    capacity = rng.uniform(80, 200, n_warehouses)
    transport_cost = np.zeros((n_warehouses, n_customers))
    for j in range(n_warehouses):
        for i in range(n_customers):
            dist = np.sqrt(np.sum((rng.uniform(0, 100, 2) - rng.uniform(0, 100, 2)) ** 2))
            transport_cost[j, i] = dist * unit_transport_cost

    sol, cost, det = solve_two_stage_supply_chain()
    print(f"最优: {int(np.sum(sol))} 个仓库, 总成本={cost:.0f}")

    n_warehouses, n_customers = old_w, old_c
    # 恢复原始数据
    _reset_data()


def _reset_data():
    global rng, build_cost, capacity, transport_cost, demands
    rng = np.random.default_rng(42)
    n_warehouses = 8
    n_customers = 20
    build_cost = rng.uniform(500, 1500, n_warehouses)
    capacity = rng.uniform(80, 200, n_warehouses)
    cust_pos = rng.uniform(0, 100, (n_customers, 2))
    ware_pos = rng.uniform(0, 100, (n_warehouses, 2))
    transport_cost = np.zeros((n_warehouses, n_customers))
    for j in range(n_warehouses):
        for i in range(n_customers):
            dist = np.sqrt(np.sum((cust_pos[i] - ware_pos[j]) ** 2))
            transport_cost[j, i] = dist * unit_transport_cost
    base_demand = rng.uniform(10, 40, n_customers)
    growth_rates = [1.4, 1.1, 0.9, 0.6]
    demands = np.zeros((n_scenarios, n_customers))
    for s in range(n_scenarios):
        demands[s] = base_demand * growth_rates[s]


def solve_medium():
    """中等规模测试"""
    print("\n=== 中等规模测试 ===")
    print("枚举 2^8=256 种组合...")
    sol, cost, det = solve_two_stage_supply_chain()
    print(f"最优: {int(np.sum(sol))} 个仓库, 期望总成本={cost:.0f}")


if __name__ == "__main__":
    solve_small()
    solve_medium()
    print("\n")
    print_summary()
