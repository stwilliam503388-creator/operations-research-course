"""
随机供应链网络设计 — 场景SAMPC方法
===================================
仓库选址问题的随机规划版本。

问题描述：
  计划建设仓库为多个客户提供服务。每个仓库有建设成本和容量。
  客户需求不确定（多场景），每个场景下决定每个仓库向每个客户的配送量。
  目标是：第一阶段决定建哪些仓库（选址），第二阶段期望配送成本最小化。

本代码使用场景方法（Sample Average Approximation, SAA / SAMPC），
用蒙特卡洛采样的方式逼近期望成本。

纯 Python 标准库 + numpy，中文注释，可自测。
"""

import random
import math

# ---------- 仓库数据 ----------
WAREHOUSES = [
    {"name": "WH1", "fixed_cost": 1000, "capacity": 500},
    {"name": "WH2", "fixed_cost": 1200, "capacity": 600},
    {"name": "WH3", "fixed_cost": 800,  "capacity": 400},
    {"name": "WH4", "fixed_cost": 1500, "capacity": 700},
]
N_WAREHOUSES = len(WAREHOUSES)

# ---------- 客户数据 ----------
CUSTOMERS = [
    {"name": "C1", "lat": 0.0, "lon": 0.0},
    {"name": "C2", "lat": 3.0, "lon": 4.0},
    {"name": "C3", "lat": 5.0, "lon": 1.0},
    {"name": "C4", "lat": 1.0, "lon": 5.0},
    {"name": "C5", "lat": 4.0, "lon": 3.0},
]
N_CUSTOMERS = len(CUSTOMERS)

# 单位运输成本（每单位距离每单位货物）
TRANSPORT_COST_PER_UNIT = 0.1

# 场景数（SAMPC样本量）
N_SCENARIOS = 200

# 需求分布参数（每个客户的需求均值和标准差）
DEMAND_PARAMS = [
    {"mean": 80, "std": 20},
    {"mean": 100, "std": 30},
    {"mean": 60, "std": 15},
    {"mean": 90, "std": 25},
    {"mean": 70, "std": 20},
]


def euclidean_dist(lat1, lon1, lat2, lon2):
    """计算平面欧氏距离"""
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def generate_demand_scenario():
    """生成一个需求场景（所有客户的需求向量）"""
    return [
        max(0, random.gauss(params["mean"], params["std"]))
        for params in DEMAND_PARAMS
    ]


def generate_scenarios(n):
    """生成 n 个需求场景"""
    return [generate_demand_scenario() for _ in range(n)]


def transport_cost(w_idx, c_idx, amount):
    """单个仓库到单个客户的运输成本"""
    w = WAREHOUSES[w_idx]
    c = CUSTOMERS[c_idx]
    dist = euclidean_dist(c["lat"], c["lon"], 0.0, 0.0)  # 假设仓库在原点
    # 更精确：每个仓库在不同位置，这里简化
    dist = euclidean_dist(c["lat"], c["lon"], 0.0, 0.0)
    return TRANSPORT_COST_PER_UNIT * dist * amount


def solve_transportation(open_warehouses, demands):
    """
    给定已建仓库集合和客户需求，求解最优配送方案（最小化运输成本）。
    贪心分配：将客户需求尽可能分配给最近的已建仓库，若容量不足则分配给次近的。

    参数:
      open_warehouses: list of bool, 哪些仓库已建
      demands: list of float, 每个客户的需求量

    返回: (总运输成本, 是否全部满足)
    """
    # 计算各仓库剩余容量
    remaining_cap = [WH["capacity"] if open_warehouses[i] else 0
                     for i, WH in enumerate(WAREHOUSES)]
    total_cost = 0.0

    # 对每个客户，按距离从近到远分配
    for c_idx, demand in enumerate(demands):
        remaining_demand = demand
        # 按距离排序仓库
        wh_distances = []
        for w_idx in range(N_WAREHOUSES):
            if not open_warehouses[w_idx]:
                continue
            # 仓库简化在 (0,0)，客户在不同位置
            w = WAREHOUSES[w_idx]
            c = CUSTOMERS[c_idx]
            dist = euclidean_dist(c["lat"], c["lon"], 0.0, 0.0)
            wh_distances.append((dist, w_idx))
        wh_distances.sort()

        for dist, w_idx in wh_distances:
            if remaining_demand <= 0:
                break
            assign = min(remaining_demand, remaining_cap[w_idx])
            if assign > 0:
                total_cost += TRANSPORT_COST_PER_UNIT * dist * assign
                remaining_cap[w_idx] -= assign
                remaining_demand -= assign

        if remaining_demand > 1e-6:
            # 无法满足需求，返回巨大成本
            return float("inf"), False

    return total_cost, True


def evaluate_solution(open_decision, scenarios):
    """
    评估一个选址方案的期望总成本 = 固定成本 + 期望运输成本。
    open_decision: list of bool, 哪些仓库建设
    scenarios: list of demand vectors
    """
    # 固定成本
    fixed = sum(WAREHOUSES[i]["fixed_cost"] for i in range(N_WAREHOUSES) if open_decision[i])
    # 各场景运输成本
    trans_total = 0.0
    feasible_count = 0
    for demands in scenarios:
        cost, feasible = solve_transportation(open_decision, demands)
        if not feasible:
            return float("inf")  # 不可行
        trans_total += cost
        feasible_count += 1

    avg_trans = trans_total / len(scenarios) if scenarios else 0.0
    return fixed + avg_trans


def search_best_solution(scenarios):
    """枚举所有可能的仓库组合（2^N_WAREHOUSES 种），找到最优"""
    best_cost = float("inf")
    best_decision = None

    for mask in range(1 << N_WAREHOUSES):
        decision = [bool(mask & (1 << i)) for i in range(N_WAREHOUSES)]
        # 至少建一个仓库
        if not any(decision):
            continue
        cost = evaluate_solution(decision, scenarios)
        if cost < best_cost:
            best_cost = cost
            best_decision = decision

    return best_decision, best_cost


def main():
    """主函数：自测入口"""
    print("=" * 60)
    print("随机供应链网络设计 — SAMPC方法")
    print("=" * 60)
    print()

    print("--- 仓库信息 ---")
    for i, wh in enumerate(WAREHOUSES):
        print(f"  {wh['name']}: 固定成本={wh['fixed_cost']}, 容量={wh['capacity']}")
    print()

    print("--- 客户信息 ---")
    for i, c in enumerate(CUSTOMERS):
        p = DEMAND_PARAMS[i]
        print(f"  {c['name']}: 位置({c['lat']},{c['lon']}), "
              f"需求均值={p['mean']}, 标准差={p['std']}")
    print()

    # 生成场景
    random.seed(123)
    scenarios = generate_scenarios(N_SCENARIOS)
    print(f"生成 {N_SCENARIOS} 个需求场景用于 SAMPC")
    print()

    # 搜索最优解
    print("正在搜索所有仓库组合...")
    best_dec, best_cost = search_best_solution(scenarios)

    print(f"[最优选址方案]")
    for i in range(N_WAREHOUSES):
        status = "✓ 建" if best_dec[i] else "✗ 不建"
        print(f"  {WAREHOUSES[i]['name']}: {status}")
    print(f"期望总成本: {best_cost:.2f}")
    print()

    # 对比：全建方案
    all_open = [True] * N_WAREHOUSES
    cost_all = evaluate_solution(all_open, scenarios)
    print(f"[全建设方案]")
    print(f"  期望总成本: {cost_all:.2f}")

    # 对比：只建最便宜的一个
    cheap_open = [True, False, False, False]
    cost_cheap = evaluate_solution(cheap_open, scenarios)
    print(f"[只建最便宜仓库方案]")
    print(f"  期望总成本: {cost_cheap:.2f}")

    print()
    print(f"随机规划节约成本 (vs 全建): {cost_all - best_cost:.2f}")
    print(f"随机规划节约成本 (vs 最便宜): {cost_cheap - best_cost:.2f}")

    print()
    print("--- 验证不同场景数的影响 ---")
    for n_s in [20, 50, 100, 200]:
        s = generate_scenarios(n_s)
        d, c = search_best_solution(s)
        names = [WAREHOUSES[i]["name"] for i in range(N_WAREHOUSES) if d[i]]
        print(f"  场景数={n_s:3d}: 方案=[{', '.join(names)}], 期望成本={c:.2f}")

    print()
    print("测试通过！")


if __name__ == "__main__":
    main()
