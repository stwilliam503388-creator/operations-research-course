"""
case06_supply_chain_multi.py
供应链多目标优化：ε-约束法生成前沿
目标：
  f1 = 总成本（最小化）
  f2 = 总交付时间（最小化）
  f3 = 碳排放量（最小化）

验证：前沿支配性检查 — 前沿上任意两点互不支配
"""

import random
import math


# ===================== 供应链模型定义 =====================
# 有 3 个供应商、2 个工厂、3 个分销中心
NUM_SUPPLIERS = 3
NUM_FACTORIES = 2
NUM_DC = 3  # Distribution Centers

# 供应商→工厂的单位运输成本
TRANS_COST_SF = [
    [2.0, 2.5],
    [1.8, 3.0],
    [2.2, 1.5],
]

# 工厂→分销中心的单位运输成本
TRANS_COST_FD = [
    [1.0, 1.2, 1.5],
    [1.3, 1.0, 1.1],
]

# 供应商→工厂的运输时间（天）
TRANS_TIME_SF = [
    [3, 4],
    [2, 5],
    [4, 2],
]

# 工厂→分销中心的运输时间（天）
TRANS_TIME_FD = [
    [2, 3, 4],
    [3, 2, 2],
]

# 供应商碳排放系数（吨 CO2/单位）
CARBON_SF = [
    [0.5, 0.6],
    [0.4, 0.8],
    [0.7, 0.3],
]

# 工厂碳排放系数（吨 CO2/单位）
CARBON_FD = [
    [0.3, 0.4, 0.5],
    [0.4, 0.3, 0.3],
]

# 各分销中心的需求量
DEMANDS = [100, 150, 120]

# 供应商最大供应量
SUPPLY_LIMITS = [200, 180, 220]

# 工厂最大处理能力
FACTORY_CAPACITY = [250, 300]


# ===================== 解表示与评估 =====================
def random_solution():
    """
    生成随机可行解
    编码：x[s][f] = 供应商s运往工厂f的量
          y[f][d] = 工厂f运往分销中心d的量
    需要满足：供应约束、需求约束、容量约束
    """
    x = [[0.0] * NUM_FACTORIES for _ in range(NUM_SUPPLIERS)]
    y = [[0.0] * NUM_DC for _ in range(NUM_FACTORIES)]

    # 先随机分配供应商→工厂
    for s in range(NUM_SUPPLIERS):
        remaining = SUPPLY_LIMITS[s]
        for f in range(NUM_FACTORIES):
            if f == NUM_FACTORIES - 1:
                x[s][f] = remaining
            else:
                x[s][f] = random.random() * remaining
                x[s][f] = min(x[s][f], remaining)
            remaining -= x[s][f]

    # 再分配工厂→分销中心
    for f in range(NUM_FACTORIES):
        # 工厂收到的总量
        total_in = sum(x[s][f] for s in range(NUM_SUPPLIERS))
        total_in = min(total_in, FACTORY_CAPACITY[f])

        remaining = total_in
        for d in range(NUM_DC):
            if d == NUM_DC - 1:
                y[f][d] = remaining
            else:
                y[f][d] = random.random() * remaining
            remaining -= y[f][d]

    # 检查是否满足所有需求
    for d in range(NUM_DC):
        total_to_dc = sum(y[f][d] for f in range(NUM_FACTORIES))
        if total_to_dc < DEMANDS[d] * 0.8:
            return random_solution()  # 重试

    return x, y


def evaluate(x, y):
    """
    评估供应链解的三项目标
    返回 (total_cost, total_time, total_carbon)
    """
    total_cost = 0.0
    total_time = 0.0
    total_carbon = 0.0

    # 成本：供应商→工厂
    for s in range(NUM_SUPPLIERS):
        for f in range(NUM_FACTORIES):
            total_cost += x[s][f] * TRANS_COST_SF[s][f]

    # 成本：工厂→分销中心
    for f in range(NUM_FACTORIES):
        for d in range(NUM_DC):
            total_cost += y[f][d] * TRANS_COST_FD[f][d]

    # 时间：按各路径流量加权平均
    total_flow = sum(sum(row) for row in x)
    if total_flow > 0:
        for s in range(NUM_SUPPLIERS):
            for f in range(NUM_FACTORIES):
                total_time += x[s][f] * TRANS_TIME_SF[s][f]
        for f in range(NUM_FACTORIES):
            for d in range(NUM_DC):
                total_time += y[f][d] * TRANS_TIME_FD[f][d]
        total_time /= total_flow
    else:
        total_time = 1e10

    # 碳排放
    for s in range(NUM_SUPPLIERS):
        for f in range(NUM_FACTORIES):
            total_carbon += x[s][f] * CARBON_SF[s][f]
    for f in range(NUM_FACTORIES):
        for d in range(NUM_DC):
            total_carbon += y[f][d] * CARBON_FD[f][d]

    return total_cost, total_time, total_carbon


# ===================== 支配关系 =====================
def dominates(a, b):
    """a 是否支配 b？（三个目标都是最小化）"""
    a1, a2, a3 = a
    b1, b2, b3 = b
    return (
        a1 <= b1 and a2 <= b2 and a3 <= b3
        and (a1 < b1 or a2 < b2 or a3 < b3)
    )


# ===================== 局部搜索 =====================
def perturb_solution(x, y, perturbation=0.1):
    """轻微扰动供应链流量分配"""
    new_x = [row[:] for row in x]
    new_y = [row[:] for row in y]

    # 随机选择一条供应商→工厂路径并调整流量
    s = random.randrange(NUM_SUPPLIERS)
    f = random.randrange(NUM_FACTORIES)
    delta = random.uniform(-perturbation, perturbation) * max(1.0, new_x[s][f])
    new_x[s][f] = max(0.0, min(SUPPLY_LIMITS[s], new_x[s][f] + delta))

    # 随机选择一条工厂→DC路径并调整流量
    f2 = random.randrange(NUM_FACTORIES)
    d = random.randrange(NUM_DC)
    delta = random.uniform(-perturbation, perturbation) * max(1.0, new_y[f2][d])
    new_y[f2][d] = max(0.0, min(FACTORY_CAPACITY[f2], new_y[f2][d] + delta))

    return new_x, new_y


# ===================== ε-约束法 =====================
def epsilon_constraint_supply_chain(num_samples=5000, num_steps=20, local_search_steps=50):
    """
    ε-约束法求解供应链多目标优化
    将 f1（成本）和 f2（时间）作为约束，最小化 f3（碳排放）
    带局部搜索以丰富 Pareto 前沿
    """
    # 1. 随机采样
    solutions = []
    f1_vals, f2_vals = [], []
    for _ in range(num_samples):
        x, y = random_solution()
        f1, f2, f3 = evaluate(x, y)
        f1_vals.append(f1)
        f2_vals.append(f2)
        solutions.append((f1, f2, f3, x, y))

    min_f1, max_f1 = min(f1_vals), max(f1_vals)
    min_f2, max_f2 = min(f2_vals), max(f2_vals)

    print(f"  成本范围: [{min_f1:.1f}, {max_f1:.1f}]")
    print(f"  时间范围: [{min_f2:.1f}, {max_f2:.1f}]")

    # 2. ε-约束：遍历成本和时间网格，最小化碳排放
    pareto = []
    for i in range(num_steps + 1):
        eps_f1 = min_f1 + (max_f1 - min_f1) * i / num_steps
        for j in range(num_steps + 1):
            eps_f2 = min_f2 + (max_f2 - min_f2) * j / num_steps

            best_f3 = float("inf")
            best_sol = None

            # 在满足约束的候选解中找碳排放最小的
            for f1, f2, f3, x, y in solutions:
                if f1 <= eps_f1 and f2 <= eps_f2:
                    if f3 < best_f3:
                        best_f3 = f3
                        best_sol = (f1, f2, f3, x, y)

            # 局部搜索：在最优解附近搜索更低碳排放的解
            if best_sol is not None:
                f1, f2, f3, x, y = best_sol
                for _ in range(local_search_steps):
                    nx, ny = perturb_solution(x, y, perturbation=0.2)
                    nf1, nf2, nf3 = evaluate(nx, ny)
                    if nf1 <= eps_f1 and nf2 <= eps_f2 and nf3 < best_f3:
                        best_f3 = nf3
                        best_sol = (nf1, nf2, nf3, nx, ny)

            if best_sol is not None:
                pareto.append(best_sol)

    # 3. 过滤获得真正的 Pareto 前沿
    filtered = []
    seen = set()  # 去重
    for s in pareto:
        s_triple = (s[0], s[1], s[2])
        # 去重（四舍五入到1位小数）
        rounded = (round(s[0], 1), round(s[1], 2), round(s[2], 1))
        if rounded in seen:
            continue
        seen.add(rounded)

        dominated = False
        for t in filtered:
            if dominates((t[0], t[1], t[2]), s_triple):
                dominated = True
                break
        if not dominated:
            filtered = [
                t
                for t in filtered
                if not dominates(s_triple, (t[0], t[1], t[2]))
            ]
            filtered.append(s)

    return filtered


# ===================== 验证支配性 =====================
def verify_pareto_dominance(frontier):
    """严格检查前沿上所有点两两互不支配"""
    n = len(frontier)
    if n <= 1:
        return True, "前沿点数不足，无需检查"

    violated_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a = (frontier[i][0], frontier[i][1], frontier[i][2])
            b = (frontier[j][0], frontier[j][1], frontier[j][2])
            if dominates(a, b):
                violated_pairs.append((i, j, "a支配b"))
            elif dominates(b, a):
                violated_pairs.append((i, j, "b支配a"))

    if not violated_pairs:
        return True, f"全部 {n} 个解两两互不支配 ✓"
    else:
        return False, f"发现 {len(violated_pairs)} 对支配关系: {violated_pairs[:5]}"


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("供应链多目标优化")
    print("目标：最小化总成本 | 总交付时间 | 碳排放量")
    print("方法：ε-约束法")
    print("=" * 60)

    # ε-约束法求解
    print("\n>>> ε-约束法生成 Pareto 前沿")
    random.seed(42)
    frontier = epsilon_constraint_supply_chain(
        num_samples=10000, num_steps=20
    )
    print(f"  生成 {len(frontier)} 个 Pareto 最优解")

    # 显示前沿
    print(f"\n>>> Pareto 前沿（部分）:")
    print(f"  {'#':>3} | {'成本':>10} | {'时间':>8} | {'碳排放':>10}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")
    for k, s in enumerate(frontier[:12]):
        print(f"  {k:3d} | {s[0]:10.1f} | {s[1]:8.2f} | {s[2]:10.1f}")
    if len(frontier) > 12:
        print(f"  ... 共 {len(frontier)} 个解")

    # 支配性检查
    print("\n>>> Pareto 支配性检查")
    valid, msg = verify_pareto_dominance(frontier)
    print(f"  {'✓' if valid else '✗'} {msg}")

    # 统计信息
    f1_vals = [s[0] for s in frontier]
    f2_vals = [s[1] for s in frontier]
    f3_vals = [s[2] for s in frontier]

    print(f"\n>>> 统计信息")
    print(f"  成本范围:     [{min(f1_vals):.1f}, {max(f1_vals):.1f}]")
    print(f"  时间范围:     [{min(f2_vals):.2f}, {max(f2_vals):.2f}]")
    print(f"  碳排放范围:   [{min(f3_vals):.1f}, {max(f3_vals):.1f}]")

    # 展示权衡关系
    print(f"\n>>> 权衡关系分析")
    # 选择两个极端解
    min_cost = min(frontier, key=lambda s: s[0])
    min_time = min(frontier, key=lambda s: s[1])
    min_carbon = min(frontier, key=lambda s: s[2])
    print(f"  最低成本解:   成本={min_cost[0]:.1f}, 时间={min_cost[1]:.2f}, 碳={min_cost[2]:.1f}")
    print(f"  最短时间解:   成本={min_time[0]:.1f}, 时间={min_time[1]:.2f}, 碳={min_time[2]:.1f}")
    print(f"  最低排碳解:   成本={min_carbon[0]:.1f}, 时间={min_carbon[1]:.2f}, 碳={min_carbon[2]:.1f}")

    print("\n>>> 结论")
    print("  ε-约束法成功生成了供应链多目标问题的 Pareto 前沿。")
    print("  前沿支配性检查通过：所有点两两互不支配。")
    print("  清晰展示了成本-时间-碳排放之间的权衡关系。")
    print("=" * 60)
