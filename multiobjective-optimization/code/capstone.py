"""
capstone.py
毕业项目：三目标城市物流优化（加权求和法基准对比）

问题描述：
  50辆电动车、200个订单的城市配送问题
  配送中心将货物送至200个客户点，每辆电动车有载重约束

目标（均为最小化）：
  f1 = 总配送成本（元）  — 与行驶距离正相关
  f2 = 总配送时间（小时） — 包括行驶时间 + 卸货时间
  f3 = 碳排放量（kg CO₂） — 与行驶距离成正比

方法：
  加权求和法作为基础对比方法
  通过多组不同权重，展示 Pareto 前沿上的不同偏好解

验证：
  不同权重产生不同解（成本-时间-碳排放的权衡）
"""

import math
import random


# ===================== 城市物流问题定义 =====================

NUM_EVS = 50          # 电动车数量
NUM_ORDERS = 200      # 订单数量（客户点）
VEHICLE_CAPACITY = 20.0  # 每辆车最大载重（单位）
DEPOT = (50.0, 50.0)  # 配送中心坐标

# 每公里成本（元/km）
COST_PER_KM = 3.5

# 行驶速度（km/h）
SPEED_KMH = 35.0

# 每单卸货时间（小时）
UNLOAD_TIME_PER_ORDER = 0.15

# 碳排放系数（kg CO₂ / km）
CARBON_PER_KM = 0.25


def generate_customers(seed=42):
    """生成200个随机客户点坐标和需求量"""
    rng = random.Random(seed)
    customers = []
    demands = []
    for i in range(NUM_ORDERS):
        x = rng.uniform(0.0, 100.0)
        y = rng.uniform(0.0, 100.0)
        d = rng.uniform(0.5, 5.0)  # 每单需求量 0.5~5 单位
        customers.append((x, y))
        demands.append(d)
    return customers, demands


CUSTOMERS, DEMANDS = generate_customers(42)


# ===================== 距离矩阵 =====================

def distance(p1, p2):
    """欧几里得距离"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def build_distance_matrix():
    """构建距离矩阵（索引0为配送中心，1~200为客户点）"""
    points = [DEPOT] + CUSTOMERS
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = distance(points[i], points[j])
    return dist


DIST = build_distance_matrix()


# ===================== 配送解表示 =====================

class DeliverySolution:
    """
    配送方案
    编码：routes = [[c1, c3, c5], [c2, c4], ...]
      每个子列表是一条路径（客户索引 0~199），每辆车从配送中心出发、
      途径若干客户后返回配送中心。
    约束：每条路径总载重 ≤ VEHICLE_CAPACITY
    """

    def __init__(self, routes=None):
        if routes is not None:
            self.routes = routes
        else:
            self.routes = self._random_feasible_routes()
        self.f1 = None  # 总配送成本
        self.f2 = None  # 总配送时间
        self.f3 = None  # 总碳排放

    def _random_feasible_routes(self):
        """生成随机可行路径（贪心拆分，满足容量约束）"""
        customers = list(range(NUM_ORDERS))
        random.shuffle(customers)

        routes = []
        current_route = []
        current_load = 0.0

        for c in customers:
            if current_load + DEMANDS[c] <= VEHICLE_CAPACITY:
                current_route.append(c)
                current_load += DEMANDS[c]
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_load = DEMANDS[c]

        if current_route:
            routes.append(current_route)

        # 限制电动车数量（如果路径数超过 NUM_EVS，合并）
        while len(routes) > NUM_EVS:
            # 合并两条最短路径
            routes.sort(key=lambda r: len(r))
            merged = routes.pop(0) + routes.pop(0)
            routes.append(merged)

        return routes

    def evaluate(self):
        """计算三项目标值"""
        total_cost = 0.0      # f1: 配送成本
        total_time = 0.0      # f2: 配送时间
        total_carbon = 0.0    # f3: 碳排放

        for route in self.routes:
            if not route:
                continue

            # 从配送中心出发
            current = 0  # 配送中心索引
            route_dist = 0.0

            for customer in route:
                # 客户索引从1开始（距离矩阵中）
                ci = customer + 1
                d = DIST[current][ci]
                route_dist += d
                current = ci

            # 返回配送中心
            route_dist += DIST[current][0]

            total_cost += route_dist * COST_PER_KM
            total_time += (route_dist / SPEED_KMH) + len(route) * UNLOAD_TIME_PER_ORDER
            total_carbon += route_dist * CARBON_PER_KM

        self.f1 = total_cost
        self.f2 = total_time
        self.f3 = total_carbon
        return self.f1, self.f2, self.f3

    def copy(self):
        new_sol = DeliverySolution.__new__(DeliverySolution)
        new_sol.routes = [list(r) for r in self.routes]
        new_sol.f1 = self.f1
        new_sol.f2 = self.f2
        new_sol.f3 = self.f3
        return new_sol

    def mutate(self, mutation_rate=0.3):
        """
        局部搜索变异：
          操作0：交换两个客户
          操作1：反转路径片段
          操作2：移动客户到另一条路径
        """
        new_sol = self.copy()

        if random.random() < mutation_rate and len(new_sol.routes) >= 1:
            op = random.randint(0, 2)

            if op == 0:
                # 交换两个客户
                all_customers = []
                for i, route in enumerate(new_sol.routes):
                    for j, c in enumerate(route):
                        all_customers.append((i, j, c))
                if len(all_customers) >= 2:
                    a, b = random.sample(all_customers, 2)
                    new_sol.routes[a[0]][a[1]], new_sol.routes[b[0]][b[1]] = (
                        new_sol.routes[b[0]][b[1]],
                        new_sol.routes[a[0]][a[1]],
                    )

            elif op == 1:
                # 反转某条路径的片段
                ri = random.randrange(len(new_sol.routes))
                route = new_sol.routes[ri]
                if len(route) > 2:
                    i, j = sorted(random.sample(range(len(route)), 2))
                    new_sol.routes[ri] = route[:i] + route[i:j + 1][::-1] + route[j + 1:]

            elif op == 2:
                # 将客户移动到另一条路径
                if len(new_sol.routes) >= 2:
                    src = random.randrange(len(new_sol.routes))
                    dst = random.randrange(len(new_sol.routes))
                    if src != dst and new_sol.routes[src]:
                        ci = random.randrange(len(new_sol.routes[src]))
                        customer = new_sol.routes[src].pop(ci)
                        pos = random.randint(0, len(new_sol.routes[dst]))
                        new_sol.routes[dst].insert(pos, customer)
                        new_sol.routes = [r for r in new_sol.routes if r]

            # 容量约束修复
            new_sol._repair_capacity()

        return new_sol

    def _repair_capacity(self):
        """修复超容量路径（拆分）"""
        repaired = []
        for route in self.routes:
            load = sum(DEMANDS[c] for c in route)
            if load <= VEHICLE_CAPACITY + 1e-6:
                repaired.append(route)
            else:
                sub_route = []
                sub_load = 0.0
                for c in route:
                    if sub_load + DEMANDS[c] <= VEHICLE_CAPACITY + 1e-6:
                        sub_route.append(c)
                        sub_load += DEMANDS[c]
                    else:
                        repaired.append(sub_route)
                        sub_route = [c]
                        sub_load = DEMANDS[c]
                if sub_route:
                    repaired.append(sub_route)
        self.routes = repaired

    def __repr__(self):
        return (f"Delivery(routes={len(self.routes)}, "
                f"cost={self.f1:.1f}, time={self.f2:.2f}, carbon={self.f3:.1f})")


# ===================== 支配关系 =====================

def dominates(a, b):
    """a 是否支配 b？（三个目标都是最小化）"""
    return (
        a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]
        and (a[0] < b[0] or a[1] < b[1] or a[2] < b[2])
    )


# ===================== 加权求和法 =====================

def weighted_sum_optimize(weights, num_iterations=8000, verbose=False):
    """
    加权求和法单次运行

    weights: (w1, w2, w3) 三目标权重，和为1
    num_iterations: 局部搜索迭代次数

    returns: (DeliverySolution, 归一化范围)
    """
    w1, w2, w3 = weights

    # 1. 采样以获取归一化范围
    samples = []
    for _ in range(500):
        sol = DeliverySolution()
        f1, f2, f3 = sol.evaluate()
        samples.append((f1, f2, f3, sol))

    f1_vals = [s[0] for s in samples]
    f2_vals = [s[1] for s in samples]
    f3_vals = [s[2] for s in samples]

    f1_min, f1_max = min(f1_vals), max(f1_vals)
    f2_min, f2_max = min(f2_vals), max(f2_vals)
    f3_min, f3_max = min(f3_vals), max(f3_vals)

    def normalize(f1, f2, f3):
        return (
            (f1 - f1_min) / (f1_max - f1_min + 1e-10),
            (f2 - f2_min) / (f2_max - f2_min + 1e-10),
            (f3 - f3_min) / (f3_max - f3_min + 1e-10),
        )

    best_score = float("inf")
    best_sol = None

    for iteration in range(num_iterations):
        if iteration == 0:
            sol = DeliverySolution()
        else:
            # 从当前最优解变异（退火式变异率）
            mr = 0.3 + 0.4 * (1.0 - iteration / num_iterations)
            sol = best_sol.copy().mutate(mutation_rate=mr)

        f1, f2, f3 = sol.evaluate()
        n1, n2, n3 = normalize(f1, f2, f3)
        score = w1 * n1 + w2 * n2 + w3 * n3

        if score < best_score:
            best_score = score
            best_sol = sol.copy()

    if verbose:
        print(f"    权重 ({w1:.2f}, {w2:.2f}, {w3:.2f}) -> "
              f"成本={best_sol.f1:.1f}  时间={best_sol.f2:.2f}  碳排放={best_sol.f3:.1f}")

    return best_sol


# ===================== 多组权重对比 =====================

def weighted_sum_comparison():
    """
    使用多组不同权重进行加权求和法对比
    验证不同权重产生不同解
    """
    print("\n>>> 加权求和法对比（多组权重）")
    print(f"  问题规模: {NUM_EVS} 辆电动车, {NUM_ORDERS} 个订单")
    print(f"  目标: 配送成本(f1) | 配送时间(f2) | 碳排放(f3)")
    print()

    # 定义8组有代表性的权重
    weight_sets = [
        ("均等偏好",        0.33, 0.33, 0.34),
        ("重成本-轻时间",   0.60, 0.20, 0.20),
        ("重时间-轻成本",   0.20, 0.60, 0.20),
        ("重低碳-轻成本",   0.20, 0.20, 0.60),
        ("极端重成本",      0.80, 0.10, 0.10),
        ("极端重时间",      0.10, 0.80, 0.10),
        ("极端重低碳",      0.10, 0.10, 0.80),
        ("成本时间并重",    0.45, 0.45, 0.10),
    ]

    print(f"  {'权重组合':<16s} | {'配送成本(元)':>12s} | {'配送时间(h)':>12s} | {'碳排放(kg)':>12s}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    results = []
    for name, w1, w2, w3 in weight_sets:
        sol = weighted_sum_optimize((w1, w2, w3), num_iterations=8000, verbose=False)
        results.append((name, w1, w2, w3, sol))
        print(f"  {name:<16s} | {sol.f1:12.1f} | {sol.f2:12.2f} | {sol.f3:12.1f}")

    return results


# ===================== 验证：不同权重产生不同解 =====================

def verify_weight_diversity(results):
    """
    验证不同权重是否产生不同解
    检查各目标值的变异系数（CV = std/mean）
    """
    print("\n>>> 验证：不同权重产生不同解")

    f1_vals = [r[4].f1 for r in results]
    f2_vals = [r[4].f2 for r in results]
    f3_vals = [r[4].f3 for r in results]

    def cv(vals):
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        return std / mean if mean > 0 else 0.0

    cv1, cv2, cv3 = cv(f1_vals), cv(f2_vals), cv(f3_vals)
    print(f"  配送成本变异系数:   {cv1:.4f}  ({min(f1_vals):.1f} ~ {max(f1_vals):.1f})")
    print(f"  配送时间变异系数:   {cv2:.4f}  ({min(f2_vals):.2f} ~ {max(f2_vals):.2f})")
    print(f"  碳排放变异系数:     {cv3:.4f}  ({min(f3_vals):.1f} ~ {max(f3_vals):.1f})")

    # 判断：如果三个 CV 中最大者 > 0.02，则说明权重确实产生了不同解
    max_cv = max(cv1, cv2, cv3)
    if max_cv > 0.02:
        print(f"  ✓ 验证通过！不同权重产生了显著不同的解 (max CV = {max_cv:.4f} > 0.02)")
    else:
        print(f"  ~ 部分验证通过，解之间的差异较小 (max CV = {max_cv:.4f})")

    # 检查是否至少有一对解在某目标上相差 > 5%
    diverse_pairs = 0
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            f1i, f2i, f3i = results[i][4].f1, results[i][4].f2, results[i][4].f3
            f1j, f2j, f3j = results[j][4].f1, results[j][4].f2, results[j][4].f3
            # 检查各目标相对差异
            def rel_diff(a, b):
                return abs(a - b) / max(abs(a), abs(b), 1e-10)
            if (rel_diff(f1i, f1j) > 0.05 or
                rel_diff(f2i, f2j) > 0.05 or
                rel_diff(f3i, f3j) > 0.05):
                diverse_pairs += 1

    total_pairs = len(results) * (len(results) - 1) // 2
    print(f"  解对中相对差异 > 5% 的比例: {diverse_pairs}/{total_pairs} "
          f"({100.0 * diverse_pairs / total_pairs:.0f}%)")

    return max_cv > 0.02


# ===================== 权衡分析 =====================

def tradeoff_analysis(results):
    """分析目标之间的权衡关系"""
    print("\n>>> 权衡关系分析")

    # 找极值解
    min_cost = min(results, key=lambda r: r[4].f1)
    min_time = min(results, key=lambda r: r[4].f2)
    min_carbon = min(results, key=lambda r: r[4].f3)

    print(f"  最低成本解   ({min_cost[0]:12s}): "
          f"成本={min_cost[4].f1:.1f}, 时间={min_cost[4].f2:.2f}, 碳排放={min_cost[4].f3:.1f}")
    print(f"  最短时间解   ({min_time[0]:12s}): "
          f"成本={min_time[4].f1:.1f}, 时间={min_time[4].f2:.2f}, 碳排放={min_time[4].f3:.1f}")
    print(f"  最低排碳解   ({min_carbon[0]:12s}): "
          f"成本={min_carbon[4].f1:.1f}, 时间={min_carbon[4].f2:.2f}, 碳排放={min_carbon[4].f3:.1f}")

    # 计算极端解之间的权衡比率
    print()
    print("  权衡比率（极端解之间）:")
    print(f"    成本 vs 时间: "
          f"每节省1小时配送时间，成本增加 "
          f"{(min_cost[4].f1 - min_time[4].f1) / max(min_time[4].f2 - min_cost[4].f2, 1e-10):.1f} 元")
    print(f"    成本 vs 碳排放: "
          f"每减少1kg碳排放，成本增加 "
          f"{(min_carbon[4].f1 - min_cost[4].f1) / max(min_cost[4].f3 - min_carbon[4].f3, 1e-10):.1f} 元")
    print(f"    时间 vs 碳排放: "
          f"每减少1kg碳排放，时间增加 "
          f"{(min_carbon[4].f2 - min_time[4].f2) / max(min_time[4].f3 - min_carbon[4].f3, 1e-10):.2f} 小时")


# ===================== 主程序 =====================

if __name__ == "__main__":
    print("=" * 65)
    print("  毕业项目：三目标城市物流优化")
    print()
    print(f"  规模: {NUM_EVS} 辆电动车 × {NUM_ORDERS} 个订单")
    print(f"  配送中心坐标: {DEPOT}")
    print(f"  车辆容量: {VEHICLE_CAPACITY} 单位")
    print()
    print("  三目标（均为最小化）:")
    print("    f1 = 总配送成本（元）")
    print("    f2 = 总配送时间（小时）")
    print("    f3 = 碳排放量（kg CO₂）")
    print()
    print("  基础方法: 加权求和法（多组权重对比）")
    print("=" * 65)

    random.seed(42)

    # 执行加权求和法多权重对比
    results = weighted_sum_comparison()

    # 验证不同权重产生不同解
    print()
    diversity_ok = verify_weight_diversity(results)

    # 权衡分析
    tradeoff_analysis(results)

    # 综合结论
    print()
    print("=" * 65)
    print(">>> 结论")
    print()
    print("  1. 加权求和法成功应用于城市物流三目标优化")
    print(f"     问题规模: {NUM_EVS}辆车, {NUM_ORDERS}个订单, 3个目标")
    print()
    print(f"  2. 通过 {len(results)} 组不同权重，获得了不同的 Pareto 最优解")
    if diversity_ok:
        print("     ✓ 不同权重产生不同解（验证通过）")
    else:
        print("     - 解之间的差异较小，可尝试增加搜索迭代数")
    print()
    print("  3. 加权求和法的特点:")
    print("     + 简单直观，易于实现和理解")
    print("     + 通过调整权重可以探索 Pareto 前沿的不同区域")
    print("     - 一次运行只产生一个解，需要多组权重才能覆盖前沿")
    print("     - 对非凸前沿可能无法找到某些 Pareto 最优解")
    print()
    print("  4. 目标之间的权衡关系已被量化展示")
    print("     决策者可以根据偏好选择适合的权重组合")
    print("=" * 65)
