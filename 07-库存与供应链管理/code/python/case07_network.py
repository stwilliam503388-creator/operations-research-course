"""
case07_network.py — 供应链网络设计（简化版选址模型）
=====================================================
演示内容：
1. N 个候选仓库选址，每个有固定成本 + 运输成本
2. 全开策略 vs 最优选址策略
3. 运输成本计算（距离 × 需求量 × 运输费率）
4. 成本构成分析（固定 vs 可变）

使用暴力枚举（N 较小）或贪心法寻找最优选址子集

仅使用 Python 标准库（math, random）
"""


# 教学注释：围绕订货、库存、契约和网络配置观察供应链决策变量。
# 重点比较成本、缺货风险与服务水平之间的权衡。



import math
import random
import itertools


# ============================================================
# 1. 数据结构
# ============================================================

class Customer:
    """客户点"""
    def __init__(self, cid: int, x: float, y: float, demand: float):
        self.id = cid
        self.x = x
        self.y = y
        self.demand = demand

    def __repr__(self):
        return f"客户{cid}({self.x},{self.y},需求={self.demand})"


class Warehouse:
    """候选仓库"""
    def __init__(self, wid: int, x: float, y: float,
                 fixed_cost: float, capacity: float):
        self.id = wid
        self.x = x
        self.y = y
        self.fixed_cost = fixed_cost   # 固定成本（建设/运营）
        self.capacity = capacity       # 容量

    def __repr__(self):
        return f"仓库{wid}({self.x},{self.y},固定={self.fixed_cost},容量={self.capacity})"


# ============================================================
# 2. 距离与运输成本
# ============================================================

def euclidean_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """欧氏距离"""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def transport_cost(warehouse: Warehouse, customer: Customer,
                   rate: float = 1.0) -> float:
    """
    从仓库到客户的运输成本

    运输成本 = 距离 × 需求量 × 运输费率
    """
    dist = euclidean_dist(warehouse.x, warehouse.y, customer.x, customer.y)
    return dist * customer.demand * rate


# ============================================================
# 3. 选址方案评估
# ============================================================

def evaluate_solution(selected_whs: list, all_warehouses: list,
                      customers: list, transport_rate: float = 1.0) -> dict:
    """
    评估一个选址方案

    每个客户分配给距离最近的所选仓库（如果容量允许；简化版容量暂不检查）

    返回:
        {
            "total_cost": 总成本,
            "fixed_cost": 总固定成本,
            "transport_cost": 总运输成本,
            "assignment": {客户id: 仓库id},
            "open_whs": 开启的仓库列表,
        }
    """
    selected = [all_warehouses[i] for i in selected_whs]

    total_fixed = sum(w.fixed_cost for w in selected)
    total_transport = 0.0
    assignment = {}

    for cust in customers:
        # 找最近的选中仓库
        best_wh = None
        best_cost = float('inf')
        for wh in selected:
            cost = transport_cost(wh, cust, transport_rate)
            if cost < best_cost:
                best_cost = cost
                best_wh = wh
        total_transport += best_cost
        assignment[cust.id] = best_wh.id

    total_cost = total_fixed + total_transport

    return {
        "total_cost": total_cost,
        "fixed_cost": total_fixed,
        "transport_cost": total_transport,
        "assignment": assignment,
        "open_whs": selected_whs,
        "num_open": len(selected_whs),
    }


# ============================================================
# 4. 最优选址搜索（暴力枚举 + 贪心）
# ============================================================

def brute_force_best(all_warehouses: list, customers: list,
                     transport_rate: float = 1.0,
                     max_combinations: int = 10000) -> dict:
    """
    暴力枚举所有可能（N 较小时使用）

    返回最优方案
    """
    n = len(all_warehouses)
    best_solution = None
    best_cost = float('inf')

    # 枚举所有非空子集
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            result = evaluate_solution(list(combo), all_warehouses, customers, transport_rate)
            if result["total_cost"] < best_cost:
                best_cost = result["total_cost"]
                best_solution = result

    return best_solution


def greedy_best(all_warehouses: list, customers: list,
                transport_rate: float = 1.0) -> dict:
    """
    贪心法：逐步添加最有价值的仓库

    1. 从空集开始
    2. 每次添加一个使总成本降低最多的仓库
    3. 直到添加任何仓库都不再降低成本
    """
    n = len(all_warehouses)
    selected = []
    current_cost = float('inf')

    while True:
        best_new = None
        best_new_cost = current_cost
        for i in range(n):
            if i in selected:
                continue
            test_set = selected + [i]
            result = evaluate_solution(test_set, all_warehouses, customers, transport_rate)
            if result["total_cost"] < best_new_cost:
                best_new_cost = result["total_cost"]
                best_new = i
        if best_new is None or best_new_cost >= current_cost:
            break
        selected.append(best_new)
        current_cost = best_new_cost

    return evaluate_solution(selected, all_warehouses, customers, transport_rate)


# ============================================================
# 5. 文本可视化
# ============================================================

def print_network_map(all_warehouses: list, customers: list,
                      solution: dict, grid_size: int = 30) -> str:
    """
    文本方式画出网络分布图

    图例：
        W = 开启的仓库
        w = 未开启的候选仓库
        C = 客户
    """
    # 收集所有点坐标
    all_x = [w.x for w in all_warehouses] + [c.x for c in customers]
    all_y = [w.y for w in all_warehouses] + [c.y for c in customers]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    x_span = max_x - min_x if max_x != min_x else 1.0
    y_span = max_y - min_y if max_y != min_y else 1.0

    lines = []
    lines.append("=" * (grid_size + 4))
    lines.append("  供应链网络分布图")
    lines.append("=" * (grid_size + 4))
    lines.append("  图例: W=开启仓库  w=未开启  C=客户")
    lines.append("")

    # 创建网格
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]

    # 放置客户
    for cust in customers:
        cx = int((cust.x - min_x) / x_span * (grid_size - 1))
        cy = int((cust.y - min_y) / y_span * (grid_size - 1))
        grid[cy][cx] = 'C'

    # 放置仓库
    open_set = set(solution["open_whs"])
    for i, wh in enumerate(all_warehouses):
        wx = int((wh.x - min_x) / x_span * (grid_size - 1))
        wy = int((wh.y - min_y) / y_span * (grid_size - 1))
        ch = 'W' if i in open_set else 'w'
        grid[wy][wx] = ch

    # 输出网格
    lines.append("  +" + "-" * grid_size + "+")
    for row in grid:
        lines.append("  |" + "".join(row) + "|")
    lines.append("  +" + "-" * grid_size + "+")

    # 成本摘要
    lines.append("")
    lines.append(f"  开启仓库数: {solution['num_open']}")
    lines.append(f"  总固定成本: {solution['fixed_cost']:.2f}")
    lines.append(f"  总运输成本: {solution['transport_cost']:.2f}")
    lines.append(f"  总成本:     {solution['total_cost']:.2f}")

    # 分配信息
    lines.append("")
    lines.append("  客户分配:")
    for cust in customers:
        wh_id = solution["assignment"][cust.id]
        wh = all_warehouses[wh_id]
        dist = euclidean_dist(cust.x, cust.y, wh.x, wh.y)
        lines.append(f"    客户{cust.id} (需求{cust.demand:.0f}) → 仓库{wh_id} (距离{dist:.1f})")

    lines.append("=" * (grid_size + 4))
    return "\n".join(lines)


# ============================================================
# 6. 主程序
# ============================================================

def main():
    """主函数：演示供应链网络设计"""
    random.seed(42)

    print("\n" + "★" * 50)
    print("  供应链网络设计（仓库选址模型）演示")
    print("★" * 50)

    # ---- 生成问题数据 ----
    N_WAREHOUSES = 8   # 候选仓库数
    N_CUSTOMERS = 12   # 客户数
    TRANSPORT_RATE = 0.5  # 运输费率（每单位距离每单位需求）

    print(f"\n▶ 问题设定")
    print(f"   候选仓库数: {N_WAREHOUSES}")
    print(f"   客户数:     {N_CUSTOMERS}")
    print(f"   运输费率:   {TRANSPORT_RATE}")

    # 生成客户（在 [0, 100] x [0, 100] 区域内）
    customers = []
    for i in range(N_CUSTOMERS):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        demand = random.uniform(50, 500)
        customers.append(Customer(i, x, y, demand))

    # 生成候选仓库
    warehouses = []
    for i in range(N_WAREHOUSES):
        x = random.uniform(10, 90)
        y = random.uniform(10, 90)
        # 固定成本在 100~500 之间，与位置有关（中心位置更贵）
        center_dist = math.sqrt((x - 50)**2 + (y - 50)**2)
        fixed_cost = 200 + 300 * (1 - center_dist / 70)  # 越中心越贵
        capacity = 2000  # 简化，统一容量
        warehouses.append(Warehouse(i, x, y, fixed_cost, capacity))

    # 打印客户和仓库信息
    print("\n▶ 客户信息")
    print("-" * 80)
    for c in customers:
        print(f"   客户{c.id:>2}: ({c.x:>6.2f}, {c.y:>6.2f})  需求={c.demand:>7.2f}")
    print("\n▶ 候选仓库信息")
    print("-" * 80)
    for w in warehouses:
        print(f"   仓库{w.id:>2}: ({w.x:>6.2f}, {w.y:>6.2f})  固定成本={w.fixed_cost:>8.2f}  容量={w.capacity}")

    # ---- (1) 全开策略 ----
    print("\n▶ 1. 全开策略（所有候选仓库都建）")
    print("-" * 70)
    all_open = evaluate_solution(list(range(N_WAREHOUSES)), warehouses, customers, TRANSPORT_RATE)
    print(f"   开启仓库: {N_WAREHOUSES} 个")
    print(f"   固定成本:     {all_open['fixed_cost']:>10.2f}")
    print(f"   运输成本:     {all_open['transport_cost']:>10.2f}")
    print(f"   总成本:       {all_open['total_cost']:>10.2f}")

    # ---- (2) 最优选址（贪心法） ----
    print("\n▶ 2. 最优选址策略（贪心法）")
    print("-" * 70)
    best_greedy = greedy_best(warehouses, customers, TRANSPORT_RATE)
    print(f"   开启仓库: {best_greedy['num_open']} 个")
    print(f"   开启仓库列表: {best_greedy['open_whs']}")
    print(f"   固定成本:     {best_greedy['fixed_cost']:>10.2f}")
    print(f"   运输成本:     {best_greedy['transport_cost']:>10.2f}")
    print(f"   总成本:       {best_greedy['total_cost']:>10.2f}")

    # 对比
    savings = all_open["total_cost"] - best_greedy["total_cost"]
    saving_pct = (savings / all_open["total_cost"]) * 100
    print(f"\n   对比全开策略:")
    print(f"   成本节约: {savings:.2f} ({saving_pct:.2f}%)")

    # ---- (3) 暴力枚举最优（N 较小） ----
    print("\n▶ 3. 暴力枚举验证最优解（N较小时适用）")
    print("-" * 70)
    best_brute = brute_force_best(warehouses, customers, TRANSPORT_RATE)
    print(f"   最优仓库数: {best_brute['num_open']}")
    print(f"   最优选择:   {best_brute['open_whs']}")
    print(f"   最优总成本: {best_brute['total_cost']:.2f}")
    print(f"   贪心法成本: {best_greedy['total_cost']:.2f}")

    diff = abs(best_brute["total_cost"] - best_greedy["total_cost"])
    if diff < 0.01:
        print(f"   ✅ 贪心法找到全局最优解！")
    else:
        print(f"   ⚠️  贪心法差距: {diff:.2f}（非全局最优，但接近）")

    # ---- (4) 网络可视化 ----
    print("\n▶ 4. 网络拓扑图 （最优方案）")
    print(print_network_map(warehouses, customers, best_greedy))

    # ---- (5) 仓库数量与总成本的关系 ----
    print("\n▶ 5. 仓库数量与总成本的关系")
    print("-" * 70)
    print(f"{'仓库数':>8} | {'最优总成本':>12} | {'固定成本':>12} | {'运输成本':>12} | {'边际效益':>10}")
    print("-" * 70)

    prev_cost = float('inf')
    for k in range(1, N_WAREHOUSES + 1):
        # 对固定数量 k 搜索最优组合
        best_k_cost = float('inf')
        best_k_sol = None
        for combo in itertools.combinations(range(N_WAREHOUSES), k):
            r = evaluate_solution(list(combo), warehouses, customers, TRANSPORT_RATE)
            if r["total_cost"] < best_k_cost:
                best_k_cost = r["total_cost"]
                best_k_sol = r
        marginal = prev_cost - best_k_cost if prev_cost != float('inf') else 0
        print(f"{k:>8} | {best_k_cost:>12.2f} | {best_k_sol['fixed_cost']:>12.2f} | "
              f"{best_k_sol['transport_cost']:>12.2f} | {marginal:>10.2f}")
        prev_cost = best_k_cost

    print("\n" + "★" * 50)
    print("  供应链网络设计演示完毕")
    print("★" * 50 + "\n")


if __name__ == "__main__":
    main()
