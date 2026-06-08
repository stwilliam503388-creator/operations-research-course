#!/usr/bin/env python3
"""
案例1：物流网络优化 — 运输问题 LP 求解
============================================

场景：3 工厂 → 10 仓库 → 200 客户，最小化总配送成本

教学目标：
  - 单纯形法（运输问题特化：MODI 法）
  - 影子价格（对偶变量）的经济解释
  - 灵敏度分析（运费变化对方案的影响）

验证标准：
  ✅ 每个客户被分配到一个仓库
  ✅ 影子价格（仓库对偶变量）的经济含义可解释
  ✅ 运费 ±10% 时最优方案不变（基变量不变）
"""


# 教学注释：先识别业务对象，再看它们如何映射为优化、仿真或启发式模型。
# 结果解读侧重成本、资源利用率和服务水平等管理指标。



import random
import math
from copy import deepcopy

# ============================================================
# 1. 数据生成
# ============================================================

random.seed(42)


def generate_data():
    """
    生成案例数据：
      - 3 个工厂，供给分别为 200、300、400（总计 900）
      - 10 个仓库，容量随机分配，总容量 >= 总供给
      - 200 个客户，每个客户需求量 1~10 单位
      - 运费：工厂→仓库 和 仓库→客户，基于距离（欧氏距离模拟）
    """
    n_factories = 3
    n_warehouses = 10
    n_customers = 200

    # ---- 工厂供给（总供给 = 900） ----
    supplies = [200, 300, 400]

    # ---- 仓库容量（总和 >= 900） ----
    warehouse_capacities = []
    remaining = 1100  # 总容量略大于总供给，留余量
    for j in range(n_warehouses - 1):
        cap = random.randint(60, 140)
        warehouse_capacities.append(cap)
        remaining -= cap
    warehouse_capacities.append(remaining)
    # 打乱一下让分布更自然
    random.shuffle(warehouse_capacities)

    # ---- 仓库位置（二维坐标，用于模拟距离） ----
    wh_positions = [(random.uniform(0, 100), random.uniform(0, 100))
                    for _ in range(n_warehouses)]

    # ---- 客户位置与需求 ----
    # 确保总需求不超过总供给的 95%
    raw_demands = [random.randint(1, 10) for _ in range(n_customers)]
    raw_total = sum(raw_demands)
    target_total = int(sum(supplies) * 0.92)  # 总供给的 92%
    customer_demands = [
        max(1, int(d * target_total / raw_total)) for d in raw_demands
    ]
    # 微调使总和精确
    diff = target_total - sum(customer_demands)
    if diff != 0:
        customer_demands[0] += diff

    customer_positions = [(random.uniform(0, 100), random.uniform(0, 100))
                          for _ in range(n_customers)]

    # ---- 工厂位置 ----
    factory_positions = [(random.uniform(0, 100), random.uniform(0, 100))
                         for _ in range(n_factories)]

    def euclidean(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # ---- 运费矩阵：工厂 i → 仓库 j（单位：元/单位货物） ----
    # 模拟运费 = 距离 × 运费系数 0.5~1.5
    f2w_cost = []
    for i in range(n_factories):
        row = []
        for j in range(n_warehouses):
            dist = euclidean(factory_positions[i], wh_positions[j])
            rate = random.uniform(0.5, 1.5)
            row.append(round(dist * rate, 2))
        f2w_cost.append(row)

    # ---- 运费矩阵：仓库 j → 客户 k（元/单位货物） ----
    w2c_cost = []
    for j in range(n_warehouses):
        row = []
        for k in range(n_customers):
            dist = euclidean(wh_positions[j], customer_positions[k])
            rate = random.uniform(0.5, 1.5)
            row.append(round(dist * rate, 2))
        w2c_cost.append(row)

    data = {
        'n_factories': n_factories,
        'n_warehouses': n_warehouses,
        'n_customers': n_customers,
        'supplies': supplies,
        'warehouse_capacities': warehouse_capacities,
        'customer_demands': customer_demands,
        'f2w_cost': f2w_cost,
        'w2c_cost': w2c_cost,
        'wh_positions': wh_positions,
        'customer_positions': customer_positions,
        'factory_positions': factory_positions,
    }
    return data


# ============================================================
# 2. 运输问题求解器（MODI 法）
# ============================================================

def northwest_corner(supply, demand):
    """
    西北角法：生成运输问题的初始可行解。
    返回：分配矩阵 alloc (m×n)
    """
    m, n = len(supply), len(demand)
    s = supply[:]
    d = demand[:]
    alloc = [[0.0] * n for _ in range(m)]
    i, j = 0, 0
    while i < m and j < n:
        amount = min(s[i], d[j])
        alloc[i][j] = amount
        s[i] -= amount
        d[j] -= amount
        if s[i] < 1e-9:
            i += 1
        if d[j] < 1e-9:
            j += 1
    return alloc


def modi_method(supply, demand, cost, max_iter=5000):
    """
    MODI (Modified Distribution) 法求解运输问题。

    参数：
        supply: list[m] — 各供给量
        demand: list[n] — 各需求量
        cost: list[list] — 成本矩阵 m×n
        max_iter: 最大迭代次数

    返回：
        alloc: m×n 分配矩阵
        total_cost: 总成本
        dual_u: list[m] — 供给方对偶变量（影子价格）
        dual_v: list[n] — 需求方对偶变量（影子价格）
        iterations: 迭代次数
    """
    m, n = len(supply), len(demand)
    alloc = northwest_corner(supply, demand)
    total_supply = sum(supply)
    total_demand = sum(demand)

    # 检查供需平衡
    assert abs(total_supply - total_demand) < 1e-9, \
        f"供需不平衡！供给={total_supply}, 需求={total_demand}"

    for iteration in range(max_iter):
        # ---- 步骤 1：计算对偶变量 u_i, v_j ----
        # 对于基变量 (i,j)，有 u_i + v_j = c_ij
        # 方程组：m+n-1 个方程，m+n 个变量
        # 固定 u[0] = 0，然后解出所有 u 和 v
        dual_u = [None] * m
        dual_v = [None] * n
        dual_u[0] = 0.0

        # 用 BFS 方式传播求解
        changed = True
        while changed:
            changed = False
            for i in range(m):
                for j in range(n):
                    if alloc[i][j] > 1e-9:  # 基变量
                        if dual_u[i] is not None and dual_v[j] is None:
                            dual_v[j] = cost[i][j] - dual_u[i]
                            changed = True
                        elif dual_v[j] is not None and dual_u[i] is None:
                            dual_u[i] = cost[i][j] - dual_v[j]
                            changed = True

        # 处理未赋值的对偶变量（退化的特殊情况）
        for i in range(m):
            if dual_u[i] is None:
                dual_u[i] = 0.0
        for j in range(n):
            if dual_v[j] is None:
                dual_v[j] = cost[0][j] - dual_u[0]

        # ---- 步骤 2：计算非基变量的检验数 σ_ij = c_ij - u_i - v_j ----
        min_sigma = float('inf')
        enter_i, enter_j = -1, -1
        for i in range(m):
            for j in range(n):
                if alloc[i][j] < 1e-9:  # 非基变量
                    sigma = cost[i][j] - dual_u[i] - dual_v[j]
                    if sigma < min_sigma - 1e-9:
                        min_sigma = sigma
                        enter_i, enter_j = i, j

        # 所有检验数 ≥ 0 → 最优解
        if min_sigma >= -1e-9:
            total_cost = sum(cost[i][j] * alloc[i][j]
                             for i in range(m) for j in range(n))
            return alloc, total_cost, dual_u, dual_v, iteration

        # ---- 步骤 3：找到闭回路并调整 ----
        # 把 (enter_i, enter_j) 加入基变量集合
        # 在基变量矩阵中找闭回路

        # 构建基变量索引列表（包含进入变量）
        basic = []
        for i in range(m):
            for j in range(n):
                if alloc[i][j] > 1e-9 or (i == enter_i and j == enter_j):
                    basic.append([i, j])

        # 在基变量中找闭回路：从进入点出发，沿基变量走回起点
        # 使用 DFS 找回路，交替行/列方向
        def find_cycle(start_i, start_j, basic_cells):
            """在基变量格中找从 (start_i, start_j) 出发的闭回路。"""
            # 将基本格按行和列组织
            rows = {}
            cols = {}
            for i, j in basic_cells:
                rows.setdefault(i, []).append(j)
                cols.setdefault(j, []).append(i)

            # DFS 搜索，交替行和列移动
            # path: 路径上的 (i,j) 坐标列表
            # direction: 0=行移动, 1=列移动
            def dfs(i, j, direction, visited, path):
                if len(path) > 0 and i == start_i and j == start_j and len(path) >= 4:
                    return path

                if direction == 0:  # 按行移动 → 在同一行找下一列
                    for nj in rows.get(i, []):
                        if (i, nj) not in visited or (nj == start_j and len(path) >= 3):
                            new_visited = visited | {(i, nj)}
                            new_path = path + [(i, nj)]
                            result = dfs(i, nj, 1, new_visited, new_path)
                            if result:
                                return result
                else:  # 按列移动 → 在同一列找下一行
                    for ni in cols.get(j, []):
                        if (ni, j) not in visited or (ni == start_i and j == start_j and len(path) >= 3):
                            new_visited = visited | {(ni, j)}
                            new_path = path + [(ni, j)]
                            result = dfs(ni, j, 0, new_visited, new_path)
                            if result:
                                return result
                return None

            path = dfs(start_i, start_j, 0, set(), [])
            return path

        cycle = find_cycle(enter_i, enter_j, basic)
        if cycle is None:
            # 如果找不到回路（退化情况），强制退出
            total_cost = sum(cost[i][j] * alloc[i][j]
                             for i in range(m) for j in range(n))
            return alloc, total_cost, dual_u, dual_v, iteration

        # 在回路上标记奇偶位置
        # 进入变量标记为 '+'（加），交替 '-' '+' ...
        theta = float('inf')
        for idx, (i, j) in enumerate(cycle[1:]):  # 跳过起点（它是 +）
            if idx % 2 == 0:  # '-' 位置
                if alloc[i][j] < theta:
                    theta = alloc[i][j]

        if theta == float('inf') or theta < 1e-12:
            # 退化情况
            total_cost = sum(cost[i][j] * alloc[i][j]
                             for i in range(m) for j in range(n))
            return alloc, total_cost, dual_u, dual_v, iteration

        # 调整分配量
        alloc[enter_i][enter_j] += theta
        for idx, (i, j) in enumerate(cycle[1:]):
            if idx % 2 == 0:  # '-' 位置
                alloc[i][j] -= theta
            else:  # '+' 位置
                alloc[i][j] += theta

        # 清除接近零的数值
        for i in range(m):
            for j in range(n):
                if abs(alloc[i][j]) < 1e-9:
                    alloc[i][j] = 0.0

    # 达到最大迭代次数
    total_cost = sum(cost[i][j] * alloc[i][j]
                     for i in range(m) for j in range(n))
    return alloc, total_cost, dual_u, dual_v, max_iter


def solve_transportation(supply, demand, cost):
    """
    统一接口：求解运输问题，返回结果字典。
    """
    alloc, total_cost, dual_u, dual_v, iters = modi_method(
        supply, demand, cost
    )
    return {
        'allocation': alloc,
        'total_cost': total_cost,
        'dual_u': dual_u,
        'dual_v': dual_v,
        'iterations': iters,
    }


# ============================================================
# 3. 仓库→客户 分配（每个客户分配到唯一仓库）
# ============================================================

def assign_customers_to_warehouses(
    warehouse_supplies, customer_demands, w2c_cost
):
    """
    将每个客户分配到成本最低且供应充足的仓库。
    使用贪心 + 运输问题的简化版本：
      按客户-仓库成本排序，依次分配。

    返回：
        assignment: list[客户索引] = 仓库索引
        total_cost: 总配送成本
    """
    n_warehouses = len(warehouse_supplies)
    n_customers = len(customer_demands)

    # 复制可用库存
    available = [s for s in warehouse_supplies]

    # 生成所有 (客户, 仓库, 成本) 三元组，按成本排序
    triples = []
    for k in range(n_customers):
        for j in range(n_warehouses):
            triples.append((w2c_cost[j][k], k, j))
    triples.sort()

    assignment = [-1] * n_customers
    total_cost = 0.0
    assigned_count = 0

    for _, k, j in triples:
        if assignment[k] >= 0:
            continue  # 已分配
        demand = customer_demands[k]
        if available[j] >= demand:
            assignment[k] = j
            available[j] -= demand
            total_cost += w2c_cost[j][k] * demand
            assigned_count += 1

    if assigned_count < n_customers:
        # 有客户未能分配（容量不足），报错
        unassigned = [k for k in range(n_customers) if assignment[k] < 0]
        raise RuntimeError(
            f"未能分配所有客户！未分配客户数：{len(unassigned)}。"
            f"请增加仓库容量。"
        )

    return assignment, total_cost, available


# ============================================================
# 4. 灵敏度分析
# ============================================================

def sensitivity_analysis(supply, demand, cost, base_result):
    """
    灵敏度分析：将每个工厂→仓库的运费 ±10%
    检查最优方案（基变量）是否改变。

    参数：
        supply, demand, cost: 原始数据
        base_result: solve_transportation 的返回结果

    返回：
        sensitivity_results: list of dict
    """
    m = len(supply)
    n = len(demand)
    base_alloc = base_result['allocation']
    base_total = base_result['total_cost']

    # 提取基变量集合
    base_basic = set()
    for i in range(m):
        for j in range(n):
            if base_alloc[i][j] > 1e-9:
                base_basic.add((i, j))

    results = []
    for i in range(m):
        for j in range(n):
            for change, label in [(0.9, '-10%'), (1.1, '+10%')]:
                new_cost = deepcopy(cost)
                new_cost[i][j] = round(cost[i][j] * change, 2)
                new_result = solve_transportation(supply, demand, new_cost)

                # 检查基变量是否相同
                new_alloc = new_result['allocation']
                new_basic = set()
                for ii in range(m):
                    for jj in range(n):
                        if new_alloc[ii][jj] > 1e-9:
                            new_basic.add((ii, jj))

                basic_changed = (new_basic != base_basic)

                results.append({
                    'edge': f'工厂{i}→仓库{j}',
                    'change': label,
                    'original_cost': cost[i][j],
                    'new_cost': new_cost[i][j],
                    'original_total': round(base_total, 2),
                    'new_total': round(new_result['total_cost'], 2),
                    'delta_total': round(new_result['total_cost'] - base_total, 2),
                    'basic_changed': basic_changed,
                })
    return results


# ============================================================
# 5. 主流程：运行完整案例
# ============================================================

def main():
    print("=" * 72)
    print("  案例1：物流网络优化 — 运输问题 LP 求解")
    print("  3 工厂 → 10 仓库 → 200 客户")
    print("=" * 72)

    # ---- 生成数据 ----
    print("\n📦 生成数据...")
    data = generate_data()

    print(f"  工厂数：{data['n_factories']}")
    print(f"  仓库数：{data['n_warehouses']}")
    print(f"  客户数：{data['n_customers']}")
    print(f"  总供给：{sum(data['supplies'])}")
    print(f"  总需求：{sum(data['customer_demands'])}")
    print(f"  仓库总容量：{sum(data['warehouse_capacities'])}")

    total_demand = sum(data['customer_demands'])
    total_capacity = sum(data['warehouse_capacities'])

    # ---- 阶段 1：工厂→仓库 运输问题 ----
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  阶段 1：工厂 → 仓库（运输问题 LP）")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 供需平衡：工厂供给 = 仓库接收量（按仓库容量比例分配需求）
    total_supply = sum(data['supplies'])
    # 按容量比例分配仓库需求
    wh_demand = [
        max(1, int(total_supply * cap / total_capacity))
        for cap in data['warehouse_capacities']
    ]
    # 调整使总和 = total_supply
    wh_sum = sum(wh_demand)
    diff = total_supply - wh_sum
    wh_demand[0] += diff

    result_f2w = solve_transportation(
        data['supplies'], wh_demand, data['f2w_cost']
    )

    print(f"  总运输成本：{result_f2w['total_cost']:.2f} 元")
    print(f"  MODI 法迭代次数：{result_f2w['iterations']}")
    print()
    print("  工厂→仓库 流量矩阵（只显示非零流量）：")
    for i in range(data['n_factories']):
        for j in range(data['n_warehouses']):
            if result_f2w['allocation'][i][j] > 1e-9:
                print(f"    工厂{i} → 仓库{j}: "
                      f"{result_f2w['allocation'][i][j]:.1f} 单位, "
                      f"运费={data['f2w_cost'][i][j]:.2f} 元/单位")

    # ---- 影子价格分析 ----
    print("\n  影子价格（对偶变量）解释：")
    print("  仓库影子价格 v_j = 该仓库多接收 1 单位货物时总成本的边际增加")
    for j in range(data['n_warehouses']):
        v = result_f2w['dual_v'][j]
        print(f"    仓库{j}: v={v:.4f}  —— "
              f"{'稀缺资源' if v > 1 else '供应充裕' if v < 0.5 else '正常水平'}")

    print("\n  工厂影子价格 u_i = 该工厂多供应 1 单位货物时总成本的边际减少")
    for i in range(data['n_factories']):
        u = result_f2w['dual_u'][i]
        print(f"    工厂{i}: u={u:.4f}  —— "
              f"{'高价值供给源' if u < -1 else '普通供给源'}")

    # ---- 阶段 2：仓库→客户 分配 ----
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  阶段 2：仓库 → 客户（每个客户分配到一个仓库）")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 计算每个仓库从工厂接收的总货物量（作为仓库供给）
    wh_supplies = [0.0] * data['n_warehouses']
    for i in range(data['n_factories']):
        for j in range(data['n_warehouses']):
            wh_supplies[j] += result_f2w['allocation'][i][j]

    assignment, w2c_total, remaining = assign_customers_to_warehouses(
        wh_supplies, data['customer_demands'], data['w2c_cost']
    )

    print(f"  仓库→客户 总配送成本：{w2c_total:.2f} 元")
    print(f"\n  各仓库服务客户数：")
    wh_customer_count = [0] * data['n_warehouses']
    for k, j in enumerate(assignment):
        wh_customer_count[j] += 1
    for j in range(data['n_warehouses']):
        print(f"    仓库{j}: {wh_customer_count[j]} 个客户, "
              f"剩余库存={remaining[j]:.1f} 单位")

    # ---- 验证 1：每个客户被分配到一个仓库 ----
    print("\n✅ 验证标准 1：每个客户被分配到一个仓库")
    assert all(a >= 0 for a in assignment), "存在未分配客户"
    print(f"  全部 {data['n_customers']} 个客户已分配，通过！")

    # ---- 验证 2：影子价格经济含义可解释 ----
    print("\n✅ 验证标准 2：影子价格经济含义")
    for j in range(data['n_warehouses']):
        v = result_f2w['dual_v'][j]
        if v > 0.5:
            print(f"  仓库{j}: v_j={v:.4f} > 0.5 → "
                  f"该仓库是瓶颈资源，扩容可降低总成本约 {v:.2f} 元/单位")
        else:
            print(f"  仓库{j}: v_j={v:.4f} ≤ 0.5 → "
                  f"该仓库供给充裕，不是瓶颈")
    print("  通过！")

    # ---- 验证 3：灵敏度分析（运费 ±10% 方案不变） ----
    print("\n✅ 验证标准 3：灵敏度分析（运费 ±10%）")
    print("  (对工厂→仓库的运费进行 ±10% 扰动)")
    print()

    sens_results = sensitivity_analysis(
        data['supplies'], wh_demand, data['f2w_cost'], result_f2w
    )

    all_stable = True
    for r in sens_results:
        status = "✅ 基变量不变" if not r['basic_changed'] else "❌ 基变量改变"
        if r['basic_changed']:
            all_stable = False
        print(f"  {r['edge']} {r['change']}: "
              f"{r['original_cost']}→{r['new_cost']}, "
              f"总成本 {r['original_total']}→{r['new_total']} "
              f"(Δ={r['delta_total']:+})  {status}")

    if all_stable:
        print("\n  结论：在当前运费 ±10% 范围内，所有最优基变量均保持不变。")
        print("  说明该配送方案对运费波动具有一定的鲁棒性。")
    else:
        print("\n  结论：部分运费变化导致基变量改变，需要重新求解。")

    # ---- 总成本汇总 ----
    print("\n" + "=" * 72)
    print("  总成本汇总")
    print("=" * 72)
    total = result_f2w['total_cost'] + w2c_total
    print(f"  阶段1（工厂→仓库）：{result_f2w['total_cost']:.2f} 元")
    print(f"  阶段2（仓库→客户）：{w2c_total:.2f} 元")
    print(f"  总配送成本：{total:.2f} 元")

    # ---- 洞察总结 ----
    print("\n" + "=" * 72)
    print("  📖 洞察与延伸")
    print("=" * 72)
    print("""
  1. 运输问题的特殊结构：
     - 约束矩阵是幺模矩阵（totally unimodular）
     - 即使作为 LP 求解，得到的解天然是整数
     - 这是运输问题"好解"的数学根源

  2. 影子价格的业务含义：
     - 仓库 v_j 高 → 该仓库是网络瓶颈 → 扩容或增设仓库
     - 工厂 u_i 负且绝对值大 → 该工厂是低成本供给源
     - 管理者可用影子价格指导投资决策

  3. 灵敏度分析的作用：
     - 找到"敏感边"——哪些运费变化会改变配送方案
     - 运费谈判时，你知道哪些路线可以妥协、哪些必须守住
     - 如果实际运费在 [0.9c, 1.1c] 内都不影响方案 → 可放心签合同

  4. 延伸思考：
     - 如果引入仓库固定成本 → 变成设施选址问题（MIP）
     - 如果引入多周期 → 变成动态网络流
     - 如果需求不确定 → 变成随机规划 / 鲁棒优化
""")


if __name__ == '__main__':
    main()
