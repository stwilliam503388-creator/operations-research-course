"""
案例1：外卖骑手路线规划 (VRP)
难度：★★☆☆☆
求解器：HiGHS (免费)
依赖：pip install highspy numpy
运行：python case01_vrp.py

简述：用 MIP 求解 5 个骑手、30 个订单的外卖配送路线，最小化总行驶距离。

技术背景：
  这是一个经典的 VRP (Vehicle Routing Problem)。我们用 MTZ (Miller-Tucker-Zemlin)
  公式来处理子环消除——每条路线上给节点编号，强制编号沿路线递增，
  这样求解器就不能把一条路线切成两个独立的环。
"""

# 教学注释：重点看变量、约束和目标函数如何把业务规则翻译成 MIP 模型。
# 求解日志或结果可用来理解分支定界、松弛和可行解质量。


import highspy
import numpy as np
import time


# ============================================================
# 数据生成（模拟数据）
# ============================================================

def generate_data(n_orders, n_riders, seed=42):
    """
    生成模拟外卖配送数据。

    返回:
        depo: 起始点坐标 (骑手出发点 / 餐站)
        orders: shape (n_orders, 2) — 每个订单的送达坐标
        rider_capacity: 每个骑手最多能同时带的单数（外卖箱容量）
    """
    rng = np.random.default_rng(seed)
    # 出发点设在城市中心
    depo = np.array([5.0, 5.0])
    # 订单随机散布在 10x10 的城区
    orders = rng.uniform(0, 10, size=(n_orders, 2))
    # 骑手外卖箱容量（实际场景中约 6-10 单）
    rider_capacity = int(np.ceil(n_orders / n_riders)) + rng.integers(1, 4)
    return depo, orders, rider_capacity


# ============================================================
# 距离矩阵
# ============================================================

def build_distance_matrix(depo, orders):
    """
    构建距离矩阵 dist[i][j]。
    节点索引: 0 = depo, 1..n = 订单 1..n。
    dist[i][j] = 从节点 i 到节点 j 的欧氏距离。
    """
    # 把所有点拼在一起: [depo, order1, order2, ...]
    points = np.vstack([depo.reshape(1, -1), orders])
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        delta = points[i] - points
        dist[i, :] = np.sqrt(np.sum(delta ** 2, axis=1))
    return dist


# ============================================================
# 辅助：打印路线
# ============================================================

def print_solution(model, x, u, dist, n_orders, n_riders):
    """
    从 MIP 解中提取并打印每条骑手路线。

    参数:
        model: HiGHS 模型对象
        x: dict, key=(i,j,k) 的决策变量
        u: dict, key=(i,k) 的 MTZ 顺序变量
        dist: 距离矩阵
        n_orders, n_riders: 订单数和骑手数
    """
    tol = 0.5  # MIP 解可能有微小浮点误差，x≈1 视为选中

    print(f"\n{'='*60}")
    print("  路线详情")
    print(f"{'='*60}")

    total_dist = 0.0
    for k in range(n_riders):
        # 提取骑手 k 的所有弧
        route_arcs = []
        for i in range(n_orders + 1):
            for j in range(n_orders + 1):
                if i != j:
                    val = model.getSolution(x[i, j, k])
                    if val > tol:
                        route_arcs.append((i, j, val))

        if not route_arcs:
            continue  # 该骑手未被使用

        # 从 depo (0) 出发，沿着弧走
        # 先构建邻接关系
        next_node = {}
        for i, j, _ in route_arcs:
            next_node[i] = j

        # 从 0 出发遍历
        seq = [0]
        current = 0
        while True:
            nxt = next_node.get(current)
            if nxt is None or nxt == 0:
                break
            seq.append(nxt)
            current = nxt
        seq.append(0)  # 回到 depo

        # 计算该骑手行驶距离
        rider_dist = 0.0
        for idx in range(len(seq) - 1):
            rider_dist += dist[seq[idx], seq[idx + 1]]

        total_dist += rider_dist

        # 格式化输出
        seq_str = " → ".join(f"订单{s}" if s != 0 else "餐站" for s in seq)
        print(f"\n骑手 {k+1}: {len(seq)-2} 单 | 行驶 {rider_dist:.1f} 公里")
        print(f"  路线: {seq_str}")

    print(f"\n{'─'*60}")
    print(f"  总行驶距离: {total_dist:.1f} 公里")
    print(f"{'='*60}\n")
    return total_dist


# ============================================================
# 主求解函数
# ============================================================

def solve_vrp(n_riders=5, n_orders=30, seed=42, max_seconds=120):
    """
    用 MIP 求解外卖骑手 VRP 问题。

    参数:
        n_riders: 骑手数量
        n_orders: 订单数量
        seed: 随机种子（保证可复现）
        max_seconds: 求解时间上限（秒）
    """
    # --- 数据准备 ---
    depo, orders, rider_cap = generate_data(n_orders, n_riders, seed)
    dist = build_distance_matrix(depo, orders)

    print(f"\n{'='*60}")
    print(f"  VRP 求解: {n_orders} 个订单, {n_riders} 个骑手")
    print(f"  骑手容量: {rider_cap} 单/人")
    print(f"{'='*60}")

    # --- 模型构建 ---
    model = highspy.Highs()
    # close all printing during solve
    model.setOptionValue("output_flag", False)
    model.setOptionValue("time_limit", max_seconds)

    # 节点集合: 0 = depo, 1..n = 订单
    N = range(n_orders + 1)                # 所有节点
    N_orders = range(1, n_orders + 1)      # 仅订单节点
    K = range(n_riders)                     # 骑手集合

    # ============================================================
    # 决策变量
    # ============================================================
    #
    # x[i,j,k] = 1 表示骑手 k 从节点 i 直接行驶到节点 j，否则 0
    #   i,j ∈ {0,1,...,n}, k ∈ K, i≠j
    #
    # u[i,k] = 骑手 k 在节点 i 的访问顺序号（MTZ 子环消除用）
    #   u[0,k] = 0 (depo 始终是起点)
    #   u[i,k] ∈ [1, n] 当 i ≥ 1
    #
    # 思路 (像对同事说话):
    #   每个决策变量 x[i,j,k] 是一个「是否走这条路」的选择。
    #   u[i,k] 是 MTZ 公式的核心——我们给每单在它所属骑手的路线上
    #   分配一个递增的序号，这样求解器就不能把路线切成两个独立的环。
    #   如果骑手 k 先送 A 再送 B，那么 u[A,k] < u[B,k]。
    #   详情见子环消除约束处的注释。

    n = n_orders

    # --- 添加所有决策变量 (逐个 addVar) ---
    #
    # highspy 最直观的 API: addVar(lb, ub) 返回列索引，
    # 然后 changeColCost / changeColIntegrality 设置属性。
    # 这是逐变量添加方式——变量多时稍慢，但对教学代码来说最清晰。

    x_cols = {}   # (i, j, k) -> col_index
    u_cols = {}   # (i, k)     -> col_index

    # --- x 变量: 弧选择变量 (0-1 整数) ---
    for k in K:
        for i in N:
            for j in N:
                if i == j:
                    continue
                idx = model.addVar(0.0, 1.0)
                # 目标函数系数: 走这条弧的距离
                model.changeColCost(idx, dist[i][j])
                # 标记为整数变量 (上下界 [0,1] 自然形成二元变量)
                model.changeColIntegrality(idx, 1)
                x_cols[(i, j, k)] = idx

    # --- u 变量: MTZ 顺序变量 (连续，辅助用) ---
    for k in K:
        # depo 的 u 固定为 0（上下界都是 0）
        idx = model.addVar(0.0, 0.0)
        model.changeColCost(idx, 0.0)
        u_cols[(0, k)] = idx

        for i in N_orders:
            idx = model.addVar(1.0, float(n))
            model.changeColCost(idx, 0.0)
            u_cols[(i, k)] = idx

    num_vars = model.getNumCol()
    print(f"  变量数: {num_vars}")

    # ============================================================
    # 约束
    # ============================================================

    # highspy 中，addRow(lower, upper, num_nz, col_indices, values)
    # 给模型加一行: lower ≤ Σ(values[s] * x[col_indices[s]]) ≤ upper

    # --- 约束1: 每个订单恰好被一个骑手访问一次 ---
    # Σ_k Σ_j x[i,j,k] = 1   for each order i
    for i in N_orders:
        cols = []
        vals = []
        for k in K:
            for j in N:
                if i == j:
                    continue
                cols.append(x_cols[(i, j, k)])
                vals.append(1.0)
        model.addRow(1.0, 1.0, len(cols), cols, vals)

    # --- 约束2: 每个节点（含 depo）的流量守恒 ---
    # 对每个骑手 k 和每个节点 j: Σ_i x[i,j,k] = Σ_i x[j,i,k]
    # 即流入 = 流出
    for k in K:
        for j in N:
            cols_in = []
            vals_in = []
            cols_out = []
            vals_out = []
            for i in N:
                if i == j:
                    continue
                cols_in.append(x_cols[(i, j, k)])
                vals_in.append(1.0)
                cols_out.append(x_cols[(j, i, k)])
                vals_out.append(-1.0)
            all_cols = cols_in + cols_out
            all_vals = vals_in + vals_out
            model.addRow(0.0, 0.0, len(all_cols), all_cols, all_vals)

    # --- 约束3: 每个骑手从 depo 出发恰好一次（如果被使用）---
    # 用 ≤1 允许骑手不被使用（可送 0 单）
    for k in K:
        cols = []
        vals = []
        # 离开 depo 的弧
        for j in N_orders:
            cols.append(x_cols[(0, j, k)])
            vals.append(1.0)
        # 返回 depo 的弧
        for i in N_orders:
            cols.append(x_cols[(i, 0, k)])
            vals.append(1.0)
        model.addRow(0.0, 2.0, len(cols), cols, vals)

    # --- 约束4: 骑手从 depo 出发的弧数 ≤ 1 ---
    for k in K:
        cols = []
        vals = []
        for j in N_orders:
            cols.append(x_cols[(0, j, k)])
            vals.append(1.0)
        model.addRow(0.0, 1.0, len(cols), cols, vals)

    # --- 约束5: 骑手容量 ---
    # Σ_i Σ_{j≠depo} x[i,j,k] ≤ rider_cap  for each rider k
    # (骑手 k 经过的订单弧总数 = 他送的订单数)
    for k in K:
        cols = []
        vals = []
        for j in N_orders:
            for i in N:
                if i == j:
                    continue
                cols.append(x_cols[(i, j, k)])
                vals.append(1.0)
        model.addRow(0.0, float(rider_cap), len(cols), cols, vals)

    # ============================================================
    # 约束6: MTZ 子环消除约束 ★★★ 核心约束 ★★★
    # ============================================================
    #
    # 为什么需要这个约束？
    # 没有子环消除，求解器可能给出这样的「数学合法但物理荒谬」的解：
    #   骑手1: 餐站 → 订单A → 订单B → 餐站
    #   骑手2: 餐站 → 订单C → 订单D → 餐站
    # 但实际上骑手1可能被切成两个独立环:
    #   环1: 餐站 → 订单A → 订单C → 餐站  (经过订单C但不归骑手2)
    #   环2: 订单B → 订单D → 订单B       (悬空的环!)
    # 求解器看到流量守恒约束全部满足，就说「这可行」——但现实中
    # 骑手不能同时在两个地方，环2完全脱离餐站也不合逻辑。
    #
    # MTZ 公式: 给每个订单在所属骑手路线上一个递增的访问序号。
    #   u[i,k] = 骑手 k 在节点 i 的访问顺序号
    #   如果 x[i,j,k] = 1 (骑手 k 从 i 去 j), 则必须有 u[i,k] + 1 ≤ u[j,k]
    #   用 Big-M 线性化:
    #     u[i,k] - u[j,k] + M * x[i,j,k] ≤ M - 1
    #   当 x[i,j,k]=1 时强制 u[i,k] + 1 ≤ u[j,k] (序号递增)
    #   当 x[i,j,k]=0 时不施加约束 (M 足够大让约束自动满足)
    #
    # Big-M 的选取:
    #   理论上 M ≥ n (最多 n 个节点)。我们用 n+1 作为安全余量。
    #   不要用 999999——过大的 M 会让 LP 松弛非常松，导致 branch-and-bound
    #   效率极低。这里 n=30，M=31 完全够用。

    M = float(n + 1)  # Big-M: 物理上限 = 路径上最多 n 个节点

    mtz_count = 0
    for k in K:
        for i in N_orders:
            for j in N_orders:
                if i == j:
                    continue
                # u[i,k] - u[j,k] + M * x[i,j,k] ≤ M - 1
                cols = [u_cols[(i, k)], u_cols[(j, k)], x_cols[(i, j, k)]]
                vals = [1.0, -1.0, M]
                model.addRow(-highspy.kHighsInf, M - 1, len(cols), cols, vals)
                mtz_count += 1

    print(f"  约束数: {model.getNumRows()}")
    print(f"  其中 MTZ 子环消除约束: {mtz_count} 条")
    print(f"  Big-M 值: {M:.0f}")

    # --- 求解 ---
    print(f"\n  求解中... (时限 {max_seconds}s)")
    start_time = time.time()
    model.run()
    solve_time = time.time() - start_time

    status = model.getModelStatus()
    status_map = {
        highspy.HighsModelStatus.kOptimal: "最优解",
        highspy.HighsModelStatus.kInfeasible: "不可行",
        highspy.HighsModelStatus.kTimeLimit: "超时(次优解)",
        highspy.HighsModelStatus.kIterationLimit: "迭代上限",
    }
    status_str = status_map.get(status, f"状态码: {status}")

    print(f"\n  求解状态: {status_str}")
    print(f"  求解耗时: {solve_time:.1f}s")

    if status not in [highspy.HighsModelStatus.kOptimal, highspy.HighsModelStatus.kTimeLimit]:
        info = model.getInfo()
        print(f"  可能原因: 无法找到可行解。尝试增加骑手容量或减少订单数。")
        return

    obj_val = model.getInfoValue("objective_function_value")
    print(f"  目标值(总距离): {obj_val:.1f} 公里")

    # --- 对比: 人工排单的简单方案 ---
    # 简单贪心: 按距离排序，每个骑手分配最近的一批单
    manual_dist = estimate_manual_total(dist, n_orders, n_riders, rider_cap)
    print(f"  人工排单估算: {manual_dist:.1f} 公里")
    print(f"  优化提升: {100*(1 - obj_val/manual_dist):.1f}%")

    # --- 打印路线 ---
    print_solution(model, x_cols, u_cols, dist, n_orders, n_riders)

    return dict(
        obj_val=obj_val,
        solve_time=solve_time,
        manual_dist=manual_dist,
        n_vars=num_vars,
        n_cons=model.getNumRows(),
        status=status_str,
    )


def estimate_manual_total(dist, n_orders, n_riders, rider_cap):
    """
    估算人工排单的总距离: 每个骑手独立 TSP（最近邻贪心）+ 从 depo 往返。

    这不是真正的 TSP 最优解，而是一个「人手工排单时常见做法」的估计：
    把订单分配给骑手后，每个骑手按最近邻贪心规划自己的路线。
    """
    rng = np.random.default_rng(99)
    # 随机分配订单给骑手
    order_list = list(range(1, n_orders + 1))
    rng.shuffle(order_list)

    # 分配给骑手
    assignments = [[] for _ in range(n_riders)]
    for idx, order in enumerate(order_list):
        k = idx % n_riders
        assignments[k].append(order)

    total = 0.0
    for k, assigned in enumerate(assignments):
        if not assigned:
            continue
        # 从 depo 出发，最近邻贪心
        current = 0  # depo
        remaining = set(assigned)
        while remaining:
            # 找最近的节点
            best_order = min(remaining, key=lambda o: dist[current, o])
            total += dist[current, best_order]
            current = best_order
            remaining.remove(best_order)
        # 回 depo
        total += dist[current, 0]

    return total


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║  案例1：外卖骑手路线规划 (VRP)                              ║
║  求解器: HiGHS  ·  算法: MIP + MTZ 子环消除                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    results = {}

    # --- 小规模测试: 3 骑手 x 8 订单 (几秒内出结果) ---
    print("\n### 小规模测试 (3骑手, 8订单) ###")
    r = solve_vrp(n_riders=3, n_orders=8, seed=123)
    results["小规模 (3x8)"] = r

    # --- 中等规模测试: 5 骑手 x 20 订单 ---
    print("\n### 中等规模测试 (5骑手, 20订单) ###")
    r = solve_vrp(n_riders=5, n_orders=20, seed=123)
    results["中规模 (5x20)"] = r

    # --- 目标规模: 5 骑手 x 30 订单 ---
    print("\n### 目标规模 (5骑手, 30订单) ###")
    r = solve_vrp(n_riders=5, n_orders=30, seed=123, max_seconds=60)
    results["目标规模 (5x30)"] = r

    # --- 结果汇总 ---
    print(f"\n{'='*70}")
    print("  结果汇总")
    print(f"{'='*70}")
    print(f"  {'测试名称':<20s} {'变量数':>8s} {'约束数':>8s} {'目标值':>10s} {'耗时':>8s} {'提升':>8s}")
    print(f"  {'─'*70}")
    for name, r in results.items():
        if r is None:
            continue
        improvement = 100 * (1 - r["obj_val"] / r["manual_dist"])
        print(f"  {name:<20s} {r['n_vars']:>8d} {r['n_cons']:>8d} "
              f"{r['obj_val']:>8.1f}km {r['solve_time']:>6.1f}s {improvement:>7.1f}%")
    print(f"  {'─'*70}")
    print(f"\n  结论: MIP 优化相比人工排单可节省 {results['目标规模 (5x30)']['obj_val']/results['目标规模 (5x30)']['manual_dist']:.0%} 的总距离。")
    print(f"  在实际业务中，这等同于每天少跑十几公里——积少成多，一年下来就是可观的成本节省。")
    print()
