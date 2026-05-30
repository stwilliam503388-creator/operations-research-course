"""
案例2：工厂生产排程 (Job Shop Scheduling)
难度：★★★☆☆
求解器：HiGHS
依赖：pip install highspy numpy
运行：python case02_scheduling.py

简述：用析取约束 MIP 求解 10 批次 5 台机器的生产排程问题，
      对比 Big-M 紧值 vs 松值的性能差异，展示对称性破坏的加速效果。

技术背景：
  这是一个经典的 JSSP (Job Shop Scheduling Problem)。我们用析取建模
  (Disjunctive Programming) 来处理机器互斥——对每台机器上的每对批次，
  引入 0-1 变量 y[j,k,m]，「要么 j 在 k 前，要么 k 在 j 前」。
  Big-M 值用物理上界 (总加工时间之和) 而非魔法数字 999999。
"""

# 教学注释：重点看变量、约束和目标函数如何把业务规则翻译成 MIP 模型。
# 求解日志或结果可用来理解分支定界、松弛和可行解质量。


import highspy
import numpy as np
import time
from collections import defaultdict


# ============================================================
# 数据生成（模拟半导体厂光刻区数据）
# ============================================================

def generate_instance(n_jobs=10, n_machines=5, seed=42):
    """
    生成模拟半导体厂光刻区排程实例。

    每个批次 (job) 有一条工艺路线 (route)：依次经过若干台机器。
    不同批次可能走不同的路线（对应不同的产品类型）。

    参数:
        n_jobs: 批次数
        n_machines: 机器数
        seed: 随机种子

    返回:
        routes: list[list[int]] — routes[j] 是批次 j 经过的机器顺序
        p: dict — p[(j,m)] = 批次 j 在机器 m 上的处理时间
        setup: dict — setup[(j,k,m)] = 机器 m 上从批次 j 切换到批次 k 的换线时间
        product_types: list[int] — product_types[j] = 批次 j 的产品类型 (1,2,3)
    """
    rng = np.random.default_rng(seed)

    # 产品类型：3 种产品，每个产品对应一条工艺路线
    n_product_types = 3
    product_types = [rng.integers(0, n_product_types) for _ in range(n_jobs)]

    # 每种产品类型的工艺路线 (机器顺序)
    # 类型 0: 机器 0→2→4 (经典 CMOS 路线)
    # 类型 1: 机器 1→0→3→4 (特殊工艺)
    # 类型 2: 机器 0→1→2→3→4 (全流程)
    type_routes = {
        0: [0, 2, 4],
        1: [1, 0, 3, 4],
        2: [0, 1, 2, 3, 4],
    }
    routes = [type_routes[pt] for pt in product_types]

    # 处理时间：不同批次在相同机器上处理时间有微小差异（模拟工艺波动）
    # 基础处理时间（分钟）: 机器 0=30, 1=25, 2=35, 3=20, 4=40
    base_times = [30, 25, 35, 20, 40]
    p = {}
    for j in range(n_jobs):
        for m in routes[j]:
            # 基础时间 ±20% 随机波动
            p[(j, m)] = max(5, int(base_times[m] * rng.uniform(0.8, 1.2)))

    # Setup time：同产品类型之间切换快 (5min)，跨类型切换慢 (20-40min)
    setup = {}
    for m in range(n_machines):
        for j in range(n_jobs):
            if m not in routes[j]:
                continue
            for k in range(n_jobs):
                if j == k or m not in routes[k]:
                    continue
                if product_types[j] == product_types[k]:
                    setup[(j, k, m)] = rng.integers(3, 8)   # 同类型：3-7 分钟
                else:
                    setup[(j, k, m)] = rng.integers(20, 45)  # 跨类型：20-44 分钟

    return routes, p, setup, product_types, n_machines


# ============================================================
# 人工排程基线：先到先服务贪心 (FCFS Greedy)
# ============================================================

def estimate_manual_makespan(routes, p, setup, n_jobs):
    """
    模拟人工排程：按批次编号顺序 (FCFS)，每台机器独立贪心排。

    这不是真正的人工排程（排程员会有更多经验法则），
    但它代表了一种常见的「默认模式」——按批次到达顺序，
    每台机器谁先到谁先做。
    """
    # 每台机器当前空闲的时刻
    machine_ready = defaultdict(float)
    # 每个批次当前工序完成后的时刻
    job_ready = [0.0] * n_jobs
    # 每台机器上一个加工的批次 (用于计算 setup)
    last_on_machine = {}

    makespan = 0.0

    # 按批次顺序轮流推进：每个批次走完它的工艺路线
    for j in range(n_jobs):
        route = routes[j]
        for step_idx, m in enumerate(route):
            # 该工序可以开始的时间：
            #   = max(批次 j 上一步完成时间, 机器 m 空闲时间)
            earliest = max(job_ready[j], machine_ready[m])

            # 如果机器 m 上之前有加工其他批次，加 setup time
            if m in last_on_machine:
                prev_j = last_on_machine[m]
                setup_key = (prev_j, j, m)
                if setup_key in setup:
                    earliest += setup[setup_key]

            start = earliest
            end = start + p[(j, m)]

            # 更新状态
            machine_ready[m] = end
            job_ready[j] = end
            last_on_machine[m] = j

            if end > makespan:
                makespan = end

    return makespan


# ============================================================
# 辅助打印函数
# ============================================================

def print_schedule(model, start_cols, p, routes, n_jobs):
    """
    从 MIP 解中提取并打印每个批次的排程时间表。

    输出格式：每个批次一行，显示在各机器上的开始-结束时间。
    """
    print(f"\n{'='*70}")
    print("  排程详情 (start → end 时刻, 单位: 分钟)")
    print(f"{'='*70}")

    makespan = 0.0
    for j in range(n_jobs):
        line = f"  批次 {j+1:2d}:  "
        route = routes[j]
        for m in route:
            s = model.getSolution(start_cols[(j, m)])
            e = s + p[(j, m)]
            line += f" M{m}({s:5.0f}→{e:5.0f})"
            if e > makespan:
                makespan = e
        print(line)

    print(f"\n  {'─'*70}")
    print(f"  Makespan (最晚完工时刻): {makespan:.0f} 分钟 ({makespan/60:.1f} 小时)")
    print(f"{'='*70}\n")
    return makespan


# ============================================================
# 主求解函数
# ============================================================

def solve_scheduling(n_jobs=10, n_machines=5, tight_big_m=True,
                     symmetry_breaking=True, seed=42, max_seconds=300):
    """
    用 MIP 求解 JSSP (Job Shop Scheduling Problem)，析取建模。

    参数:
        n_jobs: 批次数
        n_machines: 机器数
        tight_big_m: True=用物理上界, False=用 999999
        symmetry_breaking: 是否加对称性破坏约束
        seed: 随机种子
        max_seconds: 求解时间上限 (秒)
    """
    # --- 数据准备 ---
    routes, p, setup, product_types, _ = generate_instance(n_jobs, n_machines, seed)

    # 计算物理上界 Big-M：所有处理时间 + 最大可能 setup time 之和
    total_proc = sum(p.values())
    total_setup = sum(setup.values())
    M_tight = float(total_proc + total_setup)
    M_loose = 999999.0
    BigM = M_tight if tight_big_m else M_loose

    big_m_label = f"物理上界 {BigM:.0f}" if tight_big_m else f"魔法数字 {BigM:.0f}"

    print(f"\n{'='*70}")
    print(f"  JSSP 求解: {n_jobs} 批次, {n_machines} 机器")
    print(f"  Big-M: {big_m_label}")
    print(f"  对称性破坏: {'开启' if symmetry_breaking else '关闭'}")
    print(f"  总处理时间: {total_proc:.0f} min | 总 Setup: {total_setup:.0f} min")
    print(f"{'='*70}")

    # --- 模型构建 ---
    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    model.setOptionValue("time_limit", max_seconds)

    # ============================================================
    # 决策变量
    # ============================================================
    #
    # start[j,m] = 批次 j 在机器 m 上的开始时刻 (连续变量, ≥ 0)
    #   仅当机器 m 在批次 j 的工艺路线上时才有定义。
    #
    # y[j,k,m] = 1 表示在机器 m 上，批次 j 在批次 k 之前加工 (0-1 变量)
    #   只定义 j < k，避免重复变量。
    #   当 y[j,k,m]=1 时: j 在 k 前, start[k,m] ≥ start[j,m] + p[j,m] + setup[j,k,m]
    #   当 y[j,k,m]=0 时: k 在 j 前, start[j,m] ≥ start[k,m] + p[k,m] + setup[k,j,m]
    #
    # Cmax = makespan (最晚完工时刻)，目标函数即最小化 Cmax。
    #
    # 思路 (像对同事说话):
    #   对于每台机器，我们需要排出一个批次顺序。y[j,k,m] 就是
    #   「在这台机器上，j 和 k 谁先谁后」的二元选择。
    #   Big-M 用来在 y=0 时「关闭」其中一个方向的不等式，
    #   只保留实际排序方向的那条约束起作用。

    # --- 建立哪些机器上存在哪些批次的索引 ---
    # jobs_on_machine[m] = [j0, j1, ...]  在机器 m 上有工序的所有批次
    jobs_on_machine = defaultdict(list)
    for j in range(n_jobs):
        for m in routes[j]:
            jobs_on_machine[m].append(j)

    # --- start 变量: 开始时刻 ---
    start_cols = {}  # (j, m) -> col_index
    for j in range(n_jobs):
        for m in routes[j]:
            idx = model.addVar(0.0, highspy.kHighsInf)
            model.changeColCost(idx, 0.0)
            start_cols[(j, m)] = idx

    # --- Cmax 变量: makespan ---
    cmax_idx = model.addVar(0.0, highspy.kHighsInf)
    model.changeColCost(cmax_idx, 1.0)  # 目标函数系数 = 1
    model.changeColIntegrality(cmax_idx, 0)  # 连续变量

    # --- y 变量: 析取排序选择 ---
    y_cols = {}  # (j, k, m) -> col_index  其中 j < k
    for m in range(n_machines):
        jobs = jobs_on_machine[m]
        for idx_j, j in enumerate(jobs):
            for idx_k in range(idx_j + 1, len(jobs)):
                k = jobs[idx_k]
                col = model.addVar(0.0, 1.0)
                model.changeColCost(col, 0.0)
                model.changeColIntegrality(col, 1)  # 0-1 整数
                y_cols[(j, k, m)] = col

    num_vars = model.getNumCol()
    print(f"  变量数: {num_vars}")

    # ============================================================
    # 约束
    # ============================================================

    # --- 约束 1: 工艺路线前置约束 ---
    # 对于批次 j 的工艺路线 [m0, m1, m2, ...]:
    #   start[j, m_{i+1}] >= start[j, m_i] + p[j, m_i]
    # 含义: 后一道工序必须在前一道工序完成后才能开始。
    # 通俗: 光刻完了才能刻蚀，刻蚀完了才能沉积。不能跳步。
    for j in range(n_jobs):
        route = routes[j]
        for step_idx in range(len(route) - 1):
            m_curr = route[step_idx]
            m_next = route[step_idx + 1]
            # start[j, m_next] - start[j, m_curr] >= p[j, m_curr]
            cols = [start_cols[(j, m_next)], start_cols[(j, m_curr)]]
            vals = [1.0, -1.0]
            model.addRow(float(p[(j, m_curr)]), highspy.kHighsInf,
                         len(cols), cols, vals)

    # --- 约束 2: 析取约束 (机器互斥 + setup time) ★★★ 核心约束 ★★★ ---
    #
    # 对每台机器 m 上的每对批次 (j,k), j<k:
    #
    #   如果 y[j,k,m] = 1 (j 在 k 前):
    #     start[k,m] >= start[j,m] + p[j,m] + setup[j,k,m]   (方向1生效)
    #     start[j,m] >= start[k,m] + p[k,m] + setup[k,j,m]   (方向2被Big-M关闭)
    #
    #   如果 y[j,k,m] = 0 (k 在 j 前):
    #     方向1 被 Big-M 关闭，方向2 生效（两边反过来）
    #
    # 用 Big-M 线性化表达:
    #   (a) start[k,m] - start[j,m] + M*(1 - y[j,k,m]) >= p[j,m] + setup[j,k,m]
    #       当 y=1: start[k,m] >= start[j,m] + p[j,m] + setup[j,k,m]  ← 生效
    #       当 y=0: start[k,m] >= start[j,m] + p[j,m] + setup[j,k,m] - M  ← 自动满足
    #
    #   (b) start[j,m] - start[k,m] + M*y[j,k,m] >= p[k,m] + setup[k,j,m]
    #       当 y=0: start[j,m] >= start[k,m] + p[k,m] + setup[k,j,m]  ← 生效
    #       当 y=1: start[j,m] >= start[k,m] + p[k,m] + setup[k,j,m] - M  ← 自动满足
    #
    # Big-M 的选取 (见 .md §4):
    #   M = total_processing_time ≈ 物理上界。
    #   不用 M=999999——过大的 M 让 LP 松弛极弱，branch-and-bound 效率极低。

    disj_count = 0
    for m in range(n_machines):
        jobs = jobs_on_machine[m]
        for idx_j, j in enumerate(jobs):
            for idx_k in range(idx_j + 1, len(jobs)):
                k = jobs[idx_k]
                y_col = y_cols[(j, k, m)]

                # 方向 (a): start[k,m] - start[j,m] + M*(1-y) >= p[j,m] + setup
                setup_jk = setup.get((j, k, m), 0)
                rhs_a = float(p[(j, m)] + setup_jk)
                cols_a = [start_cols[(k, m)], start_cols[(j, m)], y_col]
                vals_a = [1.0, -1.0, BigM]
                # 整理: start_k - start_j + M*y >= rhs_a - M
                #   → start_k - start_j + M*y >= p_j + setup_jk - M
                # 移项: -start_j + start_k + M*y >= p_j + setup_jk - M
                model.addRow(rhs_a - BigM, highspy.kHighsInf,
                             len(cols_a), cols_a, vals_a)

                # 方向 (b): start[j,m] - start[k,m] + M*y >= p[k,m] + setup
                setup_kj = setup.get((k, j, m), 0)
                rhs_b = float(p[(k, m)] + setup_kj)
                cols_b = [start_cols[(j, m)], start_cols[(k, m)], y_col]
                vals_b = [1.0, -1.0, BigM]
                # start_j - start_k + M*y >= p_k + setup_kj
                model.addRow(rhs_b, highspy.kHighsInf,
                             len(cols_b), cols_b, vals_b)

                disj_count += 2

    print(f"  析取约束对数: {disj_count} 条")

    # --- 约束 3: Cmax 定义 ---
    # Cmax >= start[j, m_last] + p[j, m_last]  对所有批次 j
    # 含义: makespan 不能小于任何一个批次的完工时刻。
    for j in range(n_jobs):
        last_m = routes[j][-1]
        cols = [cmax_idx, start_cols[(j, last_m)]]
        vals = [1.0, -1.0]
        model.addRow(float(p[(j, last_m)]), highspy.kHighsInf,
                     len(cols), cols, vals)

    # ============================================================
    # 对称性破坏约束 ★★★ 加速关键 ★★★
    # ============================================================
    #
    # 如果两个批次的工艺路线完全相同, 它们的工序处理时间也接近,
    # 那么它们在数学上是可互换的 (symmetric)。
    # 求解器会花大量时间探索 j 在 k 前 和 k 在 j 前 两种对称分支。
    #
    # 对称性破坏: 强制相同路线的批次按编号排序第一台机器上的开始时间。
    #   如果 routes[j] == routes[k] 且 j < k:
    #     start[j, routes[j][0]] <= start[k, routes[k][0]]
    #
    # 这砍掉了约一半的对称搜索分支, 不影响最优解的存在性
    # (因为如果最优解里 k 在 j 前, 交换它们得到的解目标值相同)。

    sym_count = 0
    if symmetry_breaking:
        for j in range(n_jobs):
            for k in range(j + 1, n_jobs):
                if routes[j] == routes[k]:
                    # 强制 start[j, first_machine] <= start[k, first_machine]
                    first_m = routes[j][0]
                    cols = [start_cols[(k, first_m)], start_cols[(j, first_m)]]
                    vals = [1.0, -1.0]
                    model.addRow(0.0, highspy.kHighsInf, len(cols), cols, vals)
                    sym_count += 1
        print(f"  对称性破坏约束: {sym_count} 条 (相同路线批次对数)")

    num_cons = model.getNumRows()
    print(f"  总约束数: {num_cons}")

    # --- 求解 ---
    print(f"\n  求解中... (时限 {max_seconds}s)")

    # Minimize Cmax
    model.changeObjectiveSense(1)  # 1 = minimize (highspy convention)

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

    info = model.getInfo()
    node_count = info.get("mip_node_count", "N/A")

    print(f"\n  求解状态: {status_str}")
    print(f"  求解耗时: {solve_time:.1f}s")
    print(f"  B&B 节点数: {node_count}")

    if status not in [highspy.HighsModelStatus.kOptimal,
                       highspy.HighsModelStatus.kTimeLimit]:
        print(f"  可能原因: 无法找到可行解。尝试减少批次数或增加求解时间。")
        return None

    obj_val = model.getInfoValue("objective_function_value")
    print(f"  最优 makespan: {obj_val:.1f} 分钟 ({obj_val/60:.1f} 小时)")

    # --- 对比人工排程 ---
    manual = estimate_manual_makespan(routes, p, setup, n_jobs)
    improvement = 100 * (1 - obj_val / manual) if manual > 0 else 0
    util_opt = 100 * sum(p.values()) / (n_machines * obj_val)
    util_manual = 100 * sum(p.values()) / (n_machines * manual)

    print(f"  人工排程 makespan: {manual:.1f} 分钟")
    print(f"  优化提升: {improvement:.1f}%")
    print(f"  设备利用率: MIP={util_opt:.1f}% vs 人工={util_manual:.1f}%")

    # --- 打印排程 ---
    if status == highspy.HighsModelStatus.kOptimal:
        print_schedule(model, start_cols, p, routes, n_jobs)

    return dict(
        obj_val=obj_val,
        solve_time=solve_time,
        manual_makespan=manual,
        n_vars=num_vars,
        n_cons=num_cons,
        n_nodes=node_count,
        status=status_str,
        big_m=BigM,
        symmetry_breaking=symmetry_breaking,
        tight_big_m=tight_big_m,
    )


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║  案例2：工厂生产排程 (Job Shop Scheduling)                   ║
║  求解器: HiGHS  ·  算法: MIP + 析取建模 + 对称性破坏        ║
╚══════════════════════════════════════════════════════════════╝
""")

    results = {}

    # --- 小规模测试: 5 批次 x 3 机器 ---
    print("\n### 小规模测试 (5批次, 3机器) ###")
    r = solve_scheduling(n_jobs=5, n_machines=3, seed=42)
    if r:
        results["小规模 (5x3)"] = r

    # --- 目标规模: 10 批次 x 5 机器, 紧 Big-M, 带对称性破坏 ---
    print("\n### 目标规模 (10批次, 5机器) — 紧Big-M + 对称性破坏 ###")
    r = solve_scheduling(n_jobs=10, n_machines=5, tight_big_m=True,
                         symmetry_breaking=True, seed=42)
    if r:
        results["紧M+对称破坏 (10x5)"] = r

    # --- 对比: 紧 Big-M, 无对称性破坏 ---
    print("\n### 对比1 (10批次, 5机器) — 紧Big-M, 无对称性破坏 ###")
    r_no_sym = solve_scheduling(n_jobs=10, n_machines=5, tight_big_m=True,
                                symmetry_breaking=False, seed=42, max_seconds=120)
    if r_no_sym:
        results["紧M无对称 (10x5)"] = r_no_sym

    # --- 对比: 松 Big-M (999999), 无对称性破坏 ---
    print("\n### 对比2 (10批次, 5机器) — 松Big-M=999999, 无对称破坏 ###")
    r_loose = solve_scheduling(n_jobs=10, n_machines=5, tight_big_m=False,
                               symmetry_breaking=False, seed=42, max_seconds=300)
    if r_loose:
        results["松M无对称 (10x5)"] = r_loose

    # --- 结果汇总 ---
    print(f"\n{'='*80}")
    print("  结果汇总")
    print(f"{'='*80}")
    header = (f"  {'配置':<30s} {'变量':>6s} {'约束':>6s} "
              f"{'Big-M':>9s} {'makespan':>10s} {'耗时':>8s} {'节点':>8s}")
    print(header)
    print(f"  {'─'*80}")
    for name, r in results.items():
        if r is None:
            continue
        print(f"  {name:<30s} {r['n_vars']:>6d} {r['n_cons']:>6d} "
              f"{r['big_m']:>9.0f} {r['obj_val']:>8.0f}min "
              f"{r['solve_time']:>6.1f}s {str(r['n_nodes']):>8s}")
    print(f"  {'─'*80}")

    if len(results) >= 3:
        r_best = results.get("紧M+对称破坏 (10x5)")
        r_nosym = results.get("紧M无对称 (10x5)")
        r_loose = results.get("松M无对称 (10x5)")

        if r_best and r_nosym:
            speedup = r_nosym["solve_time"] / r_best["solve_time"] if r_best["solve_time"] > 0 else 0
            print(f"\n  对称性破坏加速: {speedup:.1f}x"
                  f" ({r_nosym['solve_time']:.1f}s → {r_best['solve_time']:.1f}s)")

        if r_best and r_loose:
            speedup = r_loose["solve_time"] / r_best["solve_time"] if r_best["solve_time"] > 0 else 0
            print(f"  紧Big-M加速: {speedup:.1f}x"
                  f" ({r_loose['solve_time']:.1f}s → {r_best['solve_time']:.1f}s)")

    # --- 业务价值计算 ---
    if r_best:
        makespan = r_best["obj_val"]
        manual = r_best["manual_makespan"]
        batches_per_day_opt = 1440 / makespan
        batches_per_day_manual = 1440 / manual
        extra_per_day = batches_per_day_opt - batches_per_day_manual
        print(f"\n  业务价值估算:")
        print(f"    人工排程产能: {batches_per_day_manual:.2f} 批次/天")
        print(f"    MIP 优化产能: {batches_per_day_opt:.2f} 批次/天")
        print(f"    每天增产:     {extra_per_day:.2f} 批次")
        print(f"    (按 8万美元/批次) 年增收 ≈ {extra_per_day * 8 * 365:.0f} 万美元")

    print()
