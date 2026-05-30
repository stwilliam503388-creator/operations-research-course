"""
案例4：发电厂调度 (Unit Commitment)
难度：★★★☆☆
求解器：HiGHS (highspy, 免费)
依赖：pip install highspy numpy
运行：python case04_energy.py

简述：虚拟电厂 10 台发电机，24 小时电价/负荷波动，用 MIP 求解
      机组组合 (Unit Commitment) 问题——每个小时决定每台机组
      开/关和出力水平，最小化总成本（燃料 + 启动成本）。
      对比人工「优先顺序法」基线，展示 MIP 优化节省 ~10%。

技术背景：
  UC 是电力系统运行中最核心的优化问题。它属于 NP-hard——
  即使不考虑爬坡和最小启停时间约束，纯组合部分也是 NP-hard。
  模型本质是混合整数线性规划 (MILP)：on[t,g] ∈ {0,1} 是离散
  决策，power[t,g] 是连续出力变量。约束涉及时间维度上的耦合
  （最小开/停机时间、爬坡速率），使得人工经验法难以全局优化。

公式逐符号解释:

  目标函数:
    min  Σ[t∈T] Σ[g∈G] fuel_cost[g] * power[t,g]
         + Σ[t∈T] Σ[g∈G] startup_cost[g] * start[t,g]

    fuel_cost[g]    → 机组 g 的燃料成本 ($/MWh)
    power[t,g]      → 机组 g 在 t 时刻的出力 (MW, 连续变量)
    startup_cost[g] → 机组 g 的启动成本 ($/次)
    start[t,g]      → 机组 g 在 t 时刻是否刚启动 (0-1 变量)

  约束:
    (1) Σ[g] power[t,g] = demand[t]         ← 功率平衡 (每个小时)
    (2) P_min[g]·on[t,g] ≤ power[t,g]       ← 出力下限 (开着必须发够)
        power[t,g] ≤ P_max[g]·on[t,g]       ← 出力上限
    (3) start[t,g] ≥ on[t,g] - on[t-1,g]    ← 启动检测 (0→1 跳变)
        start[t,g] ≤ on[t,g]
        start[t,g] ≤ 1 - on[t-1,g]
    (4) on[t,g] ≥ Σ[k=max(0,t-min_up+1)..t] start[k,g]  ← 最小开机时间
    (5) 1-on[t,g] ≥ Σ[k=max(0,t-min_down+1)..t] shut[k,g] ← 最小停机时间
    (6) |power[t,g] - power[t-1,g]| ≤ ramp[g]             ← 爬坡速率

  决策变量:
    on[t,g]    ∈ {0, 1}  → 机组 g 在时刻 t 是否开机
    start[t,g] ∈ {0, 1}  → 机组 g 在时刻 t 是否刚启动 (0→1 边沿)
    shut[t,g]  ∈ {0, 1}  → 机组 g 在时刻 t 是否刚停机 (1→0 边沿)
    power[t,g] ∈ [0, P_max] → 机组 g 在时刻 t 的出力 (MW)
"""


# 教学注释：重点看变量、约束和目标函数如何把业务规则翻译成 MIP 模型。
# 求解日志或结果可用来理解分支定界、松弛和可行解质量。



import numpy as np
import time
import sys


# ============================================================
# 数据生成
# ============================================================

def generate_generator_data(n_generators=10, seed=42):
    """
    生成虚拟电厂发电机参数。

    参数:
        n_generators: 发电机组数量
        seed: 随机种子（保证可复现）

    返回:
        dict: 包含各参数的字典
            - P_min: shape (n_generators,) — 最小出力 (MW)
            - P_max: shape (n_generators,) — 最大出力 (MW)
            - fuel_cost: shape (n_generators,) — 燃料成本 ($/MWh)
            - startup_cost: shape (n_generators,) — 启动成本 ($)
            - min_up: shape (n_generators,) — 最小连续开机时间 (h)
            - min_down: shape (n_generators,) — 最小连续停机时间 (h)
            - ramp: shape (n_generators,) — 最大爬坡速率 (MW/h)
            - gen_type: list — 机组类型标签
    """
    rng = np.random.default_rng(seed)

    # --- 10台机组分四类 ---
    # 类型 1: 大煤机 (基荷, Gen1-2) — 便宜但启动慢、爬坡慢
    # 类型 2: 中煤机 (Gen3-4) — 中等
    # 类型 3: 燃气机组 (腰荷, Gen5-7) — 较贵但灵活
    # 类型 4: 柴油调峰 (峰荷, Gen8-10) — 最贵但启动快

    gen_types = [
        "大煤机", "大煤机",
        "中煤机", "中煤机",
        "燃气机组", "燃气机组", "燃气机组",
        "柴油调峰", "柴油调峰", "柴油调峰",
    ]

    # P_min 和 P_max: 每种类型设定范围
    P_min = np.array([
        180, 180,       # 大煤机
        120, 120,       # 中煤机
        50,  50,  50,   # 燃气机组
        20,  20,  20,   # 柴油调峰
    ], dtype=float)

    P_max = np.array([
        600, 600,       # 大煤机: 满发 600MW
        400, 400,       # 中煤机: 满发 400MW
        200, 220, 180,  # 燃气机组: 略有不均
        80,  90,  70,   # 柴油调峰: 小机组
    ], dtype=float)

    # 燃料成本: 大煤机最便宜, 柴油最贵
    fuel_cost = np.array([
        20,  21,        # 大煤机: ~$20/MWh
        28,  30,        # 中煤机: ~$29/MWh
        50,  52,  55,   # 燃气机组: ~$52/MWh
        90,  95, 100,   # 柴油调峰: ~$95/MWh
    ], dtype=float)

    # 启动成本: 大煤机启动贵, 小机组启动便宜
    startup_cost = np.array([
        5000, 5200,     # 大煤机
        3500, 3600,     # 中煤机
        1200, 1300, 1400, # 燃气机组
        500,  600,  700,  # 柴油调峰
    ], dtype=float)

    # 最小开机时间: 大机组开了就不能随便停
    min_up = np.array([
        6, 6,           # 大煤机: 开了至少烧 6h
        4, 4,           # 中煤机
        2, 2, 2,        # 燃气机组: 灵活
        1, 1, 1,        # 柴油调峰: 随时可启停
    ], dtype=int)

    # 最小停机时间: 大机组停了就得歇够
    min_down = np.array([
        6, 6,           # 大煤机
        3, 3,           # 中煤机
        1, 1, 1,        # 燃气机组
        0, 0, 0,        # 柴油调峰: 可以秒停秒开
    ], dtype=int)

    # 爬坡速率: 大机组爬得慢, 小机组爬得快
    ramp = np.array([
        60,  60,        # 大煤机: 60 MW/h
        80,  80,        # 中煤机: 80 MW/h
        120, 130, 110,  # 燃气机组: ~120 MW/h
        70,  80,  60,   # 柴油调峰: 但本身容量小所以比例大
    ], dtype=float)

    return {
        "P_min": P_min[:n_generators],
        "P_max": P_max[:n_generators],
        "fuel_cost": fuel_cost[:n_generators],
        "startup_cost": startup_cost[:n_generators],
        "min_up": min_up[:n_generators],
        "min_down": min_down[:n_generators],
        "ramp": ramp[:n_generators],
        "gen_type": gen_types[:n_generators],
    }


def generate_demand_curve(n_hours=24, seed=123):
    """
    生成 24 小时负荷曲线 (MW)。

    典型日负荷曲线: 凌晨低 → 上午爬升 → 下午高位 → 傍晚峰值 → 深夜回落

    参数:
        n_hours: 小时数
        seed: 随机种子

    返回:
        demand: shape (n_hours,) — 每小时负荷需求 (MW)
    """
    rng = np.random.default_rng(seed)

    # 基础负荷曲线模板
    hours = np.arange(n_hours)
    base_shape = np.array([
        500, 480, 470, 460,  # 0-3: 凌晨低谷
        480, 520,            # 4-5: 开始爬升
        650, 750, 850,       # 6-8: 上午快速上升
        920, 960, 980,       # 9-11: 接近峰值
        1000, 1020,          # 12-13: 午间微降
        980, 960, 950,       # 14-16: 下午稳定高位
        1050, 1100, 1080,    # 17-19: 晚高峰（下班回家）
        950, 850,            # 20-21: 回落
        750, 650,            # 22-23: 深夜
    ])

    # 加少量随机噪声 (模拟实际负荷的随机波动)
    noise = rng.normal(0, 15, size=n_hours)
    demand = base_shape + noise
    # 确保非负
    demand = np.maximum(demand, 300)

    return demand


# ============================================================
# 核心求解函数：机组组合 MIP
# ============================================================

def solve_unit_commitment(gen_data, demand, time_limit_sec=60):
    """
    用 HiGHS 求解机组组合 (Unit Commitment) MIP 模型。

    参数:
        gen_data: dict, generate_generator_data() 的返回值
        demand: ndarray, shape (n_hours,), 负荷需求 (MW)
        time_limit_sec: 求解时间上限 (秒)

    返回:
        dict: 包含求解状态、成本、调度计划等信息
    """
    import highspy

    n_hours = len(demand)
    n_gens = len(gen_data["P_min"])
    T = n_hours
    G = n_gens

    # 解包参数
    P_min = gen_data["P_min"]
    P_max = gen_data["P_max"]
    fuel_cost = gen_data["fuel_cost"]
    startup_cost = gen_data["startup_cost"]
    min_up = gen_data["min_up"]
    min_down = gen_data["min_down"]
    ramp = gen_data["ramp"]

    # ============================================================
    # 构建变量索引映射
    # ============================================================
    # 四种变量: on, start, shut, power
    # 每个 (t, g) 对都有四个变量，按顺序排列
    # Index layout:
    #   block 0: on[t,g]     — T*G 个, binary
    #   block 1: start[t,g]  — T*G 个, binary
    #   block 2: shut[t,g]   — T*G 个, binary
    #   block 3: power[t,g]  — T*G 个, continuous

    VAR_ON = 0
    VAR_START = 1
    VAR_SHUT = 2
    VAR_POWER = 3
    BLOCK_SIZE = T * G

    def var_idx(var_type, t, g):
        """返回变量在模型中的全局索引。"""
        return var_type * BLOCK_SIZE + t * G + g

    total_vars = 4 * BLOCK_SIZE

    # ============================================================
    # 初始化 HiGHS 模型
    # ============================================================
    h = highspy.Highs()

    # 关闭 HiGHS 默认输出（用我们自己的进度输出）
    h.setOptionValue("output_flag", False)

    # 设置时间限制
    h.setOptionValue("time_limit", time_limit_sec)

    # --- 添加变量 ---
    # 先批量添加所有连续/二元变量
    # highspy 的 addVar 逐个添加效率较低，
    # 这里用 addVars (批量) 或逐个添加
    # 为了代码清晰，逐个添加并记录

    print(f"  构建模型: {n_hours}h × {n_gens}台 = "
          f"{T*G} 个时段, 共 {total_vars} 个变量")

    inf = highspy.kHighsInf

    for vtype_id in range(4):
        for t in range(T):
            for g in range(G):
                idx = var_idx(vtype_id, t, g)
                if vtype_id in (VAR_ON, VAR_START, VAR_SHUT):
                    # 0-1 二元变量
                    h.addVar(lb=0, ub=1)
                    h.changeColIntegrality(
                        idx,
                        highspy.HighsVarType.kInteger
                    )
                else:  # VAR_POWER
                    h.addVar(lb=0, ub=P_max[g])

    # ============================================================
    # 设置目标函数: min Σ fuel_cost * power + Σ startup_cost * start
    # ============================================================
    # 初始目标系数都是 0，修改 power 和 start 的系数
    for t in range(T):
        for g in range(G):
            # power 变量成本
            p_idx = var_idx(VAR_POWER, t, g)
            h.changeColCost(p_idx, float(fuel_cost[g]))
            # start 变量成本
            s_idx = var_idx(VAR_START, t, g)
            h.changeColCost(s_idx, float(startup_cost[g]))

    # ============================================================
    # 约束构建
    # ============================================================

    # 辅助函数: addRow(lower, upper, nnz, indices, values)
    constraint_count = 0

    def add_constraint(lower, upper, coef_pairs):
        """
        添加一条线性约束。

        参数:
            lower: 约束下界 (double or -inf)
            upper: 约束上界 (double or +inf)
            coef_pairs: list of (var_index, coefficient) tuples
        """
        nonlocal constraint_count
        indices = np.array([c[0] for c in coef_pairs], dtype=np.int32)
        values = np.array([c[1] for c in coef_pairs], dtype=np.float64)
        h.addRow(lower, upper, len(indices), indices, values)
        constraint_count += 1

    # --- 约束 1: 功率平衡 Σ[g] power[t,g] = demand[t] ---
    for t in range(T):
        coefs = [(var_idx(VAR_POWER, t, g), 1.0) for g in range(G)]
        add_constraint(demand[t], demand[t], coefs)

    # --- 约束 2: 出力上下限 ---
    # P_min[g] * on[t,g] <= power[t,g] <= P_max[g] * on[t,g]
    for t in range(T):
        for g in range(G):
            on_idx = var_idx(VAR_ON, t, g)
            p_idx = var_idx(VAR_POWER, t, g)

            # 下限: power - P_min * on >= 0
            add_constraint(0, inf, [
                (p_idx, 1.0),
                (on_idx, -P_min[g]),
            ])

            # 上限: power - P_max * on <= 0
            add_constraint(-inf, 0, [
                (p_idx, 1.0),
                (on_idx, -P_max[g]),
            ])

    # --- 约束 3: 启动检测 ---
    # start[t,g] >= on[t,g] - on[t-1,g]
    # start[t,g] <= on[t,g]
    # start[t,g] <= 1 - on[t-1,g]
    for t in range(T):
        for g in range(G):
            s_idx = var_idx(VAR_START, t, g)
            on_idx = var_idx(VAR_ON, t, g)

            # start <= on
            add_constraint(-inf, 0, [(s_idx, 1.0), (on_idx, -1.0)])

            if t == 0:
                # t=0: 假设初始状态为关机
                # start[0,g] >= on[0,g] - 0 = on[0,g]
                add_constraint(0, inf, [(s_idx, 1.0), (on_idx, -1.0)])
                # start[0,g] <= 1 - 0 = 1 (always true, skip)
            else:
                on_prev = var_idx(VAR_ON, t - 1, g)

                # start >= on[t] - on[t-1]
                add_constraint(0, inf, [
                    (s_idx, 1.0), (on_idx, -1.0), (on_prev, 1.0),
                ])

                # start <= 1 - on[t-1]
                add_constraint(-inf, 1, [
                    (s_idx, 1.0), (on_prev, 1.0),
                ])

    # --- 约束 3b: 停机检测 (shut) ---
    # shut[t,g] >= on[t-1,g] - on[t,g]
    # shut[t,g] <= 1 - on[t,g]
    # shut[t,g] <= on[t-1,g]
    for t in range(T):
        for g in range(G):
            h_idx = var_idx(VAR_SHUT, t, g)
            on_idx = var_idx(VAR_ON, t, g)

            # shut <= 1 - on[t]
            add_constraint(-inf, 1, [(h_idx, 1.0), (on_idx, 1.0)])

            if t == 0:
                # t=0: 假设初始为关，on[t-1]=0
                # shut >= 0 - on[0] = -on[0], 即 shut + on >= 0 (always true)
                # shut <= 0 (即 shut = 0, 因为初始已关机)
                add_constraint(-inf, 0, [(h_idx, 1.0)])
            else:
                on_prev = var_idx(VAR_ON, t - 1, g)

                # shut >= on[t-1] - on[t]
                add_constraint(0, inf, [
                    (h_idx, 1.0), (on_idx, 1.0), (on_prev, -1.0),
                ])

                # shut <= on[t-1]
                add_constraint(-inf, 0, [
                    (h_idx, 1.0), (on_prev, -1.0),
                ])

    # --- 约束 4: 最小开机时间 ---
    # on[t,g] >= Σ[k=max(0, t-min_up+1)..t] start[k,g]
    for t in range(1, T):
        for g in range(G):
            mu = min_up[g]
            if mu <= 1:
                continue
            window_start = max(0, t - mu + 1)
            coefs = [(var_idx(VAR_ON, t, g), 1.0)]
            for k in range(window_start, t + 1):
                coefs.append((var_idx(VAR_START, k, g), -1.0))
            add_constraint(0, inf, coefs)

    # --- 约束 5: 最小停机时间 ---
    # 1 - on[t,g] >= Σ[k=max(0, t-min_down+1)..t] shut[k,g]
    # 即: on[t,g] + Σ shut[k,g] <= 1
    for t in range(1, T):
        for g in range(G):
            md = min_down[g]
            if md <= 1:
                continue
            window_start = max(0, t - md + 1)
            coefs = [(var_idx(VAR_ON, t, g), 1.0)]
            for k in range(window_start, t + 1):
                coefs.append((var_idx(VAR_SHUT, k, g), 1.0))
            add_constraint(-inf, 1, coefs)

    # --- 约束 6: 爬坡速率 ---
    # |power[t,g] - power[t-1,g]| <= ramp[g]
    for t in range(1, T):
        for g in range(G):
            p_idx = var_idx(VAR_POWER, t, g)
            p_prev = var_idx(VAR_POWER, t - 1, g)

            # 向上爬坡: power[t] - power[t-1] <= ramp
            add_constraint(-inf, ramp[g], [(p_idx, 1.0), (p_prev, -1.0)])

            # 向下爬坡: power[t-1] - power[t] <= ramp
            add_constraint(-inf, ramp[g], [(p_prev, 1.0), (p_idx, -1.0)])

    print(f"  变量: {total_vars}, 约束: {constraint_count}")

    # ============================================================
    # 求解
    # ============================================================
    print(f"  求解中 (HiGHS, 时间上限 {time_limit_sec}s)...")
    start_time = time.time()
    h.run()
    solve_time = time.time() - start_time

    # ============================================================
    # 解析结果
    # ============================================================
    model_status = h.getModelStatus()
    status_map = {
        highspy.HighsModelStatus.kOptimal: "最优解",
        highspy.HighsModelStatus.kInfeasible: "不可行",
        highspy.HighsModelStatus.kUnbounded: "无界",
        highspy.HighsModelStatus.kTimeLimit: "超时(次优解)",
        highspy.HighsModelStatus.kUnknown: "未知",
    }
    status_str = status_map.get(model_status, f"状态码: {model_status}")

    solution = h.getSolution()
    col_values = solution.col_value

    # 提取变量值
    on = np.zeros((T, G), dtype=int)
    start = np.zeros((T, G), dtype=int)
    shut = np.zeros((T, G), dtype=int)
    power = np.zeros((T, G), dtype=float)

    has_solution = model_status in (
        highspy.HighsModelStatus.kOptimal,
        highspy.HighsModelStatus.kTimeLimit,
    )

    if has_solution:
        for t in range(T):
            for g in range(G):
                # 对二元变量做四舍五入（容差 0.5）
                on_val = col_values[var_idx(VAR_ON, t, g)]
                start_val = col_values[var_idx(VAR_START, t, g)]
                shut_val = col_values[var_idx(VAR_SHUT, t, g)]
                power_val = col_values[var_idx(VAR_POWER, t, g)]

                on[t, g] = 1 if on_val > 0.5 else 0
                start[t, g] = 1 if start_val > 0.5 else 0
                shut[t, g] = 1 if shut_val > 0.5 else 0
                power[t, g] = power_val

        # 计算总成本
        fuel_total = np.sum(fuel_cost[g] * power[t, g]
                            for t in range(T) for g in range(G))
        startup_total = np.sum(startup_cost[g] * start[t, g]
                               for t in range(T) for g in range(G))
        total_cost = fuel_total + startup_total
        obj_value = h.getInfoValue("objective_function_value")
    else:
        fuel_total = startup_total = total_cost = obj_value = 0

    return {
        "n_hours": n_hours,
        "n_generators": n_gens,
        "total_vars": total_vars,
        "total_constrs": constraint_count,
        "status": status_str,
        "solve_time": solve_time,
        "total_cost": total_cost,
        "fuel_cost": fuel_total,
        "startup_cost": startup_total,
        "obj_value": obj_value,
        "on": on,
        "start": start,
        "shut": shut,
        "power": power,
        "has_solution": has_solution,
    }


# ============================================================
# 输出展示
# ============================================================

def print_schedule(result, gen_data, demand):
    """
    从 MIP 解中提取并格式化打印每台机组的启停计划和出力。

    参数:
        result: solve_unit_commitment() 的返回值
        gen_data: 发电机参数
        demand: 负荷曲线
    """
    if not result["has_solution"]:
        print("  (无可行解，无法打印调度计划)")
        return

    on = result["on"]
    power = result["power"]
    T = result["n_hours"]
    G = result["n_generators"]
    gen_type = gen_data["gen_type"]

    print(f"\n{'─'*80}")
    print(f"  机组启停计划 (■ = 开机, · = 停机)")
    print(f"{'─'*80}")

    # 表头
    header = f"  {'机组':<12s} {'类型':<10s} "
    for t in range(T):
        header += f"{t:2d}"
    header += f"  {'启停':>4s}"
    print(header)
    print(f"  {'─'*12} {'─'*10} {'─'*48} {'─'*4}")

    for g in range(G):
        line = f"  Gen{g+1:<11d} {gen_type[g]:<10s} "
        for t in range(T):
            line += "■" if on[t, g] else "·"
            line += " " if t < T - 1 else ""
        # 统计开关次数
        switches = int(np.sum(np.abs(np.diff(on[:, g], prepend=0))))
        line += f"  {switches:>2d}次"
        print(line)

    print(f"  {'─'*12} {'─'*10} {'─'*48} {'─'*4}")

    # 每小时总出力与需求对比
    print(f"\n  每小时供需平衡:")
    print(f"  {'时刻':>6s} {'需求(MW)':>10s} {'出力(MW)':>10s} {'偏差':>10s}")
    print(f"  {'─'*40}")
    for t in range(T):
        total_power = np.sum(power[t, :])
        deviation = total_power - demand[t]
        print(f"  {t:>4d}h  {demand[t]:>10.0f}  {total_power:>10.0f}  "
              f"{deviation:>+10.1f}")

    # 每台机组出力摘要
    print(f"\n  机组出力摘要:")
    print(f"  {'机组':<12s} {'开机(h)':>8s} {'总出力(MWh)':>12s} "
          f"{'平均出力(MW)':>12s} {'利用率':>8s}")
    print(f"  {'─'*60}")
    for g in range(G):
        hours_on = int(np.sum(on[:, g]))
        total_mwh = float(np.sum(power[:, g]))
        avg_power = total_mwh / hours_on if hours_on > 0 else 0
        util = avg_power / gen_data["P_max"][g] * 100 if hours_on > 0 else 0
        print(f"  Gen{g+1:<11d} {hours_on:>8d}  {total_mwh:>12.0f}  "
              f"{avg_power:>12.0f}  {util:>7.0f}%")


# ============================================================
# 人工经验基线：优先顺序法 (Merit Order)
# ============================================================

def manual_dispatch_baseline(gen_data, demand):
    """
    模拟人工「优先顺序法」排班。

    算法: 按燃料成本从低到高排序机组。每个小时，先开最便宜的，
    开到满足负荷为止。同时加入简单的启停规则：
    - 如果某机组连续 2 小时利用率 < 30%，尝试关机
    - 如果负荷比上一小时增加超过 50MW，提前一小时开机

    参数:
        gen_data: 发电机参数
        demand: 负荷需求 (MW)

    返回:
        dict: 包含总成本、燃料成本、启动成本等
    """
    T = len(demand)
    G = len(gen_data["P_min"])
    P_min = gen_data["P_min"]
    P_max = gen_data["P_max"]
    fuel_cost = gen_data["fuel_cost"]
    startup_cost = gen_data["startup_cost"]

    # 按燃料成本排序（便宜的先）
    order = np.argsort(fuel_cost)

    on = np.zeros((T, G), dtype=int)
    power = np.zeros((T, G), dtype=float)

    for t in range(T):
        remaining = demand[t]
        for g_idx in order:
            if remaining <= 0:
                break
            # 尽量开到最大
            output = min(P_max[g_idx], remaining)
            # 但必须满足最小出力
            if output < P_min[g_idx] and remaining > P_min[g_idx]:
                # 如果剩余需求不够最小出力，跳过这台
                continue
            if output >= P_min[g_idx]:
                on[t, g_idx] = 1
                power[t, g_idx] = min(P_max[g_idx], remaining)
                remaining -= power[t, g_idx]

    # 简单启停优化: 扫描连续低利用率时段
    for g in range(G):
        for t in range(1, T - 1):
            if on[t, g] and on[t - 1, g] and on[t + 1, g]:
                # 检查中间时段是否利用率过低
                if power[t, g] < 0.3 * P_max[g]:
                    # 判断是否可以关掉
                    slack = (P_max[g] - power[t, g])
                    remaining = demand[t] - power[t, g]
                    if remaining >= P_min[g]:
                        # 关掉这台，让其他机组补足
                        # 简化: 只关掉这小时
                        redist = power[t, g]
                        power[t, g] = 0
                        on[t, g] = 0
                        # 尝试让其他开机机组多发力
                        for g2 in range(G):
                            if g2 == g or not on[t, g2]:
                                continue
                            extra = min(P_max[g2] - power[t, g2], redist)
                            if extra > 0:
                                power[t, g2] += extra
                                redist -= extra
                            if redist <= 0:
                                break

    # 计算成本
    fuel_total = np.sum(fuel_cost[g] * power[t, g]
                        for t in range(T) for g in range(G))
    # 统计启动次数
    startup_count = 0
    for g in range(G):
        startup_count += int(np.sum(np.diff(on[:, g], prepend=0) == 1))
    startup_total = np.sum(gen_data["startup_cost"][g] *
                           int(np.sum(np.diff(on[:, g], prepend=0) == 1))
                           for g in range(G))

    return {
        "total_cost": fuel_total + startup_total,
        "fuel_cost": fuel_total,
        "startup_cost": startup_total,
        "on": on,
        "power": power,
    }


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║  案例4：发电厂调度 (Unit Commitment)                        ║
║  求解器: HiGHS (free)                                        ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ============================================================
    # 测试 1: 小规模 (5 台 × 12 小时)
    # ============================================================
    print(f"\n{'═'*60}")
    print(f"  测试 1: 小规模 (5 台 × 12 小时)")
    print(f"{'═'*60}")

    gen_small = generate_generator_data(n_generators=5, seed=42)
    demand_small = generate_demand_curve(n_hours=12, seed=123)

    result1 = solve_unit_commitment(gen_small, demand_small, time_limit_sec=30)

    print(f"\n  求解状态: {result1['status']}")
    print(f"  求解耗时: {result1['solve_time']:.1f}s")
    if result1["has_solution"]:
        print(f"  目标值: ${result1['obj_value']:,.0f}")
        print(f"  总成本: ${result1['total_cost']:,.0f}")
        print(f"    燃料成本: ${result1['fuel_cost']:,.0f}")
        print(f"    启动成本: ${result1['startup_cost']:,.0f}")
        print_schedule(result1, gen_small, demand_small)

        # 人工基线对比
        baseline1 = manual_dispatch_baseline(gen_small, demand_small)
        saving1 = baseline1["total_cost"] - result1["total_cost"]
        saving_pct1 = saving1 / baseline1["total_cost"] * 100
        print(f"\n  人工基线对比:")
        print(f"    人工经验法: ${baseline1['total_cost']:,.0f}")
        print(f"    MIP 优化:   ${result1['total_cost']:,.0f}")
        print(f"    节省:       ${saving1:,.0f} ({saving_pct1:+.1f}%)")
    else:
        print("  (小规模测试无解，跳过)")

    # ============================================================
    # 测试 2: 中等规模 (10 台 × 24 小时)
    # ============================================================
    print(f"\n{'═'*60}")
    print(f"  测试 2: 中等规模 (10 台 × 24 小时)")
    print(f"{'═'*60}")

    gen_data = generate_generator_data(n_generators=10, seed=42)
    demand = generate_demand_curve(n_hours=24, seed=123)

    result2 = solve_unit_commitment(gen_data, demand, time_limit_sec=120)

    print(f"\n  求解状态: {result2['status']}")
    print(f"  求解耗时: {result2['solve_time']:.1f}s")
    if result2["has_solution"]:
        print(f"  目标值: ${result2['obj_value']:,.0f}")
        print(f"  总成本: ${result2['total_cost']:,.0f}")
        print(f"    燃料成本: ${result2['fuel_cost']:,.0f}")
        print(f"    启动成本: ${result2['startup_cost']:,.0f}")
        print_schedule(result2, gen_data, demand)

        # 人工基线对比
        baseline2 = manual_dispatch_baseline(gen_data, demand)
        saving2 = baseline2["total_cost"] - result2["total_cost"]
        saving_pct2 = saving2 / baseline2["total_cost"] * 100

        print(f"\n  {'═'*60}")
        print(f"  优化效果对比 (10台 × 24h)")
        print(f"  {'═'*60}")
        print(f"  {'方案':<16s} {'总成本 ($)':>14s} {'燃料成本 ($)':>14s} "
              f"{'启动成本 ($)':>14s}  {'说明'}")
        print(f"  {'─'*75}")
        print(f"  {'人工经验法':<16s} {baseline2['total_cost']:>14,.0f} "
              f"{baseline2['fuel_cost']:>14,.0f} "
              f"{baseline2['startup_cost']:>14,.0f}  "
              f"{'优先顺序 + 贪心启停'}")
        print(f"  {'MIP 最优':<16s} {result2['total_cost']:>14,.0f} "
              f"{result2['fuel_cost']:>14,.0f} "
              f"{result2['startup_cost']:>14,.0f}  "
              f"{'求解器全局优化'}")
        print(f"  {'─'*75}")
        print(f"  {'节省':<16s} {saving2:>14,.0f} "
              f"{baseline2['fuel_cost']-result2['fuel_cost']:>14,.0f} "
              f"{baseline2['startup_cost']-result2['startup_cost']:>14,.0f}  "
              f"{saving_pct2:+.1f}%")

        # 业务价值
        annual_saving = saving2 * 365
        print(f"\n  业务价值:")
        print(f"    日节省: ${saving2:,.0f}")
        print(f"    年节省: ${annual_saving:,.0f} (×365天)")
        if annual_saving > 1e6:
            print(f"           ≈ ${annual_saving/1e6:.0f} 百万/年")

    # ============================================================
    # 规模对比汇总
    # ============================================================
    print(f"\n{'═'*60}")
    print(f"  规模对比汇总")
    print(f"{'═'*60}")
    print(f"  {'测试规模':<16s} {'变量数':>8s} {'约束数':>8s} "
          f"{'状态':>10s} {'耗时':>8s}")
    print(f"  {'─'*56}")
    print(f"  {'5台×12h':<16s} {result1['total_vars']:>8d} "
          f"{result1['total_constrs']:>8d} {result1['status']:>10s} "
          f"{result1['solve_time']:>7.1f}s")
    print(f"  {'10台×24h':<16s} {result2['total_vars']:>8d} "
          f"{result2['total_constrs']:>8d} {result2['status']:>10s} "
          f"{result2['solve_time']:>7.1f}s")

    print(f"\n{'═'*60}")
    print(f"  完成。")
    print(f"  更多解释见: 06-case-energy.md")
    print(f"{'═'*60}\n")
