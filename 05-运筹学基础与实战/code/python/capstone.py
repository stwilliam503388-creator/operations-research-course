#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
毕业项目：智能供应链网络设计 — 骨架实现
=============================================
组合技术：工厂选址(MIP) + 运输优化(LP/网络流) + 需求仿真(敏感性分析)

教学点：
  1. MIP 建模 — 0/1 变量 + Big-M 约束
  2. LP 建模 — 流量守恒 + 多级运输
  3. 蒙特卡洛仿真 — 不确定性评估
  4. 方案对比 — 优化方案 vs 经验方案

作者：OR Course
"""


# 教学注释：先识别业务对象，再看它们如何映射为优化、仿真或启发式模型。
# 结果解读侧重成本、资源利用率和服务水平等管理指标。



import numpy as np

# ============================================================
# 1. 数据定义
# ============================================================

# --- 工厂候选地 ---
# 10 个城市：每个有建厂投资（万元）、年运营成本（万元/年）、产能（吨/年）
FACTORIES = {
    "北京":  {"invest": 8000, "operate": 1200, "capacity": 300000},
    "上海":  {"invest": 9000, "operate": 1400, "capacity": 350000},
    "广州":  {"invest": 7000, "operate": 1100, "capacity": 280000},
    "深圳":  {"invest": 7500, "operate": 1200, "capacity": 300000},
    "武汉":  {"invest": 5000, "operate": 800,  "capacity": 220000},
    "成都":  {"invest": 4800, "operate": 750,  "capacity": 200000},
    "西安":  {"invest": 4500, "operate": 700,  "capacity": 180000},
    "郑州":  {"invest": 4200, "operate": 650,  "capacity": 200000},
    "沈阳":  {"invest": 4000, "operate": 600,  "capacity": 170000},
    "昆明":  {"invest": 3800, "operate": 550,  "capacity": 160000},
}

FACTORY_NAMES = list(FACTORIES.keys())
N_FACTORIES = len(FACTORY_NAMES)

# 建厂投资折旧年限（年）
AMORT_YEARS = 10

# 总投资预算（万元）
BUDGET = 30000

# 最少/最多建厂数
MIN_FACTORIES = 3
MAX_FACTORIES = 6

# --- 配送中心（仓库）---
WAREHOUSES = {
    "华东仓": {"capacity": 200000},
    "华南仓": {"capacity": 180000},
    "华北仓": {"capacity": 150000},
    "华中仓": {"capacity": 160000},
    "西南仓": {"capacity": 140000},
}
WH_NAMES = list(WAREHOUSES.keys())
N_WH = len(WH_NAMES)

# --- 客户城市 ---
# 50 个城市，每个有年均需求量（吨）和标准差
np.random.seed(42)
CUSTOMER_NAMES = [f"客户{c+1:02d}" for c in range(50)]
N_CUSTOMERS = len(CUSTOMER_NAMES)

# 基准需求：均匀分布在 1000~8000 吨
BASE_DEMAND = np.random.uniform(1000, 8000, N_CUSTOMERS).astype(int)
# 标准差 = 15% 基准需求
DEMAND_STD = (BASE_DEMAND * 0.15).astype(int)

# --- 运输成本 ---
# 工厂→仓库 每吨成本（元/吨）——粗略按距离模拟
np.random.seed(123)
COST_FW = np.random.uniform(80, 300, (N_FACTORIES, N_WH))

# 仓库→客户 每吨成本（元/吨）
np.random.seed(456)
COST_WC = np.random.uniform(50, 250, (N_WH, N_CUSTOMERS))


# ============================================================
# 2. 子问题 1：工厂选址（MIP）
# ============================================================

class FactoryLocationSolver:
    """
    工厂选址 MIP 模型
    - 决策变量：build[f] ∈ {0,1}, produce[f] ≥ 0
    - 约束：投资预算、产能覆盖、建厂数限制
    - 目标：最小化年化成本
    """

    def __init__(self, factories=FACTORIES, budget=BUDGET,
                 min_factories=MIN_FACTORIES, max_factories=MAX_FACTORIES,
                 amort_years=AMORT_YEARS):
        self.factories = factories
        self.factory_names = list(factories.keys())
        self.n = len(self.factory_names)
        self.budget = budget
        self.min_factories = min_factories
        self.max_factories = max_factories
        self.amort_years = amort_years

    def solve(self, total_demand):
        """
        使用枚举 + 线性规划思想求解工厂选址（教学版，不依赖外部求解器）

        注意：真实场景应使用 Gurobi / COPT / HiGHS 等 MIP 求解器。
        这里用枚举所有建厂组合（≤2^10=1024 种）+ 精确求解运输成本。
        """
        print(f"\n=== 子问题 1: 工厂选址 (MIP) ===")
        print(f"候选地: {self.n}, 预算: ¥{self.budget}万")
        print(f"年总需求: {total_demand/10000:.1f}万吨")

        best_cost = float('inf')
        best_build = None
        best_produce = None

        # 枚举所有建厂组合（从 min 到 max 个）
        from itertools import combinations

        for k in range(self.min_factories, self.max_factories + 1):
            for combo in combinations(range(self.n), k):
                build = np.zeros(self.n, dtype=int)
                for idx in combo:
                    build[idx] = 1

                # 检查总投资预算
                total_invest = sum(
                    self.factories[self.factory_names[i]]["invest"]
                    for i in combo
                )
                if total_invest > self.budget:
                    continue

                # 年化投资成本
                annual_invest = total_invest / self.amort_years

                # 检查总产能
                total_capacity = sum(
                    self.factories[self.factory_names[i]]["capacity"]
                    for i in combo
                )
                if total_capacity < total_demand * 1.1:
                    continue

                # 运输成本（调用子问题 2）
                transport_cost = self._solve_transport(
                    combo, total_demand
                )

                # 年运营成本
                operate_cost = sum(
                    self.factories[self.factory_names[i]]["operate"]
                    for i in combo
                )

                total_cost = annual_invest + operate_cost + transport_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_build = build.copy()
                    best_produce = np.array([
                        self.factories[self.factory_names[i]]["capacity"]
                        if i in combo else 0
                        for i in range(self.n)
                    ])

        if best_build is None:
            print("⚠️ 未找到可行建厂方案！")
            return None, None, None

        # 输出结果
        built_names = [self.factory_names[i] for i in range(self.n) if best_build[i]]
        total_invest = sum(
            self.factories[n]["invest"] for n in built_names
        )
        annual_invest = total_invest / self.amort_years
        total_capacity = sum(best_produce)

        print(f"建厂方案: {best_build}")
        print(f"建厂城市: {', '.join(built_names)}")
        print(f"总投资: ¥{total_invest:.0f}万, 年化: ¥{annual_invest:.0f}万")
        print(f"年产能: {total_capacity/10000:.1f}万吨 (需求: {total_demand/10000:.1f}万吨)")
        print(f"年总成本: ¥{best_cost:.0f}万 (含运输)")

        return best_build, best_produce, best_cost

    def _solve_transport(self, factory_indices, total_demand):
        """简化运输成本估算"""
        # 按产能比例分配需求到各工厂
        capacities = [
            self.factories[self.factory_names[i]]["capacity"]
            for i in factory_indices
        ]
        total_cap = sum(capacities)

        # 加权平均运输成本
        avg_cost_fw = np.mean([COST_FW[i].mean() for i in factory_indices])
        avg_cost_wc = np.mean(COST_WC)

        return (avg_cost_fw + avg_cost_wc) * total_demand / 10000


# ============================================================
# 3. 子问题 2：运输优化（LP/网络流）
# ============================================================

class TransportOptimizer:
    """
    运输优化 LP 模型
    - 决策变量：工厂→仓库→客户的流量
    - 约束：产能、仓库容量、流量守恒、需求满足
    - 目标：最小化运输成本
    """

    def __init__(self, factories=FACTORIES, warehouses=WAREHOUSES,
                 cost_fw=COST_FW, cost_wc=COST_WC):
        self.factory_names = list(factories.keys())
        self.n_factories = len(self.factory_names)
        self.wh_names = list(warehouses.keys())
        self.n_wh = len(self.wh_names)
        self.factories = factories
        self.warehouses = warehouses
        self.cost_fw = cost_fw
        self.cost_wc = cost_wc

    def solve(self, build, produce, demand):
        """
        求解运输问题（教学版：用贪心分配近似最优运输）

        注意：真实场景应使用网络单纯形法或 LP 求解器。
        这里用最短路径优先的贪心分配。
        """
        print(f"\n=== 子问题 2: 运输优化 (LP) ===")

        n_cust = len(demand)

        # 哪些工厂被启用
        active_factories = [i for i in range(self.n_factories) if build[i] == 1]
        active_produce = {i: produce[i] for i in active_factories}

        # 可用产能
        available = {i: float(produce[i]) for i in active_factories}

        # 仓库容量
        wh_cap = {w: self.warehouses[self.wh_names[w]]["capacity"]
                  for w in range(self.n_wh)}

        # 贪心分配：对每个客户，找成本最低的路径
        total_cost = 0.0
        flow_fw = np.zeros((self.n_factories, self.n_wh))
        flow_wc = np.zeros((self.n_wh, n_cust))
        wh_used = np.zeros(self.n_wh)

        # 对每个客户按需求随机排序（避免系统性偏差）
        customer_order = list(range(n_cust))
        np.random.shuffle(customer_order)

        for c in customer_order:
            remaining = float(demand[c])

            # 找到所有可能的工厂→仓库路径，按成本升序排列
            paths = []
            for f in active_factories:
                if available[f] <= 1e-6:
                    continue
                for w in range(self.n_wh):
                    if wh_used[w] >= wh_cap[w] - 1e-6:
                        continue
                    cost = self.cost_fw[f, w] + self.cost_wc[w, c]
                    paths.append((cost, f, w))

            if not paths:
                print(f"  ⚠️ 客户 {c}: 无法满足需求！")
                continue

            # 按成本排序
            paths.sort()

            for cost, f, w in paths:
                if remaining <= 1e-6:
                    break

                # 可运送量 = min(工厂剩余, 仓库剩余, 客户剩余)
                max_flow = min(available[f],
                               wh_cap[w] - wh_used[w],
                               remaining)
                if max_flow <= 1e-6:
                    continue

                flow_fw[f, w] += max_flow
                flow_wc[w, c] += max_flow
                available[f] -= max_flow
                wh_used[w] += max_flow
                total_cost += max_flow * cost / 10000  # 万元
                remaining -= max_flow

            if remaining > 1e-6:
                print(f"  ⚠️ 客户 {c}: 剩余需求 {remaining:.0f} 吨未满足")

        # 统计
        total_flow = np.sum(flow_wc)
        avg_unit_cost = total_cost / (total_flow / 10000) if total_flow > 0 else 0

        print(f"总运输量: {total_flow/10000:.1f}万吨")
        print(f"总运输成本: ¥{total_cost:.0f}万")
        print(f"平均每吨成本: ¥{avg_unit_cost:.2f}")

        return total_cost, flow_fw, flow_wc


# ============================================================
# 4. 子问题 3：敏感性分析（仿真）
# ============================================================

class SensitivityAnalyzer:
    """
    需求不确定性分析（蒙特卡洛仿真）
    - 假设每个客户的需求服从 N(μ, σ²)
    - 固定建厂方案，重新求解运输问题
    """

    def __init__(self, base_demand=BASE_DEMAND, demand_std=DEMAND_STD):
        self.base_demand = base_demand
        self.demand_std = demand_std
        self.n_customers = len(base_demand)

    def run_simulation(self, build, produce, n_sim=1000):
        """
        运行蒙特卡洛仿真
        """
        print(f"\n=== 子问题 3: 敏感性分析 (仿真 {n_sim} 次) ===")

        transport = TransportOptimizer()
        costs = []
        feasible_count = 0

        # 需求相关系数矩阵（简化：所有城市间 ρ = 0.3）
        corr = 0.3
        cov_matrix = np.ones((self.n_customers, self.n_customers)) * corr
        np.fill_diagonal(cov_matrix, 1.0)
        # 转换为协方差矩阵
        std_diag = np.diag(self.demand_std)
        cov = std_diag @ cov_matrix @ std_diag

        for sim in range(n_sim):
            # 生成相关多元正态需求
            demand = np.random.multivariate_normal(
                self.base_demand, cov
            )
            demand = np.maximum(demand, 0)  # 非负

            total_demand = demand.sum()
            total_capacity = sum(produce)

            if total_demand > total_capacity * 1.1:
                # 产能不足
                continue

            cost, _, _ = transport.solve(build, produce, demand)
            costs.append(cost)
            feasible_count += 1

        costs = np.array(costs)
        feasible_rate = feasible_count / n_sim * 100

        print(f"\n  仿真结果 ({n_sim} 次):")
        print(f"  方案可行性: {feasible_rate:.1f}%")
        if len(costs) > 0:
            print(f"  成本均值: ¥{np.mean(costs):.0f}万")
            print(f"  成本 P10:  ¥{np.percentile(costs, 10):.0f}万")
            print(f"  成本 P50:  ¥{np.percentile(costs, 50):.0f}万")
            print(f"  成本 P90:  ¥{np.percentile(costs, 90):.0f}万")
            print(f"  成本标准差: ¥{np.std(costs):.0f}万")

        return costs, feasible_rate


# ============================================================
# 5. 经验方案（对照组）
# ============================================================

def empirical_solution(total_demand):
    """
    经验方案：在人口最密集的 3 个城市建厂
    按最短距离运输，不考虑仓库容量
    """
    print(f"\n=== 经验方案（对照组）===")

    # 经验选厂：北京、上海、广州（人口密集地区）
    empirical_build = np.zeros(N_FACTORIES, dtype=int)
    empirical_indices = [0, 1, 2]  # 北京、上海、广州
    for i in empirical_indices:
        empirical_build[i] = 1

    # 年化投资
    total_invest = sum(FACTORIES[FACTORY_NAMES[i]]["invest"]
                       for i in empirical_indices)
    annual_invest = total_invest / AMORT_YEARS

    # 运营成本
    operate_cost = sum(FACTORIES[FACTORY_NAMES[i]]["operate"]
                       for i in empirical_indices)

    # 产能
    total_capacity = sum(FACTORIES[FACTORY_NAMES[i]]["capacity"]
                         for i in empirical_indices)

    # 运输成本估算（经验方案不做优化，直接按平均成本估算 + 20% 浪费）
    avg_transport_cost_per_ton = 350  # 元/吨（经验值）
    transport_cost = avg_transport_cost_per_ton * total_demand / 10000

    # 浪费因子 1.2（经验方案效率低）
    transport_cost *= 1.2

    total_cost = annual_invest + operate_cost + transport_cost

    print(f"建厂城市: 北京, 上海, 广州")
    print(f"年化投资: ¥{annual_invest:.0f}万")
    print(f"运营成本: ¥{operate_cost:.0f}万")
    print(f"运输成本: ¥{transport_cost:.0f}万")
    print(f"总成本:   ¥{total_cost:.0f}万")
    print(f"总产能:   {total_capacity/10000:.1f}万吨")

    return empirical_build, total_cost


# ============================================================
# 6. 主流程
# ============================================================

def main():
    print("=" * 70)
    print("智能供应链网络设计 — 毕业项目")
    print("=" * 70)

    total_demand = BASE_DEMAND.sum()

    # ---- 1. 工厂选址 ----
    loc_solver = FactoryLocationSolver()
    build, produce, loc_cost = loc_solver.solve(total_demand)

    if build is None:
        print("❌ 工厂选址失败，终止项目。")
        return

    # ---- 2. 运输优化 ----
    transport = TransportOptimizer()
    transport_cost, flow_fw, flow_wc = transport.solve(build, produce, BASE_DEMAND)

    # 总成本
    total_invest = sum(FACTORIES[FACTORY_NAMES[i]]["invest"]
                       for i in range(N_FACTORIES) if build[i])
    annual_invest = total_invest / AMORT_YEARS
    operate_cost = sum(FACTORIES[FACTORY_NAMES[i]]["operate"]
                       for i in range(N_FACTORIES) if build[i])
    total_cost = annual_invest + operate_cost + transport_cost

    print(f"\n优化方案总成本: ¥{total_cost:.0f}万/年")

    # ---- 3. 敏感性分析 ----
    analyzer = SensitivityAnalyzer()
    sim_costs, feasible_rate = analyzer.run_simulation(build, produce)

    # ---- 4. 与经验方案对比 ----
    _, empirical_cost = empirical_solution(total_demand)
    saving = (empirical_cost - total_cost) / empirical_cost * 100
    print(f"\n{'=' * 70}")
    print(f"对比总结")
    print(f"{'=' * 70}")
    print(f"经验方案成本: ¥{empirical_cost:.0f}万/年")
    print(f"优化方案成本: ¥{total_cost:.0f}万/年")
    print(f"节省:         {saving:.1f}%")
    print(f"需求场景可行性: {feasible_rate:.1f}%")
    print()

    if saving >= 15:
        print("✅ 验证通过：优化方案 ≥ 15% 成本节省！")
    else:
        print(f"⚠️ 注意：节省 {saving:.1f}%，接近但未达到 15% 目标。")
        print("   建议：尝试更多建厂组合或调整运输参数。")

    if feasible_rate >= 95:
        print("✅ 验证通过：方案在 ≥95% 需求场景下可行！")
    else:
        print(f"⚠️ 方案可行性 {feasible_rate:.1f}% < 95%，建议增加产能。")

    print(f"\n{'=' * 70}")
    print("提示: 这是一个教学演示骨架。")
    print("生产环境建议：")
    print("  1. 使用 Gurobi / COPT / HiGHS 求解 MIP 和 LP")
    print("  2. 使用真实距离数据计算运输成本")
    print("  3. 加入多商品流、库存成本、碳约束等扩展")
    print("  4. 用 matplotlib 绘制供应链网络地图")
    print("=" * 70)


if __name__ == "__main__":
    main()
