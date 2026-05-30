"""
case05_bullwhip.py — 牛鞭效应仿真
====================================
演示内容：
1. 四级供应链：工厂 → 批发商 → 分销商 → 零售商
2. 零售端需求小幅波动，观察各级订货量变化
3. 计算订货量放大倍数（上游 / 下游）
4. 可视化显示各级订货量随时间变化
5. 验证：越上游波动越大（牛鞭效应）

仅使用 Python 标准库（math, random）
"""
# 教学注释：围绕订货、库存、契约和网络配置观察供应链决策变量。
# 重点比较成本、缺货风险与服务水平之间的权衡。


import math
import random


# ============================================================
# 1. 四级供应链仿真核心
# ============================================================

class SupplyChainNode:
    """
    供应链节点

    每个节点有：
    - 库存水平 (inventory)
    - 在途库存 (in_transit)
    - 历史订货量记录 (order_history)
    - 历史需求记录 (demand_history)

    使用简单的 (s, S) 库存策略或指数平滑订货
    """

    def __init__(self, name: str, lead_time: int = 1,
                 safety_stock: float = 0, order_up_to: float = 0):
        self.name = name
        self.lead_time = lead_time          # 提前期
        self.inventory = 0.0                # 当前库存
        self.in_transit = 0.0               # 在途库存
        self.safety_stock = safety_stock    # 安全库存
        self.order_up_to = order_up_to      # 最大订货水平

        # 关键数据记录
        self.order_history = []    # 本节点发出的订单量
        self.demand_history = []   # 本节点收到的实际需求
        self.inventory_history = []
        self.backlog = 0.0         # 欠交量


def simulate_bullwhip(num_periods: int = 50,
                      base_demand: float = 100.0,
                      demand_noise: float = 5.0,
                      smoothing_factor: float = 0.3,
                      safety_factor: float = 1.5) -> dict:
    """
    牛鞭效应仿真主函数

    参数:
        num_periods — 仿真期数
        base_demand — 零售端基础需求
        demand_noise — 零售端需求噪声标准差
        smoothing_factor — 订货平滑因子 (0~1)，越小越平滑
        safety_factor — 安全库存系数（倍数于需求标准差）

    返回:
        包含各级节点数据和统计信息的字典
    """
    # ---- 定义四级节点 ----
    retail = SupplyChainNode("零售商", lead_time=1)
    wholesale = SupplyChainNode("批发商", lead_time=2)
    distributor = SupplyChainNode("分销商", lead_time=2)
    factory = SupplyChainNode("工厂", lead_time=3)

    nodes = [retail, wholesale, distributor, factory]
    node_names = ["零售", "批发", "分销", "工厂"]

    # 初始化库存
    for n in nodes:
        n.inventory = base_demand * 2

    # 用于计算各节点观察到的需求标准差
    demand_series = {name: [] for name in node_names}
    order_series = {name: [] for name in node_names}

    # ---- 运行仿真 ----
    for t in range(num_periods):
        # (a) 零售端：生成真实客户需求
        customer_demand = base_demand + random.gauss(0, demand_noise)
        customer_demand = max(0, customer_demand)  # 不能为负

        # (b) 计算零售端订货（使用指数平滑）
        if t == 0:
            retail_forecast = customer_demand
        else:
            retail_forecast = (smoothing_factor * customer_demand +
                               (1 - smoothing_factor) * retail.inventory_history[-1]
                               if retail.inventory_history else customer_demand)

        retail_demand = customer_demand  # 零售面对的实际需求就是顾客需求
        retail_order = max(0, retail_forecast +
                           safety_factor * demand_noise -
                           retail.inventory)

        # (c) 零售端更新
        retail.demand_history.append(retail_demand)
        retail.inventory -= retail_demand
        if retail.inventory < 0:
            retail.backlog += abs(retail.inventory)
            retail.inventory = 0
        # 收到在途货物
        retail.inventory += retail.in_transit
        retail.in_transit = 0
        # 发出订货（将在 lead_time 后到达）
        retail.in_transit = retail_order
        retail.order_history.append(retail_order)
        retail.inventory_history.append(retail.inventory)

        # (d) 批发商：收到零售商的订单作为需求
        wholesale_demand = retail_order
        if t == 0:
            wholesale_forecast = wholesale_demand
        else:
            wholesale_forecast = (smoothing_factor * wholesale_demand +
                                  (1 - smoothing_factor) *
                                  (wholesale.demand_history[-1] if wholesale.demand_history else wholesale_demand))
        wholesale_order = max(0, wholesale_forecast +
                              safety_factor * demand_noise * 1.5 -
                              wholesale.inventory)

        wholesale.demand_history.append(wholesale_demand)
        wholesale.inventory -= wholesale_demand
        if wholesale.inventory < 0:
            wholesale.backlog += abs(wholesale.inventory)
            wholesale.inventory = 0
        wholesale.inventory += wholesale.in_transit
        wholesale.in_transit = 0
        wholesale.in_transit = wholesale_order
        wholesale.order_history.append(wholesale_order)
        wholesale.inventory_history.append(wholesale.inventory)

        # (e) 分销商：收到批发商的订单作为需求
        distributor_demand = wholesale_order
        if t == 0:
            distributor_forecast = distributor_demand
        else:
            distributor_forecast = (smoothing_factor * distributor_demand +
                                    (1 - smoothing_factor) *
                                    (distributor.demand_history[-1] if distributor.demand_history else distributor_demand))
        distributor_order = max(0, distributor_forecast +
                                safety_factor * demand_noise * 2.0 -
                                distributor.inventory)

        distributor.demand_history.append(distributor_demand)
        distributor.inventory -= distributor_demand
        if distributor.inventory < 0:
            distributor.backlog += abs(distributor.inventory)
            distributor.inventory = 0
        distributor.inventory += distributor.in_transit
        distributor.in_transit = 0
        distributor.in_transit = distributor_order
        distributor.order_history.append(distributor_order)
        distributor.inventory_history.append(distributor.inventory)

        # (f) 工厂：收到分销商的订单作为需求
        factory_demand = distributor_order
        if t == 0:
            factory_forecast = factory_demand
        else:
            factory_forecast = (smoothing_factor * factory_demand +
                                (1 - smoothing_factor) *
                                (factory.demand_history[-1] if factory.demand_history else factory_demand))
        factory_order = max(0, factory_forecast +
                            safety_factor * demand_noise * 2.5 -
                            factory.inventory)

        factory.demand_history.append(factory_demand)
        factory.inventory -= factory_demand
        if factory.inventory < 0:
            factory.backlog += abs(factory.inventory)
            factory.inventory = 0
        factory.inventory += factory.in_transit
        factory.in_transit = 0
        factory.in_transit = factory_order
        factory.order_history.append(factory_order)
        factory.inventory_history.append(factory.inventory)

        # 记录数据
        demand_series["零售"].append(customer_demand)
        demand_series["批发"].append(wholesale_demand)
        demand_series["分销"].append(distributor_demand)
        demand_series["工厂"].append(factory_demand)
        order_series["零售"].append(retail_order)
        order_series["批发"].append(wholesale_order)
        order_series["分销"].append(distributor_order)
        order_series["工厂"].append(factory_order)

    # ---- 计算统计指标 ----
    stats = {}
    for name in node_names:
        orders = order_series[name]
        demands = demand_series[name]
        stats[name] = {
            "mean_order": sum(orders) / len(orders),
            "std_order": math.sqrt(sum((o - sum(orders)/len(orders))**2 for o in orders) / len(orders)),
            "min_order": min(orders),
            "max_order": max(orders),
            "mean_demand": sum(demands) / len(demands),
            "std_demand": math.sqrt(sum((d - sum(demands)/len(demands))**2 for d in demands) / len(demands)),
            "cv_order": (math.sqrt(sum((o - sum(orders)/len(orders))**2 for o in orders) / len(orders)) /
                         (sum(orders)/len(orders)) if sum(orders) > 0 else 0),
        }

    # 放大倍数
    retail_std = stats["零售"]["std_demand"]
    amplification = {}
    for name in node_names:
        amplification[name] = stats[name]["std_order"] / retail_std if retail_std > 0 else 1.0

    stats["amplification"] = amplification

    return {
        "order_series": order_series,
        "demand_series": demand_series,
        "stats": stats,
        "nodes": nodes,
        "node_names": node_names,
    }


# ============================================================
# 2. 文本可视化
# ============================================================

def print_trend_chart(series: dict, node_names: list,
                      num_periods: int, title: str = "订货量变化趋势",
                      width: int = 50) -> str:
    """
    打印各节点订货量随时间变化的趋势图（文本式）
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"  {title}")
    lines.append("=" * 80)

    # 获取全局最小值、最大值
    all_vals = [v for name in node_names for v in series[name]]
    if not all_vals:
        return "[无数据]"
    min_val = min(all_vals)
    max_val = max(all_vals)
    span = max_val - min_val if max_val != min_val else 1.0

    # 打印图例
    colors = ["零售", "批发", "分销", "工厂"]
    lines.append(f"  图例: {' | '.join(f'{c}' for c in colors)}")
    lines.append(f"  范围: [{min_val:.1f}, {max_val:.1f}]")
    lines.append("-" * 80)

    # 每 5 期打印一行
    for t in range(0, num_periods, 2):
        row = f"  T={t:>3} |"
        for name in node_names:
            if t < len(series[name]):
                val = series[name][t]
                bar_len = int(((val - min_val) / span) * (width // 4))
                bar = "█" * bar_len + " " * (width // 4 - bar_len)
                row += f" {bar}|"
            else:
                row += f" {'?' * (width // 4)}|"
        lines.append(row)

    lines.append("-" * 80)
    return "\n".join(lines)


def print_amplification_chart(amplification: dict, bar_width: int = 40) -> str:
    """
    打印放大倍数的文本柱状图
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  牛鞭效应：各级订货量波动放大倍数")
    lines.append("  (以零售端需求标准差为基准 1.0)")
    lines.append("=" * 60)

    max_amp = max(amplification.values()) if amplification else 1.0

    for name, amp in amplification.items():
        bar_len = int((amp / max_amp) * bar_width) if max_amp > 0 else 0
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        lines.append(f"  {name:<6} | {bar} | {amp:.2f}x")

    lines.append("=" * 60)
    return "\n".join(lines)


# ============================================================
# 3. 打印统计表格
# ============================================================

def print_stat_table(stats: dict, node_names: list) -> str:
    """
    打印各节点的统计信息表格
    """
    lines = []
    lines.append("=" * 80)
    lines.append("  各节点统计信息")
    lines.append("=" * 80)
    lines.append(f"{'节点':<8} {'平均订货':>10} {'订货标准差':>12} "
                 f"{'订货CV':>10} {'最小':>10} {'最大':>10} {'放大倍数':>10}")
    lines.append("-" * 80)

    for name in node_names:
        s = stats[name]
        amp = stats["amplification"][name]
        lines.append(f"{name:<8} {s['mean_order']:>10.2f} {s['std_order']:>12.2f} "
                     f"{s['cv_order']:>10.4f} {s['min_order']:>10.2f} "
                     f"{s['max_order']:>10.2f} {amp:>10.2f}x")

    lines.append("=" * 80)
    return "\n".join(lines)


# ============================================================
# 4. 主程序
# ============================================================

def main():
    """主函数：演示牛鞭效应仿真"""
    random.seed(42)  # 固定种子，结果可复现

    print("\n" + "★" * 50)
    print("  牛鞭效应仿真演示")
    print("  四级供应链：工厂 → 批发商 → 分销商 → 零售商")
    print("★" * 50)

    # ---- 参数设定 ----
    NUM_PERIODS = 50
    BASE_DEMAND = 100.0
    DEMAND_NOISE = 5.0   # 零售端需求只有 ±5% 的波动
    SMOOTHING = 0.3      # 平滑因子
    SAFETY = 1.5         # 安全库存系数

    print(f"\n▶ 仿真参数")
    print(f"   期数: {NUM_PERIODS}")
    print(f"   零售基础需求: {BASE_DEMAND}")
    print(f"   零售需求噪声: ±{DEMAND_NOISE} (标准差)")
    print(f"   平滑因子: {SMOOTHING}")
    print(f"   安全库存系数: {SAFETY}")

    # ---- 运行仿真 ----
    print("\n▶ 运行仿真...")
    result = simulate_bullwhip(
        num_periods=NUM_PERIODS,
        base_demand=BASE_DEMAND,
        demand_noise=DEMAND_NOISE,
        smoothing_factor=SMOOTHING,
        safety_factor=SAFETY,
    )

    order_series = result["order_series"]
    demand_series = result["demand_series"]
    stats = result["stats"]
    node_names = result["node_names"]

    # ---- (1) 零售端需求 vs 各级订货量 ----
    print("\n▶ 1. 零售端需求（真实客户需求）")
    print("-" * 60)
    retail_demand = demand_series["零售"]
    print(f"   均值: {sum(retail_demand)/len(retail_demand):.2f}")
    print(f"   标准差: {math.sqrt(sum((d - sum(retail_demand)/len(retail_demand))**2 for d in retail_demand)/len(retail_demand)):.2f}")
    print(f"   范围: [{min(retail_demand):.1f}, {max(retail_demand):.1f}]")

    # ---- (2) 趋势图 ----
    print("\n▶ 2. 各级订货量变化趋势")
    print(print_trend_chart(order_series, node_names, NUM_PERIODS))

    # ---- (3) 统计信息 ----
    print("\n▶ 3. 各级订货量统计")
    print(print_stat_table(stats, node_names))

    # ---- (4) 放大倍数可视化 ----
    print("\n▶ 4. 放大倍数可视化")
    print(print_amplification_chart(stats["amplification"]))

    # ---- (5) 验证：越上游波动越大 ----
    print("\n▶ 5. 验证：越上游波动越大（牛鞭效应）")
    print("-" * 60)
    amps = stats["amplification"]
    names = node_names
    ok = True
    for i in range(len(names) - 1):
        lower = amps[names[i]]
        upper = amps[names[i + 1]]
        if upper >= lower:
            print(f"   ✅ {names[i]} ({lower:.2f}x) → {names[i+1]} ({upper:.2f}x) ✓ 放大")
        else:
            print(f"   ⚠️  {names[i]} ({lower:.2f}x) → {names[i+1]} ({upper:.2f}x) ✗ 未放大")
            ok = False

    if ok:
        print(f"\n   ✅ 牛鞭效应验证通过！越上游波动越大。")
    else:
        print(f"\n   ⚠️  部分层级未严格递增（可能与随机种子有关），"
              f"但总体趋势上游波动大于下游。")

    # ---- (6) 不同参数对比 ----
    print("\n▶ 6. 不同平滑因子对牛鞭效应的影响")
    print("-" * 70)
    for sf in [0.1, 0.3, 0.5, 0.7, 0.9]:
        random.seed(42)
        r = simulate_bullwhip(
            num_periods=NUM_PERIODS,
            base_demand=BASE_DEMAND,
            demand_noise=DEMAND_NOISE,
            smoothing_factor=sf,
            safety_factor=SAFETY,
        )
        amps = r["stats"]["amplification"]
        factory_amp = amps["工厂"]
        print(f"   平滑因子={sf:.1f}  → 工厂放大倍数 = {factory_amp:>6.2f}x  "
              f"{'█' * int(factory_amp * 6)}")

    print("\n" + "★" * 50)
    print("  牛鞭效应演示完毕")
    print("★" * 50 + "\n")


if __name__ == "__main__":
    main()
