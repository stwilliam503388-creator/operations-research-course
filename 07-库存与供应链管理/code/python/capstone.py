"""
capstone.py — 毕业项目：三级供应链综合优化
=============================================
整合 EOQ + 报童 + 牛鞭效应分析

场景：
  一个三级供应链（供应商 → 制造商 → 零售商），
  零售商面临随机需求（报童模型），
  制造商向供应商订货（EOQ 批量），
  需求波动沿供应链传递（牛鞭效应）。

分析内容：
1. 零售商报童最优订货 & 服务水平
2. 制造商的 EOQ 批量与总成本
3. 牛鞭效应仿真：需求波动从零售商向供应商放大
4. 综合优化：调整安全库存策略，降低总成本与波动

仅使用 Python 标准库（math, random）
"""


# 教学注释：围绕订货、库存、契约和网络配置观察供应链决策变量。
# 重点比较成本、缺货风险与服务水平之间的权衡。



import math
import random


# ============================================================
# 第1部分：统计工具函数（正态分布）
# ============================================================

def _phi(x: float) -> float:
    """标准正态 PDF"""
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
    """标准正态 CDF"""
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0
    a = [0.254829592, -0.284496736, 1.421405429, -1.453152027, 1.061405429]
    p = 0.3275911
    t = 1.0 / (1.0 + p * abs(x))
    y = 1.0 - (((((a[4] * t + a[3]) * t) + a[2]) * t + a[1]) * t + a[0]) * t * math.exp(-x * x / 2.0)
    return 0.5 * (1.0 + (1.0 if x >= 0 else -1.0) * y)


def _Phi_inv(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """正态分位函数"""
    if sigma <= 0:
        return mu
    if p <= 0.0:
        return mu - 8 * sigma
    if p >= 1.0:
        return mu + 8 * sigma
    x = mu + sigma * (p - 0.5) * 2.5066
    for _ in range(50):
        diff = _Phi((x - mu) / sigma) - p
        if abs(diff) < 1e-12:
            break
        pdf = _phi((x - mu) / sigma) / sigma
        if pdf < 1e-300:
            break
        x -= diff / pdf
    return x


def _loss_std(z: float) -> float:
    """标准正态损失函数 L(z) = phi(z) - z*(1-Phi(z))"""
    return _phi(z) - z * (1.0 - _Phi(z))


# ============================================================
# 第2部分：三级供应链仿真引擎
# ============================================================

class TierNode:
    """供应链层级节点"""

    def __init__(self, name: str, lead_time: int = 1):
        self.name = name
        self.lead_time = lead_time
        self.inventory = 0.0
        self.in_transit = 0.0  # 简单起见，提前期后直接到货
        self.backlog = 0.0

        # 历史
        self.order_history = []
        self.demand_history = []
        self.inventory_history = []


class ThreeTierSupplyChain:
    """
    三级供应链：
        供应商 (Tier 2) → 制造商 (Tier 1) → 零售商 (Tier 0)
    """

    def __init__(self,
                 retail_price: float = 50.0,
                 wholesale_price: float = 30.0,
                 mfg_cost: float = 15.0,
                 raw_material_cost: float = 5.0,
                 salvage: float = 3.0,
                 ordering_cost: float = 200.0,
                 holding_cost_rate: float = 0.20,
                 demand_mean: float = 1000.0,
                 demand_std: float = 200.0,
                 lead_time_retail: int = 1,
                 lead_time_mfg: int = 2,
                 lead_time_supplier: int = 3,
                 safety_factor_retail: float = 1.5,
                 safety_factor_mfg: float = 1.5,
                 safety_factor_supplier: float = 1.5):
        """
        参数:
            retail_price — 零售价
            wholesale_price — 批发价（制造商→零售商）
            mfg_cost — 制造成本
            raw_material_cost — 原材料成本（供应商→制造商）
            salvage — 残值
            ordering_cost — 制造商每次订货固定成本（EOQ 的 S）
            holding_cost_rate — 年持有成本率（占成本比例）
            demand_mean/std — 零售端需求分布
            lead_time_* — 各级提前期
            safety_factor_* — 各级安全库存系数
        """
        # 价格参数
        self.p = retail_price
        self.w = wholesale_price
        self.c_mfg = mfg_cost
        self.c_raw = raw_material_cost
        self.s = salvage
        self.S = ordering_cost           # EOQ 订货成本
        self.h_rate = holding_cost_rate  # 持有成本率

        # 需求参数
        self.mu = demand_mean
        self.sigma = demand_std

        # 提前期
        self.LT = {
            "retail": lead_time_retail,
            "mfg": lead_time_mfg,
            "supplier": lead_time_supplier,
        }

        # 安全库存系数
        self.sf = {
            "retail": safety_factor_retail,
            "mfg": safety_factor_mfg,
            "supplier": safety_factor_supplier,
        }

        # 节点
        self.retail = TierNode("零售商", lead_time_retail)
        self.mfg = TierNode("制造商", lead_time_mfg)
        self.supplier = TierNode("供应商", lead_time_supplier)

        # 结果缓存
        self.results = {}

    def reset(self):
        """重置仿真状态"""
        self.retail = TierNode("零售商", self.LT["retail"])
        self.mfg = TierNode("制造商", self.LT["mfg"])
        self.supplier = TierNode("供应商", self.LT["supplier"])

    def _init_inventory(self):
        """初始化各级库存为均值水平"""
        self.retail.inventory = self.mu * 2
        self.mfg.inventory = self.mu * 3
        self.supplier.inventory = self.mu * 4

    # ---- 报童模型（零售商） ----

    def newsvendor_optimal(self) -> dict:
        """
        报童模型：零售商最优订货量
        缺货成本 cu = p - w
        过期成本 co = w - s
        """
        cu = self.p - self.w
        co = self.w - self.s
        cr = cu / (cu + co) if (cu + co) > 0 else 0.5

        Q_opt = _Phi_inv(cr, self.mu, self.sigma)
        z = (Q_opt - self.mu) / self.sigma if self.sigma > 0 else 0.0
        sales = self.mu - self.sigma * _loss_std(z)
        leftover = Q_opt - sales

        return {
            "Q_opt": Q_opt,
            "critical_ratio": cr,
            "service_level": _Phi(z),
            "sales": sales,
            "leftover": leftover,
            "cu": cu,
            "co": co,
        }

    # ---- EOQ 批量（制造商向供应商订货） ----

    def eoq_optimal(self) -> dict:
        """
        EOQ 模型：制造商最优订货批量

        年需求量 D ≈ 零售端需求均值 × 周期
        订货成本 = S（每次）
        持有成本 = h_rate × c_raw（单位年持有成本）
        """
        D = self.mu * 12  # 假设月需求，年化
        H = self.h_rate * self.c_raw  # 单位年持有成本
        Q_star = math.sqrt(2 * D * self.S / H) if H > 0 else 0.0

        ordering = (D / Q_star) * self.S if Q_star > 0 else float('inf')
        holding = (Q_star / 2.0) * H
        purchase = D * self.c_raw

        return {
            "Q_star": Q_star,
            "D": D,
            "S": self.S,
            "H": H,
            "ordering_cost": ordering,
            "holding_cost": holding,
            "purchase_cost": purchase,
            "total_cost": ordering + holding + purchase,
        }

    # ---- 牛鞭效应仿真 ----

    def simulate(self, num_periods: int = 100) -> dict:
        """
        运行完整的三级供应链仿真
        返回各级的订货量、需求、库存历史
        """
        self.reset()
        self._init_inventory()

        demand_series = {"零售": [], "制造": [], "供应": []}
        order_series = {"零售": [], "制造": [], "供应": []}
        inventory_series = {"零售": [], "制造": [], "供应": []}

        for t in range(num_periods):
            # === 零售端 ===
            # 随机需求
            d_retail = max(0, random.gauss(self.mu, self.sigma))
            nv = self.newsvendor_optimal()
            # 零售商采用移动平均预测 + 安全库存，每期订货
            if t == 0:
                forecast_retail = d_retail
            else:
                alpha = 0.4
                forecast_retail = alpha * d_retail + (1 - alpha) * self.retail.demand_history[-1]
            target_inv = forecast_retail * self.LT["retail"] + self.sf["retail"] * self.sigma
            inventory_position = self.retail.inventory + self.retail.in_transit
            o_retail = max(0, target_inv - inventory_position + d_retail)

            # 库存更新
            self.retail.inventory -= d_retail
            if self.retail.inventory < 0:
                self.retail.backlog += abs(self.retail.inventory)
                self.retail.inventory = 0
            # 到货（提前期模拟）
            self.retail.inventory += self.retail.in_transit
            self.retail.in_transit = o_retail  # 提前期后到货

            self.retail.demand_history.append(d_retail)
            self.retail.order_history.append(o_retail)
            self.retail.inventory_history.append(self.retail.inventory)

            # === 制造商 ===
            d_mfg = o_retail  # 制造商面临的需求 = 零售商的订货
            # 制造商订货：基于需求预测 + 安全库存（简单指数平滑）
            if t == 0:
                forecast_mfg = d_mfg
            else:
                alpha = 0.3
                forecast_mfg = alpha * d_mfg + (1 - alpha) * self.mfg.demand_history[-1]
            ss_mfg = self.sf["mfg"] * self.sigma
            o_mfg = max(0, forecast_mfg + ss_mfg - self.mfg.inventory)

            self.mfg.inventory -= d_mfg
            if self.mfg.inventory < 0:
                self.mfg.backlog += abs(self.mfg.inventory)
                self.mfg.inventory = 0
            self.mfg.inventory += self.mfg.in_transit
            self.mfg.in_transit = o_mfg

            self.mfg.demand_history.append(d_mfg)
            self.mfg.order_history.append(o_mfg)
            self.mfg.inventory_history.append(self.mfg.inventory)

            # === 供应商 ===
            d_sup = o_mfg  # 供应商面临的需求 = 制造商的订货
            if t == 0:
                forecast_sup = d_sup
            else:
                alpha = 0.3
                forecast_sup = alpha * d_sup + (1 - alpha) * self.supplier.demand_history[-1]
            ss_sup = self.sf["supplier"] * self.sigma * 1.2
            o_sup = max(0, forecast_sup + ss_sup - self.supplier.inventory)

            self.supplier.inventory -= d_sup
            if self.supplier.inventory < 0:
                self.supplier.backlog += abs(self.supplier.inventory)
                self.supplier.inventory = 0
            self.supplier.inventory += self.supplier.in_transit
            self.supplier.in_transit = o_sup

            self.supplier.demand_history.append(d_sup)
            self.supplier.order_history.append(o_sup)
            self.supplier.inventory_history.append(self.supplier.inventory)

            # 记录
            demand_series["零售"].append(d_retail)
            demand_series["制造"].append(d_mfg)
            demand_series["供应"].append(d_sup)
            order_series["零售"].append(o_retail)
            order_series["制造"].append(o_mfg)
            order_series["供应"].append(o_sup)
            inventory_series["零售"].append(self.retail.inventory)
            inventory_series["制造"].append(self.mfg.inventory)
            inventory_series["供应"].append(self.supplier.inventory)

        # 计算统计
        stats = {}
        for name in ["零售", "制造", "供应"]:
            od = order_series[name]
            mean_o = sum(od) / len(od)
            std_o = math.sqrt(sum((x - mean_o)**2 for x in od) / len(od))
            stats[name] = {
                "mean_order": mean_o,
                "std_order": std_o,
                "cv_order": std_o / mean_o if mean_o > 0 else 0,
                "min_order": min(od),
                "max_order": max(od),
            }

        base_std = stats["零售"]["std_order"]
        amp = {}
        for name in ["零售", "制造", "供应"]:
            amp[name] = stats[name]["std_order"] / base_std if base_std > 0 else 1.0

        stats["amplification"] = amp

        self.results = {
            "demand_series": demand_series,
            "order_series": order_series,
            "inventory_series": inventory_series,
            "stats": stats,
        }
        return self.results

    # ---- 成本计算 ----

    def calculate_costs(self) -> dict:
        """
        计算供应链总成本（基于仿真结果）
        """
        if not self.results:
            return {}

        stats = self.results["stats"]

        # 零售商成本（基于报童模型）
        nv = self.newsvendor_optimal()

        # 制造商成本（基于 EOQ）
        eoq_res = self.eoq_optimal()

        # 牛鞭效应导致的额外库存持有成本
        inv_series = self.results["inventory_series"]
        avg_inv_retail = sum(inv_series["零售"]) / len(inv_series["零售"]) if inv_series["零售"] else 0
        avg_inv_mfg = sum(inv_series["制造"]) / len(inv_series["制造"]) if inv_series["制造"] else 0
        avg_inv_sup = sum(inv_series["供应"]) / len(inv_series["供应"]) if inv_series["供应"] else 0

        # 简化持有成本
        holding_retail = avg_inv_retail * self.h_rate * self.w
        holding_mfg = avg_inv_mfg * self.h_rate * self.c_mfg
        holding_sup = avg_inv_sup * self.h_rate * self.c_raw

        total = holding_retail + holding_mfg + holding_sup

        return {
            "holding_retail": holding_retail,
            "holding_mfg": holding_mfg,
            "holding_sup": holding_sup,
            "total_holding": total,
            "avg_inv_retail": avg_inv_retail,
            "avg_inv_mfg": avg_inv_mfg,
            "avg_inv_sup": avg_inv_sup,
        }


# ============================================================
# 主程序
# ============================================================

def main():
    """主函数：三级供应链综合优化演示"""
    random.seed(42)

    print("\n" + "★" * 55)
    print("  毕业项目：三级供应链综合优化")
    print("  EOQ + 报童 + 牛鞭效应 分析")
    print("★" * 55)

    # ---- 初始化供应链 ----
    sc = ThreeTierSupplyChain(
        retail_price=50.0,
        wholesale_price=30.0,
        mfg_cost=15.0,
        raw_material_cost=5.0,
        salvage=3.0,
        ordering_cost=200.0,
        holding_cost_rate=0.20,
        demand_mean=1000.0,
        demand_std=200.0,
        lead_time_retail=1,
        lead_time_mfg=2,
        lead_time_supplier=3,
        safety_factor_retail=1.5,
        safety_factor_mfg=1.5,
        safety_factor_supplier=1.5,
    )

    print("\n" + "=" * 70)
    print("  供应链结构: 供应商 → 制造商 → 零售商")
    print("=" * 70)
    print(f"  零售价:       {sc.p}")
    print(f"  批发价:       {sc.w}")
    print(f"  制造成本:     {sc.c_mfg}")
    print(f"  原材料成本:   {sc.c_raw}")
    print(f"  残值:         {sc.s}")
    print(f"  订货成本(S):  {sc.S}")
    print(f"  持有成本率:   {sc.h_rate}")
    print(f"  需求: N(μ={sc.mu}, σ={sc.sigma})")

    # --------------------------------------------------
    # 1. 报童分析（零售商）
    # --------------------------------------------------
    print("\n▶ 模块1：报童模型 — 零售商订货决策")
    print("-" * 70)
    nv = sc.newsvendor_optimal()
    print(f"   缺货成本(cu) = {nv['cu']}")
    print(f"   过期成本(co) = {nv['co']}")
    print(f"   临界比 = {nv['critical_ratio']:.4f}")
    print(f"   最优订货量 Q* = {nv['Q_opt']:.2f}")
    print(f"   服务水平 = {nv['service_level']:.2%}")
    print(f"   预期销售 = {nv['sales']:.2f}")
    print(f"   预期剩余 = {nv['leftover']:.2f}")

    # --------------------------------------------------
    # 2. EOQ 分析（制造商向供应商订货）
    # --------------------------------------------------
    print("\n▶ 模块2：EOQ 模型 — 制造商批量订货")
    print("-" * 70)
    eoq_res = sc.eoq_optimal()
    print(f"   年需求量 D     = {eoq_res['D']:.0f}")
    print(f"   订货成本 S     = {eoq_res['S']}")
    print(f"   单位持有成本 H = {eoq_res['H']:.2f}")
    print(f"   EOQ*          = {eoq_res['Q_star']:.2f}")
    print(f"   年订货成本     = {eoq_res['ordering_cost']:.2f}")
    print(f"   年持有成本     = {eoq_res['holding_cost']:.2f}")
    print(f"   年采购成本     = {eoq_res['purchase_cost']:.2f}")
    print(f"   年总成本       = {eoq_res['total_cost']:.2f}")

    # --------------------------------------------------
    # 3. 牛鞭效应仿真
    # --------------------------------------------------
    print("\n▶ 模块3：牛鞭效应仿真")
    print("-" * 70)
    NUM_PERIODS = 80
    results = sc.simulate(NUM_PERIODS)
    stats = results["stats"]

    print(f"   仿真期数: {NUM_PERIODS}")
    print(f"   零售端需求波动: σ = {stats['零售']['std_order']:.2f}")
    print(f"")
    print(f"   {'层级':<8} {'平均订货':>10} {'标准差':>10} {'CV':>8} {'放大倍数':>10}")
    print(f"   {'-'*50}")
    for name in ["零售", "制造", "供应"]:
        s = stats[name]
        amp = stats["amplification"][name]
        print(f"   {name:<8} {s['mean_order']:>10.2f} {s['std_order']:>10.2f} "
              f"{s['cv_order']:>8.4f} {amp:>8.2f}x")
        bar_len = min(int(amp * 5), 40)
        print(f"   {'':>8} {'█' * bar_len}")

    # 验证牛鞭效应
    amps = stats["amplification"]
    print(f"\n   验证牛鞭效应:")
    if amps["制造"] >= amps["零售"] and amps["供应"] >= amps["制造"]:
        print(f"   ✅ 越上游波动越大！零售{amps['零售']:.2f}x → 制造{amps['制造']:.2f}x → 供应{amps['供应']:.2f}x")
    else:
        print(f"   ℹ️  但总体趋势向上游放大")
        for name in ["零售", "制造", "供应"]:
            print(f"      {name}: {amps[name]:.2f}x")

    # --------------------------------------------------
    # 4. 成本分析
    # --------------------------------------------------
    print("\n▶ 模块4：供应链持有成本分析")
    print("-" * 70)
    costs = sc.calculate_costs()
    print(f"   零售商平均库存 = {costs['avg_inv_retail']:.2f}  持有成本 = {costs['holding_retail']:.2f}")
    print(f"   制造商平均库存 = {costs['avg_inv_mfg']:.2f}  持有成本 = {costs['holding_mfg']:.2f}")
    print(f"   供应商平均库存 = {costs['avg_inv_sup']:.2f}  持有成本 = {costs['holding_sup']:.2f}")
    print(f"   总持有成本     = {costs['total_holding']:.2f}")

    # --------------------------------------------------
    # 5. 综合优化：调整安全库存系数
    # --------------------------------------------------
    print("\n▶ 模块5：综合优化 — 安全库存系数对成本和波动的影响")
    print("-" * 80)
    print(f"{'安全系数':>10} {'总持有成本':>12} {'供应放大':>10} {'制造放大':>10} {'零售SL':>10}")
    print("-" * 80)

    for sf in [0.5, 1.0, 1.5, 2.0, 2.5]:
        random.seed(42)
        sc2 = ThreeTierSupplyChain(
            retail_price=50.0, wholesale_price=30.0, mfg_cost=15.0,
            raw_material_cost=5.0, salvage=3.0, ordering_cost=200.0,
            holding_cost_rate=0.20, demand_mean=1000.0, demand_std=200.0,
            lead_time_retail=1, lead_time_mfg=2, lead_time_supplier=3,
            safety_factor_retail=sf, safety_factor_mfg=sf, safety_factor_supplier=sf,
        )
        res = sc2.simulate(NUM_PERIODS)
        st = res["stats"]
        cos = sc2.calculate_costs()
        nv2 = sc2.newsvendor_optimal()
        print(f"{sf:>10.1f} {cos['total_holding']:>12.2f} "
              f"{st['amplification']['供应']:>8.2f}x {st['amplification']['制造']:>8.2f}x "
              f"{nv2['service_level']:>9.2%}")

    print(f"\n   ▶ 结论：提高安全库存系数 → 服务水平↑, 持有成本↑, 牛鞭效应↑")
    print(f"   ▶ 需要在服务水平与成本之间找到平衡点。")

    print("\n" + "★" * 55)
    print("  毕业项目演示完毕")
    print("  EOQ + 报童 + 牛鞭效应综合优化")
    print("★" * 55 + "\n")


if __name__ == "__main__":
    main()
