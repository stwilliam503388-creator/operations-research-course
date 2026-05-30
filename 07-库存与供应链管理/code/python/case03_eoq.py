"""
case03_eoq.py — 经济订货批量（EOQ）模型
============================================
演示内容：
1. EOQ 公式计算：Q* = sqrt(2DS/H)
2. 总成本 = 订货成本 + 持有成本 + 采购成本
3. 文本柱状图展示总成本曲线
4. 灵敏度分析：参数 ±10% 对总成本的影响
5. 验证：EOQ 公式结果与枚举最优一致

仅使用 Python 标准库（math, random）
"""


# 教学注释：围绕订货、库存、契约和网络配置观察供应链决策变量。
# 重点比较成本、缺货风险与服务水平之间的权衡。



import math
import random


# ============================================================
# 1. EOQ 核心函数
# ============================================================

def eoq(D: float, S: float, H: float) -> float:
    """
    计算经济订货批量 EOQ

    参数:
        D — 年需求量
        S — 每次订货固定成本
        H — 单位年持有成本

    返回:
        EOQ 值（最优订货批量）
    """
    return math.sqrt(2 * D * S / H)


def total_cost(Q: float, D: float, S: float, H: float, P: float = 0.0) -> float:
    """
    计算对应订货批量的总成本

    总成本 = 订货成本(D/Q*S) + 持有成本(Q/2*H) + 采购成本(D*P)
    """
    ordering_cost = (D / Q) * S if Q > 0 else float('inf')
    holding_cost = (Q / 2.0) * H
    purchase_cost = D * P
    return ordering_cost + holding_cost + purchase_cost


def enumerate_best(D: float, S: float, H: float, P: float = 0.0,
                   q_min: int = 1, q_max: int = 2000) -> tuple:
    """
    通过枚举法寻找最优订货批量，用于验证 EOQ 公式

    返回:
        (最优批量, 最小总成本)
    """
    best_q = q_min
    best_tc = total_cost(q_min, D, S, H, P)
    for q in range(q_min, q_max + 1):
        tc = total_cost(q, D, S, H, P)
        if tc < best_tc:
            best_tc = tc
            best_q = q
    return best_q, best_tc


# ============================================================
# 2. 文本总成本曲线可视化
# ============================================================

def text_cost_curve(D: float, S: float, H: float, P: float,
                    q_start: int, q_end: int, step: int = 10,
                    bar_width: int = 40) -> str:
    """
    生成文本形式的总成本曲线（横向柱状条形图）
    """
    # 计算各点的总成本
    points = []
    for q in range(q_start, q_end + 1, step):
        tc = total_cost(q, D, S, H, P)
        points.append((q, tc))

    if not points:
        return "[无数据]"

    min_tc = min(p[1] for p in points)
    max_tc = max(p[1] for p in points)
    span = max_tc - min_tc

    lines = []
    lines.append("=" * 70)
    lines.append(f"总成本曲线 (D={D}, S={S}, H={H}, P={P})")
    lines.append(f"EOQ* = {eoq(D, S, H):.1f}")
    opt_q, opt_tc = enumerate_best(D, S, H, P, q_start, q_end)
    lines.append(f"枚举最优批量 = {opt_q}, 最小总成本 = {opt_tc:.2f}")
    lines.append("=" * 70)
    lines.append(f"{'批量':>6} | {'总成本':>10} | 条形")
    lines.append("-" * 70)

    for q, tc in points:
        if span < 1e-9:
            bar_len = 0
        else:
            # 成本越高条越短（希望看到谷底最长）
            ratio = (max_tc - tc) / span  # 0~1，越小成本越高
            bar_len = int(ratio * bar_width)
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        marker = " <-- EOQ" if abs(q - eoq(D, S, H)) < step * 0.9 else ""
        lines.append(f"{q:>6} | {tc:>10.2f} | {bar}{marker}")

    lines.append("=" * 70)
    return "\n".join(lines)


# ============================================================
# 3. 灵敏度分析
# ============================================================

def sensitivity_analysis(D: float, S: float, H: float, P: float,
                         delta: float = 0.1) -> str:
    """
    灵敏度分析：各参数分别变化 ±delta（默认 ±10%），
    观察 EOQ 和总成本的变化百分比

    返回:
        格式化分析报告
    """
    base_q = eoq(D, S, H)
    base_tc = total_cost(base_q, D, S, H, P)

    params = {
        "D (年需求量)": ("D", D),
        "S (订货成本)": ("S", S),
        "H (持有成本)": ("H", H),
        "P (单价)":     ("P", P),
    }

    lines = []
    lines.append("=" * 80)
    lines.append(f"灵敏度分析 (参数 ±{delta*100:.0f}%)")
    lines.append(f"基准 EOQ = {base_q:.2f}, 基准总成本 = {base_tc:.2f}")
    lines.append("=" * 80)
    lines.append(f"{'参数':<18} {'变化':>8} {'新EOQ':>10} {'EOQ变化%':>10} "
                 f"{'新总成本':>12} {'成本变化%':>10}")
    lines.append("-" * 80)

    for pname, (pkey, pval) in params.items():
        for sign, label in [(1.0, f"+{delta*100:.0f}%"), (-1.0, f"-{delta*100:.0f}%")]:
            new_val = pval * (1.0 + sign * delta)
            # 构造新的参数字典
            kwargs = {"D": D, "S": S, "H": H, "P": P}
            kwargs[pkey] = new_val
            new_q = eoq(kwargs["D"], kwargs["S"], kwargs["H"])
            new_tc = total_cost(new_q, kwargs["D"], kwargs["S"], kwargs["H"], kwargs["P"])

            q_change = (new_q - base_q) / base_q * 100
            tc_change = (new_tc - base_tc) / base_tc * 100
            lines.append(f"{pname:<18} {label:>8} {new_q:>10.2f} "
                         f"{q_change:>+9.2f}% {new_tc:>12.2f} {tc_change:>+9.2f}%")

    lines.append("=" * 80)
    return "\n".join(lines)


# ============================================================
# 4. 主程序：自测与演示
# ============================================================

def main():
    """主函数：演示 EOQ 模型的各项功能"""
    # ---- 基本参数 ----
    D = 12000      # 年需求量 12,000 件
    S = 300        # 每次订货固定成本 300 元
    H = 10         # 单位年持有成本 10 元
    P = 50         # 单位采购成本 50 元

    print("\n" + "★" * 40)
    print("  经济订货批量 (EOQ) 模型演示")
    print("★" * 40)

    # ---- (1) 基本计算 ----
    print("\n▶ 1. EOQ 基本计算")
    print("-" * 50)
    q_star = eoq(D, S, H)
    tc_star = total_cost(q_star, D, S, H, P)
    print(f"   年需求量 D      = {D}")
    print(f"   订货成本 S      = {S}")
    print(f"   持有成本 H      = {H}")
    print(f"   采购单价 P      = {P}")
    print(f"   EOQ*           = {q_star:.2f} 件/次")
    print(f"   最优订货成本    = {tc_star:.2f} 元/年")

    # ---- (2) 枚举验证 ----
    print("\n▶ 2. 枚举法验证 EOQ 公式")
    print("-" * 50)
    opt_q, opt_tc = enumerate_best(D, S, H, P, 1, 2000)
    print(f"   枚举范围: 1 ~ 2000")
    print(f"   枚举最优批量 Q  = {opt_q} 件/次")
    print(f"   枚举最小总成本  = {opt_tc:.2f} 元/年")
    match = abs(opt_q - q_star) <= 1
    print(f"   EOQ 公式结果    = {q_star:.2f}")
    print(f"   结果一致? {'✅ 是' if match else '❌ 否'}")
    if not match:
        print(f"   [警告] 偏差: {abs(opt_q - q_star):.2f}")

    # ---- (3) 总成本曲线（文本柱状图） ----
    print("\n▶ 3. 总成本曲线")
    print(text_cost_curve(D, S, H, P, 100, 1500, 50))

    # ---- (4) 灵敏度分析 ----
    print("\n▶ 4. 灵敏度分析")
    print(sensitivity_analysis(D, S, H, P, delta=0.10))

    # ---- (5) 参数变化对 EOQ 的影响演示 ----
    print("\n▶ 5. 各因素对 EOQ 的影响概览")
    print("-" * 50)
    for factor, new_val in [("D 增加50% ", D * 1.5),
                            ("D 减少50% ", D * 0.5),
                            ("S 增加50% ", S * 1.5),
                            ("S 减少50% ", S * 0.5),
                            ("H 增加50% ", H * 1.5),
                            ("H 减少50% ", H * 0.5)]:
        q_new = eoq(new_val if factor.startswith("D") else D,
                    new_val if factor.startswith("S") else S,
                    new_val if factor.startswith("H") else H)
        print(f"   {factor:<12} → EOQ = {q_new:>8.2f}")

    print("\n" + "★" * 40)
    print("  EOQ 模型演示完毕")
    print("★" * 40 + "\n")


if __name__ == "__main__":
    main()
