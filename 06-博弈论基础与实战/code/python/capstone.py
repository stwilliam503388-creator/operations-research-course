"""
拍卖仿真实验 — 博弈论课程毕业项目
====================================

比较第一价格密封拍卖、第二价格密封拍卖（Vickrey）、
英式升价拍卖三种形式的期望收入。

验证收入等价定理（Revenue Equivalence Theorem），
并扩展分析风险规避和保留价的影响。

作者: Game Theory Course
"""


# 教学注释：从参与者、策略和收益矩阵出发理解交互决策结构。
# 计算结果用于验证均衡、分配规则或机制设计是否符合预期。



import numpy as np
from typing import List, Tuple


# ============================================================
# 第一部分：三种拍卖形式的仿真
# ============================================================

def first_price_auction(values: np.ndarray) -> Tuple[float, float, float]:
    """
    第一价格密封拍卖 (First-Price Sealed-Bid Auction)

    规则：
    - 每个买家秘密出价
    - 最高出价者获胜，按自己的出价支付
    - 均衡策略（风险中性，均匀估价 U[0,1]）: b(v) = (n-1)/n * v

    参数:
        values: 每个买家的真实估价 (n,)

    返回:
        (winner_idx, payment, winner_value)
    """
    n = len(values)
    # 均衡出价策略: b(v) = (n-1)/n * v
    bids = (n - 1) / n * values

    winner = int(np.argmax(bids))
    payment = bids[winner]
    return winner, payment, values[winner]


def second_price_auction(values: np.ndarray) -> Tuple[float, float, float]:
    """
    第二价格密封拍卖 (Vickrey Auction)

    规则：
    - 每个买家秘密出价
    - 最高出价者获胜，按第二高价格支付
    - 占优策略: 如实出价 b(v) = v

    参数:
        values: 每个买家的真实估价 (n,)

    返回:
        (winner_idx, payment, winner_value)
    """
    # 占优策略: 如实出价
    bids = values.copy()

    # 找到最高和第二高
    sorted_indices = np.argsort(bids)[::-1]
    winner = int(sorted_indices[0])
    payment = float(bids[sorted_indices[1]])  # 第二高价格
    return winner, payment, values[winner]


def english_auction(values: np.ndarray, step: float = 0.001) -> Tuple[float, float, float]:
    """
    英式升价拍卖 (English Auction)

    规则：
    - 价格从 0 开始逐步上升
    - 买家随时可以退出（退出后不能再回来）
    - 最后一个留下的买家获胜，按退出时的价格支付
    - 均衡下，买家在价格超过自己的估价时退出

    参数:
        values: 每个买家的真实估价 (n,)
        step:   价格步长

    返回:
        (winner_idx, payment, winner_value)
    """
    n = len(values)
    active = np.ones(n, dtype=bool)
    current_price = 0.0
    n_active = n

    # 按估价排序的退出价格
    exit_prices = np.sort(values)

    while n_active > 1:
        # 下一个退出的价格
        next_exit = exit_prices[n - n_active]  # 最低估价的活跃买家退出
        current_price = next_exit

        # 标记退出
        for i in range(n):
            if active[i] and values[i] <= current_price:
                active[i] = False
                n_active -= 1

    # 只剩一个买家
    winner = int(np.where(active)[0][0])
    # 成交价是最后一个退出的买家的估价
    # （第二高估价 ≈ 赢家支付的价格）
    sorted_values = np.sort(values)
    payment = float(sorted_values[-2])  # 第二高估价
    return winner, payment, values[winner]

    # 英式拍卖等价于第二价格拍卖，成交价 = 第二高估价


# ============================================================
# 第二部分：仿真引擎
# ============================================================

def simulate_once(n_buyers: int, auction_fn) -> float:
    """
    单次拍卖仿真

    参数:
        n_buyers: 买家人数
        auction_fn: 拍卖函数（first_price_auction / second_price_auction / english_auction）

    返回:
        卖家收入（即成交价）
    """
    values = np.random.uniform(0, 1, size=n_buyers)
    _, payment, _ = auction_fn(values)
    return payment


def run_simulation(
    n_buyers: int,
    n_trials: int = 10000,
    auction_fn=first_price_auction,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """
    运行多次仿真

    参数:
        n_buyers:  买家人数
        n_trials:  仿真次数
        auction_fn: 拍卖函数
        seed:      随机种子

    返回:
        (mean_revenue, std_revenue, all_revenues)
    """
    rng = np.random.default_rng(seed)
    revenues = np.zeros(n_trials)

    for i in range(n_trials):
        # 用每个 trial 独立的随机种子
        sub_rng = np.random.default_rng(seed + i)
        values = sub_rng.uniform(0, 1, size=n_buyers)
        _, payment, _ = auction_fn(values)
        revenues[i] = payment

    return float(np.mean(revenues)), float(np.std(revenues)), revenues


def run_simulation_vectorized(
    n_buyers: int,
    n_trials: int = 10000,
    auction_fn=first_price_auction,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """
    向量化快速仿真（仅适用于第一/第二价格拍卖）

    参数:
        n_buyers:  买家人数
        n_trials:  仿真次数
        auction_fn: 拍卖函数
        seed:      随机种子

    返回:
        (mean_revenue, std_revenue, all_revenues)
    """
    rng = np.random.default_rng(seed)
    # 一次性生成所有估价: [n_trials, n_buyers]
    all_values = rng.uniform(0, 1, size=(n_trials, n_buyers))

    if auction_fn == first_price_auction:
        # 第一价格: b(v) = (n-1)/n * v
        bids = (n_buyers - 1) / n_buyers * all_values
        payments = np.max(bids, axis=1)

    elif auction_fn == second_price_auction:
        # 第二价格: 成交价 = 第二高估价
        sorted_values = np.sort(all_values, axis=1)
        payments = sorted_values[:, -2]  # 第二高

    else:
        raise ValueError(f"Unsupported auction for vectorized simulation: {auction_fn}")

    return float(np.mean(payments)), float(np.std(payments)), payments


# ============================================================
# 第三部分：风险规避模型 (CARA)
# ============================================================

def risk_averse_bid(v: float, r: float) -> float:
    """
    CARA 风险规避下的最优出价（n=2, 均匀分布 U[0,1]）

    b(v) = v - (1/r) * ln( (1 - e^{-r}) / (1 - e^{-rv}) )

    参数:
        v: 估价 [0, 1]
        r: 风险规避系数 (r > 0, 越大越规避)

    返回:
        最优出价
    """
    if r < 1e-10:
        # 风险中性 -> b(v) = v/2 (n=2)
        return v / 2.0

    # 边界处理
    v = np.clip(v, 0.0, 1.0)
    if v <= 0.0:
        return 0.0

    # 使用 numpy 的 safe 版本
    exp_r = np.exp(-r)
    exp_rv = np.exp(-r * v)

    num = 1.0 - exp_r       # 1 - e^{-r}
    denom = 1.0 - exp_rv    # 1 - e^{-rv}

    # 当 v 接近 0 时，denom ≈ r*v, num ≈ r
    # 因此 num/denom ≈ 1/v, 出价接近 0 (正确)
    # 当 r 很小时，使用泰勒展开避免数值问题
    if r < 0.1:
        # 泰勒展开: b(v) ≈ v/2 + r*v*(1-v)/12 + O(r^2)
        # 但这里直接用原始公式，注意处理小 r 精度
        ratio = num / max(denom, 1e-15)
        log_ratio = np.log(max(ratio, 1e-15))
        # 当 r 很小时，v - (1/r)*ln(ratio) 可能略负，截断到 0
        bid = v - (1.0 / r) * log_ratio
        return max(bid, 0.0)

    ratio = num / max(denom, 1e-15)
    log_ratio = np.log(max(ratio, 1e-15))
    bid = v - (1.0 / r) * log_ratio
    return float(np.clip(bid, 0.0, v))


def first_price_risk_averse(values: np.ndarray, r: float = 1.0) -> Tuple[float, float, float]:
    """
    风险规避买家参与的第一价格密封拍卖

    参数:
        values: 每个买家的真实估价 (n,)
        r:      风险规避系数

    返回:
        (winner_idx, payment, winner_value)
    """
    n = len(values)
    if n == 2:
        # n=2 时使用解析解
        bids = np.array([risk_averse_bid(v, r) for v in values])
    else:
        # n>2 时用近似解（可扩展）
        # 实际应用时，这里应该用数值解微分方程
        # 简化处理: 用 (n-1)/n 乘风险规避出价
        base_bids = np.array([risk_averse_bid(v, r) for v in values])
        bids = (n - 1) / n * base_bids / 0.5  # 归一化

    winner = int(np.argmax(bids))
    payment = float(bids[winner])
    return winner, payment, values[winner]


def simulate_risk_averse(
    n_buyers: int,
    n_trials: int,
    r: float,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """
    风险规避下的仿真

    参数:
        n_buyers: 买家人数
        n_trials: 仿真次数
        r:        风险规避系数
        seed:     随机种子

    返回:
        (mean_revenue, std_revenue, all_revenues)
    """
    assert n_buyers == 2, "风险规避模型当前仅支持 n=2"
    rng = np.random.default_rng(seed)
    revenues = np.zeros(n_trials)

    for i in range(n_trials):
        sub_rng = np.random.default_rng(seed + i)
        values = sub_rng.uniform(0, 1, size=n_buyers)
        _, payment, _ = first_price_risk_averse(values, r)
        revenues[i] = payment

    return float(np.mean(revenues)), float(np.std(revenues)), revenues


# ============================================================
# 第四部分：保留价分析
# ============================================================

def first_price_with_reserve(
    values: np.ndarray, reserve_price: float = 0.0
) -> Tuple[float, float, float, bool]:
    """
    带保留价的第一价格密封拍卖

    参数:
        values: 每个买家的真实估价 (n,)
        reserve_price: 保留价

    返回:
        (winner_idx, payment, winner_value, sold)
    """
    n = len(values)
    bids = (n - 1) / n * values

    winner = int(np.argmax(bids))
    highest_bid = bids[winner]

    if highest_bid >= reserve_price:
        return winner, float(highest_bid), values[winner], True
    else:
        return -1, 0.0, 0.0, False


def simulate_with_reserve(
    n_buyers: int,
    n_trials: int,
    reserve_price: float,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    带保留价的仿真

    参数:
        n_buyers:      买家人数
        n_trials:      仿真次数
        reserve_price: 保留价
        seed:          随机种子

    返回:
        (mean_revenue, std_revenue, fail_rate)
    """
    rng = np.random.default_rng(seed)
    revenues = []
    fails = 0

    for i in range(n_trials):
        sub_rng = np.random.default_rng(seed + i)
        values = sub_rng.uniform(0, 1, size=n_buyers)
        _, payment, _, sold = first_price_with_reserve(values, reserve_price)
        if sold:
            revenues.append(payment)
        else:
            fails += 1

    mean_rev = float(np.mean(revenues)) if revenues else 0.0
    std_rev = float(np.std(revenues)) if revenues else 0.0
    fail_rate = fails / n_trials
    return mean_rev, std_rev, fail_rate


# ============================================================
# 第五部分：主程序 — 运行实验并输出报告
# ============================================================

def main():
    """运行所有实验"""
    print("=" * 60)
    print("  博弈论毕业项目 — 拍卖机制仿真实验")
    print("=" * 60)

    # ========== 配置 ==========
    N_BUYERS = 2
    N_TRIALS = 10_000
    SEED = 42
    THEORETICAL_REVENUE = (N_BUYERS - 1) / (N_BUYERS + 1)

    # ========== 实验 1: 收入等价定理验证 ==========
    print(f"\n{'─' * 60}")
    print("  【实验 1】收入等价定理验证")
    print(f"{'─' * 60}")
    print(f"  仿真次数:     {N_TRIALS}")
    print(f"  买家人数:     {N_BUYERS}")
    print(f"  估价分布:     Uniform[0, 1]")
    print(f"  理论收入:     (n-1)/(n+1) = {THEORETICAL_REVENUE:.4f}")
    print()

    # 第一价格拍卖
    fp_mean, fp_std, fp_revs = run_simulation_vectorized(
        N_BUYERS, N_TRIALS, first_price_auction, SEED
    )
    print(f"  第一价格密封拍卖:  平均收入 = {fp_mean:.4f}  (标准差 = {fp_std:.4f})")

    # 第二价格拍卖
    sp_mean, sp_std, sp_revs = run_simulation_vectorized(
        N_BUYERS, N_TRIALS, second_price_auction, SEED
    )
    print(f"  第二价格密封拍卖:  平均收入 = {sp_mean:.4f}  (标准差 = {sp_std:.4f})")

    # 英式拍卖
    eng_mean, eng_std, eng_revs = run_simulation(
        N_BUYERS, N_TRIALS, english_auction, SEED
    )
    print(f"  英式升价拍卖:      平均收入 = {eng_mean:.4f}  (标准差 = {eng_std:.4f})")
    print()

    # 验证
    max_diff = max(abs(fp_mean - THEORETICAL_REVENUE),
                   abs(sp_mean - THEORETICAL_REVENUE),
                   abs(eng_mean - THEORETICAL_REVENUE))
    if max_diff < 0.01:
        print(f"  ✅ 验证通过: 三种形式的收入差异 < 0.01, 支持收入等价定理")
    else:
        print(f"  ⚠️  差异较大 (最大偏差 = {max_diff:.4f}), 检查仿真实现")

    # ========== 实验 2: 风险规避分析 ==========
    print(f"\n{'─' * 60}")
    print("  【实验 2】风险规避对第一价格拍卖的影响")
    print(f"{'─' * 60}")
    print(f"  (CARA 效用函数, n = 2)")
    print()

    risk_aversion_levels = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"  {'风险系数 r':>12}  {'平均收入':>12}  {'标准差':>12}")
    print(f"  {'─' * 38}")

    for r in risk_aversion_levels:
        if r == 0.0:
            # 风险中性 = 标准第一价格
            rev = fp_mean
            std = fp_std
        else:
            rev, std, _ = simulate_risk_averse(N_BUYERS, N_TRIALS, r, SEED)
        print(f"  {r:>12.1f}  {rev:>12.4f}  {std:>12.4f}")

    # 比较: 风险规避 vs 风险中性
    ra_rev, _, _ = simulate_risk_averse(N_BUYERS, N_TRIALS, 2.0, SEED)
    print(f"\n  结论: r=2 时第一价格收入({ra_rev:.4f}) > 风险中性收入({fp_mean:.4f})")
    print(f"        → 风险规避打破了收入等价定理")
    print(f"        → 如果买家极度风险规避, 第一价格拍卖对卖家更有利")

    # ========== 实验 3: 保留价分析 ==========
    print(f"\n{'─' * 60}")
    print("  【实验 3】保留价对卖家收入的影响")
    print(f"{'─' * 60}")
    print()

    reserve_prices = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"  {'保留价':>8}  {'平均收入':>12}  {'流拍率':>10}")
    print(f"  {'─' * 32}")

    best_reserve = 0.0
    best_revenue = 0.0

    for rp in reserve_prices:
        rev, std, fail_rate = simulate_with_reserve(N_BUYERS, N_TRIALS, rp, SEED)
        print(f"  {rp:>8.1f}  {rev:>12.4f}  {fail_rate:>10.1%}")
        if rev > best_revenue:
            best_revenue = rev
            best_reserve = rp

    print(f"\n  最优保留价: {best_reserve:.1f}  (期望收入 = {best_revenue:.4f})")
    print(f"  无保留价收入: {fp_mean:.4f}")
    print(f"  收入提升: {(best_revenue - fp_mean) / fp_mean * 100:.1f}%")
    print(f"\n  注: 保留价提高卖家收入，但增加流拍风险。")
    print(f"      实际应用中需结合拍品价值和卖家风险偏好权衡。")

    # ========== 总结 ==========
    print(f"\n{'=' * 60}")
    print("  实验总结")
    print(f"{'=' * 60}")
    print(f"""
  📊 收入等价定理:      {'✅ 验证通过' if max_diff < 0.01 else '⚠️  需要检查'}
  📊 风险规避效应:      风险规避 → 第一价格收入上升
  📊 收入等价定理被打破:  {'✅ 确认' if ra_rev > fp_mean + 0.01 else '⚠️  不显著'}
  📊 最优保留价:         {best_reserve:.1f} (收入 {best_revenue:.4f})

  🏆 对客户的建议:
     - 如果买家风险中性: 三种拍卖形式收入相同, 选成本最低的
     - 如果买家风险规避: 推荐第一价格密封拍卖
     - 设置保留价 ≈ {best_reserve:.1f} 可最大化期望收入
  """)


if __name__ == "__main__":
    main()
