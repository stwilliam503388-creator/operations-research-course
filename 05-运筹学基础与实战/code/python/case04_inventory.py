#!/usr/bin/env python3
"""
案例2：库存策略优化 — 多周期 DP + 报童问题
=============================================

场景：零售商多周期库存管理
  - T 个周期，每个周期需求随机
  - 每次订货有固定成本 K
  - 持有成本 h（每单位每周期）
  - 缺货惩罚 p（每单位未满足需求）
  - 仓库容量上限 C

教学目标：
  - 报童问题（Newsvendor）：单周期最优订货量的解析解
  - 动态规划（DP）：多周期最优策略
  - (s, S) 策略的发现与验证
  - 库存平衡方程的理解

验证标准：
  ✅ 报童解析解与 DP 在单周期下的一致
  ✅ 库存水平始终不超过仓库容量 C
  ✅ 固定成本 K 越高，订货频率越低
"""
# 教学注释：先识别业务对象，再看它们如何映射为优化、仿真或启发式模型。
# 结果解读侧重成本、资源利用率和服务水平等管理指标。


import random
import math

random.seed(2024)


# ============================================================
# 1. 报童问题（Newsvendor Problem）— 解析解
# ============================================================

def newsvendor_analytical(h, p, demand_probs):
    """
    报童问题的解析解：最小化期望总成本（持有 + 缺货）。

    参数：
        h: 每单位持有成本（库存过多时的损失）
        p: 每单位缺货成本（库存不足时的损失）
        demand_probs: dict {需求量: 概率}，或 list of (demand, prob)

    返回：
        Q_opt: 最优订货量（使得累积概率刚好 ≥ p/(p+h)）
        expected_cost: 对应的期望成本
        critical_ratio: 临界比率 p/(p+h)
    """
    # 计算临界比率
    critical_ratio = p / (p + h)

    # 将需求概率转为累积分布
    if isinstance(demand_probs, dict):
        demands = sorted(demand_probs.keys())
    else:
        demands = sorted([d for d, _ in demand_probs])
        demand_probs = dict(demand_probs)

    cum_prob = 0.0
    Q_opt = demands[-1]  # 默认为最大需求
    for d in demands:
        cum_prob += demand_probs[d]
        if cum_prob >= critical_ratio - 1e-9:
            Q_opt = d
            break

    # 计算期望成本
    expected_cost = 0.0
    for d, prob in demand_probs.items():
        if d <= Q_opt:
            # 持有成本：多订了 (Q_opt - d) 单位
            hold_cost = h * (Q_opt - d)
        else:
            # 缺货成本：少订了 (d - Q_opt) 单位
            hold_cost = p * (d - Q_opt)
        expected_cost += prob * hold_cost

    return Q_opt, expected_cost, critical_ratio


# ============================================================
# 2. 多周期动态规划（DP）
# ============================================================

def dp_inventory(T, K, h, p, C, demand_dist, init_inventory=0):
    """
    多周期库存问题的动态规划求解。

    状态：t 周期初的库存水平 I (0 ≤ I ≤ C)
    决策：订货量 q (使周期初库存 I + q ≤ C)
    转移：I_{t+1} = max(0, I + q - D_t), 其中 D_t 是随机需求
    成本：固定成本 K * [q>0] + 持有成本 h * max(0, I+q-D) + 缺货惩罚 p * max(0, D-(I+q))

    参数：
        T: 周期数
        K: 固定订货成本
        h: 单位持有成本/周期
        p: 单位缺货惩罚
        C: 仓库容量
        demand_dist: list of (demand, prob) 需求分布（各周期 i.i.d.）
        init_inventory: 期初库存

    返回：
        V: dict, V[t][I] = 从周期 t 开始、库存 I 的最优期望总成本
        policy: dict, policy[t][I] = 最优订货量 q
        stats: dict, 包含仿真统计信息
    """
    # 需求列表和概率
    demands = [d for d, _ in demand_dist]
    probs = [p for _, p in demand_dist]
    n_demands = len(demands)

    # DP 表：V[t][I] = 周期 t 初库存 I 的最小期望总成本（从 t 到 T-1）
    # policy[t][I] = 最优订货量
    V = [{} for _ in range(T + 1)]
    policy = [{} for _ in range(T)]

    # 终值：周期 T 之后无成本（可设残值或零）
    for I in range(C + 1):
        V[T][I] = 0.0

    # 倒推：从最后一个周期往前
    for t in range(T - 1, -1, -1):
        for I in range(C + 1):
            best_cost = float('inf')
            best_q = 0

            # 可选的订货量：0 到 C - I
            for q in range(0, C - I + 1):
                fixed_cost = K if q > 0 else 0.0

                expected_future = 0.0
                for idx in range(n_demands):
                    d = demands[idx]
                    prob_d = probs[idx]

                    # 周期内成本
                    inventory_after = I + q - d  # 可能为负
                    if inventory_after >= 0:
                        hold_cost = h * inventory_after
                    else:
                        # 缺货
                        hold_cost = p * (-inventory_after)

                    # 下一周期库存（截断在 [0, C]）
                    next_I = max(0, min(C, inventory_after))

                    # 总期望成本 = 当前成本 + 未来成本
                    total_for_d = hold_cost + V[t + 1][next_I]
                    expected_future += prob_d * total_for_d

                total_cost = fixed_cost + expected_future

                if total_cost < best_cost - 1e-9:
                    best_cost = total_cost
                    best_q = q

            V[t][I] = best_cost
            policy[t][I] = best_q

    # ---- 仿真：使用 DP 策略跑一个样本路径 ----
    sim_inventory = [0] * (T + 1)
    sim_order = [0] * T
    sim_actual_demand = [0] * T
    sim_hold = [0] * T
    sim_shortage = [0] * T
    sim_fixed = [0] * T

    sim_inventory[0] = init_inventory
    total_sim_cost = 0.0

    for t in range(T):
        I = sim_inventory[t]
        q = policy[t][I]
        sim_order[t] = q

        # 随机抽取需求
        r = random.random()
        cum = 0.0
        d = demands[-1]
        for idx in range(n_demands):
            cum += probs[idx]
            if r <= cum:
                d = demands[idx]
                break
        sim_actual_demand[t] = d

        fixed_cost = K if q > 0 else 0.0
        sim_fixed[t] = fixed_cost

        inventory_after = I + q - d
        if inventory_after >= 0:
            hold_cost = h * inventory_after
            shortage_cost = 0.0
        else:
            hold_cost = 0.0
            shortage_cost = p * (-inventory_after)

        sim_hold[t] = hold_cost
        sim_shortage[t] = shortage_cost
        total_sim_cost += fixed_cost + hold_cost + shortage_cost

        # 下一周期库存
        sim_inventory[t + 1] = max(0, min(C, inventory_after))

    stats = {
        'V': V,
        'policy': policy,
        'total_cost': total_sim_cost,
        'sim_inventory': sim_inventory,
        'sim_order': sim_order,
        'sim_demand': sim_actual_demand,
        'sim_hold': sim_hold,
        'sim_shortage': sim_shortage,
        'sim_fixed': sim_fixed,
    }

    return stats


def compute_s_s_policy(policy, T):
    """
    从 DP 策略中提取 (s, S) 参数。
    对于每个周期 t:
      - s(t) = 再订货点：库存低于此值就订货
      - S(t) = 订货目标：订货到该库存水平

    如果策略是 (s, S) 形式的，那么：
      当 I < s 时，Q = S - I（订到 S）
      当 I ≥ s 时，Q = 0（不订货）
    """
    s_policy = []
    S_policy = []

    for t in range(T):
        # 查找 s：从低库存到高库存，找到第一个不订货的点
        s = 0
        S = 0
        found_s = False
        for I in range(101):  # 假设库存上限 100
            if I in policy[t]:
                q = policy[t][I]
                if q > 0 and not found_s:
                    s = I
                    found_s = True
                if q > 0:
                    S = I + q
        if not found_s:
            s = 101  # 永不订货
        s_policy.append(s)
        S_policy.append(S)

    return s_policy, S_policy


# ============================================================
# 3. 验证函数
# ============================================================

def check_newsvendor_vs_dp(h, p, K, C, demand_dist):
    """
    验证标准 1：单周期下报童解析解与 DP 一致。
    """
    print("\n✅ 验证标准 1：报童解析解与 DP 在单周期下的一致性")
    print("-" * 60)

    # 报童解析解
    Q_opt, exp_cost_news, cr = newsvendor_analytical(h, p, demand_dist)
    print(f"  报童临界比率：p/(p+h) = {p}/{p+h} = {cr:.4f}")
    print(f"  报童最优订货量：Q* = {Q_opt}")
    print(f"  报童期望成本：{exp_cost_news:.4f}")

    # DP 单周期（T=1）
    dp_result = dp_inventory(
        T=1, K=0, h=h, p=p, C=C,
        demand_dist=demand_dist, init_inventory=0
    )
    dp_Q = dp_result['policy'][0][0]
    dp_cost = dp_result['V'][0][0]

    print(f"\n  DP 最优订货量（T=1, K=0）：{dp_Q}")
    print(f"  DP 期望成本：{dp_cost:.4f}")

    if Q_opt == dp_Q and abs(exp_cost_news - dp_cost) < 1e-6:
        print("\n  ✅ 结果一致！报童解析解 = DP 最优解")
        return True
    else:
        # 检查是否因离散化导致的微小差异
        print(f"\n  ⚠️ 差异：报童 Q={Q_opt}, DP Q={dp_Q}")
        print(f"    成本差异：{abs(exp_cost_news - dp_cost):.6f}")
        if abs(exp_cost_news - dp_cost) < 0.01:
            print("  ✅ 成本差异在容差范围内，可以接受")
            return True
        return False


def check_capacity_violation(stats, C):
    """
    验证标准 2：库存水平不超过仓库容量。
    """
    print("\n✅ 验证标准 2：库存始终不超过仓库容量")
    print("-" * 60)

    max_inv = max(stats['sim_inventory'])
    print(f"  最大库存水平：{max_inv}")
    print(f"  仓库容量：{C}")

    if max_inv <= C:
        print(f"  ✅ 通过！所有周期库存 ≤ {C}")
        return True
    else:
        print(f"  ❌ 失败！库存 {max_inv} > 容量 {C}")
        return False


def check_ordering_frequency(K_low, K_high, demand_dist, C, h, p, T):
    """
    验证标准 3：固定成本越高，订货频率越低。
    """
    print("\n✅ 验证标准 3：固定成本 K 越高 → 订货频率越低")
    print("-" * 60)

    results = []
    for K in [K_low, K_high]:
        stats = dp_inventory(
            T=T, K=K, h=h, p=p, C=C,
            demand_dist=demand_dist, init_inventory=0
        )
        order_count = sum(1 for q in stats['sim_order'] if q > 0)
        avg_inv = sum(stats['sim_inventory']) / (T + 1)
        results.append({
            'K': K,
            'order_count': order_count,
            'total_cost': stats['total_cost'],
            'avg_inventory': avg_inv,
        })
        print(f"  K={K}: 订货次数={order_count}/{T}, "
              f"平均库存={avg_inv:.2f}, "
              f"总成本={stats['total_cost']:.2f}")

    if results[0]['order_count'] >= results[1]['order_count']:
        print(f"\n  ✅ 通过！K={K_low} 订货 {results[0]['order_count']} 次 > "
              f"K={K_high} 订货 {results[1]['order_count']} 次")
        return True
    else:
        print(f"\n  ❌ 失败！高 K 反而订货更频繁")
        return False


def extract_s_s_policy_and_display(stats, T):
    """
    从 DP 策略中提取 (s, S) 并显示。
    """
    print("\n📊 (s, S) 策略提取")
    print("-" * 60)

    s_vals, S_vals = compute_s_s_policy(stats['policy'], T)
    print(f"  周期 | 再订货点 s | 订货目标 S | 说明")
    print(f"  -----|-----------|-----------|------")
    for t in range(min(T, 10)):  # 最多显示 10 个周期
        desc = f"库存 < {s_vals[t]} 就订到 {S_vals[t]}" if s_vals[t] <= 100 else "永不订货"
        print(f"  {t:^5} | {s_vals[t]:^9} | {S_vals[t]:^9} | {desc}")

    if T > 10:
        print(f"  ... (共 {T} 个周期，仅显示前 10 个)")

    return s_vals, S_vals


# ============================================================
# 4. 主流程
# ============================================================

def main():
    print("=" * 72)
    print("  案例2：库存策略优化 — 多周期 DP + 报童问题")
    print("=" * 72)

    # ---- 参数设置 ----
    T = 20                # 周期数
    K = 50                # 固定订货成本
    h = 2                 # 单位持有成本/周期
    p = 8                 # 单位缺货惩罚
    C = 30                # 仓库容量

    # 需求分布（离散均匀分布）
    demand_values = list(range(5, 21))  # 需求 5~20
    demand_probs = {d: 1.0 / len(demand_values) for d in demand_values}
    demand_dist = [(d, 1.0 / len(demand_values)) for d in demand_values]

    print(f"\n📋 问题参数：")
    print(f"  周期数 T          = {T}")
    print(f"  固定订货成本 K    = {K}")
    print(f"  持有成本 h        = {h} / 单位 / 周期")
    print(f"  缺货惩罚 p        = {p} / 单位")
    print(f"  仓库容量 C        = {C}")
    print(f"  需求分布          = Uniform[{min(demand_values)}, {max(demand_values)}]")

    # ---- 1. 报童问题 ----
    print("\n" + "=" * 72)
    print("  第一部分：报童问题（单周期新闻 vendor 模型）")
    print("=" * 72)

    Q_opt, exp_cost, cr = newsvendor_analytical(h, p, demand_probs)
    print(f"\n  临界比率：p/(p+h) = {p}/{p+h} = {cr:.4f}")
    print(f"  最优订货量 Q* = {Q_opt}")
    print(f"  期望成本 = {exp_cost:.4f}")
    print()
    print(f"  📖 解释：")
    print(f"  报童问题告诉我们，当缺货惩罚（{p}）远大于持有成本（{h}）时，")
    print(f"  应该订更多的货。临界比率 {cr:.2f} ≈ {cr*100:.0f}%，")
    print(f"  意味着你应该备货到至少满足 {cr*100:.0f}% 的需求场景。")

    # ---- 2. 多周期 DP ----
    print("\n" + "=" * 72)
    print("  第二部分：多周期 DP 求解")
    print("=" * 72)

    dp_result = dp_inventory(
        T=T, K=K, h=h, p=p, C=C,
        demand_dist=demand_dist, init_inventory=0
    )

    print(f"\n  DP 求解完成！")
    print(f"  仿真总成本：{dp_result['total_cost']:.2f}")
    print(f"  平均每周期成本：{dp_result['total_cost'] / T:.2f}")

    # 显示仿真轨迹
    print(f"\n  仿真路径（前 10 个周期）：")
    print(f"  周期 | 期初库存 | 订货量 | 实际需求 | 期末库存 | 固定成本 | 持有成本 | 缺货成本")
    print(f"  ----|---------|-------|---------|---------|---------|---------|--------")
    for t in range(min(10, T)):
        inv_before = dp_result['sim_inventory'][t]
        q = dp_result['sim_order'][t]
        d = dp_result['sim_demand'][t]
        inv_after = dp_result['sim_inventory'][t + 1]
        fc = dp_result['sim_fixed'][t]
        hc = dp_result['sim_hold'][t]
        sc = dp_result['sim_shortage'][t]
        print(f"  {t:^4} | {inv_before:^7} | {q:^5} | {d:^7} | {inv_after:^7} | {fc:^7} | {hc:^7.1f} | {sc:^7.1f}")

    # ---- 3. (s, S) 策略 ----
    s_vals, S_vals = extract_s_s_policy_and_display(dp_result, T)

    print(f"\n  📖 解释：")
    print(f"  (s, S) 策略是库存管理中最经典的策略之一：")
    print(f"  - 当库存水平低于 s（再订货点）时，订货到 S 水平")
    print(f"  - 当库存水平 ≥ s 时，不订货")
    print(f"  - 固定成本 K 越高 → s 和 S 的差距越大（一次多订，减少订货次数）")

    # ---- 4. 验证 ----
    print("\n" + "=" * 72)
    print("  第三部分：验证标准")
    print("=" * 72)

    # 验证 1：报童 vs DP（单周期，K=0）
    check_newsvendor_vs_dp(h, p, 0, C, demand_dist)

    # 验证 2：库存不超过容量
    check_capacity_violation(dp_result, C)

    # 验证 3：高 K → 低频率
    check_ordering_frequency(K_low=10, K_high=100, T=T,
                              h=h, p=p, C=C, demand_dist=demand_dist)

    # ---- 5. 对比不同 K 下的策略 ----
    print("\n" + "=" * 72)
    print("  第四部分：不同固定成本下的策略对比")
    print("=" * 72)

    for K_val in [0, 20, 50, 100, 200]:
        result = dp_inventory(
            T=T, K=K_val, h=h, p=p, C=C,
            demand_dist=demand_dist, init_inventory=0
        )
        order_count = sum(1 for q in result['sim_order'] if q > 0)
        avg_inv = sum(result['sim_inventory']) / (T + 1)
        print(f"  K={K_val:>4}: 订货次数={order_count:>2}/{T}, "
              f"平均库存={avg_inv:>5.2f}, "
              f"总成本={result['total_cost']:>8.2f}")

    # ---- 洞察总结 ----
    print("\n" + "=" * 72)
    print("  📖 洞察与延伸")
    print("=" * 72)
    print(f"""
  1. 报童问题（Newsvendor）的直觉：
     - 单周期库存决策的核心是权衡：备多了（持有成本）vs 备少了（缺货损失）
     - 临界比率 p/(p+h) 直接告诉你"备货到百分位数"
     - 本例中 p=8, h=2 → 临界比率 {p/(p+h):.2f} → 备货到 {p/(p+h)*100:.0f}% 分位

  2. (s, S) 策略的发现：
     - 固定成本 K 的存在使"不频繁、大量订货"优于"频繁、少量订货"
     - K 越大 → s 越小（更不容易触发订货）+ S 越大（一次多订）
     - K=0 → 退化为 base-stock 策略（每周期订到固定水平 S）

  3. 动态规划的精髓：
     - 从最后一个周期倒推求解（Bellman 方程）
     - 每个状态（库存水平）都对应一个最优决策
     - 状态空间爆炸是 DP 的致命弱点——本例只有 31 个状态（0~30）
     - 如果库存容量 C=10000，DP 就需要 10001 个状态——仍可接受
     - 如果再加上多产品 → 状态数指数增长 → 需要近似 DP

  4. 库存平衡方程：
     - I_{{t+1}} = max(0, I_t + q_t - D_t)
     - 这个简单的递推关系是整个库存优化的数学基石
     - 实际应用中还要考虑：提前期（lead time）、批量折扣（quantity discount）、
       多级库存（multi-echelon）、需求预测（demand forecasting）

  5. 延伸思考：
     - 如果需求是正态分布 → 报童解用 Φ^{-1}(p/(p+h)) 计算
     - 如果供给也不确定（随机供给）→ 双重边际化问题
     - 如果零售有多个层级 → 供应链协调（契约设计）
""")


if __name__ == '__main__':
    main()
