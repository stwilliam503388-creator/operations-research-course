#!/usr/bin/env python3
"""
案例1：定价博弈与囚徒困境 ★★☆☆☆

包含：
1. 收益矩阵定义与展示
2. 下划线法找纯策略纳什均衡
3. 占优策略分析
4. 帕累托差距计算
5. 重复博弈模拟（以牙还牙策略）
"""
# 教学注释：从参与者、策略和收益矩阵出发理解交互决策结构。
# 计算结果用于验证均衡、分配规则或机制设计是否符合预期。


import numpy as np
from typing import List, Tuple

# ============================================================================
# 1. 收益矩阵定义
# ============================================================================

# 策略名称
A_STRATEGIES = ["高价", "低价"]
B_STRATEGIES = ["高价", "低价"]

# 收益矩阵: payoffs[row][col] = (A的收益, B的收益)
#   row 0 = A选高价, row 1 = A选低价
#   col 0 = B选高价, col 1 = B选低价
PAYOFFS: List[List[Tuple[int, int]]] = [
    [(5, 5), (2, 8)],   # A高价
    [(8, 2), (3, 3)],   # A低价
]

# ============================================================================
# 2. 辅助函数
# ============================================================================

def print_matrix():
    """打印格式化的收益矩阵"""
    print("          ", end="")
    for j, s in enumerate(B_STRATEGIES):
        print(f"  B_{s:4s}", end="")
    print()
    for i, s_a in enumerate(A_STRATEGIES):
        print(f"A_{s_a:4s} ", end="")
        for j in range(len(B_STRATEGIES)):
            a_pay, b_pay = PAYOFFS[i][j]
            print(f" ({a_pay},{b_pay}) ", end="")
        print()
    print()


def find_best_responses() -> Tuple[List[int], List[int]]:
    """
    找双方的最优反应。

    Returns:
        a_best[i] = B选策略i时 A的最优策略索引
        b_best[j] = A选策略j时 B的最优策略索引
    """
    n_a = len(A_STRATEGIES)
    n_b = len(B_STRATEGIES)

    # A的最优反应：固定B的选择，找A的最大收益
    a_best = []
    for j in range(n_b):
        best_i = max(range(n_a), key=lambda i: PAYOFFS[i][j][0])
        a_best.append(best_i)

    # B的最优反应：固定A的选择，找B的最大收益
    b_best = []
    for i in range(n_a):
        best_j = max(range(n_b), key=lambda j: PAYOFFS[i][j][1])
        b_best.append(best_j)

    return a_best, b_best


# ============================================================================
# 3. 下划线法求纳什均衡
# ============================================================================

def find_nash_equilibria():
    """
    使用下划线法找出所有纯策略纳什均衡。

    原理：给每个参与者的最优反应收益画下划线，
    同时有下划线的格子就是纳什均衡。
    """
    print("=" * 60)
    print("下划线法找均衡")
    print("=" * 60)

    n_a = len(A_STRATEGIES)
    n_b = len(B_STRATEGIES)
    a_best, b_best = find_best_responses()

    equilibria = []

    for i in range(n_a):
        for j in range(n_b):
            a_pay, b_pay = PAYOFFS[i][j]

            # 检查A：这是否是A在B选j时的最优反应？
            a_optimal = (i == a_best[j])
            # 检查B：这是否是B在A选i时的最优反应？
            b_optimal = (j == b_best[i])

            print(f"\n检查格 ({A_STRATEGIES[i]},{B_STRATEGIES[j]}):")

            # A的最优检查
            best_a_pay = max(PAYOFFS[k][j][0] for k in range(n_a))
            if a_optimal:
                print(f"  → A的收益{a_pay} == B选{B_STRATEGIES[j]}时最大{best_a_pay} → A的最优 ✅")
            else:
                print(f"  → A的收益{a_pay} < B选{B_STRATEGIES[j]}时最大{best_a_pay} → 不是A的最优")

            # B的最优检查
            best_b_pay = max(PAYOFFS[i][k][1] for k in range(n_b))
            if b_optimal:
                print(f"  → B的收益{b_pay} == A选{A_STRATEGIES[i]}时最大{best_b_pay} → B的最优 ✅")
            else:
                print(f"  → B的收益{b_pay} < A选{A_STRATEGIES[i]}时最大{best_b_pay} → 不是B的最优")

            if a_optimal and b_optimal:
                print(f"  ✓ 纳什均衡！")
                equilibria.append((i, j))
            else:
                print(f"  ✗ 不是纳什均衡")

    return equilibria


# ============================================================================
# 4. 占优策略分析
# ============================================================================

def dominant_strategy_analysis():
    """分析双方是否有占优策略"""
    print("=" * 60)
    print("占优策略分析")
    print("=" * 60)

    n_a = len(A_STRATEGIES)
    n_b = len(B_STRATEGIES)

    # A是否有占优策略？
    print("\nA的视角：")
    a_has_dominant = True
    a_dominant_strategy = None
    for i in range(n_a):
        better_in_all = True
        for j in range(n_b):
            # 检查策略i是否在所有B的策略下都是最优的
            best_i_for_j = max(range(n_a), key=lambda k: PAYOFFS[k][j][0])
            if i != best_i_for_j:
                better_in_all = False
                break
        if better_in_all:
            a_dominant_strategy = i
            break

    for j in range(n_b):
        best_i = max(range(n_a), key=lambda k: PAYOFFS[k][j][0])
        worst_i = min(range(n_a), key=lambda k: PAYOFFS[k][j][0])
        print(f"  B选{B_STRATEGIES[j]}时：{A_STRATEGIES[best_i]}({PAYOFFS[best_i][j][0]}) > {A_STRATEGIES[worst_i]}({PAYOFFS[worst_i][j][0]})")

    if a_dominant_strategy is not None:
        print(f"  → {A_STRATEGIES[a_dominant_strategy]}是A的占优策略 ✅")
    else:
        print(f"  → A没有占优策略")

    # B是否有占优策略？
    print("\nB的视角：")
    b_has_dominant = True
    b_dominant_strategy = None
    for j in range(n_b):
        better_in_all = True
        for i in range(n_a):
            best_j_for_i = max(range(n_b), key=lambda k: PAYOFFS[i][k][1])
            if j != best_j_for_i:
                better_in_all = False
                break
        if better_in_all:
            b_dominant_strategy = j
            break

    for i in range(n_a):
        best_j = max(range(n_b), key=lambda k: PAYOFFS[i][k][1])
        worst_j = min(range(n_b), key=lambda k: PAYOFFS[i][k][1])
        print(f"  A选{A_STRATEGIES[i]}时：{B_STRATEGIES[best_j]}({PAYOFFS[i][best_j][1]}) > {B_STRATEGIES[worst_j]}({PAYOFFS[i][worst_j][1]})")

    if b_dominant_strategy is not None:
        print(f"  → {B_STRATEGIES[b_dominant_strategy]}是B的占优策略 ✅")
    else:
        print(f"  → B没有占优策略")

    if a_dominant_strategy is not None and b_dominant_strategy is not None:
        print(f"\n结论：双方都有占优策略「{A_STRATEGIES[a_dominant_strategy]}」")
        print("强制降价是各自的最佳选择")

    return a_dominant_strategy, b_dominant_strategy


# ============================================================================
# 5. 帕累托差距
# ============================================================================

def pareto_gap_analysis(equilibria: List[Tuple[int, int]]):
    """计算合作解（都高价）与纳什均衡（都低价）的效率差距"""
    print("=" * 60)
    print("帕累托差距分析")
    print("=" * 60)

    # 合作解：都选高价 (0, 0)
    coop = PAYOFFS[0][0]
    coop_total = coop[0] + coop[1]
    print(f"\n合作解（都高价）：{coop}，总收益 = {coop_total}")

    # 纳什均衡
    for i, j in equilibria:
        eq = PAYOFFS[i][j]
        eq_total = eq[0] + eq[1]
        print(f"纳什均衡（{A_STRATEGIES[i]},{B_STRATEGIES[j]}）：{eq}，总收益 = {eq_total}")

        if coop_total > 0:
            loss = (coop_total - eq_total) / coop_total * 100
            print(f"帕累托效率损失：{loss:.1f}%")
            print(f"→ 每个人都理性选择，结果比合作差了 {loss:.0f}%")

    print()


# ============================================================================
# 6. 重复博弈模拟（以牙还牙策略）
# ============================================================================

def tit_for_tat_simulation(rounds: int = 10, seed: int = 42):
    """
    模拟重复囚徒困境，双方使用以牙还牙策略。

    以牙还牙（Tit-for-Tat）规则：
    1. 第一回合：合作（高价）
    2. 后续回合：复制对方上一回合的选择
    """
    print("=" * 60)
    print(f"重复博弈模拟（以牙还牙） - {rounds} 回合")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    # 策略索引：0=高价（合作），1=低价（背叛）
    COOP = 0  # 合作 = 高价
    DEFECT = 1  # 背叛 = 低价

    # A和B都使用以牙还牙
    a_prev = COOP  # A记得B上一轮的选择
    b_prev = COOP  # B记得A上一轮的选择

    a_choice = COOP  # A第一回合合作
    b_choice = COOP  # B第一回合合作

    history_a = []
    history_b = []
    total_a = 0
    total_b = 0

    cooperation_count = 0

    for t in range(1, rounds + 1):
        if t > 1:
            # 以牙还牙：复制对方上一回合的选择
            a_choice = b_prev
            b_choice = a_prev

        a_pay, b_pay = PAYOFFS[a_choice][b_choice]
        history_a.append(a_choice)
        history_b.append(b_choice)
        total_a += a_pay
        total_b += b_pay

        if a_choice == COOP and b_choice == COOP:
            cooperation_count += 1

        # 显示每一回合
        a_action = "合作(高价)" if a_choice == COOP else "背叛(低价)"
        b_action = "合作(高价)" if b_choice == COOP else "背叛(低价)"
        print(f"  回合 {t:2d}: A={a_action} B={b_action} → ({a_pay},{b_pay})")

        # 更新记忆
        a_prev = b_choice
        b_prev = a_choice

    print(f"\n最终累计收益：A={total_a}, B={total_b}，总收益={total_a+total_b}")
    print(f"合作回合占比：{cooperation_count}/{rounds} = {cooperation_count/rounds*100:.0f}%")

    if cooperation_count == rounds:
        print("→ 以牙还牙策略在重复博弈中维持了合作！")
    elif cooperation_count > rounds * 0.5:
        print("→ 大部分回合维持了合作")
    else:
        print("→ 合作被破坏")

    # 测试背叛触发惩罚
    print("\n--- 背叛触发验证 ---")
    print("场景：A在第2回合单方面背叛，之后恢复合作")
    a_prev = COOP  # A上次看到B的选择
    b_prev = COOP  # B上次看到A的选择
    for t in range(1, 8):
        if t == 1:
            a_choice = COOP  # 第一回合合作
            b_choice = COOP
        elif t == 2:
            a_choice = DEFECT  # A在第2回合背叛
            b_choice = a_prev  # B继续以牙还牙（合作）
        else:
            a_choice = COOP  # A从第3回合起恢复合作
            b_choice = b_prev  # B复制A上一轮的选择

        a_pay, b_pay = PAYOFFS[a_choice][b_choice]
        a_action = "合作" if a_choice == COOP else "背叛"
        b_action = "合作" if b_choice == COOP else "背叛"
        marker = " ← A背叛!" if t == 2 else ""
        print(f"  回合 {t}: A={a_action}, B={b_action} → ({a_pay},{b_pay}){marker}")

        a_prev = b_choice  # A记住B这轮的选择
        b_prev = a_choice  # B记住A这轮的选择

    print("\n→ 以牙还牙机制演示：")
    print("  (1) B在第3回合立即惩罚A的背叛 → 也选背叛")
    print("  (2) A第3回合恢复合作 → B第4回合也恢复合作")
    print("  → 以牙还牙既惩罚背叛，又宽容悔改")


# ============================================================================
# 7. 主程序
# ============================================================================

def main():
    print("=" * 60)
    print("博弈论案例1：定价博弈与囚徒困境")
    print("=" * 60)

    # 1. 展示收益矩阵
    print("\n收益矩阵：")
    print("-" * 40)
    print_matrix()

    # 2. 下划线法找均衡
    equilibria = find_nash_equilibria()
    print()

    # 3. 占优策略分析
    dominant_strategy_analysis()
    print()

    # 4. 帕累托差距
    pareto_gap_analysis(equilibria)
    print()

    # 5. 重复博弈模拟
    tit_for_tat_simulation(rounds=10)
    print()


if __name__ == "__main__":
    main()
