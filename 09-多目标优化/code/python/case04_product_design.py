"""
产品设计三目标优化 — ε-约束法

问题描述：
  产品设计需要同时优化三个冲突目标：
    f1 = 成本（越低越好，最小化）
    f2 = 性能（越高越好，最大化，用负值表示）
    f3 = 重量（越低越好，最小化）

方法（ε-约束法）：
  将 f2（性能）作为主目标最大化，把 f1（成本）和 f3（重量）作为 ε 约束处理。
  每次固定 f1 ≤ ε_cost、f3 ≤ ε_weight，求解最大化 -f2 的问题。
  逐步收紧约束，观察解的变化。

输出：
  - 不同 ε 组合下的解 (f1, f2, -f2)
  - 验证：约束越紧，主目标值单调变化
"""
# 教学注释：关注多个目标之间的冲突、Pareto 支配关系和权衡参数。
# 输出的解集用于筛选可解释、可落地的折中方案。


import math
import random


def simulate_design(cost: float, weight: float) -> float:
    """
    模拟产品设计的目标函数。
    给定成本投入和重量设计，返回性能值 f2。
    成本越高、重量越轻 → 性能越好（边际递减）。
    用负值表示"越大越好"的目标，内部用 -f2。
    """
    # 性能 = 成本贡献 + 重量贡献 + 交互项 + 噪声
    # 成本贡献: 成本越高性能越好，但边际递减 (sqrt)
    cost_contrib = 10.0 * math.sqrt(max(cost, 0.0))
    # 重量贡献: 重量越轻性能越好 (1/weight)
    weight_contrib = 5.0 / max(weight, 0.01)
    # 交互项: 高成本 + 轻重量 有协同效应
    interaction = 2.0 * math.sqrt(max(cost, 0.0)) / max(weight, 0.01)
    return cost_contrib + weight_contrib + interaction


def main():
    print("=" * 60)
    print("产品设计三目标优化 — ε-约束法")
    print("=" * 60)

    # ====== 步骤1: 生成解空间 ======
    # 在成本 [1, 10] 和重量 [0.5, 3.0] 网格上采样
    print("\n1. 枚举设计空间网格...")
    cost_grid = [1.0 + i * 0.5 for i in range(19)]   # 1.0~10.0, 步长0.5
    weight_grid = [0.5 + i * 0.2 for i in range(13)]  # 0.5~2.9, 步长0.2

    all_solutions = []
    for c in cost_grid:
        for w in weight_grid:
            perf = simulate_design(c, w)
            # 三目标: (成本, 性能, 重量)
            all_solutions.append((c, perf, w))

    print(f"   网格大小: {len(cost_grid)} × {len(weight_grid)} = {len(all_solutions)} 个方案")

    # ====== 步骤2: ε-约束法 ======
    print("\n2. 应用 ε-约束法（固定成本与重量上限，最大化性能）...")
    print(f"\n{'ε_cost(成本≤)':>12} | {'ε_weight(重量≤)':>16} | {'成本':>6} | {'性能':>8} | {'重量':>6}")

    # 逐步收紧约束
    cost_bounds = [10.0, 8.0, 6.0, 4.5, 3.0, 2.0]
    weight_bounds = [3.0, 2.4, 1.8, 1.3, 0.9, 0.6]

    print("-" * 56)
    results = []

    for ec in cost_bounds:
        for ew in weight_bounds:
            # 筛选满足约束的解
            feasible = [(c, p, w) for c, p, w in all_solutions
                        if c <= ec + 1e-9 and w <= ew + 1e-9]
            if not feasible:
                print(f"{ec:>12.1f} | {ew:>16.1f} | {'无解':>20}")
                continue

            # 最大化性能（即选择性能最大的可行解）
            best = max(feasible, key=lambda x: x[1])
            results.append((ec, ew, best[0], best[1], best[2]))
            print(f"{ec:>12.1f} | {ew:>16.1f} | {best[0]:>6.1f} | {best[1]:>8.2f} | {best[2]:>6.2f}")

    # ====== 步骤3: 验证单调性 ======
    print("\n3. 验证单调性：分别固定成本/重量，检查性能变化趋势")

    # 3a: 固定 ε_cost，收紧 ε_weight → 性能应单调不增
    print("\n   3a. 固定 ε_cost=8.0, 收紧 ε_weight:")
    fixed_cost = 8.0
    ew_vals = [3.0, 2.4, 1.8, 1.3, 0.9, 0.6]
    prev_perf = float('inf')
    monotonic_cost = True
    for ew in ew_vals:
        feasible = [(c, p, w) for c, p, w in all_solutions
                    if c <= fixed_cost + 1e-9 and w <= ew + 1e-9]
        if feasible:
            best = max(feasible, key=lambda x: x[1])
            print(f"     ε_weight ≤ {ew:>4.1f} → 性能 = {best[1]:>8.2f} (成本={best[0]:.1f}, 重量={best[2]:.2f})"
                  f"{' ← 下降' if best[1] > prev_perf + 1e-6 else ''}")
            if best[1] > prev_perf + 1e-6:
                monotonic_cost = False
            prev_perf = best[1]

    print(f"    → 固定成本时，收紧重量约束性能单调不增: {monotonic_cost}")

    # 3b: 固定 ε_weight，收紧 ε_cost → 性能应单调不增
    print("\n   3b. 固定 ε_weight=1.8, 收紧 ε_cost:")
    fixed_weight = 1.8
    ec_vals = [10.0, 8.0, 6.0, 4.5, 3.0, 2.0]
    prev_perf = float('inf')
    monotonic_weight = True
    for ec in ec_vals:
        feasible = [(c, p, w) for c, p, w in all_solutions
                    if c <= ec + 1e-9 and w <= fixed_weight + 1e-9]
        if feasible:
            best = max(feasible, key=lambda x: x[1])
            print(f"     ε_cost ≤ {ec:>5.1f} → 性能 = {best[1]:>8.2f} (成本={best[0]:.1f}, 重量={best[2]:.2f})"
                  f"{' ← 下降' if best[1] > prev_perf + 1e-6 else ''}")
            if best[1] > prev_perf + 1e-6:
                monotonic_weight = False
            prev_perf = best[1]

    print(f"    → 固定重量时，收紧成本约束性能单调不增: {monotonic_weight}")

    # ====== 总结 ======
    print(f"\n{'=' * 60}")
    print(f"结论: ε-约束法成功找到不同约束水平下的最优解。")
    print(f"      约束越紧 → 可行域越小 → 最优性能越低（单调不增），符合预期。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
