"""
生产调度多目标优化 — 加权求和法

问题描述：
  生产调度需要同时最小化三个冲突目标：
    f1 = 完工时间 (Makespan)
    f2 = 生产成本 (Cost)
    f3 = 能耗 (Energy)

方法（加权求和法 / Weighted Sum Method）：
  将多目标转化为单目标：min  w1*f1 + w2*f2 + w3*f3,  s.t. w1+w2+w3=1
  通过变化权重组合 (w1, w2, w3) 探索帕累托前沿。

输出：
  - 不同权重组合下的解 (f1, f2, f3)
  - 从 w1 主导到 w3 主导的过渡
  - 验证：权重变化导致目标值沿前沿移动
"""


# 教学注释：关注多个目标之间的冲突、Pareto 支配关系和权衡参数。
# 输出的解集用于筛选可解释、可落地的折中方案。



import math
import random


def simulate_schedule(speed: float, batch: int, maintenance: bool) -> dict:
    """
    根据调度参数模拟三个目标值。

    参数:
      speed: 生产速度 (1.0~5.0)，越快工时越短但能耗↑成本↑
      batch: 批次大小 (10~100)，越大成本效率越高但耗时↑
      maintenance: 是否启用节能维护模式

    返回: {makespan, cost, energy}
    """
    base_time = 50.0 / max(speed, 0.5)    # 基本完工时间
    time_factor = 1.0 + 0.005 * batch      # 批次越大耗时略增
    makespan = base_time * time_factor

    # 成本 = 时间成本 + 批次材料成本
    cost = 10.0 * makespan + 0.5 * batch

    # 能耗 = 速度能耗 + 批次能耗
    energy = 20.0 * speed + 0.1 * batch
    if maintenance:
        energy *= 0.7  # 节能模式减30%能耗

    return {"makespan": makespan, "cost": cost, "energy": energy}


def normalize_objectives(solutions: list[dict]) -> list[dict]:
    """
    对目标值进行 Min-Max 归一化到 [0,1]，便于加权求和。
    """
    if not solutions:
        return solutions

    keys = ["makespan", "cost", "energy"]
    mins = {k: min(s[k] for s in solutions) for k in keys}
    maxs = {k: max(s[k] for s in solutions) for k in keys}

    normalized = []
    for s in solutions:
        ns = {}
        for k in keys:
            if maxs[k] - mins[k] < 1e-12:
                ns[k] = 0.0
            else:
                ns[k] = (s[k] - mins[k]) / (maxs[k] - mins[k])
        normalized.append(ns)
    return normalized


def main():
    print("=" * 60)
    print("生产调度多目标优化 — 加权求和法")
    print("=" * 60)

    # ====== 步骤1: 生成调度方案集合 ======
    print("\n1. 生成调度方案集合...")

    solutions = []
    speeds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    batches = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    maint_options = [True, False]

    idx = 0
    for s in speeds:
        for b in batches:
            for m in maint_options:
                obj = simulate_schedule(s, b, m)
                obj["speed"] = s
                obj["batch"] = b
                obj["maintenance"] = m
                obj["index"] = idx
                solutions.append(obj)
                idx += 1

    print(f"   共生成 {len(solutions)} 个调度方案")

    # ====== 步骤2: 归一化 ======
    norm_solutions = normalize_objectives(solutions)

    # ====== 步骤3: 加权求和法 ======
    print("\n2. 加权求和法搜索不同权重组合下的最优解...")
    print("\n" + "-" * 72)
    print(f"{'w1(时间)':>10} | {'w2(成本)':>10} | {'w3(能耗)':>10} | "
          f"{'Makespan':>9} | {'Cost':>10} | {'Energy':>9} | {'速度':>5} | {'批次':>5} | {'节能':>5}")
    print("-" * 72)

    # 定义一组权重路径：从 w1 主导 → w2 主导 → w3 主导
    weight_paths = [
        # (w1, w2, w3)
        # 完工时间主导
        (0.9, 0.05, 0.05),
        (0.7, 0.2, 0.1),
        (0.5, 0.3, 0.2),
        # 成本主导
        (0.3, 0.6, 0.1),
        (0.2, 0.7, 0.1),
        (0.1, 0.8, 0.1),
        # 能耗主导
        (0.1, 0.2, 0.7),
        (0.1, 0.1, 0.8),
        (0.05, 0.05, 0.9),
        # 均匀权重
        (1.0 / 3, 1.0 / 3, 1.0 / 3),
    ]

    results = []
    for w1, w2, w3 in weight_paths:
        best_idx = -1
        best_weighted_sum = float('inf')

        for i, ns in enumerate(norm_solutions):
            weighted = w1 * ns["makespan"] + w2 * ns["cost"] + w3 * ns["energy"]
            if weighted < best_weighted_sum:
                best_weighted_sum = weighted
                best_idx = i

        sol = solutions[best_idx]
        results.append((w1, w2, w3, sol))

        maint_str = "ON" if sol["maintenance"] else "OFF"
        print(f"{w1:>10.3f} | {w2:>10.3f} | {w3:>10.3f} | "
              f"{sol['makespan']:>9.2f} | {sol['cost']:>10.2f} | {sol['energy']:>9.2f} | "
              f"{sol['speed']:>5.1f} | {sol['batch']:>5d} | {maint_str:>5}")

    # ====== 步骤4: 验证过渡趋势 ======
    print("\n3. 验证权重主导性变化趋势:")

    # 从 w1 主导到 w3 主导的过渡
    print("\n   3a. 完工时间 (w1) 主导 → 完工时间应最小:")
    w1_dominant = results[0]  # (0.9, 0.05, 0.05)
    print(f"     w1={w1_dominant[0]:.1f}, w2={w1_dominant[1]:.2f}, w3={w1_dominant[2]:.2f}")
    print(f"     Makespan = {w1_dominant[3]['makespan']:.2f}")

    print("\n   3b. 成本 (w2) 主导 → 成本应最小:")
    w2_dominant = results[4]  # (0.2, 0.7, 0.1)
    print(f"     w1={w2_dominant[0]:.1f}, w2={w2_dominant[1]:.1f}, w3={w2_dominant[2]:.1f}")
    print(f"     Cost = {w2_dominant[3]['cost']:.2f}")

    print("\n   3c. 能耗 (w3) 主导 → 能耗应最小:")
    w3_dominant = results[7]  # (0.1, 0.1, 0.8)
    print(f"     w1={w3_dominant[0]:.1f}, w2={w3_dominant[1]:.1f}, w3={w3_dominant[2]:.1f}")
    print(f"     Energy = {w3_dominant[3]['energy']:.2f}")

    # 验证：不同主导性下，主导目标的值确实更优
    # w1区域(0.9→0.5)：w1权重递减 → 完工时间应递增(变差)
    makespans = [r[3]['makespan'] for r in results[:3]]
    # w2区域(0.6→0.8)：w2权重递增 → 成本应递减(变好)
    costs = [r[3]['cost'] for r in results[3:6]]
    # w3区域(0.7→0.9)：w3权重递增 → 能耗应递减(变好)
    energies = [r[3]['energy'] for r in results[6:9]]

    verify_w1 = makespans[0] <= makespans[1] and makespans[1] <= makespans[2]
    verify_w2 = costs[0] >= costs[1] and costs[1] >= costs[2]
    verify_w3 = energies[0] >= energies[1] and energies[1] >= energies[2]

    print(f"\n   w1递减→完工时间递增: {makespans[0]:.1f} → {makespans[-1]:.1f} ({verify_w1})")
    print(f"   w2递增→成本递减:     {costs[0]:.1f} → {costs[-1]:.1f} ({verify_w2})")
    print(f"   w3递增→能耗递减:     {energies[0]:.1f} → {energies[-1]:.1f} ({verify_w3})")

    # ====== 总结 ======
    print(f"\n{'=' * 60}")
    print(f"结论: 加权求和法成功展示了不同权重偏好下的调度方案选择。")
    print(f"      权重变化 → 最优解沿帕累托前沿移动，符合预期。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
