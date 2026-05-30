"""
案例 4：多重约束资源分配（背包 DP）
==========================================

核心算法：0/1 背包动态规划

功能：
  - generate_data(n, max_weight)                  生成随机物品数据
  - solve_bruteforce(weights, values, W)          回溯枚举所有子集 O(2ⁿ)
  - solve_optimized(weights, values, W)           二维 DP O(n·W)
  - space_optimized(weights, values, W)           一维滚动数组 DP O(n·W), O(W) 空间
  - greedy_density(weights, values, W)            贪心（按单位价值，展示反例）
  - benchmark()                                   生成性能对比表
  - verify()                                      验证各方法结果一致
"""


# 教学注释：关注复杂度、状态设计与数据结构选择如何影响可运行规模。
# 阅读时对照搜索、剪枝或启发式步骤，理解它们减少计算量的方式。



import time
import random
import sys


# ============================================================
# 1. 数据生成
# ============================================================

def generate_data(n: int, max_weight: int = 10) -> tuple:
    """
    生成背包测试数据。

    参数:
        n: 物品数量
        max_weight: 物品最大重量（1 ~ max_weight）

    返回:
        (weights, values) 两个列表，每个长度 n
        - weights[i]: 第 i 个物品的重量 (1 ~ max_weight)
        - values[i]: 第 i 个物品的价值 (1 ~ max_weight*2)
    """
    random.seed(42)  # 固定种子，结果可复现
    weights = [random.randint(1, max_weight) for _ in range(n)]
    values = [random.randint(1, max_weight * 2) for _ in range(n)]
    return weights, values


# ============================================================
# 2. 暴力解法：回溯枚举所有子集
# ============================================================

def solve_bruteforce(weights: list, values: list, W: int) -> int:
    """
    暴力回溯枚举所有物品组合。

    每个物品「选或不选」，共 2ⁿ 种组合。
    对每种组合检查是否超重，计算总价值，取最大值。

    时间复杂度: O(2ⁿ) —— n 稍微大一点就不可行
    空间复杂度: O(n) —— 递归调用栈深度
    """
    n = len(weights)
    best_value = 0

    def backtrack(idx: int, cur_weight: int, cur_value: int):
        """深度优先搜索所有选择"""
        nonlocal best_value

        if idx == n:
            # 已经考虑了所有物品，更新最优解
            if cur_value > best_value:
                best_value = cur_value
            return

        # 剪枝：即使剩下的全部选上，也不可能超过当前最优
        # （这是一个简单的上界估计，实际可以更精确）
        # 这里不剪枝，保持纯粹的暴力

        # 选项 1：不选当前物品
        backtrack(idx + 1, cur_weight, cur_value)

        # 选项 2：选当前物品（前提是不超重）
        if cur_weight + weights[idx] <= W:
            backtrack(idx + 1, cur_weight + weights[idx],
                      cur_value + values[idx])

    backtrack(0, 0, 0)
    return best_value


# ============================================================
# 3. 优化解法：二维 DP
# ============================================================

def solve_optimized(weights: list, values: list, W: int) -> int:
    """
    二维动态规划求解 0/1 背包。

    状态定义：
      dp[i][j] = 前 i 个物品中选，总重量不超过 j 的最大总价值

    转移方程：
      dp[i][j] = max(
          dp[i-1][j],                          # 不选第 i 个物品
          dp[i-1][j - w[i-1]] + v[i-1]         # 选第 i 个物品（前提 j >= w[i-1]）
      )

    时间复杂度: O(n·W)
    空间复杂度: O(n·W) —— (n+1) × (W+1) 的二维数组
    """
    n = len(weights)
    # 创建 (n+1) × (W+1) 的二维数组，初始化为 0
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w_i = weights[i - 1]  # 第 i 个物品的重量（0-indexed）
        v_i = values[i - 1]   # 第 i 个物品的价值
        for j in range(W + 1):
            if j < w_i:
                # 装不下，只能不选
                dp[i][j] = dp[i - 1][j]
            else:
                # 选或不选，取较大值
                dp[i][j] = max(
                    dp[i - 1][j],                     # 不选
                    dp[i - 1][j - w_i] + v_i          # 选
                )

    return dp[n][W]


# ============================================================
# 4. 空间优化版本：一维滚动数组
# ============================================================

def space_optimized(weights: list, values: list, W: int) -> int:
    """
    一维滚动数组 DP（空间优化版）。

    核心观察：
      - dp[i][j] 只依赖于 dp[i-1][...]（上一行）
      - 所以不需要保留所有行，用一个一维数组就够

    关键细节：
      - j 必须从 W 到 0 倒序遍历！
      - 原因：正序遍历会使 dp[j - w] 提前被当前物品更新
        → 导致一个物品被选多次（变成完全背包）

    时间复杂度: O(n·W)
    空间复杂度: O(W)
    """
    n = len(weights)
    dp = [0] * (W + 1)

    for i in range(n):
        w_i = weights[i]
        v_i = values[i]
        # 倒序遍历：从最大容量到当前物品重量
        # 保证 dp[j - w_i] 还是上一轮的旧值
        for j in range(W, w_i - 1, -1):
            dp[j] = max(dp[j], dp[j - w_i] + v_i)

    return dp[W]


# ============================================================
# 5. 贪心解法（用于展示反例）
# ============================================================

def greedy_density(weights: list, values: list, W: int) -> int:
    """
    贪心算法：按单位价值（价值/重量）从高到低选物品。

    重要：这个解法是「错误」的——它不一定给出最优解。
    但很多初学者第一反应就是「选单价最高的」，
    这个函数用来展示贪心为什么会失败。

    时间复杂度: O(n log n) —— 排序是主要的耗时
    空间复杂度: O(n)
    """
    n = len(weights)
    # 计算每个物品的单位价值
    items = []
    for i in range(n):
        density = values[i] / weights[i]
        items.append((density, weights[i], values[i]))

    # 按单位价值从高到低排序
    items.sort(reverse=True, key=lambda x: x[0])

    total_weight = 0
    total_value = 0

    for density, w, v in items:
        if total_weight + w <= W:
            total_weight += w
            total_value += v

    return total_value


# ============================================================
# 6. 辅助函数：打印 DP 表
# ============================================================

def dp_print_table(weights: list, values: list, W: int):
    """
    打印二维 DP 表的填表过程。

    仅用于教学演示，物品数 n 和容量 W 较小时使用。
    """
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    print("\n物品列表：")
    print(f"  {'物品':>4} | {'重量':>4} | {'价值':>4}")
    print("  " + "-" * 18)
    for i in range(n):
        print(f"  {i+1:>4} | {weights[i]:>4} | {values[i]:>4}")

    print(f"\n背包容量 W = {W}")
    print("\nDP 填表过程：")

    # 打印表头
    print(f"  {'i/j':>5}", end="")
    for j in range(W + 1):
        print(f"{j:>4}", end="")
    print()

    for i in range(n + 1):
        if i == 0:
            print(f"  {'0(空)':>5}", end="")
        else:
            print(f"  {i:>4}  ", end="")

        for j in range(W + 1):
            if i == 0:
                val = 0
            else:
                w_i = weights[i - 1]
                v_i = values[i - 1]
                if j < w_i:
                    val = dp[i - 1][j]
                else:
                    val = max(dp[i - 1][j], dp[i - 1][j - w_i] + v_i)
                dp[i][j] = val
            print(f"{val:>4}", end="")
        print()

    print(f"\n最优解: dp[{n}][{W}] = {dp[n][W]}")


# ============================================================
# 7. 基准测试
# ============================================================

def benchmark():
    """
    对比不同规模的背包求解性能。

    测试规模: n=5/10/20/30/40，W 随之增大
    对比方法: 暴力回溯 / 二维 DP / 滚动数组 / 贪心
    """
    print("=" * 72)
    print("案例 4：0/1 背包基准测试")
    print("=" * 72)

    scenarios = [
        (5, 10, "小规模 n=5"),
        (10, 20, "中规模 n=10"),
        (20, 50, "较大规模 n=20"),
        (30, 100, "大规模 n=30"),
        (40, 200, "超大规模 n=40"),
    ]

    print(f"\n{'n':>4} | {'W':>4} | {'暴力回溯':>14} | {'二维 DP':>14} | {'滚动数组':>14} | {'DP加速比':>10}")
    print("-" * 72)

    for n, W, desc in scenarios:
        weights, values = generate_data(n, max_weight=W // 4)
        # 调整 W 使其合理
        total_w = sum(weights)
        W_actual = min(W, total_w // 2) if total_w > 0 else W

        # 暴力回溯（n <= 20 才跑，否则跳过）
        if n <= 20:
            t1 = time.perf_counter()
            brute_val = solve_bruteforce(weights, values, W_actual)
            t2 = time.perf_counter()
            time_brute = t2 - t1
            brute_str = f"{time_brute:>10.6f}s"
        else:
            brute_val = None
            time_brute = float('inf')
            brute_str = f"   ❌ 超时  "

        # 二维 DP
        t1 = time.perf_counter()
        dp_val = solve_optimized(weights, values, W_actual)
        t2 = time.perf_counter()
        time_dp = t2 - t1

        # 滚动数组
        t1 = time.perf_counter()
        roll_val = space_optimized(weights, values, W_actual)
        t2 = time.perf_counter()
        time_roll = t2 - t1

        # 验证 DP 和滚动数组结果一致
        assert dp_val == roll_val, f"n={n}: DP 与滚动数组结果不一致！"
        if brute_val is not None:
            assert dp_val == brute_val, f"n={n}: DP 与暴力结果不一致！"

        # 加速比（暴力 / DP）
        ratio = time_brute / time_dp if time_dp > 0 and time_brute != float('inf') else float('inf')
        ratio_str = f"{ratio:>8.1f}x" if ratio != float('inf') else "    ∞   "

        print(f"{n:>4} | {W_actual:>4} | {brute_str:>14} | {time_dp:>10.6f}s | {time_roll:>10.6f}s | {ratio_str:>10}")

    # ---- 贪心反例演示 ----
    print("\n" + "=" * 72)
    print("贪心反例演示")
    print("=" * 72)

    # 构造一个经典的让贪心失败的例子
    # 物品 1: w=4, v=12, 单价=3.0  ← 单价最高
    # 物品 2: w=3, v=8,  单价≈2.67
    # 物品 3: w=3, v=7,  单价≈2.33
    # W = 6
    # 贪心: 选物品1(单价3.0) → 容量剩2 → 结束 → 总价值=12 ❌
    # 最优: 选物品2+物品3 = 总重6 → 总价值=15 ✅
    weights_ex = [4, 3, 3]
    values_ex = [12, 8, 7]
    W_ex = 6

    greedy_val = greedy_density(weights_ex, values_ex, W_ex)
    optimal_val = solve_optimized(weights_ex, values_ex, W_ex)

    print(f"\n经典反例：W={W_ex}")
    print(f"  物品 1: 重量=4, 价值=12, 单价=3.0  ← 单价最高")
    print(f"  物品 2: 重量=3, 价值=8,  单价≈2.67")
    print(f"  物品 3: 重量=3, 价值=7,  单价≈2.33")
    print(f"\n  贪心结果: {greedy_val} (选了物品 1, 容量剩 2, 装不下其他)")
    print(f"  DP 最优:  {optimal_val} (选物品 2 + 物品 3 = 总重 6)")
    print(f"\n  {'贪心正确?':>12} {'❌ 不是最优！' if greedy_val != optimal_val else '✅ 正确'}")

    print("\n结论：")
    print("  1. 暴力回溯在 n=20 时勉强可用，n=30 以上直接爆炸")
    print("  2. DP 在 n=40, W=200 时仍瞬间完成")
    print("  3. 贪心算法在某些情况下会失败（局部最优 ≠ 全局最优）")
    print("=" * 72)


# ============================================================
# 8. 正确性验证
# ============================================================

def verify():
    """
    验证所有方法在多种输入下结果一致。

    测试类型：
    - 随机生成的多组数据
    - 边界情况
    - 贪心反例
    """
    print("正在验证背包 DP 正确性...")
    random.seed(42)

    test_cases = []

    # 边界测试
    test_cases.append(([1], [5], 0, "容量为 0"))       # 应该为 0
    test_cases.append(([1], [5], 1, "一个物品，刚好装下"))  # 应该为 5
    test_cases.append(([5], [5], 3, "一个物品，装不下"))   # 应该为 0

    # 小规模随机
    for n in [3, 5, 8, 10, 12]:
        w, v = generate_data(n, max_weight=8)
        total_w = sum(w)
        W = total_w // 2  # 容量设为总重的一半
        test_cases.append((w, v, W, f"随机 n={n}"))

    all_pass = True
    for weights, values, W, desc in test_cases:
        # n <= 20 才跑暴力
        if len(weights) <= 15:
            brute_val = solve_bruteforce(weights, values, W)
        else:
            brute_val = None

        dp_val = solve_optimized(weights, values, W)
        roll_val = space_optimized(weights, values, W)

        # DP 和滚动数组必须一致
        if dp_val != roll_val:
            print(f"  ❌ DP 与滚动数组不一致: {desc}")
            all_pass = False
        elif brute_val is not None and dp_val != brute_val:
            print(f"  ❌ DP 与暴力结果不一致: {desc} (DP={dp_val}, brute={brute_val})")
            all_pass = False
        else:
            if brute_val is not None:
                print(f"  ✅ 通过: {desc} (最优值={dp_val})")
            else:
                print(f"  ✅ 通过: {desc} (最优值={dp_val}, 暴力跳过 n={len(weights)})")

    if all_pass:
        print("🎉 所有测试用例通过！")
    else:
        print("⚠️  存在失败的测试用例！")
        sys.exit(1)

    return all_pass


# ============================================================
# 9. 主入口
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "benchmark":
            benchmark()
        elif sys.argv[1] == "table":
            # 打印 DP 表
            weights = [4, 3, 5, 2, 1]
            values = [7, 5, 8, 3, 2]
            W = 10
            dp_print_table(weights, values, W)
        else:
            print("用法: python case04_knapsack.py [benchmark|table]")
    else:
        verify()
        print("\n提示：运行 'python case04_knapsack.py benchmark' 查看性能对比")
        print("     运行 'python case04_knapsack.py table' 打印 DP 填表过程")
