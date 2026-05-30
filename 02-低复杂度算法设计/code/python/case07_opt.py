"""
案例7：实时数据流优化（滑动窗口 + 单调栈）
===========================================

教学演示文件，包含：
  - 两个问题场景：滑动窗口最大值 + 下一个更大元素
  - 暴力解 vs 优化解（滑动窗口/单调栈/deque）
  - benchmark() 输出对比表
  - 摊还分析调试辅助

兼容 Python 3.9+，无外部依赖。
"""
# 教学注释：关注复杂度、状态设计与数据结构选择如何影响可运行规模。
# 阅读时对照搜索、剪枝或启发式步骤，理解它们减少计算量的方式。


import time
import random
from collections import deque


# ============================================================
# 需求 A：滑动窗口最大值
# ============================================================

def sliding_max_brute(arr, k):
    """
    暴力解法 O(n·k)
    对每个窗口独立扫描求最大值。
    """
    n = len(arr)
    if n == 0 or k <= 0:
        return []
    return [max(arr[i:i+k]) for i in range(n - k + 1)]


def sliding_max_deque(arr, k):
    """
    单调队列解法 O(n)
    维护一个单调递减的 deque（存下标）。
    每个元素入队一次、出队一次 → 摊还 O(n)。
    """
    n = len(arr)
    if n == 0 or k <= 0:
        return []
    dq = deque()
    result = []
    for i, v in enumerate(arr):
        # 入队：弹出所有比 v 小的队尾
        while dq and arr[dq[-1]] <= v:
            dq.pop()
        dq.append(i)
        # 出队：队头超出窗口左边界
        if dq[0] == i - k:
            dq.popleft()
        # 记录：窗口形成后取队头
        if i >= k - 1:
            result.append(arr[dq[0]])
    return result


# ============================================================
# 需求 B：下一个更大元素
# ============================================================

def next_greater_brute(arr):
    """
    暴力解法 O(n²)
    对每个元素往后扫描找第一个更大的。
    """
    n = len(arr)
    return [
        next((arr[j] for j in range(i + 1, n) if arr[j] > arr[i]), -1)
        for i in range(n)
    ]


def next_greater_monotonic(arr):
    """
    单调栈解法 O(n)
    从右往左扫描，维护单调递减栈。
    每个元素入栈一次、出栈一次 → 摊还 O(n)。
    """
    n = len(arr)
    res = [-1] * n
    stack = []  # 存下标，值单调递减（从栈底到栈顶）
    for i in range(n - 1, -1, -1):
        # 弹出所有 ≤ 当前值的栈顶
        while stack and arr[stack[-1]] <= arr[i]:
            stack.pop()
        # 栈顶（如果存在）就是下一个更大元素
        if stack:
            res[i] = arr[stack[-1]]
        stack.append(i)
    return res


# ============================================================
# 正确性验证
# ============================================================

def verify_correctness():
    """在随机小规模数据上验证优化解与暴力解一致。"""
    print("=" * 65)
    print("  正确性验证")
    print("=" * 65)

    for n in [1, 2, 5, 10, 20]:
        arr = [random.randint(-100, 100) for _ in range(n)]
        k = max(1, n // 3) if n > 1 else 1

        # 滑动窗口最大值
        r1 = sliding_max_brute(arr, k)
        r2 = sliding_max_deque(arr, k)
        ok1 = "✅" if r1 == r2 else "❌"

        # 下一个更大元素
        r3 = next_greater_brute(arr)
        r4 = next_greater_monotonic(arr)
        ok2 = "✅" if r3 == r4 else "❌"

        print(f"  n={n:3d}, k={k:2d}  | 窗口最大值: {ok1}  | 下一个更大: {ok2}")

    print()


# ============================================================
# 性能对比 benchmark
# ============================================================

def benchmark_one(name, brute_fn, opt_fn, *args):
    """运行一组对比测试，返回 (暴力耗时, 优化耗时, 加速比)。"""
    # 暴力
    start = time.perf_counter()
    brute_fn(*args)
    t_brute = time.perf_counter() - start

    # 优化
    start = time.perf_counter()
    opt_fn(*args)
    t_opt = time.perf_counter() - start

    speedup = t_brute / t_opt if t_opt > 0 else float('inf')
    return t_brute, t_opt, speedup


def benchmark():
    """输出完整的性能对比表。"""
    print("=" * 65)
    print("  性能对比 (n=10000, k=1000)")
    print("=" * 65)

    # 生成测试数据
    random.seed(42)
    arr = [random.randint(0, 10000) for _ in range(10000)]
    k = 1000

    # --- 滑动窗口最大值 ---
    t1, t2, s1 = benchmark_one(
        "滑动窗口最大值",
        sliding_max_brute, sliding_max_deque,
        arr, k
    )
    print(f"\n  {'滑动窗口最大值':<20}  暴力 O(n·k)  |  {t1:.4f} 秒")
    print(f"  {'':<20}  单调队列 O(n) |  {t2:.4f} 秒")
    print(f"  {'':<20}  加速比        |  {s1:.1f}x")

    # --- 下一个更大元素 ---
    t3, t4, s2 = benchmark_one(
        "下一个更大元素",
        next_greater_brute, next_greater_monotonic,
        arr
    )
    print(f"\n  {'下一个更大元素':<20}  暴力 O(n²)   |  {t3:.4f} 秒")
    print(f"  {'':<20}  单调栈 O(n)  |  {t4:.4f} 秒")
    print(f"  {'':<20}  加速比        |  {s2:.1f}x")

    print()
    print("=" * 65)
    print("  对比汇总")
    print("=" * 65)
    print(f"  {'问题':<20} | {'复杂度':<18} | {'耗时(秒)':<12} | {'加速比':<8}")
    print(f"  {'-'*19} | {'-'*17} | {'-'*11} | {'-'*7}")
    print(f"  {'窗口最大值(暴力)':<20} | {'O(n·k)':<18} | {t1:<12.4f} | {1.0:<8.1f}")
    print(f"  {'窗口最大值(单调队列)':<20} | {'O(n)':<18} | {t2:<12.4f} | {s1:<8.1f}")
    print(f"  {'下一个更大(暴力)':<20} | {'O(n²)':<18} | {t3:<12.4f} | {1.0:<8.1f}")
    print(f"  {'下一个更大(单调栈)':<20} | {'O(n)':<18} | {t4:<12.4f} | {s2:<8.1f}")
    print()


# ============================================================
# 摊还分析演示辅助
# ============================================================

def amortized_demo_deque(arr, k):
    """
    演示单调队列的摊还性质：
    统计每个元素入队/出队的次数。
    """
    n = len(arr)
    dq = deque()
    push_count = [0] * n
    pop_count = [0] * n

    for i, v in enumerate(arr):
        while dq and arr[dq[-1]] <= v:
            popped = dq.pop()
            pop_count[popped] += 1
        dq.append(i)
        push_count[i] += 1
        if dq[0] == i - k:
            dq.popleft()
            # 被 popleft 也算一次出队（这里简化统计）

    total_push = sum(push_count)
    total_pop = sum(pop_count)
    print(f"  [摊还分析 - 单调队列] 总入队={total_push}, 总出队={total_pop}")
    print(f"  每个元素入队={total_push}/{n}={total_push/n:.1f}次")
    print(f"  每个元素出队={total_pop}/{n}={total_pop/n:.1f}次")
    assert total_push == n, f"每个元素应恰好入队 1 次, 实际={total_push}"
    return total_push, total_pop


def amortized_demo_stack(arr):
    """
    演示单调栈的摊还性质：
    统计每个元素入栈/出栈的次数。
    """
    n = len(arr)
    stack = []
    push_count = [0] * n
    pop_count = [0] * n

    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] <= arr[i]:
            popped = stack.pop()
            pop_count[popped] += 1
        stack.append(i)
        push_count[i] += 1

    total_push = sum(push_count)
    total_pop = sum(pop_count)
    print(f"  [摊还分析 - 单调栈]   总入栈={total_push}, 总出栈={total_pop}")
    print(f"  每个元素入栈={total_push}/{n}={total_push/n:.1f}次")
    print(f"  每个元素出栈={total_pop}/{n}={total_pop/n:.1f}次")
    assert total_push == n, f"每个元素应恰好入栈 1 次, 实际={total_push}"
    return total_push, total_pop


# ============================================================
# main
# ============================================================

def main():
    print()
    print("=" * 65)
    print("  案例7：实时数据流优化（滑动窗口 + 单调栈）")
    print("=" * 65)

    # 1. 正确性验证
    verify_correctness()

    # 2. 摊还分析演示
    print("=" * 65)
    print("  摊还分析演示 (n=20)")
    print("=" * 65)
    random.seed(123)
    demo_arr = [random.randint(0, 100) for _ in range(20)]
    print(f"  数据: {demo_arr}")
    amortized_demo_deque(demo_arr, k=5)
    amortized_demo_stack(demo_arr)
    print()

    # 3. 性能对比
    benchmark()

    print("=" * 65)
    print("  关键结论")
    print("=" * 65)
    print("  • 滑动窗口最大值：暴力 O(n·k) → 单调队列 O(n)")
    print("  • 下一个更大元素：暴力 O(n²)  → 单调栈 O(n)")
    print("  • 每个元素入队/入栈恰好 1 次 → 摊还 O(n)")
    print("  • 双端队列 (deque) 是滑动窗口最值的核心数据结构")
    print()


if __name__ == "__main__":
    main()
