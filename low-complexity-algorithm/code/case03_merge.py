"""
案例 3：大规模数据合并与排序（分治+归并）
==========================================

核心算法：归并排序（Merge Sort）

功能：
  - generate_data(n, near_sorted=False)  生成测试数据
  - solve_bruteforce(arr)                Python 内置排序（TimSort）
  - solve_optimized(arr)                 手写归并排序
  - solve_insertion_sort(arr)            插入排序（接近有序对比）
  - benchmark()                          生成性能对比表
  - verify()                             验证所有排序结果一致
"""

import time
import random
import sys


# ============================================================
# 1. 数据生成
# ============================================================

def generate_data(n: int, near_sorted: bool = False) -> list:
    """
    生成测试数据。

    参数:
        n: 数据规模（元素个数）
        near_sorted: 是否生成「接近有序」的数据

    返回:
        包含 n 个整数的列表

    说明:
        - near_sorted=False: 完全随机排列
        - near_sorted=True: 先有序，再随机挑 1% 的元素打乱
    """
    if near_sorted:
        # 先生成一个有序数组
        arr = list(range(1, n + 1))
        # 随机挑 1% 的位置打乱（至少 1 个）
        swap_count = max(1, n // 100)
        indices = random.sample(range(n), swap_count * 2)
        for i in range(0, len(indices), 2):
            if i + 1 < len(indices):
                a, b = indices[i], indices[i + 1]
                arr[a], arr[b] = arr[b], arr[a]
        return arr
    else:
        # 完全随机
        return random.sample(range(1, n * 10), n)


# ============================================================
# 2. 暴力解法：Python 内置排序（TimSort）
# ============================================================

def solve_bruteforce(arr: list) -> list:
    """
    暴力 baseline：Python 内置的 TimSort。

    TimSort 是归并排序 + 插入排序的混合体，
    在部分有序数据上性能极好。

    时间复杂度: O(n log n) 平均情况
    空间复杂度: O(n)
    """
    return sorted(arr)


# ============================================================
# 3. 优化解法：归并排序（分治）
# ============================================================

def merge(left: list, right: list) -> list:
    """
    合并两个有序数组。

    双指针法：
    - 同时遍历 left 和 right
    - 每次取较小的那个放入结果
    - 某一方的元素用完后，直接把另一方的剩余元素全部追加

    时间复杂度: O(len(left) + len(right))
    空间复杂度: O(len(left) + len(right))
    """
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # 剩余元素直接追加（这两个 extend 最多只有一个真正执行）
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort(arr: list) -> list:
    """
    归并排序（递归实现）。

    分治三步曲：
    1. 分（Divide）：从中间切两半
    2. 解（Conquer）：递归排序两半
    3. 合（Combine）：合并两个有序数组

    时间复杂度: O(n log n)
       - 每层合并 O(n)
       - 共 log₂n 层
    空间复杂度: O(n)
       - 每层合并需要临时数组
       - 递归调用栈 O(log n)
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # 递归排序左半
    right = merge_sort(arr[mid:])   # 递归排序右半
    return merge(left, right)       # 合并


def solve_optimized(arr: list) -> list:
    """
    手写归并排序入口。

    对输入做一次拷贝，避免修改原始数据。
    """
    return merge_sort(list(arr))


# ============================================================
# 4. 插入排序（用于接近有序数据的对比）
# ============================================================

def solve_insertion_sort(arr: list) -> list:
    """
    插入排序。

    工作原理：像整理扑克牌——新拿到的牌插入到手中已排好序的牌的正确位置。

    时间复杂度:
      - 平均/最差: O(n²) —— 完全逆序时每次都要移动所有元素
      - 最好: O(n) —— 数据已经有序，每次只需一次比较，零移动
    空间复杂度: O(1) —— 原地排序
    """
    a = list(arr)  # 拷贝，不修改原数组
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        # 把比 key 大的元素向右移动一位
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


# ============================================================
# 5. 基准测试
# ============================================================

def benchmark():
    """
    在多种数据规模下对比各排序算法的性能。

    测试规模: n = 100, 1K, 10K, 100K
    测试场景: 随机数据 + 接近有序数据
    """
    print("=" * 72)
    print("案例 3：归并排序基准测试")
    print("=" * 72)

    # ---- 场景 1：随机完全无序 ----
    print("\n[场景 1] 随机数据（完全无序）")
    print("-" * 72)
    print(f"{'n':>8} | {'内置 TimSort':>14} | {'归并排序':>14} | {'加速比':>10}")
    print("-" * 72)

    for n in [100, 1_000, 10_000, 100_000]:
        data = generate_data(n, near_sorted=False)

        # 内置排序
        t1 = time.perf_counter()
        res1 = solve_bruteforce(data)
        t2 = time.perf_counter()
        time_builtin = t2 - t1

        # 手写归并排序
        t1 = time.perf_counter()
        res2 = solve_optimized(data)
        t2 = time.perf_counter()
        time_merge = t2 - t1

        # 验证结果一致
        assert res1 == res2, f"n={n}: 结果不一致！"
        ratio = time_builtin / time_merge if time_merge > 0 else float('inf')

        print(f"{n:>8} | {time_builtin:>10.6f}s | {time_merge:>10.6f}s | {ratio:>8.2f}x")

    # ---- 场景 2：接近有序数据（打破思维定势） ----
    print("\n[场景 2] 接近有序数据（打破思维定势）")
    print("-" * 72)
    print(f"{'n':>8} | {'归并排序':>14} | {'插入排序':>14} | {'加速比':>10}")
    print("-" * 72)

    for n in [100, 1_000, 10_000, 100_000]:
        data = generate_data(n, near_sorted=True)

        # 归并排序
        t1 = time.perf_counter()
        res1 = solve_optimized(data)
        t2 = time.perf_counter()
        time_merge = t2 - t1

        # 插入排序
        t1 = time.perf_counter()
        res2 = solve_insertion_sort(data)
        t2 = time.perf_counter()
        time_insert = t2 - t1

        # 验证结果一致
        assert res1 == res2, f"n={n}: 结果不一致！"
        ratio = time_merge / time_insert if time_insert > 0 else float('inf')

        print(f"{n:>8} | {time_merge:>10.6f}s | {time_insert:>10.6f}s | {ratio:>8.2f}x")

    print("\n结论：")
    print("  1. 对完全随机数据，Python 内置 TimSort 碾压手写归并排序")
    print("     （因为 TimSort 是 C 实现，手写是纯 Python 递归）")
    print("  2. 对接近有序数据，插入排序 O(n) 碾压归并排序 O(n log n)")
    print("  3. 算法没有绝对好坏，取决于数据特征！")
    print("=" * 72)


# ============================================================
# 6. 正确性验证
# ============================================================

def verify():
    """
    验证所有排序方法在多种输入下结果一致。

    测试类型：
    - 随机数据（n=10 ~ 100）
    - 接近有序数据（n=10 ~ 100）
    - 边界数据：空数组、单元素、重复元素
    """
    print("正在验证排序正确性...")

    test_cases = [
        ([], "空数组"),
        ([1], "单元素"),
        ([5, 5, 5, 5], "全部相同"),
        ([3, 1, 2], "小规模随机"),
        ([10, 9, 8, 7, 6, 5], "完全逆序"),
        ([1, 2, 3, 4, 5], "已经有序"),
    ]

    # 随机生成更多测试用例
    random.seed(42)
    for n in [5, 10, 20, 50, 100]:
        test_cases.append((generate_data(n), f"随机 n={n}"))
        test_cases.append((generate_data(n, near_sorted=True), f"接近有序 n={n}"))

    all_pass = True
    for arr, desc in test_cases:
        res_builtin = solve_bruteforce(arr)
        res_merge = solve_optimized(arr)
        res_insert = solve_insertion_sort(arr)

        if not (res_builtin == res_merge == res_insert):
            print(f"  ❌ 失败: {desc}")
            all_pass = False
        else:
            print(f"  ✅ 通过: {desc}")

    if all_pass:
        print("🎉 所有测试用例通过！")
    else:
        print("⚠️  存在失败的测试用例！")
        sys.exit(1)

    return all_pass


# ============================================================
# 7. 主入口
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark()
    else:
        # 默认运行验证
        verify()
        print("\n提示：运行 'python case03_merge.py benchmark' 查看性能对比")
