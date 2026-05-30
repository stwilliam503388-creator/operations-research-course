"""
case05_search.py — 二分答案教学案例

本文件演示「二分答案」范式的三种经典场景：
  1. 木材切割（最大化每段长度）
  2. 书籍抄写（最小化完成时间）
  3. 分巧克力（最大化正方形边长）

所有场景共享同一个二分模板，仅仅判定函数不同。
"""


# 教学注释：关注复杂度、状态设计与数据结构选择如何影响可运行规模。
# 阅读时对照搜索、剪枝或启发式步骤，理解它们减少计算量的方式。



import time
import random


# ============================================================
# 场景 A：木材切割
# ============================================================

def can_cut(length: int, woods: list[int], k: int) -> bool:
    """
    判定函数：每段长度为 length 时，能否切出至少 k 段？

    对每根钢管，能切的段数 = 钢管长度 // length。
    把总段数和 k 比较即可。

    参数:
        length: 当前猜测的每段长度
        woods:  每根钢管的长度列表
        k:      目标段数
    返回:
        True 如果总段数 >= k，否则 False
    """
    total = 0
    for w in woods:
        total += w // length
        # 提前终止：已经达到目标就可以停
        if total >= k:
            return True
    return total >= k


def solve_wood(woods: list[int], k: int) -> int:
    """
    木材切割 — 二分答案找最大可行长度。

    答案范围: 1 ~ max(woods)
    单调性: length 越小 → 段数越多 → 越可行
            length 增大 → 段数减少 → 最终不可行
    我们要找的是「最大可行值」，用右边界收缩模板。

    时间复杂度: O(N log max(woods))
    """
    lo, hi = 1, max(woods)

    while lo < hi:
        # 注意 +1：避免 lo=mid 时陷入死循环
        mid = (lo + hi + 1) // 2
        if can_cut(mid, woods, k):
            lo = mid       # mid 可行，试试更大的
        else:
            hi = mid - 1   # mid 不可行，试试更小的

    return lo


# ============================================================
# 场景 B：书籍抄写（二分答案 + 贪心判定）
# ============================================================

def can_copy(limit: int, pages: list[int], m: int) -> bool:
    """
    判定函数：每个抄写员最多抄 limit 页，能否用 ≤ m 人完成？

    贪心分配：
      从第一本书开始，依次分配给当前抄写员。
      如果当前抄写员加上这本书会超限，就换下一个人。

    参数:
        limit: 每个抄写员的时间上限（页数）
        pages: 每本书的页数
        m:     抄写员人数
    返回:
        True 如果能在 m 人以内完成，否则 False
    """
    # 如果任何一本书超过 limit，直接不可能
    if max(pages) > limit:
        return False

    workers = 1       # 当前已使用的抄写员数量
    current = 0       # 当前抄写员已分配的总页数

    for p in pages:
        if current + p > limit:
            workers += 1       # 换下一个人
            current = p        # 新抄写员从这本书开始
            if workers > m:
                return False   # 人不够了
        else:
            current += p

    return True


def solve_book(pages: list[int], m: int) -> int:
    """
    书籍抄写 — 二分答案找最小可行时间。

    答案范围: max(pages) ~ sum(pages)
    单调性: limit 越大 → 需要的抄写员越少 → 越可行
    我们要找「最小可行值」，用左边界收缩模板。

    时间复杂度: O(M log sum(pages))
    """
    lo, hi = max(pages), sum(pages)

    while lo < hi:
        mid = (lo + hi) // 2
        if can_copy(mid, pages, m):
            hi = mid       # 可行就试试更小的 limit
        else:
            lo = mid + 1   # 不可行就加大 limit

    return lo


# ============================================================
# 场景 C：分巧克力（二分答案）
# ============================================================

def can_chop(side: int, chocolates: list[tuple[int, int]], k: int) -> bool:
    """
    判定函数：边长为 side 时，能切出至少 k 块正方形巧克力？

    每块巧克力 (H, W) 能切出的块数 = (H // side) * (W // side)。

    参数:
        side:       正方形边长（当前猜测值）
        chocolates: 每块巧克力的 (高, 宽) 列表
        k:          目标块数
    返回:
        True 如果总块数 >= k，否则 False
    """
    total = 0
    for h, w in chocolates:
        total += (h // side) * (w // side)
        if total >= k:
            return True
    return total >= k


def solve_choco(chocolates: list[tuple[int, int]], k: int) -> int:
    """
    分巧克力 — 二分答案找最大可行边长。

    答案范围: 1 ~ max(所有边长)
    单调性: side 越小 → 块数越多 → 越可行
    找「最大可行值」，用右边界收缩模板。

    时间复杂度: O(N log max(H, W))
    """
    max_side = max(max(h, w) for h, w in chocolates)
    lo, hi = 1, max_side

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if can_chop(mid, chocolates, k):
            lo = mid
        else:
            hi = mid - 1

    return lo


# ============================================================
# 线性尝试法（用于和二分答案做性能对比）
# ============================================================

def linear_wood(woods: list[int], k: int) -> int:
    """
    线性尝试法：从最大可能长度往下一个个试。
    纯暴力对照，用于 benchmark 对比。
    """
    for length in range(max(woods), 0, -1):
        if can_cut(length, woods, k):
            return length
    return 0


# ============================================================
# Benchmark：二分答案 vs 线性尝试
# ============================================================

def benchmark():
    """比较二分答案和线性尝试在木材切割上的性能差异。"""
    print("=" * 60)
    print("二分答案 Benchmark")
    print("=" * 60)

    # ── 小规模测试（验证正确性） ──
    woods_small = [103, 217, 158, 94, 312]
    k_small = 50
    print(f"\n木材切割（{len(woods_small)}根钢管，目标{k_small}段）：")

    start = time.perf_counter()
    lin_result = linear_wood(woods_small, k_small)
    lin_time = time.perf_counter() - start

    start = time.perf_counter()
    bin_result = solve_wood(woods_small, k_small)
    bin_time = time.perf_counter() - start

    print(f"  线性尝试:  最大长度 = {lin_result}, 耗时 {lin_time:.6f} 秒")
    print(f"  二分答案:  最大长度 = {bin_result}, 耗时 {bin_time:.6f} 秒")
    assert lin_result == bin_result, "结果不一致！"
    print(f"  ✅ 结果一致，速度 ×{lin_time/bin_time:.0f} 倍")

    # ── 书籍抄写 ──
    pages = [random.randint(50, 200) for _ in range(10)]
    scribes = 3
    print(f"\n书籍抄写（{len(pages)}本书，{scribes}个抄写员）：")
    result = solve_book(pages, scribes)
    print(f"  二 分 答 案: 最短耗时 = {result} 页")

    # ── 分巧克力 ──
    chocolates = [(random.randint(5, 15), random.randint(5, 15)) for _ in range(10)]
    target = 6
    print(f"\n分巧克力（{len(chocolates)}块，目标{target}块）：")
    result = solve_choco(chocolates, target)
    print(f"  二 分 答 案: 最大边长 = {result}")

    # ── 大规模测试 ──
    print(f"\n── 大规模测试 ──")
    woods_large = [random.randint(100, 500) for _ in range(500)]
    k_large = 100000
    print(f"木材 {len(woods_large)} 根, 目标 {k_large} 段:")

    start = time.perf_counter()
    bin_result = solve_wood(woods_large, k_large)
    bin_time = time.perf_counter() - start
    print(f"  二分答案: 最大长度 = {bin_result}, 耗时 {bin_time:.6f} 秒")

    print("\n✅ 所有测试完成。")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    benchmark()
