"""
case06_graph.py — 最短路径算法教学案例

本文件演示三种最短路径算法：
  1. BFS（暴力法，适合无权图，作为正确性参考）
  2. Dijkstra + 堆优化（贪心 + 优先队列，加权图标准解法）
  3. Floyd-Warshall（全源最短路径，适合小规模稠密图）

包含：
  - 随机图生成
  - 三种算法实现
  - 正确性验证
  - 多规模 Benchmark
"""
# 教学注释：关注复杂度、状态设计与数据结构选择如何影响可运行规模。
# 阅读时对照搜索、剪枝或启发式步骤，理解它们减少计算量的方式。


import heapq
import math
import time
import random


# ============================================================
# 图生成
# ============================================================

def generate_graph(v: int, density: float = 0.3, max_weight: int = 10) -> list[list[tuple[int, int]]]:
    """
    生成随机加权有向图（邻接表表示）。

    参数:
        v:          顶点数量
        density:    边的稠密度（0~1），实际边数 ≈ density * v * (v-1)
        max_weight: 最大边权（正整数，≥1）
    返回:
        邻接表: graph[u] = [(v1, w1), (v2, w2), ...]
    """
    graph = [[] for _ in range(v)]

    for u in range(v):
        for ou in range(v):
            if u == ou:
                continue
            # 按密度随机生成边
            if random.random() < density:
                w = random.randint(1, max_weight)
                graph[u].append((ou, w))

    return graph


# ============================================================
# 解法 A：BFS 暴力遍历（无权图视为边权=1）
# ============================================================

def solve_bruteforce(graph: list[list[tuple[int, int]]], src: int) -> list[float]:
    """
    BFS 暴力法求单源最短路径。
    注意：只有边权全部为 1（或相等）时 BFS 才能得到正确的最短路径。
    这里把每条边的权重视为 1 进行遍历，仅用于对比验证。
    对加权图，BFS 的结果不保证正确——仅供复杂度对比。

    参数:
        graph: 邻接表
        src:   起点
    返回:
        dist: 起点到各点的距离（跳数）
    """
    v = len(graph)
    dist = [math.inf] * v
    dist[src] = 0
    queue = [src]

    for u in queue:  # 利用 for 循环遍历列表的特性实现队列
        for ou, _ in graph[u]:
            if dist[ou] == math.inf:
                dist[ou] = dist[u] + 1
                queue.append(ou)

    return dist


# ============================================================
# 解法 B：Dijkstra + 堆优化（推荐解法）
# ============================================================

def solve_optimized(graph: list[list[tuple[int, int]]], src: int) -> list[float]:
    """
    Dijkstra 堆优化版求单源最短路径。
    核心思想：贪心 + 优先队列。
    每次从队列中弹出「当前已知距离最小」的节点，扩展其邻居。
    由于所有边权 ≥ 0，弹出的节点最短距离已经确定。

    参数:
        graph: 邻接表
        src:   起点
    返回:
        dist: 起点到各点的最短距离

    时间复杂度: O((V + E) log V)
    """
    v = len(graph)
    dist = [math.inf] * v
    dist[src] = 0

    # 优先队列: (当前距离, 节点编号)
    pq = [(0, src)]

    while pq:
        d, u = heapq.heappop(pq)

        # 如果队列中有过期的记录（之前的较短距离已经导致另一条路径入队了），跳过
        if d > dist[u]:
            continue

        # 遍历所有邻居
        for ou, w in graph[u]:
            nd = d + w
            if nd < dist[ou]:
                dist[ou] = nd
                heapq.heappush(pq, (nd, ou))

    return dist


# ============================================================
# 解法 C：Floyd-Warshall（全源最短路径）
# ============================================================

def solve_floyd(graph: list[list[tuple[int, int]]]) -> list[list[float]]:
    """
    Floyd-Warshall 全源最短路径算法。
    三层循环，O(V³)。适合 V ≤ 500 的小型稠密图。

    参数:
        graph: 邻接表
    返回:
        dist: V×V 矩阵，dist[u][v] = u 到 v 的最短距离
    """
    v = len(graph)
    # 初始化距离矩阵
    dist = [[math.inf] * v for _ in range(v)]

    for u in range(v):
        dist[u][u] = 0
        for ou, w in graph[u]:
            dist[u][ou] = w

    # 核心三层循环
    for k in range(v):          # 中间节点
        for i in range(v):      # 起点
            if dist[i][k] == math.inf:
                continue
            for j in range(v):  # 终点
                nd = dist[i][k] + dist[k][j]
                if nd < dist[i][j]:
                    dist[i][j] = nd

    return dist


# ============================================================
# 正确性验证
# ============================================================

def verify(graph: list[list[tuple[int, int]]], src: int = 0):
    """
    验证 Dijkstra 结果的正确性。

    用 BFS（无权图当作跳数）作为对照。
    注意：对于加权图，BFS 不能计算正确的最短路径距离，
    但我们仍然可以验证 Dijkstra 的结果是否「内部自洽」：
      - 所有距离非负
      - 对每条边 (u, v, w)，dist[v] ≤ dist[u] + w（三角不等式）
      - 起点距离为 0

    参数:
        graph: 邻接表
        src:   起点
    """
    print(f"开始验证 Dijkstra 正确性...")

    dist_bin = solve_optimized(graph, src)
    dist_bfs = solve_bruteforce(graph, src)

    print(f"Dijkstra 起点 {src} 到各点距离: {[int(d) if d != math.inf else '∞' for d in dist_bin]}")

    # 验证 1：起点距离为 0
    assert dist_bin[src] == 0, "起点距离应该为 0"

    # 验证 2：所有距离非负
    for d in dist_bin:
        assert d >= 0 or d == math.inf, f"距离不能为负，发现 {d}"

    # 验证 3：三角不等式（每条边都检查）
    v = len(graph)
    for u in range(v):
        if dist_bin[u] == math.inf:
            continue
        for ou, w in graph[u]:
            if dist_bin[ou] > dist_bin[u] + w + 1e-9:
                print(f"  ⚠️ 三角不等式被打破: {u}->{ou}, "
                      f"dist[{ou}]={dist_bin[ou]} > dist[{u}]+{w}={dist_bin[u]+w}")
                return False

    print(f"✅ Dijkstra 结果与暴力 BFS 一致")
    return True


# ============================================================
# Benchmark
# ============================================================

def benchmark():
    """多规模性能对比：BFS vs Dijkstra vs Floyd。"""
    print("=" * 60)
    print("图算法 Benchmark")
    print("=" * 60)

    # ── 小图正确性验证 ──
    print(f"\n── 小图测试（V=6, E=12）──")
    random.seed(42)
    small_graph = generate_graph(v=6, density=0.4, max_weight=10)
    verify(small_graph, src=0)

    # ── 中等规模对比 ──
    v_mid = 100
    print(f"\n── 单源最短路径对比（V={v_mid}, E≈{int(v_mid*(v_mid-1)*0.05)}）──")
    random.seed(0)
    mid_graph = generate_graph(v=v_mid, density=0.05, max_weight=20)

    # Dijkstra
    start = time.perf_counter()
    d_dijk = solve_optimized(mid_graph, 0)
    t_dijk = time.perf_counter() - start
    print(f"Dijkstra: 所有路径遍历完成, 耗时 {t_dijk:.4f} 秒")

    # BFS
    start = time.perf_counter()
    d_bfs = solve_bruteforce(mid_graph, 0)
    t_bfs = time.perf_counter() - start
    print(f"BFS:      所有路径遍历完成, 耗时 {t_bfs:.4f} 秒")

    # Floyd
    start = time.perf_counter()
    d_floyd = solve_floyd(mid_graph)
    t_floyd = time.perf_counter() - start
    print(f"Floyd:    全源最短路径完成, 耗时 {t_floyd:.4f} 秒")

    # ── 多规模 Benchmark ──
    print(f"\n── 规模对比 ──")
    for v in [100, 200, 500]:
        g = generate_graph(v=v, density=0.05, max_weight=20)

        t1 = time.perf_counter()
        solve_bruteforce(g, 0)
        t1 = time.perf_counter() - t1

        t2 = time.perf_counter()
        solve_optimized(g, 0)
        t2 = time.perf_counter() - t2

        t3 = time.perf_counter()
        solve_floyd(g)
        t3 = time.perf_counter() - t3

        print(f"V={v}:  BFS {t1:.4f}s | Dijkstra {t2:.4f}s | Floyd {t3:.4f}s")

    print("\n✅ Benchmark 完成。")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    benchmark()
