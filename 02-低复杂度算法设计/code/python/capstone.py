"""
毕业项目：智能外卖调度系统 (Capstone)
=====================================

脚手架代码，包含：
  - 随机数据生成器
  - 方案一：分治 + DP (TSP) + 贪心
  - 方案二：纯贪心快速版
  - 方案三：模拟退火（可运行的订单序列邻域搜索）
  - evaluate() 评估函数
  - 对比测试入口

兼容 Python 3.9+，无外部依赖（仅标准库）。
matplotlib 为可视化可选依赖。
"""


# 教学注释：关注复杂度、状态设计与数据结构选择如何影响可运行规模。
# 阅读时对照搜索、剪枝或启发式步骤，理解它们减少计算量的方式。



import math
import random
import time
import itertools


# ============================================================
# 数据生成器
# ============================================================

def generate_data(
    n_orders=50,
    n_restaurants=10,
    n_riders=8,
    capacity=5,
    speed=0.5,
    seed=42
):
    """
    生成随机外卖调度测试数据。

    参数:
        n_orders: 订单数量
        n_restaurants: 餐厅数量
        n_riders: 骑手数量
        capacity: 每个骑手容量
        speed: 骑手速度 (km/min)
        seed: 随机种子

    返回:
        dict: {restaurants, orders, riders, speed}
    """
    rng = random.Random(seed)

    # 1. 生成随机餐厅位置 (0~10 千米范围)
    restaurants = []
    for i in range(n_restaurants):
        restaurants.append({
            "id": i,
            "x": round(rng.uniform(0, 10), 2),
            "y": round(rng.uniform(0, 10), 2),
        })

    # 2. 生成随机订单
    orders = []
    for i in range(n_orders):
        rest_id = rng.randint(0, n_restaurants - 1)
        rest = restaurants[rest_id]
        # 顾客位置在餐厅附近随机偏移 (0~5km)
        orders.append({
            "id": i,
            "restaurant_id": rest_id,
            "customer_x": round(rest["x"] + rng.uniform(-5, 5), 2),
            "customer_y": round(rest["y"] + rng.uniform(-5, 5), 2),
            "deadline": rng.randint(15, 60),   # 15~60 分钟
            "penalty": round(rng.uniform(1.0, 10.0), 1),
        })

    # 3. 生成骑手 (起点在原点附近)
    riders = []
    for i in range(n_riders):
        riders.append({
            "id": i,
            "start_x": round(rng.uniform(0, 2), 2),
            "start_y": round(rng.uniform(0, 2), 2),
            "capacity": capacity,
        })

    return {
        "restaurants": restaurants,
        "orders": orders,
        "riders": riders,
        "speed": speed,
    }


# ============================================================
# 工具函数
# ============================================================

def euclidean(p1, p2):
    """欧几里得距离。"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_distance_matrix(points):
    """计算所有点之间的成对距离矩阵。"""
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = euclidean(points[i], points[j])
    return dist


# ============================================================
# 方案一：分治 + DP (TSP) + 贪心
# ============================================================

class DeliveryScheduler:
    """
    标准调度器：分治区域划分 → 贪心分配骑手 → DP 求解路径。
    """

    def __init__(self, data):
        self.data = data
        self.restaurants = {r["id"]: r for r in data["restaurants"]}
        self.orders = data["orders"]
        self.riders = data["riders"]
        self.speed = data["speed"]

    # ---- Step 1: 分治 - 区域划分 ----

    def _cluster_regions(self, n_clusters=3, max_iter=20):
        """
        用 K-Means 将顾客位置划分为 n_clusters 个区域。
        返回每个订单所属的区域编号。
        """
        # 提取所有顾客坐标
        points = [(o["customer_x"], o["customer_y"]) for o in self.orders]
        n = len(points)
        if n <= n_clusters:
            return list(range(n))

        # 初始化：取前 n_clusters 个顾客为初始中心
        centroids = points[:n_clusters]

        labels = [0] * n
        for _ in range(max_iter):
            # 分配：每个点到最近的中心
            new_labels = []
            for p in points:
                dists = [euclidean(p, c) for c in centroids]
                new_labels.append(min(range(len(dists)), key=lambda i: dists[i]))

            # 更新中心
            new_centroids = []
            for cid in range(n_clusters):
                cluster_pts = [points[i] for i in range(n) if new_labels[i] == cid]
                if cluster_pts:
                    avg_x = sum(p[0] for p in cluster_pts) / len(cluster_pts)
                    avg_y = sum(p[1] for p in cluster_pts) / len(cluster_pts)
                    new_centroids.append((avg_x, avg_y))
                else:
                    new_centroids.append(centroids[cid])

            if new_labels == labels:
                break
            labels = new_labels
            centroids = new_centroids

        return labels

    # ---- Step 2A: 贪心分配骑手 ----

    def _assign_riders(self, region_labels):
        """
        按区域订单比例，贪心分配骑手到最近区域。
        """
        n_regions = max(region_labels) + 1
        region_order_counts = [0] * n_regions
        for lbl in region_labels:
            region_order_counts[lbl] += 1

        # 统计每个区域的「重心」
        region_centers = []
        for r in range(n_regions):
            pts = [
                (self.orders[i]["customer_x"], self.orders[i]["customer_y"])
                for i, lbl in enumerate(region_labels) if lbl == r
            ]
            if pts:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
            else:
                cx = cy = 5.0
            region_centers.append((cx, cy))

        # 每个骑手分配到最近区域
        rider_assignments = [[] for _ in range(n_regions)]
        for rider in self.riders:
            start = (rider["start_x"], rider["start_y"])
            nearest = min(
                range(n_regions),
                key=lambda r: euclidean(start, region_centers[r])
            )
            rider_assignments[nearest].append(rider["id"])

        return rider_assignments, region_centers

    # ---- Step 2B: TSP DP (Held-Karp) ----

    def _tsp_dp(self, points):
        """
        TSP 精确求解 (Held-Karp DP)。
        状态: dp[mask][last] = 已访问 mask, 当前在 last 的最小距离。

        参数:
            points: 需要访问的路径点列表 [(x,y), ...]
                    第一个点是起点，必须首先访问。

        返回:
            (min_distance, route) 其中 route 是点的原始索引顺序
        """
        m = len(points)
        if m <= 1:
            return 0.0, [0]
        if m > 12:
            # 超过 12 个点用最近邻贪心回退
            return self._tsp_nearest_neighbor(points)

        dist = compute_distance_matrix(points)
        INF = float('inf')
        size = 1 << m
        dp = [[INF] * m for _ in range(size)]
        parent = [[-1] * m for _ in range(size)]

        # 初始化：从起点 (0) 到每个点
        dp[1][0] = 0.0  # 起点已经在 mask 中

        for mask in range(1, size):
            for last in range(m):
                if dp[mask][last] == INF:
                    continue
                # 尝试去下一个未访问的点
                for nxt in range(m):
                    if mask & (1 << nxt):
                        continue
                    new_mask = mask | (1 << nxt)
                    new_dist = dp[mask][last] + dist[last][nxt]
                    if new_dist < dp[new_mask][nxt]:
                        dp[new_mask][nxt] = new_dist
                        parent[new_mask][nxt] = last

        # 找到最优终点（回到起点）
        full_mask = size - 1
        best = INF
        best_last = -1
        for last in range(1, m):  # 不能以起点结束（除非回到起点）
            total = dp[full_mask][last] + dist[last][0]
            if total < best:
                best = total
                best_last = last

        if best == INF:
            return 0.0, list(range(m))

        # 回溯路径
        route = []
        mask = full_mask
        last = best_last
        while last != -1:
            route.append(last)
            prev = parent[mask][last]
            mask ^= (1 << last)
            last = prev
        route.reverse()

        return best, route

    def _tsp_nearest_neighbor(self, points):
        """最近邻贪心 TSP (回退方案)。"""
        m = len(points)
        if m <= 1:
            return 0.0, [0]
        dist = compute_distance_matrix(points)
        visited = [False] * m
        visited[0] = True
        route = [0]
        total = 0.0
        current = 0
        for _ in range(m - 1):
            nearest = -1
            min_d = float('inf')
            for j in range(m):
                if not visited[j] and dist[current][j] < min_d:
                    min_d = dist[current][j]
                    nearest = j
            if nearest == -1:
                break
            visited[nearest] = True
            route.append(nearest)
            total += min_d
            current = nearest
        total += dist[current][0]  # 回到起点
        return total, route

    def _compute_penalty(self, route_points, order_deadlines, speed):
        """
        计算一条路径的超时罚金。

        参数:
            route_points: 路径点序列 [(x,y), ...]
            order_deadlines: 每个订单的 deadline 列表
            speed: 骑手速度

        返回:
            (total_penalty, arrival_times)
        """
        total_penalty = 0.0
        arrival_times = [0.0]
        current_time = 0.0
        for i in range(1, len(route_points)):
            dist = euclidean(route_points[i-1], route_points[i])
            current_time += dist / speed
            arrival_times.append(current_time)
            if i <= len(order_deadlines):
                delay = current_time - order_deadlines[i - 1]
                if delay > 0:
                    total_penalty += delay  # * penalty 简化处理
        return total_penalty, arrival_times

    # ---- 主求解方法 ----

    def solve(self):
        """
        执行完整的分治 + DP + 贪心调度方案。

        返回:
            dict: {assignments, total_cost}
        """
        # Step 1: 区域划分
        region_labels = self._cluster_regions(n_clusters=3)

        # Step 2: 骑手分配
        rider_assignments, region_centers = self._assign_riders(region_labels)

        # Step 3: 对每个区域的每个骑手，规划路径
        assignments = []
        total_cost = 0.0

        n_regions = max(region_labels) + 1
        for region_id in range(n_regions):
            # 该区域的订单
            region_order_ids = [
                i for i, lbl in enumerate(region_labels) if lbl == region_id
            ]
            rider_ids = rider_assignments[region_id]

            if not region_order_ids or not rider_ids:
                continue

            # 按 deadline 排序分组（先到先服务）
            region_orders = sorted(
                [self.orders[i] for i in region_order_ids],
                key=lambda o: o["deadline"]
            )

            # 平均分配订单给该区域的骑手
            orders_per_rider = max(1, len(region_orders) // len(rider_ids))
            for idx, rider_id in enumerate(rider_ids):
                batch = region_orders[
                    idx * orders_per_rider:(idx + 1) * orders_per_rider
                ]
                if not batch:
                    continue

                rider = self.riders[rider_id]
                # 构造路径点：起点 → 各餐厅取餐 → 各顾客送餐
                points = [(rider["start_x"], rider["start_y"])]
                order_deadlines = []
                for o in batch:
                    rest = self.restaurants[o["restaurant_id"]]
                    points.append((rest["x"], rest["y"]))  # 取餐
                    points.append((o["customer_x"], o["customer_y"]))  # 送餐
                    order_deadlines.append(o["deadline"])

                # 用 TSP DP 求最优路径
                min_dist, route = self._tsp_dp(points)

                # 计算路径罚金
                route_pts = [points[i] for i in route]
                penalty, arrivals = self._compute_penalty(
                    route_pts, order_deadlines, self.speed
                )

                # 构造返回格式
                route_steps = []
                pt_idx = 0
                for o_idx, o in enumerate(batch):
                    route_steps.append({
                        "type": "pickup",
                        "order_id": o["id"],
                        "x": self.restaurants[o["restaurant_id"]]["x"],
                        "y": self.restaurants[o["restaurant_id"]]["y"],
                    })
                    route_steps.append({
                        "type": "deliver",
                        "order_id": o["id"],
                        "x": o["customer_x"],
                        "y": o["customer_y"],
                    })

                assignment = {
                    "rider_id": rider_id,
                    "route": route_steps,
                    "total_distance": round(min_dist, 2),
                    "total_penalty": round(penalty, 2),
                }
                assignments.append(assignment)
                total_cost += min_dist + penalty

        return {
            "assignments": assignments,
            "total_cost": round(total_cost, 2),
        }


# ============================================================
# 方案二：纯贪心快速版
# ============================================================

def solve_greedy(data, order_sequence=None):
    """
    纯贪心快速版：
    - 订单按 deadline 排序
    - 分配给最近且未满的骑手
    - 骑手用最近邻走路线

    返回:
        dict: {assignments, total_cost}
    """
    if order_sequence is None:
        orders = sorted(data["orders"], key=lambda o: o["deadline"])
    else:
        order_by_id = {o["id"]: o for o in data["orders"]}
        orders = [order_by_id[order_id] for order_id in order_sequence if order_id in order_by_id]
    riders = [
        {**r, "current_x": r["start_x"], "current_y": r["start_y"],
         "load": 0, "route": [], "distance": 0.0, "penalty": 0.0,
         "time": 0.0}
        for r in data["riders"]
    ]
    restaurants = {r["id"]: r for r in data["restaurants"]}
    speed = data["speed"]

    for o in orders:
        rest = restaurants[o["restaurant_id"]]
        # 找最近且未满的骑手
        best_rider = None
        best_cost = float('inf')
        for r in riders:
            if r["load"] >= r["capacity"]:
                continue
            d = euclidean((r["current_x"], r["current_y"]), (rest["x"], rest["y"]))
            if d < best_cost:
                best_cost = d
                best_rider = r

        if best_rider is None:
            continue  # 没有可用骑手

        r = best_rider
        # 移动到餐厅取餐
        travel_dist = euclidean(
            (r["current_x"], r["current_y"]), (rest["x"], rest["y"])
        )
        r["time"] += travel_dist / speed
        r["distance"] += travel_dist
        r["current_x"], r["current_y"] = rest["x"], rest["y"]

        # 移动到顾客送餐
        travel_dist = euclidean(
            (r["current_x"], r["current_y"]), (o["customer_x"], o["customer_y"])
        )
        r["time"] += travel_dist / speed
        r["distance"] += travel_dist
        r["current_x"], r["current_y"] = o["customer_x"], o["customer_y"]

        # 超时罚金
        delay = r["time"] - o["deadline"]
        if delay > 0:
            r["penalty"] += delay

        r["load"] += 1
        r["route"].append({
            "type": "pickup", "order_id": o["id"],
            "x": rest["x"], "y": rest["y"],
        })
        r["route"].append({
            "type": "deliver", "order_id": o["id"],
            "x": o["customer_x"], "y": o["customer_y"],
        })

    assignments = []
    total_cost = 0.0
    for r in riders:
        if not r["route"]:
            continue
        ass = {
            "rider_id": r["id"],
            "route": r["route"],
            "total_distance": round(r["distance"], 2),
            "total_penalty": round(r["penalty"], 2),
        }
        assignments.append(ass)
        total_cost += r["distance"] + r["penalty"]

    return {"assignments": assignments, "total_cost": round(total_cost, 2)}


# ============================================================
# 方案三：模拟退火（骨架）
# ============================================================

def solve_simulated_annealing(data, T_init=20.0, T_min=0.05, alpha=0.97, max_iter=300, seed=7):
    """
    模拟退火版：在订单访问顺序上做邻域搜索。

    邻域操作很简单：随机交换两个订单在调度序列中的位置，然后用同一个
    贪心分配器评估总成本。它不是工业级 SA，但能完整演示“当前解 →
    邻域解 → 按温度接受”的优化循环。

    返回:
        dict: {assignments, total_cost}
    """
    rng = random.Random(seed)
    current_order = [o["id"] for o in sorted(data["orders"], key=lambda o: o["deadline"])]
    current_solution = solve_greedy(data, current_order)
    current_cost = current_solution["total_cost"]
    best_order = current_order[:]
    best_solution = current_solution
    best_cost = current_cost

    T = T_init
    iteration = 0
    while T > T_min and iteration < max_iter and len(current_order) >= 2:
        neighbor_order = current_order[:]
        i, j = rng.sample(range(len(neighbor_order)), 2)
        neighbor_order[i], neighbor_order[j] = neighbor_order[j], neighbor_order[i]

        neighbor_solution = solve_greedy(data, neighbor_order)
        neighbor_cost = neighbor_solution["total_cost"]
        delta = neighbor_cost - current_cost

        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_order = neighbor_order
            current_solution = neighbor_solution
            current_cost = neighbor_cost

            if current_cost < best_cost:
                best_order = current_order[:]
                best_solution = current_solution
                best_cost = current_cost

        T *= alpha
        iteration += 1

    best_solution["search_iterations"] = iteration
    best_solution["order_sequence"] = best_order
    return best_solution


# ============================================================
# 评估函数
# ============================================================

def evaluate(result, data):
    """
    评估调度结果。

    返回:
        dict: 各项指标
    """
    total_distance = sum(
        a["total_distance"] for a in result["assignments"]
    )
    total_penalty = sum(
        a["total_penalty"] for a in result["assignments"]
    )

    # 骑手利用率
    total_capacity = sum(r["capacity"] for r in data["riders"])
    total_assigned = sum(
        len(a["route"]) // 2 for a in result["assignments"]
    )
    utilization = total_assigned / total_capacity if total_capacity > 0 else 0

    # 理论下界：所有订单的直线距离之和（从餐厅到顾客）
    lower_bound = 0.0
    for o in data["orders"]:
        rest = data["restaurants"][o["restaurant_id"]]
        lower_bound += euclidean(
            (rest["x"], rest["y"]),
            (o["customer_x"], o["customer_y"])
        )

    return {
        "total_distance": round(total_distance, 2),
        "total_penalty": round(total_penalty, 2),
        "total_cost": round(total_distance + total_penalty, 2),
        "utilization": round(utilization, 3),
        "lower_bound": round(lower_bound, 2),
        "gap_to_lower_bound": round(
            (total_distance + total_penalty) / lower_bound
            if lower_bound > 0 else float('inf'),
            3
        ),
        "n_assignments": len(result["assignments"]),
    }


# ============================================================
# 主对比入口
# ============================================================

def main():
    print()
    print("=" * 65)
    print("  🏆 毕业项目：智能外卖调度系统")
    print("=" * 65)

    for n_orders in [20, 50, 100]:
        print(f"\n  --- 测试规模: {n_orders} 订单 ---")
        data = generate_data(n_orders=n_orders, seed=42)

        # 方案一：分治 + DP + 贪心
        start = time.perf_counter()
        scheduler = DeliveryScheduler(data)
        result1 = scheduler.solve()
        t1 = time.perf_counter() - start
        eval1 = evaluate(result1, data)

        # 方案二：纯贪心
        start = time.perf_counter()
        result2 = solve_greedy(data)
        t2 = time.perf_counter() - start
        eval2 = evaluate(result2, data)

        # 方案三：模拟退火
        start = time.perf_counter()
        result3 = solve_simulated_annealing(data)
        t3 = time.perf_counter() - start
        eval3 = evaluate(result3, data)

        # 对比输出
        print(f"  {'指标':<20} | {'分治+DP+贪心':<18} | {'纯贪心':<12} | {'模拟退火':<12}")
        print(f"  {'-'*19} | {'-'*17} | {'-'*11} | {'-'*11}")
        print(f"  {'总成本':<20} | {eval1['total_cost']:<18.2f} | {eval2['total_cost']:<12.2f} | {eval3['total_cost']:<12.2f}")
        print(f"  {'总距离':<20} | {eval1['total_distance']:<18.2f} | {eval2['total_distance']:<12.2f} | {eval3['total_distance']:<12.2f}")
        print(f"  {'总罚金':<20} | {eval1['total_penalty']:<18.2f} | {eval2['total_penalty']:<12.2f} | {eval3['total_penalty']:<12.2f}")
        print(f"  {'骑手利用率':<20} | {eval1['utilization']:<18.3f} | {eval2['utilization']:<12.3f} | {eval3['utilization']:<12.3f}")
        print(f"  {'与下界差距':<20} | {eval1['gap_to_lower_bound']:<18.3f} | {eval2['gap_to_lower_bound']:<12.3f} | {eval3['gap_to_lower_bound']:<12.3f}")
        print(f"  {'耗时(秒)':<20} | {t1:<18.4f} | {t2:<12.4f} | {t3:<12.4f}")

    print()
    print("=" * 65)
    print("  总结")
    print("=" * 65)
    print("  • 分治+DP+贪心：成本最优，但耗时较长（适合离线调度）")
    print("  • 纯贪心：极快，但成本通常较高（适合在线调度）")
    print("  • 模拟退火：在贪心序列附近搜索，适合做轻量改进")
    print("  • 推荐组合：分治缩小规模 → 每个小规模精确解 → 元启发式微调")
    print()


if __name__ == "__main__":
    main()
