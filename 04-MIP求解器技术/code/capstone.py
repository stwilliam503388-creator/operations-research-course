"""
🏆 毕业项目：快递站调度系统
难度：★★★☆☆
求解器：HiGHS (免费)
依赖：pip install highspy numpy matplotlib
运行：python capstone.py

简述：3 个快递员（载重不同）、20 个包裹，用 MIP 最小化总配送距离。
对比随机分配基线 + 贪心最近邻，支持时间窗扩展和路线可视化。

=== 你的任务 ===
本文件提供了代码骨架。你需要填充标记为 TODO 的部分。
参考案例 1 (VRP) 的 MTZ 子环消除实现。
"""

import highspy
import numpy as np
import time

# matplotlib 可选
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib 未安装，可视化功能不可用")


# ============================================================
# 数据生成
# ============================================================

def generate_packages(n_packages=20, seed=42):
    """生成包裹数据。
    
    Returns:
        weights:  重量数组 (kg)，范围 3-25
        coords:   坐标数组 (km)，范围 0.1-15
        deadlines: 截止时间数组 (小时，从 8:00 开始算)
    """
    rng = np.random.default_rng(seed)
    weights = rng.uniform(3, 25, n_packages)  # kg
    coords = rng.uniform(0.1, 15, (n_packages, 2))  # 坐标 (x, y)
    
    # 时间窗：每个包裹截止时间 = 8+随机小时，有 30% 是 17:00 前必须送到
    deadlines = np.where(
        rng.random(n_packages) < 0.3,
        17 - 8,  # 17:00
        rng.uniform(12, 20) - 8  # 随机 12:00-20:00
    )
    
    return weights, coords, deadlines


# ============================================================
# MIP 模型求解
# ============================================================

def solve_mip(riders, weights, coords, deadlines=None, use_time_windows=False):
    """用 HiGHS 求解快递员调度 MIP。
    
    Args:
        riders:    快递员列表 [{"id": 0, "name": "小张", "capacity": 100, "max_pkgs": 8}, ...]
        weights:   包裹重量
        coords:    包裹坐标（不含站点，站点在原点的索引 0）
        deadlines: 包裹截止时间（None 表示无时间约束）
        use_time_windows: 是否加入时间窗约束
    
    Returns:
        dict: 含 routes（每个快递员的包裹顺序）、total_distance、solve_time
    """
    n_riders = len(riders)
    n_pkg = len(weights)
    
    # 构建扩展坐标：索引 0 = 快递站（原点），1..n_pkg = 包裹
    all_coords = np.vstack([[[0, 0]], coords])  # (n_pkg+1, 2)
    
    # 距离矩阵（曼哈顿）
    dist = np.zeros((n_pkg + 1, n_pkg + 1))
    for i in range(n_pkg + 1):
        for j in range(n_pkg + 1):
            dist[i, j] = np.abs(all_coords[i, 0] - all_coords[j, 0]) + \
                         np.abs(all_coords[i, 1] - all_coords[j, 1])
    
    # Big-M: 快递员容量上限 × 最大点间距离 × 包裹数
    max_dist = np.max(dist)
    capacity_sum = sum(r["capacity"] for r in riders)
    BIG_M = capacity_sum * max_dist * n_pkg
    
    h = highspy.Highs()
    h.setOptionValue("time_limit", 300.0)
    h.setOptionValue("mip_rel_gap", 0.02)
    h.setOptionValue("output_flag", False)
    
    # 变量：x[a,i,j] — 快递员 a 从点 i 到点 j（i=0 为站点）
    # 变量：y[a,i] — 包裹 i 分配给快递员 a
    # 变量：u[a,i] — MTZ 序位变量（消除子环）
    
    # TODO: 添加变量定义和约束
    # 提示：参考 case01_vrp.py 中的 addVar 方式
    # 约束包括：
    #   1. 每个包裹只分配给一个快递员
    #   2. 快递员容量限制
    #   3. 流量守恒（进来的包裹必须出去）
    #   4. MTZ 子环消除
    #   5. 时间窗约束（如启用）
    
    # 占位：返回模拟结果
    return {
        "routes": {},
        "total_distance": 185.0,
        "solve_time": 2.3,
        "status": "skeleton_placeholder"
    }


# ============================================================
# 基线方法
# ============================================================

def random_baseline(riders, weights, coords, n_trials=100):
    """随机分配 100 次取最好。"""
    n_pkg = len(weights)
    best_dist = float("inf")
    best_routes = None
    
    for _ in range(n_trials):
        # 随机打乱包裹顺序
        order = np.random.permutation(n_pkg)
        routes = {r["id"]: [] for r in riders}
        rider_loads = {r["id"]: [] for r in riders}
        
        pkg_idx = 0
        for rider in riders:
            capacity = rider["capacity"]
            count = rider.get("max_pkgs", n_pkg)
            load = 0
            assigned = []
            
            while pkg_idx < n_pkg and len(assigned) < count:
                pkg = order[pkg_idx]
                if load + weights[pkg] <= capacity:
                    load += weights[pkg]
                    assigned.append(pkg)
                pkg_idx += 1
            
            routes[rider["id"]] = assigned
        
        # 计算距离（每个快递员从站点出发，最近邻顺序配送）
        total = 0
        for rid, pkg_list in routes.items():
            if not pkg_list:
                continue
            prev = 0  # 从站点出发
            for pkg in pkg_list:
                total += np.abs(coords[pkg, 0] - (0 if prev == 0 else coords[prev, 0])) + \
                         np.abs(coords[pkg, 1] - (0 if prev == 0 else coords[prev, 1]))
                prev = pkg + 1  # 包裹索引偏移
            total += np.abs(coords[prev - 1, 0]) + np.abs(coords[prev - 1, 1])  # 回站
        
        if total < best_dist:
            best_dist = total
            best_routes = routes
    
    return best_dist, best_routes


def greedy_baseline(riders, weights, coords):
    """贪心最近邻：每次都把最近的未分配包裹派给容量未满的最近快递员。"""
    n_pkg = len(weights)
    assigned = set()
    routes = {r["id"]: [] for r in riders}
    rider_loads = {r["id"]: 0.0 for r in riders}
    rider_positions = {r["id"]: np.array([0.0, 0.0]) for r in riders}
    
    while len(assigned) < n_pkg:
        best_pkg = None
        best_rider = None
        best_dist = float("inf")
        
        for pkg in range(n_pkg):
            if pkg in assigned:
                continue
            for rider in riders:
                rid = rider["id"]
                if len(routes[rid]) >= rider.get("max_pkgs", n_pkg):
                    continue
                if rider_loads[rid] + weights[pkg] > rider["capacity"]:
                    continue
                
                d = np.abs(coords[pkg, 0] - rider_positions[rid][0]) + \
                    np.abs(coords[pkg, 1] - rider_positions[rid][1])
                
                if d < best_dist:
                    best_dist = d
                    best_pkg = pkg
                    best_rider = rid
        
        if best_pkg is None:
            break
        
        routes[best_rider].append(best_pkg)
        rider_loads[best_rider] += weights[best_pkg]
        rider_positions[best_rider] = coords[best_pkg]
        assigned.add(best_pkg)
    
    # 计算总距离
    total = 0.0
    for rid, pkg_list in routes.items():
        if not pkg_list:
            continue
        prev = np.array([0.0, 0.0])
        for pkg in pkg_list:
            total += np.abs(coords[pkg, 0] - prev[0]) + np.abs(coords[pkg, 1] - prev[1])
            prev = coords[pkg]
        total += np.abs(prev[0]) + np.abs(prev[1])  # 回站
    
    return total, routes


# ============================================================
# 可视化
# ============================================================

def visualize(routes, coords, title="配送路线图"):
    """用 matplotlib 画路线图。"""
    if not HAS_MPL:
        print("[WARNING] 无法可视化：matplotlib 未安装")
        return
    
    colors = ["#E74C3C", "#2ECC71", "#3498DB"]  # 红绿蓝
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 站点
    ax.scatter(0, 0, c="black", marker="*", s=300, zorder=5, label="配送站")
    
    # 包裹点
    ax.scatter(coords[:, 0], coords[:, 1], c="gray", s=50, alpha=0.5, label="包裹")
    
    # 每个快递员的路线
    for i, (rid, pkg_list) in enumerate(routes.items()):
        color = colors[i % len(colors)]
        if not pkg_list:
            continue
        
        # 路线：站点 → 第1个 → 第2个 → ... → 站点
        path = [np.array([0.0, 0.0])] + [coords[p] for p in pkg_list] + [np.array([0.0, 0.0])]
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-o", color=color, linewidth=2, markersize=6,
                label=f"快递员 {rid}")
    
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    
    plt.tight_layout()
    plt.savefig("capstone_routes.png", dpi=150)
    print("[INFO] 路线图已保存到 capstone_routes.png")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # 快递员定义
    riders = [
        {"id": 0, "name": "小张", "capacity": 100.0, "max_pkgs": 8},
        {"id": 1, "name": "小李", "capacity": 80.0, "max_pkgs": 8},
        {"id": 2, "name": "小王", "capacity": 120.0, "max_pkgs": 8},
    ]
    
    print("="*70)
    print("🏆 毕业项目：快递站调度系统")
    print("="*70)
    
    # 生成数据
    weights, coords, deadlines = generate_packages(n_packages=20)
    print(f"\n包裹总数: {len(weights)}, 总重量: {weights.sum():.1f} kg")
    print(f"快递员: {len(riders)} 人, 总载货上限: {sum(r['capacity'] for r in riders)} kg")
    
    # 基线
    print("\n--- 基线方法 ---")
    
    t0 = time.time()
    rand_dist, rand_routes = random_baseline(riders, weights, coords, n_trials=100)
    rand_time = time.time() - t0
    print(f"随机分配 (100次): {rand_dist:.1f} km  (耗时 {rand_time:.1f}s)")
    
    t0 = time.time()
    greedy_dist, greedy_routes = greedy_baseline(riders, weights, coords)
    greedy_time = time.time() - t0
    print(f"贪心最近邻:       {greedy_dist:.1f} km  (耗时 {greedy_time:.1f}s)")
    
    # MIP 求解
    print("\n--- MIP 求解 ---")
    result = solve_mip(riders, weights, coords)
    print(f"MIP 优化:          {result['total_distance']:.1f} km  "
          f"(耗时 {result['solve_time']:.1f}s, 状态: {result['status']})")
    
    # 对比汇总
    print(f"\n{'='*70}")
    print(f"{'方法':<15} {'总距离 (km)':<14} {'vs MIP':<10} {'耗时 (s)':<10}")
    print(f"{'-'*50}")
    print(f"{'随机分配':<15} {rand_dist:>12.1f}  {'--':>8}  {rand_time:>8.1f}")
    print(f"{'贪心最近邻':<15} {greedy_dist:>12.1f}  {'--':>8}  {greedy_time:>8.1f}")
    print(f"{'MIP 优化':<15} {result['total_distance']:>12.1f}  {'基线':>8}  {result['solve_time']:>8.1f}")
    print(f"{'='*70}")
    
    # 可视化
    visualize(rand_routes, coords, title="随机分配路线")
    
    print("\n💡 提示: 以上 MIP 结果为骨架占位值。")
    print("请填充 solve_mip() 函数中的 TODO 部分以获取真实优化结果。")
    print("完成后对比应显示 MIP 距离 < 贪心最近邻 < 随机分配。")
