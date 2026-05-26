"""
案例4: 1D 波动方程 FDM 求解 — 地震波传播模拟
=================================================

物理方程: ∂²u/∂t² = c² ∂²u/∂x²
数值方法: 显式中心差分格式
边界条件: 反射边界 (固定端) / 吸收边界 (一阶 Mur)

参考: 04-case-wave.md
"""

import numpy as np


# ============================================================
# 参数设置
# ============================================================
def set_parameters():
    """
    设置物理和数值参数。

    场景: 地震波在地层中传播
      - 地层长度 L = 1000 m
      - 波速 c = 3000 m/s (典型地壳 P 波速度)
      - 初始条件: 高斯波包 (模拟震源)
      - 边界: 可选反射或吸收

    返回: dict, 包含所有参数
    """
    params = {
        # 物理参数
        "L": 1000.0,            # 地层长度 [m]
        "c": 3000.0,            # 波速 [m/s]
        # 源参数 (高斯波包)
        "source_x0": 200.0,     # 震源中心位置 [m]
        "source_sigma": 30.0,   # 高斯波包宽度 [m]
        "source_amplitude": 1.0, # 初始振幅
        # 数值参数
        "nx": 200,              # 空间网格数
        "nt": 500,              # 时间步数
        "boundary_type": "reflective",  # "reflective" 或 "absorbing"
    }
    return params


# ============================================================
# 网格构建
# ============================================================
def build_mesh(params):
    """
    构建一维网格并计算 CFL 条件。

    参数:
        params: dict, 来自 set_parameters()

    返回:
        dict: x, dx, dt, dt_cfl, courant 数
    """
    L = params["L"]
    nx = params["nx"]
    c = params["c"]

    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    # CFL 条件: c·Δt/Δx ≤ 1
    dt_cfl = dx / c
    dt = dt_cfl * 0.9  # 取 90% 保证安全

    return {
        "x": x,
        "dx": dx,
        "dt": dt,
        "dt_cfl": dt_cfl,
        "courant": c * dt / dx,
        "nx": nx,
    }


# ============================================================
# 初始条件: 高斯波包
# ============================================================
def initial_condition(x, params):
    """
    设置初始位移 u(x, 0) — 高斯波包。

    模拟震源: 在 x0 处的一个初始隆起。

    参数:
        x: ndarray, 空间坐标
        params: dict, 参数集

    返回:
        ndarray: 初始位移场
    """
    x0 = params["source_x0"]
    sigma = params["source_sigma"]
    A = params["source_amplitude"]
    return A * np.exp(-((x - x0)**2) / (2 * sigma**2))


def initial_velocity(x, params):
    """
    设置初始速度 ∂u/∂t(x, 0) — 通常为 0。

    参数:
        x: ndarray, 空间坐标
        params: dict, 参数集

    返回:
        ndarray: 初始速度场 (全零)
    """
    return np.zeros_like(x)


# ============================================================
# CFL 条件验证
# ============================================================
def check_cfl(mesh):
    """
    验证 CFL 条件是否满足。

    返回: bool, True 表示稳定
    """
    courant = mesh["courant"]
    cfl_limit = mesh["dt_cfl"]
    dt_used = mesh["dt"]

    print(f"[CFL 验证]")
    print(f"  Courant 数: σ = c·Δt/Δx = {courant:.4f}")
    print(f"  CFL 条件: σ ≤ 1.0    → {'✅ 稳定' if courant <= 1.0 else '❌ 不稳定'}")
    print(f"  dt_used = {dt_used*1e3:.3f} ms")
    print(f"  dt_max (CFL) = {cfl_limit*1e3:.3f} ms")

    if courant > 1.0:
        print("  ⚠️  警告: CFL 条件不满足, 数值解可能发散!")

    return courant <= 1.0


# ============================================================
# 显式时间推进
# ============================================================
def solve_wave(params, mesh):
    """
    显式中心差分格式求解 1D 波动方程。

    离散格式:
      u^{n+1}_i = 2u^n_i - u^{n-1}_i
                  + σ² · (u^n_{i-1} - 2u^n_i + u^n_{i+1})

    其中 σ = c·Δt/Δx (Courant 数)

    第一步 (n=0) 需要特殊处理, 使用初始速度条件。

    参数:
        params: dict, 参数集
        mesh: dict, 网格信息

    返回:
        dict: u (完整演化), x, t 数组
    """
    nx = mesh["nx"]
    nt = params["nt"]
    dx = mesh["dx"]
    dt = mesh["dt"]
    c = params["c"]
    x = mesh["x"]

    sigma = c * dt / dx  # Courant 数
    sigma2 = sigma**2

    boundary_type = params["boundary_type"]

    print(f"\n[波动方程求解]")
    print(f"  σ = c·Δt/Δx = {sigma:.4f}")
    print(f"  边界类型: {boundary_type}")
    print()

    # 存储所有时刻的位移
    u = np.zeros((nt + 1, nx))

    # 初始条件
    u[0, :] = initial_condition(x, params)
    v0 = initial_velocity(x, params)

    # 第一步 (n=0): 使用初始速度
    # u^1_i ≈ u^0_i + v0_i·Δt + (σ²/2)·(u^0_{i-1} - 2u^0_i + u^0_{i+1})
    u[1, 1:-1] = (u[0, 1:-1]
                  + v0[1:-1] * dt
                  + 0.5 * sigma2 * (u[0, :-2] - 2 * u[0, 1:-1] + u[0, 2:]))
    apply_boundary(u, 1, mesh, params)

    # 时间推进 (n >= 1)
    for n in range(1, nt):
        # 内部点: 中心差分
        u[n + 1, 1:-1] = (2 * u[n, 1:-1] - u[n - 1, 1:-1]
                          + sigma2 * (u[n, :-2] - 2 * u[n, 1:-1] + u[n, 2:]))

        # 应用边界条件
        apply_boundary(u, n + 1, mesh, params)

    t = np.arange(nt + 1) * dt * 1e3  # 毫秒

    return {"u": u, "x": x, "t": t, "u_final": u[-1, :]}


# ============================================================
# 边界处理
# ============================================================
def apply_boundary(u, n, mesh, params):
    """
    在指定时刻 n 应用边界条件。

    反射边界 (固定端): u = 0
    吸收边界 (一阶 Mur): ∂u/∂t ± c·∂u/∂x = 0

    参数:
        u: ndarray, 完整位移场 [nt+1, nx]
        n: int, 当前时刻索引
        mesh: dict, 网格信息
        params: dict, 参数集
    """
    nx = mesh["nx"]
    dx = mesh["dx"]
    dt = mesh["dt"]
    c = params["c"]
    boundary_type = params["boundary_type"]

    if boundary_type == "reflective":
        # 反射边界: 固定端 (u = 0)
        u[n, 0] = 0.0
        u[n, -1] = 0.0

    elif boundary_type == "absorbing":
        # 一阶 Mur 吸收边界
        # 左边界 (x=0): ∂u/∂t - c·∂u/∂x = 0 (入射波向左)
        # 离散: u^{n+1}_0 = u^n_1 + (cΔt - Δx)/(cΔt + Δx)·(u^{n+1}_1 - u^n_0)
        # 简化一阶: u^{n+1}_0 = u^n_1 + (σ - 1)/(σ + 1)·(u^{n+1}_1 - u^n_0)
        sigma = c * dt / dx
        if n > 0:
            u[n, 0] = (u[n - 1, 1]
                       + (sigma - 1) / (sigma + 1) * (u[n, 1] - u[n - 1, 0]))
            u[n, -1] = (u[n - 1, -2]
                        + (sigma - 1) / (sigma + 1) * (u[n, -2] - u[n - 1, -1]))


# ============================================================
# 达朗贝尔解析解
# ============================================================
def d_alembert_solution(x, t, params):
    """
    达朗贝尔公式: 无限域上 1D 波动方程的解析解。

    u(x,t) = [f(x - ct) + f(x + ct)] / 2
    + (1/2c) ∫_{x-ct}^{x+ct} g(ξ) dξ

    其中 f(x) = u(x,0), g(x) = ∂u/∂t(x,0)

    对于纯初始位移 (g=0):
      u(x,t) = [f(x-ct) + f(x+ct)] / 2

    参数:
        x: ndarray, 空间坐标
        t: float, 时间
        params: dict, 参数集

    返回:
        ndarray: 解析解
    """
    c = params["c"]
    x0 = params["source_x0"]
    sigma = params["source_sigma"]
    A = params["source_amplitude"]

    def f(xi):
        """初始位移函数 f(x) = u(x, 0)"""
        return A * np.exp(-((xi - x0)**2) / (2 * sigma**2))

    # 达朗贝尔公式 (g=0)
    return 0.5 * (f(x - c * t) + f(x + c * t))


# ============================================================
# 解析解对比验证
# ============================================================
def verify_with_analytical(params, mesh):
    """
    将数值解与达朗贝尔解析解对比。
    在 CFL 稳定条件下, 在波到达边界之前进行比较。

    参数:
        params: dict, 参数集
        mesh: dict, 网格信息
    """
    print("=" * 60)
    print("解析验证: 与达朗贝尔公式对比")
    print("=" * 60)

    dt = mesh["dt"]
    # 计算到波到达左边界或右边界之前
    # 源在 x0=200, 到左边界距离=200, 到右边界距离=800
    # 波速 c=3000, 到左边界时间 ≈ 200/3000 ≈ 0.067s
    # 取安全时间 T = 0.05s
    T_verify = 0.05
    nt_verify = int(T_verify / dt)

    # 重新计算到短时间
    params_verify = params.copy()
    params_verify["nt"] = nt_verify
    # 使用吸收边界避免反射干扰
    params_verify["boundary_type"] = "absorbing"

    result = solve_wave(params_verify, mesh)
    x = mesh["x"]
    t_final = nt_verify * dt

    u_numerical = result["u"][-1, :]
    u_analytical = d_alembert_solution(x, t_final, params_verify)

    # 计算误差 (只在波包范围内, 忽略边界)
    # 波包大致在 [x0 - c*t - 3σ, x0 + c*t + 3σ]
    c = params["c"]
    x0 = params["source_x0"]
    sigma = params["source_sigma"]
    x_min = max(0, x0 - c * t_final - 3 * sigma)
    x_max = min(params["L"], x0 + c * t_final + 3 * sigma)
    mask = (x >= x_min) & (x <= x_max)

    if np.any(mask):
        error = np.max(np.abs(u_numerical[mask] - u_analytical[mask]))
        max_amp = np.max(np.abs(u_analytical[mask]))
        rel_error = error / max_amp * 100 if max_amp > 0 else 0.0
        print(f"  时间: t = {t_final*1000:.2f} ms")
        print(f"  比较区域: [{x_min:.1f}, {x_max:.1f}] m")
        print(f"  最大绝对误差: {error:.6f}")
        print(f"  相对误差: {rel_error:.2f}%")
    else:
        print("  警告: 波包已超出计算域, 无法进行有效比较")
        error = None

    # 抽样对比
    if error is not None and np.any(mask):
        sample_indices = np.where(mask)[0]
        step = max(1, len(sample_indices) // 5)
        sample_indices = sample_indices[::step]
        print(f"\n  位置(m)  |  数值解  |  解析解  |  误差")
        print("  " + "-" * 45)
        for i in sample_indices[:8]:
            print(f"  {x[i]:8.1f} | {u_numerical[i]:8.4f} | "
                  f"{u_analytical[i]:8.4f} | "
                  f"{abs(u_numerical[i] - u_analytical[i]):8.4f}")

    print()


# ============================================================
# 可视化 (文本输出)
# ============================================================
def visualize(result, mesh, params, title="波动传播"):
    """
    以文本形式输出波动传播状态。

    输出:
      - 选取几个时刻的快照
      - 文本条形图显示波场

    参数:
        result: dict, 求解结果
        mesh: dict, 网格信息
        params: dict, 参数集
        title: str, 标题
    """
    u = result["u"]
    x = mesh["x"]
    nt = params["nt"]

    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  网格: {mesh['nx']} 点, "
          f"dx = {mesh['dx']:.1f} m, "
          f"dt = {mesh['dt']*1e3:.3f} ms")
    print(f"  波速: c = {params['c']} m/s")
    print()

    # 选取若干时刻输出
    n_snapshots = 6
    snap_indices = np.linspace(0, nt, n_snapshots, dtype=int)

    for idx in snap_indices:
        t_ms = idx * mesh["dt"] * 1e3
        print(f"  t = {t_ms:8.2f} ms")
        print("  " + "-" * 55)

        # 文本条形图
        bar_scale = 40 / max(1e-10, np.max(np.abs(u[idx, :])))
        for i in range(0, len(x), max(1, len(x) // 8)):
            xi = x[i]
            bar_len = int(u[idx, i] * bar_scale)
            if bar_len >= 0:
                bar = " " * 10 + "█" * min(bar_len, 40)
            else:
                bar = " " * (10 + bar_len + 40) + "█" * min(-bar_len, 40)
            print(f"    {xi:6.0f}m | {u[idx, i]:+7.4f} {bar}")
        print()

    print(f"  最大振幅: {np.max(np.abs(u)):.4f}")
    print()


# ============================================================
# 边界对比演示
# ============================================================
def compare_boundaries(params, mesh):
    """
    对比反射边界和吸收边界的效果。

    参数:
        params: dict, 参数集
        mesh: dict, 网格信息
    """
    print("=" * 60)
    print("边界对比: 反射边界 vs 吸收边界")
    print("=" * 60)

    # 反射边界
    params_r = params.copy()
    params_r["boundary_type"] = "reflective"
    params_r["nt"] = 400
    result_r = solve_wave(params_r, mesh)

    # 吸收边界
    params_a = params.copy()
    params_a["boundary_type"] = "absorbing"
    params_a["nt"] = 400
    result_a = solve_wave(params_a, mesh)

    x = mesh["x"]
    L = params["L"]

    # 在后期时刻观察边界行为
    check_time = 350  # 时间步索引
    if check_time < len(result_r["u"]):
        print(f"\n  t = {check_time * mesh['dt'] * 1e3:.1f} ms 时的边界情况:")

        # 检查左边界的位移
        print(f"\n  左边界 (x=0):")
        print(f"    反射边界 u0 = {result_r['u'][check_time, 0]:+.6f}")
        print(f"    吸收边界 u0 = {result_a['u'][check_time, 0]:+.6f}")

        # 检查右边界的位移
        print(f"\n  右边界 (x={L:.0f}):")
        print(f"    反射边界 uN = {result_r['u'][check_time, -1]:+.6f}")
        print(f"    吸收边界 uN = {result_a['u'][check_time, -1]:+.6f}")

        # 能量分析
        energy_r = np.sum(result_r['u'][check_time, :]**2) * mesh["dx"]
        energy_a = np.sum(result_a['u'][check_time, :]**2) * mesh["dx"]
        energy_initial = np.sum(result_r['u'][0, :]**2) * mesh["dx"]

        print(f"\n  系统能量 (近似):")
        print(f"    初始能量:         {energy_initial:.4f}")
        print(f"    反射边界 (t≈{check_time*mesh['dt']*1e3:.0f}ms): {energy_r:.4f}")
        print(f"    吸收边界 (t≈{check_time*mesh['dt']*1e3:.0f}ms): {energy_a:.4f}")

        if energy_initial > 0:
            print(f"    反射边界能量比:  {energy_r/energy_initial:.2%}")
            print(f"    吸收边界能量比:  {energy_a/energy_initial:.2%}")

    print()


# ============================================================
# 稳定性演示 (CFL 破坏)
# ============================================================
def demonstrate_instability(params, mesh):
    """
    演示 CFL 条件不满足时数值解发散。

    参数:
        params: dict, 参数集
        mesh: dict, 网格信息
    """
    print("=" * 60)
    print("稳定性演示: CFL 条件破坏")
    print("=" * 60)

    # 使用 CFL = 1.5 的不稳定参数
    dt_unstable = 1.5 * mesh["dt_cfl"]
    dx = mesh["dx"]
    c = params["c"]
    sigma_unstable = c * dt_unstable / dx
    nx = mesh["nx"]
    nt = 50
    x = mesh["x"]

    print(f"  Courant 数: σ = {sigma_unstable:.2f} (> 1.0, 不稳定!)")
    print(f"  dt = {dt_unstable*1e3:.3f} ms")
    print()

    u = np.zeros((nt + 1, nx))
    u[0, :] = initial_condition(x, params)
    u[1, 1:-1] = (u[0, 1:-1]
                  + 0.5 * sigma_unstable**2
                  * (u[0, :-2] - 2 * u[0, 1:-1] + u[0, 2:]))
    u[1, 0] = 0
    u[1, -1] = 0

    max_vals = [np.max(np.abs(u[0]))]
    for n in range(1, nt):
        u[n + 1, 1:-1] = (2 * u[n, 1:-1] - u[n - 1, 1:-1]
                          + sigma_unstable**2
                          * (u[n, :-2] - 2 * u[n, 1:-1] + u[n, 2:]))
        u[n + 1, 0] = 0
        u[n + 1, -1] = 0
        max_vals.append(np.max(np.abs(u[n + 1])))

    print("  时间步 | 最大振幅 | 状态")
    print("  " + "-" * 35)
    n_samples = list(range(0, len(max_vals), max(1, len(max_vals) // 6)))
    for n in n_samples[:6]:
        val = max_vals[n] if n < len(max_vals) else 0.0
        status = "💥 发散!" if val > 10 * max_vals[0] else "..."
        print(f"  {n:6d} | {val:8.4f} | {status}")

    print()
    print(f"  结论: σ = {sigma_unstable:.2f} > 1, 数值解在 {nt} 步内发散")
    print(f"  稳定条件: σ = c·Δt/Δx ≤ 1.0")
    print()


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║   1D 波动方程 FDM 求解 — 地震波传播模拟        ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # 1. 设置参数
    params = set_parameters()
    print("物理参数:")
    print(f"  地层长度: L = {params['L']:.0f} m")
    print(f"  波速: c = {params['c']} m/s")
    print(f"  震源位置: x0 = {params['source_x0']:.0f} m")
    print(f"  震源宽度: σ = {params['source_sigma']:.0f} m")
    print(f"  边界类型: {params['boundary_type']}")
    print()

    # 2. 构建网格
    mesh = build_mesh(params)
    print(f"空间网格: nx = {mesh['nx']}, dx = {mesh['dx']:.1f} m")
    print(f"时间步长: dt = {mesh['dt']*1e3:.3f} ms")
    print()

    # 3. 检查 CFL 条件
    check_cfl(mesh)

    # 4. 波动方程求解
    result = solve_wave(params, mesh)
    visualize(result, mesh, params, f"波动传播 (边界: {params['boundary_type']})")

    # 5. 与达朗贝尔解析解对比
    verify_with_analytical(params, mesh)

    # 6. 边界对比
    compare_boundaries(params, mesh)

    # 7. 稳定性演示
    demonstrate_instability(params, mesh)

    print()
    print("✓ 波动传播模拟完成")
