"""
案例3: 1D 热传导方程 FDM 求解 — 芯片散热模拟
=================================================

物理方程: ∂u/∂t = α ∂²u/∂x²
数值方法: FTCS 显式格式 / 隐式格式 (Crank-Nicolson)
边界条件: 左边界 Dirichlet (固定温度), 右边界 Neumann (绝热)

参考: 03-case-heat.md
"""
# 教学注释：把离散格式、时间步长和稳定性条件联系到原始物理模型。
# 输出的温度、波形或场量用于检查数值解是否符合物理直觉。


import numpy as np


# ============================================================
# 参数设置
# ============================================================
def set_parameters():
    """
    设置物理和数值参数。

    场景: 手机芯片满负荷运行
      - 芯片长度 L = 10 mm
      - 热扩散率 α = 1.0 × 10⁻⁴ m²/s (硅的典型值)
      - 初始温度 u(x,0) = 25 °C (环境温度)
      - 左边界温度 u(0,t) = 85 °C (芯片热点)
      - 右边界绝热 ∂u/∂x(L,t) = 0

    返回: dict, 包含所有参数
    """
    params = {
        # 物理参数
        "L": 0.01,              # 芯片长度 [m] = 10 mm
        "alpha": 1.0e-4,        # 热扩散率 [m²/s]
        "T_left": 85.0,         # 左边界温度 [°C] (Dirichlet)
        "T_initial": 25.0,      # 初始温度 [°C]
        # 数值参数
        "nx": 50,               # 空间网格数
        "nt": 2000,             # 时间步数 (显式)
        "nt_implicit": 200,     # 时间步数 (隐式, 可大步长)
    }
    return params


# ============================================================
# 网格构建
# ============================================================
def build_mesh(params):
    """
    构建一维网格。

    参数:
        params: dict, 来自 set_parameters()

    返回:
        dict: 包含 x (坐标数组), dx (网格间距), dt (时间步长),
              dt_cfl (CFL 限制下的最大稳定时间步长)
    """
    L = params["L"]
    nx = params["nx"]
    alpha = params["alpha"]

    dx = L / (nx - 1)          # 空间步长
    x = np.linspace(0, L, nx)  # 空间网格点坐标

    # CFL 稳定性条件: α·dt/dx² ≤ 0.5
    dt_cfl = 0.5 * dx**2 / alpha
    dt = dt_cfl * 0.9          # 取 90% 以保证安全

    return {
        "x": x,
        "dx": dx,
        "dt": dt,
        "dt_cfl": dt_cfl,
        "nx": nx,
    }


# ============================================================
# 初值条件
# ============================================================
def initial_condition(x, params):
    """
    设置初始温度分布 u(x, 0)。

    芯片初始温度均匀，等于环境温度。

    参数:
        x: ndarray, 空间坐标数组
        params: dict, 参数集

    返回:
        ndarray: 初始温度场
    """
    return np.full_like(x, params["T_initial"])


# ============================================================
# 显式 FTCS 求解
# ============================================================
def solve_explicit(params, mesh):
    """
    FTCS 显式格式求解 1D 热传导方程。

    u^{n+1}_i = u^n_i + r · (u^n_{i-1} - 2u^n_i + u^n_{i+1})
    其中 r = α·Δt/Δx²

    边界条件:
      - 左边界 (x=0): Dirichlet, u = T_left
      - 右边界 (x=L): Neumann, ∂u/∂x = 0 (绝热)

    参数:
        params: dict, 物理参数
        mesh: dict, 网格信息

    返回:
        dict: {"u": 完整时间演化 [nt+1, nx], "u_final": 最后时刻温度场}
    """
    alpha = params["alpha"]
    T_left = params["T_left"]
    nx = mesh["nx"]
    nt = params["nt"]
    dx = mesh["dx"]
    dt = mesh["dt"]
    x = mesh["x"]

    r = alpha * dt / dx**2  # CFL 数 (扩散情况下)
    print(f"[显式] r = α·Δt/Δx² = {r:.4f}  (稳定条件: r ≤ 0.5)")

    # 存储所有时刻的温度场
    u = np.zeros((nt + 1, nx))
    u[0, :] = initial_condition(x, params)

    for n in range(nt):
        # 内部点: FTCS 更新
        u[n + 1, 1:-1] = (u[n, 1:-1]
                          + r * (u[n, :-2] - 2 * u[n, 1:-1] + u[n, 2:]))

        # 左边界: Dirichlet (固定温度)
        u[n + 1, 0] = T_left

        # 右边界: Neumann (绝热, ∂u/∂x=0 → u_{nx-1} = u_{nx-2})
        u[n + 1, -1] = u[n + 1, -2]

    return {"u": u, "u_final": u[-1, :]}


# ============================================================
# 隐式 Crank-Nicolson 求解
# ============================================================
def solve_implicit(params, mesh):
    """
    Crank-Nicolson 隐式格式求解 1D 热传导方程。

    公式:
      -r·u^{n+1}_{i-1} + (2+2r)·u^{n+1}_i - r·u^{n+1}_{i+1}
        = r·u^n_{i-1} + (2-2r)·u^n_i + r·u^n_{i+1}

    其中 r = α·Δt/Δx²

    Crank-Nicolson 格式是无条件稳定的 (对任意 r 值)。

    参数:
        params: dict, 物理参数
        mesh: dict, 网格信息

    返回:
        dict: {"u": 完整时间演化 [nt+1, nx], "u_final": 最后时刻温度场}
    """
    alpha = params["alpha"]
    T_left = params["T_left"]
    nx = mesh["nx"]
    nt = params["nt_implicit"]
    dx = mesh["dx"]
    x = mesh["x"]

    # 隐式格式可以用更大的时间步长
    # 使用 10 倍于显式格式的最大稳定步长
    dt = mesh["dt_cfl"] * 10.0
    r = alpha * dt / dx**2

    print(f"[隐式 C-N] r = α·Δt/Δx² = {r:.2f}  (无条件稳定)")

    # 构造系数矩阵 A (三对角)
    # A·u^{n+1} = B·u^n
    # 对角线: 2+2r
    # 次对角线: -r
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nx))

    # 内部点
    for i in range(1, nx - 1):
        A[i, i - 1] = -r
        A[i, i] = 2 + 2 * r
        A[i, i + 1] = -r

        B[i, i - 1] = r
        B[i, i] = 2 - 2 * r
        B[i, i + 1] = r

    # 左边界: Dirichlet
    A[0, 0] = 1
    B[0, 0] = 1

    # 右边界: Neumann (∂u/∂x=0 → u[-1] - u[-2] = 0)
    # 使用一阶前向差分近似: u_{nx-1} - u_{nx-2} = 0
    A[-1, -1] = 1
    A[-1, -2] = -1
    B[-1, -1] = 1
    B[-1, -2] = -1

    # 时间推进
    u = np.zeros((nt + 1, nx))
    u[0, :] = initial_condition(x, params)

    for n in range(nt):
        rhs = B @ u[n, :]
        # 边界条件修正: 左边界固定温度
        rhs[0] = T_left
        u[n + 1, :] = np.linalg.solve(A, rhs)

    return {"u": u, "u_final": u[-1, :]}


# ============================================================
# 解析解 (稳态)
# ============================================================
def analytical_steady_state(x, params):
    """
    稳态解析解。

    对 1D 热传导方程 ∂u/∂t = α ∂²u/∂x² 的稳态 (∂u/∂t=0):
      d²u/dx² = 0 → u(x) = Ax + B

    边界条件:
      u(0) = T_left
      ∂u/∂x(L) = 0 (绝热)

    解: u(x) = T_left  (因为绝热边界下, 整根棒温度均匀化到 T_left)

    参数:
        x: ndarray, 坐标
        params: dict, 参数集

    返回:
        ndarray: 稳态温度分布
    """
    return np.full_like(x, params["T_left"])


def analytical_transient(x, t, params):
    """
    瞬态解析解 (仅用于 Dirichlet-Dirichlet 边界作为参考)。

    当左右边界都为 Dirichlet 时:
      u(0,t)=T0, u(L,t)=T1, u(x,0)=Ti

    解: 分离变量法 + 傅里叶级数
      u(x,t) = u_ss(x) + Σ A_k sin(kπx/L) exp(-αk²π²t/L²)

    参数:
        x: ndarray, 坐标
        t: float, 时间
        params: dict, 参数集

    返回:
        ndarray: 解析温度分布
    """
    L = params["L"]
    alpha = params["alpha"]
    T_left = params["T_left"]
    T_initial = params["T_initial"]
    # 假设右边界固定为 T_initial (Dirichlet)
    T_right = T_initial

    # 稳态解: 线性分布
    u_ss = T_left + (T_right - T_left) * x / L

    # 傅里叶级数
    u = u_ss.copy()
    n_terms = 200
    for k in range(1, n_terms + 1):
        Ak = (2 / L) * (
            (T_initial - T_left)
            * (1 - (-1)**k) / (k * np.pi)
            + (T_left - T_right) * (-1)**k / (k * np.pi)
        )
        u += Ak * np.sin(k * np.pi * x / L) * np.exp(
            -alpha * (k * np.pi / L)**2 * t
        )

    return u


# ============================================================
# 与解析解对比验证
# ============================================================
def verify_with_analytical(params, mesh):
    """
    将数值解与解析解对比，验证代码正确性。

    对 Dirichlet-Dirichlet 边界 (左右都固定温度) 进行比较。

    参数:
        params: dict, 参数集 (会修改边界条件为双 Dirichlet)
        mesh: dict, 网格信息
    """
    print("=" * 60)
    print("解析验证: 与傅里叶级数解析解对比")
    print("=" * 60)

    # 使用 Dirichlet-Dirichlet 边界进行验证
    T_left = params["T_left"]
    T_initial = params["T_initial"]
    # 假设右边界固定为 T_initial
    T_right = T_initial

    nx = mesh["nx"]
    dx = mesh["dx"]
    x = mesh["x"]
    dt = mesh["dt"]
    alpha = params["alpha"]
    r = alpha * dt / dx**2

    # 用显式 FTCS 计算到 t=0.5s
    nt_verify = int(0.5 / dt)
    u_num = np.zeros((nt_verify + 1, nx))
    u_num[0, :] = initial_condition(x, params)

    for n in range(nt_verify):
        u_num[n + 1, 1:-1] = (u_num[n, 1:-1]
                              + r * (u_num[n, :-2] - 2 * u_num[n, 1:-1]
                                     + u_num[n, 2:]))
        u_num[n + 1, 0] = T_left
        u_num[n + 1, -1] = T_right  # Dirichlet-Dirichlet

    t_final = nt_verify * dt
    u_analytical = analytical_transient(x, t_final, params)

    # 计算误差
    error = np.max(np.abs(u_num[-1, :] - u_analytical))
    print(f"  时间: t = {t_final:.3f}s")
    print(f"  最大绝对误差: {error:.6f} °C")
    print(f"  相对误差: {error / (T_left - T_initial) * 100:.2f}%")
    print()

    # 在几个点抽样对比
    sample_indices = [0, nx // 4, nx // 2, 3 * nx // 4, nx - 1]
    print(f"  {'位置(mm)':>8} {'数值解(°C)':>12} {'解析解(°C)':>12} {'误差':>10}")
    print("  " + "-" * 44)
    for i in sample_indices:
        xi = x[i] * 1000  # 转 mm
        print(f"  {xi:8.2f} {u_num[-1, i]:12.4f} {u_analytical[i]:12.4f} "
              f"{abs(u_num[-1, i] - u_analytical[i]):10.4f}")


# ============================================================
# 可视化 (文本输出)
# ============================================================
def visualize(result, mesh, params, title="温度分布"):
    """
    以文本形式输出温度分布。

    参数:
        result: dict, 求解结果
        mesh: dict, 网格信息
        params: dict, 参数集
        title: str, 输出标题
    """
    x = mesh["x"]
    u_final = result["u_final"]

    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  网格点数: {mesh['nx']}, "
          f"dx = {mesh['dx']*1000:.3f} mm, "
          f"dt = {mesh['dt']*1e6:.1f} μs")
    print()

    # 文本条形图
    print("  位置(mm) | 温度(°C) | 分布")
    print("  " + "-" * 50)
    for i in range(0, len(x), max(1, len(x) // 10)):
        xi = x[i] * 1000  # 转 mm
        bar_len = int((u_final[i] - params["T_initial"]) / 1.5)
        bar = "█" * max(0, min(bar_len, 40))
        print(f"  {xi:8.2f}  | {u_final[i]:8.2f} | {bar}")

    # 关键数据
    print()
    print(f"  最高温度: {np.max(u_final):.2f} °C")
    print(f"  最低温度: {np.min(u_final):.2f} °C")
    print(f"  平均温度: {np.mean(u_final):.2f} °C")
    print()


# ============================================================
# 性能基准测试
# ============================================================
def benchmark():
    """
    比较显式格式和隐式格式的性能。

    测试不同网格规模下的求解时间和精度。
    """
    print("=" * 60)
    print("性能基准测试: 显式 vs 隐式")
    print("=" * 60)

    for nx in [20, 50, 100]:
        params = set_parameters()
        params["nx"] = nx
        mesh = build_mesh(params)

        # 显式
        import time
        t0 = time.time()
        result_ex = solve_explicit(params, mesh)
        t_ex = time.time() - t0

        # 隐式 (Crank-Nicolson)
        t0 = time.time()
        result_im = solve_implicit(params, mesh)
        t_im = time.time() - t0

        # 稳态对比
        u_ss = analytical_steady_state(mesh["x"], params)
        err_ex = np.max(np.abs(result_ex["u_final"] - u_ss))
        err_im = np.max(np.abs(result_im["u_final"] - u_ss))

        print(f"\n  网格: nx = {nx}")
        print(f"    显式: {t_ex:.3f}s, 误差={err_ex:.4f}, "
              f"步数={params['nt']}")
        print(f"    隐式: {t_im:.3f}s, 误差={err_im:.4f}, "
              f"步数={params['nt_implicit']}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║     1D 热传导方程 FDM 求解 — 芯片散热模拟       ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # 1. 设置参数
    params = set_parameters()
    print("物理参数:")
    print(f"  芯片长度: {params['L']*1000:.1f} mm")
    print(f"  热扩散率: {params['alpha']:.2e} m²/s")
    print(f"  左边界温度: {params['T_left']} °C (Dirichlet)")
    print(f"  右边界: 绝热 (Neumann, ∂u/∂x=0)")
    print(f"  初始温度: {params['T_initial']} °C")
    print()

    # 2. 构建网格
    mesh = build_mesh(params)
    print(f"空间网格: nx = {mesh['nx']}, dx = {mesh['dx']*1000:.3f} mm")
    print(f"时间步长: dt = {mesh['dt']*1e6:.1f} μs")
    print(f"CFL 限制: dt ≤ {mesh['dt_cfl']*1e6:.1f} μs "
          f"(α·dt/dx² ≤ 0.5)")
    print()

    # 3. 显式求解
    result_explicit = solve_explicit(params, mesh)
    visualize(result_explicit, mesh, params, "显式 FTCS 格式结果")

    # 4. 隐式求解
    result_implicit = solve_implicit(params, mesh)
    visualize(result_implicit, mesh, params, "隐式 Crank-Nicolson 格式结果")

    # 5. 解析验证
    verify_with_analytical(params, mesh)

    # 6. 基准测试
    benchmark()

    print()
    print("✓ 热传导模拟完成")
