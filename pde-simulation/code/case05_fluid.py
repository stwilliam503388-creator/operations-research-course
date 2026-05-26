"""
case05_fluid.py — 1D Burgers 方程 FDM 求解
============================================

方程:  u_t + u u_x = ν u_xx

物理含义: Burgers 方程是 Navier-Stokes 方程的一维简化版。
  - u u_x: 非线性对流项（迎风格式处理）
  - ν u_xx: 黏性扩散项（中心差分处理）
  - ν 越小 → 雷诺数 Re = UL/ν 越大 → 激波越陡

本例演示:
  (1) 正弦波初始条件随时间的演化
  (2) 不同雷诺数（黏性系数）对解的影响
  (3) 迎风格式与中心差分的配合
  (4) CFL 稳定性条件

作者: pde-course
"""

import numpy as np
import matplotlib.pyplot as plt


def burgers_fdm(nx=200, nt=600, L=2.0, T=1.0, nu=0.01, ic='sin'):
    """
    求解 1D Burgers 方程: u_t + u*u_x = nu*u_xx

    参数:
        nx  : 空间网格点数
        nt  : 时间步数
        L   : 空间域长度 [0, L]
        T   : 总模拟时间
        nu  : 黏性系数 (1/Re)
        ic  : 初始条件类型 ('sin' 或 'shock')

    返回:
        x   : 空间网格 (nx,)
        u   : 最终时刻的解 (nx,)
        U   : 全部时刻的解 (nt+1, nx)
    """
    # --- 网格 ---
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    dt = T / nt

    # --- CFL 检查与自动调整 ---
    cfl_conv = 1.0 * dt / dx
    cfl_diff = nu * dt / dx**2
    print(f"Grid: nx={nx}, nt={nt}")
    print(f"dx = {dx:.5f}, dt = {dt:.5e}")
    print(f"Conv CFL (max|u|=1): {cfl_conv:.4f}  (should be <= 1)")
    print(f"Diff CFL: {cfl_diff:.4f}  (should be <= 0.5)")

    if cfl_diff > 0.5:
        # auto-adjust to satisfy diffusion CFL
        nt_new = int(nt * cfl_diff / 0.45) + 1
        print(f"Diff CFL exceeded, auto-adjusting nt: {nt} -> {nt_new}")
        nt = nt_new
        dt = T / nt
        cfl_diff = nu * dt / dx**2
        print(f"  Adjusted: dt = {dt:.5e}, diff CFL = {cfl_diff:.4f}")
    if cfl_conv > 1.0:
        nt_new = int(nt * cfl_conv / 0.9) + 1
        print(f"Conv CFL exceeded, auto-adjusting nt: {nt} -> {nt_new}")
        nt = nt_new
        dt = T / nt

    # --- 初始条件 ---
    U = np.zeros((nt + 1, nx))
    if ic == 'sin':
        U[0, :] = np.sin(np.pi * x / L)
        print("IC: sine wave u(x,0) = sin(pi x/L)")
    elif ic == 'shock':
        U[0, :] = 1.0 * (x < L / 2) + 0.0 * (x >= L / 2)
        print("IC: shock/step u(x,0) = 1 (x<L/2), 0 (x>=L/2)")
    else:
        raise ValueError(f"Unknown IC type: {ic}")

    # --- 时间推进（迎风 + 中心差分）---
    for n in range(nt):
        u = U[n, :].copy()
        u_new = u.copy()

        # 内部点 (i=1..nx-2)
        for i in range(1, nx - 1):
            # ---- 对流项: u * u_x ----
            # 迎风格式: 根据 u 的符号选择前向/后向差分
            if u[i] >= 0:
                u_x = (u[i] - u[i - 1]) / dx  # 后向差分
            else:
                u_x = (u[i + 1] - u[i]) / dx  # 前向差分

            # ---- 扩散项: nu * u_xx ----
            u_xx = (u[i - 1] - 2 * u[i] + u[i + 1]) / (dx * dx)

            # ---- 时间推进 ----
            u_new[i] = u[i] + dt * (-u[i] * u_x + nu * u_xx)

        # 边界条件: Dirichlet 零值（两端固定为 0）
        u_new[0] = 0.0
        u_new[-1] = 0.0

        U[n + 1, :] = u_new

    return x, U[-1, :], U


def compare_reynolds():
    """
    对比不同黏性系数（不同雷诺数）下的 Burgers 方程解
    """
    nu_values = [0.10, 0.05, 0.02, 0.01]
    L, T = 2.0, 1.0
    nx, nt = 200, 600

    plt.figure(figsize=(12, 8))

    for i, nu in enumerate(nu_values):
        x, u_final, U = burgers_fdm(nx=nx, nt=nt, L=L, T=T, nu=nu, ic='sin')
        Re = 1.0 / nu  # 特征速度 1, 特征长度 L/2? 简单取 Re = 1/nu
        plt.subplot(2, 2, i + 1)
        plt.plot(x, U[0, :], 'k--', label='t=0 (initial)', alpha=0.5)
        plt.plot(x, u_final, 'r-', label=f't={T}s')
        plt.title(f'nu = {nu:.2f} (Re ~ {Re:.0f})')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.2, 1.2)

    plt.suptitle('Burgers Eqn: Comparison of Different Viscosity (nu)', fontsize=14)
    plt.tight_layout()
    plt.savefig('/Users/liuwei/pde-course/code/fig_burgers_comparison.png', dpi=150)
    print("\nComparison figure saved: fig_burgers_comparison.png")
    plt.show()


def time_evolution():
    """
    展示 Burgers 方程解随时间的演化过程
    """
    nx, nt, L, T = 200, 800, 2.0, 1.2
    nu = 0.02

    x, u_final, U = burgers_fdm(nx=nx, nt=nt, L=L, T=T, nu=nu, ic='sin')

    # 选取几个时间层
    snapshots = [0, nt // 4, nt // 2, 3 * nt // 4, nt]
    labels = [f't = {i * T / nt:.2f}s' for i in snapshots]

    plt.figure(figsize=(10, 6))
    for idx, label in zip(snapshots, labels):
        plt.plot(x, U[idx, :], label=label)

    plt.title(f'Burgers Eqn Time Evolution (nu={nu}, Re~{1/nu:.0f})')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/Users/liuwei/pde-course/code/fig_burgers_evolution.png', dpi=150)
    print("Evolution figure saved: fig_burgers_evolution.png")
    plt.show()


def shock_evolution():
    """
    展示间断初始条件（激波）在 Burgers 方程中的演化
    """
    nx, nt, L, T = 300, 1000, 2.0, 0.8
    nu = 0.005

    x, u_final, U = burgers_fdm(nx=nx, nt=nt, L=L, T=T, nu=nu, ic='shock')

    snapshots = [0, nt // 4, nt // 2, 3 * nt // 4, nt]
    labels = [f't = {i * T / nt:.2f}s' for i in snapshots]

    plt.figure(figsize=(10, 6))
    for idx, label in zip(snapshots, labels):
        plt.plot(x, U[idx, :], label=label)

    plt.title(f'Shock Evolution (nu={nu}, Re~{1/nu:.0f})')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/Users/liuwei/pde-course/code/fig_burgers_shock.png', dpi=150)
    print("Shock figure saved: fig_burgers_shock.png")
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("Case 5: 1D Burgers Equation - FDM Solver")
    print("=" * 60)

    print("\n>>> 1. Basic Solver (Sine IC)")
    x, u_final, U = burgers_fdm(nx=200, nt=600, L=2.0, T=1.0, nu=0.02, ic='sin')

    print("\n>>> 2. Reynolds Number Comparison")
    compare_reynolds()

    print("\n>>> 3. Time Evolution")
    time_evolution()

    print("\n>>> 4. Shock Evolution")
    shock_evolution()

    print("\n" + "=" * 60)
    print("All results generated.")
    print("=" * 60)
