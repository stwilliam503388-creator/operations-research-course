"""
case06_em.py — 2D Laplace 方程 Gauss-Seidel 迭代求解
=====================================================

方程:  ∂²φ/∂x² + ∂²φ/∂y² = 0    (拉普拉斯方程)

物理含义: 平行板电容器内部的静电场分布。
  - φ 为电位（电势）
  - 上板: φ = +V0, 下板: φ = -V0
  - 左右边界: φ = 0 (自由边界)
  - 电场 E = -∇φ

数值方法:
  - 五点差分格式离散拉普拉斯算子
  - Gauss-Seidel 迭代（逐点更新）
  - 相对残差收敛检测

与解析解对比:
  平行板电容器内部电位沿 y 方向线性变化:
  φ_analytic(y) = V0 * (2*y/H - 1)  (中心区域)

作者: pde-course
"""

import numpy as np
from pathlib import Path
import os


def maybe_show(plt):
    """Only open an interactive window when explicitly requested."""
    if os.environ.get("SHOW_PLOTS") == "1":
        plt.show()
    else:
        plt.close()


def laplace_gauss_seidel(nx=40, ny=40, Lx=1.0, Ly=1.0, V0=1.0,
                         max_iter=10000, tol=1e-6, verbose=True):
    """
    求解 2D Laplace 方程 ∇²φ = 0 在矩形域上的 Dirichlet 问题。

    边界条件:
      - 上边 (y=Ly): φ = +V0
      - 下边 (y=0): φ = -V0
      - 左边 (x=0): φ = 0
      - 右边 (x=Lx): φ = 0

    参数:
        nx, ny   : x, y 方向网格点数（含边界）
        Lx, Ly   : 区域尺寸 [m]
        V0       : 电极板电位 [V]
        max_iter : 最大迭代步数
        tol      : 收敛容差（相对残差）
        verbose  : 是否输出迭代过程

    返回:
        x, y     : 网格坐标 (nx,), (ny,)
        phi      : 电位分布 (ny, nx)
        n_iter   : 实际迭代步数
        history  : 残差历史 (n_iter,)
    """
    # --- 网格 ---
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # --- 初始化 ---
    phi = np.zeros((ny, nx))

    # 设置 Dirichlet 边界条件
    phi[-1, :] = +V0   # 上边界 (y=Ly): +V0
    phi[0, :]  = -V0   # 下边界 (y=0):  -V0
    # 左右边界保持为 0 (已在初始化中设定)

    if verbose:
        print(f"Grid: {nx}x{ny}, dx={dx:.4f}, dy={dy:.4f}")
        print(f"BC: top=+{V0}V, bottom=-{V0}V, left=right=0V")
        print(f"Max iter: {max_iter}, tol: {tol:.1e}")

    # --- Gauss-Seidel 迭代 ---
    history = []

    for n_iter in range(1, max_iter + 1):
        phi_old = phi.copy()
        max_residual = 0.0

        # 逐点更新（跳过边界点）
        for i in range(1, ny - 1):       # y 方向
            for j in range(1, nx - 1):   # x 方向
                # 五点差分格式
                # φ(i,j) = [φ(i-1,j) + φ(i+1,j) + φ(i,j-1) + φ(i,j+1)] / 4
                phi[i, j] = 0.25 * (phi[i - 1, j] + phi[i + 1, j]
                                    + phi[i, j - 1] + phi[i, j + 1])

        # 计算最大残差（相对值）
        diff = np.abs(phi - phi_old)
        current_max = np.max(phi) - np.min(phi)
        if current_max > 0:
            residual = np.max(diff) / current_max
        else:
            residual = np.max(diff)

        history.append(residual)

        if verbose and (n_iter <= 20 or n_iter % 500 == 0):
            print(f"  Iter {n_iter:5d}: rel residual = {residual:.2e}")

        if residual < tol:
            if verbose:
                print(f"\nConverged! Iter: {n_iter}, final residual: {residual:.2e}")
            return x, y, phi, n_iter, np.array(history)

    if verbose:
        print(f"\nMax iter {max_iter} reached, final residual: {residual:.2e}")
    return x, y, phi, max_iter, np.array(history)


def compute_electric_field(x, y, phi):
    """
    从电位分布计算电场: E = -∇φ

    使用中心差分计算梯度分量:
      Ex = -∂φ/∂x,  Ey = -∂φ/∂y

    返回:
        X, Y   : 网格坐标矩阵 (ny, nx)
        Ex, Ey : 电场分量 (ny, nx)
        E_mag  : 电场强度幅值
    """
    ny, nx = phi.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    Ex = np.zeros((ny, nx))
    Ey = np.zeros((ny, nx))

    # 内部点: 中心差分
    Ex[1:-1, 1:-1] = -(phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * dx)
    Ey[1:-1, 1:-1] = -(phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * dy)

    # 边界: 单侧差分
    Ex[:, 0]   = -(phi[:, 1] - phi[:, 0]) / dx
    Ex[:, -1]  = -(phi[:, -1] - phi[:, -2]) / dx
    Ey[0, :]   = -(phi[1, :] - phi[0, :]) / dy
    Ey[-1, :]  = -(phi[-1, :] - phi[-2, :]) / dy

    E_mag = np.sqrt(Ex**2 + Ey**2)

    X, Y = np.meshgrid(x, y)
    return X, Y, Ex, Ey, E_mag


def analytic_solution(x, y, V0=1.0, Ly=1.0):
    """
    平行板电容器内部电位解析解（忽略边缘效应）:
      φ(y) = V0 * (2*y/Ly - 1)

    注意: 严格来讲，平行板电容器在中心区域的电位沿 y 方向线性变化。
          我们的数值解在边界条件上做了简化（左右边界为 0），
          因此与纯一维解析解只在中心区域接近。
    """
    return V0 * (2 * y / Ly - 1)


def compare_with_analytical():
    """将数值解与解析解对比"""
    nx, ny = 40, 40
    Lx, Ly = 1.0, 1.0
    V0 = 1.0

    x, y, phi, n_iter, history = laplace_gauss_seidel(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, V0=V0, max_iter=5000, tol=1e-6
    )

    # 解析解（沿 y 方向，取 x 中心）
    j_center = nx // 2
    phi_center = phi[:, j_center]
    phi_analytic = analytic_solution(x[j_center], y, V0=V0, Ly=Ly)

    print("\n" + "=" * 60)
    print("Numerical vs Analytical Solution (x = Lx/2 centerline)")
    print("=" * 60)
    print(f"{'y':>8s}  {'phi_num':>10s}  {'phi_ana':>10s}  {'error':>10s}")
    print("-" * 42)
    errors = []
    for i in range(0, ny, 4):  # every 4th point
        err = abs(phi_center[i] - phi_analytic[i])
        errors.append(err)
        print(f"{y[i]:>8.3f}  {phi_center[i]:>10.6f}  "
              f"{phi_analytic[i]:>10.6f}  {err:>10.2e}")
    print("-" * 42)
    print(f"Max error: {max(errors):.2e}")
    print(f"Mean error: {np.mean(errors):.2e}")

    return x, y, phi, history


def text_visualization(phi, x, y, title="电位分布"):
    """
    文本方式的电位分布可视化（适用于终端显示）
    使用字符灰度近似显示
    """
    ny, nx = phi.shape
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    # normalize to 0~1 for display
    p_min, p_max = phi.min(), phi.max()
    p_norm = (phi - p_min) / (p_max - p_min + 1e-16)

    # char mapping
    chars = ' .:-=+*#%@'

    print(f"  Top (y={y[-1]:.2f}): phi = +{phi[-1, 0]:.1f}")
    for i in range(ny - 1, -1, -1):  # top to bottom
        line = ''
        for j in range(nx):
            idx = min(int(p_norm[i, j] * (len(chars) - 1)), len(chars) - 1)
            line += chars[idx] * 2
        print(f"  |{line}|  y={y[i]:.3f}")
    print(f"  Bottom (y={y[0]:.2f}): phi = {phi[0, 0]:.1f}")
    print(f"  Left phi=0                         Right phi=0")
    print(f"  phi range: [{p_min:.3f}, {p_max:.3f}]")
    print(f"{'=' * 60}")


def plot_results(x, y, phi, history, X, Y, Ex, Ey, E_mag):
    """使用 matplotlib 绘制结果图"""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 电位分布（云图）
        ax1 = axes[0, 0]
        contour = ax1.contourf(X, Y, phi, levels=20, cmap='RdBu_r')
        plt.colorbar(contour, ax=ax1, label='φ [V]')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.set_title('Potential phi(x,y)')
        ax1.set_aspect('equal')

        # 2. E-field vector + magnitude
        ax2 = axes[0, 1]
        skip = 3
        c = ax2.contourf(X, Y, E_mag, levels=20, cmap='viridis')
        plt.colorbar(c, ax=ax2, label='|E| [V/m]')
        ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                   Ex[::skip, ::skip], Ey[::skip, ::skip],
                   color='white', alpha=0.7, scale=1.5, width=0.003)
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('y [m]')
        ax2.set_title('E-field |E| and direction')
        ax2.set_aspect('equal')

        # 3. Centerline comparison with analytical
        ax3 = axes[1, 0]
        j_center = nx // 2
        phi_analytic = analytic_solution(X[0, j_center], y, V0=1.0, Ly=1.0)
        ax3.plot(y, phi[:, j_center], 'b-o', label='Numerical', markersize=3)
        ax3.plot(y, phi_analytic, 'r--', label='Analytical', linewidth=2)
        ax3.set_xlabel('y [m]')
        ax3.set_ylabel('phi [V]')
        ax3.set_title(f'Centerline (x={x[j_center]:.2f}m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Convergence
        ax4 = axes[1, 1]
        ax4.semilogy(history, 'b-')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Rel residual')
        ax4.set_title('Gauss-Seidel Convergence')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(__file__).with_name('fig_laplace_results.png'), dpi=150)
        print("Result figure saved: fig_laplace_results.png")
        maybe_show(plt)

    except ImportError:
        print("matplotlib 未安装，跳过图形化绘图。")


def convergence_study():
    """Mesh refinement convergence study"""
    print("\n" + "=" * 60)
    print("Mesh Refinement Convergence Study")
    print("=" * 60)

    nx_list = [10, 20, 40, 80, 160]
    Lx, Ly, V0 = 1.0, 1.0, 1.0

    results = []
    for nx in nx_list:
        ny = nx
        x, y, phi, n_iter, history = laplace_gauss_seidel(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, V0=V0,
            max_iter=20000, tol=1e-6, verbose=False
        )

        # compare with analytical on centerline
        j_center = nx // 2
        phi_analytic = analytic_solution(x[j_center], y, V0=V0, Ly=Ly)
        error = np.max(np.abs(phi[:, j_center] - phi_analytic))

        results.append((nx, n_iter, error))
        print(f"  N={nx:3d}: iter {n_iter:5d}, max error = {error:.2e}")

    print("\n  Error scaling with mesh refinement:")
    for i in range(1, len(results)):
        ratio = results[i-1][2] / results[i][2] if results[i][2] > 0 else float('inf')
        print(f"    N={results[i-1][0]:3d}->{results[i][0]:3d}: "
              f"error ratio ~ {ratio:.1f}  (2nd order expects: 4.0)")


if __name__ == '__main__':
    print("=" * 60)
    print("Case 6: 2D Laplace Eqn - Gauss-Seidel Solver")
    print("=" * 60)

    # --- 1. Basic solve ---
    print("\n>>> 1. Gauss-Seidel Iteration")
    nx, ny = 40, 40
    x, y, phi, n_iter, history = laplace_gauss_seidel(
        nx=nx, ny=ny, Lx=1.0, Ly=1.0, V0=1.0,
        max_iter=10000, tol=1e-6
    )

    # --- 2. Text visualization ---
    print("\n>>> 2. Potential Distribution (Text Viz)")
    text_visualization(phi, x, y, "Parallel Plate Capacitor Potential")

    # --- 3. Electric field ---
    print("\n>>> 3. Electric Field Computation")
    X, Y, Ex, Ey, E_mag = compute_electric_field(x, y, phi)
    print(f"  Max |E|: {E_mag.max():.4f} V/m")
    print(f"  Min |E|: {E_mag.min():.4f} V/m")

    # --- 4. Compare with analytical ---
    print("\n>>> 4. Comparison with Analytical Solution")
    compare_with_analytical()

    # --- 5. Plot ---
    print("\n>>> 5. Result Visualization")
    plot_results(x, y, phi, history, X, Y, Ex, Ey, E_mag)

    # --- 6. Convergence study ---
    convergence_study()

    print("\n" + "=" * 60)
    print("All results generated.")
    print("=" * 60)
