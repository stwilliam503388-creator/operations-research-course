"""
🏆 毕业项目：芯片封装多物理场模拟（骨架代码）
==============================================
问题：芯片工作时产生热量 → 温度升高 → 热膨胀 → 结构应力
组合 PDE：
  (1) 热传导方程（稳态或瞬态）：∇·(k∇T) + Q = ρcₚ ∂T/∂t
  (2) 热弹性方程（结构应力）：∇·σ + f = 0, σ = C:(ε - αΔT·I)

作者：pde-course
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# 1. 问题定义与参数
# ============================================================
class ChipPackage:
    """芯片封装几何与材料参数"""

    def __init__(self):
        # 几何尺寸 [m]
        self.Lx = 10e-3       # 封装 x 方向长度 10mm
        self.Ly = 10e-3       # 封装 y 方向长度 10mm
        self.Lz = 1e-3        # 封装 z 方向厚度 1mm（平面应力近似）

        # 材料参数：硅芯片
        self.k_si = 130.0     # 导热系数 [W/(m·K)]
        self.rho_si = 2330.0  # 密度 [kg/m³]
        self.cp_si = 700.0    # 比热容 [J/(kg·K)]
        self.E_si = 170e9     # 弹性模量 [Pa]
        self.nu_si = 0.28     # 泊松比
        self.alpha_si = 2.6e-6  # 热膨胀系数 [1/K]

        # 材料参数：封装基板 (FR4)
        self.k_fr4 = 0.3
        self.rho_fr4 = 1900.0
        self.cp_fr4 = 600.0
        self.E_fr4 = 22e9
        self.nu_fr4 = 0.15
        self.alpha_fr4 = 14e-6

        # 热源（芯片热点）
        self.Q_hotspot = 1e9   # 热点热源密度 [W/m³]
        self.T_ambient = 25.0  # 环境温度 [°C]
        self.h_conv = 50.0     # 对流换热系数 [W/(m²·K)]

        # 求解控制
        self.Nx = 40           # x 方向网格数
        self.Ny = 40           # y 方向网格数
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny


# ============================================================
# 2. 热传导求解器 (FDM + 隐式时间推进)
# ============================================================
class HeatSolver2D:
    """
    二维热传导方程求解器
    ∂T/∂t = α (∂²T/∂x² + ∂²T/∂y²) + Q/(ρcₚ)
    方法：交替方向隐式 (ADI) — 将 2D 问题分解为两个 1D 问题
    """

    def __init__(self, chip: ChipPackage):
        self.chip = chip
        self.Nx = chip.Nx
        self.Ny = chip.Ny
        self.dx = chip.dx
        self.dy = chip.dy

        # 温度场 [°C]
        self.T = np.ones((self.Ny, self.Nx)) * chip.T_ambient

        # 热源场
        self.Q = np.zeros((self.Ny, self.Nx))
        self._setup_heat_source()

        # 材料热扩散系数 α = k/(ρcₚ)
        self.alpha_mat = chip.k_si / (chip.rho_si * chip.cp_si)

    def _setup_heat_source(self):
        """在芯片中心区域设置热点热源"""
        chip = self.chip
        cx, cy = chip.Nx // 2, chip.Ny // 2
        hotspot_radius = 5  # 热点半径（网格数）
        for j in range(chip.Ny):
            for i in range(chip.Nx):
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                if dist <= hotspot_radius:
                    self.Q[j, i] = chip.Q_hotspot

    def solve_steady_state(self, max_iter=10000, tol=1e-8):
        """
        稳态热传导求解 (Gauss-Seidel 迭代)
        ∇·(k∇T) + Q = 0
        """
        chip = self.chip
        T = self.T.copy()
        Q = self.Q
        k = chip.k_si
        dx2 = self.dx**2
        dy2 = self.dy**2

        # 四边对流边界 (Robin BC): -k ∂T/∂n = h(T - T_ambient)
        T_amb = chip.T_ambient
        h = chip.h_conv

        print("求解稳态温度场...")
        for iteration in range(max_iter):
            T_old = T.copy()

            # 内部节点更新 (FDM five-point stencil)
            for j in range(1, self.Ny - 1):
                for i in range(1, self.Nx - 1):
                    T[j, i] = (
                        (T[j, i - 1] + T[j, i + 1]) / dx2 +
                        (T[j - 1, i] + T[j + 1, i]) / dy2 +
                        Q[j, i] / k
                    ) / (2 / dx2 + 2 / dy2)

            # 边界条件 (简化：Dirichlet 近似)
            T[0, :] = T_amb      # 上边界
            T[-1, :] = T_amb     # 下边界
            T[:, 0] = T_amb      # 左边界
            T[:, -1] = T_amb     # 右边界

            # 收敛判断
            error = np.max(np.abs(T - T_old))
            if iteration % 500 == 0:
                print(f"  迭代 {iteration:5d}, 最大变化 = {error:.2e}")

            if error < tol:
                print(f"  收敛于迭代 {iteration}, 最大变化 = {error:.2e}")
                break

        self.T = T
        return T

    def solve_transient(self, t_end=10.0, dt=0.01):
        """
        瞬态热传导 (ADI 方法)
        半步: x 方向隐式, y 方向显式
        半步: y 方向隐式, x 方向显式
        """
        chip = self.chip
        alpha = self.alpha_mat
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        T = self.T.copy()
        T_amb = chip.T_ambient

        rx = alpha * dt / (2 * dx**2)
        ry = alpha * dt / (2 * dy**2)

        n_steps = int(t_end / dt)
        print(f"瞬态求解: t_end={t_end}s, dt={dt}s, n_steps={n_steps}")

        # 预组装三对角矩阵 (x 方向)
        Ax = np.zeros((Nx, Nx))
        for i in range(1, Nx - 1):
            Ax[i, i - 1] = -rx
            Ax[i, i] = 1 + 2 * rx
            Ax[i, i + 1] = -rx
        Ax[0, 0] = 1.0
        Ax[-1, -1] = 1.0

        # 预组装三对角矩阵 (y 方向)
        Ay = np.zeros((Ny, Ny))
        for j in range(1, Ny - 1):
            Ay[j, j - 1] = -ry
            Ay[j, j] = 1 + 2 * ry
            Ay[j, j + 1] = -ry
        Ay[0, 0] = 1.0
        Ay[-1, -1] = 1.0

        def solve_tridiag(a, b, c, d):
            """Thomas 算法求解三对角系统"""
            n = len(d)
            cp = np.zeros(n)
            dp = np.zeros(n)
            x = np.zeros(n)

            cp[0] = c[0] / b[0]
            dp[0] = d[0] / b[0]

            for i in range(1, n):
                denom = b[i] - a[i] * cp[i - 1]
                cp[i] = c[i] / denom
                dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

            x[-1] = dp[-1]
            for i in range(n - 2, -1, -1):
                x[i] = dp[i] - cp[i] * x[i + 1]

            return x

        for step in range(n_steps):
            T_old = T.copy()

            # 半步1: x 方向隐式 (逐行求解)
            for j in range(1, Ny - 1):
                d = T[j, :] + ry * (T[j - 1, :] - 2 * T[j, :] + T[j + 1, :])
                d[0] = T_amb
                d[-1] = T_amb
                # 提取三对角系数
                a = np.array([0] + [-rx] * (Nx - 2) + [0])
                b = np.array([1] + [1 + 2 * rx] * (Nx - 2) + [1])
                c = np.array([0] + [-rx] * (Nx - 2) + [0])
                T[j, :] = solve_tridiag(a, b, c, d)

            # 半步2: y 方向隐式 (逐列求解)
            for i in range(1, Nx - 1):
                d = T[:, i] + rx * (T[:, i - 1] - 2 * T[:, i] + T[:, i + 1])
                d[0] = T_amb
                d[-1] = T_amb
                a = np.array([0] + [-ry] * (Ny - 2) + [0])
                b = np.array([1] + [1 + 2 * ry] * (Ny - 2) + [1])
                c = np.array([0] + [-ry] * (Ny - 2) + [0])
                T[:, i] = solve_tridiag(a, b, c, d)

            # 边界条件
            T[0, :] = T_amb
            T[-1, :] = T_amb
            T[:, 0] = T_amb
            T[:, -1] = T_amb

            if step % 100 == 0:
                T_max = np.max(T)
                print(f"  t = {(step + 1) * dt:.2f}s, T_max = {T_max:.2f}°C")

        self.T = T
        return T

    def plot_temperature(self):
        """绘制温度场云图"""
        chip = self.chip
        x = np.linspace(0, chip.Lx * 1e3, chip.Nx)  # mm
        y = np.linspace(0, chip.Ly * 1e3, chip.Ny)  # mm
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots(figsize=(8, 6))
        cf = ax.contourf(X, Y, self.T, levels=30, cmap='hot')
        plt.colorbar(cf, ax=ax, label='Temperature [°C]')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'芯片封装稳态温度场 (T_max = {np.max(self.T):.1f}°C)')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig('/Users/liuwei/pde-course/code/capstone_temperature.png', dpi=150)
        print("温度场图形已保存: capstone_temperature.png")


# ============================================================
# 3. 热应力求解器 (平面应力近似)
# ============================================================
class StressSolver2D:
    """
    二维热弹性方程求解器
    控制方程：
      ∂σ_xx/∂x + ∂σ_xy/∂y = 0
      ∂σ_xy/∂x + ∂σ_yy/∂y = 0
    
    本构关系（平面应力）：
      σ_xx = E/(1-ν²)[ε_xx + ν·ε_yy - α·ΔT·(1+ν)]
      σ_yy = E/(1-ν²)[ε_yy + ν·ε_xx - α·ΔT·(1+ν)]
      σ_xy = E/(1+ν)·ε_xy
    
    应变-位移关系：
      ε_xx = ∂u/∂x,  ε_yy = ∂v/∂y,  ε_xy = ½(∂u/∂y + ∂v/∂x)
    """

    def __init__(self, chip: ChipPackage, T_field: np.ndarray):
        self.chip = chip
        self.T = T_field                    # 温度场 [°C]
        self.T_ref = chip.T_ambient         # 参考温度 [°C]
        self.delta_T = T_field - self.T_ref # 温差 [°C]

        # 位移场 (初始化)
        self.u = np.zeros_like(T_field)     # x 方向位移
        self.v = np.zeros_like(T_field)     # y 方向位移

        # 应力场 (初始化)
        self.sxx = np.zeros_like(T_field)
        self.syy = np.zeros_like(T_field)
        self.sxy = np.zeros_like(T_field)

    def compute_stresses(self):
        """
        简化: 给定温度场，用热弹性本构估算热应力
        （完整实现需要 FEM; 此处为概念演示）
        
        在一维近似下：σ ≈ -E·α·ΔT/(1-ν)
        """
        chip = self.chip
        E = chip.E_si
        nu = chip.nu_si
        alpha = chip.alpha_si

        # 约束热应力估算 (完全约束的情况)
        self.sxx = -E * alpha * self.delta_T / (1 - nu)
        self.syy = -E * alpha * self.delta_T / (1 - nu)
        self.sxy = np.zeros_like(self.delta_T)

        # von Mises 等效应力
        self.von_mises = np.sqrt(
            self.sxx**2 + self.syy**2 - self.sxx * self.syy + 3 * self.sxy**2
        )

        return self.von_mises

    def plot_stress(self):
        """绘制热应力云图"""
        chip = self.chip
        x = np.linspace(0, chip.Lx * 1e3, chip.Nx)
        y = np.linspace(0, chip.Ly * 1e3, chip.Ny)
        X, Y = np.meshgrid(x, y)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        titles = [r'σ_xx [MPa]', r'σ_yy [MPa]',
                  r'σ_xy [MPa]', r'von Mises [MPa]']
        fields = [self.sxx, self.syy, self.sxy, self.von_mises]

        for ax, title, field in zip(axes.flat, titles, fields):
            cf = ax.contourf(X, Y, field / 1e6, levels=30, cmap='coolwarm')
            plt.colorbar(cf, ax=ax)
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            ax.set_title(title)
            ax.set_aspect('equal')

        plt.suptitle('芯片封装热应力分布', fontsize=14)
        plt.tight_layout()
        plt.savefig('/Users/liuwei/pde-course/code/capstone_stress.png', dpi=150)
        print("热应力图形已保存: capstone_stress.png")


# ============================================================
# 4. 主程序：多物理场耦合流程
# ============================================================
def main():
    """完整的多物理场耦合模拟骨架"""

    print("=" * 60)
    print("芯片封装多物理场模拟")
    print("=" * 60)

    # Step 1: 初始化芯片封装模型
    chip = ChipPackage()
    print(f"\n[Step 1] 芯片封装参数:")
    print(f"  尺寸: {chip.Lx*1e3:.1f} × {chip.Ly*1e3:.1f} mm")
    print(f"  网格: {chip.Nx} × {chip.Ny}")
    print(f"  热源密度: {chip.Q_hotspot/1e6:.0f} MW/m³")

    # Step 2: 求解温度场
    print(f"\n[Step 2] 求解温度场...")
    heat_solver = HeatSolver2D(chip)
    T_steady = heat_solver.solve_steady_state()
    print(f"  最高温度: {np.max(T_steady):.1f}°C")
    print(f"  最低温度: {np.min(T_steady):.1f}°C")

    # Step 3: 转瞬态（演示使用）
    print(f"\n[Step 3] 瞬态热分析 (可选)...")
    # 注释掉瞬态计算以节省时间，使用稳态结果
    # T_transient = heat_solver.solve_transient(t_end=5.0, dt=0.05)

    # Step 4: 计算热应力（单向耦合）
    print(f"\n[Step 4] 计算热应力（单向耦合: T → σ）...")
    stress_solver = StressSolver2D(chip, T_steady)
    vm = stress_solver.compute_stresses()
    print(f"  最大 von Mises 应力: {np.max(vm)/1e6:.1f} MPa")
    print(f"  硅抗拉强度 (~150 MPa): ", end="")
    if np.max(vm) < 150e6:
        print("✅ 安全")
    else:
        print("❌ 可能失效")

    # Step 5: 可视化
    print(f"\n[Step 5] 可视化...")
    heat_solver.plot_temperature()
    stress_solver.plot_stress()

    # Step 6: 评估与报告
    print(f"\n[Step 6] 评估报告")
    print(f"{'─' * 50}")
    print(f"  物理场耦合: 热传导 → 热应力 (单向)")
    print(f"  网格分辨率: {chip.Nx} × {chip.Ny}")
    print(f"  最高温度:   {np.max(T_steady):.1f} °C")
    print(f"  最大热应力: {np.max(vm)/1e6:.1f} MPa")
    print(f"  安全系数:   {150e6/np.max(vm):.2f} (vs 硅抗拉强度 150MPa)")
    print(f"  求解方法:   FDM (热传导) + 本构方程 (热应力)")
    print(f"\n  扩展方向：")
    print(f"    - 双向耦合（应力影响热传导：裂纹导致的导热退化）")
    print(f"    - 瞬态非线性（材料参数随温度变化）")
    print(f"    - FEM 网格（处理复杂封装几何）")
    print(f"    - 并行加速（GPU: CuPy / JAX）")
    print(f"{'─' * 50}")

    # plt.show()  # 取消注释以显示图形


if __name__ == "__main__":
    main()
