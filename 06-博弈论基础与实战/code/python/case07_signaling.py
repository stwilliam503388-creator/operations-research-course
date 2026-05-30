"""
案例5：信号传递与市场 —— Spence 劳动力市场信号模型
====================================================

场景：
  · 工人有高能力 (θ=H) 或低能力 (θ=L)，比例分别为 p 和 1-p
  · 工人可以选择受教育水平 e（连续变量，e ≥ 0）
  · 教育成本：高能力 c_H * e，低能力 c_L * e （c_H < c_L）
  · 企业竞争雇佣工人，付工资等于对工人能力的期望

教学点：
  · 分离均衡（Separating Equilibrium）：不同类型选择不同教育水平
  · 混同均衡（Pooling Equilibrium）：不同类型选择相同教育水平
  · 信号成本条件：c_H < c_L 是分离可能的必要条件
  · 直观标准（Intuitive Criterion）剔除不合理均衡

关键结论：
  分离均衡存在条件：c_L > c_H（信号成本与能力负相关）
  具体地，存在 e* ∈ [Δθ/c_L, Δθ/c_H] 使得：
    高能力选 e*, 低能力选 0, 企业据此区分
"""


# 教学注释：从参与者、策略和收益矩阵出发理解交互决策结构。
# 计算结果用于验证均衡、分配规则或机制设计是否符合预期。



import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ──────────────────────────────────────────────────────
# 1. 模型参数（可调）
# ──────────────────────────────────────────────────────

class SpenceModel:
    """Spence 劳动力市场信号模型"""

    def __init__(self,
                 theta_L=50,    # 低能力产出
                 theta_H=100,   # 高能力产出
                 c_L=8,         # 低能力教育成本系数
                 c_H=4,         # 高能力教育成本系数
                 p=0.5):        # 高能力比例
        self.theta_L = theta_L
        self.theta_H = theta_H
        self.c_L = c_L
        self.c_H = c_H
        self.p = p

    # ── 收益函数 ──────────────────────────────────

    def payoff_H(self, e, w):
        """高能力工人选择教育 e、拿工资 w 的收益"""
        return w - self.c_H * e

    def payoff_L(self, e, w):
        """低能力工人选择教育 e、拿工资 w 的收益"""
        return w - self.c_L * e

    def firm_payoff(self, w, theta):
        """企业雇佣一个工人的利润"""
        return theta - w

    # ── 条件 ───────────────────────────────────────

    @property
    def delta_theta(self):
        """能力差距"""
        return self.theta_H - self.theta_L

    @property
    def separating_feasible(self):
        """分离均衡存在的基本条件：c_L > c_H"""
        return self.c_L > self.c_H

    def separating_education_range(self):
        """
        分离均衡中教育水平 e* 的取值范围：
          Δθ / c_L ≤ e* ≤ Δθ / c_H

        解释：
          · e* ≥ Δθ/c_L 确保低能力不会模仿（模仿成本太高）
          · e* ≤ Δθ/c_H 确保高能力愿意选择 e*（而不是模仿低能力）
        """
        if not self.separating_feasible:
            return None
        return (self.delta_theta / self.c_L, self.delta_theta / self.c_H)

    def pooling_wage(self):
        """混同均衡中的工资 = 期望产出"""
        return self.p * self.theta_H + (1 - self.p) * self.theta_L

    # ── 均衡求解 ──────────────────────────────────

    def separating_equilibrium(self, e_star=None):
        """
        求解分离均衡。

        如果 e_star 未指定，取中间值 e* = Δθ / (c_L + c_H) * 2
        （满足分离条件的最小教育水平+边际）
        """
        if not self.separating_feasible:
            return None

        e_range = self.separating_education_range()
        if e_star is None:
            # 取中间值
            e_star = (e_range[0] + e_range[1]) / 2

        # 均衡策略
        # 高能力选 e_star，低能力选 0
        # 企业信念：看到 e ≥ e* 认为是高能力，否则低能力

        # 高能力收益
        w_H = self.theta_H
        payoff_H = self.payoff_H(e_star, w_H)

        # 低能力收益（选择 0）
        w_L = self.theta_L
        payoff_L = self.payoff_L(0, w_L)

        # 检验激励相容
        # 低能力不模仿高能力
        ic_L = self.payoff_L(e_star, w_H) <= payoff_L
        # 高能力不模仿低能力
        ic_H = self.payoff_H(0, w_L) <= payoff_H

        return {
            'type': '分离均衡',
            'e_H': e_star,
            'e_L': 0,
            'w_H': w_H,
            'w_L': w_L,
            'payoff_H': payoff_H,
            'payoff_L': payoff_L,
            'ic_L': ic_L,
            'ic_H': ic_H,
            'e_min': e_range[0],
            'e_max': e_range[1],
        }

    def pooling_equilibrium(self, e_pool=0):
        """
        求解混同均衡。

        所有工人选相同教育水平 e_pool，
        企业付期望工资 w = p*θ_H + (1-p)*θ_L
        """
        w_pool = self.pooling_wage()

        payoff_H = self.payoff_H(e_pool, w_pool)
        payoff_L = self.payoff_L(e_pool, w_pool)

        # 检验无利可图的偏离
        # 如果工人选 e=0（如果 e_pool > 0）
        # 企业信念：看到 e ≠ e_pool 认为是低能力
        w_dev = self.theta_L
        dev_payoff_H = self.payoff_H(0, w_dev) if e_pool > 0 else None
        dev_payoff_L = self.payoff_L(0, w_dev) if e_pool > 0 else None

        no_dev_H = dev_payoff_H is None or dev_payoff_H <= payoff_H
        no_dev_L = dev_payoff_L is None or dev_payoff_L <= payoff_L

        return {
            'type': '混同均衡',
            'e_pool': e_pool,
            'w_pool': w_pool,
            'payoff_H': payoff_H,
            'payoff_L': payoff_L,
            'no_deviation_H': no_dev_H,
            'no_deviation_L': no_dev_L,
        }


# ──────────────────────────────────────────────────────
# 2. 运行与输出
# ──────────────────────────────────────────────────────

model = SpenceModel(theta_L=50, theta_H=100, c_L=8, c_H=4, p=0.5)

print("=" * 65)
print("  Spence 劳动力市场信号模型")
print("=" * 65)
print(f"  参数:")
print(f"    低能力产出 θ_L = {model.theta_L}")
print(f"    高能力产出 θ_H = {model.theta_H}")
print(f"    低能力教育成本 c_L = {model.c_L}")
print(f"    高能力教育成本 c_H = {model.c_H}")
print(f"    高能力比例 p = {model.p}")
print(f"    能力差距 Δθ = {model.delta_theta}")
print()

# ── 基本条件检查 ──────────────────────────────────

print("=" * 65)
print("  基本条件检查")
print("=" * 65)
print(f"  信号成本条件 (c_L > c_H): {model.separating_feasible}")
if model.separating_feasible:
    e_min, e_max = model.separating_education_range()
    print(f"  分离均衡 e* 取值范围: [{e_min:.2f}, {e_max:.2f}]")
print()

# ── 分离均衡 ──────────────────────────────────────

print("=" * 65)
print("  分离均衡（Separating Equilibrium）")
print("=" * 65)
sep = model.separating_equilibrium()
if sep:
    print(f"  高能力工人: e* = {sep['e_H']:.2f}, 工资 w = {sep['w_H']:.1f}, 收益 = {sep['payoff_H']:.1f}")
    print(f"  低能力工人: e* = {sep['e_L']:.2f}, 工资 w = {sep['w_L']:.1f}, 收益 = {sep['payoff_L']:.1f}")
    print(f"  e* 允许范围: [{sep['e_min']:.2f}, {sep['e_max']:.2f}]")
    print(f"  低能力不模仿 (IC_L): {'✅' if sep['ic_L'] else '❌'}")
    print(f"  高能力不模仿 (IC_H): {'✅' if sep['ic_H'] else '❌'}")
print()

# ── 混同均衡 ──────────────────────────────────────

print("=" * 65)
print("  混同均衡（Pooling Equilibrium）")
print("=" * 65)
pool = model.pooling_equilibrium(e_pool=0)
print(f"  混同教育水平 e_pool = {pool['e_pool']}")
print(f"  混同工资 w_pool = {pool['w_pool']:.1f}")
print(f"  高能力收益 = {pool['payoff_H']:.1f}")
print(f"  低能力收益 = {pool['payoff_L']:.1f}")
print(f"  高能力无偏差: {'✅' if pool['no_deviation_H'] else '❌'}")
print(f"  低能力无偏差: {'✅' if pool['no_deviation_L'] else '❌'}")
print()

# 尝试带教育水平的混同均衡
print("—" * 40)
# 看不同 e_pool 的混同均衡
for e in [0, 2, 4, 6]:
    p_eq = model.pooling_equilibrium(e_pool=e)
    ok = "✅" if p_eq['no_deviation_H'] and p_eq['no_deviation_L'] else "❌"
    print(f"  e_pool={e}: w={p_eq['w_pool']:.1f}, "
          f"U_H={p_eq['payoff_H']:.1f}, U_L={p_eq['payoff_L']:.1f}, 可行={ok}")

print()

# ── 灵敏度分析：c_L 和 c_H 的影响 ──────────────────

print("=" * 65)
print("  灵敏度分析：c_H 对分离均衡的影响")
print("=" * 65)
for c_H in [2, 4, 6, 7.5]:
    m = SpenceModel(c_H=c_H)
    if m.separating_feasible:
        e_min, e_max = m.separating_education_range()
        print(f"  c_H={c_H:.1f}: e*范围=[{e_min:.2f}, {e_max:.2f}], 范围宽度={e_max-e_min:.2f}")
    else:
        print(f"  c_H={c_H:.1f}: ❌ 分离不可行（c_L=8 ≤ c_H={c_H})")

print()

# ── 分离不可能的情况 ──────────────────────────────

print("=" * 65)
print("  分离不可能的情况 (c_L = c_H)")
print("=" * 65)
model_no = SpenceModel(c_L=5, c_H=5)
print(f"  c_L = c_H = 5 → 分离可行? {model_no.separating_feasible}")
print(f"  解释：如果教育成本与能力无关，高能力无法通过教育传递信号")
print()

# ──────────────────────────────────────────────────────
# 3. 可视化
# ──────────────────────────────────────────────────────

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# ── 图1：无差异曲线与分离均衡 ──
e_range = np.linspace(0, 20, 200)

# 高能力无差异曲线（达到分离均衡收益所需工资）
w_for_H = model.c_H * e_range + sep['payoff_H']
# 低能力无差异曲线
w_for_L = model.c_L * e_range + sep['payoff_L']

ax1.plot(e_range, w_for_H, 'b-', linewidth=2, label='High-type indifference')
ax1.plot(e_range, w_for_L, 'r-', linewidth=2, label='Low-type indifference')
# 标记均衡点
ax1.plot(sep['e_H'], sep['w_H'], 'bo', markersize=10, label=f"High-type equilibrium (e={sep['e_H']:.1f})")
ax1.plot(sep['e_L'], sep['w_L'], 'ro', markersize=10, label=f"Low-type equilibrium (e={sep['e_L']:.1f})")
# 显示教育成本差异
ax1.annotate(f'Δθ/c_L={sep["e_min"]:.1f}', xy=(sep['e_min'], 75), fontsize=9, color='gray')
ax1.annotate(f'Δθ/c_H={sep["e_max"]:.1f}', xy=(sep['e_max'], 75), fontsize=9, color='gray')
ax1.axvline(sep['e_min'], color='gray', linestyle=':', alpha=0.5)
ax1.axvline(sep['e_max'], color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Education level e')
ax1.set_ylabel('Wage w')
ax1.set_title('Separating equilibrium: indifference curves')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# ── 图2：分离均衡 e* 取值范围 vs c_H ──
c_H_vals = np.linspace(0.5, 7.5, 30)
e_mins = []
e_maxs = []
feasible = []
for ch in c_H_vals:
    m = SpenceModel(c_H=ch)
    feas = m.separating_feasible
    feasible.append(feas)
    if feas:
        emin, emax = m.separating_education_range()
        e_mins.append(emin)
        e_maxs.append(emax)
    else:
        e_mins.append(np.nan)
        e_maxs.append(np.nan)

ax2.plot(c_H_vals, e_mins, 'g--', label='Lower bound (dtheta/c_L)', linewidth=2)
ax2.plot(c_H_vals, e_maxs, 'b-', label='Upper bound (dtheta/c_H)', linewidth=2)
ax2.fill_between(c_H_vals, e_mins, e_maxs, alpha=0.2, color='blue', label='Feasible e* range')
ax2.axvline(8, color='red', linestyle=':', alpha=0.7, label='c_L = 8')
ax2.set_xlabel('High-type education cost c_H')
ax2.set_ylabel('Education level e*')
ax2.set_title('Feasible range of separating e*')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── 图3：能力差距 Δθ 对分离的影响 ──
theta_H_vals = np.linspace(60, 150, 20)
e_mins2 = []
e_maxs2 = []
for th in theta_H_vals:
    m = SpenceModel(theta_H=th)
    if m.separating_feasible:
        emin, emax = m.separating_education_range()
        e_mins2.append(emin)
        e_maxs2.append(emax)
    else:
        e_mins2.append(np.nan)
        e_maxs2.append(np.nan)

ax3.plot(theta_H_vals - 50, e_mins2, 'g--', label='Lower bound', linewidth=2)
ax3.plot(theta_H_vals - 50, e_maxs2, 'b-', label='Upper bound', linewidth=2)
ax3.fill_between(theta_H_vals - 50, e_mins2, e_maxs2, alpha=0.2, color='blue')
ax3.set_xlabel('Ability gap dtheta')
ax3.set_ylabel('Education level e*')
ax3.set_title('Ability gap impact on separating equilibrium')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── 图4：混同均衡下各类型收益 ──
e_pool_vals = np.linspace(0, 10, 50)
u_H_pool = [model.payoff_H(e, model.pooling_wage()) for e in e_pool_vals]
u_L_pool = [model.payoff_L(e, model.pooling_wage()) for e in e_pool_vals]

ax4.plot(e_pool_vals, u_H_pool, 'b-', linewidth=2, label='High-type payoff')
ax4.plot(e_pool_vals, u_L_pool, 'r-', linewidth=2, label='Low-type payoff')
ax4.axhline(model.payoff_H(0, model.theta_L), color='blue', linestyle='--', alpha=0.5, label='High-type deviation payoff')
ax4.axhline(model.payoff_L(0, model.theta_L), color='red', linestyle='--', alpha=0.5, label='Low-type deviation payoff')
ax4.set_xlabel('Pooling education level e_pool')
ax4.set_ylabel('Worker payoff')
ax4.set_title('Pooling equilibrium: type payoffs')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).with_name('case07_signaling.png'), dpi=150)
print("📊 可视化已保存: code/case07_signaling.png")
plt.close()

# ──────────────────────────────────────────────────────
# 4. 验证标准自动检查
# ──────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  ✅ 验证标准自动检查")
print("=" * 65)

# 标准1：分离均衡存在条件
check1 = model.separating_feasible
print(f"  [分离可行] c_L={model.c_L} > c_H={model.c_H} → {'✅ 通过' if check1 else '❌ 未通过'}")

# 标准2：IC 条件满足
check2 = sep['ic_L'] and sep['ic_H']
print(f"  [IC条件] 激励相容 → {'✅ 通过' if check2 else '❌ 未通过'}")
if sep['ic_L']:
    dev_L = model.payoff_L(sep['e_H'], sep['w_H'])
    print(f"    低能力不模仿: U_L(e*, w_H)={dev_L:.2f} ≤ U_L(0, θ_L)={sep['payoff_L']:.2f}")
if sep['ic_H']:
    dev_H = model.payoff_H(0, sep['w_L'])
    print(f"    高能力不模仿: U_H(0, θ_L)={dev_H:.2f} ≤ U_H(e*, θ_H)={sep['payoff_H']:.2f}")

# 标准3：混同均衡条件
check3 = pool['no_deviation_H'] and pool['no_deviation_L']
print(f"  [混同均衡] 无偏差动机 → {'✅ 通过' if check3 else '❌ 未通过'}")

# 标准4：分离不可行条件
model_infeas = SpenceModel(c_L=5, c_H=5)
check4 = not model_infeas.separating_feasible
print(f"  [分离不可行] c_L=c_H=5 → {'✅ 通过' if check4 else '❌ 未通过'}")
