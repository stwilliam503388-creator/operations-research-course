"""
案例4：合作博弈与公平分配 —— 沙普利值计算
============================================

场景：三个部门合作完成一个项目
  · A部门（研发）：单独可创造 20 万利润
  · B部门（市场）：单独可创造 30 万利润
  · C部门（运营）：单独可创造 25 万利润
  · A+B：60 万（协同效应）
  · A+C：55 万
  · B+C：70 万（市场和运营配合好）
  · A+B+C：100 万

教学点：
  · 边际贡献的概念
  · 沙普利值（Shapley Value）计算公式
  · 核（Core）的概念与合理性检验
  · 效率、个体理性、联盟理性
"""

import itertools
import math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────
# 1. 特征函数 (Characteristic Function)
# ──────────────────────────────────────────────────────
# 用位掩码表示联盟：A=1, B=2, C=4

PLAYERS = ['A', 'B', 'C']
N = len(PLAYERS)

def coalition_value(mask: int) -> float:
    """给定联盟（位掩码），返回可创造的利润（万元）"""
    values = {
        0b000: 0,      # 空集
        0b001: 20,     # A 单独
        0b010: 30,     # B 单独
        0b100: 25,     # C 单独
        0b011: 60,     # A+B
        0b101: 55,     # A+C
        0b110: 70,     # B+C
        0b111: 100,    # A+B+C
    }
    return values.get(mask, 0)


# ──────────────────────────────────────────────────────
# 2. 沙普利值计算（枚举所有排列）
# ──────────────────────────────────────────────────────

def shapley_values(players: list, v_func) -> dict:
    """
    计算沙普利值。
    φ_i = (1/n!) * Σ_{π ∈ Π} [v(S_π(i) ∪ {i}) - v(S_π(i))]
    其中 S_π(i) 是在排列 π 中排在 i 之前的玩家集合。
    """
    n = len(players)
    contributions = defaultdict(float)

    # 遍历所有排列
    for perm in itertools.permutations(range(n)):
        # 按排列顺序依次加入联盟，计算边际贡献
        coalition_mask = 0
        for idx in perm:
            # 加入前的联盟价值
            prev_val = v_func(coalition_mask)
            # 加入后的联盟价值
            new_mask = coalition_mask | (1 << idx)
            new_val = v_func(new_mask)
            # 边际贡献
            marginal = new_val - prev_val
            contributions[players[idx]] += marginal
            coalition_mask = new_mask

    # 除以排列数得到期望边际贡献
    n_fact = math.factorial(n)
    return {p: contributions[p] / n_fact for p in players}


# ──────────────────────────────────────────────────────
# 3. 核 (Core) 检验
# ──────────────────────────────────────────────────────

def check_core(allocation: dict, players: list, v_func) -> list:
    """
    检查给定分配是否在核中。
    核条件：
      1. 效率（总和 = 大联盟价值）
      2. 个体理性（每人至少得到单独行动的价值）
      3. 联盟理性（任何子联盟的总和 ≥ 该联盟单独的价值）
    返回违反的条件列表（空列表 = 在核中）
    """
    violations = []
    n = len(players)
    total = sum(allocation[p] for p in players)
    grand_val = v_func((1 << n) - 1)

    # 效率
    if abs(total - grand_val) > 1e-9:
        violations.append(f"❌ 效率违反: 分配总和 {total:.2f} ≠ 大联盟价值 {grand_val}")

    # 个体理性
    for i, p in enumerate(players):
        alone = v_func(1 << i)
        if allocation[p] + 1e-9 < alone:
            violations.append(f"❌ 个体理性违反: {p} 得 {allocation[p]:.2f} < 单独 {alone}")

    # 联盟理性（所有非空真子集）
    for mask in range(1, (1 << n) - 1):
        coalition_val = v_func(mask)
        coalition_allocation = 0
        members = []
        for i, p in enumerate(players):
            if mask & (1 << i):
                coalition_allocation += allocation[p]
                members.append(p)
        if coalition_allocation + 1e-9 < coalition_val:
            violations.append(
                f"❌ 联盟理性违反: {{{','.join(members)}}} "
                f"得 {coalition_allocation:.2f} < 联盟价值 {coalition_val}"
            )

    return violations


# ──────────────────────────────────────────────────────
# 4. 运行与输出
# ──────────────────────────────────────────────────────

print("=" * 65)
print("  合作博弈特征函数")
print("=" * 65)
coalition_names = {
    0b001: "{A}", 0b010: "{B}", 0b100: "{C}",
    0b011: "{A,B}", 0b101: "{A,C}", 0b110: "{B,C}", 0b111: "{A,B,C}",
}
for mask, name in coalition_names.items():
    print(f"  v({name:>8}) = {coalition_value(mask):3d} 万元")
print()

# 计算沙普利值
sv = shapley_values(PLAYERS, coalition_value)

print("=" * 65)
print("  沙普利值计算结果")
print("=" * 65)
total_sv = sum(sv.values())
for p in PLAYERS:
    print(f"  φ({p}) = {sv[p]:>6.2f} 万元  ({sv[p]/total_sv*100:.1f}%)")
print(f"  ─────────────────────")
print(f"  合计    {total_sv:.2f} 万元")
print()

# 核检验
print("=" * 65)
print("  核 (Core) 合理性检验")
print("=" * 65)
violations = check_core(sv, PLAYERS, coalition_value)
if not violations:
    print("  ✅ 沙普利值在核中！所有合理性条件满足。")
else:
    for v in violations:
        print(f"  {v}")
print()

# ──────────────────────────────────────────────────────
# 5. 对比：不同分配方案的核检验
# ──────────────────────────────────────────────────────

print("=" * 65)
print("  不同分配方案对比")
print("=" * 65)

# 方案1：平均分配
equal_split = {p: 100 / 3 for p in PLAYERS}
# 方案2：按单独价值比例
alone_vals = [coalition_value(1 << i) for i in range(N)]
total_alone = sum(alone_vals)
proportional = {p: coalition_value(1 << i) / total_alone * 100 for i, p in enumerate(PLAYERS)}
# 方案3：沙普利值
# 方案4：核仁近似（B 和 C 有强议价力 -> B=35, C=35, A=30）
custom = {'A': 30, 'B': 35, 'C': 35}

schemes = [
    ("平均分配", equal_split),
    ("按单独价值比例", proportional),
    ("沙普利值", sv),
    ("自定义 (B=C=35, A=30)", custom),
]

for name, alloc in schemes:
    violations = check_core(alloc, PLAYERS, coalition_value)
    in_core = len(violations) == 0
    total = sum(alloc.values())
    print(f"\n  {name}:")
    print(f"    A={alloc['A']:.1f}, B={alloc['B']:.1f}, C={alloc['C']:.1f}  (合计={total:.1f})")
    print(f"    {'✅ 在核中' if in_core else '❌ 不在核中'}")
    if not in_core:
        for v in violations[:2]:  # 只显示前两条
            print(f"      {v}")

# ──────────────────────────────────────────────────────
# 6. 可视化
# ──────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# 左图：沙普利值分解
players_idx = range(N)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax1.bar(players_idx, [sv[p] for p in PLAYERS], color=colors, width=0.5)
ax1.set_xticks(players_idx)
ax1.set_xticklabels([f'Dept {p}\nphi={sv[p]:.1f}' for p in PLAYERS])
ax1.set_ylabel('Allocation amount')
ax1.set_title('Shapley value allocation')
ax1.set_ylim(0, 55)
for bar, val in zip(bars, [sv[p] for p in PLAYERS]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}', ha='center', fontweight='bold')
ax1.axhline(100/3, color='gray', linestyle='--', alpha=0.5, label='Equal split')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 右图：沙普利值 vs 平均 vs 单独比例
x = np.arange(N)
width = 0.25
ax2.bar(x - width, [equal_split[p] for p in PLAYERS], width, label='Equal split', color='gray', alpha=0.6)
ax2.bar(x, [sv[p] for p in PLAYERS], width, label='Shapley value', color='#FF6B6B')
ax2.bar(x + width, [proportional[p] for p in PLAYERS], width, label='Standalone ratio', color='#4ECDC4')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Dept {p}' for p in PLAYERS])
ax2.set_ylabel('Allocation amount')
ax2.set_title('Allocation scheme comparison')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(Path(__file__).with_name('case06_cooperation.png'), dpi=150)
print("\n📊 可视化已保存: code/case06_cooperation.png")
plt.close()

# ──────────────────────────────────────────────────────
# 7. 验证标准自动检查
# ──────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  ✅ 验证标准自动检查")
print("=" * 65)

# 标准1：效率（总和 = 总收益）
check1 = abs(total_sv - 100) < 1e-9
print(f"  [效率] 总和={total_sv:.4f}, 期望=100 → {'✅ 通过' if check1 else '❌ 未通过'}")

# 标准2：个体理性（每人 ≥ 单独价值）
check2_all = True
for i, p in enumerate(PLAYERS):
    alone_val = coalition_value(1 << i)
    ok = sv[p] >= alone_val - 1e-9
    if not ok:
        check2_all = False
    print(f"  [个体理性] φ({p})={sv[p]:.2f} ≥ v({{{p}}})={alone_val:.2f} → {'✅ 通过' if ok else '❌ 未通过'}")

# 标准3：联盟理性示范
print(f"  [联盟理性] 沙普利值在核中 → {'✅ 通过' if in_core else '❌ 未通过'}")
