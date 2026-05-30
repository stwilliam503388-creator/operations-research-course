"""
案例3：谈判与讨价还价博弈 —— 逆向归纳求解
============================================

场景：两人分 100 元，轮流出价，贴现因子 δ

教学点：
  · 逆向归纳求解子博弈完美均衡
  · 先手优势（first-mover advantage）
  · 耐心（贴现因子 δ）对分配结果的影响
  · 轮数 → ∞ 时趋向平均分配

关键结论（Rubinstein, 1982）：
  无限轮博弈的唯一子博弈完美均衡：
  A 拿 (1 - δ) / (1 - δ²) × 100,  B 拿 δ(1 - δ) / (1 - δ²) × 100
 当 δ → 1 时，分配趋向 (50, 50)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ──────────────────────────────────────────────────────
# 1. 有限轮逆向归纳（数值求解）
# ──────────────────────────────────────────────────────

def finite_round_bargaining(delta: float, total_rounds: int, total=100.0):
    """
    有限轮轮流出价博弈，逆向归纳求解。

    规则：
      · 奇数轮 → A 出价，B 决定接受或拒绝
      · 偶数轮 → B 出价，A 决定接受或拒绝
      · 若最后一轮被拒绝，双方得 0

    返回: (a_share, b_share)  当前（第一轮）的均衡分配
    """
    # 从最后一轮开始逆向递推
    # offer_share: 当前出价人给自己的份额（在剩余蛋糕上的比例）
    proposer_share = total  # 最后一轮：出价人拿全部，对方接受任何非负

    for r in range(total_rounds - 1, 0, -1):  # r = total_rounds-1 ... 1
        # 本轮出价人给对方的 offer 需让对方在「接受」和「等到下轮」之间无差异
        # 对方下轮的期望收益 = delta * proposer_share（等一轮有贴现）
        responder_gets = delta * proposer_share
        # 出价人自己拿剩下的
        proposer_share = total - responder_gets

    # 第一轮出价人始终是 A（奇数轮 = A 出价）
    # 逆向归纳后 proposer_share 即第一轮出价人的份额
    a_share = proposer_share
    b_share = total - proposer_share

    return a_share, b_share


# ──────────────────────────────────────────────────────
# 2. 无限轮解析解（Rubinstein 公式）
# ──────────────────────────────────────────────────────

def rubinstein_solution(delta: float, total=100.0):
    """
    无限轮 Rubinstein 谈判博弈的解析解。

    A 先出价时的子博弈完美均衡：
      A_share = total * (1 - δ) / (1 - δ²)
      B_share = total - A_share
    """
    if delta >= 1:
        return total / 2, total / 2
    a_share = total * (1 - delta) / (1 - delta ** 2)
    b_share = total - a_share
    return a_share, b_share


# ──────────────────────────────────────────────────────
# 3. 运行与输出
# ──────────────────────────────────────────────────────

print("=" * 65)
print("  例1：一轮博弈（最后通牒）")
print("=" * 65)
a, b = finite_round_bargaining(delta=0.9, total_rounds=1)
print(f"  δ=0.9, 1轮 → A拿 {a:.2f}元, B拿 {b:.2f}元")
print(f"  ✅ 验证：一轮博弈 A 拿走几乎全部 (99.9+?)")
print()

print("=" * 65)
print("  例2：两轮博弈")
print("=" * 65)
for d in [0.3, 0.5, 0.8, 0.9, 0.99]:
    a, b = finite_round_bargaining(delta=d, total_rounds=2)
    print(f"  δ={d:.2f} → A拿 {a:.2f}元, B拿 {b:.2f}元")
print(f"  ✅ 验证：两轮博弈 A 拿 1-δ（蛋糕归一化后）")
print()

print("=" * 65)
print("  例3：轮数增加 → 趋向各半")
print("=" * 65)
for rounds in [2, 5, 10, 50, 100]:
    a, b = finite_round_bargaining(delta=0.8, total_rounds=rounds)
    print(f"  δ=0.8, {rounds:3d}轮 → A拿 {a:6.2f}元, B拿 {b:6.2f}元")
print()

print("=" * 65)
print("  例4：无限轮解析解（Rubinstein 1982）")
print("=" * 65)
for d in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    a, b = rubinstein_solution(delta=d)
    print(f"  δ={d:.2f} → A拿 {a:7.2f}元 ({a/100*100:.1f}%), B拿 {b:7.2f}元 ({b/100*100:.1f}%)")
print()

# ──────────────────────────────────────────────────────
# 4. 可视化：δ 和轮数的影响
# ──────────────────────────────────────────────────────

# 4a. δ 对分配的影响（无限轮）
deltas = np.linspace(0.01, 0.99, 50)
a_shares_inf = [rubinstein_solution(d)[0] for d in deltas]

# 4b. 轮数对分配的影响（固定 δ）
rounds_list = np.arange(1, 21)
a_by_rounds_09 = [finite_round_bargaining(0.9, r)[0] for r in rounds_list]
a_by_rounds_05 = [finite_round_bargaining(0.5, r)[0] for r in rounds_list]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# 左图：δ 的影响
ax1.plot(deltas, a_shares_inf, 'b-', linewidth=2, label='A share')
ax1.plot(deltas, [100 - s for s in a_shares_inf], 'r--', linewidth=2, label='B share')
ax1.axhline(50, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Discount factor delta')
ax1.set_ylabel('Allocation amount')
ax1.set_title('Infinite-round bargaining: discount factor impact')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：轮数的影响
ax2.plot(rounds_list, a_by_rounds_09, 'b-o', label='δ = 0.9')
ax2.plot(rounds_list, a_by_rounds_05, 'r-s', label='δ = 0.5')
ax2.axhline(50, color='gray', linestyle=':', alpha=0.5, label='Equal split')
ax2.set_xlabel('Rounds')
ax2.set_ylabel('A share')
ax2.set_title('Finite-round bargaining: rounds impact')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).with_name('case05_bargaining.png'), dpi=150)
print("📊 可视化已保存: code/case05_bargaining.png")
plt.close()

# ──────────────────────────────────────────────────────
# 5. 验证标准自动检查
# ──────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  ✅ 验证标准自动检查")
print("=" * 65)

# 标准1：一轮博弈 A 拿几乎全部
a1, b1 = finite_round_bargaining(delta=0.9, total_rounds=1)
check1 = a1 >= 99.9
print(f"  [一轮A拿99.9+] A={a1:.4f} → {'✅ 通过' if check1 else '❌ 未通过'}")

# 标准2：两轮博弈 A 拿 1-δ（蛋糕=1 归一化）
for d in [0.3, 0.5, 0.8]:
    a2, _ = finite_round_bargaining(delta=d, total_rounds=2, total=1.0)
    expected = 1 - d
    check2 = abs(a2 - expected) < 1e-10
    print(f"  [两轮A=1-δ] δ={d}: A={a2:.6f}, 期望={expected:.6f} → {'✅ 通过' if check2 else '❌ 未通过'}")

# 标准3：轮数增加 → 无限轮解析解（δ=0.5 时极限 66.67）
a_100, _ = finite_round_bargaining(delta=0.5, total_rounds=100)
inf_solution = rubinstein_solution(delta=0.5)[0]
check3 = abs(a_100 - inf_solution) < 0.01
print(f"  [多轮→无限轮] δ=0.5, 100轮: A={a_100:.4f}, 无限轮解={inf_solution:.4f} → {'✅ 通过' if check3 else '❌ 未通过'}")

# 标准4：无限轮 δ→1 时趋向 50
a_inf_99, _ = rubinstein_solution(delta=0.999)
check4 = abs(a_inf_99 - 50) < 0.1
print(f"  [δ→1→50] δ=0.999: A={a_inf_99:.4f} → {'✅ 通过' if check4 else '❌ 未通过'}")
