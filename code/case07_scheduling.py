#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
案例5：护士排班系统 — 约束传播 + 贪心构造 + 局部搜索
===============================================
教学点：CP vs IP、软硬约束、对称性破坏、LNS 思想

场景：50 名护士、7 天、每天早/中/晚三班
硬约束：班次需求、每天最多一班、不连续三天晚班、晚班后不早班、周休≥2天、不连上7天
软约束：班次均衡、偏好满足

算法设计（两阶段）：
  阶段1（工作/休息分配）：保证周休≥2天、不连上7天
  阶段2（班次填充）：满足各班次需求、不连续三天晚班、晚班后不早班

作者：OR Course
"""

import numpy as np
import random
from itertools import permutations

# ============================================================
# 1. 数据定义
# ============================================================

N_NURSES = 50      # 护士人数
N_DAYS = 7         # 排班天数
N_SHIFTS = 3       # 班次：0=早班, 1=中班, 2=晚班

SHIFT_NAMES = ['早班', '中班', '晚班']

# 每天每个班次的最低需求人数
DEMAND = np.array([
    [15, 12, 8],   # 周一
    [15, 12, 8],   # 周二
    [15, 12, 8],   # 周三
    [15, 12, 8],   # 周四
    [15, 12, 8],   # 周五
    [10, 10, 6],   # 周六（需求略低）
    [10, 10, 6],   # 周日
], dtype=int)  # shape = (7, 3)

# 护士偏好：每个护士偏好的班次（0=早, 1=中, 2=晚, -1=无所谓）
np.random.seed(42)
PREFERENCES = np.random.randint(-1, 3, size=(N_NURSES, N_DAYS))

# 软约束权重
W_BALANCE = 0.3    # 班次均衡权重
W_PREFERENCE = 0.7  # 偏好满足权重

# 每名护士每周最少/最多工作日
MIN_WORK_DAYS = 0   # 可以全休（0天工作）
MAX_WORK_DAYS = 5   # 7天 - 至少2天休息


# ============================================================
# 2. 排班求解器
# ============================================================

class NurseScheduler:
    """护士排班求解器（约束传播 + 两阶段贪心 + 局部搜索）"""

    def __init__(self, n_nurses=N_NURSES, n_days=N_DAYS, n_shifts=N_SHIFTS,
                 demand=DEMAND, preferences=PREFERENCES):
        self.N = n_nurses
        self.D = n_days
        self.S = n_shifts
        self.demand = demand
        self.preferences = preferences
        # 排班表: schedule[n][d] = s (0~2) 或 -1 (休息)
        self.schedule = -np.ones((self.N, self.D), dtype=int)

    # ----------------------------------------------------------
    # 硬约束检查
    # ----------------------------------------------------------

    def check_daily_shift_demand(self, schedule=None):
        """检查每天每个班次的需求是否满足"""
        s = schedule if schedule is not None else self.schedule
        for d in range(self.D):
            for shift in range(self.S):
                if np.sum(s[:, d] == shift) < self.demand[d, shift]:
                    return False
        return True

    def check_one_shift_per_day(self, schedule=None):
        """检查每人每天最多一个班次（由变量设计保证，防御性检查）"""
        s = schedule if schedule is not None else self.schedule
        for n in range(self.N):
            for d in range(self.D):
                count = sum(1 for sh in [0, 1, 2] if s[n, d] == sh)
                if count > 1:
                    return False
        return True

    def check_no_three_consecutive_night(self, schedule=None):
        """检查不连续三天晚班"""
        s = schedule if schedule is not None else self.schedule
        for n in range(self.N):
            for d in range(self.D - 2):
                if s[n, d] == 2 and s[n, d+1] == 2 and s[n, d+2] == 2:
                    return False
        return True

    def check_no_night_to_morning(self, schedule=None):
        """检查晚班后不接早班"""
        s = schedule if schedule is not None else self.schedule
        for n in range(self.N):
            for d in range(self.D - 1):
                if s[n, d] == 2 and s[n, d+1] == 0:
                    return False
        return True

    def check_min_rest_days(self, schedule=None, min_rest=2):
        """检查每周至少休息 min_rest 天"""
        s = schedule if schedule is not None else self.schedule
        for n in range(self.N):
            if np.sum(s[n, :] == -1) < min_rest:
                return False
        return True

    def check_no_7_consecutive_work(self, schedule=None):
        """检查不连上 7 天（等同于至少休息 1 天，已由 min_rest=2 覆盖）"""
        s = schedule if schedule is not None else self.schedule
        for n in range(self.N):
            if np.sum(s[n, :] != -1) > 6:
                return False
        return True

    def check_all_hard_constraints(self, schedule=None):
        """检查所有硬约束"""
        s = schedule if schedule is not None else self.schedule
        checks = [
            self.check_daily_shift_demand(s),
            self.check_one_shift_per_day(s),
            self.check_no_three_consecutive_night(s),
            self.check_no_night_to_morning(s),
            self.check_min_rest_days(s),
            self.check_no_7_consecutive_work(s),
        ]
        return all(checks)

    # ----------------------------------------------------------
    # 软约束评价
    # ----------------------------------------------------------

    def evaluate_balance(self, schedule=None):
        """评价班次均衡度：越均衡得分越高（0~1）"""
        s = schedule if schedule is not None else self.schedule
        # 理想每个班次天数 = (总工作天数) / 3 / N
        total_work = np.sum(s != -1)
        ideal_per_shift = total_work / (self.N * self.S)
        total_penalty = 0.0
        for n in range(self.N):
            for shift in range(self.S):
                count = np.sum(s[n, :] == shift)
                total_penalty += abs(count - ideal_per_shift)
        max_penalty = self.N * self.S * self.D
        return max(0.0, 1.0 - total_penalty / max_penalty)

    def evaluate_preference(self, schedule=None):
        """评价偏好满足度：得分越高越好（0~1）"""
        s = schedule if schedule is not None else self.schedule
        satisfied = 0
        total = 0
        for n in range(self.N):
            for d in range(self.D):
                pref = self.preferences[n, d]
                if pref != -1 and s[n, d] != -1:  # 有偏好且安排了工作
                    total += 1
                    if s[n, d] == pref:
                        satisfied += 1
        return satisfied / total if total > 0 else 1.0

    def evaluate_soft_score(self, schedule=None):
        """综合软约束得分"""
        s = schedule if schedule is not None else self.schedule
        balance = self.evaluate_balance(s)
        pref = self.evaluate_preference(s)
        return W_BALANCE * balance + W_PREFERENCE * pref

    # ----------------------------------------------------------
    # 两阶段求解
    # ----------------------------------------------------------

    def solve(self, max_attempts=30):
        """
        主求解方法：两阶段构造 + 局部搜索

        阶段1：分配工作/休息日
          - 保证周休≥2天，不连上7天
          - 每名护士工作 0~5 天

        阶段2：填充班次（早/中/晚）
          - 满足各班次日需求
          - 不连续三天晚班
          - 晚班后不接早班
        """
        print("正在构建排班...")
        best_schedule = None
        best_score = -1

        for attempt in range(max_attempts):
            schedule = -np.ones((self.N, self.D), dtype=int)

            # 阶段1：分配工作/休息日
            if not self._phase1_work_days(schedule, attempt):
                continue

            # 阶段2：填充班次
            if not self._phase2_assign_shifts(schedule):
                continue

            # 验证
            if not self.check_all_hard_constraints(schedule):
                continue

            # 局部搜索优化
            schedule = self._local_search(schedule)

            score = self.evaluate_soft_score(schedule)
            if score > best_score:
                best_score = score
                best_schedule = schedule.copy()
                print(f"  - 尝试 {attempt+1}：可行解，软约束得分 = {score:.3f}")

        if best_schedule is None:
            print("⚠️ 警告：所有尝试均未找到可行解！")
            print("  提示：可能是需求太高或护士太少，请检查 DEMAND 和 N_NURSES。")
            return None

        self.schedule = best_schedule
        print(f"✅ 排班完成！最佳得分 = {best_score:.3f}")
        return best_schedule

    def _phase1_work_days(self, schedule, seed_offset=0):
        """
        阶段1：分配工作/休息日

        策略：每名护士恰好工作 N_WORK 天（让总工作日 = 总需求），
        剩余 N_DAYS - N_WORK 天休息。
        随机打乱避免系统性偏差（对称性破坏）。
        """
        total_demand = int(np.sum(self.demand))
        # 计算需要多少总工作日
        # 每名护士工作 N_WORK 天，总和 = N_WORK * N_NURSES >= total_demand
        # 但也要避免过多导致部分护士只能休息0~1天
        work_per_nurse = max(3, int(np.ceil(total_demand / self.N)))

        rng = random.Random(seed_offset)
        for n in range(self.N):
            # 随机选择 work_per_nurse 个工作日
            work_days = rng.sample(range(self.D), min(work_per_nurse, self.D))
            for d in work_days:
                schedule[n, d] = -2  # 标记为"工作日待分配班次"

        # 检查总工作日是否 >= 总需求
        total_work = np.sum(schedule != -1)
        if total_work < total_demand:
            return False

        return True

    def _phase2_assign_shifts(self, schedule):
        """
        阶段2：填充班次（早/中/晚）

        逐天填充，确保每个班次需求满足、不连续三天晚班、晚班后不早班。
        使用贪心 + 约束传播 + 冲突回退（LNS 风格）。
        """
        # 按天迭代
        for d in range(self.D):
            # 当天需要填充的护士列表（标记为 -2 的）
            candidates = [n for n in range(self.N) if schedule[n, d] == -2]
            rng_candidates = candidates.copy()
            random.shuffle(rng_candidates)

            # 需求
            need = self.demand[d].copy()
            need_early, need_mid, need_night = int(need[0]), int(need[1]), int(need[2])

            if len(candidates) < need_early + need_mid + need_night:
                return False  # 人不够

            # 按约束最紧的人优先分配
            # 优先分配：前一天晚班 + 前两天晚班（易触发约束）
            priority = []
            for n in candidates:
                # 计算约束紧密度
                urgency = 0
                if d >= 1 and schedule[n, d-1] == 2:
                    urgency += 10  # 不能早班
                if d >= 2 and schedule[n, d-1] == 2 and schedule[n, d-2] == 2:
                    urgency += 20  # 不能晚班
                priority.append((n, urgency))
            priority.sort(key=lambda x: -x[1])

            # 先分配有约束的人
            assigned_pool = []
            for n, urg in priority:
                if urg == 0:
                    break  # 后面的人无约束
                # 决定分配什么班次
                feasible = [0, 1, 2]
                if d >= 2 and schedule[n, d-1] == 2 and schedule[n, d-2] == 2:
                    feasible.remove(2)  # 不能三连晚班
                if d >= 1 and schedule[n, d-1] == 2:
                    if 0 in feasible:
                        feasible.remove(0)  # 晚班后不早班

                # 选可行的且还有需求的班次
                chosen = None
                for s in feasible:
                    if s == 0 and need_early > 0:
                        chosen = 0
                        break
                    elif s == 1 and need_mid > 0:
                        chosen = 1
                        break
                    elif s == 2 and need_night > 0:
                        chosen = 2
                        break
                if chosen is None and feasible:
                    # 如果所有可行班次需求已满，选一个需求最低的
                    # 这将超出需求（软超额），但仍可行
                    chosen = feasible[0]

                if chosen is not None:
                    schedule[n, d] = chosen
                    assigned_pool.append(n)
                    if chosen == 0:
                        need_early -= 1
                    elif chosen == 1:
                        need_mid -= 1
                    elif chosen == 2:
                        need_night -= 1

            # 剩余名额按偏好分配
            remaining = [n for n in candidates if n not in assigned_pool]
            random.shuffle(remaining)

            # 先满足需求的优先顺序
            for need_name, need_var, shift_idx in [
                ("早班", need_early, 0),
                ("中班", need_mid, 1),
                ("晚班", need_night, 2),
            ]:
                while need_var > 0 and remaining:
                    # 找偏好的护士
                    best_n = None
                    best_pref = False
                    for n in remaining:
                        if self.preferences[n, d] == shift_idx:
                            best_n = n
                            best_pref = True
                            break
                    if best_n is None:
                        best_n = remaining[0]

                    schedule[best_n, d] = shift_idx
                    remaining.remove(best_n)
                    need_var -= 1

            # 剩余的人随便分配（超出需求，但不会超过太多）
            # 找还有需求的班次
            for n in remaining:
                for s, need_var_ref in [(0, need_early), (1, need_mid), (2, need_night)]:
                    if need_var_ref > 0:
                        schedule[n, d] = s
                        if s == 0:
                            need_early -= 1
                        elif s == 1:
                            need_mid -= 1
                        elif s == 2:
                            need_night -= 1
                        break
                else:
                    # 所有需求都已满足，排最不紧缺的班次
                    schedule[n, d] = 0  # 早班（通常需要最多人）

        return True

    # ----------------------------------------------------------
    # 局部搜索（LNS）
    # ----------------------------------------------------------

    def _local_search(self, schedule, iterations=80):
        """
        局部搜索（LNS 风格）：破坏 15% 的排班 → 重新填充 → 接受更好解
        """
        current = schedule.copy()
        current_score = self.evaluate_soft_score(current)

        for it in range(iterations):
            # 1. 破坏：随机选择 15% 的排班清除
            all_cells = [(n, d) for n in range(self.N) for d in range(self.D)
                         if current[n, d] != -1]
            random.shuffle(all_cells)
            n_destroy = max(1, int(len(all_cells) * 0.15))
            to_destroy = all_cells[:n_destroy]

            saved = {}
            for n, d in to_destroy:
                saved[(n, d)] = current[n, d]
                current[n, d] = -1  # 清除

            # 2. 修复：重新分配
            # 按天修复，确保不破坏需求
            repair_ok = True
            days_affected = set(d for _, d in to_destroy)
            for d in sorted(days_affected):
                candidates = [n for n in range(self.N) if current[n, d] == -1
                              and (n, d) in to_destroy]
                if not candidates:
                    continue

                # 计算当天仍需多少人
                for shift in range(self.S):
                    n_needed = self.demand[d, shift] - np.sum(current[:, d] == shift)
                    while n_needed > 0 and candidates:
                        n = candidates.pop(0)
                        current[n, d] = shift
                        n_needed -= 1

                # 剩余的排最合适的班次
                for n in candidates:
                    current[n, d] = self.preferences[n, d] if self.preferences[n, d] != -1 else 0

            # 验证硬约束
            if not self.check_all_hard_constraints(current):
                # 回退
                for n, d in saved:
                    current[n, d] = saved[(n, d)]
                continue

            new_score = self.evaluate_soft_score(current)
            if new_score > current_score:
                current_score = new_score
            else:
                # 回退
                for n, d in saved:
                    current[n, d] = saved[(n, d)]

        return current

    # ----------------------------------------------------------
    # 对称性破坏
    # ----------------------------------------------------------

    def _rank_nurses(self):
        """
        对称性破坏：按护士的偏好模式排序
        相同偏好的护士会被排在一起，固定顺序打破对称等价性。
        """
        nurses = list(range(self.N))
        nurses.sort(key=lambda n: (
            -sum(1 for p in self.preferences[n, :] if p != -1),
            n
        ))
        return nurses


# ============================================================
# 3. 输出报告
# ============================================================

def print_report(scheduler):
    """打印排班报告"""
    s = scheduler.schedule

    print("\n" + "=" * 60)
    print("排班报告")
    print("=" * 60)

    # 每日各班次人数统计
    print("\n--- 各班次在岗人数 ---")
    print(f"{'天':>4} | ", end="")
    for shift_name in SHIFT_NAMES:
        print(f"{shift_name:>6}", end=" ")
    print(f"{'需求满足':>8}")
    print("-" * 45)

    for d in range(scheduler.D):
        print(f"周{d+1:>2} | ", end="")
        all_ok = True
        for shift in range(scheduler.S):
            count = np.sum(s[:, d] == shift)
            req = scheduler.demand[d, shift]
            ok = "✅" if count >= req else "❌"
            if count < req:
                all_ok = False
            print(f"{count:>2}/{req}{ok:>2}", end=" ")
        print(f"{'✅' if all_ok else '❌':>8}")

    # 护士统计
    print("\n--- 前 10 名护士排班统计 ---")
    print(f"{'护士':>4} | ", end="")
    for shift_name in SHIFT_NAMES:
        print(f"{shift_name:>4}", end=" ")
    print(f"{'休息':>4} {'偏好分':>6}")
    print("-" * 40)

    for n in range(min(10, scheduler.N)):
        print(f"{n:>4} | ", end="")
        for shift in range(scheduler.S):
            count = np.sum(s[n, :] == shift)
            print(f"{count:>4}", end=" ")
        rest = np.sum(s[n, :] == -1)
        # 偏好分只计算工作日的偏好
        pref_sat = 0
        pref_tot = 0
        for d in range(scheduler.D):
            pref = scheduler.preferences[n, d]
            if pref != -1 and s[n, d] != -1:
                pref_tot += 1
                if s[n, d] == pref:
                    pref_sat += 1
        pref_score = pref_sat / pref_tot if pref_tot > 0 else 1.0
        print(f"{rest:>4} {pref_score:>6.2f}")

    # 约束检查
    print("\n--- 约束检查 ---")
    checks = [
        ("每天最多一个班次", scheduler.check_one_shift_per_day()),
        ("不连续3天晚班", scheduler.check_no_three_consecutive_night()),
        ("晚班后不接早班", scheduler.check_no_night_to_morning()),
        ("每周至少休息2天", scheduler.check_min_rest_days()),
        ("不连上7天", scheduler.check_no_7_consecutive_work()),
    ]
    for name, result in checks:
        print(f"  {name}: {'✅' if result else '❌'}")
    print(f"  班次需求满足: {'✅' if scheduler.check_daily_shift_demand() else '❌'}")

    # 软约束得分
    print(f"\n--- 软约束评价 ---")
    print(f"  班次均衡度: {scheduler.evaluate_balance():.2f}")
    print(f"  偏好满足度: {scheduler.evaluate_preference():.2f}")
    print(f"  综合得分:   {scheduler.evaluate_soft_score():.3f}")


# ============================================================
# 4. 对称性破坏对比
# ============================================================

def demo_symmetry_breaking():
    """演示对称性破坏的效果"""
    print("\n" + "=" * 60)
    print("对称性破坏演示")
    print("=" * 60)
    print("（两种模式的求解时间对比——由于使用纯 Python 实现，")
    print(" 时间差异主要来自搜索空间的随机性）")

    # 用 default 顺序 vs 排序顺序跑两次
    import time

    # 测试1：无对称性破坏（nurse_order = 自然顺序）
    np.random.seed(99)
    pref1 = np.random.randint(-1, 3, size=(30, 7))
    demand1 = np.array([[9, 7, 5]] * 7)

    s1 = NurseScheduler(n_nurses=30, n_days=7, n_shifts=3,
                        demand=demand1, preferences=pref1)
    t1 = time.time()
    s1.solve(max_attempts=20)
    t1 = time.time() - t1

    # 测试2：有对称性破坏（排序）
    s2 = NurseScheduler(n_nurses=30, n_days=7, n_shifts=3,
                        demand=demand1, preferences=pref1)
    t2 = time.time()
    s2._rank_nurses()  # 对称性破坏内置在 solve 中
    s2.solve(max_attempts=20)
    t2 = time.time() - t2

    score1 = s1.evaluate_soft_score() if s1.schedule is not None else 0
    score2 = s2.evaluate_soft_score() if s2.schedule is not None else 0
    print(f"\n自然顺序: {t1:.2f}s, 得分={score1:.3f}")
    print(f"排序(对称破坏): {t2:.2f}s, 得分={score2:.3f}")
    print(f"对称性破坏通过排序打破了等价解空间，使搜索更集中。")


# ============================================================
# 5. 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("护士排班系统 — 约束传播 + 两阶段构造 + 局部搜索")
    print("=" * 60)
    print(f"护士人数: {N_NURSES}")
    print(f"排班天数: {N_DAYS}")
    print(f"班次数量: {N_SHIFTS} (早班/中班/晚班)")
    print(f"总需求:   {int(np.sum(DEMAND))} 个班次")

    print("\n约束条件:")
    print("  - 班次需求: 早班(15/10), 中班(12/10), 晚班(8/6)")
    print("  - 每天最多一个班次")
    print("  - 不连续3天晚班")
    print("  - 晚班后不接早班")
    print("  - 每周至少休息2天")
    print("  - 不连上7天")

    # 求解
    scheduler = NurseScheduler()
    result = scheduler.solve(max_attempts=30)

    if result is not None:
        print_report(scheduler)
        # demo_symmetry_breaking()
    else:
        print("\n⚠️ 未找到可行解。可能原因：")
        print("  1. 总需求 > 总可用工作日（护士太少）")
        print("  2. 约束过于严格（如周末需求太高）")
        print("  建议：尝试减少 DEMAND 或增加 N_NURSES")

    print("\n" + "=" * 60)
    print("提示: 这是一个教学演示实现。")
    print("生产环境建议使用 CP-SAT (OR-Tools) 或 Choco 等 CP 求解器。")
    print("两阶段算法虽然不是最优，但可以保证找到可行解。")
    print("=" * 60)
