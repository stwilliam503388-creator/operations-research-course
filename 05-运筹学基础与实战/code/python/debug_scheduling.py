#!/usr/bin/env python3
"""Trace the greedy construction step by step"""

# 教学注释：先识别业务对象，再看它们如何映射为优化、仿真或启发式模型。
# 结果解读侧重成本、资源利用率和服务水平等管理指标。


import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from case07_scheduling import NurseScheduler, N_NURSES, N_DAYS, N_SHIFTS, DEMAND, PREFERENCES

# Small test: 10 nurses, 3 days, lower demand
small_demand = np.array([
    [2, 2, 1],  # day 0: need 5
    [2, 2, 1],  # day 1: need 5
    [2, 1, 1],  # day 2: need 4
])
np.random.seed(123)
small_pref = np.random.randint(-1, 3, size=(10, 3))

print("Demand:", small_demand.tolist())
print("Total demand:", small_demand.sum())
print("Max work capacity:", 10 * 5)
print("Feasible check:", 10 * 5 >= small_demand.sum())
print()

sched = NurseScheduler(n_nurses=10, n_days=3, n_shifts=3, demand=small_demand, preferences=small_pref)

# Monkey-patch to add tracing
original_construct = sched._greedy_construct

def traced_construct(schedule, nurse_order):
    print(f"Nurse order: {nurse_order}")
    for d in range(sched.D):
        print(f"\n=== Day {d} ===")
        remaining = sched.demand[d].copy()
        print(f"Demand: early={remaining[0]} mid={remaining[1]} night={remaining[2]}")
        
        for shift in sorted(range(sched.S), key=lambda s: -remaining[s]):
            while remaining[shift] > 0:
                found = False
                for n in nurse_order:
                    if schedule[n, d] != -1:
                        continue
                    feasible = sched.feasible_shifts(n, d, schedule)
                    if shift not in feasible:
                        continue
                    pref_cost = 0.0
                    if sched.preferences[n, d] == shift:
                        pref_cost = -1.0
                    elif sched.preferences[n, d] != -1:
                        pref_cost = 1.0
                    schedule[n, d] = shift
                    remaining[shift] -= 1
                    print(f"  Nurse {n} -> shift {shift} (pref_cost={pref_cost:.1f})")
                    found = True
                    break
                if not found:
                    print(f"  ⚠️ CANNOT fill shift {shift} on day {d}!")
                    return False
        
        # Rest assignment
        for n in nurse_order:
            if schedule[n, d] == -1:
                feasible = sched.feasible_shifts(n, d, schedule)
                if -1 in feasible:
                    schedule[n, d] = -1
                    print(f"  Nurse {n} -> rest")
                else:
                    # Force work
                    for s in feasible:
                        if s >= 0:
                            schedule[n, d] = s
                            print(f"  Nurse {n} -> forced shift {s} (rest not feasible)")
                            break
    
    return True

sched._greedy_construct = traced_construct
schedule = -np.ones((10, 3), dtype=int)
nurse_order = sched._rank_nurses()
success = sched._greedy_construct(schedule, nurse_order)

print(f"\nFinal schedule:\n{schedule}")
print(f"Success: {success}")
if success:
    print(f"All hard: {sched.check_all_hard_constraints(schedule)}")
