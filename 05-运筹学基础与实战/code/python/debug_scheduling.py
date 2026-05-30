#!/usr/bin/env python3
"""Trace the two-phase scheduling construction step by step."""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from case07_scheduling import NurseScheduler

# Small test: 10 nurses, 7 days, lower demand.
# Keep the 7-day horizon so the "at least 2 rest days" weekly constraint still applies.
small_demand = np.array([
    [2, 2, 1],  # day 0: need 5
    [2, 2, 1],  # day 1: need 5
    [2, 1, 1],  # day 2: need 4
    [1, 1, 1],  # day 3: need 3
    [1, 1, 1],  # day 4: need 3
    [1, 1, 0],  # day 5: need 2
    [1, 1, 0],  # day 6: need 2
])
np.random.seed(123)
small_pref = np.random.randint(-1, 3, size=(10, 7))


def main():
    print("Demand:", small_demand.tolist())
    print("Total demand:", small_demand.sum())
    print("Max work capacity:", 10 * 5)
    print("Feasible check:", 10 * 5 >= small_demand.sum())
    print()

    sched = NurseScheduler(n_nurses=10, n_days=7, n_shifts=3, demand=small_demand, preferences=small_pref)
    schedule = sched.solve(max_attempts=30)

    print(f"\nFinal schedule:\n{schedule}")
    success = schedule is not None
    print(f"Success: {success}")
    if success:
        print("Work-day counts by day:", np.sum(schedule != -1, axis=0).tolist())
        print(f"All hard: {sched.check_all_hard_constraints(schedule)}")


if __name__ == "__main__":
    main()
