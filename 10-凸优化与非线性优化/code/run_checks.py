"""Run lightweight checks for the convex optimization course examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


SCRIPTS = [
    "case03_least_squares.py",
    "case04_portfolio_qp.py",
    "case05_logistic_regression.py",
    "case06_constrained_design.py",
    "case07_nonconvex_pitfalls.py",
    "capstone.py",
]


def run_script(script: str) -> bool:
    print(f"\n=== RUN {script} ===")
    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    ok = result.returncode == 0
    print(f"{script}: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    results = {script: run_script(script) for script in SCRIPTS}
    passed = sum(results.values())
    print(f"\nSummary: {passed}/{len(SCRIPTS)} passed")
    if passed != len(SCRIPTS):
        failed = [script for script, ok in results.items() if not ok]
        raise SystemExit(f"Failed scripts: {failed}")


if __name__ == "__main__":
    main()
