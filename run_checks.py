"""Repository-level smoke checks for course examples.

The goal is fast confidence, not exhaustive numerical validation:
1. compile every Python script under course code directories;
2. compile every C++17 example;
3. run the full checks for the newest convex optimization course.
"""

from __future__ import annotations

import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def iter_python_files() -> list[Path]:
    return sorted(path for path in ROOT.glob("*/code/*.py") if path.is_file())


def iter_cpp_files() -> list[Path]:
    return sorted(path for path in ROOT.glob("*/code/cpp/*.cpp") if path.is_file())


def compile_python() -> None:
    files = iter_python_files()
    print(f"Python compile checks: {len(files)} files")
    for path in files:
        py_compile.compile(str(path), doraise=True)


def compile_cpp() -> None:
    files = iter_cpp_files()
    print(f"C++17 compile checks: {len(files)} files")
    for path in files:
        output = Path(tempfile.gettempdir()) / f"{path.stem}_check"
        result = subprocess.run(
            ["g++", "-std=c++17", "-O2", str(path), "-o", str(output)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f"C++ compile failed: {path}")


def run_convex_course_checks() -> None:
    script = ROOT / "10-凸优化与非线性优化" / "code" / "run_checks.py"
    print(f"Running {script.relative_to(ROOT)}")
    result = subprocess.run([sys.executable, str(script)], cwd=script.parent, check=False)
    if result.returncode != 0:
        raise RuntimeError("Convex optimization course checks failed")


def main() -> None:
    compile_python()
    compile_cpp()
    run_convex_course_checks()
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
