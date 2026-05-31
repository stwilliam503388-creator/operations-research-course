"""Repository-level smoke checks for course examples.

The goal is fast confidence, not exhaustive numerical validation:
1. compile every Python script under course code/python directories;
2. compile every C++17 example;
3. run each course's local smoke checks.

This script intentionally avoids writing Python bytecode caches into the
repository so it can run in read-only CI or sandboxed environments.
"""

from __future__ import annotations

import py_compile
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def iter_python_files() -> list[Path]:
    course_files = [path for path in ROOT.glob("*/code/python/*.py") if path.is_file()]
    common_files = [path for path in (ROOT / "common").glob("*.py") if path.is_file()]
    return sorted(course_files + common_files)


def iter_cpp_files() -> list[Path]:
    return sorted(path for path in ROOT.glob("*/code/cpp/*.cpp") if path.is_file())


def compile_python() -> None:
    files = iter_python_files()
    print(f"Python compile checks: {len(files)} files")
    cache_root = Path(os.environ.get("PYTHONPYCACHEPREFIX", tempfile.gettempdir())) / "or_course_pycache"
    for path in files:
        cache_path = cache_root / path.relative_to(ROOT).with_suffix(".pyc")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        py_compile.compile(str(path), cfile=str(cache_path), doraise=True)


def compile_cpp() -> None:
    files = iter_cpp_files()
    print(f"C++17 compile checks: {len(files)} files")
    for path in files:
        output = Path(tempfile.gettempdir()) / f"{path.stem}_check"
        try:
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
        finally:
            output.unlink(missing_ok=True)


def run_unit_tests() -> None:
    test_dirs = sorted({path.parent for path in ROOT.glob("**/test_*.py") if path.is_file()})
    print(f"Unit test discovery: {len(test_dirs)} directories")
    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "pycache-or-course"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for test_dir in test_dirs:
        print(f"Running unit tests in {test_dir.relative_to(ROOT)}")
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", str(test_dir), "-p", "test_*.py"],
            cwd=ROOT,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Unit tests failed in: {test_dir.relative_to(ROOT)}")


def iter_course_check_scripts() -> list[Path]:
    return sorted(path for path in ROOT.glob("*/code/python/run_checks.py") if path.is_file())


def run_course_checks() -> None:
    scripts = iter_course_check_scripts()
    print(f"Course smoke checks: {len(scripts)} scripts")
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig-or-course"))
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "pycache-or-course"))
    for script in scripts:
        print(f"Running {script.relative_to(ROOT)}")
        result = subprocess.run([sys.executable, str(script)], cwd=script.parent, env=env, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Course checks failed: {script.relative_to(ROOT)}")


def main() -> None:
    compile_python()
    compile_cpp()
    run_unit_tests()
    run_course_checks()
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
