"""Reusable smoke-check helpers for course code directories."""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path


def compile_python_files(code_dir: Path) -> None:
    files = sorted(path for path in code_dir.glob("*.py") if path.name != "run_checks.py")
    print(f"Python compile checks: {len(files)} files")
    cache_root = Path(os.environ.get("PYTHONPYCACHEPREFIX", tempfile.gettempdir())) / "or_course_pycache"
    for path in files:
        cache_path = cache_root / path.with_suffix(".pyc").name
        py_compile.compile(str(path), cfile=str(cache_path), doraise=True)


def run_scripts(code_dir: Path, scripts: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig-or-course"))
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "pycache-or-course"))

    for script_name in scripts:
        script = code_dir / script_name
        if not script.exists():
            continue
        print(f"Running {script.relative_to(code_dir)}")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=code_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f"Course script failed: {script}")


def main(code_dir: Path, scripts: list[str] | None = None) -> None:
    compile_python_files(code_dir)
    run_scripts(code_dir, scripts or ["capstone.py"])
    print("Course smoke checks passed.")
