"""Smoke checks for Course 09 examples."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.course_checks import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
