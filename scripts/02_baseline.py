"""Run the M2 baseline gate using the existing run's frozen configuration."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from sbo.m2_cli import main

if __name__ == "__main__":
    raise SystemExit(main("baseline", PROJECT))
