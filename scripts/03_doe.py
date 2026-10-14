"""Collect M2 training/holdout data and durably resume unfinished work."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from sbo.m2_cli import main

if __name__ == "__main__":
    raise SystemExit(main("doe", PROJECT))
