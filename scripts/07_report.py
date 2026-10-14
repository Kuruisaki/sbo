"""Generate final evidence and geometry exports without new aerodynamic calls."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from sbo.reporting import run_report
from sbo.stages import resolve_run


def main():
    parser = argparse.ArgumentParser(description="M5 revalidated final report and recommendation export")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        report = run_report(resolve_run(PROJECT, args.run_id))
        print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
        return 0 if report["status"] == "success" else 2
    except (KeyboardInterrupt, Exception) as exc:
        print(json.dumps({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
