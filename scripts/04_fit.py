"""Fit the initial GP/LOO and save one unassessed M3 proposal in a frozen run."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from sbo.fitting import run_fit
from sbo.stages import resolve_run


def main():
    parser = argparse.ArgumentParser(description="M3 initial GP, independent LOO and candidate proposal")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-folds", type=int, help="Recovery diagnostic: stop after N newly completed LOO folds")
    args = parser.parse_args()
    try:
        result = run_fit(resolve_run(PROJECT, args.run_id), resume=args.resume, stop_after_folds=args.stop_after_folds)
        print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    except (KeyboardInterrupt, Exception) as exc:
        print(json.dumps({"status": getattr(exc, "status", "failed"), "reason": f"{type(exc).__name__}: {exc}"}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
