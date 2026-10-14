"""Run/resume the M4 sequential loop on an existing frozen experiment."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from sbo.evaluator import Evaluator
from sbo.optimizer import run_optimize
from sbo.stages import resolve_run, run_ownership


def main():
    parser = argparse.ArgumentParser(description="M4 serial GP/EI/XFOIL loop with persistent stopping and budget accounting")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-evaluations", type=int, help="Recovery diagnostic: stop after N outcomes and model update")
    parser.add_argument("--review-eval-id")
    parser.add_argument("--review-decision", choices=("accept", "reject"))
    parser.add_argument("--review-reason")
    args = parser.parse_args()
    try:
        directory = resolve_run(PROJECT, args.run_id)
        review = (args.review_eval_id, args.review_decision, args.review_reason)
        if any(review):
            if not all(review) or not args.resume:
                parser.error("review requires all review arguments and --resume")
            with Evaluator(directory) as evaluator, run_ownership(evaluator.store):
                state = evaluator.store.stage("optimize")
                if not state or not state.get("pending") or state["pending"]["eval_id"] != args.review_eval_id:
                    raise ValueError("review must identify this loop's pending evaluation")
                evaluator.review(*review, include_holdout=False)
        report = run_optimize(directory, resume=args.resume, stop_after_evaluations=args.stop_after_evaluations,
            on_progress=lambda event: print(json.dumps(event, allow_nan=False), file=sys.stderr, flush=True))
        print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
        return 0 if report["status"] == "success" else 2
    except (KeyboardInterrupt, Exception) as exc:
        print(json.dumps({"status": getattr(exc, "status", "failed"), "reason": f"{type(exc).__name__}: {exc}"}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
