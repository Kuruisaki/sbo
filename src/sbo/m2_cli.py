"""Shared CLI for frozen-run M2 stages; exit 2 means a reviewable incomplete stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import run_baseline
from .doe import run_doe
from .evaluator import Evaluator
from .stages import resolve_run, run_ownership


def main(stage: str, project: Path) -> int:
    parser = argparse.ArgumentParser(description=f"Execute M2 {stage} in an existing frozen run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--review-eval-id")
    parser.add_argument("--review-decision", choices=("accept", "reject"))
    parser.add_argument("--review-reason")
    if stage == "doe":
        parser.add_argument("--stop-after", type=int, help="Recovery diagnostic: stop after this many evaluation outcomes")
    args = parser.parse_args()
    try:
        directory = resolve_run(project, args.run_id)
        review = (args.review_eval_id, args.review_decision, args.review_reason)
        if any(review):
            if not all(review) or not args.resume:
                parser.error("review requires all three review arguments and --resume")
            with Evaluator(directory) as evaluator, run_ownership(evaluator.store):
                evaluator.review(*review)
        report = run_baseline(directory, resume=args.resume) if stage == "baseline" else run_doe(
            directory, resume=args.resume, stop_after=args.stop_after)
        print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    except (KeyboardInterrupt, Exception) as exc:
        print(json.dumps({"status": getattr(exc, "status", "failed"), "reason": f"{type(exc).__name__}: {exc}"}, ensure_ascii=True))
        return 2
