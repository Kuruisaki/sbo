"""Durable 16-point training DOE plus six sealed holdout labels, with replacements."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.spatial.distance import pdist

from .config import canonical_json, write_json
from .evaluator import Evaluator
from .sampling import initial_plan, normalize, replace_slot
from .stages import (StageIncomplete, checkpoint_failure, prerequisites, receipt,
                     run_ownership, update_manifest, verify_receipt)


def _reserve_design(store, slot, config):
    m, p = slot["design"]
    t = config["design"]["t"]
    identity = sha256(canonical_json([float(m).hex(), float(p).hex(), float(t).hex()]).encode("ascii")).hexdigest()
    store.register_design({"design_id": identity, "m": m, "p": p, "t": t,
                           "u_m": slot["u"][0], "u_p": slot["u"][1], "split": slot["split"]})


def _plot(directory, plan, config):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for split, marker, label in (("train", "o", "Training DOE"), ("holdout", "x", "Reserved holdout (coordinates only)")):
        points = np.array([slot["u"] for slot in plan["slots"] if slot["split"] == split])
        axis.scatter(points[:, 0], points[:, 1], marker=marker, label=label)
    baseline = normalize([config["design"]["baseline"]], config["design"]["bounds"])[0]
    axis.scatter(*baseline, marker="*", s=150, label="Reused baseline")
    failed = [history["u"] for slot in plan["slots"] for history in slot["history"]]
    if failed:
        points = np.asarray(failed)
        axis.scatter(points[:, 0], points[:, 1], marker="+", label="Failed / replaced coordinates")
    axis.set(xlabel="Normalized m", ylabel="Normalized p", xlim=(0, 1), ylim=(0, 1), title="M2 design distribution")
    axis.grid(alpha=.25)
    axis.legend()
    figure.savefig(directory / "doe.png", dpi=160)
    plt.close(figure)


def _validate_results(evaluator, state):
    baseline = evaluator.store.get_result(state["baseline_eval_id"])
    if not baseline or not baseline.usable_for_training or not evaluator._cache_valid(baseline, True):
        raise RuntimeError("DOE baseline no longer validates")
    for slot in state["slots"]:
        if slot["status"] == "success":
            result = evaluator.store.get_result(slot["eval_id"])
            if not result or result.status != "success" or result.split != slot["split"] or not evaluator._cache_valid(result, False):
                raise RuntimeError("completed DOE result no longer validates: " + slot["slot_id"])


def run_doe(run_dir: str | Path, *, resume: bool = False, stop_after: int | None = None) -> dict:
    """stop_after is an explicit recovery diagnostic, stopping only at DB boundaries."""
    if stop_after is not None and (type(stop_after) is not int or stop_after <= 0):
        raise ValueError("stop_after must be a positive integer")
    with Evaluator(run_dir) as evaluator, run_ownership(evaluator.store):
        manifest = prerequisites(evaluator, "doe")
        state = evaluator.store.stage("doe")
        if state and state["status"] == "success":
            verify_receipt(evaluator.run_dir, state)
            _validate_results(evaluator, state)
            evaluator.store.export_csv()
            evaluator.store.write_checkpoint()
            return json.loads((evaluator.run_dir / "doe.json").read_text(encoding="utf-8"))
        if state and not resume:
            raise StageIncomplete("DOE already started; use --resume")
        if state is None:
            occupied = [list(row) for row in evaluator.store.connection.execute("SELECT u_m,u_p FROM designs ORDER BY rowid")]
            state = initial_plan(evaluator.config, excluded=occupied)
            state["baseline_eval_id"] = manifest["baseline_eval_id"]
            if len(evaluator.store.training_data()) != 1:
                raise RuntimeError("DOE must start with only the accepted baseline training point")
        try:
            # Coordinates/split reservation is idempotent even if a prior checkpoint write failed.
            for slot in state["slots"]:
                _reserve_design(evaluator.store, slot, evaluator.config)
            evaluator.store.save_stage("doe", "running", state)
            _validate_results(evaluator, state)
            update_manifest(evaluator, "doe", "running")
            completed_this_call = 0
            # Holdout outcomes never change later training proposals: train slots finish first.
            ordered = sorted(state["slots"], key=lambda slot: slot["split"] == "holdout")
            for slot in ordered:
                while slot["status"] != "success":
                    if slot["eval_id"] is None:
                        slot["eval_id"] = "eval_" + uuid4().hex
                    slot["status"] = "running"
                    _reserve_design(evaluator.store, slot, evaluator.config)
                    # This proposal, RNG state and exclusion set commit BEFORE solver launch.
                    evaluator.store.save_stage("doe", "running", state)
                    result = evaluator.evaluate(slot["design"], split=slot["split"], purpose="doe",
                                                evaluation_id=slot["eval_id"], resume=resume)
                    slot.update(eval_id=result.eval_id, status=result.status)
                    evaluator.store.save_stage("doe", "running", state)
                    evaluator.store.export_csv()
                    if result.status == "needs_review":
                        raise StageIncomplete("explicit review required for " + result.eval_id, "needs_review")
                    if result.status == "skipped" or (result.failure_reason or "").startswith("retry_budget_exhausted"):
                        raise StageIncomplete("DOE quota incomplete: " + str(result.failure_reason), "skipped")
                    completed_this_call += 1
                    if result.status == "failed":
                        systemic = {"launch_failed", "cleanup_failed", "solver_identity_mismatch", "solver_setting_mismatch", "aborted"}
                        if any(item["runner"].get("status") in systemic for item in result.attempts):
                            raise StageIncomplete("DOE solver/cleanup failure; no replacement launch: " + str(result.failure_reason), "failed")
                        counts = evaluator.store.budget_snapshot()
                        ordinary_cap = counts["max_process_launches"] - counts["reserved_final_verification_launches"]
                        if counts["distinct_standard_designs"] >= counts["max_distinct_standard_designs"] or counts["process_launches"] >= ordinary_cap:
                            raise StageIncomplete("DOE quota incomplete: replacement budget exhausted", "skipped")
                        replace_slot(state, slot, evaluator.config)
                        _reserve_design(evaluator.store, slot, evaluator.config)
                        evaluator.store.save_stage("doe", "running", state)
                    if stop_after is not None and completed_this_call >= stop_after:
                        raise StageIncomplete("requested recovery diagnostic boundary reached")
            _validate_results(evaluator, state)
            training_count = len(evaluator.store.training_data())
            expected_train = evaluator.config["doe"]["initial_points"] + 1
            holdout_count = sum(slot["status"] == "success" and slot["split"] == "holdout" for slot in state["slots"])
            if training_count != expected_train or holdout_count != evaluator.config["doe"]["holdout_points"]:
                raise StageIncomplete("DOE valid-sample quota mismatch", "failed")
            distances = pdist(np.asarray(state["excluded"]))
            minimum = float(distances.min())
            if minimum < evaluator.config["doe"]["min_normalized_distance"]:
                raise StageIncomplete("DOE normalized separation invariant failed", "failed")
            files = evaluator.store.export_csv()
            report = {"status": "success", "config_hash": evaluator.config.config_hash,
                      "training_points": training_count, "sealed_holdout_points": holdout_count,
                      "holdout_labels_sealed": True, "strict_lhs": state["strict_lhs"], "seeds": state["seeds"],
                      "replacement_designs": sum(len(slot["history"]) for slot in state["slots"]),
                      "minimum_normalized_distance": minimum, "datasets": files,
                      "budget": evaluator.store.budget_snapshot(),
                      "scope": "M2 data collection only; no surrogate fit, candidate ranking or holdout metric."}
            write_json(evaluator.run_dir / "doe.json", report)
            write_json(evaluator.run_dir / "doe-plan.json", state)
            _plot(evaluator.run_dir, state, evaluator.config)
            lines = ["# M2 DOE 数据验收", "", f"有效训练点：{training_count}（含复用基准）。",
                     f"封存留出点：{holdout_count}；标签不进入训练查询、候选排名或停止判断。", "",
                     f"种子：训练 {state['seeds']['train']}，留出 {state['seeds']['holdout']}。",
                     f"失败替补数：{report['replacement_designs']}；严格 LHS 状态：{state['strict_lhs']}。",
                     f"最小归一化距离：{minimum:.8g}。", "",
                     f"累计进程启动：{report['budget']['process_launches']}；不同标准设计：{report['budget']['distinct_standard_designs']}。",
                     "", "![采样分布](doe.png)", "", "训练数据：samples.csv；失败训练记录：failures.csv；封存留出审计：holdout.csv。",
                     "SQLite 是预算、结果及恢复状态事实源；CSV 与 checkpoint 为可重建快照。",
                     "M3–M5 的代理拟合、序贯优化和收益复核尚未执行。"]
            (evaluator.run_dir / "doe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            state["artifacts"] = receipt(evaluator.run_dir, ["doe.json", "doe.md", "doe.png", "doe-plan.json"])
            evaluator.store.save_stage("doe", "success", state)
            update_manifest(evaluator, "doe", "success", doe_passed=True, m2_passed=True)
            return report
        except BaseException as exc:
            checkpoint_failure(evaluator, "doe", state, exc)
            raise
