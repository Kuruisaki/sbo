"""Serial M4: persisted proposal -> real evaluation -> verified data -> GP/LOO.

SQLite is the outcome/launch/proposal truth. Reviews halt without counting a
failure; final design outcomes (not individual attempts) drive stopping. Holdout
labels are never queried, including exports and failure checkpoints.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Callable
from uuid import uuid4

import numpy as np

from .acquisition import propose
from .config import file_sha256, write_json
from .evaluator import Evaluator, _runner_from_dict
from .fitting import _plots, _validate_proposal, _write_csv, run_fit
from .polar import parse_attempt
from .sampling import nearest_distances
from .stages import (StageIncomplete, checkpoint_failure, prerequisites, receipt,
                     run_ownership, update_manifest, verify_receipt)
from .surrogate import (SurrogateFitError, digest, fit_gp, load_model, loo_cv,
                        prepare_training_data, save_model)

OPTIMIZER_VERSION = "serial-sbo-v1"


def validate_stopping(config):
    stop, convergence = config["stopping"], config["stopping"]["convergence"]
    fixed = {"failure_rate_comparison": "greater_than", "require_full_failure_rate_window": True,
             "failure_count_policy": "final_design_outcome_after_retry", "stop_on_budget_exhaustion": True}
    if any(stop[key] != value for key, value in fixed.items()):
        raise ValueError("unsupported stopping/failure-counting policy")
    if (convergence["combine_conditions"] != "all" or convergence["improvement_comparison"] != "less_than_or_equal"
            or convergence["ei_comparison"] != "less_than"):
        raise ValueError("convergence requires improvement<=threshold AND valid pool EI<threshold")
    if (stop["max_consecutive_failed_designs"] <= 0 or stop["failure_rate_window"] <= 0
            or not 0 <= stop["max_failure_rate"] <= 1 or convergence["successful_points_window"] <= 0
            or convergence["min_successful_sequential_points"] < convergence["successful_points_window"]
            or convergence["max_best_cd_improvement"] < 0 or convergence["max_pool_ei"] < 0):
        raise ValueError("invalid stopping thresholds/windows")


def stopping_decision(history, budget, config, initial_best_cd: float, max_pool_ei: float | None) -> dict:
    """Recent W successful additions compare B[n-W] to B[n] (W transitions)."""
    validate_stopping(config)
    outcomes = [item for item in history if item["status"] in {"success", "failed"}]
    successes = [item for item in outcomes if item["status"] == "success"]
    n = len(successes)
    consecutive = 0
    for item in reversed(outcomes):
        if item["status"] != "failed":
            break
        consecutive += 1
    stop = config["stopping"]
    window = stop["failure_rate_window"]
    failure_rate = sum(item["status"] == "failed" for item in outcomes[-window:]) / window if len(outcomes) >= window else None
    convergence = stop["convergence"]
    improvement = None
    if n >= convergence["successful_points_window"]:
        values = [initial_best_cd, *[item["best_cd"] for item in successes]]
        improvement = float(values[-convergence["successful_points_window"] - 1] - values[-1])
    reason = None
    if consecutive >= stop["max_consecutive_failed_designs"]:
        reason = "consecutive_failed_designs"
    elif failure_rate is not None and failure_rate > stop["max_failure_rate"]:
        reason = "failure_rate_exceeded"
    elif n >= config["budget"]["max_successful_sequential_points"]:
        reason = "max_successful_sequential_points"
    elif config["doe"]["initial_points"] + 1 + config["doe"]["holdout_points"] + n >= config["budget"]["max_planned_valid_standard_designs"]:
        reason = "max_planned_valid_standard_designs"
    elif budget["distinct_standard_designs"] >= budget["max_distinct_standard_designs"]:
        reason = "distinct_design_budget"
    elif budget["process_launches"] + budget["pending_launch_reservations"] >= budget["max_process_launches"] - budget["reserved_final_verification_launches"]:
        reason = "process_launch_budget_or_final_reserve"
    elif (n >= convergence["min_successful_sequential_points"] and improvement is not None
          and improvement <= convergence["max_best_cd_improvement"] and max_pool_ei is not None
          and np.isfinite(max_pool_ei) and max_pool_ei < convergence["max_pool_ei"]):
        reason = "converged"
    return {"stop": reason is not None, "reason": reason, "successful_sequential_points": n,
            "consecutive_failed_designs": consecutive, "failure_rate": failure_rate,
            "failure_rate_window_full": len(outcomes) >= window,
            "successful_window_improvement_cd": improvement, "max_pool_ei": max_pool_ei}


def _verified_data(evaluator, state):
    expected = set(state["initial_eval_ids"]) | {item["eval_id"] for item in state["history"] if item["status"] == "success"}
    rows = evaluator.store.training_data()
    actual = {row["eval_id"] for row in rows}
    allowed_pending = {state["pending"]["eval_id"]} if state.get("pending") else set()
    if expected - actual or actual - expected - allowed_pending:
        raise RuntimeError("training ledger differs from tracked initial/sequential evaluations")
    rows = [row for row in rows if row["eval_id"] in expected]
    for row in rows:
        result = evaluator.store.get_result(row["eval_id"])
        if not result or not result.usable_for_training or not evaluator._cache_valid(result, False):
            raise RuntimeError("training solver artifacts no longer validate: " + row["eval_id"])
        selected = next(item for item in result.attempts if item["attempt_number"] == result.selected_attempt)
        parsed = parse_attempt(_runner_from_dict(selected["runner"]), evaluator.config)
        identity = digest([float(row[name]).hex() for name in ("m", "p", "t")])
        if (identity != row["design_id"] or result.design_id != identity or parsed.aerodynamics is None
                or row["cd"] != parsed.aerodynamics.cd or row["cl"] != parsed.aerodynamics.cl):
            raise RuntimeError("training labels/coordinates differ from audited solver evidence")
    return prepare_training_data(rows, evaluator.config)


def _register_model(evaluator, directory, metadata):
    evaluator.store.save_model_record({"model_id": directory.name, "config_hash": evaluator.config.config_hash,
        "directory": str(directory.relative_to(evaluator.run_dir)).replace("\\", "/"),
        "metadata": metadata, "metadata_sha256": file_sha256(directory / "metadata.json")})


def _ensure_model(evaluator, state, data, *, resume):
    successes = sum(item["status"] == "success" for item in state["history"])
    model_id = f"iter_{successes:03d}"
    directory = evaluator.run_dir / "models" / model_id
    if successes == 0:
        registered = evaluator.store.model("iter_000")
        if not registered:
            raise RuntimeError("M3 initial model registry is missing")
        model = load_model(directory, registered["metadata"], data, evaluator.config)
        loo = registered["metadata"]["loo"]
    else:
        progress = state.setdefault("model_progress", {}).setdefault(model_id, {
            "training_data_hash": data.data_hash, "loo_folds": [], "complete": False})
        if progress["training_data_hash"] != data.data_hash:
            raise RuntimeError("model progress belongs to different training data")
        registered = evaluator.store.model(model_id)
        model = None
        if registered:
            try:
                model = load_model(directory, registered["metadata"], data, evaluator.config)
                if progress.get("complete"):
                    for name, sha in progress["artifacts"].items():
                        if file_sha256(directory / name) != sha:
                            raise ValueError("model diagnostic artifact mismatch")
            except (OSError, ValueError, KeyError) as exc:
                final = evaluator.store.stage("verify")
                if final and final.get("freeze") and final["freeze"]["model"]["model_id"] == model_id:
                    raise RuntimeError("M5 final GP is frozen; cannot refit it after holdout unsealing") from exc
                if not resume:
                    raise StageIncomplete("derived model mismatch; use --resume: " + str(exc)) from exc
                backup = directory / "recovery" / uuid4().hex
                backup.mkdir(parents=True)
                for path in directory.iterdir():
                    if path.is_file() and path.suffix != ".tmp":
                        shutil.copyfile(path, backup / path.name)
                evaluator.store.event("sequential_model_rebuild", "", {"model_id": model_id, "archive": str(backup)})
                model, progress["complete"] = None, False
        if model is None:
            model = fit_gp(data, evaluator.config)
            metadata = save_model(model, directory, evaluator.config)
            _register_model(evaluator, directory, metadata)
        else:
            metadata = registered["metadata"]
        if not progress["complete"]:
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "training.json", {"config_hash": data.config_hash, "training_data_hash": data.data_hash,
                                                     "records": list(data.records), "holdout_labels_used": False})
            _write_csv(directory / "training.csv", data.records, list(data.records[0]))

            def on_fold(fold):
                progress["loo_folds"].append(fold)
                evaluator.store.save_stage("optimize", "running", state)

            loo = loo_cv(data, evaluator.config, state["baseline_cd"], completed_folds=progress["loo_folds"], on_fold=on_fold)
            write_json(directory / "loo.json", loo)
            _write_csv(directory / "loo.csv", loo["folds"], ["held_out_design_id", "held_out_eval_id", "observed_cd", "predicted_cd", "sigma_cd", "residual_cd"])
            _plots(directory, data, loo)
            metadata.update(iteration=successes, loo={key: value for key, value in loo.items() if key != "folds"})
            write_json(directory / "metadata.json", metadata)
            _register_model(evaluator, directory, metadata)
            progress.update(complete=True, artifacts={name: file_sha256(directory / name) for name in
                            ("gp.joblib", "metadata.json", "training.json", "training.csv", "loo.json", "loo.csv", "loo.png")})
        else:
            loo = metadata["loo"]
    state.update(current_model_id=model_id, training_data_hash=data.data_hash,
                 training_ids=list(data.design_ids), successful_sequential_points=successes,
                 loo_quality_passed=loo["quality_passed"], explore_next_iteration=loo["explore_next_iteration"])
    evaluator.store.save_stage("optimize", "running", state)
    return model, loo


def _best_plot(evaluator, state):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    rows = state["history"]
    values = [state["initial_best_cd"], *[row["best_cd"] for row in rows]]
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.step(np.arange(len(values)), values, where="post")
    axis.set(xlabel="Sequential design outcomes (after retry)", ylabel="Best accepted standard CD",
             title="M4 best-so-far; verification pending")
    axis.grid(alpha=.25)
    figure.savefig(evaluator.run_dir / "best-so-far.png", dpi=160)
    plt.close(figure)


def _report(evaluator, state, status, reason, *, frozen_budget=None):
    history = state["history"]
    successes = [item for item in history if item["status"] == "success"]
    best = state["initial_best"]
    for item in successes:
        if item["cd"] < best["cd"]:
            best = {"design_id": item["design_id"], "eval_id": item["eval_id"], "m": item["m"], "p": item["p"], "cd": item["cd"]}
    report = {"status": status, "stage": "optimize", "stop_reason": reason, "optimizer_version": OPTIMIZER_VERSION,
              "config_hash": evaluator.config.config_hash, "loop_updated": bool(successes),
              "successful_sequential_points": len(successes), "failed_sequential_designs": sum(row["status"] == "failed" for row in history),
              "initial_training_points": len(state["initial_eval_ids"]), "training_points": len(state["initial_eval_ids"]) + len(successes),
              "current_model_id": state.get("current_model_id", "iter_000"), "training_data_hash": state.get("training_data_hash"),
              "initial_best_cd": state["initial_best_cd"], "best_standard": best,
              "baseline_cd": state["baseline_cd"], "history": history, "pending": state.get("pending"),
              "budget_start": state["budget_start"], "budget": frozen_budget or evaluator.store.budget_snapshot(),
              "stopping": state.get("stop_diagnostics"), "holdout_labels_used": False,
              "improvement_certified": False, "scope": "Standard-fidelity sequential updates only; M5 independent verification required."}
    write_json(evaluator.run_dir / "optimization.json", report)
    _write_csv(evaluator.run_dir / "history.csv", history, ["iteration", "eval_id", "design_id", "m", "p", "status", "failure_reason",
               "cd", "best_cd", "best_design_id", "successful_sequence_number", "model_before", "model_after", "selected_attempt", "process_launches"])
    _best_plot(evaluator, state)
    lines = ["# M4 序贯优化记录", "", f"状态：{status}；停止原因：`{reason}`。",
             f"成功新增：{len(successes)}；最终失败设计：{report['failed_sequential_designs']}；可训练点：{report['training_points']}。",
             f"当前模型：`{report['current_model_id']}`；数据 SHA256：`{report['training_data_hash']}`。", "",
             "| 轮次 | m | p | 结果 | CD | best-so-far | 模型更新 |",
             "|---|---:|---:|---|---:|---:|---|"]
    for row in history:
        lines.append(f"| {row['iteration']} | {row['m']:.10g} | {row['p']:.10g} | {row['status']} | {row['cd'] if row['cd'] is not None else ''} | {row['best_cd']} | {row['model_before']}→{row.get('model_after', '')} |")
    lines += ["", f"标准面元下最佳已接受 CD={best['cd']}，m={best['m']:.12g}，p={best['p']:.12g}。",
              f"累计启动={report['budget']['process_launches']}；不同标准设计={report['budget']['distinct_standard_designs']}；最终复核预留={report['budget']['reserved_final_verification_launches']}。",
              "", "![best-so-far](best-so-far.png)", "", "失败点没有 CD 罚值；重试成功只计一个成功设计，异常点等待显式复查。",
              "封存留出标签未读取。以上为标准 N 的直接求解和代理更新记录，M5 多面元/独立验证尚未完成，未认证优化收益。"]
    (evaluator.run_dir / "optimization.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_optimize(run_dir: str | Path, *, resume: bool = False, stop_after_evaluations: int | None = None,
                 on_progress: Callable[[dict], None] | None = None) -> dict:
    if stop_after_evaluations is not None and (type(stop_after_evaluations) is not int or stop_after_evaluations <= 0):
        raise ValueError("stop_after_evaluations must be a positive integer")
    run_dir = Path(run_dir).resolve()
    # M3 owns its own lease; validate/rebuild it before obtaining the M4 lease.
    with Evaluator(run_dir) as check:
        initial_stage = check.store.stage("fit")
        if initial_stage is None or initial_stage["status"] != "success":
            raise RuntimeError("M3 initial fitting must pass before M4")
    run_fit(run_dir, resume=resume)
    with Evaluator(run_dir) as evaluator, run_ownership(evaluator.store):
        validate_stopping(evaluator.config)
        manifest = prerequisites(evaluator, "doe")
        fit_state = evaluator.store.stage("fit")
        state = evaluator.store.stage("optimize")
        if state and not resume and state["status"] not in {"success", "stopped", "skipped"}:
            raise StageIncomplete("optimization already started; use --resume")
        if state is None:
            data = prepare_training_data(evaluator.store.training_data(), evaluator.config)
            if set(data.evaluation_ids) != set(fit_state["training_eval_ids"]):
                raise RuntimeError("untracked training data exist before the first sequential proposal")
            initial_best = min(data.records, key=lambda row: (row["cd"], row["design_id"]))
            baseline = next(row for row in data.records if row["eval_id"] == manifest["baseline_eval_id"])
            state = {"optimizer_version": OPTIMIZER_VERSION, "initial_eval_ids": list(data.evaluation_ids),
                     "initial_best_cd": initial_best["cd"], "initial_best": {key: initial_best[key] for key in ("design_id", "eval_id", "m", "p", "cd")},
                     "baseline_cd": baseline["cd"], "budget_start": evaluator.store.budget_snapshot(),
                     "history": [], "pending": None, "model_progress": {}, "current_model_id": "iter_000",
                     "training_data_hash": data.data_hash, "successful_sequential_points": 0,
                     "next_iteration": evaluator.config["acquisition"]["iteration_start"],
                     "holdout_labels_used": False, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        if state["optimizer_version"] != OPTIMIZER_VERSION:
            raise RuntimeError("optimizer checkpoint version mismatch")
        pending_id = state["pending"]["eval_id"] if state.get("pending") else None
        if set(evaluator.store.unfinished()) - ({pending_id} if pending_id else set()):
            raise RuntimeError("unrelated unfinished evaluations must be recovered first")
        terminal_status = state.get("terminal_status")
        if terminal_status is None and state.get("status") in {"success", "stopped", "skipped"}:
            terminal_status = state["status"]
        if terminal_status is None and state.get("ended_at_utc") and state.get("stop_reason"):
            # Compatibility with a completed v1 checkpoint whose later artifact
            # validation failure changed only its current stage status.
            normal = state["stop_reason"] not in {"consecutive_failed_designs", "failure_rate_exceeded", "solver_or_cleanup_error"}
            terminal_status = "success" if normal and any(row["status"] == "success" for row in state["history"]) else "stopped"
        terminal = terminal_status is not None
        evaluations_this_call = 0
        try:
            if terminal:
                state["terminal_status"] = terminal_status
                valid_receipt = True
                try:
                    verify_receipt(evaluator.run_dir, state)
                except (OSError, RuntimeError):
                    valid_receipt = False
                    if not resume:
                        raise StageIncomplete("optimization artifacts changed; use --resume to rebuild projections")
                original_report = json.loads((evaluator.run_dir / "optimization.json").read_text(encoding="utf-8")) if valid_receipt else None
                if "terminal_budget" not in state and original_report:
                    state["terminal_budget"] = original_report["budget"]
                data = _verified_data(evaluator, state)
                _ensure_model(evaluator, state, data, resume=resume)
                report = original_report if valid_receipt else _report(evaluator, state, terminal_status, state["stop_reason"],
                                                                       frozen_budget=state.get("terminal_budget"))
                state["artifacts"] = receipt(evaluator.run_dir, ["optimization.json", "optimization.md", "history.csv", "best-so-far.png"])
                evaluator.store.save_stage("optimize", terminal_status, state)
                if manifest.get("stage") == "optimize" and manifest.get("status") != terminal_status:
                    update_manifest(evaluator, "optimize", terminal_status, m4_passed=terminal_status == "success",
                                    loop_updated=report["loop_updated"], model_id=state["current_model_id"],
                                    ended_at_utc=state["ended_at_utc"])
                return report
            evaluator.store.save_stage("optimize", "running", state)
            update_manifest(evaluator, "optimize", "running")
            while True:
                if state.get("pending"):
                    pending = state["pending"]
                    data_before = _verified_data(evaluator, state)
                    _validate_proposal(pending["proposal"], {"excluded": pending["excluded"]}, data_before, evaluator.config)
                    path = evaluator.run_dir / pending["proposal_file"]
                    if path.parent != evaluator.run_dir / "proposals":
                        raise RuntimeError("pending proposal path is outside proposals/")
                    try:
                        projected = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        projected = None
                    if projected != pending:
                        if path.is_file():
                            shutil.copyfile(path, path.with_name(path.name + ".invalid-" + uuid4().hex))
                        path.parent.mkdir(parents=True, exist_ok=True)
                        write_json(path, pending)
                        evaluator.store.event("proposal_projection_restored", pending["eval_id"], {"proposal_file": pending["proposal_file"]})
                    # Fresh exclusions may add unrelated designs; allow only this pending design's own reservation.
                    point_id = digest([float(pending["proposal"][key]).hex() for key in ("m", "p", "t")])
                    others = [[row["u_m"], row["u_p"]] for row in evaluator.store.exclusion_coordinates() if row["design_id"] != point_id]
                    if nearest_distances([pending["proposal"]["u"]], others)[0] < evaluator.config["acquisition"]["min_normalized_distance"]:
                        raise RuntimeError("pending proposal became an excluded/too-close design")
                    result = evaluator.evaluate(pending["proposal"]["design"], purpose="sequential",
                                                evaluation_id=pending["eval_id"], resume=resume)
                    if result.eval_id != pending["eval_id"] or result.purpose != "sequential":
                        raise RuntimeError("a cached external design cannot count as a new sequential point")
                    if result.status == "needs_review":
                        raise StageIncomplete("explicit review required for " + result.eval_id, "needs_review")
                    if result.status not in {"success", "failed", "skipped"}:
                        raise StageIncomplete("sequential evaluation is incomplete")
                    previous_best = state["history"][-1]["best_cd"] if state["history"] else state["initial_best_cd"]
                    best_id = state["history"][-1]["best_design_id"] if state["history"] else state["initial_best"]["design_id"]
                    cd = result.aerodynamics["cd"] if result.usable_for_training else None
                    if result.status == "success" and cd is None:
                        raise RuntimeError("sequential success is not valid standard training data")
                    if cd is not None and cd < previous_best:
                        previous_best, best_id = cd, result.design_id
                    successes = sum(row["status"] == "success" for row in state["history"]) + int(result.status == "success")
                    row = {"iteration": pending["iteration"], "eval_id": result.eval_id, "design_id": result.design_id,
                           "m": pending["proposal"]["m"], "p": pending["proposal"]["p"], "status": result.status,
                           "failure_reason": result.failure_reason, "cd": cd, "best_cd": previous_best, "best_design_id": best_id,
                           "successful_sequence_number": successes if result.status == "success" else None,
                           "model_before": pending["proposal"]["model_id"], "model_after": f"iter_{successes:03d}",
                           "selected_attempt": result.selected_attempt, "attempt_count": len(result.attempts),
                           "process_launches": evaluator.store.budget_snapshot()["process_launches"],
                           "proposal_file": pending["proposal_file"]}
                    state["history"].append(row)
                    state.update(pending=None, next_iteration=pending["iteration"] + 1)
                    systemic = {"launch_failed", "cleanup_failed", "solver_identity_mismatch", "solver_setting_mismatch", "aborted"}
                    if any(item["runner"].get("status") in systemic for item in result.attempts):
                        state["forced_stop"] = "solver_or_cleanup_error"
                    elif result.status == "skipped":
                        state["forced_stop"] = "evaluation_budget_skipped"
                    evaluator.store.save_stage("optimize", "running", state)
                    evaluator.store.event("sequential_outcome_committed", result.eval_id, row)
                    evaluator.store.export_csv(include_holdout=False)
                    evaluations_this_call += 1
                    if on_progress:
                        on_progress({"iteration": row["iteration"], "status": row["status"], "cd": row["cd"],
                                     "successful_points": successes, "best_cd": row["best_cd"], "process_launches": row["process_launches"]})
                data = _verified_data(evaluator, state)
                model, loo = _ensure_model(evaluator, state, data, resume=resume)
                if stop_after_evaluations is not None and evaluations_this_call >= stop_after_evaluations:
                    raise StageIncomplete("requested sequential recovery diagnostic boundary reached")
                decision = stopping_decision(state["history"], evaluator.store.budget_snapshot(), evaluator.config,
                                             state["initial_best_cd"], None)
                if state.get("forced_stop"):
                    decision.update(stop=True, reason=state["forced_stop"])
                if not decision["stop"]:
                    iteration = state["next_iteration"]
                    excluded = [[row["u_m"], row["u_p"]] for row in evaluator.store.exclusion_coordinates()]
                    if not state["history"] and iteration == evaluator.config["acquisition"]["iteration_start"] and nearest_distances([fit_state["proposal"]["u"]], excluded)[0] >= evaluator.config["acquisition"]["min_normalized_distance"]:
                        proposal = dict(fit_state["proposal"])
                    else:
                        proposal = propose(model, evaluator.config, excluded, f_best=float(data.y.min()), iteration=iteration,
                            successful_sequential_points=decision["successful_sequential_points"], poor_loo=loo["explore_next_iteration"])
                        proposal.update(model_id=state["current_model_id"], training_data_hash=data.data_hash, config_hash=data.config_hash)
                    decision = stopping_decision(state["history"], evaluator.store.budget_snapshot(), evaluator.config,
                                                 state["initial_best_cd"], proposal["max_pool_ei"])
                state["stop_diagnostics"] = decision
                if decision["stop"]:
                    reason = decision["reason"]
                    normal = reason not in {"consecutive_failed_designs", "failure_rate_exceeded", "solver_or_cleanup_error"}
                    status = "success" if normal and decision["successful_sequential_points"] > 0 else "stopped"
                    state.update(stop_reason=reason, ended_at_utc=datetime.now(timezone.utc).isoformat())
                    state["terminal_budget"] = evaluator.store.budget_snapshot()
                    state["terminal_status"] = status
                    report = _report(evaluator, state, status, reason)
                    state["artifacts"] = receipt(evaluator.run_dir, ["optimization.json", "optimization.md", "history.csv", "best-so-far.png"])
                    evaluator.store.save_stage("optimize", status, state)
                    update_manifest(evaluator, "optimize", status, m4_passed=status == "success", loop_updated=report["loop_updated"],
                                    model_id=state["current_model_id"], ended_at_utc=state["ended_at_utc"])
                    return report
                _validate_proposal(proposal, {"excluded": excluded}, data, evaluator.config)
                proposal_file = f"proposals/iter_{iteration:03d}.json"
                state["pending"] = {"iteration": iteration, "eval_id": "eval_" + uuid4().hex,
                                    "proposal": proposal, "excluded": excluded, "proposal_file": proposal_file}
                # A fixed proposal/seed/eval ID commit precedes every possible solver launch.
                evaluator.store.save_stage("optimize", "running", state)
                path = evaluator.run_dir / proposal_file
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, state["pending"])
                evaluator.store.event("sequential_proposal_committed", state["pending"]["eval_id"], {
                    "iteration": iteration, "design": proposal["design"], "seed": proposal["seed"], "model_id": proposal["model_id"]})
        except BaseException as exc:
            if isinstance(exc, SurrogateFitError):
                state["fit_failure_attempts"] = exc.attempts
            checkpoint_failure(evaluator, "optimize", state, exc, export_datasets=False)
            _report(evaluator, state, getattr(exc, "status", "interrupted"), str(exc))
            raise
