"""M3 initial training/model/LOO/proposal stage with resumable derived artifacts."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from .acquisition import ACQUISITION_VERSION, propose
from .config import file_sha256, write_json
from .evaluator import Evaluator, _runner_from_dict
from .polar import parse_attempt
from .sampling import nearest_distances
from .stages import (StageIncomplete, checkpoint_failure, prerequisites, receipt,
                     run_ownership, update_manifest, verify_receipt)
from .surrogate import (SURROGATE_VERSION, SurrogateFitError, digest, fit_gp, load_model,
                        loo_cv, prepare_training_data, save_model)


def _initial_training(evaluator, manifest):
    # Initial model remains tied to its DOE IDs even after M4 eventually adds points.
    doe = evaluator.store.stage("doe")
    if doe is None or doe["status"] != "success":
        raise RuntimeError("M2 DOE must pass before M3 fitting")
    verify_receipt(evaluator.run_dir, doe)
    expected_ids = {manifest["baseline_eval_id"], *[slot["eval_id"] for slot in doe["slots"]
                    if slot["split"] == "train" and slot["status"] == "success"]}
    records = [row for row in evaluator.store.training_data() if row["eval_id"] in expected_ids]
    if len(records) != evaluator.config["doe"]["initial_points"] + 1 or len(records) != len(expected_ids):
        raise RuntimeError("initial model requires the accepted baseline plus all DOE training slots")
    for row in records:
        result = evaluator.store.get_result(row["eval_id"])
        if not result or not result.usable_for_training or not evaluator._cache_valid(result, False):
            raise RuntimeError("training solver artifacts no longer validate: " + row["eval_id"])
        identity = digest([float(row[key]).hex() for key in ("m", "p", "t")])
        selected = next(item for item in result.attempts if item["attempt_number"] == result.selected_attempt)
        parsed = parse_attempt(_runner_from_dict(selected["runner"]), evaluator.config)
        if (identity != row["design_id"] or result.design_id != identity or parsed.aerodynamics is None
                or row["cd"] != parsed.aerodynamics.cd or row["cl"] != parsed.aerodynamics.cl):
            raise RuntimeError("training coordinates/labels differ from audited solver identity")
    baseline = next(row for row in records if row["eval_id"] == manifest["baseline_eval_id"])
    return prepare_training_data(records, evaluator.config), baseline["cd"]


def _write_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(stream.getvalue(), encoding="utf-8", newline="")
    temporary.replace(path)


def _register(evaluator, directory, metadata):
    evaluator.store.save_model_record({"model_id": "iter_000", "config_hash": evaluator.config.config_hash,
        "directory": "models/iter_000", "metadata": metadata, "metadata_sha256": file_sha256(directory / "metadata.json")})


def _archive_derived(evaluator, directory, state):
    backup = directory / "recovery" / uuid4().hex
    backup.mkdir(parents=True)
    # Preserve corrupt/incompatible derived files for audit; raw solver data never changes.
    names = ("gp.joblib", "metadata.json", "training.json", "training.csv", "loo.json", "loo.csv", "loo.png", "proposal.json")
    for name in names:
        path = directory / name
        if path.is_file():
            shutil.copyfile(path, backup / name)
    state["rebuilds"] = state.get("rebuilds", 0) + 1
    state["model_rebuild_pending"] = True
    state.pop("artifacts", None)
    state.pop("proposal", None)
    evaluator.store.event("model_rebuild_requested", "", {"model_id": "iter_000", "archive": str(backup)})


def _validate_proposal(proposal, state, data, config):
    from .geometry import generate_from_config
    from .sampling import normalize
    if (proposal["assessed"] is not False or proposal["status"] != "proposed" or proposal["holdout_labels_used"] is not False
            or proposal["acquisition_version"] != ACQUISITION_VERSION or proposal["training_data_hash"] != data.data_hash):
        raise ValueError("proposal identity differs from training/acquisition state")
    point = normalize([proposal["design"]], config["design"]["bounds"])[0]
    if not np.allclose(point, proposal["u"], rtol=0, atol=1e-14):
        raise ValueError("proposal does not use fixed-bound coordinates")
    distance = nearest_distances([point], state["excluded"])[0]
    if distance < config["acquisition"]["min_normalized_distance"]:
        raise ValueError("proposal repeats an excluded point")
    generated = generate_from_config(config, proposal["design"])
    if generated.geometry_hash != proposal["geometry_hash"] or generated.file_sha256 != proposal["geometry_file_sha256"]:
        raise ValueError("proposal geometry identity mismatch")


def _plots(directory, data, loo):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    observed = np.asarray([fold["observed_cd"] for fold in loo["folds"]])
    predicted = np.asarray([fold["predicted_cd"] for fold in loo["folds"]])
    residual = predicted - observed
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].scatter(observed, predicted)
    lower, upper = float(min(observed.min(), predicted.min())), float(max(observed.max(), predicted.max()))
    axes[0].plot([lower, upper], [lower, upper], "--", color="gray")
    axes[0].set(xlabel="Observed XFOIL CD", ylabel="LOO predicted CD", title="Independent GP refit in each fold")
    axes[1].scatter(np.arange(1, len(data.records) + 1), residual)
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set(xlabel="Training design (stable ID order)", ylabel="Predicted - observed CD", title="LOO residuals in original CD units")
    for axis in axes:
        axis.grid(alpha=.25)
    figure.savefig(directory / "loo.png", dpi=160)
    plt.close(figure)


def _report(evaluator, directory, data, metadata, loo, proposal, before):
    after = evaluator.store.budget_snapshot()
    if after != before:
        raise RuntimeError("M3 must not spend solver budget")
    report = {"status": "success", "stage": "fit", "model_id": "iter_000",
              "config_hash": evaluator.config.config_hash, "training_data_hash": data.data_hash,
              "training_points": len(data.records), "training_ids": list(data.design_ids),
              "holdout_labels_used": False, "fit": metadata["fit"],
              "loo": {key: value for key, value in loo.items() if key != "folds"},
              "proposal": proposal, "budget_before": before, "budget_after": after,
              "new_solver_launches": 0, "scope": "Initial surrogate and one unassessed proposal; no sequential update or improvement certification."}
    write_json(evaluator.run_dir / "fit.json", report)
    lines = ["# M3 初始 GP / LOO / 候选提议", "", f"成功训练点：{len(data.records)}；封存留出标签未读取。", "",
             f"训练快照 SHA256：`{data.data_hash}`。", f"核：`{metadata['fit']['kernel']}`。",
             f"标准化输出正则 alpha：{metadata['fit']['alpha_regularization']:.8g}。", "",
             "| LOO 指标（CD 原尺度） | 数值 |", "|---|---:|",
             f"| MAE | {loo['mae_cd']:.8g} |", f"| RMSE | {loo['rmse_cd']:.8g} |",
             f"| 最大绝对误差 | {loo['max_abs_error_cd']:.8g} |", f"| MAE 门槛 | {loo['threshold_cd']:.8g} |", "",
             f"LOO 门槛通过：{loo['quality_passed']}；下一轮探索标志：{loo['explore_next_iteration']}。",
             "每个折都重新拟合核与输出标准化；折内训练 ID、警告和正则尝试见 `models/iter_000/loo.json`。", "",
             "![LOO 预测与残差](models/iter_000/loo.png)", "", "## 未评估候选", "",
             f"m={proposal['m']:.12g}，p={proposal['p']:.12g}，t={proposal['t']:.12g}。",
             f"选择原因：`{proposal['reason']}`；来源：`{proposal['source']}`。",
             f"预测 CD={proposal['mu_cd']}，预测标准差={proposal['sigma_cd']}，EI={proposal['ei']}。",
             f"最小排除距离={proposal['min_excluded_distance']:.8g}；边界命中={proposal['boundary_hit']}。", "",
             "这些数值是代理预测；候选尚未运行 XFOIL，也没有写入训练样本。",
             "本阶段新增 XFOIL 启动为 0；拟合警告记录在模型元数据中，核边界命中不会被隐藏。",
             "M4 序贯闭环及 M5 独立验证/收益复核尚未执行。"]
    (evaluator.run_dir / "fit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_fit(run_dir: str | Path, *, resume: bool = False, stop_after_folds: int | None = None) -> dict:
    """Fit iter_000 and propose iter_001. Fold-boundary stop is a recovery diagnostic."""
    if stop_after_folds is not None and (type(stop_after_folds) is not int or stop_after_folds <= 0):
        raise ValueError("stop_after_folds must be a positive integer")
    with Evaluator(run_dir) as evaluator, run_ownership(evaluator.store):
        manifest = prerequisites(evaluator, "doe")
        if evaluator.store.unfinished():
            raise RuntimeError("unfinished solver evaluations must be recovered before M3")
        data, baseline_cd = _initial_training(evaluator, manifest)
        directory = evaluator.run_dir / "models/iter_000"
        state = evaluator.store.stage("fit")
        before = evaluator.store.budget_snapshot()
        if state and state["training_data_hash"] != data.data_hash:
            raise RuntimeError("initial model training snapshot changed; restore verified data before resuming")
        if state and state["status"] == "success":
            try:
                verify_receipt(evaluator.run_dir, state)
                record = evaluator.store.model("iter_000")
                if not record:
                    raise ValueError("completed stage has no model registry entry")
                load_model(directory, record["metadata"], data, evaluator.config)
                _validate_proposal(state["proposal"], state, data, evaluator.config)
                evaluator.store.write_checkpoint()
                return json.loads((evaluator.run_dir / "fit.json").read_text(encoding="utf-8"))
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                if not resume:
                    raise StageIncomplete("model/artifact mismatch; use --resume to rebuild verified training data: " + str(exc)) from exc
                _archive_derived(evaluator, directory, state)
        elif state and not resume:
            raise StageIncomplete("initial fitting already started; use --resume")
        if state is None:
            state = {"model_id": "iter_000", "training_data_hash": data.data_hash,
                     "training_ids": list(data.design_ids), "training_eval_ids": list(data.evaluation_ids),
                     "excluded": [[row["u_m"], row["u_p"]] for row in evaluator.store.exclusion_coordinates()],
                     "loo_folds": [], "iteration": evaluator.config["acquisition"]["iteration_start"],
                     "seed": evaluator.config["acquisition"]["seed_base"] + evaluator.config["acquisition"]["iteration_start"],
                     "holdout_labels_used": False, "surrogate_version": SURROGATE_VERSION, "acquisition_version": ACQUISITION_VERSION}
        try:
            evaluator.store.save_stage("fit", "running", state)
            state["surrogate_version"] = SURROGATE_VERSION
            state["acquisition_version"] = ACQUISITION_VERSION
            update_manifest(evaluator, "fit", "running")
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "training.json", {"config_hash": data.config_hash, "training_data_hash": data.data_hash,
                                                      "records": list(data.records), "holdout_labels_used": False})
            _write_csv(directory / "training.csv", data.records, list(data.records[0]))
            registered = evaluator.store.model("iter_000")
            model = None
            if registered and not state.get("model_rebuild_pending"):
                try:
                    model = load_model(directory, registered["metadata"], data, evaluator.config)
                except (OSError, ValueError, KeyError) as exc:
                    if not resume:
                        raise StageIncomplete("model artifact mismatch; use --resume") from exc
                    _archive_derived(evaluator, directory, state)
            if model is None:
                model = fit_gp(data, evaluator.config)
                metadata = save_model(model, directory, evaluator.config)
                _register(evaluator, directory, metadata)
                state["model_rebuild_pending"] = False
                evaluator.store.save_stage("fit", "running", state)
            else:
                metadata = registered["metadata"]
            folds_this_call = 0

            def on_fold(fold):
                nonlocal folds_this_call
                state["loo_folds"].append(fold)
                evaluator.store.save_stage("fit", "running", state)
                folds_this_call += 1
                if stop_after_folds is not None and folds_this_call >= stop_after_folds:
                    raise StageIncomplete("requested LOO recovery diagnostic boundary reached")

            loo = loo_cv(data, evaluator.config, baseline_cd, completed_folds=state["loo_folds"], on_fold=on_fold)
            write_json(directory / "loo.json", loo)
            _write_csv(directory / "loo.csv", loo["folds"], ["held_out_design_id", "held_out_eval_id", "observed_cd", "predicted_cd", "sigma_cd", "residual_cd"])
            _plots(directory, data, loo)
            # Proposal is committed in SQLite before derived files; a resume reuses it.
            if "proposal" not in state:
                proposal = propose(model, evaluator.config, state["excluded"], f_best=float(np.min(data.y)),
                                   iteration=state["iteration"], poor_loo=loo["explore_next_iteration"])
                proposal.update(model_id="iter_000", training_data_hash=data.data_hash, config_hash=data.config_hash)
                state["proposal"] = proposal
                evaluator.store.save_stage("fit", "running", state)
                evaluator.store.event("candidate_proposed", "", {"model_id": "iter_000", "iteration": proposal["iteration"],
                                      "design": proposal["design"], "reason": proposal["reason"], "assessed": False})
            proposal = state["proposal"]
            _validate_proposal(proposal, state, data, evaluator.config)
            write_json(directory / "proposal.json", proposal)
            metadata.update(iteration=0, loo={key: value for key, value in loo.items() if key != "folds"}, proposal_file="proposal.json")
            write_json(directory / "metadata.json", metadata)
            _register(evaluator, directory, metadata)
            report = _report(evaluator, directory, data, metadata, loo, proposal, before)
            names = ["fit.json", "fit.md", *["models/iter_000/" + name for name in
                     ("gp.joblib", "metadata.json", "training.json", "training.csv", "loo.json", "loo.csv", "loo.png", "proposal.json")]]
            state["artifacts"] = receipt(evaluator.run_dir, names)
            state["loo_quality_passed"] = loo["quality_passed"]
            state["explore_next_iteration"] = loo["explore_next_iteration"]
            evaluator.store.save_stage("fit", "success", state)
            update_manifest(evaluator, "fit", "success", initial_model_passed=True, m3_passed=True,
                            loo_quality_passed=loo["quality_passed"], model_id="iter_000")
            return report
        except BaseException as exc:
            if isinstance(exc, SurrogateFitError):
                state["fit_failure_attempts"] = exc.attempts
            checkpoint_failure(evaluator, "fit", state, exc, export_datasets=False)
            raise
