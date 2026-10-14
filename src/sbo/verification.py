"""M5: immutable final model/candidates, sealed holdout and independent solves.

Verification never fits a model or appends training labels. Each slot has a new
evaluation ID committed before launch; only explicit recovery reuses that ID.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .baseline import pressure_diagnostics
from .config import file_sha256, write_json
from .evaluator import Evaluator, _runner_from_dict
from .geometry import generate_from_config
from .optimizer import _verified_data
from .polar import parse_attempt
from .stages import (StageIncomplete, checkpoint_failure, prerequisites, receipt,
                     run_ownership, update_manifest, verify_receipt)
from .surrogate import digest, load_model

VERIFICATION_VERSION = "independent-verification-v1"
SLOTS = (("160", 160), ("200", 200), ("240", 240), ("200_repeat", 200))


def holdout_metrics(observed, predicted, sigma, config, baseline_cd: float) -> dict:
    y, mean, std = (np.asarray(value, dtype=float) for value in (observed, predicted, sigma))
    if (y.ndim != 1 or not len(y) or mean.shape != y.shape or std.shape != y.shape
            or not all(np.isfinite(value).all() for value in (y, mean, std))
            or np.any(y <= 0) or np.any(std < 0)):
        raise ValueError("holdout metrics require aligned finite CD predictions and nonnegative sigma")
    residual = y - mean
    variance = float(np.mean((y - y.mean()) ** 2))
    # XFOIL's saved CD resolution is 1e-5. R2 below that variance scale is unstable.
    floor = config["verification"]["numerical_uncertainty_floor"] ** 2
    quality = config["quality"]["holdout"]
    threshold = max(quality["absolute_cd_threshold"], quality["relative_baseline_cd_threshold"] * baseline_cd)
    mae = float(np.mean(np.abs(residual)))
    interval = config["verification"]["holdout_interval_sigma"] * std
    roundoff = 16 * np.finfo(float).eps * np.maximum(np.abs(y), np.abs(mean))
    return {"count": len(y), "mae": mae, "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "max_absolute_error": float(np.max(np.abs(residual))),
            "r2": float(1 - np.mean(residual ** 2) / variance) if len(y) >= 2 and variance > floor else None,
            "r2_not_applicable_reason": None if len(y) >= 2 and variance > floor else "too_few_points_or_target_variance_at_resolution_floor",
            "target_variance": variance, "r2_variance_floor": floor,
            "two_sigma_coverage": float(np.mean(np.abs(residual) <= interval + roundoff)),
            "interval_sigma": config["verification"]["holdout_interval_sigma"], "coverage_is_diagnostic_only": True,
            "coverage_roundoff_policy": "16_machine_epsilon_times_CD_magnitude",
            "mae_threshold": threshold, "quality_passed": mae <= threshold,
            "used_for_training_or_candidate_selection": False}


def shape_quality(aerodynamics: dict, pressure: dict, config) -> dict:
    valid = all(aerodynamics.get(name) is not None for name, _ in SLOTS)
    if not valid:
        return {"passed": False, "all_target_points_valid": False, "epsilon": None,
                "panel_convergence": False, "repeatability": False, "transition_cp_screening": False}
    cd = {name: aerodynamics[name]["cd"] for name, _ in SLOTS}
    if any(not np.isfinite(value) or value <= 0 for value in cd.values()):
        raise ValueError("invalid CD in verification quality calculation")
    panel = abs(cd["200"] - cd["240"])
    repeat = abs(cd["200"] - cd["200_repeat"])
    quality = config["quality"]["panel_convergence"]
    threshold = max(quality["absolute_cd_threshold"], quality["relative_cd_threshold"] * cd["240"])
    components = [abs(cd["160"] - cd["200"]), panel, repeat, config["verification"]["numerical_uncertainty_floor"]]
    roundoff = float(16 * np.finfo(float).eps * max(cd.values()))
    checks = {"all_target_points_valid": True, "panel_convergence": panel <= threshold + roundoff,
              "repeatability": repeat <= config["verification"]["max_repeat_cd_difference"] + roundoff,
              "transition_cp_screening": all(pressure.get(name, {}).get("passed", False) for name, _ in SLOTS)}
    return {**checks, "passed": all(checks.values()), "panel_cd_difference": panel,
            "panel_threshold": threshold, "repeat_cd_difference": repeat,
            "repeat_threshold": config["verification"]["max_repeat_cd_difference"],
            "epsilon": max(components), "epsilon_components": components, "gate_roundoff_tolerance_cd": roundoff,
            "epsilon_is_statistical_confidence_interval": False}


def certify(shapes: list[dict]) -> dict:
    baseline = next(shape for shape in shapes if shape["role"] == "baseline")
    candidates = [shape for shape in shapes if shape["role"] != "baseline" and shape["quality"]["passed"]]
    candidates.sort(key=lambda shape: (shape["aerodynamics"]["240"]["cd"], shape["design_id"]))
    comparisons = []
    if baseline["quality"]["passed"]:
        for rank, shape in enumerate(candidates, 1):
            delta = baseline["aerodynamics"]["240"]["cd"] - shape["aerodynamics"]["240"]["cd"]
            scale = baseline["quality"]["epsilon"] + shape["quality"]["epsilon"]
            roundoff = float(16 * np.finfo(float).eps * max(baseline["aerodynamics"]["240"]["cd"], shape["aerodynamics"]["240"]["cd"]))
            comparisons.append({"design_id": shape["design_id"], "rank_240": rank,
                "delta_cd_240": delta, "uncertainty_sum": scale, "certified": delta > scale + roundoff,
                "strict_gain_roundoff_tolerance_cd": roundoff,
                "relative_improvement_240": delta / baseline["aerodynamics"]["240"]["cd"],
                "delta_cd_160": baseline["aerodynamics"]["160"]["cd"] - shape["aerodynamics"]["160"]["cd"]})
        # Rank is frozen by CD_240; an alternative can qualify with a smaller epsilon.
        accepted = next((row for row in comparisons if row["certified"]), None)
        recommendation = accepted["design_id"] if accepted else baseline["design_id"]
        reason = "certified_improvement" if accepted else "retain_baseline_no_certified_improvement"
    else:
        recommendation, reason, accepted = None, "baseline_verification_failed", None
    return {"comparisons": comparisons, "ranked_valid_candidate_ids": [shape["design_id"] for shape in candidates],
            "recommendation_design_id": recommendation, "recommendation_reason": reason,
            "improvement_certified": accepted is not None, "baseline_passed": baseline["quality"]["passed"]}


def _policy(config):
    v = config["verification"]
    required = {"candidates_from": "successful_nonbaseline_training_designs", "preliminary_rank_by": "standard_cd",
                "include_baseline": True, "bypass_cache": True, "save_cp": True, "save_transition_locations": True,
                "require_all_target_points_valid": True, "apply_panel_convergence_quality_gate": True,
                "uncertainty_combination": "max", "final_rank_n": 240, "repeat_n": 200,
                "certify_improvement_above": "sum_of_baseline_and_candidate_uncertainties",
                "report_standard_n_improvement": True, "recommendation_without_certified_improvement": "baseline",
                "recommendation_on_baseline_verification_failure": "none", "allow_verification_data_in_training": False}
    if (any(v[key] != value for key, value in required.items()) or tuple(v["panel_counts"]) != (160, 200, 240)
            or tuple(v["numerical_uncertainty_components"]) != ("abs_cd_160_minus_200", "abs_cd_200_minus_240", "abs_cd_200_repeat_difference")
            or tuple(v["holdout_metrics"]) != ("mae", "rmse", "max_absolute_error", "r2", "two_sigma_coverage")
            or config["quality"]["holdout"]["use_for_training_or_candidate_selection"]
            or config["quality"]["holdout"]["on_exceedance"] != "report_surrogate_quality_failure"):
        raise ValueError("unsupported M5 verification policy")


def _context(evaluator):
    manifest = prerequisites(evaluator, "doe")
    optimization = evaluator.store.stage("optimize")
    if (not optimization or optimization.get("terminal_status", optimization["status"]) != "success"
            or not manifest.get("m4_passed") or optimization.get("pending")):
        raise RuntimeError("M4 must finish successfully before final model/candidate freeze")
    verify_receipt(evaluator.run_dir, optimization)
    data = _verified_data(evaluator, optimization)
    model_id = optimization["current_model_id"]
    registered = evaluator.store.model(model_id)
    if not registered or registered["metadata"]["training_data_hash"] != data.data_hash:
        raise RuntimeError("final GP registry/data identity mismatch")
    directory = (evaluator.run_dir / registered["directory"]).resolve()
    if not directory.is_relative_to(evaluator.run_dir) or file_sha256(directory / "metadata.json") != registered["metadata_sha256"]:
        raise RuntimeError("final model metadata identity mismatch")
    model = load_model(directory, registered["metadata"], data, evaluator.config)
    baseline = next(row for row in data.records if row["eval_id"] == manifest["baseline_eval_id"])
    ranked = sorted((row for row in data.records if row["design_id"] != baseline["design_id"]),
                    key=lambda row: (row["cd"], row["design_id"]))
    count = evaluator.config["verification"]["candidate_count"]
    if len(ranked) < count:
        raise RuntimeError("insufficient distinct nonbaseline accepted training candidates")
    shapes = [{**{key: row[key] for key in ("design_id", "eval_id", "m", "p", "t", "cd")},
               "role": "baseline" if i == 0 else f"candidate_{i}"} for i, row in enumerate([baseline, *ranked[:count]])]
    freeze = {"verification_version": VERIFICATION_VERSION, "config_hash": evaluator.config.config_hash,
              "model": registered, "training_data_hash": data.data_hash, "training_ids": list(data.design_ids),
              "training_eval_ids": list(data.evaluation_ids), "shapes": shapes,
              "holdout_coordinates": evaluator.store.holdout_coordinates()}
    return freeze, model


def _audit(evaluator, result, design, panel_count, *, pressure):
    identity = digest([float(value).hex() for value in (*design, evaluator.config["design"]["t"])])
    geometry = generate_from_config(evaluator.config, design)
    if (result is None or result.status != "success" or result.design_id != identity
            or result.config_hash != evaluator.config.config_hash or result.fidelity["panel_count"] != panel_count
            or result.geometry_file_sha256 != geometry.file_sha256 or not evaluator._cache_valid(result, pressure)):
        raise RuntimeError("final verification source evidence/coordinate binding no longer validates")
    selected = next(item for item in result.attempts if item["attempt_number"] == result.selected_attempt)
    parsed = parse_attempt(_runner_from_dict(selected["runner"]), evaluator.config)
    if parsed.aerodynamics is None or any(result.aerodynamics[key] != getattr(parsed.aerodynamics, key) for key in result.aerodynamics):
        raise RuntimeError("aerodynamic labels differ from raw target-point evidence")


def _validate_slots(state):
    expected = {(shape["design_id"], name, panel) for shape in state["freeze"]["shapes"] for name, panel in SLOTS}
    slots = state["slots"]
    if (len(slots) != len(expected) or len({slot["eval_id"] for slot in slots}) != len(slots)
            or {(slot["design_id"], slot["name"], slot["panel_count"]) for slot in slots} != expected):
        raise RuntimeError("independent verification slot identities/panels are duplicated or changed")


def _holdout(evaluator, state, model):
    rows = evaluator.store.final_holdout_data(state["freeze_hash"])
    coordinates = state["freeze"]["holdout_coordinates"]
    doe = evaluator.store.stage("doe")
    expected = {slot["eval_id"] for slot in doe["slots"] if slot["split"] == "holdout" and slot["status"] == "success"}
    if (len(rows) != evaluator.config["doe"]["holdout_points"] or {row["eval_id"] for row in rows} != expected
            or {row["design_id"] for row in rows} != {row["design_id"] for row in coordinates}):
        raise RuntimeError("independent holdout labels differ from frozen DOE split")
    for row in rows:
        result = evaluator.store.get_result(row["eval_id"])
        if result.split != "holdout" or result.purpose != "doe":
            raise RuntimeError("invalid holdout evaluation identity")
        _audit(evaluator, result, [row["m"], row["p"]], evaluator.config["xfoil"]["standard_n"], pressure=False)
        if row["cd"] != result.aerodynamics["cd"]:
            raise RuntimeError("holdout dataset CD differs from audited result")
    mean, std = model.predict([[row["m"], row["p"]] for row in rows])
    predictions = [{"design_id": row["design_id"], "eval_id": row["eval_id"], "m": row["m"], "p": row["p"],
                    "observed_cd": row["cd"], "predicted_cd": float(mu), "sigma_cd": float(sigma),
                    "residual_cd": float(row["cd"] - mu)} for row, mu, sigma in zip(rows, mean, std)]
    return {"metrics": holdout_metrics([row["cd"] for row in rows], mean, std, evaluator.config,
                                      state["freeze"]["shapes"][0]["cd"]), "predictions": predictions,
            "freeze_hash": state["freeze_hash"], "model_id": state["freeze"]["model"]["model_id"]}


def verification_summary(evaluator, state) -> dict:
    _validate_slots(state)
    shapes = []
    for shape in state["freeze"]["shapes"]:
        aero, pressure, evaluations = {}, {}, {}
        for name, panel in SLOTS:
            slot = next(item for item in state["slots"] if item["design_id"] == shape["design_id"] and item["name"] == name)
            result = evaluator.store.get_result(slot["eval_id"])
            evaluations[name] = {"eval_id": slot["eval_id"], "status": result.status if result else "not_run",
                                 "failure_reason": result.failure_reason if result else None, "panel_count": panel}
            aero[name] = None
            if result and result.status == "success":
                if result.purpose != "verification" or result.split != "train":
                    raise RuntimeError("independent slot reused a nonverification result")
                request = evaluator.store.get_request(result.eval_id)
                if (not request["save_cp"] or not request["use_reserved_budget"]
                        or datetime.fromisoformat(result.started_at_utc) < datetime.fromisoformat(state["started_at_utc"])):
                    raise RuntimeError("independent slot reused a pre-freeze/cache evaluation")
                _audit(evaluator, result, [shape["m"], shape["p"]], panel, pressure=True)
                aero[name] = result.aerodynamics
                pressure[name], _ = pressure_diagnostics(result)
        shapes.append({**shape, "standard_cd": shape["cd"], "evaluations": evaluations,
                       "aerodynamics": aero, "pressure": pressure, "quality": shape_quality(aero, pressure, evaluator.config)})
    certification = certify(shapes)
    checks = {"independent_holdout": state["holdout"]["metrics"]["quality_passed"],
              "all_shapes_verified": all(shape["quality"]["passed"] for shape in shapes)}
    return {"stage": "verify", "status": "success" if all(checks.values()) else "failed",
            "verification_version": VERIFICATION_VERSION, "config_hash": evaluator.config.config_hash,
            "freeze_hash": state["freeze_hash"], "model_id": state["freeze"]["model"]["model_id"],
            "model_sha256": state["freeze"]["model"]["metadata"]["artifact"]["sha256"],
            "training_data_hash": state["freeze"]["training_data_hash"], "training_points": len(state["freeze"]["training_ids"]),
            "checks": checks, "holdout": state["holdout"], "shapes": shapes, "certification": certification,
            "budget_start": state["budget_start"], "budget": state.get("terminal_budget", evaluator.store.budget_snapshot()),
            "verification_labels_used_for_training": False,
            "scope": "Two-dimensional XFOIL at frozen conditions; numerical comparison scale, not experimental/RANS/3D validation."}


def run_verify(run_dir: str | Path, *, resume: bool = False, stop_after_evaluations: int | None = None,
               on_progress=None) -> dict:
    if stop_after_evaluations is not None and (type(stop_after_evaluations) is not int or stop_after_evaluations <= 0):
        raise ValueError("stop_after_evaluations must be a positive integer")
    with Evaluator(run_dir) as evaluator, run_ownership(evaluator.store):
        _policy(evaluator.config)
        freeze, model = _context(evaluator)
        state = evaluator.store.stage("verify")
        if state and (state["freeze_hash"] != digest(freeze) or digest(state["freeze"]) != state["freeze_hash"]):
            raise RuntimeError("final model/training/candidate freeze changed; create a new experiment")
        if state and not state.get("complete") and not resume:
            raise StageIncomplete("verification already started; use --resume")
        if state is None:
            budget = evaluator.store.budget_snapshot()
            required = len(freeze["shapes"]) * len(SLOTS)
            if evaluator.store.unfinished() or budget["max_process_launches"] - budget["process_launches"] - budget["pending_launch_reservations"] < required:
                raise StageIncomplete("unfinished evaluations or insufficient final verification launch budget", "failed")
            state = {"freeze": freeze, "freeze_hash": digest(freeze), "budget_start": budget,
                     "started_at_utc": datetime.now(timezone.utc).isoformat(), "complete": False,
                     "slots": [{"design_id": shape["design_id"], "name": name, "panel_count": panel,
                                "eval_id": "eval_" + uuid4().hex, "status": "planned"}
                               for shape in freeze["shapes"] for name, panel in SLOTS]}
            evaluator.store.save_stage("verify", "running", state)
        try:
            _validate_slots(state)
            allowed = {slot["eval_id"] for slot in state["slots"]}
            if set(evaluator.store.unfinished()) - allowed:
                raise RuntimeError("unrelated unfinished evaluation must be recovered first")
            unsealed = evaluator.store.connection.execute("SELECT payload_json FROM events WHERE kind='holdout_unsealed'").fetchall()
            if not unsealed:
                evaluator.store.event("holdout_unsealed", "", {"freeze_hash": state["freeze_hash"], "model_id": freeze["model"]["model_id"],
                                      "candidate_ids": [shape["design_id"] for shape in freeze["shapes"][1:]]})
            holdout = _holdout(evaluator, state, model)
            if state.get("holdout") and state["holdout"] != holdout:
                raise RuntimeError("frozen holdout predictions/evidence changed")
            state["holdout"] = holdout
            evaluator.store.save_stage("verify", "running", state)
            update_manifest(evaluator, "verify", "running", holdout_unsealed=True)
            processed = 0
            if not state.get("complete"):
                for slot in state["slots"]:
                    if slot["status"] in {"success", "failed", "skipped"}:
                        continue
                    shape = next(item for item in freeze["shapes"] if item["design_id"] == slot["design_id"])
                    result = evaluator.evaluate([shape["m"], shape["p"]], purpose="verification", panel_count=slot["panel_count"],
                        save_cp=True, bypass_cache=True, use_reserved_budget=True, evaluation_id=slot["eval_id"], resume=resume)
                    slot.update(status=result.status, failure_reason=result.failure_reason)
                    evaluator.store.save_stage("verify", "running", state)
                    if on_progress:
                        on_progress({"stage": "verify", "role": shape["role"], "slot": slot["name"], "eval_id": result.eval_id,
                                     "status": result.status, "budget": evaluator.store.budget_snapshot()})
                    if result.status == "needs_review":
                        raise StageIncomplete("verification requires an explicit review decision: " + result.eval_id, "needs_review")
                    processed += 1
                    if stop_after_evaluations is not None and processed >= stop_after_evaluations:
                        raise StageIncomplete("explicit verification recovery boundary")
                state.update(complete=True, ended_at_utc=datetime.now(timezone.utc).isoformat(),
                             terminal_budget=evaluator.store.budget_snapshot())
            report = verification_summary(evaluator, state)
            from .reporting import write_verification_artifacts
            if state.get("artifacts"):
                try:
                    verify_receipt(evaluator.run_dir, state)
                except (OSError, RuntimeError):
                    if not resume:
                        raise StageIncomplete("verification derived artifacts changed; use --resume")
                    from .reporting import archive_artifacts
                    archive_artifacts(evaluator.run_dir, state["artifacts"])
            names = write_verification_artifacts(evaluator.run_dir, report)
            state.update(report=report, artifacts=receipt(evaluator.run_dir, names))
            evaluator.store.save_stage("verify", report["status"], state)
            update_manifest(evaluator, "verify", report["status"], m5_passed=report["status"] == "success",
                            improvement_certified=report["certification"]["improvement_certified"])
            return report
        except BaseException as exc:
            checkpoint_failure(evaluator, "verify", state, exc, export_datasets=False)
            raise
