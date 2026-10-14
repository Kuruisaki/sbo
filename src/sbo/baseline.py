"""Baseline gate: independent 200/240/repeated-160 solves and geometric/Cp diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from .config import file_sha256, write_json
from .evaluator import Evaluator
from .geometry import generate_from_config, check_geometry
from .stages import (StageIncomplete, checkpoint_failure, prerequisites, receipt,
                     run_ownership, update_manifest, verify_receipt)


def geometry_diagnostics(config) -> tuple[dict, np.ndarray]:
    geometry = generate_from_config(config)
    reference_root = Path(config["runtime"]["reference_root"])
    path = reference_root / "naca2412-uiuc.dat"
    metadata = json.loads(path.with_suffix(".source.json").read_text(encoding="utf-8"))
    if file_sha256(path) != metadata["sha256"]:
        raise RuntimeError("independent geometry reference hash mismatch")
    reference = np.loadtxt(path, skiprows=1)
    if not check_geometry(reference).valid or not geometry.checks.valid:
        raise RuntimeError("baseline geometry validation failed")
    # UIUC reference is the NACA 2412, not a template for arbitrary baselines.
    if not np.allclose([geometry.m, geometry.p, geometry.t], [.02, .4, .12], rtol=0, atol=1e-14):
        raise ValueError("this baseline gate's UIUC reference requires NACA 2412")
    le = int(np.argmin(reference[:, 0]))
    generated_le = geometry.settings.points_per_side - 1
    errors = []
    for measured, analytic in ((reference[:le], geometry.coordinates[:generated_le + 1]),
                               (reference[le + 1:], geometry.coordinates[generated_le:])):
        order = np.argsort(analytic[:, 0])
        predicted = np.interp(measured[:, 0], analytic[order, 0], analytic[order, 1])
        errors.append(float(np.max(np.abs(predicted - measured[:, 1]))))
    threshold = .0002  # Reference ordinates rounded to four decimals, as in M1 geometry QA.
    return {"passed": max(errors) <= threshold, "max_surface_y_errors": errors,
            "threshold_chord_units": threshold, "reference": metadata,
            "geometry_file_sha256": geometry.file_sha256, "geometry_hash": geometry.geometry_hash}, reference


def pressure_diagnostics(result) -> tuple[dict, np.ndarray]:
    selected = next(item for item in result.attempts if item["attempt_number"] == result.selected_attempt)
    path = Path(selected["runner"]["cp_path"])
    lines = path.read_text(encoding="ascii").splitlines()
    header = next(i for i, line in enumerate(lines) if line.lstrip().startswith("#") and "Cp" in line.split())
    values = np.loadtxt(lines[header + 1:])
    if values.shape != (result.fidelity["panel_count"], 3) or not np.isfinite(values).all():
        raise ValueError("expected finite x,y,Cp per panel")
    if file_sha256(path) != selected["pressure"]["sha256"]:
        raise ValueError("Cp hash mismatch")
    aero = result.aerodynamics
    transitions = all(aero[key] is not None and 0 <= aero[key] <= 1 for key in ("top_xtr", "bot_xtr"))
    # Conservative screening of obvious corruption; engineering thresholds recorded explicitly.
    cp_range = bool(np.all((values[:, 2] >= -20) & (values[:, 2] <= 2)))
    te_difference = float(abs(values[0, 2] - values[-1, 2]))
    return {"passed": transitions and cp_range and te_difference <= .15,
            "top_xtr": aero["top_xtr"], "bot_xtr": aero["bot_xtr"],
            "cp_min": float(values[:, 2].min()), "cp_max": float(values[:, 2].max()),
            "te_cp_difference": te_difference, "te_cp_difference_threshold": .15,
            "cp_screening_range": [-20, 2], "cp_sha256": file_sha256(path)}, values


def _plots(evaluator, results, reference, pressure):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    contour = generate_from_config(evaluator.config).coordinates
    axes[0].plot(contour[:, 0], contour[:, 1], label="Analytic continuous NACA 2412")
    axes[0].plot(reference[:, 0], reference[:, 1], ".", label="UIUC reference")
    axes[0].set_aspect("equal")
    axes[0].set(xlabel="x/c", ylabel="y/c", title="Baseline geometry")
    for name, values in pressure.items():
        axes[1].plot(values[:, 0], values[:, 2], label=name)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="x/c", ylabel="Cp", title="Surface pressure at target CL")
    for name, result in results.items():
        axes[2].scatter(result.fidelity["panel_count"], result.aerodynamics["cd"], label=name)
    axes[2].set(xlabel="Panel count N", ylabel="CD", title="Independent panel / repeat check")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    figure.savefig(evaluator.run_dir / "baseline.png", dpi=160)
    plt.close(figure)


def run_baseline(run_dir: str | Path, *, resume: bool = False) -> dict:
    with Evaluator(run_dir) as evaluator, run_ownership(evaluator.store):
        manifest = prerequisites(evaluator, "baseline")
        state = evaluator.store.stage("baseline")
        if state and state["status"] == "success":
            verify_receipt(evaluator.run_dir, state)
            for eval_id in state["evaluations"].values():
                result = evaluator.store.get_result(eval_id)
                if not evaluator._cache_valid(result, True):
                    raise RuntimeError("completed baseline artifacts changed")
            evaluator.store.write_checkpoint()
            return json.loads((evaluator.run_dir / "baseline.json").read_text(encoding="utf-8"))
        if state and not resume:
            raise StageIncomplete("baseline already started; use --resume")
        if state is None:
            state = {"evaluations": {"160": manifest["baseline_eval_id"],
                     **{name: "eval_" + uuid4().hex for name in ("200", "240", "160_repeat")}}}
        try:
            quality = evaluator.config["quality"]
            if tuple(quality["baseline"]["panel_counts"]) != (160, 200, 240) or quality["baseline"]["repeat_n"] != 160:
                raise ValueError("M2 baseline gate requires panels 160/200/240 and repeat 160")
            evaluator.store.save_stage("baseline", "running", state)
            update_manifest(evaluator, "baseline", "running")
            results = {}
            baseline = evaluator.store.get_result(state["evaluations"]["160"])
            if baseline is None or not baseline.usable_for_training or not evaluator._cache_valid(baseline, True):
                raise StageIncomplete("accepted preflight baseline artifacts no longer validate", "failed")
            results["160"] = baseline
            for name in ("200", "240", "160_repeat"):
                result = evaluator.evaluate(purpose="verification", panel_count=int(name.split('_')[0]), save_cp=True,
                                            evaluation_id=state["evaluations"][name], resume=resume)
                if result.status != "success":
                    raise StageIncomplete(f"baseline {name}: {result.status}/{result.failure_reason}",
                                          "needs_review" if result.status == "needs_review" else "failed")
                results[name] = result
                evaluator.store.save_stage("baseline", "running", state)
            geometry, reference = geometry_diagnostics(evaluator.config)
            diagnostic, pressure = {}, {}
            for name, result in results.items():
                diagnostic[name], pressure[name] = pressure_diagnostics(result)
            repeat_delta = abs(results["160"].aerodynamics["cd"] - results["160_repeat"].aerodynamics["cd"])
            panel_delta = abs(results["200"].aerodynamics["cd"] - results["240"].aerodynamics["cd"])
            threshold = max(quality["panel_convergence"]["absolute_cd_threshold"],
                            quality["panel_convergence"]["relative_cd_threshold"] * results["240"].aerodynamics["cd"])
            checks = {"geometry": geometry["passed"], "transition_cp_screening": all(d["passed"] for d in diagnostic.values()),
                      "repeatability": repeat_delta <= quality["baseline"]["max_repeat_cd_difference"],
                      "panel_convergence": panel_delta <= threshold}
            report = {"config_hash": evaluator.config.config_hash, "status": "success" if all(checks.values()) else "failed",
                      "checks": checks, "evaluations": {k: v.eval_id for k, v in results.items()},
                      "aerodynamics": {k: v.aerodynamics for k, v in results.items()}, "geometry": geometry,
                      "transition_cp_diagnostics": diagnostic, "repeat_cd_difference": repeat_delta,
                      "repeat_threshold": quality["baseline"]["max_repeat_cd_difference"],
                      "panel_cd_difference": panel_delta, "panel_threshold": threshold,
                      "diagnostic_method": "Automated geometry/transition/Cp screening plus archived visual evidence; anomaly stops the gate.",
                      "budget": evaluator.store.budget_snapshot()}
            write_json(evaluator.run_dir / "baseline.json", report)
            _plots(evaluator, results, reference, pressure)
            lines = ["# M2 基准质量关卡", "", f"状态：{report['status']}", "",
                     "| 求解 | N | CL | CD | alpha (deg) | CM | top xtr | bot xtr |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for name, result in results.items():
                a = result.aerodynamics
                lines.append(f"| {name} | {result.fidelity['panel_count']} | {a['cl']} | {a['cd']} | {a['alpha_deg']} | {a['cm']} | {a['top_xtr']} | {a['bot_xtr']} |")
            lines.extend(["", f"160 重复差值：{repeat_delta:.8g}；门槛 {report['repeat_threshold']:.8g}。",
                          f"200/240 差值：{panel_delta:.8g}；门槛 {threshold:.8g}。", "",
                          "几何与 UIUC 独立参考比较，转捩位置和 Cp 自动筛查异常；诊断阈值及原始数据见 baseline.json。",
                          "![几何、压力与离散检查](baseline.png)", "", "该关卡不认证优化收益。"])
            (evaluator.run_dir / "baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            if not all(checks.values()):
                raise StageIncomplete("baseline quality thresholds failed; see baseline.json", "failed")
            state["artifacts"] = receipt(evaluator.run_dir, ["baseline.json", "baseline.md", "baseline.png"])
            evaluator.store.save_stage("baseline", "success", state)
            evaluator.store.export_csv()
            update_manifest(evaluator, "baseline", "success", baseline_passed=True)
            return report
        except BaseException as exc:
            checkpoint_failure(evaluator, "baseline", state, exc)
            raise
