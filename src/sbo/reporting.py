"""Read-only M5 evidence reporting; no solver launch or model fitting."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from .config import file_sha256, write_json
from .evaluator import Evaluator
from .fitting import _write_csv
from .geometry import generate_from_config
from .stages import receipt, run_ownership, update_manifest


def archive_artifacts(directory: Path, names):
    archive = directory / "report-recovery" / uuid4().hex
    for name in names:
        source = (directory / name).resolve()
        if not source.is_relative_to(directory.resolve()):
            raise ValueError("report artifact outside experiment")
        if source.is_file():
            destination = archive / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    return archive


def verification_rows(report):
    rows = []
    for shape in report["shapes"]:
        for name, evaluation in shape["evaluations"].items():
            rows.append({"role": shape["role"], "design_id": shape["design_id"], "m": shape["m"], "p": shape["p"], "t": shape["t"],
                         "slot": name, **evaluation, **(shape["aerodynamics"][name] or {}),
                         "shape_passed": shape["quality"]["passed"], "epsilon": shape["quality"]["epsilon"]})
    return rows


def _verification_plot(directory, report):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    predictions = report["holdout"]["predictions"]
    if predictions:
        y = [row["observed_cd"] for row in predictions]
        mean = [row["predicted_cd"] for row in predictions]
        error = [report["holdout"]["metrics"]["interval_sigma"] * row["sigma_cd"] for row in predictions]
        axes[0].errorbar(y, mean, yerr=error, fmt="o", capsize=3, label="Frozen GP mean +/- 2 sigma")
        limits = [min(*y, *mean), max(*y, *mean)]
        axes[0].plot(limits, limits, "k--", label="Identity")
    axes[0].set(xlabel="Independent observed CD (N=160)", ylabel="GP predicted CD", title="Sealed holdout, opened after freeze")
    for shape in report["shapes"]:
        points = [(evaluation["panel_count"], shape["aerodynamics"][name]["cd"])
                  for name, evaluation in shape["evaluations"].items() if shape["aerodynamics"][name]]
        if points:
            axes[1].plot(*np.asarray(points[:3]).T, "o-", label=shape["role"])
            if len(points) == 4:
                axes[1].scatter(*points[3], marker="x", color="black")
    axes[1].set(xlabel="Panel count", ylabel="Independent CD", title="Panel sensitivity; x marks N=200 repeat")
    for axis in axes:
        if axis.lines:
            axis.legend(fontsize=8)
        axis.grid(alpha=.25)
    figure.savefig(directory / "verification.png", dpi=160)
    plt.close(figure)


def write_verification_artifacts(directory: Path, report: dict) -> list[str]:
    # SQLite's canonical serialization sorts keys. Derived JSON must have the
    # same bytes before and after a reload, not merely equivalent values.
    write_json(directory / "verification.json", json.loads(json.dumps(report, sort_keys=True, allow_nan=False)))
    rows = verification_rows(report)
    _write_csv(directory / "verification.csv", rows,
        ["role", "design_id", "m", "p", "t", "slot", "eval_id", "status", "failure_reason", "panel_count",
         "alpha_deg", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr", "cl_error", "shape_passed", "epsilon"])
    _write_csv(directory / "holdout-predictions.csv", report["holdout"]["predictions"],
               ["design_id", "eval_id", "m", "p", "observed_cd", "predicted_cd", "sigma_cd", "residual_cd"])
    _verification_plot(directory, report)
    return ["verification.json", "verification.csv", "holdout-predictions.csv", "verification.png"]


def _final_plot(evaluator, report, verification):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    if verification:
        from .baseline import pressure_diagnostics
        for shape in verification["shapes"]:
            geometry = generate_from_config(evaluator.config, [shape["m"], shape["p"]])
            axes[0, 0].plot(*geometry.coordinates.T, label=shape["role"])
            evaluation = shape["evaluations"]["240"]
            if evaluation["status"] == "success":
                _, cp = pressure_diagnostics(evaluator.store.get_result(evaluation["eval_id"]))
                axes[0, 1].plot(cp[:, 0], cp[:, 2], label=shape["role"])
        axes[0, 1].invert_yaxis()
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set(xlabel="x/c", ylabel="y/c", title="Frozen candidates and baseline")
    axes[0, 1].set(xlabel="x/c", ylabel="Cp", title="Independent N=240 surface pressure")
    data = evaluator.store.training_data()
    axes[1, 0].scatter([row["m"] for row in data], [row["p"] for row in data], label="Accepted training", s=20)
    coordinates = evaluator.store.holdout_coordinates()
    axes[1, 0].scatter([row["m"] for row in coordinates], [row["p"] for row in coordinates], marker="x", label="Reserved holdout")
    axes[1, 0].set(xlabel="m", ylabel="p", title="Sampling distribution; fixed bounds")
    axes[1, 0].set_xlim(*evaluator.config["design"]["bounds"][0])
    axes[1, 0].set_ylim(*evaluator.config["design"]["bounds"][1])
    optimization = evaluator.store.stage("optimize")
    if optimization:
        values = [optimization["initial_best_cd"], *[row["best_cd"] for row in optimization["history"]]]
        axes[1, 1].step(range(len(values)), values, where="post", label="Accepted N=160 best")
    axes[1, 1].set(xlabel="Sequential design outcomes", ylabel="CD", title="Standard-fidelity best-so-far")
    for axis in axes.flat:
        axis.grid(alpha=.25)
        if axis.lines or axis.collections:
            axis.legend(fontsize=8)
    if verification:
        axes[0, 0].legend(loc="lower center", bbox_to_anchor=(.5, 1.55), ncol=3, fontsize=8)
    figure.savefig(evaluator.run_dir / "report.png", dpi=160)
    plt.close(figure)


def _recommendation(evaluator, verification):
    selected = verification["certification"]["recommendation_design_id"] if verification else None
    paths = [evaluator.run_dir / "best_airfoil.dat", evaluator.run_dir / "best_airfoil.json"]
    if selected is None:
        # These two names are exclusively owned by this stage. Preserve a prior
        # export as evidence, but remove it from the active recommendation path.
        if any(path.is_file() for path in paths):
            archive_artifacts(evaluator.run_dir, [path.name for path in paths])
            for path in paths:
                path.unlink(missing_ok=True)
        return None
    shape = next(shape for shape in verification["shapes"] if shape["design_id"] == selected)
    geometry = generate_from_config(evaluator.config, [shape["m"], shape["p"]])
    temp = paths[0].with_suffix(".dat.tmp")
    temp.write_bytes(geometry.dat_bytes)
    temp.replace(paths[0])
    metadata = {"config_hash": evaluator.config.config_hash, "freeze_hash": verification["freeze_hash"],
                "design_id": selected, "m": shape["m"], "p": shape["p"], "t": shape["t"],
                "reason": verification["certification"]["recommendation_reason"],
                "geometry_hash": geometry.geometry_hash, "file_sha256": file_sha256(paths[0]),
                "generator_version": evaluator.config["geometry"]["generator_version"],
                "independent_evaluations": shape["evaluations"], "quality": shape["quality"],
                "surrogate_quality_passed": verification["checks"]["independent_holdout"],
                "boundary_hits": {name: shape[name] in evaluator.config["design"]["bounds"][i]
                                  for i, name in enumerate(("m", "p"))},
                "scope": verification["scope"]}
    write_json(paths[1], metadata)
    return metadata


def _markdown(report, verification):
    lines = ["# SBO 最终报告", "", f"状态：{report['status']}；阶段门槛通过：{report['m5_passed']}。", "",
             f"冻结配置 SHA256：`{report['config_hash']}`。", "",
             f"工况：Re={report['flow']['reynolds']:g}，Mach={report['flow']['mach']:g}，CL={report['flow']['target_cl']:g}，"
             f"Ncrit={report['flow']['ncrit']:g}，XTR={report['flow']['xtr']}，TYPE={report['flow']['type']}；t={report['fixed_t']}。", "",
             f"累计 XFOIL 启动：{report['budget']['process_launches']}；不同标准设计：{report['budget']['distinct_standard_designs']}。",
             f"失败评估数（重试后最终结果）：{report['failed_evaluations']}；失败尝试数：{report['failed_attempts']}。", ""]
    if not verification:
        lines.extend(["复核尚未完成或证据校验失败，不能给出收益认证或最终推荐。", "", report.get("diagnostic_reason", "缺少完整 M5 记录。")])
        return "\n".join(lines) + "\n"
    metrics = verification["holdout"]["metrics"]
    lines.extend([f"最终模型：`{verification['model_id']}`；训练点 {verification['training_points']}；模型 SHA256：`{verification['model_sha256']}`。",
                  f"最终 LOO MAE：{report['loo']['mae_cd']:.8g}；独立留出 {metrics['count']} 点 MAE：{metrics['mae']:.8g}，"
                  f"RMSE：{metrics['rmse']:.8g}，最大绝对误差：{metrics['max_absolute_error']:.8g}；MAE 门槛：{metrics['mae_threshold']:.8g}。",
                  f"留出质量通过：{metrics['quality_passed']}；R²：{metrics['r2']}；均值±2σ覆盖率：{metrics['two_sigma_coverage']:.1%}（仅诊断）。",
                  "留出标签在模型/候选冻结后开放，没有进入训练、调参、停止或候选选择。", "",
                  "| 翼型 | m | p | CD160 | CD200 | CD240 | CD200重复 | ε | 复核通过 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for shape in verification["shapes"]:
        values = [f"{shape['aerodynamics'][name]['cd']:.8g}" if shape['aerodynamics'][name] else "—" for name in ("160", "200", "240", "200_repeat")]
        lines.append(f"| {shape['role']} | {shape['m']:.13g} | {shape['p']:.13g} | " + " | ".join(values) +
                     f" | {format(shape['quality']['epsilon'], '.8g') if shape['quality']['epsilon'] is not None else '—'} | {shape['quality']['passed']} |")
    lines.extend(["", "ε=max(|CD160−CD200|,|CD200−CD240|,|两次CD200之差|,1e−5)，是工程对比尺度，不是统计置信区间。", "",
                  "| N=240 排名 | 候选 | ΔCD240（基准−候选） | 相对改善 | ε基准+ε候选 | 超过数值差异 | ΔCD160 |",
                  "|---|---|---:|---:|---:|---|---:|"])
    for comparison in verification["certification"]["comparisons"]:
        role = next(shape["role"] for shape in verification["shapes"] if shape["design_id"] == comparison["design_id"])
        lines.append(f"| {comparison['rank_240']} | {role} | {comparison['delta_cd_240']:.8g} | {comparison['relative_improvement_240']:.2%} | {comparison['uncertainty_sum']:.8g} | {comparison['certified']} | {comparison['delta_cd_160']:.8g} |")
    recommendation = report["recommendation"]
    if recommendation:
        lines.extend(["", f"最终推荐：m={recommendation['m']:.13g}，p={recommendation['p']:.13g}，t={recommendation['t']}；原因 `{recommendation['reason']}`。",
                      "几何为 `best_airfoil.dat`，完整来源及哈希为 `best_airfoil.json`。"])
    else:
        lines.extend(["", "基准复核未通过，本轮没有最终推荐几何，收益结论暂缓。"])
    if not metrics["quality_passed"]:
        lines.extend(["", "独立留出质量关卡失败。没有重调 GP，也没有用留出点替换候选；直接求解的数值比较与代理泛化结论分别保存。"])
    lines.extend(["", "![留出误差和面元敏感性](verification.png)", "", "![几何、压力、采样与序贯历史](report.png)", "",
                  "结论仅适用于本次二维 XFOIL 模型、工况和设计边界；不是实验真值、RANS 或三维机翼验证，也不证明全局最优。",
                  "Cp 和转捩经过记录阈值的自动异常筛查；原始会话、压力数据、每点完整气动值和失败原因均保留。"])
    return "\n".join(lines) + "\n"


def run_report(run_dir: str | Path) -> dict:
    """Revalidate committed evidence and create reports; incomplete runs stay diagnostic."""
    from .verification import _context, _holdout, verification_summary
    with Evaluator(run_dir) as evaluator, run_ownership(evaluator.store):
        state = evaluator.store.stage("verify")
        verification, diagnostic = None, None
        try:
            if not state or not state.get("complete"):
                raise ValueError("M5 independent verification has not completed")
            freeze, model = _context(evaluator)
            from .surrogate import digest
            if digest(freeze) != state["freeze_hash"] or digest(state["freeze"]) != state["freeze_hash"]:
                raise ValueError("final verification freeze mismatch")
            if _holdout(evaluator, state, model) != state["holdout"]:
                raise ValueError("holdout evidence/predictions changed")
            verification = verification_summary(evaluator, state)
            if verification != state["report"]:
                raise ValueError("verification ledger summary changed")
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            verification = None
            diagnostic = f"{type(exc).__name__}: {exc}"
        db = evaluator.store.connection
        model = evaluator.store.model(verification["model_id"]) if verification else None
        report = {"status": verification["status"] if verification else "incomplete", "stage": "report",
                  "config_hash": evaluator.config.config_hash, "flow": dict(evaluator.config["flow"]),
                  "fixed_t": evaluator.config["design"]["t"], "budget": evaluator.store.budget_snapshot(),
                  "failed_evaluations": db.execute("SELECT COUNT(*) FROM evaluations WHERE status='failed'").fetchone()[0],
                  "failed_attempts": db.execute("SELECT COUNT(*) FROM attempts WHERE status IN ('failed','interrupted')").fetchone()[0],
                  "unaccepted_evaluations": [dict(row) for row in db.execute(
                      "SELECT e.eval_id,e.design_id,e.purpose,e.panel_count,e.status,d.m,d.p,d.t, "
                      "json_extract(e.result_json,'$.failure_reason') AS failure_reason "
                      "FROM evaluations e JOIN designs d USING(design_id) WHERE e.status IN ('failed','skipped','needs_review') ORDER BY e.rowid")],
                  "m5_passed": verification is not None and verification["status"] == "success",
                  "loo": model["metadata"]["loo"] if model else None, "diagnostic_reason": diagnostic,
                  "verification": verification, "recommendation": _recommendation(evaluator, verification)}
        names = ["report.json", "report.md", "report.png"]
        if verification:
            changed = [name for name, sha in state.get("artifacts", {}).items()
                       if not (evaluator.run_dir / name).is_file() or file_sha256(evaluator.run_dir / name) != sha]
            if changed:
                archive_artifacts(evaluator.run_dir, changed)
            verification_names = write_verification_artifacts(evaluator.run_dir, verification)
            names.extend(verification_names)
            state["artifacts"] = receipt(evaluator.run_dir, verification_names)
            evaluator.store.save_stage("verify", verification["status"], state)
        _final_plot(evaluator, report, verification)
        write_json(evaluator.run_dir / "report.json", report)
        (evaluator.run_dir / "report.md").write_text(_markdown(report, verification), encoding="utf-8")
        if report["recommendation"]:
            names.extend(["best_airfoil.dat", "best_airfoil.json"])
        # Preserve the documented reports/ layout, alongside the run-root
        # stage summaries used consistently by M2-M5.
        reports = evaluator.run_dir / "reports"
        (reports / "figures").mkdir(parents=True, exist_ok=True)
        summary = _markdown(report, verification).replace("(verification.png)", "(figures/verification.png)").replace("(report.png)", "(figures/report.png)")
        (reports / "summary.md").write_text(summary, encoding="utf-8")
        copies = {"report.json": "reports/metrics.json", "report.png": "reports/figures/report.png"}
        if verification:
            copies.update({"verification.csv": "reports/final_comparison.csv", "verification.png": "reports/figures/verification.png"})
        else:
            _write_csv(reports / "final_comparison.csv", [], ["role", "design_id", "slot", "status", "cd"])
            names.append("reports/final_comparison.csv")
        if report["recommendation"]:
            copies.update({"best_airfoil.dat": "reports/best_airfoil.dat", "best_airfoil.json": "reports/best_airfoil.json"})
        else:
            old = ["reports/best_airfoil.dat", "reports/best_airfoil.json"]
            if any((evaluator.run_dir / name).is_file() for name in old):
                archive_artifacts(evaluator.run_dir, old)
                for name in old:
                    (evaluator.run_dir / name).unlink(missing_ok=True)
        for source, destination in copies.items():
            shutil.copyfile(evaluator.run_dir / source, evaluator.run_dir / destination)
            names.append(destination)
        names.append("reports/summary.md")
        evaluator.store.save_stage("report", report["status"], {"artifacts": receipt(evaluator.run_dir, names),
            "freeze_hash": verification["freeze_hash"] if verification else None, "m5_passed": report["m5_passed"]})
        update_manifest(evaluator, "report", report["status"], m5_passed=report["m5_passed"],
                        improvement_certified=bool(verification and verification["certification"]["improvement_certified"]))
        return report
