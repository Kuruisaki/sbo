"""Static documentation checks; read-only evidence, no sbo import/CLI/test run."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess
from urllib.parse import unquote
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/development_record"
RAW = DOC / "raw"
MAIN = ROOT / "runs/m1_20261013_02"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    assert path.resolve().is_relative_to(DOC.resolve())
    path.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tree():
    lines = ["Kuruisaki/sbo — local snapshot; Git HEAD d670b51", "Maximum depth 5; .git, pycache, environment/build caches and large run raw files folded.", ""]
    def walk(path, depth=0):
        children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        for child in children:
            if child.name in {".git", "__pycache__", ".venv", "conda", ".pytest_cache", "build", "dist"} or child.name.endswith(".egg-info"):
                continue
            indent = "  " * depth
            if child.name.startswith("pytest-cache-files-"):
                lines.append(indent + child.name + "/ [retained test cache; not expanded]")
                continue
            if child == ROOT / "runs":
                directories = [p for p in child.iterdir() if p.is_dir()]
                lines.append(indent + "runs/ [ignored by Git; " + str(len(directories)) + " existing directories]")
                for selected in (ROOT / "runs/m1_20261013_01", MAIN):
                    lines.append(indent + "  " + selected.name + "/")
                    if selected != MAIN:
                        lines.append(indent + "    preflight.json / environment-probe logs [failed environment gate; 0 solver launches]")
                    else:
                        names = sorted(p.name for p in selected.iterdir() if p.is_file())
                        for name in names:
                            lines.append(indent + "    " + name)
                        lines.extend([indent + "    evaluations/ [51 eval directories; raw files folded]",
                                      indent + "      eval_<UUID>/result.json",
                                      indent + "        attempt_1/ [airfoil.dat, input.txt, stdout.log, stderr.log, polar.txt, optional cp.txt, process.json, result.json]",
                                      indent + "    models/ [9 models]",
                                      indent + "      iter_000/ [gp.joblib, metadata.json, training.json/csv, loo.json/csv/png, proposal.json]",
                                      indent + "      iter_001..008/ [gp.joblib, metadata.json, training.json/csv, loo.json/csv/png; recovery folded if present]",
                                      indent + "    proposals/ [iter_001.json..iter_008.json; fixed pending/eval identities]"])
                        if (selected / "reports").is_dir():
                            for p in sorted((selected / "reports").rglob("*")):
                                lines.append(indent + "    " + str(p.relative_to(selected)).replace("\\", "/") + ("/" if p.is_dir() else ""))
                        for p in sorted(selected.iterdir()):
                            if p.is_dir() and p.name not in {"evaluations", "models", "proposals", "reports"}:
                                lines.append(indent + "    " + p.name + "/ [diagnostic/recovery content folded]")
                other = [p for p in directories if p.name not in {"m1_20261013_01", MAIN.name}]
                for p in sorted(other):
                    lines.append(indent + "  " + p.name + "/ [validation/integration contents folded; see raw/run_inventory.json]")
                continue
            lines.append(indent + child.name + ("/" if child.is_dir() else ""))
            if child.is_dir() and depth < 4:
                walk(child, depth + 1)
            elif child.is_dir():
                lines.append(indent + "  [maximum-depth fold]")
    walk(ROOT)
    write(DOC / "project_tree.txt", "\n".join(lines) + "\n")


def main():
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "checking", "checks": {}, "errors": [],
              "solver_launched_during_archive": False, "new_sbo_run_created": False,
              "method": "AST + existing YAML/JSON/CSV/XML/file hashes + SQLite mode=ro; no project import, CLI, pytest, fit, install or XFOIL."}
    def check(name, condition, detail=None):
        result["checks"][name] = {"passed": bool(condition), "detail": detail}
        if not condition:
            result["errors"].append(name)
    write(RAW / "consistency_check.json", result)
    tree()
    required = ["README.md", "01_项目开发流程.md", "02_软件架构.md", "03_实际执行流程.md", "04_SBO数学流程.md", "05_各文件功能.md", "project_tree.txt"]
    check("required_documents", all((DOC / name).is_file() for name in required), required)
    check("required_raw_git", all((RAW / name).is_file() for name in ("git_status.txt", "git_log.txt", "git_diff.patch", "git_branches.txt", "history_recovery_notes.md")))
    documents = {name: (DOC / name).read_text(encoding="utf-8") for name in required if name.endswith(".md")}
    bodies = "\n".join(documents.values())
    broken = []
    for file in [*(DOC / name for name in documents), RAW / "history_recovery_notes.md"]:
        for link in re.findall(r"!?\[[^\]\n]+\]\(([^)\n]+)\)", file.read_text(encoding="utf-8")):
            if link.startswith(("https://", "http://", "#", "codex://")):
                continue
            target = (file.parent / unquote(link.split("#", 1)[0])).resolve()
            if not target.exists():
                broken.append({"file": str(file.relative_to(DOC)), "target": link})
    check("local_links_resolve", not broken, broken)

    config = yaml.safe_load((ROOT / "configs/naca2412.yaml").read_text(encoding="utf-8"))
    frozen = yaml.safe_load((MAIN / "config.frozen.yaml").read_text(encoding="utf-8"))
    def leaves(x):
        return sum(leaves(v) for v in x.values()) if isinstance(x, dict) else 1
    check("yaml_sections_and_leaves", len(config) == 12 and leaves(config) == 202, {"sections": list(config), "leaves": leaves(config)})
    check("source_config_equals_main_frozen_values", config == frozen)
    check("documented_default_design", config["design"]["variables"] == ["m", "p"] and config["design"]["bounds"] == [[.01, .04], [.25, .60]] and config["design"]["t"] == .12)
    check("documented_physics", config["flow"]["reynolds"] == 1e6 and config["flow"]["mach"] == .1 and config["flow"]["target_cl"] == .5 and config["xfoil"]["standard_n"] == 160)
    check("documented_doe_and_gp", config["doe"]["initial_points"] == 16 and config["doe"]["holdout_points"] == 6 and config["doe"]["seed"] == 42 and config["doe"]["holdout_seed"] == 2026 and config["gp"]["matern_nu"] == 2.5 and config["gp"]["alpha_regularization"] == 1e-6 and config["gp"]["n_restarts_optimizer"] == 5)
    check("documented_ei_stopping_verification", config["acquisition"]["xi"] == 1e-5 and config["stopping"]["convergence"]["successful_points_window"] == 5 and config["verification"]["panel_counts"] == [160,200,240] and config["verification"]["repeat_n"] == 200)

    inventory = read(RAW / "source_inventory.json")
    core = [row for row in inventory if row["path"].startswith("src/sbo/")]
    check("core_module_catalogue", len(core) == 19 and all("### " + Path(row["path"]).name in documents["05_各文件功能.md"] for row in core))
    declared_symbols = {
        "config.py": ["FrozenConfig","load_config","create_run","verify_run_config"],
        "geometry.py": ["GeometryResult","naca4_surfaces","generate_from_config","check_geometry"],
        "xfoil_runner.py": ["XfoilSettings","XfoilAttemptResult","build_session","run_xfoil"],
        "polar.py": ["PolarResult","AerodynamicResult","parse_polar_text","parse_attempt"],
        "records.py": ["EvaluationResult","BudgetExhausted"], "evaluator.py": ["Evaluator","evaluate","resume","review","build_cache_key"],
        "store.py": ["SQLiteEvaluationStore","training_data","final_holdout_data","save_model_record"],
        "stages.py": ["StageIncomplete","run_ownership","resolve_run","verify_receipt"],
        "sampling.py": ["normalize","lhs","sobol","maximin","initial_plan","replace_slot"],
        "doe.py": ["run_doe"], "m2_cli.py": ["main"], "baseline.py": ["run_baseline","geometry_diagnostics","pressure_diagnostics"],
        "surrogate.py": ["TrainingData","SurrogateModel","fit_gp","loo_cv","load_model","save_model"],
        "fitting.py": ["run_fit"], "acquisition.py": ["expected_improvement","candidate_pool","propose"],
        "optimizer.py": ["run_optimize","stopping_decision","_verified_data","_ensure_model"],
        "verification.py": ["run_verify","holdout_metrics","shape_quality","certify","verification_summary"],
        "reporting.py": ["run_report","write_verification_artifacts","archive_artifacts"]}
    missing_symbols=[]
    for row in core:
        name=Path(row["path"]).name
        actual={x["name"] for x in row["functions"]+row["classes"]}
        missing_symbols.extend(name+":"+symbol for symbol in declared_symbols.get(name,[]) if symbol not in actual or symbol not in documents["05_各文件功能.md"])
    check("catalogue_symbols_exist_in_actual_ast",not missing_symbols,missing_symbols)
    check("ast_sources_not_changed", all(sha(ROOT / row["path"]) == row["sha256"] for row in inventory))
    options = {}
    for row in inventory:
        if row["path"].startswith("scripts/") or row["path"] == "src/sbo/m2_cli.py":
            options[row["path"]] = [x["call"] for x in row["cli_arguments"]]
    check("all_seven_entry_points", all((ROOT / "scripts" / name).is_file() for name in ("01_preflight.py","02_baseline.py","03_doe.py","04_fit.py","05_optimize.py","06_verify.py","07_report.py")), options)
    report_cli = next(row for row in inventory if row["path"] == "scripts/07_report.py")
    check("report_cli_only_run_id", len(report_cli["cli_arguments"]) == 1 and "--run-id" in report_cli["cli_arguments"][0]["call"])
    check("no_invented_console_entry", "[project.scripts]" not in (ROOT / "pyproject.toml").read_text())
    check("ei_formula_includes_xi", "\\xi" in documents["04_SBO数学流程.md"] and "f_best - mean - xi" in (ROOT / "src/sbo/acquisition.py").read_text())
    check("stopping_window_uses_five_transitions", "B_{n-5}-B_n" in documents["04_SBO数学流程.md"] and '-convergence["successful_points_window"] - 1' in (ROOT / "src/sbo/optimizer.py").read_text())

    manifest = read(MAIN / "manifest.json")
    check("frozen_file_receipt", sha(MAIN / "config.frozen.yaml") == manifest["config_file_sha256"])
    exe_sha=read(MAIN/"preflight.json")["xfoil"]["executable_sha256"]
    check("solver_identity_documented_without_execution",exe_sha in documents["03_实际执行流程.md"] and sha(Path(config["xfoil"]["executable"]))==exe_sha)
    reference=read(ROOT/"data/reference/naca2412-uiuc.source.json")
    check("reference_geometry_identity",reference["sha256"] in documents["05_各文件功能.md"] and sha(ROOT/"data/reference/naca2412-uiuc.dat")==reference["sha256"])
    with sqlite3.connect((MAIN / "samples.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        launches = db.execute("SELECT SUM(launched) FROM attempts").fetchone()[0]
        train = db.execute("SELECT COUNT(*) FROM designs WHERE split='train'").fetchone()[0]
        held = db.execute("SELECT COUNT(*) FROM designs WHERE split='holdout'").fetchone()[0]
        models = db.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        stages = db.execute("SELECT name,status FROM stages ORDER BY name").fetchall()
        identity = db.execute("SELECT value FROM metadata WHERE key='config_hash'").fetchone()[0]
        pending = db.execute("SELECT COUNT(*) FROM attempts WHERE status='reserved' AND launched=0").fetchone()[0]
        check("sqlite_main_counts", (launches,train,held,models,pending) == (51,25,6,9,0), {"launches":launches,"train":train,"holdout":held,"models":models,"pending":pending})
        check("sqlite_all_six_stages_success", len(stages) == 6 and all(status == "success" for _,status in stages), stages)
        check("sqlite_config_identity", identity == manifest["config_hash"])
    verify = read(MAIN / "verification.json")
    opt = read(MAIN / "optimization.json")
    fit = read(MAIN / "fit.json")
    check("historical_budget_snapshots", opt["budget"]["process_launches"] == 39 and fit["budget_after"]["process_launches"] == 31 and verify["budget"]["process_launches"] == 51)
    check("documented_main_result", verify["holdout"]["metrics"]["quality_passed"] and verify["certification"]["improvement_certified"] and opt["successful_sequential_points"] == 8 and opt["stop_reason"] == "converged")
    check("same_fidelity_cd", [shape["aerodynamics"]["240"]["cd"] for shape in verify["shapes"]] == [.006,.00545,.00546])
    check("unique_final_slots", len({row["eval_id"] for shape in verify["shapes"] for row in shape["evaluations"].values()}) == 12)
    for label in ("iter_000","iter_008"):
        metadata = read(MAIN / "models" / label / "metadata.json")
        loo = read(MAIN / "models" / label / "loo.json")
        residuals = [fold["predicted_cd"] - fold["observed_cd"] for fold in loo["folds"]]
        mae = sum(abs(x) for x in residuals)/len(residuals)
        rmse = math.sqrt(sum(x*x for x in residuals)/len(residuals))
        check(label+"_model_and_loo", sha(MAIN/"models"/label/"gp.joblib") == metadata["artifact"]["sha256"] and math.isclose(mae,loo["mae_cd"],abs_tol=1e-15) and math.isclose(rmse,loo["rmse_cd"],abs_tol=1e-15), {"folds":len(residuals),"mae":mae,"rmse":rmse})
    point_rows = verify["holdout"]["predictions"]
    mae_h = sum(abs(row["observed_cd"]-row["predicted_cd"]) for row in point_rows)/len(point_rows)
    check("holdout_metrics_from_saved_rows", len(point_rows) == 6 and math.isclose(mae_h,verify["holdout"]["metrics"]["mae"],abs_tol=1e-15))
    check("reports_mirror_integrity", all(sha(MAIN/source)==sha(MAIN/target) for source,target in (("report.json","reports/metrics.json"),("verification.csv","reports/final_comparison.csv"),("best_airfoil.dat","reports/best_airfoil.dat"),("best_airfoil.json","reports/best_airfoil.json"))))

    latest = {}
    execution_summary = []
    for name in ("m5-tests.xml","m5-final-tests.xml","m5-freeze-tests.xml","m5-slot-tests.xml"):
        cases = ET.parse(MAIN/name).findall(".//testcase")
        execution_summary.append({"file":name,"cases":len(cases),"failures":sum(x.find("failure") is not None or x.find("error") is not None for x in cases)})
        for case in cases:
            latest[(case.get("classname"),case.get("name"))] = all(case.find(tag) is None for tag in ("failure","error","skipped"))
    check("saved_test_latest_results", len(latest)==300 and all(latest.values()), {"distinct":len(latest),"integration":sum("_integration" in key[0] for key in latest),"executions":execution_summary,"rerun_now":False})
    hist = read(RAW/"codex_conversation.json")
    check("public_history_no_internal_reasoning", all(item["type"] not in {"reasoning","analysis"} and (item["type"]!="agentMessage" or item.get("phase") in {"commentary","final_answer"}) for thread in hist["threads"] for turn in thread["turns"] for item in turn["items"]))
    check("recovered_history_counts",len(hist["threads"])==3 and sum(len(x["turns"]) for x in hist["threads"])==13 and len(read(RAW/"command_history.json"))==210)
    events = re.findall(r"^### 开发事件(\d+)：(.+)$",documents["01_项目开发流程.md"],re.M)
    check("27_development_events", [int(number) for number,_ in events]==list(range(1,28)), [title for _,title in events])
    write(RAW/"development_events.json", {"kind":"Evidence-linked summaries in chapter01; not fabricated independent prompts.","count":len(events),"events":[{"id":int(n),"title":title} for n,title in events]})
    baseline = read(RAW/"baseline_integrity.json")
    changed = [name for name,value in baseline["files"].items() if not (ROOT/name).is_file() or sha(ROOT/name)!=value]
    check("all_protected_files_byte_unchanged", not changed, {"count":len(baseline["files"]),"changed":changed})
    head = subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    status = subprocess.run(["git","status","--short","--branch"],cwd=ROOT,capture_output=True,text=True).stdout
    write(RAW/"git_status.txt",status)
    check("git_head_unchanged",head==baseline["head"])
    check("only_new_archive_uncommitted",[line for line in status.splitlines() if not line.startswith("##")]==["?? docs/"],status)
    result["confirmed_existing_description_differences"] = ["README calls models table reserved although actively populated", "07 is zero-solver/zero-fit but writes stage/checkpoint/manifest", "duplicate_proposal config label has no same-name maximin branch", "completed DOE revalidation/export reads stored holdout labels within M2", "historical stage report scope is not current pipeline status"]
    result["status"] = "success" if not result["errors"] else "failed"
    write(RAW/"consistency_check.json",result)
    print(json.dumps({"status":result["status"],"checks":len(result["checks"]),"errors":result["errors"],"protected_files":len(baseline["files"]),"solver_launches_before_and_after":launches,"development_events":len(events),"git_status":status},ensure_ascii=False))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
