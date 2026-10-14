"""M4 stopping/retry/review/recovery tests. Archived outputs stay in disposable runs."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from test_evaluator import lab, solver, ROOT
from sbo.config import FrozenConfig, create_run, file_sha256, load_config, write_json
from sbo.evaluator import Evaluator
from sbo.fitting import run_fit
from sbo.optimizer import run_optimize, stopping_decision
from sbo.stages import StageIncomplete
from sbo.store import SQLiteEvaluationStore


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/naca2412.yaml")


def budget():
    return {"process_launches": 31, "pending_launch_reservations": 0,
            "distinct_standard_designs": 23, "max_process_launches": 120,
            "max_distinct_standard_designs": 45, "reserved_final_verification_launches": 12}


def test_m5_final_model_damage_stops_before_refit_or_file_write(lab, monkeypatch):
    from types import SimpleNamespace
    import sbo.optimizer as module
    frozen = {"freeze": {"model": {"model_id": "iter_001"}}}
    store = SimpleNamespace(model=lambda _: {"metadata": {}}, stage=lambda _: frozen)
    evaluator = SimpleNamespace(run_dir=lab, store=store, config={})
    state = {"history": [{"status": "success"}]}

    def damaged(*_):
        raise ValueError("deliberately damaged final model")

    monkeypatch.setattr(module, "load_model", damaged)
    monkeypatch.setattr(module, "fit_gp", lambda *_: pytest.fail("unsealed final GP must never be refitted"))
    with pytest.raises(RuntimeError, match="M5 final GP is frozen"):
        module._ensure_model(evaluator, state, SimpleNamespace(data_hash="test-only"), resume=True)
    assert not (lab / "models").exists()


def history(statuses, improvement=0):
    best, rows = .006, []
    for status in statuses:
        if status == "success":
            best -= improvement
        rows.append({"status": status, "best_cd": best})
    return rows


def test_failure_window_uses_design_outcome_not_retry_attempt(config):
    rows = history(["success", "failed", "success", "success", "failed", "success", "success", "success", "success", "success"])
    rows[0]["attempt_statuses"] = ["failed", "success"]
    result = stopping_decision(rows, budget(), config, .006, 1e-4)
    assert result["failure_rate"] == .2 and not result["stop"]
    rows[3]["status"] = "failed"
    assert stopping_decision(rows, budget(), config, .006, 1e-4)["reason"] == "failure_rate_exceeded"


def test_consecutive_failure_and_incomplete_window(config):
    result = stopping_decision(history(["failed", "failed"]), budget(), config, .006, 0)
    assert result["failure_rate"] is None and not result["stop"]
    assert stopping_decision(history(["failed"] * 3), budget(), config, .006, 0)["reason"] == "consecutive_failed_designs"
    rows = history(["failed", "failed", "needs_review", "skipped", "success"])
    result = stopping_decision(rows, budget(), config, .006, 0)
    assert result["successful_sequential_points"] == 1 and result["consecutive_failed_designs"] == 0


def test_convergence_requires_min_count_and_both_conditions(config):
    assert not stopping_decision(history(["success"] * 7), budget(), config, .006, 0)["stop"]
    rows = history(["success"] * 8)
    assert stopping_decision(rows, budget(), config, .006, 9e-6)["reason"] == "converged"
    assert not stopping_decision(rows, budget(), config, .006, 1e-5)["stop"]  # Strict EI < threshold.
    assert not stopping_decision(rows, budget(), config, .006, None)["stop"]
    assert not stopping_decision(history(["success"] * 8, 1e-5), budget(), config, .006, 0)["stop"]
    result = stopping_decision(history(["success"] * 8, 1e-6), budget(), config, .006, 0)
    assert result["successful_window_improvement_cd"] == pytest.approx(5e-6)


@pytest.mark.parametrize("field,value,reason", [("distinct_standard_designs", 45, "distinct_design_budget"),
    ("process_launches", 108, "process_launch_budget_or_final_reserve"),
    ("pending_launch_reservations", 77, "process_launch_budget_or_final_reserve")])
def test_hard_budgets_protect_final_reserve(config, field, value, reason):
    snapshot = budget()
    snapshot[field] = value
    assert stopping_decision([], snapshot, config, .006, None)["reason"] == reason


def test_success_limit_and_planned_valid_cap(config):
    assert stopping_decision(history(["success"] * 12), budget(), config, .006, 1)["reason"] == "max_successful_sequential_points"
    values = config.to_dict()
    values["budget"]["max_planned_valid_standard_designs"] = 24  # 17 train + 6 sealed coords + one new success.
    changed = FrozenConfig(values)
    assert stopping_decision(history(["success"]), budget(), changed, .006, 1)["reason"] == "max_planned_valid_standard_designs"


def ready(lab, *, iteration_start=1, **overrides):
    values = load_config(ROOT / "configs/naca2412.yaml").to_dict()
    values["runtime"]["runs_root"] = str(lab)
    values["doe"]["initial_points"], values["doe"]["holdout_points"] = 3, 1
    values["gp"]["n_restarts_optimizer"] = 0
    values["acquisition"]["iteration_start"] = iteration_start
    values["budget"]["max_successful_sequential_points"] = 2
    values["budget"].update(overrides)
    config = FrozenConfig(values)
    directory = create_run(config, "m4")
    with Evaluator(directory) as evaluator:
        baseline = evaluator.evaluate(save_cp=True)
        slots = []
        for design in [[.012, .28], [.03, .5], [.038, .59]]:
            result = evaluator.evaluate(design, purpose="doe")
            assert result.usable_for_training
            slots.append({"split": "train", "status": "success", "eval_id": result.eval_id})
        holdout = evaluator.evaluate([.022, .48], split="holdout", purpose="doe")
        slots.append({"split": "holdout", "status": "success", "eval_id": holdout.eval_id})
        evaluator.store.save_stage("baseline", "success", {"evaluations": {"160": baseline.eval_id}, "artifacts": {}})
        evaluator.store.save_stage("doe", "success", {"slots": slots, "artifacts": {}})
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest.update(preflight_passed=True, baseline_eval_id=baseline.eval_id)
        write_json(directory / "manifest.json", manifest)
        evaluator.store.export_csv()
    run_fit(directory)
    return directory, config


def test_success_updates_training_model_and_does_not_open_sealed_labels(lab, solver, monkeypatch):
    directory, config = ready(lab)
    sealed_hash = file_sha256(directory / "holdout.csv")
    actual_dataset = SQLiteEvaluationStore._dataset

    def train_only(self, split, **kwargs):
        assert split == "train", "M4 cannot query sealed labels"
        return actual_dataset(self, split, **kwargs)

    monkeypatch.setattr(SQLiteEvaluationStore, "_dataset", train_only)
    original_open = Path.open

    def no_holdout_file(self, *args, **kwargs):
        assert self.name != "holdout.csv", "M4 cannot open sealed CSV"
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_holdout_file)
    report = run_optimize(directory)
    assert report["status"] == "success" and report["stop_reason"] == "max_successful_sequential_points"
    assert report["successful_sequential_points"] == 2 and report["training_points"] == 6
    assert report["current_model_id"] == "iter_002" and report["budget"]["process_launches"] == 7
    assert all(row["cd"] > 0 for row in report["history"])
    assert not report["holdout_labels_used"] and not report["improvement_certified"]
    count = len(solver[0])
    assert run_optimize(directory) == report
    assert len(solver[0]) == count
    monkeypatch.setattr(Path, "open", original_open)
    assert file_sha256(directory / "holdout.csv") == sealed_hash
    with Evaluator(directory) as evaluator:
        assert len(evaluator.store.training_data()) == 6
        assert evaluator.store.model("iter_002")["metadata"]["training_data_hash"] != evaluator.store.model("iter_000")["metadata"]["training_data_hash"]


def test_retry_success_counts_once_and_terminal_failure_is_null(lab, solver):
    directory, _ = ready(lab)
    calls, modes = solver
    modes.extend(["failed", "success", "failed", "failed", "success"])
    report = run_optimize(directory)
    assert [row["status"] for row in report["history"]] == ["success", "failed", "success"]
    assert report["history"][0]["attempt_count"] == 2 and report["history"][1]["cd"] is None
    assert report["successful_sequential_points"] == 2 and report["failed_sequential_designs"] == 1
    assert report["budget"]["process_launches"] == 10
    failed = [row for row in csv.DictReader((directory / "history.csv").open(encoding="utf-8")) if row["status"] == "failed"]
    assert failed[0]["cd"] == ""


def test_consecutive_failure_stops_without_infinite_replacements(lab, solver):
    directory, _ = ready(lab)
    solver[1].extend(["failed"] * 6)
    report = run_optimize(directory)
    assert report["status"] == "stopped" and report["stop_reason"] == "consecutive_failed_designs"
    assert report["successful_sequential_points"] == 0 and report["failed_sequential_designs"] == 3
    assert report["budget"]["process_launches"] == 11


def test_model_boundary_resume_preserves_history_and_budget(lab, solver):
    directory, _ = ready(lab)
    with pytest.raises(StageIncomplete, match="boundary"):
        run_optimize(directory, stop_after_evaluations=1)
    with Evaluator(directory) as evaluator:
        state = evaluator.store.stage("optimize")
        assert len(state["history"]) == 1 and state["current_model_id"] == "iter_001"
        saved = deepcopy(state["history"])
        prior = evaluator.store.budget_snapshot()
    model_hash = file_sha256(directory / "models/iter_001/gp.joblib")
    write_json(directory / "checkpoint.json", {"history": [], "budget": 0})
    with pytest.raises(StageIncomplete, match="--resume"):
        run_optimize(directory)
    report = run_optimize(directory, resume=True)
    assert report["history"][:1] == saved and report["budget"]["process_launches"] == prior["process_launches"] + 1
    assert file_sha256(directory / "models/iter_001/gp.joblib") == model_hash


def test_completed_evaluation_crash_gap_is_committed_once(lab, solver, monkeypatch):
    directory, _ = ready(lab)
    original = Evaluator.evaluate

    def interrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if kwargs.get("purpose") == "sequential":
            raise KeyboardInterrupt("injected after evaluation DB commit")
        return result

    monkeypatch.setattr(Evaluator, "evaluate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_optimize(directory)
    with Evaluator(directory) as evaluator:
        state = evaluator.store.stage("optimize")
        assert not state["history"] and state["pending"]
        pending_id = state["pending"]["eval_id"]
        assert evaluator.store.get_result(pending_id).status == "success"
        (directory / state["pending"]["proposal_file"]).unlink()
    monkeypatch.setattr(Evaluator, "evaluate", original)
    report = run_optimize(directory, resume=True)
    assert report["history"][0]["eval_id"] == pending_id and len(report["history"]) == 2
    assert report["budget"]["process_launches"] == 7
    assert (directory / report["history"][0]["proposal_file"]).is_file()


@pytest.mark.parametrize("decision", ["accept", "reject"])
def test_explicit_review_controls_success_and_replacement(lab, solver, decision):
    directory, _ = ready(lab)
    solver[1].append("review")
    with pytest.raises(StageIncomplete, match="review required"):
        run_optimize(directory)
    with Evaluator(directory) as evaluator:
        state = evaluator.store.stage("optimize")
        assert not state["history"]
        evaluator.review(state["pending"]["eval_id"], decision, "explicit disposable-test decision", include_holdout=False)
    report = run_optimize(directory, resume=True)
    assert report["successful_sequential_points"] == 2
    assert report["failed_sequential_designs"] == (1 if decision == "reject" else 0)


def test_launch_failure_is_systemic_stop(lab, solver):
    directory, _ = ready(lab)
    solver[1].append("launch_failed")
    report = run_optimize(directory)
    assert report["stop_reason"] == "solver_or_cleanup_error" and report["status"] == "stopped"
    assert report["budget"]["process_launches"] == 5


def test_final_reserve_stops_before_any_new_launch(lab, solver):
    directory, _ = ready(lab, max_process_launches=17, reserved_final_verification_launches=12)
    count = len(solver[0])
    report = run_optimize(directory)
    assert report["status"] == "stopped" and not report["loop_updated"]
    assert report["stop_reason"] == "process_launch_budget_or_final_reserve" and len(solver[0]) == count


def test_completed_model_damage_rebuilds_without_new_solver_calls(lab, solver):
    directory, _ = ready(lab)
    original = run_optimize(directory)
    path = directory / "models/iter_002/gp.joblib"
    path.write_bytes(b"damaged derived model")
    count = len(solver[0])
    with pytest.raises(StageIncomplete, match="--resume"):
        run_optimize(directory)
    # A later verification may change the live budget while the M4 derived
    # artifact awaits repair. It must not be attributed to sequential work.
    with Evaluator(directory) as evaluator:
        assert evaluator.evaluate(purpose="verification").status == "success"
    rebuilt = run_optimize(directory, resume=True)
    assert rebuilt["history"] == original["history"] and rebuilt["budget"] == original["budget"]
    assert len(solver[0]) == count + 1
    assert list((directory / "models/iter_002/recovery").glob("*/gp.joblib"))


def test_raw_sequential_source_damage_cannot_be_hidden_by_retraining(lab, solver):
    directory, _ = ready(lab)
    original = run_optimize(directory)
    with Evaluator(directory) as evaluator:
        result = evaluator.store.get_result(original["history"][0]["eval_id"])
        Path(result.attempts[0]["runner"]["polar_path"]).write_bytes(b"damaged raw source")
    count = len(solver[0])
    with pytest.raises(RuntimeError, match="no longer validate"):
        run_optimize(directory, resume=True)
    assert len(solver[0]) == count


def test_completed_stage_keeps_budget_snapshot_after_later_verification(lab, solver):
    directory, _ = ready(lab)
    original = run_optimize(directory)
    with Evaluator(directory) as evaluator:
        result = evaluator.evaluate(purpose="verification")
        assert result.status == "success"
        assert evaluator.store.budget_snapshot()["process_launches"] == original["budget"]["process_launches"] + 1
    assert run_optimize(directory) == original


def test_iteration_counter_and_seed_follow_frozen_start(lab, solver):
    directory, _ = ready(lab, iteration_start=5)
    report = run_optimize(directory)
    assert [row["iteration"] for row in report["history"]] == [5, 6]
    for row in report["history"]:
        proposal = json.loads((directory / row["proposal_file"]).read_text())["proposal"]
        assert proposal["seed"] == 1000 + row["iteration"]
