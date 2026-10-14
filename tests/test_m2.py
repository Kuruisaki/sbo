"""M2 state, separation, budget and crash-gap tests use isolated archived-output runs."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path

import numpy as np
import psutil
import pytest
from scipy.spatial.distance import pdist

from test_evaluator import lab, solver, make_run
from sbo.baseline import geometry_diagnostics
from sbo.config import canonical_json, file_sha256, write_json
from sbo.doe import run_doe
from sbo.evaluator import Evaluator
from sbo.sampling import (denormalize, distance_filter, initial_plan, lhs, maximin,
                          nearest_distances, normalize, replace_slot, sobol)
from sbo.stages import StageIncomplete, run_ownership
from sbo.store import SQLiteEvaluationStore


def ready_run(lab, name, solver, **limits):
    directory, config = make_run(lab, name, **limits)
    with Evaluator(directory) as evaluator:
        baseline = evaluator.evaluate(save_cp=True)
        assert baseline.usable_for_training
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest.update(preflight_passed=True, baseline_eval_id=baseline.eval_id)
        write_json(directory / "manifest.json", manifest)
        evaluator.store.save_stage("baseline", "success", {"evaluations": {"160": baseline.eval_id}, "artifacts": {}})
    return directory, config


def test_seeded_lhs_sobol_and_normalization(lab):
    _, config = make_run(lab)
    for n, seed in ((16, 42), (6, 2026)):
        points = lhs(n, seed)
        np.testing.assert_array_equal(points, lhs(n, seed))
        assert all(len(np.unique(np.floor(points[:, col] * n))) == n for col in range(2))
        physical = denormalize(points, config["design"]["bounds"])
        np.testing.assert_allclose(normalize(physical, config["design"]["bounds"]), points, atol=1e-15)
    np.testing.assert_array_equal(sobol(16, 10), sobol(16, 10))
    with pytest.raises(ValueError):
        sobol(15, 10)
    with pytest.raises(ValueError):
        normalize([[.9, .4]], config["design"]["bounds"])


def test_plan_reserves_holdout_and_replacements_restore_rng(lab):
    _, config = make_run(lab)
    plan = initial_plan(config)
    assert len(plan["slots"]) == 22 and len(plan["excluded"]) == 23
    assert plan["strict_lhs"] == {"train": True, "holdout": True}
    assert pdist(np.array(plan["excluded"])).min() >= .001
    restored = json.loads(canonical_json(plan))
    replace_slot(plan, plan["slots"][7], config)
    replace_slot(restored, restored["slots"][7], config)
    assert plan == restored and plan["slots"][7]["generation"] == 1
    assert not plan["strict_lhs"]["train"] and plan["strict_lhs"]["holdout"]
    assert nearest_distances([plan["slots"][7]["u"]], plan["excluded"][:-1])[0] >= .001


def test_distance_excludes_failed_reserved_and_each_accepted_candidate():
    points = distance_filter([[0, 0], [.0005, 0], [.5, .5], [.5001, .5]], [[0, 0]], .001)
    np.testing.assert_array_equal(points, [[.5, .5]])
    with pytest.raises(RuntimeError):
        maximin([[.5, .5]], np.random.default_rng(1), 2)


def test_m1_ledger_migrates_without_reset(lab, solver):
    directory, config = make_run(lab)
    with Evaluator(directory) as evaluator:
        baseline = evaluator.evaluate()
        before = evaluator.store.budget_snapshot()
        evaluator.store.connection.execute("UPDATE metadata SET value='m1-ledger-v1' WHERE key='schema_version'")
    store = SQLiteEvaluationStore(directory, config)
    assert store.budget_snapshot() == before
    assert store.get_result(baseline.eval_id).status == "success"
    assert store.connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0] == store.VERSION
    store.close()


@pytest.mark.parametrize("lose_launch", [False, True])
def test_completed_process_crash_gap_imports_without_launch(lab, solver, lose_launch):
    directory, _ = make_run(lab)
    calls, _ = solver
    with Evaluator(directory) as evaluator:
        original = evaluator.evaluate(evaluation_id="eval_crash", save_cp=True)
        db = evaluator.store.connection
        db.execute("UPDATE evaluations SET result_json=NULL,status='running'")
        db.execute("UPDATE attempts SET record_json=NULL,status=?,launched=?", ("reserved" if lose_launch else "running", 0 if lose_launch else 1))
        original.result_path.unlink()
    with Evaluator(directory) as evaluator:
        with pytest.raises(RuntimeError, match="unfinished"):
            evaluator.evaluate(save_cp=True)
        recovered = evaluator.resume("eval_crash")
        assert recovered.status == "success" and recovered.eval_id == original.eval_id
        assert recovered.aerodynamics == original.aerodynamics
        assert evaluator.store.budget_snapshot()["process_launches"] == 1
        assert evaluator.store.budget_snapshot()["pending_launch_reservations"] == 0
    assert len(calls) == 1


def test_dead_interrupted_attempt_uses_only_remaining_retry(lab, solver):
    directory, _ = make_run(lab)
    calls, _ = solver
    with Evaluator(directory) as evaluator:
        original = evaluator.evaluate(evaluation_id="eval_crash")
        audit_path = Path(original.attempts[0]["runner"]["process_path"])
        write_json(audit_path, {"status": "running", "launched": True,
                               "pid": os.getpid(), "process_create_time": psutil.Process().create_time() - 10})
        # PID has been reused: do not kill this test process or import its stale logs.
        evaluator.store.connection.execute("UPDATE evaluations SET result_json=NULL,status='running'")
        evaluator.store.connection.execute("UPDATE attempts SET record_json=NULL,status='running'")
    with Evaluator(directory) as evaluator:
        recovered = evaluator.resume("eval_crash")
        assert recovered.selected_attempt == 2 and recovered.status == "success"
        assert recovered.attempts[0]["status"] == "interrupted"
        assert evaluator.store.budget_snapshot()["process_launches"] == 2
    assert len(calls) == 2 and calls[1].attempt_number == 2


def test_live_solver_identity_blocks_resume(lab, solver):
    directory, _ = make_run(lab)
    with Evaluator(directory) as evaluator:
        result = evaluator.evaluate(evaluation_id="eval_live")
        path = Path(result.attempts[0]["runner"]["process_path"])
        write_json(path, {"status": "running", "launched": True, "pid": os.getpid(),
                          "process_create_time": psutil.Process().create_time()})
        evaluator.store.connection.execute("UPDATE evaluations SET result_json=NULL,status='running'")
        evaluator.store.connection.execute("UPDATE attempts SET record_json=NULL,status='running'")
        with pytest.raises(RuntimeError, match="still alive"):
            evaluator.resume("eval_live")
        assert evaluator.store.budget_snapshot()["process_launches"] == 1


def test_resume_rejects_changed_request_and_budget_is_not_reset(lab, solver):
    directory, _ = make_run(lab, max_process_launches=2, reserved_final_verification_launches=1)
    with Evaluator(directory) as evaluator:
        result = evaluator.evaluate(evaluation_id="eval_frozen")
        with pytest.raises(ValueError, match="differs"):
            evaluator.evaluate([.03, .4], evaluation_id=result.eval_id, resume=True)
        path = Path(result.attempts[0]["runner"]["process_path"])
        write_json(path, {"status": "running", "launched": True})
        evaluator.store.connection.execute("UPDATE evaluations SET result_json=NULL,status='running'")
        evaluator.store.connection.execute("UPDATE attempts SET record_json=NULL,status='running'")
        recovered = evaluator.resume(result.eval_id)
        assert recovered.status == "failed" and recovered.aerodynamics is None
        assert recovered.failure_reason.startswith("retry_budget_exhausted")
        assert evaluator.store.budget_snapshot()["process_launches"] == 1


def test_store_exports_null_failures_and_separates_holdout(lab, solver):
    directory, _ = make_run(lab)
    _, modes = solver
    with Evaluator(directory) as evaluator:
        evaluator.evaluate()
        modes.extend(["failed", "failed"])
        failure = evaluator.evaluate([.03, .4], purpose="doe")
        evaluator.evaluate([.024, .44], split="holdout", purpose="doe")
        evaluator.evaluate(purpose="verification")
        files = evaluator.store.export_csv()
        assert files["samples.csv"]["rows"] == 1 and files["holdout.csv"]["rows"] == 1
        assert len(evaluator.store.training_data()) == 1
        assert "cd" not in evaluator.store.holdout_coordinates()[0]
        failed = list(csv.DictReader((directory / "failures.csv").open(encoding="utf-8")))[0]
        assert failed["cd"] == "" and failed["cl"] == "" and failed["eval_id"] == failure.eval_id
        assert "holdout" not in (directory / "samples.csv").read_text()


@pytest.mark.parametrize("decision", ["accept", "reject"])
def test_explicit_review_event_controls_training(lab, solver, decision):
    directory, _ = make_run(lab)
    calls, modes = solver
    modes.append("review")
    with Evaluator(directory) as evaluator:
        result = evaluator.evaluate()
        assert len(evaluator.store.training_data()) == 0
        updated = evaluator.review(result.eval_id, decision, "explicit unit-test decision, not production acceptance")
        assert len(evaluator.store.training_data()) == (1 if decision == "accept" else 0)
        assert evaluator.evaluate().status == updated.status
        assert evaluator.store.connection.execute("SELECT COUNT(*) FROM events WHERE kind LIKE 'review_%'").fetchone()[0] == 1
    assert len(calls) == 1


def test_stage_ownership_blocks_concurrent_runner(lab, solver):
    directory, config = make_run(lab)
    with Evaluator(directory) as evaluator, run_ownership(evaluator.store):
        other = SQLiteEvaluationStore(directory, config)
        with pytest.raises(RuntimeError, match="owns"):
            with run_ownership(other):
                pass
        other.close()


def test_stage_prerequisites_and_reference(lab, solver):
    directory, config = make_run(lab)
    with pytest.raises(RuntimeError, match="preflight"):
        run_doe(directory)
    diagnostics, _ = geometry_diagnostics(config)
    assert diagnostics["passed"]


def test_doe_resume_matches_uninterrupted_with_failed_replacement(lab, solver):
    calls, modes = solver
    whole, _ = ready_run(lab, "whole", solver)
    modes.extend(["failed", "failed"])
    first = run_doe(whole)
    partial, _ = ready_run(lab, "partial", solver)
    modes.extend(["failed", "failed"])
    with pytest.raises(StageIncomplete, match="boundary"):
        run_doe(partial, stop_after=3)
    with Evaluator(partial) as evaluator:
        before = evaluator.store.budget_snapshot()
        persisted = evaluator.store.stage("doe")
    # A missing/tampered JSON projection never drives recovery or changes proposals.
    write_json(partial / "checkpoint.json", {"budget": {"process_launches": 0}})
    with pytest.raises(StageIncomplete, match="--resume"):
        run_doe(partial)
    report = run_doe(partial, resume=True)
    assert report["training_points"] == first["training_points"] == 17
    assert report["sealed_holdout_points"] == first["sealed_holdout_points"] == 6
    assert report["replacement_designs"] == 1 and not report["strict_lhs"]["train"]
    assert report["budget"] == first["budget"]
    plans = [json.loads((path / "doe-plan.json").read_text()) for path in (whole, partial)]
    assert [slot["design"] for slot in plans[0]["slots"]] == [slot["design"] for slot in plans[1]["slots"]]
    assert plans[0]["rng_state"] == plans[1]["rng_state"]
    assert persisted["slots"][6]["history"]  # First train slot failed, its coordinate remains excluded.
    count = len(calls)
    assert run_doe(partial)["budget"] == report["budget"]
    assert len(calls) == count and before["process_launches"] < report["budget"]["process_launches"]


def test_doe_budget_stops_with_incomplete_quota(lab, solver):
    directory, _ = ready_run(lab, "cap", solver, max_distinct_standard_designs=2)
    calls, _ = solver
    with pytest.raises(StageIncomplete, match="quota incomplete"):
        run_doe(directory)
    with Evaluator(directory) as evaluator:
        assert evaluator.store.budget_snapshot()["process_launches"] == 2
        assert evaluator.store.stage("doe")["status"] == "skipped"
    assert len(calls) == 2


def test_doe_review_stops_then_explicit_reject_replaces(lab, solver):
    directory, _ = ready_run(lab, "review", solver)
    _, modes = solver
    modes.append("review")
    with pytest.raises(StageIncomplete, match="review required"):
        run_doe(directory)
    with Evaluator(directory) as evaluator:
        state = evaluator.store.stage("doe")
        slot = next(slot for slot in state["slots"] if slot["status"] == "needs_review")
        evaluator.review(slot["eval_id"], "reject", "explicit test rejection")
    report = run_doe(directory, resume=True)
    assert report["training_points"] == 17 and report["replacement_designs"] == 1


def test_doe_launch_failure_does_not_replace_forever(lab, solver):
    directory, _ = ready_run(lab, "spawn", solver)
    calls, modes = solver
    modes.append("launch_failed")
    with pytest.raises(StageIncomplete, match="no replacement launch"):
        run_doe(directory)
    assert len(calls) == 2
    with Evaluator(directory) as evaluator:
        assert evaluator.store.stage("doe")["status"] == "failed"
        assert evaluator.store.budget_snapshot()["process_launches"] == 1


def test_recovery_refuses_changed_committed_output(lab, solver):
    directory, _ = make_run(lab)
    with Evaluator(directory) as evaluator:
        result = evaluator.evaluate(evaluation_id="eval_committed")
        evaluator.store.connection.execute("UPDATE evaluations SET result_json=NULL,status='running'")
        path = Path(result.attempts[0]["runner"]["stdout_path"])
        path.write_bytes(path.read_bytes() + b'\nchanged committed output\n')
        with pytest.raises(RuntimeError, match="committed hash"):
            evaluator.resume(result.eval_id)


def test_initial_plan_excludes_existing_failed_coordinates(lab):
    _, config = make_run(lab)
    existing = lhs(16, 42)[0]
    plan = initial_plan(config, excluded=[existing])
    assert plan["slots"][6]["method"] == "maximin"
    assert not plan["strict_lhs"]["train"]
    assert nearest_distances([plan["slots"][6]["u"]], [existing])[0] >= .001


def test_keyboard_interrupt_cannot_start_a_fresh_attempt_budget(lab, solver, monkeypatch):
    import sbo.evaluator as module
    directory, _ = make_run(lab)
    actual = module.run_xfoil

    def interrupt_after_exit(*args, **kwargs):
        actual(*args, **kwargs)
        raise KeyboardInterrupt("injected after finalized process audit")

    monkeypatch.setattr(module, "run_xfoil", interrupt_after_exit)
    with Evaluator(directory) as evaluator:
        with pytest.raises(KeyboardInterrupt):
            evaluator.evaluate(evaluation_id="eval_interrupt")
        with pytest.raises(RuntimeError, match="resume"):
            evaluator.evaluate()
        result = evaluator.resume("eval_interrupt")
        assert result.status == "success"
        assert evaluator.store.budget_snapshot()["process_launches"] == 1
