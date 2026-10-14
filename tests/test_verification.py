"""M5 decision risks; modified archived output is confined to disposable runs."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from test_evaluator import lab, solver, ROOT
from test_optimizer import ready
from sbo.config import file_sha256, load_config, write_json
from sbo.evaluator import Evaluator
from sbo.optimizer import run_optimize
from sbo.reporting import run_report
from sbo.stages import StageIncomplete
from sbo.store import SQLiteEvaluationStore
from sbo.verification import certify, holdout_metrics, run_verify, shape_quality


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/naca2412.yaml")


def aero(values):
    return {key: {"cd": value} if value is not None else None
            for key, value in zip(("160", "200", "240", "200_repeat"), values)}


def pressure():
    return {key: {"passed": True} for key in ("160", "200", "240", "200_repeat")}


def shape(role, values, config):
    a = aero(values)
    return {"role": role, "design_id": role, "aerodynamics": a, "quality": shape_quality(a, pressure(), config)}


def test_metrics_and_resolution_limited_r2(config):
    metrics = holdout_metrics([.006, .007], [.0061, .0068], [.0001, .0001], config, .006)
    assert metrics["mae"] == pytest.approx(.00015)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5e-8))
    assert metrics["max_absolute_error"] == pytest.approx(.0002)
    assert metrics["r2"] == pytest.approx(.9)
    assert metrics["quality_passed"] and metrics["two_sigma_coverage"] == 1
    tiny = holdout_metrics([.006, .006], [.006, .006], [0, 0], config, .006)
    assert tiny["r2"] is None and tiny["r2_not_applicable_reason"]
    assert tiny["two_sigma_coverage"] == 1
    poor = holdout_metrics([.006], [.007], [.00001], config, .006)
    assert not poor["quality_passed"] and poor["two_sigma_coverage"] == 0


@pytest.mark.parametrize("y,mean,std", [([], [], []), ([np.nan], [.006], [0]), ([.006], [.006], [-1]),
    ([.006], [.006, .006], [0]), ([0], [.006], [0])])
def test_invalid_metrics_never_serialize_nonfinite(config, y, mean, std):
    with pytest.raises(ValueError):
        holdout_metrics(y, mean, std, config, .006)


def test_shape_quality_uses_all_three_differences_and_repeat(config):
    q = shape_quality(aero([.006, .00604, .00606, .006045]), pressure(), config)
    assert q["passed"] and q["epsilon"] == pytest.approx(.00004)
    assert q["panel_threshold"] == pytest.approx(.0000606)
    q = shape_quality(aero([.006, .00604, .00614, .00604]), pressure(), config)
    assert not q["passed"] and not q["panel_convergence"]
    q = shape_quality(aero([.006, .006, .006, .00602]), pressure(), config)
    assert not q["repeatability"]
    assert not shape_quality(aero([.006, None, .006, .006]), pressure(), config)["passed"]
    p = pressure()
    p["240"]["passed"] = False
    assert not shape_quality(aero([.006] * 4), p, config)["passed"]


def test_same_fidelity_ranking_and_certification(config):
    shapes = [shape("baseline", [.006] * 4, config),
              shape("candidate_1", [.0055, .0055, .00555, .0055], config),
              shape("candidate_2", [.0056, .0056, .00554, .0056], config)]
    # candidate_2 fails panel threshold; it cannot win despite lower CD240.
    result = certify(shapes)
    assert result["ranked_valid_candidate_ids"] == ["candidate_1"]
    assert result["improvement_certified"] and result["recommendation_design_id"] == "candidate_1"
    assert result["comparisons"][0]["delta_cd_160"] == pytest.approx(.0005)
    # Exact equality is insufficient for the strict gain gate.
    shapes[1]["quality"]["epsilon"] = .006 - .00555 - shapes[0]["quality"]["epsilon"]
    result = certify(shapes)
    assert not result["improvement_certified"] and result["recommendation_design_id"] == "baseline"
    shapes[0]["quality"]["passed"] = False
    assert certify(shapes)["recommendation_design_id"] is None


@pytest.fixture
def final_run(lab, solver, monkeypatch):
    directory, config = ready(lab)
    run_optimize(directory)
    import sbo.evaluator as module
    original = module.run_xfoil

    def panels(geometry, settings, path, **kwargs):
        result = original(geometry, settings, path, **kwargs)
        # Unit test-only synthesized higher-N Cp and panel report. Actual physics
        # and independent multi-panel behavior are checked by integration tests.
        if settings.panel_count != 160:
            log = result.stdout_path.read_text()
            result.stdout_path.write_text(log.replace("160", str(settings.panel_count)))
            if result.cp_path and result.cp_path.is_file():
                values = np.loadtxt(result.cp_path, skiprows=3)
                x = np.linspace(0, 1, len(values))
                xi = np.linspace(0, 1, settings.panel_count)
                sampled = np.column_stack([np.interp(xi, x, values[:, i]) for i in range(3)])
                np.savetxt(result.cp_path, sampled, header="x y Cp", comments="# ")
        write_json(result.process_path, result.to_dict())
        return result

    monkeypatch.setattr(module, "run_xfoil", panels)
    return directory, config


def test_holdout_opens_only_after_committed_model_candidate_freeze(final_run, solver, monkeypatch):
    directory, _ = final_run
    old = SQLiteEvaluationStore._dataset

    def guarded(self, split, **kwargs):
        if split == "holdout":
            state = self.stage("verify")
            assert state["freeze"]["model"]["model_id"] == "iter_002"
            assert len(state["freeze"]["shapes"]) == 3 and len(state["slots"]) == 12
            assert self.connection.execute("SELECT COUNT(*) FROM events WHERE kind='holdout_unsealed'").fetchone()[0] == 1
        return old(self, split, **kwargs)

    monkeypatch.setattr(SQLiteEvaluationStore, "_dataset", guarded)
    count = len(solver[0])
    sealed = file_sha256(directory / "holdout.csv")
    report = run_verify(directory)
    assert report["status"] == "success" and len(solver[0]) == count + 12
    assert not report["certification"]["improvement_certified"]
    assert report["certification"]["recommendation_reason"] == "retain_baseline_no_certified_improvement"
    with Evaluator(directory) as e:
        assert len(e.store.training_data()) == 6
        assert e.store.budget_snapshot()["distinct_standard_designs"] == 7
        slots = e.store.stage("verify")["slots"]
        assert len({slot["eval_id"] for slot in slots}) == 12
        assert all(e.store.get_request(slot["eval_id"])["use_reserved_budget"] for slot in slots)
    assert run_verify(directory) == report and len(solver[0]) == count + 12
    final = run_report(directory)
    assert final["m5_passed"] and final["recommendation"]["m"] == .02
    assert len(solver[0]) == count + 12 and file_sha256(directory / "holdout.csv") == sealed
    assert file_sha256(directory / "best_airfoil.dat") == final["recommendation"]["file_sha256"]
    assert run_verify(directory) == report
    with Evaluator(directory) as e:
        with pytest.raises(ValueError, match="forbids new"):
            e.evaluate([.019, .39], purpose="sequential")
        record = deepcopy(e.store.model("iter_002"))
        record["metadata"]["fit"]["seed"] = 999
        with pytest.raises(ValueError, match="frozen M5"):
            e.store.save_model_record(record)
        corrupted = e.store.stage("verify")
        # Both slots request N=200, so ordinary fidelity checks alone would not
        # catch reusing the first result as an independent repeat.
        corrupted["slots"][3]["eval_id"] = corrupted["slots"][1]["eval_id"]
        e.store.save_stage("verify", "success", corrupted)
    with pytest.raises(RuntimeError, match="slot identities"):
        run_verify(directory, resume=True)


def test_cross_stage_resume_and_derived_report_repair(final_run, solver):
    directory, _ = final_run
    before = len(solver[0])
    with pytest.raises(StageIncomplete, match="boundary"):
        run_verify(directory, stop_after_evaluations=4)
    with Evaluator(directory) as e:
        original = deepcopy(e.store.stage("verify"))
        assert [slot["status"] for slot in original["slots"]].count("success") == 4
    with pytest.raises(StageIncomplete, match="--resume"):
        run_verify(directory)
    report = run_verify(directory, resume=True)
    assert len(solver[0]) == before + 12
    with Evaluator(directory) as e:
        resumed = e.store.stage("verify")
        assert resumed["freeze_hash"] == original["freeze_hash"] and resumed["slots"][:4] == original["slots"][:4]
    (directory / "verification.csv").write_text("damaged derived report")
    with pytest.raises(StageIncomplete, match="--resume"):
        run_verify(directory)
    assert run_verify(directory, resume=True) == report
    assert list((directory / "report-recovery").glob("*/verification.csv"))
    assert len(solver[0]) == before + 12


@pytest.mark.parametrize("decision", ["accept", "reject"])
def test_review_pauses_until_explicit_decision(final_run, solver, decision):
    directory, _ = final_run
    solver[1].append("review")
    before = len(solver[0])
    with pytest.raises(StageIncomplete) as caught:
        run_verify(directory)
    assert caught.value.status == "needs_review" and len(solver[0]) == before + 1
    with pytest.raises(StageIncomplete):
        run_verify(directory, resume=True)
    assert len(solver[0]) == before + 1
    with Evaluator(directory) as e:
        slot = e.store.stage("verify")["slots"][0]
        reviewed = e.review(slot["eval_id"], decision, "explicit unit test decision", include_holdout=False)
        if decision == "accept":
            assert e._cache_valid(reviewed, True)
    report = run_verify(directory, resume=True)
    final = run_report(directory)
    if decision == "reject":
        assert report["status"] == "failed" and report["shapes"][0]["aerodynamics"]["160"] is None
        assert report["certification"]["recommendation_design_id"] is None
        assert final["status"] == "failed" and not (directory / "best_airfoil.dat").exists()
    else:
        assert report["status"] == "success" and (directory / "best_airfoil.dat").is_file()


def test_raw_evidence_damage_removes_stale_recommendation(final_run):
    directory, _ = final_run
    run_verify(directory)
    assert run_report(directory)["recommendation"]
    with Evaluator(directory) as e:
        slot = e.store.stage("verify")["slots"][0]
        result = e.store.get_result(slot["eval_id"])
        Path(result.attempts[0]["runner"]["polar_path"]).write_text("damaged original")
    with pytest.raises(RuntimeError, match="no longer validates"):
        run_verify(directory, resume=True)
    final = run_report(directory)
    assert final["status"] == "incomplete" and final["recommendation"] is None
    assert not (directory / "best_airfoil.dat").exists()
    assert list((directory / "report-recovery").glob("*/best_airfoil.dat"))


def test_short_budget_stops_before_unsealing_or_launch(final_run, solver, monkeypatch):
    directory, _ = final_run
    original = SQLiteEvaluationStore.budget_snapshot

    def shortage(self):
        snapshot = original(self)
        snapshot["max_process_launches"] = snapshot["process_launches"] + 11
        return snapshot

    monkeypatch.setattr(SQLiteEvaluationStore, "budget_snapshot", shortage)
    before = len(solver[0])
    with pytest.raises(StageIncomplete, match="insufficient"):
        run_verify(directory)
    assert len(solver[0]) == before
    with Evaluator(directory) as e:
        assert e.store.stage("verify") is None
        with pytest.raises(ValueError, match="committed"):
            e.store.final_holdout_data("wrong")


def test_incomplete_report_is_diagnostic_without_label_access(final_run, solver, monkeypatch):
    directory, _ = final_run
    monkeypatch.setattr(SQLiteEvaluationStore, "final_holdout_data", lambda *_: pytest.fail("sealed labels opened"))
    before = len(solver[0])
    report = run_report(directory)
    assert report["status"] == "incomplete" and not report["m5_passed"] and report["recommendation"] is None
    assert len(solver[0]) == before and (directory / "report.md").is_file()


def test_poor_holdout_is_reported_without_refit_or_candidate_reselection(final_run, solver, monkeypatch):
    directory, config = final_run
    import sbo.verification as module
    original = module._holdout

    def poor(*args):
        result = original(*args)
        for row in result["predictions"]:
            row.update(predicted_cd=row["observed_cd"] + .001, residual_cd=-.001)
        result["metrics"] = holdout_metrics([.00599], [.00699], [.00001], config, .00599)
        return result

    monkeypatch.setattr(module, "_holdout", poor)
    before = len(solver[0])
    with Evaluator(directory) as e:
        model_sha = e.store.model("iter_002")["metadata"]["artifact"]["sha256"]
    report = run_verify(directory)
    assert report["status"] == "failed" and report["checks"]["all_shapes_verified"]
    assert not report["checks"]["independent_holdout"] and len(solver[0]) == before + 12
    final = run_report(directory)
    assert final["status"] == "failed" and not final["m5_passed"]
    assert final["recommendation"]["m"] == .02
    with Evaluator(directory) as e:
        assert len(e.store.training_data()) == 6
        assert e.store.model("iter_002")["metadata"]["artifact"]["sha256"] == model_sha
