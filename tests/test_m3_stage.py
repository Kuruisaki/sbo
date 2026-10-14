"""M3 lifecycle tests use archived solver output in disposable test runs only."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from test_evaluator import lab, solver, ROOT
import sbo.fitting as fitting
from sbo.config import FrozenConfig, file_sha256, load_config, write_json
from sbo.evaluator import Evaluator
from sbo.fitting import run_fit
from sbo.sampling import normalize
from sbo.stages import StageIncomplete
from sbo.surrogate import digest, SurrogateFitError
from sbo.config import create_run


def ready(lab):
    values = load_config(ROOT / "configs/naca2412.yaml").to_dict()
    values["runtime"]["runs_root"] = str(lab)
    values["doe"]["initial_points"] = 3
    values["doe"]["holdout_points"] = 1
    values["gp"]["n_restarts_optimizer"] = 0
    config = FrozenConfig(values)
    directory = create_run(config, "m3")
    with Evaluator(directory) as evaluator:
        baseline = evaluator.evaluate(save_cp=True)
        slots = []
        for i, design in enumerate([[.012, .28], [.03, .5], [.038, .59]]):
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
    return directory, config


def test_m3_stage_does_not_read_holdout_labels_or_launch_solver(lab, solver, monkeypatch):
    directory, config = ready(lab)
    calls, _ = solver
    count = len(calls)
    actual = Evaluator.__init__

    def guarded_init(self, *args, **kwargs):
        actual(self, *args, **kwargs)
        # An authorizer proves that M3 cannot select evaluation data joined to holdout.
        # The public coordinate query is allowed; every training query remains explicit.
        raw = self.store._dataset

        def only_training(split, **kwargs):
            assert split == "train", "sealed labels were queried"
            return raw(split, **kwargs)

        self.store._dataset = only_training
        source_get = self.store.get_result
        holdout_id = self.store.connection.execute("SELECT e.eval_id FROM evaluations e JOIN designs d USING(design_id) WHERE d.split='holdout'").fetchone()[0]

        def guarded_result(eval_id):
            assert eval_id != holdout_id, "holdout aerodynamic result was opened"
            return source_get(eval_id)

        self.store.get_result = guarded_result

    monkeypatch.setattr(Evaluator, "__init__", guarded_init)
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        assert self.name != "holdout.csv", "holdout label CSV was opened"
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = run_fit(directory)
    assert report["training_points"] == 4 and not report["holdout_labels_used"]
    assert report["budget_before"] == report["budget_after"] and len(calls) == count
    proposal = report["proposal"]
    assert not proposal["assessed"] and proposal["min_excluded_distance"] >= .001
    assert proposal["exclusion_count"] == 5
    assert len(json.loads((directory / "models/iter_000/loo.json").read_text())["folds"]) == 4
    with Evaluator(directory) as evaluator:
        assert len(evaluator.store.training_data()) == 4
        assert evaluator.store.model("iter_000")["metadata"]["training_data_hash"] == report["training_data_hash"]
        assert evaluator.store.connection.execute("SELECT COUNT(*) FROM designs").fetchone()[0] == 5
    again = run_fit(directory)
    assert again == report and len(calls) == count


def test_fold_resume_reuses_model_and_completed_folds(lab, solver, monkeypatch):
    directory, _ = ready(lab)
    with pytest.raises(StageIncomplete, match="boundary"):
        run_fit(directory, stop_after_folds=2)
    model_path = directory / "models/iter_000/gp.joblib"
    model_hash = file_sha256(model_path)
    with Evaluator(directory) as evaluator:
        saved = evaluator.store.stage("fit")
        assert len(saved["loo_folds"]) == 2
        before = evaluator.store.budget_snapshot()
    write_json(directory / "checkpoint.json", {"fake": "never used as recovery truth"})
    with pytest.raises(StageIncomplete, match="--resume"):
        run_fit(directory)
    # Remaining folds fit through surrogate.fit_gp, whereas the full fit must load.
    monkeypatch.setattr(fitting, "fit_gp", lambda *_: pytest.fail("full GP must be reused"))
    report = run_fit(directory, resume=True)
    assert file_sha256(model_path) == model_hash
    assert report["budget_after"] == before
    folds = json.loads((directory / "models/iter_000/loo.json").read_text())["folds"]
    assert folds[:2] == saved["loo_folds"]


def test_changed_model_requires_resume_and_is_rebuilt_from_same_training(lab, solver):
    directory, _ = ready(lab)
    before = run_fit(directory)
    path = directory / "models/iter_000/gp.joblib"
    path.write_bytes(b"deliberately invalid derived artifact")
    with pytest.raises(StageIncomplete, match="--resume"):
        run_fit(directory)
    rebuilt = run_fit(directory, resume=True)
    assert rebuilt["training_data_hash"] == before["training_data_hash"]
    assert rebuilt["budget_after"] == before["budget_after"]
    np.testing.assert_allclose(rebuilt["proposal"]["design"], before["proposal"]["design"], atol=1e-12)
    assert list((directory / "models/iter_000/recovery").glob("*/gp.joblib"))
    assert run_fit(directory)["status"] == "success"


def test_raw_training_damage_blocks_rebuild(lab, solver):
    directory, _ = ready(lab)
    with Evaluator(directory) as evaluator:
        row = evaluator.store.training_data()[1]
        result = evaluator.store.get_result(row["eval_id"])
        Path(result.attempts[0]["runner"]["polar_path"]).write_bytes(b"corrupt source data")
    with pytest.raises(RuntimeError, match="no longer validate"):
        run_fit(directory, resume=True)
    assert not (directory / "models/iter_000/gp.joblib").exists()


def test_database_label_change_cannot_replace_solver_evidence(lab, solver):
    directory, _ = ready(lab)
    with Evaluator(directory) as evaluator:
        row = evaluator.store.training_data()[1]
        result = evaluator.store.get_result(row["eval_id"])
        corrupted = replace(result, aerodynamics={**result.aerodynamics, "cd": .001})
        evaluator.store.finish_evaluation(corrupted)
    with pytest.raises(RuntimeError, match="labels differ"):
        run_fit(directory)


def test_model_registry_rejects_identity_reuse_for_other_training(lab, solver):
    directory, config = ready(lab)
    with Evaluator(directory) as evaluator:
        record = {"model_id": "test", "config_hash": config.config_hash, "metadata": {
                  "training_data_hash": "one", "artifact": {"sha256": "binary"}}}
        evaluator.store.save_model_record(record)
        record["metadata"]["training_data_hash"] = "two"
        with pytest.raises(ValueError, match="different training"):
            evaluator.store.save_model_record(record)


def test_failed_fit_does_not_open_sealed_dataset(lab, solver, monkeypatch):
    directory, _ = ready(lab)
    csv_hash = file_sha256(directory / "config.frozen.yaml")

    def fail(*args):
        raise SurrogateFitError("injected exhausted Cholesky fallback", [{"alpha_regularization": 1e-4}])

    monkeypatch.setattr(fitting, "fit_gp", fail)
    monkeypatch.setattr(fitting.Evaluator, "evaluate", lambda *args, **kwargs: pytest.fail("M3 failure cannot launch solver"))
    from sbo.store import SQLiteEvaluationStore
    monkeypatch.setattr(SQLiteEvaluationStore, "export_csv", lambda *args: pytest.fail("M3 failure cannot open holdout labels"))
    with pytest.raises(SurrogateFitError):
        run_fit(directory)
    assert file_sha256(directory / "config.frozen.yaml") == csv_hash
    with Evaluator(directory) as evaluator:
        assert evaluator.store.stage("fit")["status"] == "failed"
        assert evaluator.store.stage("fit")["fit_failure_attempts"]
