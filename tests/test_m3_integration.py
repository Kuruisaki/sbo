"""Real M2 labels and raw artifacts, with an isolated M3 ledger/derived outputs.

The validation run inherits the source budget snapshot; it performs no solver
launch and does not claim to be an independently collected aerodynamic dataset.
"""
from datetime import datetime
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from uuid import uuid4

import pytest

from sbo.config import create_run, file_sha256, load_config, write_json
from sbo.evaluator import Evaluator
from sbo.fitting import run_fit
from sbo.stages import StageIncomplete

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


def test_real_training_model_and_cross_process_fold_resume():
    source = ROOT / "runs/m1_20261013_02"
    if not (source / "doe.json").is_file():
        pytest.skip("persisted real M2 dataset unavailable; this does not confirm M3 numerical validation")
    config = load_config(source / "config.frozen.yaml")
    label = "m3-real-validation-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    directory = create_run(config, label)
    original, copied = sqlite3.connect(source / "samples.sqlite"), sqlite3.connect(directory / "samples.sqlite")
    try:
        original.backup(copied)
        final = copied.execute("SELECT state_json FROM stages WHERE name='verify'").fetchone()
        if final:
            for slot in json.loads(final[0])["slots"]:
                for table in ("attempts", "events", "evaluations"):
                    copied.execute(f"DELETE FROM {table} WHERE eval_id=?", (slot["eval_id"],))
        copied.execute("DELETE FROM stages WHERE name IN ('verify','report')")
        copied.execute("DELETE FROM events WHERE kind='holdout_unsealed'")
        # Restore the M2 portion of this OWN validation copy when the source has
        # subsequently completed M4; retain the original solver files read-only.
        copied.execute("DELETE FROM attempts WHERE eval_id IN (SELECT eval_id FROM evaluations WHERE purpose='sequential')")
        copied.execute("DELETE FROM events WHERE eval_id IN (SELECT eval_id FROM evaluations WHERE purpose='sequential')")
        copied.execute("DELETE FROM evaluations WHERE purpose='sequential'")
        copied.execute("DELETE FROM designs WHERE design_id NOT IN (SELECT design_id FROM evaluations)")
        copied.execute("DELETE FROM stages WHERE name IN ('fit','optimize')")
        copied.execute("DELETE FROM models")
        copied.execute("DELETE FROM run_lease")
        copied.commit()
    finally:
        original.close()
        copied.close()
    with Evaluator(directory) as evaluator:
        for name in ("baseline", "doe"):
            for relative in evaluator.store.stage(name)["artifacts"]:
                destination = directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, destination)
        source_manifest = json.loads((source / "manifest.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest.update(preflight_passed=True, baseline_eval_id=source_manifest["baseline_eval_id"],
                        source_run=source_manifest["run_id"], validation_scope="Read-only real M2 artifacts; copied budget ledger; independent M3 derived files.")
        write_json(directory / "manifest.json", manifest)
        before = evaluator.store.budget_snapshot()
    with pytest.raises(StageIncomplete, match="boundary"):
        run_fit(directory, stop_after_folds=3)
    serialized_hash = file_sha256(directory / "models/iter_000/gp.joblib")
    with Evaluator(directory) as evaluator:
        saved = evaluator.store.stage("fit")
        assert len(saved["loo_folds"]) == 3
    child = subprocess.run([sys.executable, str(ROOT / "scripts/04_fit.py"), "--run-id", label, "--resume"],
                           capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    report = json.loads(child.stdout)
    assert report["training_points"] == 17 and report["loo"]["quality_passed"]
    assert report["budget_before"] == report["budget_after"] == before
    assert file_sha256(directory / "models/iter_000/gp.joblib") == serialized_hash
    assert report["proposal"]["exclusion_count"] == 23
    assert not report["proposal"]["assessed"] and not report["holdout_labels_used"]
    assert report["proposal"]["ei"] >= report["proposal"]["max_pool_ei"]
    with Evaluator(directory) as evaluator:
        completed = evaluator.store.stage("fit")
        assert len(completed["loo_folds"]) == 17 and completed["loo_folds"][:3] == saved["loo_folds"]
    assert run_fit(directory) == report
    write_json(directory / "m3-validation.json", {
        "status": "success", "source_run": str(source), "training_points": 17,
        "model_reused_byte_for_byte": True, "completed_folds_reused": 3, "remaining_folds_completed": 14,
        "cross_process_resume": True, "holdout_labels_used": False,
        "budget_unchanged": before, "proposal": report["proposal"]["design"],
    })
