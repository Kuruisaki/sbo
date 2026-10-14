"""Real M5 solves recover a hard exit after evaluation commit, before slot commit."""
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
from sbo.reporting import run_report
from sbo.stages import StageIncomplete
from sbo.verification import run_verify

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


def test_real_final_verification_recovers_commit_gap_and_certifies():
    source = ROOT / "runs/m1_20261013_02"
    if not (source / "optimization.json").is_file():
        pytest.skip("real M4 source unavailable; numerical M5 remains unverified by this test")
    config = load_config(source / "config.frozen.yaml")
    if not Path(config["xfoil"]["executable"]).is_file():
        pytest.skip("actual configured XFOIL unavailable")
    directory = create_run(config, "m5-real-validation-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
    original, copied = sqlite3.connect(source / "samples.sqlite"), sqlite3.connect(directory / "samples.sqlite")
    try:
        original.backup(copied)
        old = copied.execute("SELECT state_json FROM stages WHERE name='verify'").fetchone()
        if old:
            # Remove only M5 slots from this OWN copy; retain inherited M1/M2
            # verification calls, training/holdout and their historical budgets.
            for slot in json.loads(old[0])["slots"]:
                for table in ("attempts", "events", "evaluations"):
                    copied.execute(f"DELETE FROM {table} WHERE eval_id=?", (slot["eval_id"],))
        copied.execute("DELETE FROM stages WHERE name IN ('verify','report')")
        copied.execute("DELETE FROM events WHERE kind='holdout_unsealed'")
        copied.execute("DELETE FROM run_lease")
        copied.commit()
    finally:
        original.close()
        copied.close()
    with Evaluator(directory) as evaluator:
        for stage in ("baseline", "doe", "fit", "optimize"):
            for relative in evaluator.store.stage(stage)["artifacts"]:
                destination = directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, destination)
        model_id = evaluator.store.stage("optimize")["current_model_id"]
        shutil.copytree(source / "models" / model_id, directory / "models" / model_id)
        source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(preflight_passed=True, m4_passed=True, baseline_eval_id=source_manifest["baseline_eval_id"],
                        source_run=str(source), validation_scope="Inherited M1-M4 ledger; read-only source artifacts; twelve new real verification solves.")
        write_json(directory / "manifest.json", manifest)
        before = evaluator.store.budget_snapshot()
        assert before["process_launches"] == 39 and len(evaluator.store.training_data()) == 25
    model_sha = file_sha256(directory / "models" / model_id / "gp.joblib")
    sealed_sha = file_sha256(source / "holdout.csv")
    code = """
import os,sys
from sbo.evaluator import Evaluator
from sbo.verification import run_verify
original=Evaluator.evaluate
def crash(self,*args,**kwargs):
    result=original(self,*args,**kwargs)
    if kwargs.get('purpose')=='verification':
        os._exit(75)
    return result
Evaluator.evaluate=crash
run_verify(sys.argv[1])
"""
    child = subprocess.run([sys.executable, "-c", code, str(directory)], capture_output=True, text=True, timeout=90)
    assert child.returncode == 75, child.stdout + child.stderr
    with Evaluator(directory) as evaluator:
        state = evaluator.store.stage("verify")
        first = state["slots"][0]
        assert first["status"] == "planned" and evaluator.store.get_result(first["eval_id"]).status == "success"
        assert evaluator.store.budget_snapshot()["process_launches"] == 40
        freeze_hash = state["freeze_hash"]
    with pytest.raises(StageIncomplete, match="boundary"):
        run_verify(directory, resume=True, stop_after_evaluations=1)
    with Evaluator(directory) as evaluator:
        assert evaluator.store.budget_snapshot()["process_launches"] == 40
        assert evaluator.store.stage("verify")["slots"][0]["eval_id"] == first["eval_id"]
    verified = run_verify(directory, resume=True)
    assert verified["status"] == "success" and verified["freeze_hash"] == freeze_hash
    assert verified["budget"]["process_launches"] == 51 and verified["budget"]["distinct_standard_designs"] == 31
    assert verified["certification"]["improvement_certified"]
    assert verified["holdout"]["metrics"]["quality_passed"]
    final = run_report(directory)
    assert final["m5_passed"] and final["recommendation"]["reason"] == "certified_improvement"
    assert run_verify(directory) == verified
    assert file_sha256(directory / "models" / model_id / "gp.joblib") == model_sha
    assert file_sha256(source / "holdout.csv") == sealed_sha
    with Evaluator(directory) as evaluator:
        assert len(evaluator.store.training_data()) == 25
        assert evaluator.store.budget_snapshot()["process_launches"] == 51
        assert evaluator.store.budget_snapshot()["pending_launch_reservations"] == 0
        slots = evaluator.store.stage("verify")["slots"]
        assert len({slot["eval_id"] for slot in slots}) == 12
        write_json(directory / "m5-validation.json", {"status": "success", "source_run": str(source),
                   "hard_exit_code": 75, "freeze_hash": freeze_hash, "recovered_eval_id": first["eval_id"],
                   "new_real_solver_launches": 12, "training_unchanged": True, "final_model_unchanged": True,
                   "source_holdout_csv_unchanged": True, "budget": evaluator.store.budget_snapshot(),
                   "holdout_metrics": verified["holdout"]["metrics"], "certification": verified["certification"]})
