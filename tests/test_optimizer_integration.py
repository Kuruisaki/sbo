"""Real new XFOIL point after a hard parent exit, using an isolated inherited ledger."""
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
from sbo.optimizer import run_optimize
from sbo.stages import StageIncomplete

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


def test_actual_closed_loop_recovers_evaluation_commit_before_history():
    source = ROOT / "runs/m1_20261013_02"
    if not (source / "fit.json").is_file():
        pytest.skip("real M3 source experiment unavailable; M4 remains unverified by this test")
    config = load_config(source / "config.frozen.yaml")
    if not Path(config["xfoil"]["executable"]).is_file():
        pytest.skip("actual XFOIL executable unavailable")
    label = "m4-real-validation-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
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
        # In this OWN validation copy retain exactly the M2/M3 source experiment;
        # source solver files and the production ledger are strictly read-only.
        copied.execute("DELETE FROM attempts WHERE eval_id IN (SELECT eval_id FROM evaluations WHERE purpose='sequential')")
        copied.execute("DELETE FROM events WHERE eval_id IN (SELECT eval_id FROM evaluations WHERE purpose='sequential')")
        copied.execute("DELETE FROM evaluations WHERE purpose='sequential'")
        copied.execute("DELETE FROM designs WHERE design_id NOT IN (SELECT design_id FROM evaluations)")
        copied.execute("DELETE FROM stages WHERE name='optimize'")
        copied.execute("DELETE FROM models WHERE model_id!='iter_000'")
        copied.execute("DELETE FROM run_lease")
        copied.commit()
    finally:
        original.close()
        copied.close()
    with Evaluator(directory) as evaluator:
        for name in ("baseline", "doe", "fit"):
            for relative in evaluator.store.stage(name)["artifacts"]:
                destination = directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, destination)
        source_manifest = json.loads((source / "manifest.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest.update(preflight_passed=True, baseline_eval_id=source_manifest["baseline_eval_id"],
                        source_run=source_manifest["run_id"], validation_scope="Inherited M2/M3 ledger; read-only source artifacts; independent real sequential point.")
        write_json(directory / "manifest.json", manifest)
        before = evaluator.store.budget_snapshot()
        assert before["process_launches"] == 31 and len(evaluator.store.training_data()) == 17
    code = """
import os,sys
from sbo.evaluator import Evaluator
from sbo.optimizer import run_optimize
actual=Evaluator.evaluate
def crash(self,*args,**kwargs):
    result=actual(self,*args,**kwargs)
    if kwargs.get('purpose')=='sequential':
        os._exit(74)
    return result
Evaluator.evaluate=crash
run_optimize(sys.argv[1])
"""
    child = subprocess.run([sys.executable, "-c", code, str(directory)], capture_output=True, text=True, timeout=60)
    assert child.returncode == 74, child.stdout + child.stderr
    with Evaluator(directory) as evaluator:
        state = evaluator.store.stage("optimize")
        assert state["pending"] and not state["history"]
        pending = state["pending"]
        assert evaluator.store.get_result(pending["eval_id"]).status == "success"
        assert evaluator.store.budget_snapshot()["process_launches"] == before["process_launches"] + 1
    with pytest.raises(StageIncomplete, match="boundary"):
        run_optimize(directory, resume=True, stop_after_evaluations=1)
    with Evaluator(directory) as evaluator:
        state = evaluator.store.stage("optimize")
        assert len(state["history"]) == 1 and state["history"][0]["eval_id"] == pending["eval_id"]
        assert state["current_model_id"] == "iter_001" and len(evaluator.store.training_data()) == 18
        assert evaluator.store.model("iter_001")["metadata"]["training_data_hash"] != evaluator.store.model("iter_000")["metadata"]["training_data_hash"]
        assert evaluator.store.budget_snapshot()["process_launches"] == before["process_launches"] + 1
        assert evaluator.store.budget_snapshot()["pending_launch_reservations"] == 0
        assert state["history"][0]["cd"] > 0
        write_json(directory / "m4-validation.json", {
            "status": "success", "source_run": str(source), "hard_exit_code": child.returncode,
            "recovered_evaluation_id": pending["eval_id"], "proposal_unchanged": state["history"][0]["m"] == pending["proposal"]["m"],
            "training_before": 17, "training_after": 18, "model_before": "iter_000", "model_after": "iter_001",
            "new_real_solver_launches": 1, "budget": evaluator.store.budget_snapshot(),
            "holdout_labels_used": False, "improvement_certified": False,
        })
