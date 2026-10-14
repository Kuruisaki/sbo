"""Hard process exits exercise real SQLite/Windows Job/XFOIL recovery boundaries."""
import json
import subprocess
import sys
import time

import psutil
import pytest

from test_evaluator_integration import real_runs
from sbo.config import write_json
from sbo.evaluator import Evaluator

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("boundary", ["after_process_exit", "after_launch_event"])
def test_actual_hard_exit_and_remaining_attempt_recovery(real_runs, boundary):
    directory = real_runs("hard_exit_" + boundary)
    code = """
import os, sys
from sbo.evaluator import Evaluator
import sbo.evaluator as module
actual = module.run_xfoil
def crash(geometry, settings, directory, *, on_launch):
    if sys.argv[2] == 'after_launch_event':
        def callback(event):
            on_launch(event)
            os._exit(74)
        return actual(geometry, settings, directory, on_launch=callback)
    actual(geometry, settings, directory, on_launch=on_launch)
    os._exit(73)
module.run_xfoil = crash
with Evaluator(sys.argv[1]) as evaluator:
    evaluator.evaluate([.0237, .436], purpose='doe', save_cp=True, evaluation_id='eval_hard_exit')
"""
    child = subprocess.run([sys.executable, "-c", code, str(directory), boundary],
                           capture_output=True, text=True, timeout=30)
    assert child.returncode == (73 if boundary == "after_process_exit" else 74), child.stderr
    with Evaluator(directory) as evaluator:
        assert evaluator.store.budget_snapshot()["process_launches"] == 1
        assert evaluator.store.unfinished() == ["eval_hard_exit"]
        process_path = directory / "evaluations/eval_hard_exit/attempt_1/process.json"
        audit = json.loads(process_path.read_text())
        # Closing the abruptly exited owner's Windows Job kills its suspended solver.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                process = psutil.Process(audit["pid"])
                if process.create_time() != audit["process_create_time"] or not process.is_running():
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.05)
        recovered = evaluator.resume("eval_hard_exit")
        expected = 1 if boundary == "after_process_exit" else 2
        assert recovered.status == "success" and recovered.selected_attempt == expected
        assert recovered.aerodynamics["cl"] == .5
        assert evaluator.store.budget_snapshot()["process_launches"] == expected
        assert evaluator.store.budget_snapshot()["pending_launch_reservations"] == 0
        assert len(recovered.attempts) == expected
        assert evaluator.evaluate([.0237, .436], purpose="doe", save_cp=True).cache_hit
        write_json(directory / "recovery-validation.json", {
            "status": "success", "boundary": boundary, "child_exit_code": child.returncode,
            "same_evaluation_id": recovered.eval_id, "selected_attempt": recovered.selected_attempt,
            "budget": evaluator.store.budget_snapshot(), "no_synthetic_aerodynamics": True,
        })
