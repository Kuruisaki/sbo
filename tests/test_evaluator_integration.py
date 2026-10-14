"""Actual end-to-end evaluator tests; source evidence remains under runs/."""
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from sbo.config import FrozenConfig, create_run, write_json
from sbo.evaluator import Evaluator

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_runs():
    data = yaml.safe_load((ROOT / "configs/naca2412.yaml").read_text(encoding="utf-8"))
    if not Path(data["xfoil"]["executable"]).is_file():
        pytest.skip("actual XFOIL executable is unavailable; M1 remains unverified")
    label = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = ROOT / "runs" / f"evaluator-integration-{label}-{uuid4().hex[:8]}"
    root.mkdir()
    records = []

    def make(name, first_iterations=None):
        values = json.loads(json.dumps(data))
        values["runtime"]["runs_root"] = str(root)
        if first_iterations is not None:
            values["xfoil"]["initial_attempt"]["iteration_limit"] = first_iterations
        directory = create_run(FrozenConfig(values), name)
        records.append(str(directory))
        return directory
    yield make
    write_json(root / "validation.json", {"scope": "real evaluator/cache/retry/split validation", "runs": records})


def test_actual_baseline_cache_and_independent_verification(real_runs):
    directory = real_runs("baseline")
    with Evaluator(directory) as evaluator:
        first = evaluator.evaluate(save_cp=True)
        assert first.usable_for_training and first.aerodynamics["cl"] == .5
        assert first.aerodynamics["cd"] > 0
        assert evaluator.evaluate(save_cp=True).cache_hit
        verification = evaluator.evaluate(purpose="verification", panel_count=240, save_cp=True)
        assert verification.status == "success" and not verification.usable_for_training
        assert verification.fidelity["panel_count"] == 240
        assert evaluator.store.budget_snapshot()["process_launches"] == 2
        assert evaluator.store.budget_snapshot()["distinct_standard_designs"] == 1
    with Evaluator(directory) as reopened:
        assert reopened.evaluate(save_cp=True).cache_hit
        assert reopened.store.budget_snapshot()["process_launches"] == 2


def test_actual_unconverged_first_attempt_is_recovered_once(real_runs):
    directory = real_runs("forced_retry", first_iterations=1)
    with Evaluator(directory) as evaluator:
        result = evaluator.evaluate(save_cp=True)
        assert result.status == "success" and result.selected_attempt == 2
        assert len(result.attempts) == 2
        assert result.attempts[0]["polar"]["failure_reason"] == "target_convergence_failed"
        assert result.attempts[0]["polar"]["aerodynamics"] is None
        assert result.attempts[1]["runner"]["cl_sequence"] == (.2, .3, .4, .5)
        assert result.aerodynamics["cl"] == .5 and result.aerodynamics["cd"] > 0
        assert evaluator.evaluate(save_cp=True).cache_hit
        assert evaluator.store.budget_snapshot()["process_launches"] == 2


def test_actual_continuous_holdout_is_not_a_training_point(real_runs):
    directory = real_runs("holdout")
    with Evaluator(directory) as evaluator:
        result = evaluator.evaluate([.0237, .436], split="holdout", purpose="doe")
        assert result.status == "success" and not result.usable_for_training
        assert result.aerodynamics["cl_error"] <= evaluator.config["flow"]["cl_tolerance"]
        assert evaluator.store.budget_snapshot()["distinct_standard_designs"] == 1
