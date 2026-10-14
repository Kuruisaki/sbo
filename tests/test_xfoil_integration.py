"""Real solver tests. Evidence remains under runs/xfoil-runner-integration-*.

Run with ``python -m pytest -m integration`` and src on PYTHONPATH.
These calls validate the runner, not the full preflight or numerical quality gates.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import yaml

from sbo.geometry import generate_from_config
from sbo.polar import parse_attempt
from sbo.xfoil_runner import XfoilSettings, run_xfoil

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


def _polar_rows(path):
    # Independent integration assertions only; production parsing belongs to polar.py.
    lines = path.read_text(encoding="ascii").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.split()[:2] == ["alpha", "CL"])
    columns = lines[header_index].split()
    records = []
    for line in lines[header_index + 2:]:
        tokens = line.split()
        if not tokens:
            continue
        assert len(tokens) == len(columns)
        values = [float(value.replace("D", "E")) for value in tokens]
        assert np.isfinite(values).all()
        records.append(dict(zip(columns, values)))
    return records


@pytest.fixture(scope="module")
def real_solver():
    config_path = ROOT / "configs/naca2412.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    executable = Path(config["xfoil"]["executable"])
    if not executable.is_file():
        pytest.skip("configured XFOIL executable is missing; real solver validation remains incomplete")
    label = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = ROOT / "runs" / f"xfoil-runner-integration-{label}-{uuid4().hex[:8]}"
    directory.mkdir()
    (directory / "config.frozen.yaml").write_bytes(config_path.read_bytes())
    geometry = generate_from_config(config)
    events, results = [], []

    def evaluate(label, *, attempt_number=1, panel_count=None):
        settings = XfoilSettings.from_config(config, attempt_number=attempt_number, panel_count=panel_count, save_cp=True)
        result = run_xfoil(geometry, settings, directory / label / f"attempt_{attempt_number}", on_launch=events.append)
        results.append(result.to_dict())
        assert result.completed_normally, result.to_dict()
        assert not result.remaining_pids and not result.cleanup_errors
        assert result.detected_version == config["xfoil"]["expected_version"]
        assert result.actual_panel_count == settings.panel_count
        parsed = parse_attempt(result, config)
        assert parsed.usable_for_training, parsed.to_dict()
        rows = _polar_rows(result.polar_path)
        target = [row for row in rows if abs(row["CL"] - config["flow"]["target_cl"]) <= config["flow"]["cl_tolerance"]]
        assert len(target) == 1
        assert target[0]["CD"] > 0 and np.isfinite([target[0][key] for key in ("alpha", "CL", "CD", "CM")]).all()
        cp_lines = result.cp_path.read_text(encoding="ascii").splitlines()
        cp_header = next(i for i, line in enumerate(cp_lines) if line.lstrip().startswith("#") and "Cp" in line.split())
        cp = np.loadtxt(cp_lines[cp_header + 1:])
        assert cp.ndim == 2 and len(cp) == settings.panel_count and cp.shape[1] >= 2
        assert np.isfinite(cp).all()
        text = result.stdout_path.read_bytes().decode("utf-8", errors="replace")
        assert "not recognized" not in text.lower()
        return result, rows, target[0]

    try:
        yield evaluate
    finally:
        report = {
            "scope": "XFOIL process/session validation, not full SBO preflight or optimization",
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "real_process_launches": len(events),
            "attempts": results,
        }
        (directory / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def test_five_consecutive_independent_calls(real_solver):
    results, cds = [], []
    for index in range(5):
        result, rows, target = real_solver(f"repeat_160_{index + 1}")
        assert len(rows) == 1
        results.append(result)
        cds.append(target["CD"])
    assert len({result.attempt_dir for result in results}) == 5
    assert max(cds) - min(cds) <= 1e-5
    assert len({result.session_sha256 for result in results}) == 1


@pytest.mark.parametrize("panel_count", [200, 240])
def test_verification_panel_counts(real_solver, panel_count):
    result, rows, _ = real_solver(f"panels_{panel_count}", panel_count=panel_count)
    assert result.actual_panel_count == panel_count and len(rows) == 1


def test_fresh_process_continuation_attempt(real_solver):
    result, rows, _ = real_solver("continuation_160", attempt_number=2)
    assert result.iteration_limit == 400 and result.timeout_seconds == 120
    assert result.cl_sequence == (0.2, 0.3, 0.4, 0.5)
    np.testing.assert_allclose([row["CL"] for row in rows], result.cl_sequence, atol=0.001)
