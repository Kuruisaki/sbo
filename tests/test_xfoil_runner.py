"""Process/session regressions; no real XFOIL calls are made in this module."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from uuid import uuid4

import psutil
import pytest
import yaml

from sbo.geometry import generate_from_config
from sbo.xfoil_runner import XfoilConfigurationError, XfoilSettings, build_session, run_xfoil

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "configs/naca2412.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def lab():
    # Inherit Windows permissions instead of pytest's restrictive tmp_path mode.
    directory = ROOT / "runs" / f"xfoil-runner-tests-{uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        assert directory.resolve().parent == (ROOT / "runs").resolve()
        shutil.rmtree(directory)


@pytest.fixture
def fake_process(monkeypatch):
    original = subprocess.Popen
    calls = []

    def select(mode):
        def launch(command, **options):
            calls.append((command, options))
            return original([sys.executable, "-u", str(ROOT / "tests/fixtures/fake_xfoil_process.py"), mode], **options)
        monkeypatch.setattr("sbo.xfoil_runner.subprocess.Popen", launch)
        return calls
    return select


def test_session_menu_blanks_and_fixed_physics(config):
    session = build_session(XfoilSettings.from_config(config, save_cp=True))
    assert "XB 1 1\n\n\nPANE\nOPER\nTYPE 1\nVISC 1000000\nMACH 0.1\nITER 200\n" in session
    assert "VPAR\nN 9\nXTR 1 1\n\nPACC\npolar.txt\n\nCL 0.5\n" in session
    assert session.endswith("CPWR cp.txt\nPACC\n\nQUIT\n")
    assert session.startswith("PLOP\nG\n\nLOAD airfoil.dat\n")
    assert "CLI " not in session and "NORM" not in session and "\r" not in session


def test_retry_and_verification_change_only_allowed_settings(config):
    first = XfoilSettings.from_config(config)
    retry = XfoilSettings.from_config(config, attempt_number=2)
    assert retry.iteration_limit == 400 and retry.timeout_seconds == 120
    assert retry.cl_sequence == (0.2, 0.3, 0.4, 0.5)
    for name in ("reynolds", "mach", "flow_type", "ncrit", "xtr", "panel_count", "panel_bunching"):
        assert getattr(first, name) == getattr(retry, name)
    verification = XfoilSettings.from_config(config, panel_count=240, save_cp=True)
    assert "N 240\n" in build_session(verification)
    assert "CL 0.2\nCL 0.3\nCL 0.4\nCL 0.5\n" in build_session(retry)


@pytest.mark.parametrize("field,value", [
    ("timeout_seconds", 0), ("timeout_seconds", float("nan")), ("panel_count", True),
    ("panel_count", 160.5), ("iteration_limit", 0), ("mach", 1), ("ncrit", -1),
    ("xtr", (1,)), ("xtr", (1, 1.1)), ("top_refinement", (0.9, 0.1)),
    ("cl_sequence", (0.2,)), ("cl_sequence", ()), ("attempt_number", 3),
    ("flow_type", 2), ("flow_type", 1.0), ("expected_version", 6.99), ("save_cp", "true"),
    ("executable", Path("xfoil.exe")),
])
def test_invalid_settings_fail_before_creation(config, field, value, lab):
    settings = replace(XfoilSettings.from_config(config), **{field: value})
    destination = lab / "attempt_1"
    with pytest.raises(XfoilConfigurationError):
        run_xfoil(generate_from_config(config), settings, destination)
    assert not destination.exists()


def test_missing_configuration_is_not_defaulted(config):
    del config["xfoil"]["ppar"]["xt"]
    with pytest.raises(XfoilConfigurationError):
        XfoilSettings.from_config(config)


@pytest.mark.parametrize("group", ["solver", "flow", "ppar", "attempt"])
def test_unknown_configuration_fields_are_rejected(config, group):
    target = {
        "solver": config["xfoil"], "flow": config["flow"],
        "ppar": config["xfoil"]["ppar"], "attempt": config["xfoil"]["initial_attempt"],
    }[group]
    target["misspelled_option"] = 1
    with pytest.raises(XfoilConfigurationError, match="unknown"):
        XfoilSettings.from_config(config)


@pytest.mark.parametrize("dirty", [False, True])
def test_existing_attempt_directory_is_rejected(config, lab, dirty):
    destination = lab / "attempt_1"
    destination.mkdir()
    if dirty:
        (destination / "xfoil.def").write_text("old settings")
        (destination / "polar.txt").write_text("old polar")
    with pytest.raises(FileExistsError):
        run_xfoil(generate_from_config(config), XfoilSettings.from_config(config), destination)
    if dirty:
        assert (destination / "polar.txt").read_text() == "old polar"


@pytest.mark.parametrize("mode,expected", [
    ("normal", "completed"), ("wrong_version", "solver_identity_mismatch"),
    ("missing_identity", "solver_identity_mismatch"), ("wrong_panels", "solver_setting_mismatch"),
    ("crash", "crashed"), ("flood", "completed"),
])
def test_lifecycle_and_streamed_logs(config, lab, fake_process, mode, expected):
    calls = fake_process(mode)
    events = []
    result = run_xfoil(generate_from_config(config), XfoilSettings.from_config(config), lab / "attempt_1", on_launch=events.append)
    assert result.status == expected
    assert result.launched and len(events) == len(calls) == 1
    assert events[0].pid == result.pid
    assert calls[0][0] == [str(Path(config["xfoil"]["executable"]).resolve())]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["cwd"] == result.attempt_dir
    assert not result.polar_path.exists()  # completed_normally is not sample validity.
    metadata = json.loads(result.process_path.read_text(encoding="utf-8"))
    assert metadata["status"] == expected and metadata["pid"] == result.pid
    assert result.elapsed_seconds < 10 and not result.remaining_pids and not result.cleanup_errors
    if mode == "flood":
        assert result.stdout_path.stat().st_size > 1024 * 1024
        assert b"\xff" in result.stdout_path.read_bytes()
    if mode == "crash":
        assert result.returncode == 7 and "simulated crash" in result.stderr_path.read_text()


def _assert_gone(pid):
    deadline = time.monotonic() + 2
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.02)
    assert not psutil.pid_exists(pid)


@pytest.mark.parametrize("mode", ["timeout_child", "orphan_child"])
def test_child_processes_are_removed(config, lab, fake_process, mode):
    fake_process(mode)
    settings = replace(XfoilSettings.from_config(config), timeout_seconds=0.8)
    result = run_xfoil(generate_from_config(config), settings, lab / "attempt_1")
    assert result.status == ("timeout" if mode == "timeout_child" else "completed")
    assert result.elapsed_seconds < 8 and not result.remaining_pids
    child = int((result.attempt_dir / "child.pid").read_text())
    _assert_gone(child)
    _assert_gone(result.pid)
    assert result.input_path.is_file() and result.stdout_path.is_file()


@pytest.mark.parametrize("error_type", [RuntimeError, OSError, KeyboardInterrupt])
def test_callback_failure_stops_process_and_preserves_audit(config, lab, fake_process, error_type):
    fake_process("timeout_child")
    observed = []
    directory = lab / "attempt_1"

    def fail(event):
        assert json.loads((directory / "process.json").read_text())["status"] == "running"
        observed.append(event)
        raise error_type("event-store failure")
    with pytest.raises(error_type, match="event-store failure"):
        run_xfoil(generate_from_config(config), XfoilSettings.from_config(config), directory, on_launch=fail)
    _assert_gone(observed[0].pid)
    metadata = json.loads((directory / "process.json").read_text())
    assert metadata["status"] == "aborted" and metadata["launched"] is True
    assert not metadata["remaining_pids"]


def test_launch_failure_has_no_launch_event(config, lab, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated spawn failure")
    monkeypatch.setattr("sbo.xfoil_runner.subprocess.Popen", fail)
    events = []
    result = run_xfoil(generate_from_config(config), XfoilSettings.from_config(config), lab / "attempt_1", on_launch=events.append)
    assert result.status == "launch_failed" and not result.launched and result.returncode is None
    assert events == []
    assert "simulated spawn failure" in result.failure_reason
