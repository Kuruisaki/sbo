"""Orchestration tests use archived solver output in disposable test runs only."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import psutil
import pytest
import yaml

from sbo.config import FrozenConfig, create_run, file_sha256, write_json
from sbo.evaluator import Evaluator, build_cache_key
from sbo.geometry import GeometryError, generate_from_config
from sbo.xfoil_runner import LaunchEvent, XfoilAttemptResult, XfoilSettings, build_session

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lab():
    path = ROOT / "runs" / f"evaluator-tests-{uuid4().hex}"
    path.mkdir()
    yield path
    assert path.resolve().parent == (ROOT / "runs").resolve()
    shutil.rmtree(path)


def make_run(lab, name="case", **limits):
    data = yaml.safe_load((ROOT / "configs/naca2412.yaml").read_text(encoding="utf-8"))
    data["runtime"]["runs_root"] = str(lab)
    data["budget"].update(limits)
    config = FrozenConfig(data)
    return create_run(config, name), config


@pytest.fixture
def solver(monkeypatch):
    calls = []
    modes = []

    def fake(geometry, settings, directory, *, on_launch):
        calls.append(settings)
        mode = modes.pop(0) if modes else "success"
        source = ROOT / "tests/fixtures" / ("xfoil699_unconverged" if mode == "failed" else "xfoil699_baseline")
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("polar.txt", "stdout.log", "stderr.log"):
            shutil.copyfile(source / name, directory / name)
        if settings.save_cp:
            shutil.copyfile(ROOT / "tests/fixtures/xfoil699_baseline/cp.txt", directory / "cp.txt")
        if mode == "review":
            polar = (directory / "polar.txt").read_text().replace("2.335", "9.000")
            log = (directory / "stdout.log").read_text().replace("a =  2.335", "a =  9.000")
            (directory / "polar.txt").write_text(polar)
            (directory / "stdout.log").write_text(log)
        (directory / "airfoil.dat").write_bytes(geometry.dat_bytes)
        (directory / "input.txt").write_bytes(build_session(settings).encode("ascii"))
        stamp = datetime.now(timezone.utc).isoformat()
        pid, created = os.getpid(), psutil.Process().create_time()
        event = LaunchEvent(pid, created, settings.attempt_number, directory, stamp)
        audit = {"launched": mode != "launch_failed", "pid": pid, "status": "running"}
        write_json(directory / "process.json", audit)
        if mode != "launch_failed":
            on_launch(event)
        if mode == "exception":
            raise RuntimeError("injected internal failure")
        result = XfoilAttemptResult(
            status="launch_failed" if mode == "launch_failed" else "completed",
            failure_reason="injected spawn failure" if mode == "launch_failed" else None,
            launched=mode != "launch_failed", pid=pid, process_create_time=created,
            returncode=None if mode == "launch_failed" else 0,
            started_at_utc=stamp, ended_at_utc=stamp, elapsed_seconds=.001,
            executable=settings.executable, executable_sha256=file_sha256(settings.executable),
            geometry_file_sha256=geometry.file_sha256, session_sha256=file_sha256(directory / "input.txt"),
            evaluation_policy_version=settings.evaluation_policy_version, expected_version=settings.expected_version,
            detected_version=settings.expected_version, actual_panel_count=settings.panel_count,
            attempt_number=settings.attempt_number, panel_count=settings.panel_count,
            iteration_limit=settings.iteration_limit, timeout_seconds=float(settings.timeout_seconds),
            cl_sequence=settings.cl_sequence, remaining_pids=(), cleanup_errors=(),
            attempt_dir=directory, airfoil_path=directory / "airfoil.dat", input_path=directory / "input.txt",
            stdout_path=directory / "stdout.log", stderr_path=directory / "stderr.log",
            polar_path=directory / "polar.txt", cp_path=directory / "cp.txt" if settings.save_cp else None,
            process_path=directory / "process.json",
        )
        write_json(result.process_path, result.to_dict())
        return result

    monkeypatch.setattr("sbo.evaluator.run_xfoil", fake)
    return calls, modes


def test_success_cache_and_persistent_budget(lab, solver):
    run, _ = make_run(lab)
    calls, _ = solver
    with Evaluator(run) as evaluator:
        first = evaluator.evaluate(save_cp=True)
        assert first.usable_for_training and first.selected_attempt == 1
        cached = evaluator.evaluate(save_cp=True)
        assert cached.cache_hit and cached.eval_id == first.eval_id
        assert evaluator.store.budget_snapshot()["process_launches"] == 1
    with Evaluator(run) as reopened:
        assert reopened.evaluate(save_cp=True).cache_hit
        assert reopened.store.budget_snapshot()["process_launches"] == 1
    assert len(calls) == 1


def test_failure_retries_once_with_same_physics(lab, solver):
    run, _ = make_run(lab)
    calls, modes = solver
    modes.extend(["failed", "success"])
    with Evaluator(run) as evaluator:
        result = evaluator.evaluate()
        assert result.usable_for_training and result.selected_attempt == 2
        assert len(result.attempts) == 2
        assert evaluator.store.budget_snapshot()["process_launches"] == 2
        assert evaluator.store.budget_snapshot()["distinct_standard_designs"] == 1
    assert calls[0].iteration_limit == 200 and calls[1].iteration_limit == 400
    assert calls[1].cl_sequence == (.2, .3, .4, .5)
    assert calls[0].reynolds == calls[1].reynolds and calls[0].ncrit == calls[1].ncrit
    assert result.attempts[0]["runner"]["attempt_dir"] != result.attempts[1]["runner"]["attempt_dir"]


def test_two_failures_never_fill_penalty_or_repeat_again(lab, solver):
    run, _ = make_run(lab)
    calls, modes = solver
    modes.extend(["failed", "failed"])
    with Evaluator(run) as evaluator:
        result = evaluator.evaluate()
        assert result.status == "failed" and result.aerodynamics is None
        assert evaluator.evaluate().status == "failed"
        assert evaluator.store.budget_snapshot()["process_launches"] == 2
        json.dumps(result.to_dict(), allow_nan=False)
    assert len(calls) == 2


def test_review_stops_without_retry_and_is_not_cache_success(lab, solver):
    run, _ = make_run(lab)
    calls, modes = solver
    modes.append("review")
    with Evaluator(run) as evaluator:
        result = evaluator.evaluate()
        assert result.status == "needs_review" and not result.usable_for_training
        assert result.aerodynamics["alpha_deg"] == 9
        assert evaluator.evaluate().status == "needs_review"
        assert evaluator.store.budget_snapshot()["process_launches"] == 1
    assert len(calls) == 1


def test_final_reserve_blocks_retry_before_spawn(lab, solver):
    run, _ = make_run(lab, max_process_launches=2, reserved_final_verification_launches=1)
    calls, modes = solver
    modes.append("failed")
    with Evaluator(run) as evaluator:
        result = evaluator.evaluate()
        assert result.status == "failed" and result.failure_reason.startswith("retry_budget_exhausted")
        assert evaluator.store.budget_snapshot()["process_launches"] == 1
    assert len(calls) == 1


def test_distinct_design_cap_and_verification_exclusion(lab, solver):
    run, _ = make_run(lab, max_distinct_standard_designs=1)
    with Evaluator(run) as evaluator:
        assert evaluator.evaluate().status == "success"
        skipped = evaluator.evaluate([.03, .4], purpose="doe")
        assert skipped.status == "skipped" and skipped.aerodynamics is None
        repeat = evaluator.evaluate([.03, .4], purpose="verification")
        assert repeat.status == "success" and not repeat.usable_for_training
        assert evaluator.store.budget_snapshot()["distinct_standard_designs"] == 1
        assert evaluator.store.budget_snapshot()["process_launches"] == 2


def test_verification_bypasses_cache_without_poisoning_standard(lab, solver):
    run, _ = make_run(lab)
    with Evaluator(run) as evaluator:
        baseline = evaluator.evaluate()
        repeat = evaluator.evaluate(purpose="verification")
        assert repeat.eval_id != baseline.eval_id and not repeat.usable_for_training
        assert evaluator.evaluate().eval_id == baseline.eval_id
        assert evaluator.store.budget_snapshot()["process_launches"] == 2


def test_cache_revalidates_artifacts_and_never_reruns_silently(lab, solver):
    run, _ = make_run(lab)
    calls, _ = solver
    with Evaluator(run) as evaluator:
        result = evaluator.evaluate()
        Path(result.attempts[0]["runner"]["polar_path"]).write_text("tampered")
        rejected = evaluator.evaluate()
        assert rejected.failure_reason == "cache_artifact_mismatch" and rejected.aerodynamics is None
        assert evaluator.evaluate().status == "failed"
    assert len(calls) == 1


def test_explicit_reserved_budget_is_only_for_final_verification(lab, solver):
    run, _ = make_run(lab, max_process_launches=2, reserved_final_verification_launches=1)
    with Evaluator(run) as evaluator:
        evaluator.evaluate()
        assert evaluator.evaluate(purpose="verification").status == "skipped"
        assert evaluator.evaluate(purpose="verification", use_reserved_budget=True).status == "success"
        with pytest.raises(ValueError):
            evaluator.evaluate(use_reserved_budget=True)
        assert evaluator.store.budget_snapshot()["process_launches"] == 2


def test_holdout_label_never_becomes_training_or_changes_split(lab, solver):
    run, _ = make_run(lab)
    with Evaluator(run) as evaluator:
        result = evaluator.evaluate(split="holdout", purpose="doe")
        assert result.status == "success" and not result.usable_for_training
        with pytest.raises(ValueError, match="split"):
            evaluator.evaluate(split="train")


def test_internal_error_preserves_launch_and_stops(lab, solver):
    run, _ = make_run(lab)
    calls, modes = solver
    modes.append("exception")
    with Evaluator(run) as evaluator:
        with pytest.raises(RuntimeError, match="injected"):
            evaluator.evaluate()
        assert evaluator.store.budget_snapshot()["process_launches"] == 1
        assert evaluator.store.budget_snapshot()["pending_launch_reservations"] == 0
    assert len(calls) == 1


def test_launch_failure_does_not_charge_a_launch(lab, solver):
    run, _ = make_run(lab)
    _, modes = solver
    modes.append("launch_failed")
    with Evaluator(run) as evaluator:
        result = evaluator.evaluate()
        assert result.status == "failed"
        assert evaluator.store.budget_snapshot()["process_launches"] == 0


def test_no_implicit_defaults_for_invalid_design_or_changed_frozen_config(lab, solver):
    run, _ = make_run(lab)
    with Evaluator(run) as evaluator:
        with pytest.raises(GeometryError):
            evaluator.evaluate([.5, .4])
        (run / "config.frozen.yaml").write_text("changed")
        with pytest.raises(ValueError, match="changed"):
            evaluator.evaluate()


def test_missing_cached_cp_requires_independent_verification(lab, solver):
    run, _ = make_run(lab)
    with Evaluator(run) as evaluator:
        evaluator.evaluate()
        with pytest.raises(ValueError, match="verification"):
            evaluator.evaluate(save_cp=True)
        assert evaluator.store.budget_snapshot()["process_launches"] == 1


def test_cache_key_covers_full_precision_fidelity_and_solver(lab):
    _, config = make_run(lab)
    g = generate_from_config(config)
    settings = XfoilSettings.from_config(config)
    key = build_cache_key(g, config, settings, "solver-one")
    assert key != build_cache_key(generate_from_config(config, [.020000000001, .4]), config, settings, "solver-one")
    assert key != build_cache_key(g, config, replace(settings, panel_count=240), "solver-one")
    assert key != build_cache_key(g, config, replace(settings, ncrit=8), "solver-one")
    assert key != build_cache_key(g, config, settings, "solver-two")
