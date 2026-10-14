"""Evaluation pipeline: geometry -> cache -> bounded attempt(s) -> acceptance.

SQLite stores designs, attempts, launch events and immutable evaluation requests.
Explicit recovery imports completed artifacts or continues the remaining attempt
budget without resampling a design. Stage/dataset workflows live in store.py.

Create a run with config.create_run, then use::

    with Evaluator(run_directory) as evaluator:
        result = evaluator.evaluate()  # frozen-config baseline

Only success/train/standard/baseline-doe-sequential results are trainable. Final
verification bypasses cache; spending its reserved launches requires an explicit
use_reserved_budget=True. No failure is represented by a penalty CD.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import psutil

from .config import FrozenConfig, canonical_json, file_sha256, load_config, verify_run_config, write_json
from .records import BudgetExhausted, EvaluationResult
from .store import SQLiteEvaluationStore
from .geometry import GeometryResult, generate_from_config
from .polar import PARSER_VERSION, parse_attempt
from .xfoil_runner import LaunchEvent, XfoilAttemptResult, XfoilSettings, build_session, run_xfoil


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvaluationStore(Protocol):
    """Single-run store contract; reservations and launch events must be atomic."""
    run_dir: Path
    config_hash: str

    def register_design(self, design: Mapping[str, Any]) -> None: ...
    def latest(self, cache_key: str) -> EvaluationResult | None: ...
    def begin_evaluation(self, record: Mapping[str, Any]) -> None: ...
    def reserve_attempt(self, eval_id: str, attempt_id: str, number: int, use_reserved_budget: bool) -> None: ...
    def record_launch(self, attempt_id: str, event: LaunchEvent) -> None: ...
    def finish_attempt(self, attempt_id: str, record: Mapping[str, Any]) -> None: ...
    def finish_evaluation(self, result: EvaluationResult) -> None: ...
    def event(self, kind: str, eval_id: str, payload: Mapping[str, Any]) -> None: ...
    def budget_snapshot(self) -> dict[str, int]: ...
    def get_result(self, eval_id: str) -> EvaluationResult | None: ...
    def get_request(self, eval_id: str) -> dict[str, Any]: ...
    def attempt_rows(self, eval_id: str) -> list[dict[str, Any]]: ...
    def has_evaluation(self, eval_id: str) -> bool: ...
    def review_accepted(self, eval_id: str) -> bool: ...
    def save_review(self, result: EvaluationResult, decision: str, reason: str) -> None: ...
    def export_csv(self, *, include_holdout: bool = True) -> dict[str, Any]: ...
    def close(self) -> None: ...


def _runner_from_dict(payload: Mapping[str, Any]) -> XfoilAttemptResult:
    values = {item.name: payload[item.name] for item in fields(XfoilAttemptResult)}
    for name, value in tuple(values.items()):
        if name.endswith("_path") or name in {"executable", "attempt_dir"}:
            values[name] = Path(value) if value is not None else None
    for name in ("cl_sequence", "remaining_pids", "cleanup_errors"):
        values[name] = tuple(values[name])
    return XfoilAttemptResult(**values)


def _pressure_check(path: Path, panels: int) -> dict[str, Any]:
    raw = path.read_bytes()
    lines = raw.decode("ascii").splitlines()
    header = next(i for i, line in enumerate(lines) if line.lstrip().startswith("#") and "Cp" in line.split())
    values = np.loadtxt(lines[header + 1:])
    if values.ndim != 2 or len(values) != panels or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("invalid/nonfinite pressure-distribution data")
    return {"path": str(path), "sha256": sha256(raw).hexdigest(), "rows": len(values), "columns": values.shape[1]}


def build_cache_key(geometry: GeometryResult, config: FrozenConfig, settings: XfoilSettings, solver_sha256: str) -> str:
    solver_settings = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()}
    solver_settings.pop("save_cp")  # Optional output does not change aerodynamic fidelity.
    identity = {
        "config_hash": config.config_hash, "m": geometry.m.hex(), "p": geometry.p.hex(), "t": geometry.t.hex(),
        "geometry_file_sha256": geometry.file_sha256, "generator_version": geometry.settings.generator_version,
        "solver_sha256": solver_sha256, "parser_version": PARSER_VERSION,
        "settings": solver_settings,
    }
    return sha256(canonical_json(identity).encode("utf-8")).hexdigest()


class Evaluator:
    def __init__(self, run_dir: str | Path, *, store: EvaluationStore | None = None):
        self.run_dir = Path(run_dir).resolve()
        self.config = load_config(self.run_dir / "config.frozen.yaml")
        verify_run_config(self.run_dir, self.config)
        self.solver_sha256 = file_sha256(self.config["xfoil"]["executable"])
        self._owns_store = store is None
        self.store = SQLiteEvaluationStore(self.run_dir, self.config) if store is None else store
        if self.store.config_hash != self.config.config_hash or Path(self.store.run_dir).resolve() != self.run_dir:
            raise ValueError("store must belong to the same frozen run")

    def close(self):
        if self._owns_store:
            self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _cache_valid(self, result: EvaluationResult, save_cp: bool) -> bool:
        try:
            selected = next(record for record in result.attempts if record["attempt_number"] == result.selected_attempt)
            runner = _runner_from_dict(selected["runner"])
            if (result.config_hash != self.config.config_hash or runner.executable_sha256 != self.solver_sha256
                    or file_sha256(runner.executable) != self.solver_sha256
                    or runner.panel_count != result.fidelity["panel_count"]
                    or runner.geometry_file_sha256 != result.geometry_file_sha256
                    or runner.attempt_dir.resolve().parent != result.evaluation_dir.resolve()):
                return False
            expected = XfoilSettings.from_config(self.config, attempt_number=runner.attempt_number,
                                                 panel_count=runner.panel_count, save_cp=runner.cp_path is not None)
            if (runner.session_sha256 != sha256(build_session(expected).encode("ascii")).hexdigest()
                    or runner.actual_panel_count != expected.panel_count or runner.detected_version != expected.expected_version):
                return False
            hashes = dict(selected["polar"]["artifact_hashes"])
            for name in ("polar", "stdout", "stderr"):
                if file_sha256(getattr(runner, name + "_path")) != hashes[name]:
                    return False
            parsed = parse_attempt(runner, self.config)
            accepted = self.store.review_accepted(result.eval_id)
            if not parsed.usable_for_training and not (accepted and parsed.status == "needs_review"):
                return False
            if save_cp and _pressure_check(runner.cp_path, runner.panel_count)["sha256"] != selected["pressure"]["sha256"]:
                return False
            return True
        except (OSError, ValueError, KeyError, StopIteration, TypeError):
            return False

    def evaluate(
        self, design: Sequence[float] | None = None, *, split: str = "train", purpose: str = "baseline",
        panel_count: int | None = None, save_cp: bool = False,
        bypass_cache: bool = False, use_reserved_budget: bool = False,
        evaluation_id: str | None = None, resume: bool = False,
    ) -> EvaluationResult:
        verify_run_config(self.run_dir, self.config)
        if split not in {"train", "holdout"} or purpose not in {"baseline", "doe", "sequential", "verification"}:
            raise ValueError("invalid split or evaluation purpose")
        if use_reserved_budget and purpose != "verification":
            raise ValueError("only explicit final verification may spend reserved launches")
        if purpose == "verification":
            bypass_cache = True
        elif bypass_cache:
            raise ValueError("independent repeats must use purpose='verification'")
        geometry = generate_from_config(self.config, design)
        settings = XfoilSettings.from_config(self.config, panel_count=panel_count, save_cp=save_cp)
        if settings.panel_count != self.config["xfoil"]["standard_n"] and purpose != "verification":
            raise ValueError("nonstandard panel counts must have verification purpose")
        if file_sha256(settings.executable) != self.solver_sha256:
            raise ValueError("solver executable changed during the run")
        key = build_cache_key(geometry, self.config, settings, self.solver_sha256)
        design_id = sha256(canonical_json([geometry.m.hex(), geometry.p.hex(), geometry.t.hex()]).encode("ascii")).hexdigest()
        bounds = self.config["design"]["bounds"]
        self.store.register_design({"design_id": design_id, "m": geometry.m, "p": geometry.p, "t": geometry.t,
                                    "u_m": (geometry.m - bounds[0][0]) / (bounds[0][1] - bounds[0][0]),
                                    "u_p": (geometry.p - bounds[1][0]) / (bounds[1][1] - bounds[1][0]), "split": split})
        existing_request = None
        if evaluation_id is not None:
            if not re.fullmatch(r"eval_[A-Za-z0-9_]{1,80}", evaluation_id):
                raise ValueError("invalid evaluation_id")
            if self.store.has_evaluation(evaluation_id):
                existing_request = self.store.get_request(evaluation_id)
                for name, value in {"cache_key": key, "split": split, "purpose": purpose,
                                    "save_cp": save_cp, "use_reserved_budget": use_reserved_budget}.items():
                    if existing_request.get(name) != value:
                        raise ValueError("resume request differs from the durable evaluation request")
                done = self.store.get_result(evaluation_id)
                if done and done.status != "interrupted":
                    if done.status == "success" and not self._cache_valid(done, save_cp):
                        raise RuntimeError("completed evaluation artifacts changed")
                    return replace(done, cache_hit=done.status == "success")
                if not resume:
                    raise RuntimeError("unfinished evaluation requires --resume")
        if not bypass_cache and existing_request is None:
            cached = self.store.latest(key)
            if cached and cached.status in {"success", "needs_review", "failed"}:
                if cached.status == "success" and save_cp:
                    selected = next(record for record in cached.attempts if record["attempt_number"] == cached.selected_attempt)
                    if selected.get("pressure") is None:
                        raise ValueError("cached point has no Cp output; use verification purpose for an independent pressure evaluation")
                if cached.status == "success" and not self._cache_valid(cached, save_cp):
                    invalid = replace(cached, status="failed", failure_reason="cache_artifact_mismatch",
                                      aerodynamics=None, selected_attempt=None, ended_at_utc=_now())
                    write_json(invalid.result_path, invalid.to_dict())
                    self.store.finish_evaluation(invalid)
                    self.store.event("cache_invalidated", invalid.eval_id, {"reason": invalid.failure_reason})
                    return invalid
                self.store.event("cache_hit" if cached.status == "success" else "terminal_record_reused",
                                 cached.eval_id, {"status": cached.status, "request_purpose": purpose})
                return replace(cached, cache_hit=cached.status == "success")

        eval_id = evaluation_id or "eval_" + uuid4().hex
        directory = self.run_dir / "evaluations" / eval_id
        if existing_request is None:
            if evaluation_id and directory.is_dir() and not any(directory.iterdir()):
                pass  # Crash between directory creation and the durable request commit.
            else:
                directory.mkdir(parents=True, exist_ok=False)
        started = existing_request["started_at_utc"] if existing_request else _now()
        fidelity = {"panel_count": settings.panel_count, "standard_panel_count": self.config["xfoil"]["standard_n"],
                    "solver_version": settings.expected_version, "solver_sha256": self.solver_sha256,
                    "generator_version": geometry.settings.generator_version,
                    "flow": self.config.to_dict()["flow"], "ppar": self.config.to_dict()["xfoil"]["ppar"]}
        if existing_request is None:
            self.store.begin_evaluation({"eval_id": eval_id, "design_id": design_id, "cache_key": key,
                "config_hash": self.config.config_hash, "panel_count": settings.panel_count, "purpose": purpose,
                "split": split, "save_cp": save_cp, "use_reserved_budget": use_reserved_budget,
                "design": [geometry.m, geometry.p], "started_at_utc": started})
        attempts: list[dict[str, Any]] = []

        def finish(status, reason=None, selected=None, aerodynamic=None):
            result = EvaluationResult(
                eval_id, design_id, self.config.config_hash, key, split, purpose, fidelity, status, reason,
                selected, aerodynamic, tuple(attempts), False, started, _now(), directory,
                directory / "result.json", geometry.file_sha256, geometry.geometry_hash,
            )
            write_json(result.result_path, result.to_dict())
            self.store.finish_evaluation(result)
            return result

        if existing_request:
            for row in self.store.attempt_rows(eval_id):
                record = self._recover_attempt(row, directory, geometry, settings.panel_count, save_cp)
                self.store.finish_attempt(row["attempt_id"], record)
                attempts.append(record)
                if record["status"] in {"success", "needs_review"}:
                    data = record["polar"]["aerodynamics"]
                    aerodynamic = {name: data[name] for name in ("alpha_deg", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr", "cl_error")}
                    self.store.event("evaluation_recovered", eval_id, {"attempt": row["attempt_number"]})
                    return finish(record["status"], "manual_review_required" if record["status"] == "needs_review" else None,
                                  row["attempt_number"], aerodynamic)
                if record["runner"].get("status") in {"launch_failed", "cleanup_failed", "solver_identity_mismatch", "solver_setting_mismatch"}:
                    return finish("failed", record["runner"].get("failure_reason"))
        for number in range(len(attempts) + 1, self.config["xfoil"]["max_attempts"] + 1):
            attempt_id = eval_id + f"_attempt_{number}"
            attempt_dir = directory / f"attempt_{number}"
            try:
                self.store.reserve_attempt(eval_id, attempt_id, number, use_reserved_budget)
            except BudgetExhausted as exc:
                return finish("failed" if attempts else "skipped", "retry_budget_exhausted:" + str(exc) if attempts else str(exc))
            active_settings = XfoilSettings.from_config(self.config, attempt_number=number, panel_count=settings.panel_count, save_cp=save_cp)
            try:
                runner = run_xfoil(geometry, active_settings, attempt_dir,
                                   on_launch=lambda event: self.store.record_launch(attempt_id, event))
                if runner.executable_sha256 != self.solver_sha256:
                    runner = replace(runner, status="solver_identity_mismatch", failure_reason="solver_file_hash_changed")
                parsed = parse_attempt(runner, self.config)
                pressure = None
                if parsed.status in {"success", "needs_review"} and save_cp:
                    try:
                        pressure = _pressure_check(runner.cp_path, runner.panel_count)
                    except (OSError, ValueError, StopIteration, UnicodeError) as exc:
                        parsed = replace(parsed, status="failed", failure_reason="pressure_output_invalid",
                                         failure_details=str(exc), aerodynamics=None)
                record = {"attempt_id": attempt_id, "eval_id": eval_id, "attempt_number": number,
                          "status": parsed.status, "runner": runner.to_dict(), "polar": parsed.to_dict(), "pressure": pressure}
                write_json(attempt_dir / "result.json", record)
                self.store.finish_attempt(attempt_id, record)
                attempts.append(record)
            except BaseException as exc:
                # Runner cleans up before propagating; preserve and reconcile its audit.
                process_path = attempt_dir / "process.json"
                audit = json.loads(process_path.read_text(encoding="utf-8")) if process_path.exists() else {"launched": False}
                record = {"attempt_id": attempt_id, "eval_id": eval_id, "attempt_number": number,
                          "status": "failed", "runner": audit, "failure_reason": f"{type(exc).__name__}: {exc}"}
                if attempt_dir.exists():
                    write_json(attempt_dir / "result.json", record)
                self.store.finish_attempt(attempt_id, record)
                attempts.append(record)
                finish("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                       "evaluation_interrupted_or_internal_error")
                raise
            if parsed.status in {"success", "needs_review"}:
                data = parsed.to_dict()["aerodynamics"]
                aerodynamic = {name: data[name] for name in ("alpha_deg", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr", "cl_error")}
                return finish(parsed.status, "manual_review_required" if parsed.status == "needs_review" else None, number, aerodynamic)
            if runner.status in {"launch_failed", "cleanup_failed", "solver_identity_mismatch", "solver_setting_mismatch", "aborted"}:
                return finish("failed", runner.failure_reason or parsed.failure_reason)
        return finish("failed", attempts[-1].get("polar", {}).get("failure_reason", "attempt_budget_exhausted_after_interruption"))

    def _recover_attempt(self, row, directory, geometry, panel_count, save_cp):
        """A stale running PID is identified by creation time; no unrelated PID is killed."""
        attempt_dir = directory / f"attempt_{row['attempt_number']}"
        path = attempt_dir / "process.json"
        audit = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"launched": False}
        if attempt_dir.exists() and not audit.get("pid") and (audit.get("status") == "planned" or not path.exists()):
            raise RuntimeError("launch identity is incomplete; cannot certify absence of an untracked solver")
        # Fully finalized process audits certify normal exit; logs alone do not.
        if audit.get("status") in {"running", "planned"} and audit.get("pid"):
            try:
                process = psutil.Process(audit["pid"])
                if process.create_time() == audit.get("process_create_time") and process.is_running():
                    raise RuntimeError("the original solver process is still alive; resume must wait for its owner")
            except psutil.NoSuchProcess:
                pass
        base = {"attempt_id": row["attempt_id"], "eval_id": row["eval_id"], "attempt_number": row["attempt_number"]}
        try:
            runner = _runner_from_dict(audit)
        except KeyError:
            return {**base, "status": "interrupted", "runner": audit, "failure_reason": "incomplete_process_audit"}
        if runner.attempt_dir.resolve() != attempt_dir.resolve() or runner.geometry_file_sha256 != geometry.file_sha256:
            raise RuntimeError("recovery artifact identity mismatch")
        if runner.executable_sha256 != self.solver_sha256:
            raise RuntimeError("recovery solver identity mismatch")
        if runner.panel_count != panel_count or runner.attempt_number != row["attempt_number"]:
            raise RuntimeError("recovery panel/attempt identity mismatch")
        requested = XfoilSettings.from_config(self.config, attempt_number=row["attempt_number"],
                                              panel_count=runner.panel_count, save_cp=save_cp)
        if (runner.expected_version != requested.expected_version or
            runner.evaluation_policy_version != requested.evaluation_policy_version or
            runner.iteration_limit != requested.iteration_limit or tuple(runner.cl_sequence) != requested.cl_sequence):
            raise RuntimeError("recovery policy metadata mismatch")
        if runner.status == "completed" and (runner.detected_version != requested.expected_version or runner.actual_panel_count != panel_count):
            raise RuntimeError("recovery solver diagnostic mismatch")
        if runner.session_sha256 != sha256(build_session(requested).encode('ascii')).hexdigest():
            raise RuntimeError("recovery session differs from frozen settings")
        if row["record_json"]:
            previous = json.loads(row["record_json"])
            for name, digest in previous.get("polar", {}).get("artifact_hashes", []):
                if file_sha256(getattr(runner, name + "_path")) != digest:
                    raise RuntimeError("recovery artifact differs from its committed hash")
            if previous.get("pressure") and file_sha256(runner.cp_path) != previous["pressure"]["sha256"]:
                raise RuntimeError("recovery Cp differs from its committed hash")
        parsed = parse_attempt(runner, self.config)
        pressure = None
        if parsed.status in {"success", "needs_review"} and save_cp:
            try:
                pressure = _pressure_check(runner.cp_path, runner.panel_count)
            except (OSError, ValueError, StopIteration, UnicodeError) as exc:
                parsed = replace(parsed, status="failed", failure_reason="pressure_output_invalid",
                                 failure_details=str(exc), aerodynamics=None)
        record = {**base, "status": parsed.status, "runner": runner.to_dict(), "polar": parsed.to_dict(), "pressure": pressure}
        write_json(attempt_dir / "result.json", record)
        return record

    def resume(self, eval_id: str) -> EvaluationResult:
        request = self.store.get_request(eval_id)
        return self.evaluate(request["design"], split=request["split"], purpose=request["purpose"],
            panel_count=request["panel_count"], save_cp=request["save_cp"], use_reserved_budget=request["use_reserved_budget"],
            evaluation_id=eval_id, resume=True)

    def review(self, eval_id: str, decision: str, reason: str, *, include_holdout: bool = True) -> EvaluationResult:
        """Explicit human decision, retained as an event; never automatically accept."""
        if decision not in {"accept", "reject"} or not reason.strip():
            raise ValueError("review requires accept/reject and a nonempty reason")
        result = self.store.get_result(eval_id)
        if result is None or result.status != "needs_review":
            raise ValueError("only needs_review evaluations can be reviewed")
        if decision == "accept":
            selected = next(record for record in result.attempts if record["attempt_number"] == result.selected_attempt)
            runner = _runner_from_dict(selected["runner"])
            parsed = parse_attempt(runner, self.config)
            if parsed.status != "needs_review" or parsed.aerodynamics is None:
                raise ValueError("review artifacts no longer validate")
            for name, digest in selected["polar"]["artifact_hashes"]:
                if file_sha256(getattr(runner, name + "_path")) != digest:
                    raise ValueError("review artifact hash changed")
            if selected.get("pressure") and _pressure_check(runner.cp_path, runner.panel_count)["sha256"] != selected["pressure"]["sha256"]:
                raise ValueError("review Cp artifact hash changed")
        updated = replace(result, status="success" if decision == "accept" else "failed",
                          failure_reason=None if decision == "accept" else "manual_review_rejected",
                          aerodynamics=result.aerodynamics if decision == "accept" else None, ended_at_utc=_now())
        self.store.save_review(updated, decision, reason)
        write_json(updated.result_path, updated.to_dict())
        self.store.export_csv(include_holdout=include_holdout)
        return updated
