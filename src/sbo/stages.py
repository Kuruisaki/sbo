"""M2 stage prerequisites, exclusive ownership and artifact receipts."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4

import psutil

from .config import file_sha256, verify_run_config, write_json


class StageIncomplete(RuntimeError):
    def __init__(self, message: str, status: str = "interrupted"):
        super().__init__(message)
        self.status = status


@contextmanager
def run_ownership(store):
    """Atomic lease with PID + creation time; a stale lease never resets budgets."""
    db = store.connection
    db.execute("CREATE TABLE IF NOT EXISTS run_lease (id INTEGER PRIMARY KEY CHECK(id=1), token TEXT, pid INTEGER, created REAL)")
    token, pid, created = uuid4().hex, os.getpid(), psutil.Process().create_time()
    db.execute("BEGIN IMMEDIATE")
    try:
        old = db.execute("SELECT pid,created FROM run_lease WHERE id=1").fetchone()
        if old:
            try:
                owner = psutil.Process(old["pid"])
                if owner.create_time() == old["created"] and owner.is_running():
                    raise RuntimeError("another stage owns this run")
            except psutil.NoSuchProcess:
                pass
            store.event("stale_run_lease_recovered", "", {"pid": old["pid"], "created": old["created"]})
        db.execute("INSERT OR REPLACE INTO run_lease VALUES (1,?,?,?)", (token, pid, created))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    try:
        yield
    finally:
        db.execute("DELETE FROM run_lease WHERE token=?", (token,))


def prerequisites(evaluator, stage: str) -> dict:
    verify_run_config(evaluator.run_dir, evaluator.config)
    manifest = json.loads((evaluator.run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("preflight_passed") is not True:
        raise RuntimeError("M1 preflight must pass before M2")
    if stage == "doe":
        baseline = evaluator.store.stage("baseline")
        if baseline is None or baseline["status"] != "success":
            raise RuntimeError("baseline quality gate must pass before DOE")
        verify_receipt(evaluator.run_dir, baseline)
        for eval_id in baseline["evaluations"].values():
            result = evaluator.store.get_result(eval_id)
            if result is None or result.status != "success" or not evaluator._cache_valid(result, True):
                raise RuntimeError("baseline evidence no longer validates")
    return manifest


def verify_receipt(directory: Path, state: dict) -> None:
    for name, digest in state.get("artifacts", {}).items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha256(path) != digest:
            raise RuntimeError("completed stage artifact differs from its receipt: " + name)


def receipt(directory: Path, names: list[str]) -> dict[str, str]:
    return {name: file_sha256(directory / name) for name in names}


def update_manifest(evaluator, stage: str, status: str, **values):
    path = evaluator.run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(stage=stage, status=status, budget=evaluator.store.budget_snapshot(),
                    updated_at_utc=datetime.now(timezone.utc).isoformat(), **values)
    write_json(path, manifest)


def checkpoint_failure(evaluator, name: str, state: dict, exc: BaseException, *, export_datasets: bool = True):
    status = getattr(exc, "status", "interrupted")
    state["last_error"] = f"{type(exc).__name__}: {exc}"
    evaluator.store.save_stage(name, status, state)
    if export_datasets:
        evaluator.store.export_csv()
    update_manifest(evaluator, name, status)


def resolve_run(project: Path, run_id: str) -> Path:
    directory = (project / "runs" / run_id).resolve()
    if directory.parent != (project / "runs").resolve() or not (directory / "config.frozen.yaml").is_file():
        raise ValueError("run-id must identify an existing immediate child of runs/")
    return directory
