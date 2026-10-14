"""Durable SQLite ledger, stage checkpoints and segregated datasets."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path
import sqlite3
from typing import Any
from .config import FrozenConfig, canonical_json, write_json, file_sha256
from .records import BudgetExhausted, EvaluationResult
from .xfoil_runner import LaunchEvent

def _now():
    return datetime.now(timezone.utc).isoformat()

class SQLiteEvaluationStore:
    """Single-run transaction ledger; v1 migration never resets launch counts."""
    VERSION = "m2-ledger-v2"

    def __init__(self, run_dir: str | Path, config: FrozenConfig):
        self.run_dir = Path(run_dir).resolve()
        self.config_hash = config.config_hash
        self.limits = dict(config["budget"])
        self.standard_n = config["xfoil"]["standard_n"]
        self.connection = sqlite3.connect(self.run_dir / "samples.sqlite", timeout=10, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS designs (
                design_id TEXT PRIMARY KEY, m REAL NOT NULL, p REAL NOT NULL, t REAL NOT NULL,
                u_m REAL NOT NULL, u_p REAL NOT NULL, split TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evaluations (
                eval_id TEXT PRIMARY KEY, design_id TEXT NOT NULL REFERENCES designs(design_id),
                cache_key TEXT NOT NULL, config_hash TEXT NOT NULL, panel_count INTEGER NOT NULL,
                purpose TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT);
            CREATE INDEX IF NOT EXISTS evaluation_cache ON evaluations(cache_key);
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY, eval_id TEXT NOT NULL REFERENCES evaluations(eval_id),
                attempt_number INTEGER NOT NULL, status TEXT NOT NULL, launched INTEGER NOT NULL DEFAULT 0,
                record_json TEXT, UNIQUE(eval_id, attempt_number));
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, eval_id TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL, payload_json TEXT NOT NULL);
        """)
        wanted = {"schema_version": self.VERSION, "config_hash": config.config_hash,
                  "budget_limits": canonical_json(self.limits), "standard_n": str(self.standard_n)}
        existing = dict(self.connection.execute("SELECT key,value FROM metadata").fetchall())
        legacy = {**wanted, "schema_version": "m1-ledger-v1"}
        if existing == legacy:
            self.connection.execute("UPDATE metadata SET value=? WHERE key='schema_version'", (self.VERSION,))
            existing = wanted
        if existing and existing != wanted:
            self.connection.close()
            raise ValueError("ledger schema/configuration/budget limits differ from this frozen run")
        if not existing:
            self.connection.executemany("INSERT INTO metadata(key,value) VALUES (?,?)", wanted.items())
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(evaluations)")}
        if "request_json" not in columns:
            self.connection.execute("ALTER TABLE evaluations ADD COLUMN request_json TEXT")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS stages (
                name TEXT PRIMARY KEY, status TEXT NOT NULL, state_json TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY, config_hash TEXT NOT NULL, record_json TEXT NOT NULL);
        """)

    def close(self) -> None:
        self.connection.close()

    def event(self, kind: str, eval_id: str, payload: Mapping[str, Any]) -> None:
        self.connection.execute("INSERT INTO events(kind,eval_id,timestamp_utc,payload_json) VALUES (?,?,?,?)",
                                (kind, eval_id, _now(), canonical_json(dict(payload))))

    def register_design(self, design: Mapping[str, Any]) -> None:
        old = self.connection.execute("SELECT split FROM designs WHERE design_id=?", (design["design_id"],)).fetchone()
        if old and old["split"] != design["split"]:
            raise ValueError("a reserved/holdout design cannot change its split")
        self.connection.execute(
            "INSERT OR IGNORE INTO designs(design_id,m,p,t,u_m,u_p,split) VALUES (?,?,?,?,?,?,?)",
            tuple(design[key] for key in ("design_id", "m", "p", "t", "u_m", "u_p", "split")))

    def latest(self, cache_key: str) -> EvaluationResult | None:
        row = self.connection.execute(
            "SELECT status,result_json FROM evaluations WHERE cache_key=? AND purpose!='verification' ORDER BY rowid DESC LIMIT 1", (cache_key,)).fetchone()
        if not row:
            return None
        if row["status"] in {"planned", "running", "interrupted"} or row["result_json"] is None:
            raise RuntimeError("an unfinished matching evaluation exists; explicit resume is required")
        return EvaluationResult.from_dict(json.loads(row["result_json"]))

    def begin_evaluation(self, record: Mapping[str, Any]) -> None:
        final = self.stage("verify")
        if final and final.get("freeze_hash") and record["purpose"] != "verification":
            raise ValueError("M5 final model/candidate freeze forbids new training/holdout evaluations; create a new run")
        self.connection.execute(
            "INSERT INTO evaluations(eval_id,design_id,cache_key,config_hash,panel_count,purpose,status,request_json) VALUES (?,?,?,?,?,?,?,?)",
            tuple(record[key] for key in ("eval_id", "design_id", "cache_key", "config_hash", "panel_count", "purpose"))
            + ("planned", canonical_json(dict(record))))

    def budget_snapshot(self) -> dict[str, int]:
        counts = self.connection.execute("""
            SELECT COALESCE(SUM(launched),0) AS launches,
                   COALESCE(SUM(CASE WHEN status='reserved' AND launched=0 THEN 1 ELSE 0 END),0) AS reserved
            FROM attempts
        """).fetchone()
        designs = self.connection.execute("""
            SELECT COUNT(DISTINCT e.design_id) FROM evaluations e JOIN attempts a ON a.eval_id=e.eval_id
            WHERE e.panel_count=? AND e.purpose!='verification' AND (a.launched=1 OR a.status='reserved')
        """, (self.standard_n,)).fetchone()[0]
        return {"process_launches": counts["launches"], "pending_launch_reservations": counts["reserved"],
                "distinct_standard_designs": designs,
                "max_process_launches": self.limits["max_process_launches"],
                "max_distinct_standard_designs": self.limits["max_distinct_standard_designs"],
                "reserved_final_verification_launches": self.limits["reserved_final_verification_launches"]}

    def reserve_attempt(self, eval_id: str, attempt_id: str, number: int, use_reserved_budget: bool) -> None:
        db = self.connection
        db.execute("BEGIN IMMEDIATE")
        try:
            pending = db.execute("SELECT COUNT(*) FROM attempts WHERE status IN ('reserved','running')").fetchone()[0]
            if pending:
                raise RuntimeError("an attempt is pending/running; serial work or M2 recovery must finish first")
            snapshot = self.budget_snapshot()
            cap = self.limits["max_process_launches"] - (0 if use_reserved_budget else self.limits["reserved_final_verification_launches"])
            if snapshot["process_launches"] + snapshot["pending_launch_reservations"] >= cap:
                raise BudgetExhausted("process_launch_budget_or_final_reserve")
            evaluation = db.execute("SELECT design_id,panel_count,purpose FROM evaluations WHERE eval_id=?", (eval_id,)).fetchone()
            occupied = db.execute("""
                SELECT 1 FROM evaluations e JOIN attempts a ON a.eval_id=e.eval_id
                WHERE e.design_id=? AND e.panel_count=? AND e.purpose!='verification' AND (a.launched=1 OR a.status='reserved') LIMIT 1
            """, (evaluation["design_id"], self.standard_n)).fetchone()
            if (evaluation["purpose"] != "verification" and evaluation["panel_count"] == self.standard_n and not occupied
                    and snapshot["distinct_standard_designs"] >= self.limits["max_distinct_standard_designs"]):
                raise BudgetExhausted("distinct_standard_design_budget")
            db.execute("INSERT INTO attempts(attempt_id,eval_id,attempt_number,status) VALUES (?,?,?,'reserved')", (attempt_id, eval_id, number))
            self.event("launch_reserved", eval_id, {"attempt_id": attempt_id, "use_reserved_budget": use_reserved_budget})
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise

    def record_launch(self, attempt_id: str, event: LaunchEvent) -> None:
        db = self.connection
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute("SELECT eval_id,launched FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                raise RuntimeError("launch has no budget reservation")
            if not row["launched"]:
                payload = asdict(event)
                payload["attempt_dir"] = str(payload["attempt_dir"])
                db.execute("UPDATE attempts SET launched=1,status='running' WHERE attempt_id=?", (attempt_id,))
                db.execute("UPDATE evaluations SET status='running' WHERE eval_id=?", (row["eval_id"],))
                self.event("process_launched", row["eval_id"], {"attempt_id": attempt_id, **payload})
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise

    def finish_attempt(self, attempt_id: str, record: Mapping[str, Any]) -> None:
        # Reconcile a launch that happened before a runner setup/callback failure.
        runner = record["runner"]
        db = self.connection
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute("SELECT eval_id,launched FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if runner.get("launched") and not row["launched"]:
                self.event("process_launch_reconciled", row["eval_id"], {"attempt_id": attempt_id, "pid": runner.get("pid")})
            db.execute("UPDATE attempts SET launched=MAX(launched,?),status=?,record_json=? WHERE attempt_id=?",
                       (int(bool(runner.get("launched"))), record["status"], canonical_json(dict(record)), attempt_id))
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise

    def finish_evaluation(self, result: EvaluationResult) -> None:
        self.connection.execute("UPDATE evaluations SET status=?,result_json=? WHERE eval_id=?",
                                (result.status, canonical_json(result.to_dict()), result.eval_id))

    def get_result(self, eval_id: str) -> EvaluationResult | None:
        row = self.connection.execute("SELECT result_json FROM evaluations WHERE eval_id=?", (eval_id,)).fetchone()
        return EvaluationResult.from_dict(json.loads(row[0])) if row and row[0] else None

    def get_request(self, eval_id: str) -> dict[str, Any]:
        row = self.connection.execute("SELECT request_json FROM evaluations WHERE eval_id=?", (eval_id,)).fetchone()
        if not row or not row[0]:
            raise RuntimeError("missing durable evaluation request; cannot safely resume a legacy pending evaluation")
        return json.loads(row[0])

    def has_evaluation(self, eval_id: str) -> bool:
        return self.connection.execute("SELECT 1 FROM evaluations WHERE eval_id=?", (eval_id,)).fetchone() is not None

    def review_accepted(self, eval_id: str) -> bool:
        return self.connection.execute("SELECT 1 FROM events WHERE kind='review_accepted' AND eval_id=?", (eval_id,)).fetchone() is not None

    def save_review(self, result: EvaluationResult, decision: str, reason: str) -> None:
        db = self.connection
        db.execute("BEGIN IMMEDIATE")
        try:
            current = db.execute("SELECT status FROM evaluations WHERE eval_id=?", (result.eval_id,)).fetchone()
            if not current or current[0] != "needs_review" or decision not in {"accept", "reject"} or not reason.strip():
                raise ValueError("invalid or already resolved review")
            self.event("review_accepted" if decision == "accept" else "review_rejected", result.eval_id, {"reason": reason})
            self.finish_evaluation(result)
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise

    def unfinished(self) -> list[str]:
        return [row[0] for row in self.connection.execute(
            "SELECT eval_id FROM evaluations WHERE status IN ('planned','running','interrupted') ORDER BY rowid")]

    def attempt_rows(self, eval_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM attempts WHERE eval_id=? ORDER BY attempt_number", (eval_id,))]

    def stage(self, name: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT status,state_json FROM stages WHERE name=?", (name,)).fetchone()
        return {"status": row[0], **json.loads(row[1])} if row else None

    def save_stage(self, name: str, status: str, state: Mapping[str, Any]) -> None:
        """The DB commits first; checkpoint.json is a recoverable projection."""
        values = dict(state)
        values.pop("status", None)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute("INSERT INTO stages VALUES (?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                                    "status=excluded.status,state_json=excluded.state_json,updated_at_utc=excluded.updated_at_utc",
                                    (name, status, canonical_json(values), _now()))
            self.event("stage_checkpoint", "", {"stage": name, "status": status})
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        self.write_checkpoint()

    def write_checkpoint(self) -> None:
        stages = {row[0]: {"status": row[1], **json.loads(row[2])} for row in
                  self.connection.execute("SELECT name,status,state_json FROM stages")}
        write_json(self.run_dir / "checkpoint.json", {
            "version": self.VERSION, "config_hash": self.config_hash,
            "budget": self.budget_snapshot(), "stages": stages,
        })

    def training_data(self) -> list[dict[str, Any]]:
        """Only accepted standard train data; holdout labels cannot reach this API."""
        return self._dataset("train", accepted_only=True)

    def final_holdout_data(self, freeze_hash: str) -> list[dict[str, Any]]:
        """Expose labels only to the durable, frozen M5 experiment."""
        from .surrogate import digest
        state = self.stage("verify")
        if (not state or state.get("freeze_hash") != freeze_hash
                or digest(state.get("freeze")) != freeze_hash):
            raise ValueError("holdout access requires the committed M5 model/candidate freeze")
        event = self.connection.execute(
            "SELECT payload_json FROM events WHERE kind='holdout_unsealed' ORDER BY event_id DESC LIMIT 1").fetchone()
        if not event or json.loads(event[0]).get("freeze_hash") != freeze_hash:
            raise ValueError("holdout access requires an explicit unsealing event")
        return self._dataset("holdout", accepted_only=True)

    def exclusion_coordinates(self) -> list[dict[str, Any]]:
        """Coordinate-only view of successes/failures/reservations; never exposes labels."""
        return [dict(row) for row in self.connection.execute(
            "SELECT design_id,m,p,u_m,u_p FROM designs ORDER BY design_id")]

    def model(self, model_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT config_hash,record_json FROM models WHERE model_id=?", (model_id,)).fetchone()
        if row is None:
            return None
        if row[0] != self.config_hash:
            raise ValueError("model registry configuration mismatch")
        return json.loads(row[1])

    def save_model_record(self, record: Mapping[str, Any]) -> None:
        if record["config_hash"] != self.config_hash:
            raise ValueError("cannot register a model from another frozen configuration")
        final = self.stage("verify")
        if (final and final.get("freeze") and final["freeze"]["model"]["model_id"] == record["model_id"]
                and final["freeze"]["model"] != dict(record)):
            raise ValueError("cannot change the frozen M5 final GP registry")
        db = self.connection
        db.execute("BEGIN IMMEDIATE")
        try:
            previous = self.model(record["model_id"])
            if previous and previous["metadata"]["training_data_hash"] != record["metadata"]["training_data_hash"]:
                raise ValueError("model ID cannot be reused for different training data")
            if previous != record:
                db.execute("INSERT INTO models VALUES (?,?,?) ON CONFLICT(model_id) DO UPDATE SET record_json=excluded.record_json",
                           (record["model_id"], self.config_hash, canonical_json(dict(record))))
                self.event("model_updated" if previous else "model_registered", "", {
                    "model_id": record["model_id"], "training_data_hash": record["metadata"]["training_data_hash"],
                    "model_sha256": record["metadata"]["artifact"]["sha256"]})
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise

    def holdout_coordinates(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute(
            "SELECT design_id,m,p,t,u_m,u_p,split FROM designs WHERE split='holdout' ORDER BY rowid")]

    def _dataset(self, split: str, *, accepted_only: bool) -> list[dict[str, Any]]:
        rows = self.connection.execute("""
            SELECT d.*,e.eval_id,e.status,e.result_json FROM evaluations e JOIN designs d USING(design_id)
            WHERE d.split=? AND e.config_hash=? AND e.panel_count=?
              AND e.purpose IN ('baseline','doe','sequential') ORDER BY e.rowid
        """, (split, self.config_hash, self.standard_n))
        latest = {}
        for row in rows:
            if not row["result_json"]:
                continue
            result = EvaluationResult.from_dict(json.loads(row["result_json"]))
            if accepted_only and (result.status != "success" or not result.aerodynamics):
                continue
            record = {key: row[key] for key in ("design_id", "m", "p", "t", "u_m", "u_p", "split", "eval_id", "status")}
            record.update(config_hash=self.config_hash, panel_count=self.standard_n,
                          purpose=result.purpose, failure_reason=result.failure_reason)
            # Review values remain in audited results, never masquerade as accepted CSV labels.
            record.update({key: (result.aerodynamics or {}).get(key) if result.status == "success" else None
                           for key in ("alpha_deg", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr", "cl_error")})
            latest[result.design_id] = record
        return list(latest.values())

    def export_csv(self, *, include_holdout: bool = True) -> dict[str, Any]:
        fields = ["design_id", "m", "p", "t", "u_m", "u_p", "split", "eval_id", "status", "config_hash",
                  "panel_count", "purpose", "failure_reason", "alpha_deg", "cl", "cd", "cdp", "cm", "top_xtr", "bot_xtr", "cl_error"]
        outputs = {}
        datasets_path = self.run_dir / "datasets.json"
        if not include_holdout and datasets_path.is_file():
            previous = json.loads(datasets_path.read_text(encoding="utf-8"))
            if previous["config_hash"] != self.config_hash:
                raise ValueError("dataset snapshot configuration mismatch")
            if "holdout.csv" in previous["files"]:
                outputs["holdout.csv"] = previous["files"]["holdout.csv"]
        for name, split, accepted in (("samples.csv", "train", True), ("failures.csv", "train", False),
                                      ("holdout.csv", "holdout", False)):
            if split == "holdout" and not include_holdout:
                continue
            records = self._dataset(split, accepted_only=accepted)
            if name == "failures.csv":
                records = [row for row in records if row["status"] != "success"]
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
            path = self.run_dir / name
            temp = path.with_suffix(".csv.tmp")
            temp.write_text(buffer.getvalue(), encoding="utf-8", newline="")
            temp.replace(path)
            outputs[name] = {"rows": len(records), "sha256": file_sha256(path)}
        write_json(self.run_dir / "datasets.json", {
            "config_hash": self.config_hash, "holdout_labels_sealed": True,
            "holdout_policy": "Audit storage only. Do not train, rank candidates or stop with holdout labels.",
            "files": outputs,
        })
        return outputs


