"""Archive collector: AST/file reads, read-only SQLite and read-only Git commands.

Only docs/development_record is written. No sbo import, solver, fitting or tests.
Codex exports retain public messages/tool evidence, never internal reasoning.
"""
from __future__ import annotations
import ast
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "docs/development_record"
RAW = ARCHIVE / "raw"
CUTOFF = datetime.fromtimestamp(1791993288, timezone.utc)
THREADS = {"01a133bb-c9f7-7ba2-ba1d-5db0b83e1224": "planning",
           "01a13494-856c-7fd2-84bb-26ef79b25c2a": "main",
           "01a13ac2-29d7-7df0-a1d7-ceaa9b88c3a3": "git"}


def hash_file(path):
    return sha256(path.read_bytes()).hexdigest()


def write(name, value):
    target = (ARCHIVE / name).resolve()
    assert target.is_relative_to(ARCHIVE.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(*args):
    p = subprocess.run(["git", "-c", "core.quotepath=false", *args], cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace")
    return {"command": ["git", *args], "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def redact(text):
    text = re.sub(r"(?i)(authorization\s*:\s*(?:basic|bearer)\s+)[A-Za-z0-9+/=_\-.]+", r"\1[REDACTED]", text)
    text = re.sub(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})", "[REDACTED_TOKEN]", text)
    return text


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    git_results = {"status": git("status", "--short", "--branch"),
                   "log": git("log", "--oneline", "--decorate", "--graph", "--all"),
                   "diff": git("diff"), "staged": git("diff", "--staged"), "branches": git("branch", "-vv"),
                   "stat": git("log", "--stat"), "names": git("log", "--name-status"),
                   "remote": git("remote", "-v"), "head": git("rev-parse", "HEAD"), "tracked": git("ls-files")}
    for key, filename in [("status", "git_status.txt"), ("log", "git_log.txt"), ("branches", "git_branches.txt"),
                          ("stat", "git_log_stat.txt"), ("names", "git_log_name_status.txt"), ("remote", "git_remote.txt")]:
        result = git_results[key]
        write("raw/" + filename, result["stdout"] + result["stderr"])
    write("raw/git_diff.patch", git_results["diff"]["stdout"] + git_results["staged"]["stdout"])
    write("raw/git_commands.json", git_results)

    sources = [*sorted((ROOT / "src/sbo").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py")), *sorted((ROOT / "tests").glob("*.py"))]
    inventory = []
    for path in sources:
        body = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(body, filename=str(path))
        functions = [{"name": node.name, "line": node.lineno, "end_line": node.end_lineno,
                      "signature": ast.unparse(node.args), "docstring": ast.get_docstring(node),
                      "calls": sorted({ast.unparse(call.func) for call in ast.walk(node) if isinstance(call, ast.Call)})}
                     for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        arguments = [{"line": node.lineno, "call": ast.unparse(node)} for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"]
        inventory.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": hash_file(path), "lines": len(body.splitlines()),
                          "docstring": ast.get_docstring(tree), "imports": [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))],
                          "classes": [{"name": node.name, "line": node.lineno, "docstring": ast.get_docstring(node)} for node in ast.walk(tree) if isinstance(node, ast.ClassDef)],
                          "functions": functions, "cli_arguments": arguments})
    write("raw/source_inventory.json", inventory)
    config = yaml.safe_load((ROOT / "configs/naca2412.yaml").read_text(encoding="utf-8"))
    write("raw/config_snapshot.json", {"path": "configs/naca2412.yaml", "file_sha256": hash_file(ROOT / "configs/naca2412.yaml"), "config": config})

    main_run = ROOT / "runs/m1_20261013_02"
    snapshot_files = ["manifest.json", "preflight.json", "baseline.json", "doe.json", "datasets.json", "fit.json", "optimization.json",
                      "verification.json", "report.json", "checkpoint.json", "m2-recovery-validation.json", "m3-recovery-validation.json",
                      "m4-recovery-validation.json", "m5-recovery-validation.json", "m4-test-summary.json", "m5-test-summary.json",
                      "recovery-validation.json", "test-summary.json", "m2-test-summary.json", "m3-test-summary.json", "environment.json"]
    artifacts = {}
    for filename in snapshot_files:
        path = main_run / filename
        if path.is_file():
            artifacts[filename] = {"sha256": hash_file(path), "data": json.loads(path.read_text(encoding="utf-8"))}
    db_path = main_run / "samples.sqlite"
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = [dict(row) for row in db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")]
        counts = {row["name"]: db.execute('SELECT COUNT(*) FROM "' + row["name"] + '"').fetchone()[0] for row in tables}
        events = [dict(row) for row in db.execute("SELECT event_id,kind,eval_id,timestamp_utc,payload_json FROM events ORDER BY event_id")]
        ledger = {"open_mode": "read-only URI mode=ro", "tables": tables, "row_counts": counts,
                  "evaluation_counts": [dict(row) for row in db.execute("SELECT purpose,status,panel_count,COUNT(*) AS count FROM evaluations GROUP BY purpose,status,panel_count ORDER BY purpose,panel_count")],
                  "launches": db.execute("SELECT COALESCE(SUM(launched),0) FROM attempts").fetchone()[0],
                  "reservations": db.execute("SELECT COUNT(*) FROM attempts WHERE status='reserved' AND launched=0").fetchone()[0],
                  "stages": [dict(row) for row in db.execute("SELECT name,status,updated_at_utc FROM stages ORDER BY name")],
                  "split_counts": [dict(row) for row in db.execute("SELECT split,COUNT(*) AS count FROM designs GROUP BY split")],
                  "model_ids": [row[0] for row in db.execute("SELECT model_id FROM models ORDER BY model_id")], "events": events}
    write("raw/run_evidence.json", {"run_id": main_run.name, "artifacts": artifacts, "sqlite": ledger})

    run_inventory = []
    for directory in sorted((ROOT / "runs").iterdir()):
        if not directory.is_dir():
            continue
        row = {"name": directory.name, "top_level_files": sorted(path.name for path in directory.iterdir() if path.is_file())}
        manifest = directory / "manifest.json"
        if manifest.is_file():
            try:
                content = json.loads(manifest.read_text(encoding="utf-8-sig"))
                row["manifest_summary"] = {k: content.get(k) for k in ("run_id", "status", "stage", "preflight_passed", "m5_passed", "source_run", "validation_scope", "budget", "failure_reason")}
            except (ValueError, OSError) as exc:
                row["read_error"] = str(exc)
        validations = {}
        for path in directory.glob("*validation.json"):
            try:
                validations[path.name] = json.loads(path.read_text(encoding="utf-8-sig"))
            except (ValueError, OSError):
                pass
        row["validations"] = validations
        run_inventory.append(row)
    write("raw/run_inventory.json", run_inventory)

    local_records, local_sources, public_counts = [], [], Counter()
    for path in sorted(Path(r"C:\Users\13626\.codex\sessions").rglob("*.jsonl")):
        if not any(thread in path.name for thread in THREADS):
            continue
        thread = next(thread for thread in THREADS if thread in path.name)
        local_sources.append({"thread_id": thread, "label": THREADS[thread], "filename": path.name, "sha256_at_snapshot": hash_file(path)})
        calls = {}
        turn_id = None
        for number, line in enumerate(path.open(encoding="utf-8"), 1):
            try:
                item = json.loads(line)
                time = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
            except (ValueError, KeyError):
                continue
            if time >= CUTOFF:
                continue
            payload = item.get("payload", {})
            if item.get("type") == "event_msg" and payload.get("type") == "task_started":
                turn_id = payload.get("turn_id")
            if item.get("type") != "response_item":
                continue
            kind = payload.get("type")
            base = {"thread_id": thread, "thread_label": THREADS[thread], "turn_id": turn_id, "timestamp": item["timestamp"], "source_line": number}
            if kind in {"function_call", "custom_tool_call"}:
                name = payload.get("name", "")
                argument = payload.get("arguments", payload.get("input", ""))
                if name.endswith("exec") or name.endswith("exec_command") or name.endswith("write_stdin") or name.endswith("apply_patch"):
                    if any(term in str(argument) for term in ("read_thread", "auth.json", "credentials", "sandbox-secrets")):
                        continue
                    call_id = payload.get("call_id")
                    # Keep request provenance, but do not republish external web
                    # page bodies in this local execution archive.
                    if "web__run" not in str(argument):
                        calls[call_id] = name
                    local_records.append({**base, "kind": "tool_call", "name": name, "call_id": call_id, "arguments": redact(str(argument))})
                    public_counts["tool_calls"] += 1
            elif kind in {"function_call_output", "custom_tool_call_output"} and payload.get("call_id") in calls:
                output = payload.get("output", "")
                if isinstance(output, list):
                    output = "\n".join(block.get("text", "") for block in output
                                       if isinstance(block, dict) and block.get("type") in {"text", "input_text"})
                if isinstance(output, str):
                    local_records.append({**base, "kind": "tool_output", "name": calls[payload["call_id"]],
                                          "call_id": payload["call_id"], "output": redact(output)})
                    public_counts["tool_outputs"] += 1
    local_records.sort(key=lambda row: (row["timestamp"], row["thread_id"], row["source_line"]))
    write("raw/execution_log.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in local_records))
    write("raw/local_history_sources.json", {"cutoff_utc": CUTOFF.isoformat(), "sources": local_sources, "export_counts": public_counts,
                                             "excluded": ["internal reasoning/analysis", "system/developer messages", "creator/account identifiers", "unrelated sessions", "archive-generation turn", "credential reads", "external web-page response bodies", "image payloads"],
                                             "redaction": "Authorization headers and known API/GitHub token patterns are redacted."})

    protected = [*sources, ROOT / "configs/naca2412.yaml", ROOT / "README.md", ROOT / "SBO_代码目录与接口设计.md", ROOT / "SBO_流程与部署指南.md",
                 ROOT / "environment.yml", ROOT / "environment-win-64.lock.txt", ROOT / "pyproject.toml", ROOT / ".gitignore", db_path]
    protected.extend(path for path in main_run.rglob("*") if path.is_file())
    if not (RAW / "baseline_integrity.json").exists():
        write("raw/baseline_integrity.json", {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "head": git_results["head"]["stdout"].strip(),
              "files": {str(path.relative_to(ROOT)).replace("\\", "/"): hash_file(path) for path in protected},
              "note": "Compare again after archive generation; protected files are read only throughout."})
    print(json.dumps({"source_files": len(inventory), "src_modules": sum(row["path"].startswith("src/") for row in inventory),
                      "run_directories": len(run_inventory), "main_launches": ledger["launches"], "local_history": dict(public_counts),
                      "git_head": git_results["head"]["stdout"].strip()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
