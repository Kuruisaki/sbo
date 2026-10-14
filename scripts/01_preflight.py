"""Create a new frozen run and execute the M1 deployment/real-point gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import subprocess
import sys
import os
import fnmatch
import warnings
import time

import psutil

# Keep the documented source-checkout command usable before editable installation.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import yaml
from sbo import __version__
from sbo.config import create_run, file_sha256, load_config, write_json
from sbo.evaluator import Evaluator
from sbo.xfoil_runner import _WindowsJob, _cleanup


def check_environment(config):
    declaration = yaml.safe_load((Path(config["runtime"]["project_root"]) / "environment.yml").read_text(encoding="utf-8"))
    imports = {"numpy": "numpy", "scipy": "scipy", "scikit-learn": "sklearn", "pandas": "pandas",
               "matplotlib": "matplotlib", "pyyaml": "yaml", "joblib": "joblib", "psutil": "psutil",
               "pytest": "pytest", "setuptools": "setuptools", "pip": "pip"}
    packages, errors = {}, []
    for specification in declaration["dependencies"]:
        name, expected = specification.split("=", 1)
        if name == "python":
            actual = ".".join(map(str, sys.version_info[:2]))
            if actual != expected:
                errors.append(f"Python expected {expected}, found {actual}")
            continue
        if name in {"libblas", "libcblas", "liblapack"}:
            _, version_pattern, build_pattern = specification.split("=", 2)
            candidates = list((Path(sys.prefix) / "conda-meta").glob(name + "-*.json"))
            metadata = json.loads(candidates[0].read_text(encoding="utf-8")) if candidates else {}
            matches = (fnmatch.fnmatchcase(metadata.get("version", ""), version_pattern)
                       and fnmatch.fnmatchcase(metadata.get("build", ""), build_pattern))
            packages[name] = {"version": metadata.get("version"), "build": metadata.get("build"), "matches": matches}
            if not matches:
                errors.append(f"{name} does not match the declared BLAS backend")
            continue
        try:
            # Import setuptools before pip to avoid pip's vendored-distutils ordering.
            if name != "pip":
                importlib.import_module(imports[name])
            actual = version(name)
            packages[name] = {"expected": expected, "actual": actual, "matches": actual == expected}
            if actual != expected:
                errors.append(f"{name} expected {expected}, found {actual}")
        except (ImportError, KeyError) as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError("environment checks failed: " + "; ".join(errors))
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    gp = config["gp"]
    kernel = ConstantKernel(gp["constant_value_initial"], gp["constant_value_bounds"]) * Matern(
        length_scale=gp["length_scale_initial"], length_scale_bounds=gp["length_scale_bounds"], nu=gp["matern_nu"])
    # Synthetic dependency probe only: never written into the aerodynamic ledger.
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]], dtype=float)
    y = np.array([0, 1, 1, 2, 1], dtype=float)
    from threadpoolctl import threadpool_limits
    with warnings.catch_warnings(record=True) as notices, threadpool_limits(limits=1):
        model = GaussianProcessRegressor(kernel=kernel, alpha=gp["alpha_regularization"], normalize_y=gp["normalize_y"],
                                         n_restarts_optimizer=gp["n_restarts_optimizer"], random_state=gp["seed"]).fit(x, y)
        mean, std = model.predict([[.25, .75]], return_std=True)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("synthetic GP fit/prediction failed")
    return {"python": sys.version.split()[0], "python_executable": sys.executable,
            "packages": packages, "synthetic_gp_probe_passed": True, "gp_probe_warnings": [str(notice.message) for notice in notices],
            "sbo_version": __version__}


def probe_environment(directory):
    """Contain native-library crashes/hangs so the parent can report a failed gate."""
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(directory / "config.frozen.yaml"), "--dependency-probe"]
    process = root = job = None
    tracked = {}
    deadline = time.monotonic() + 30
    timed_out = False
    try:
        with (directory / "environment-probe.stdout.log").open("wb") as stdout, (directory / "environment-probe.stderr.log").open("wb") as stderr:
            options = {}
            if os.name == "nt":
                job = _WindowsJob()
                options["creationflags"] = subprocess.CREATE_NO_WINDOW | 0x4
            else:
                options["start_new_session"] = True
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, shell=False, **options)
            root = psutil.Process(process.pid)
            if job:
                job.assign(process.pid)
                root.resume()
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=.05)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        if process:
            try:
                remaining, cleanup_errors = _cleanup(process, root, tracked, job)
            finally:
                if job:
                    job.close()
        elif job:
            job.close()
    if timed_out or process.returncode or remaining or cleanup_errors:
        raise RuntimeError(f"dependency probe failed (timeout={timed_out}, returncode={process.returncode}); see environment-probe logs")
    return json.loads((directory / "environment.json").read_text(encoding="utf-8"))


def export_environment_lock(directory):
    # Conda installs its own Python/entrypoint outside the target environment.
    prefix = Path(sys.prefix)
    root = prefix.parent.parent if prefix.parent.name.lower() == "envs" else prefix
    interpreter = root / ("python.exe" if os.name == "nt" else "bin/python")
    entry = root / ("Scripts/conda-script.py" if os.name == "nt" else "bin/conda")
    if not interpreter.is_file() or not entry.is_file():
        raise RuntimeError("cannot find Conda's local entrypoint to export the verified environment")
    lock = directory / "environment-win-64.lock.txt"
    command = [str(interpreter), str(entry), "export", "--prefix", sys.prefix, "--format=explicit", "--file", str(lock)]
    completed = subprocess.run(command, capture_output=True, timeout=30, shell=False,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    (directory / "environment-export.stdout.log").write_bytes(completed.stdout)
    (directory / "environment-export.stderr.log").write_bytes(completed.stderr)
    if completed.returncode or not lock.is_file() or "@EXPLICIT" not in lock.read_text(encoding="utf-8"):
        raise RuntimeError("verified environment lock export failed; see environment-export logs")
    return {"path": str(lock), "sha256": file_sha256(lock)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/naca2412.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--dependency-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.dependency_probe:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetErrorMode(0x8003)
        result = check_environment(load_config(args.config))
        write_json(args.config.parent / "environment.json", result)
        return 0
    if not args.run_id:
        parser.error("--run-id is required")
    try:
        config = load_config(args.config)
        directory = create_run(config, args.run_id)
    except (OSError, ValueError) as exc:
        print(f"Preflight not started: {exc}", file=sys.stderr)
        return 2
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(started_at_utc=datetime.now(timezone.utc).isoformat(), status="running", sbo_version=__version__)
    write_json(manifest_path, manifest)
    report = {"stage": "M1", "run_id": args.run_id, "status": "running", "checks": {}}
    exit_code = 1
    try:
        report["environment"] = probe_environment(directory)
        executable = Path(config["xfoil"]["executable"])
        report["xfoil"] = {"executable": str(executable), "executable_sha256": file_sha256(executable)}
        package = Path(config["runtime"]["project_root"]) / "tools/XFOIL6.99.zip"
        if package.is_file():
            report["xfoil"]["package_sha256"] = file_sha256(package)
        with Evaluator(directory) as evaluator:
            baseline = evaluator.evaluate(save_cp=True)
            report["baseline"] = baseline.to_dict()
            if not baseline.usable_for_training:
                raise RuntimeError(f"baseline did not pass: {baseline.status}/{baseline.failure_reason}")
            times, ids = [], []
            for index in range(5):
                repeat = evaluator.evaluate(purpose="verification", save_cp=True)
                ids.append(repeat.eval_id)
                if repeat.status != "success":
                    raise RuntimeError(f"automatic call {index + 1} did not pass: {repeat.failure_reason}")
                times.append(sum(record["runner"]["elapsed_seconds"] for record in repeat.attempts))
            before = evaluator.store.budget_snapshot()
            cached = evaluator.evaluate(save_cp=True)
            after = evaluator.store.budget_snapshot()
            if not cached.cache_hit or cached.eval_id != baseline.eval_id or before != after:
                raise RuntimeError("accepted baseline cache reuse or budget accounting failed")
            report["automatic_calls"] = {"count": 5, "evaluation_ids": ids, "median_seconds": statistics.median(times),
                                         "slowest_seconds": max(times), "all_passed": True}
            report["budget"] = after
            report["checks"] = {"dependencies_and_gp": True, "baseline_target_point": True,
                                 "five_serial_independent_calls": True, "cache_reuse_without_launch": True,
                                 "no_remaining_solver_processes": True}
        report["environment_lock"] = export_environment_lock(directory)
        report["status"] = "success"
        manifest.update(status="success", preflight_passed=True, environment_lock=report["environment_lock"],
                        baseline_eval_id=baseline.eval_id, budget=report["budget"])
        exit_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
        manifest.update(status="failed", preflight_passed=False, failure_reason=report["failure_reason"])
    finally:
        manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "preflight.json", report)
        write_json(manifest_path, manifest)
        baseline_data = report.get("baseline", {}).get("aerodynamics")
        lines = [f"# M1 预检：{args.run_id}", "", f"状态：{report['status']}", "",
                 "M1 检查环境、真实基准点、串行调用、进程清理和缓存预算。M2 基准离散关卡、DOE 与优化尚未执行。", ""]
        if baseline_data:
            lines += ["| alpha_deg | CL | CD | CM |", "|---:|---:|---:|---:|",
                      f"| {baseline_data['alpha_deg']} | {baseline_data['cl']} | {baseline_data['cd']} | {baseline_data['cm']} |", ""]
        lines += [f"失败原因：{report['failure_reason']}" if report.get("failure_reason") else "全部 M1 关卡通过。", "",
                  "完整证据见 `preflight.json`、`manifest.json`、`samples.sqlite` 和 `evaluations/`。"]
        (directory / "preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "run_directory": str(directory),
                      "failure_reason": report.get("failure_reason")}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
