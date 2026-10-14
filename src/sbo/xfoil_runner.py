"""One isolated XFOIL process per attempt, with bounded execution and audit files.

Example::

    settings = XfoilSettings.from_config(config, attempt_number=1)
    attempt = run_xfoil(geometry, settings, attempt_directory)

``completed`` means normal process execution and matching solver diagnostics, not
a valid aerodynamic sample. polar.py/evaluator.py must check the target polar row,
CL tolerance, convergence and review flags. Retries, budgets and caching belong to
the evaluator; this module never launches an automatic second attempt.

Menu reference: https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt
The two PPAR exit blanks were checked against 6.99 GETPAN in xfoil.f.
Windows uses a kill-on-close job with a suspended initial process so every child
is contained before execution. POSIX uses a fresh process group. psutil records
process identities and provides additional cleanup diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any

import psutil


class XfoilConfigurationError(ValueError):
    """Unsupported or incomplete solver configuration, detected before launch."""


def _keys(values: Any, expected: set[str], name: str) -> None:
    if not isinstance(values, Mapping):
        raise XfoilConfigurationError(f"{name} must be a mapping")
    if set(values) != expected:
        raise XfoilConfigurationError(
            f"{name}: missing={sorted(expected - set(values))}, unknown={sorted(set(values) - expected)}"
        )


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise XfoilConfigurationError(f"{name} must be a finite real number")
    return float(value)


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise XfoilConfigurationError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class XfoilSettings:
    executable: Path
    expected_version: str
    evaluation_policy_version: str
    attempt_number: int
    panel_count: int
    panel_bunching: float
    te_le_density_ratio: float
    refined_le_density_ratio: float
    top_refinement: tuple[float, float]
    bottom_refinement: tuple[float, float]
    reynolds: float
    mach: float
    flow_type: int
    ncrit: float
    xtr: tuple[float, float]
    target_cl: float
    iteration_limit: int
    timeout_seconds: float
    cl_sequence: tuple[float, ...]
    save_cp: bool

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, attempt_number: int = 1,
        panel_count: int | None = None, save_cp: bool = False,
    ) -> XfoilSettings:
        """Read the first or continuation attempt without changing physics."""
        attempt_number = _integer(attempt_number, "attempt_number", 1)
        if attempt_number not in (1, 2):
            raise XfoilConfigurationError("only attempt numbers 1 and 2 are supported")
        try:
            solver, flow = config["xfoil"], config["flow"]
            _keys(solver, {
                "executable", "expected_version", "evaluation_policy_version", "disable_graphics",
                "standard_n", "ppar", "max_attempts", "initial_attempt", "retry_attempt",
                "fresh_process_per_attempt", "fresh_directory_per_attempt", "kill_process_tree_on_timeout",
                "accept_only_target_cl", "require_normal_exit", "require_target_convergence",
                "required_finite_columns", "require_positive_cd",
            }, "xfoil")
            _keys(flow, {"reynolds", "mach", "target_cl", "type", "ncrit", "xtr", "cl_tolerance", "viscous"}, "flow")
            for flag in ("disable_graphics", "fresh_process_per_attempt",
                         "fresh_directory_per_attempt", "kill_process_tree_on_timeout"):
                if solver[flag] is not True:
                    raise XfoilConfigurationError(f"xfoil.{flag} must be true")
            if flow["viscous"] is not True:
                raise XfoilConfigurationError("flow.viscous must be true")
            if _integer(solver["max_attempts"], "max_attempts", 1) != 2:
                raise XfoilConfigurationError("this evaluation policy requires max_attempts=2")
            attempt = solver["initial_attempt" if attempt_number == 1 else "retry_attempt"]
            panel = solver["ppar"]
            _keys(attempt, {"iteration_limit", "timeout_seconds", "cl_sequence"}, "attempt")
            _keys(panel, {"p", "t", "r", "xt", "xb"}, "ppar")
            settings = cls(
                executable=Path(solver["executable"]),
                expected_version=solver["expected_version"],
                evaluation_policy_version=solver["evaluation_policy_version"],
                attempt_number=attempt_number,
                panel_count=solver["standard_n"] if panel_count is None else panel_count,
                panel_bunching=panel["p"], te_le_density_ratio=panel["t"],
                refined_le_density_ratio=panel["r"],
                top_refinement=tuple(panel["xt"]), bottom_refinement=tuple(panel["xb"]),
                reynolds=flow["reynolds"], mach=flow["mach"], flow_type=flow["type"],
                ncrit=flow["ncrit"], xtr=tuple(flow["xtr"]), target_cl=flow["target_cl"],
                iteration_limit=attempt["iteration_limit"], timeout_seconds=attempt["timeout_seconds"],
                cl_sequence=tuple(attempt["cl_sequence"]), save_cp=save_cp,
            )
        except (KeyError, TypeError) as exc:
            raise XfoilConfigurationError(f"missing or malformed XFOIL configuration: {exc}") from exc
        settings.validate()
        return settings

    def validate(self) -> None:
        if not isinstance(self.executable, Path) or not self.executable.is_absolute():
            raise XfoilConfigurationError("executable must be an absolute pathlib.Path")
        if not isinstance(self.expected_version, str) or not re.fullmatch(r"\d+\.\d+", self.expected_version):
            raise XfoilConfigurationError("expected_version must be a version string, e.g. '6.99'")
        if not isinstance(self.evaluation_policy_version, str) or not self.evaluation_policy_version:
            raise XfoilConfigurationError("evaluation_policy_version must be a nonempty string")
        if _integer(self.attempt_number, "attempt_number", 1) not in (1, 2):
            raise XfoilConfigurationError("only two attempts are supported")
        _integer(self.panel_count, "panel_count", 20)
        _integer(self.iteration_limit, "iteration_limit", 1)
        if _real(self.timeout_seconds, "timeout_seconds") <= 0:
            raise XfoilConfigurationError("timeout_seconds must be positive")
        if not 0 <= _real(self.panel_bunching, "panel_bunching") <= 1:
            raise XfoilConfigurationError("panel_bunching must lie in [0, 1]")
        for label in ("te_le_density_ratio", "refined_le_density_ratio", "reynolds", "ncrit"):
            if _real(getattr(self, label), label) <= 0:
                raise XfoilConfigurationError(f"{label} must be positive")
        if not 0 <= _real(self.mach, "mach") < 1:
            raise XfoilConfigurationError("mach must lie in [0, 1)")
        if _integer(self.flow_type, "flow_type", 1) != 1:
            raise XfoilConfigurationError("only fixed Re/Mach TYPE=1 is supported")
        for label in ("top_refinement", "bottom_refinement", "xtr"):
            values = getattr(self, label)
            if len(values) != 2 or any(not 0 <= _real(v, label) <= 1 for v in values):
                raise XfoilConfigurationError(f"{label} must be two values in [0, 1]")
            if label != "xtr" and values[0] > values[1]:
                raise XfoilConfigurationError(f"{label} endpoints must be increasing or equal")
        target = _real(self.target_cl, "target_cl")
        if not self.cl_sequence or any(not math.isfinite(_real(v, "cl_sequence")) for v in self.cl_sequence):
            raise XfoilConfigurationError("cl_sequence must be nonempty and finite")
        if self.cl_sequence[-1] != target:
            raise XfoilConfigurationError("cl_sequence must end at target_cl")
        if not isinstance(self.save_cp, bool):
            raise XfoilConfigurationError("save_cp must be boolean")


def build_session(settings: XfoilSettings) -> str:
    """Create an ASCII menu session using fixed relative filenames and LF endings."""
    settings.validate()
    number = lambda value: format(float(value), ".15g")
    pair = lambda values: " ".join(number(v) for v in values)
    lines = [
        "PLOP", "G", "", "LOAD airfoil.dat", "PPAR",
        f"N {settings.panel_count}", f"P {number(settings.panel_bunching)}",
        f"T {number(settings.te_le_density_ratio)}", f"R {number(settings.refined_le_density_ratio)}",
        f"XT {pair(settings.top_refinement)}", f"XB {pair(settings.bottom_refinement)}",
        "", "",  # First blank applies paneling and re-enters PPAR; second exits.
        "PANE", "OPER", f"TYPE {settings.flow_type}", f"VISC {number(settings.reynolds)}",
        f"MACH {number(settings.mach)}", f"ITER {settings.iteration_limit}",
        "VPAR", f"N {number(settings.ncrit)}", f"XTR {pair(settings.xtr)}", "",
        "PACC", "polar.txt", "",  # Empty dump filename, not a menu exit.
    ]
    lines.extend(f"CL {number(cl)}" for cl in settings.cl_sequence)
    if settings.save_cp:
        lines.append("CPWR cp.txt")
    lines.extend(["PACC", "", "QUIT"])
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class LaunchEvent:
    pid: int
    process_create_time: float
    attempt_number: int
    attempt_dir: Path
    started_at_utc: str


@dataclass(frozen=True)
class XfoilAttemptResult:
    status: str
    failure_reason: str | None
    launched: bool
    pid: int | None
    process_create_time: float | None
    returncode: int | None
    started_at_utc: str
    ended_at_utc: str
    elapsed_seconds: float
    executable: Path
    executable_sha256: str
    geometry_file_sha256: str
    session_sha256: str
    evaluation_policy_version: str
    expected_version: str
    detected_version: str | None
    actual_panel_count: int | None
    attempt_number: int
    panel_count: int
    iteration_limit: int
    timeout_seconds: float
    cl_sequence: tuple[float, ...]
    remaining_pids: tuple[int, ...]
    cleanup_errors: tuple[str, ...]
    attempt_dir: Path
    airfoil_path: Path
    input_path: Path
    stdout_path: Path
    stderr_path: Path
    polar_path: Path
    cp_path: Path | None
    process_path: Path

    @property
    def completed_normally(self) -> bool:
        """Process/identity check only; never a convergence or aerodynamic check."""
        return self.status == "completed" and self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


def _json_write(path: Path, values: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _solver_diagnostics(path: Path) -> tuple[str | None, int | None]:
    """Read bounded chunks even if a failing solver emits very large log lines."""
    version, panels = None, None
    if path.exists():
        with path.open("rb") as stream:
            while raw := stream.readline(65536):
                text = raw.decode("utf-8", errors="replace")
                match = re.search(r"\bXFOIL\s+(?:Version\s+)?(\d+\.\d+)\b", text, re.IGNORECASE)
                if match:
                    version = match.group(1)
                for count in re.findall(r"Number of panel nodes\s+(\d+)", text, re.IGNORECASE):
                    panels = int(count)
    return version, panels


class _WindowsJob:
    """Contain a suspended process before resuming it, including future children."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64),
                        ("Flags", wintypes.DWORD), ("MinWorkingSet", ctypes.c_size_t),
                        ("MaxWorkingSet", ctypes.c_size_t), ("ActiveProcesses", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("Priority", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("Basic", BasicLimits), ("Io", IoCounters),
                        ("ProcessMemory", ctypes.c_size_t), ("JobMemory", ctypes.c_size_t),
                        ("PeakProcessMemory", ctypes.c_size_t), ("PeakJobMemory", ctypes.c_size_t)]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.SetInformationJobObject.restype = wintypes.BOOL
        self.api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenProcess.restype = wintypes.HANDLE
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype = wintypes.BOOL
        self.api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.api.TerminateJobObject.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, pid: int) -> None:
        import ctypes
        handle = self.api.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.api.AssignProcessToJobObject(self.handle, handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.api.CloseHandle(handle)

    def terminate(self) -> None:
        import ctypes
        if self.handle and not self.api.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def _remember_children(root: psutil.Process, tracked: dict[int, psutil.Process]) -> None:
    try:
        for child in root.children(recursive=True):
            tracked[child.pid] = child
    except psutil.NoSuchProcess:
        pass


def _cleanup(
    process: subprocess.Popen[bytes], root: psutil.Process | None,
    tracked: dict[int, psutil.Process], job: _WindowsJob | None,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    errors: list[str] = []
    if root is not None:
        try:
            _remember_children(root, tracked)
        except psutil.AccessDenied as exc:
            errors.append(str(exc))
        tracked[root.pid] = root
    try:
        if job is not None:
            job.terminate()
        elif os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        errors.append(str(exc))
    for identity in reversed(tuple(tracked.values())):
        try:
            identity.kill()  # psutil guards against PID reuse using process identity.
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as exc:
            errors.append(str(exc))
    try:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(str(exc))
    _, alive = psutil.wait_procs(list(tracked.values()), timeout=3.0)
    # Zombies on POSIX cannot execute; their parent/init is responsible for reaping.
    remaining = []
    for identity in alive:
        try:
            if identity.is_running() and identity.status() != psutil.STATUS_ZOMBIE:
                remaining.append(identity.pid)
        except psutil.NoSuchProcess:
            pass
    return tuple(remaining), tuple(errors)


def run_xfoil(
    geometry: Any, settings: XfoilSettings, attempt_dir: str | Path, *,
    on_launch: Callable[[LaunchEvent], None] | None = None,
) -> XfoilAttemptResult:
    """Run exactly one attempt in a directory that must not already exist.

    geometry is the checked GeometryResult from geometry.py. Existing directories
    (even empty), non-ASCII working paths and missing executables fail before a
    launch. ``on_launch`` lets the evaluator record a budget/identity event; it is
    called once after creation and persistence of process.json, before resuming a
    Windows process. A callback error or KeyboardInterrupt cleans up and re-raises.
    Raw logs are streamed directly to disk, so verbose solver output cannot block
    on an undrained pipe. process.json records lifecycle only; result.json is left
    to the aerodynamic evaluator.
    """
    settings.validate()
    executable = settings.executable.resolve(strict=True)
    if not executable.is_file():
        raise XfoilConfigurationError("executable must be a regular file")
    directory = Path(attempt_dir).resolve()
    if not str(directory).isascii():
        raise XfoilConfigurationError("attempt directory must have an ASCII path")
    if not geometry.checks.valid or sha256(geometry.dat_bytes).hexdigest() != geometry.file_sha256:
        raise XfoilConfigurationError("geometry must be checked and match its file hash")
    session_bytes = build_session(settings).encode("ascii")
    exe_hash = sha256(executable.read_bytes()).hexdigest()
    directory.mkdir(parents=True, exist_ok=False)
    airfoil_path, input_path = directory / "airfoil.dat", directory / "input.txt"
    stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
    polar_path = directory / "polar.txt"
    cp_path = directory / "cp.txt" if settings.save_cp else None
    process_path = directory / "process.json"
    airfoil_path.write_bytes(geometry.dat_bytes)
    input_path.write_bytes(session_bytes)
    started = datetime.now(timezone.utc).isoformat()
    clock_start = time.monotonic()
    state: dict[str, Any] = {
        "status": "planned", "launched": False, "pid": None, "process_create_time": None,
        "started_at_utc": started, "attempt_number": int(settings.attempt_number),
        "executable": str(executable), "executable_sha256": exe_hash,
        "geometry_file_sha256": geometry.file_sha256,
        "session_sha256": sha256(session_bytes).hexdigest(),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(settings).items()},
    }
    _json_write(process_path, state)
    process = None
    root = None
    job = None
    tracked: dict[int, psutil.Process] = {}
    status, reason = "launch_failed", None
    remaining: tuple[int, ...] = ()
    cleanup_errors: tuple[str, ...] = ()
    deferred: BaseException | None = None
    callback_active = False
    try:
        with input_path.open("rb") as stdin, stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            options: dict[str, Any] = {}
            if os.name == "nt":
                job = _WindowsJob()
                options["creationflags"] = subprocess.CREATE_NO_WINDOW | 0x00000004  # CREATE_SUSPENDED
            else:
                options["start_new_session"] = True
            process = subprocess.Popen(
                [str(executable)], stdin=stdin, stdout=stdout, stderr=stderr,
                cwd=directory, shell=False, **options,
            )
            state.update(launched=True, pid=process.pid, status="running")
            root = psutil.Process(process.pid)
            state["process_create_time"] = root.create_time()
            if job is not None:
                job.assign(process.pid)
            _json_write(process_path, state)
            if on_launch is not None:
                callback_active = True
                on_launch(LaunchEvent(process.pid, state["process_create_time"], settings.attempt_number, directory, started))
                callback_active = False
            if os.name == "nt":
                root.resume()
            deadline = clock_start + float(settings.timeout_seconds)
            while True:
                _remember_children(root, tracked)
                if process.poll() is not None:
                    status = "completed" if process.returncode == 0 else "crashed"
                    if process.returncode != 0:
                        reason = f"nonzero_returncode:{process.returncode}"
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    status, reason = "timeout", "execution_timeout"
                    break
                try:
                    process.wait(timeout=min(left, 0.05))
                except subprocess.TimeoutExpired:
                    pass
    except BaseException as exc:
        status = "aborted" if process is not None else "launch_failed"
        reason = f"{type(exc).__name__}: {exc}"
        # Operating-system launch/setup failures are returned as diagnostics.
        # Callback failures and interruption must propagate after cleanup.
        if callback_active or not isinstance(exc, (OSError, psutil.Error)):
            deferred = exc
    finally:
        if process is not None:
            try:
                remaining, cleanup_errors = _cleanup(process, root, tracked, job)
            finally:
                if job is not None:
                    job.close()
        elif job is not None:
            job.close()

    detected_version, actual_panels = _solver_diagnostics(stdout_path)
    if remaining:
        status, reason = "cleanup_failed", f"remaining_processes:{remaining}"
    elif cleanup_errors:
        status, reason = "cleanup_failed", "; ".join(cleanup_errors)
    elif status == "completed" and detected_version != settings.expected_version:
        status, reason = "solver_identity_mismatch", f"expected_version:{settings.expected_version}; detected:{detected_version}"
    elif status == "completed" and actual_panels != settings.panel_count:
        status, reason = "solver_setting_mismatch", f"expected_panels:{settings.panel_count}; detected:{actual_panels}"
    result = XfoilAttemptResult(
        status=status, failure_reason=reason, launched=state["launched"], pid=state["pid"],
        process_create_time=state["process_create_time"], returncode=process.returncode if process is not None else None,
        started_at_utc=started, ended_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - clock_start, executable=executable, executable_sha256=exe_hash,
        geometry_file_sha256=geometry.file_sha256, session_sha256=state["session_sha256"],
        evaluation_policy_version=settings.evaluation_policy_version, expected_version=settings.expected_version,
        detected_version=detected_version, actual_panel_count=actual_panels,
        attempt_number=settings.attempt_number, panel_count=settings.panel_count,
        iteration_limit=settings.iteration_limit, timeout_seconds=float(settings.timeout_seconds),
        cl_sequence=settings.cl_sequence, remaining_pids=remaining, cleanup_errors=cleanup_errors,
        attempt_dir=directory, airfoil_path=airfoil_path, input_path=input_path,
        stdout_path=stdout_path, stderr_path=stderr_path, polar_path=polar_path,
        cp_path=cp_path, process_path=process_path,
    )
    _json_write(process_path, {**state, **result.to_dict()})
    if deferred is not None:
        raise deferred
    return result
