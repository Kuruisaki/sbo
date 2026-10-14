"""Read and validate XFOIL polar output without changing experiment data.

Use ``parse_attempt(attempt, config)`` after xfoil_runner.run_xfoil. Lower-level
text/file APIs support offline diagnostics and regression fixtures. Only a
``success`` result is usable for training; ``needs_review`` retains the measured
point but requires the evaluator's explicit human-review event.

Headers determine column positions. No fixed line skip or last-row assumption is
used. A valid target must match the final solve in the log, including viscous
mode and a save-file write. XFOIL 6.99 xoper.f calls PLRADD only when LVCONV is
true in viscous mode; xpol.f emits that write marker. MRCHDU warnings during
iterations do not by themselves invalidate a later converged solution.

Solver source: tools/xfoil/Xfoil699src.zip, Xfoil699src/src/{xoper,xpol}.f.
This parser verifies attempt-local artifacts, not database/cache provenance or
the baseline/final panel-convergence gates, which belong to the evaluator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any


PARSER_VERSION = "xfoil-polar-v1"
_REQUIRED = frozenset({"alpha_deg", "cl", "cd", "cm"})
_ALIASES = {"alpha": "alpha_deg", "alfa": "alpha_deg", "bottom_xtr": "bot_xtr"}
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?\Z")
_OPER = re.compile(r"\.OPER([a-z]*)\s+c>", re.IGNORECASE)
_ALPHA_CL = re.compile(r"\b(?:a|alpha)\s*=\s*(\S+)\s+CL\s*=\s*(\S+)", re.IGNORECASE)
_CM_CD = re.compile(r"\bCm\s*=\s*(\S+)\s+CD\s*=\s*(\S+)", re.IGNORECASE)
_WRITE = re.compile(r"Point written to save file\s+(.+?)\s*$", re.IGNORECASE)
_TERMINAL_FAILURE = re.compile(
    r"VISCAL\s*:\s*Convergence failed|^\s*(?:viscous\s+)?Convergence failed\s*$"
    r"|viscous solution (?:failed|not converged)|Sequence halted.*did not converge", re.IGNORECASE | re.MULTILINE,
)
_FATAL = re.compile(
    r"forrtl\s*:\s*severe|segmentation fault|SIGSEGV|floating[ -]point exception"
    r"|access violation|fatal error|Traceback \(most recent call last\)|input not recognized",
    re.IGNORECASE,
)


class PolarConfigurationError(ValueError):
    """Invalid acceptance settings; critical thresholds are never guessed."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise PolarConfigurationError(f"{name} must be a finite real number")
    return float(value)


def _number(token: str) -> float:
    if not _NUMBER.fullmatch(token):
        raise ValueError(f"invalid numeric token: {token!r}")
    value = float(token.replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ValueError(f"nonfinite numeric token: {token!r}")
    return value


@dataclass(frozen=True)
class PolarSettings:
    target_cl: float
    cl_tolerance: float
    abs_alpha_deg_review_threshold: float
    cd_review_threshold: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PolarSettings:
        try:
            flow, solver = config["flow"], config["xfoil"]
            review = config["quality"]["manual_review"]
            for key in ("accept_only_target_cl", "require_normal_exit", "require_target_convergence", "require_positive_cd"):
                if solver[key] is not True:
                    raise PolarConfigurationError(f"xfoil.{key} must be true")
            columns = solver["required_finite_columns"]
            if not isinstance(columns, (list, tuple)) or len(columns) != 4 or set(columns) != _REQUIRED:
                raise PolarConfigurationError("required_finite_columns must contain alpha_deg, cl, cd, cm")
            if flow["viscous"] is not True:
                raise PolarConfigurationError("flow.viscous must be true")
            if review["status"] != "needs_review" or review["exclude_from_training_until_accepted"] is not True:
                raise PolarConfigurationError("manual-review points must be withheld pending explicit acceptance")
            settings = cls(
                _finite(flow["target_cl"], "target_cl"), _finite(flow["cl_tolerance"], "cl_tolerance"),
                _finite(review["abs_alpha_deg_above"], "abs_alpha_deg_above"),
                _finite(review["cd_above"], "cd_above"),
            )
        except (KeyError, TypeError) as exc:
            raise PolarConfigurationError(f"missing or malformed polar configuration: {exc}") from exc
        settings.validate()
        return settings

    def validate(self) -> None:
        _finite(self.target_cl, "target_cl")
        for name in ("cl_tolerance", "abs_alpha_deg_review_threshold", "cd_review_threshold"):
            value = _finite(getattr(self, name), name)
            if value < 0 or (name != "cl_tolerance" and value == 0):
                raise PolarConfigurationError(f"{name} must be {'nonnegative' if name == 'cl_tolerance' else 'positive'}")


@dataclass(frozen=True)
class ParseIssue:
    code: str
    line_number: int
    message: str
    cl: float | None = None


@dataclass(frozen=True, kw_only=True)
class PolarRow:
    alpha_deg: float
    cl: float
    cd: float
    cm: float
    cdp: float | None = None
    top_xtr: float | None = None
    bot_xtr: float | None = None
    line_number: int
    extra_columns: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, kw_only=True)
class AerodynamicResult(PolarRow):
    cl_error: float


@dataclass(frozen=True)
class PolarTable:
    columns: tuple[str, ...] = ()
    rows: tuple[PolarRow, ...] = ()
    issues: tuple[ParseIssue, ...] = ()
    fatal_error: str | None = None


def _columns(line: str) -> tuple[str, ...]:
    return tuple(_ALIASES.get(word.lower(), word.lower()) for word in line.lstrip("# ").split())


def parse_polar_text(text: str) -> PolarTable:
    """Parse a single polar table, retaining finite rows and damaged-row diagnostics.

    Optional columns absent from the header become None. Extra named numeric
    columns are retained. A damaged row's CL is trusted only when its field count
    matches the header; otherwise its target identity remains unknown.
    """
    if not text.strip():
        return PolarTable(fatal_error="empty_polar")
    header: tuple[str, ...] = ()
    rows, issues = [], []
    for line_number, line in enumerate(text.splitlines(), 1):
        candidate = _columns(line)
        if header and line.lstrip().startswith("#"):
            if _REQUIRED.issubset(candidate):
                return PolarTable(header, tuple(rows), tuple(issues), "multiple_polar_tables")
            continue
        is_header = len(_REQUIRED.intersection(candidate)) >= 2
        if is_header:
            if header:
                return PolarTable(header, tuple(rows), tuple(issues), "multiple_polar_tables")
            if len(candidate) != len(set(candidate)):
                return PolarTable(fatal_error="duplicate_columns")
            if not _REQUIRED.issubset(candidate):
                return PolarTable(candidate, fatal_error="missing_required_columns")
            header = candidate
            continue
        if not header or not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = line.split()
        if all(set(token).issubset({"-", "="}) for token in tokens):
            continue
        if len(tokens) != len(header):
            issues.append(ParseIssue("wrong_field_count", line_number, "row length differs from header"))
            continue
        cl = None
        try:
            cl = _number(tokens[header.index("cl")])
        except ValueError:
            pass
        try:
            values = {name: _number(token) for name, token in zip(header, tokens)}
        except ValueError as exc:
            issues.append(ParseIssue("invalid_numeric_row", line_number, str(exc), cl))
            continue
        standard = _REQUIRED | {"cdp", "top_xtr", "bot_xtr"}
        rows.append(PolarRow(
            alpha_deg=values["alpha_deg"], cl=values["cl"], cd=values["cd"], cm=values["cm"],
            cdp=values.get("cdp"), top_xtr=values.get("top_xtr"), bot_xtr=values.get("bot_xtr"),
            line_number=line_number,
            extra_columns=tuple((name, values[name]) for name in header if name not in standard),
        ))
    return PolarTable(header, tuple(rows), tuple(issues), None if header else "missing_column_header")


@dataclass(frozen=True)
class SolveEvidence:
    start_line: int
    end_line: int
    viscous: bool | None
    alpha_deg: float | None
    cl: float | None
    cd: float | None
    cm: float | None
    written_file: str | None
    failures: tuple[str, ...]
    iteration_warnings: int


@dataclass
class _SolveBlock:
    start_line: int
    viscous: bool | None = None
    alpha_deg: float | None = None
    cl: float | None = None
    cd: float | None = None
    cm: float | None = None
    written_file: str | None = None
    failures: list[str] = field(default_factory=list)
    iteration_warnings: int = 0
    has_solve: bool = False

    def freeze(self, line: int) -> SolveEvidence:
        return SolveEvidence(
            self.start_line, line, self.viscous, self.alpha_deg, self.cl, self.cd,
            self.cm, self.written_file, tuple(dict.fromkeys(self.failures)), self.iteration_warnings,
        )


def parse_log_evidence(text: str) -> tuple[SolveEvidence, ...]:
    """Group solve evidence by OPER prompts, retaining the latest iteration state."""
    blocks = []
    current = _SolveBlock(1)
    lines = text.splitlines()
    for line_number, line in enumerate(lines, 1):
        prompt = _OPER.search(line)
        if prompt:
            if current.has_solve:
                blocks.append(current.freeze(line_number - 1))
            current = _SolveBlock(line_number, viscous="v" in prompt.group(1).lower())
        alpha_cl = _ALPHA_CL.search(line)
        if alpha_cl:
            current.has_solve = True
            current.cd = current.cm = None  # Never pair new CL with an earlier iteration's CD.
            current.written_file = None  # A write before the latest state proves nothing about it.
            try:
                current.alpha_deg, current.cl = map(_number, alpha_cl.groups())
            except ValueError:
                current.alpha_deg = current.cl = None
                current.failures.append("target_log_numeric_error")
        cm_cd = _CM_CD.search(line)
        if cm_cd:
            current.has_solve = True
            current.written_file = None
            try:
                current.cm, current.cd = map(_number, cm_cd.groups())
            except ValueError:
                current.cm = current.cd = None
                current.failures.append("target_log_numeric_error")
        if _TERMINAL_FAILURE.search(line):
            current.has_solve = True
            current.failures.append("target_convergence_failed")
        if "MRCHDU:" in line and "Convergence failed" in line:
            current.iteration_warnings += 1
        written = _WRITE.search(line)
        if written:
            current.has_solve = True
            current.written_file = written.group(1).strip().strip("\"'").replace("\\", "/").rsplit("/", 1)[-1]
    if current.has_solve:
        blocks.append(current.freeze(len(lines)))
    return tuple(blocks)


@dataclass(frozen=True)
class PolarResult:
    status: str
    failure_reason: str | None = None
    failure_details: str | None = None
    aerodynamics: AerodynamicResult | None = None
    table: PolarTable | None = None
    evidence: SolveEvidence | None = None
    review_flags: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    artifact_hashes: tuple[tuple[str, str], ...] = ()
    parser_version: str = PARSER_VERSION

    @property
    def usable_for_training(self) -> bool:
        return self.status == "success" and self.aerodynamics is not None

    def to_dict(self) -> dict[str, Any]:
        """Contains only finite numbers or None; safe for json.dumps(allow_nan=False)."""
        return asdict(self)


def _process_failure(process_status: str, returncode: int | None) -> str | None:
    if process_status != "completed":
        return "process_timeout" if process_status == "timeout" else "process_not_completed"
    if isinstance(returncode, bool) or not isinstance(returncode, int) or returncode != 0:
        return "process_returncode_invalid"
    return None


def _within_cl(cl: float, settings: PolarSettings) -> bool:
    # Account only for binary roundoff at the inclusive CL tolerance boundary.
    rounding = 16.0 * math.ulp(max(abs(cl), abs(settings.target_cl), 1.0))
    return abs(cl - settings.target_cl) <= settings.cl_tolerance + rounding


def validate_polar_text(
    polar_text: str, stdout_text: str, stderr_text: str, *, settings: PolarSettings,
    process_status: str, returncode: int | None, polar_filename: str = "polar.txt",
) -> PolarResult:
    """Select one target row and verify its final viscous solve/write evidence."""
    settings.validate()
    process_error = _process_failure(process_status, returncode)
    if process_error:
        return PolarResult("failed", process_error, f"status={process_status}; returncode={returncode}")
    table = parse_polar_text(polar_text)
    fail = lambda code, **details: PolarResult("failed", code, table=table, **details)
    if table.fatal_error:
        return fail(table.fatal_error)
    if _FATAL.search(stdout_text) or _FATAL.search(stderr_text):
        return fail("fatal_solver_log")
    # stderr has no reliable solve boundaries; a terminal failure here is ambiguous.
    if _TERMINAL_FAILURE.search(stderr_text):
        return fail("unscoped_stderr_convergence_failure")
    for issue in table.issues:
        if issue.cl is None or _within_cl(issue.cl, settings):
            return fail("damaged_target_or_unidentified_row", failure_details=f"line {issue.line_number}: {issue.message}")
    blocks = parse_log_evidence(stdout_text)
    evidence = blocks[-1] if blocks else None
    if evidence is not None and evidence.failures:
        return fail(evidence.failures[-1], evidence=evidence)
    candidates = [row for row in table.rows if _within_cl(row.cl, settings)]
    if not candidates:
        return fail("target_row_missing")
    if len(candidates) != 1:
        return fail("ambiguous_target_rows")
    row = candidates[0]
    if row.cd <= 0:
        return fail("nonpositive_cd")
    if any(value is not None and not 0 <= value <= 1 for value in (row.top_xtr, row.bot_xtr)):
        return fail("invalid_transition_location")
    if evidence is None:
        return fail("target_state_unconfirmed")
    # File row order may differ; the target solve must finish the session.
    if evidence.viscous is not True:
        return fail("target_viscous_mode_unconfirmed", evidence=evidence)
    if evidence.written_file != polar_filename:
        return fail("target_polar_write_unconfirmed", evidence=evidence)
    if any(getattr(evidence, name) is None for name in _REQUIRED):
        return fail("target_state_unconfirmed", evidence=evidence)
    # XFOIL stdout prints alpha/CL/CD/CM at 3/4/5/4 decimals. Permit half a
    # printed unit plus machine roundoff when a polar has higher precision.
    precisions = {"alpha_deg": 0.0005, "cl": 0.00005, "cd": 0.000005, "cm": 0.00005}
    for name, tolerance in precisions.items():
        if not math.isclose(getattr(evidence, name), getattr(row, name), rel_tol=0.0, abs_tol=tolerance + 1e-14):
            return fail("polar_log_mismatch", failure_details=name, evidence=evidence)
    if not _within_cl(evidence.cl, settings):
        return fail("target_log_cl_mismatch", evidence=evidence)
    review = []
    if abs(row.alpha_deg) > settings.abs_alpha_deg_review_threshold:
        review.append("abs_alpha_deg_above_threshold")
    if row.cd > settings.cd_review_threshold:
        review.append("cd_above_threshold")
    warnings = tuple(f"ignored_damaged_nontarget_row:{issue.line_number}" for issue in table.issues)
    if any(block.iteration_warnings for block in blocks):
        warnings += ("intermediate_boundary_layer_iteration_warnings",)
    aerodynamic = AerodynamicResult(**asdict(row), cl_error=abs(row.cl - settings.target_cl))
    return PolarResult(
        "needs_review" if review else "success", aerodynamics=aerodynamic,
        table=table, evidence=evidence, review_flags=tuple(review), warnings=warnings,
    )


def parse_polar_file(
    polar_path: str | Path, stdout_path: str | Path, stderr_path: str | Path, *,
    settings: PolarSettings, process_status: str, returncode: int | None,
) -> PolarResult:
    """Read one attempt's files. File presence alone never certifies a sample."""
    settings.validate()
    process_error = _process_failure(process_status, returncode)
    if process_error:
        return PolarResult("failed", process_error, f"status={process_status}; returncode={returncode}")
    paths = tuple(Path(path) for path in (polar_path, stdout_path, stderr_path))
    if len({path.resolve().parent for path in paths}) != 1:
        return PolarResult("failed", "mixed_attempt_directories")
    raw = []
    for label, path in zip(("polar", "stdout", "stderr"), paths):
        try:
            raw.append(path.read_bytes())
        except OSError as exc:
            return PolarResult("failed", f"{label}_file_unreadable", str(exc))
    result = validate_polar_text(
        *(data.decode("utf-8", errors="replace") for data in raw), settings=settings,
        process_status=process_status, returncode=returncode, polar_filename=paths[0].name,
    )
    return replace(result, artifact_hashes=tuple((label, sha256(data).hexdigest()) for label, data in zip(("polar", "stdout", "stderr"), raw)))


def parse_attempt(attempt: Any, config: Mapping[str, Any] | PolarSettings) -> PolarResult:
    """Validate an XfoilAttemptResult, binding input hashes and attempt-local paths.

    No files, database state or review decisions are modified. The caller still
    owns frozen-config identity, flow/fidelity checks and training-data selection.
    """
    settings = config if isinstance(config, PolarSettings) else PolarSettings.from_config(config)
    settings.validate()
    process_error = _process_failure(attempt.status, attempt.returncode)
    if process_error:
        return PolarResult("failed", process_error, attempt.failure_reason)
    if not attempt.launched or attempt.remaining_pids or attempt.cleanup_errors:
        return PolarResult("failed", "attempt_lifecycle_unconfirmed")
    if not attempt.cl_sequence or attempt.cl_sequence[-1] != settings.target_cl:
        return PolarResult("failed", "attempt_target_mismatch")
    directory = Path(attempt.attempt_dir).resolve()
    for attribute, filename in (
        ("polar_path", "polar.txt"), ("stdout_path", "stdout.log"), ("stderr_path", "stderr.log"),
        ("input_path", "input.txt"), ("airfoil_path", "airfoil.dat"),
    ):
        if Path(getattr(attempt, attribute)).resolve() != directory / filename:
            return PolarResult("failed", "attempt_artifact_path_mismatch", attribute)
    for attribute, expected in (("input_path", attempt.session_sha256), ("airfoil_path", attempt.geometry_file_sha256)):
        try:
            actual = sha256(Path(getattr(attempt, attribute)).read_bytes()).hexdigest()
        except OSError as exc:
            return PolarResult("failed", "attempt_input_unreadable", str(exc))
        if actual != expected:
            return PolarResult("failed", "attempt_input_hash_mismatch", attribute)
    return parse_polar_file(
        attempt.polar_path, attempt.stdout_path, attempt.stderr_path,
        settings=settings, process_status=attempt.status, returncode=attempt.returncode,
    )
