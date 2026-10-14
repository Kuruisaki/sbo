"""Polar parsing/acceptance regressions; synthetic rows are only test inputs.

Real fixtures contain archived XFOIL output with source hashes. No test in this
module launches the solver, fills training data or modifies recorded evidence.
"""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4

import pytest
import yaml

from sbo.polar import (
    PolarConfigurationError, PolarSettings, parse_attempt, parse_log_evidence,
    parse_polar_file, parse_polar_text, validate_polar_text,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "configs/naca2412.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def settings(config):
    return PolarSettings.from_config(config)


@pytest.fixture
def lab():
    directory = ROOT / "runs" / f"polar-tests-{uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        assert directory.resolve().parent == (ROOT / "runs").resolve()
        shutil.rmtree(directory)


def _polar(rows=("2.0 0.5 0.006 -0.05",), header="alpha CL CD CM", padding=0):
    return "Synthetic test table\n" + "metadata\n" * padding + header + "\n----- ----- ----- -----\n" + "\n".join(rows) + "\n"


def _log(alpha=2.0, cl=0.5, cd=0.006, cm=-0.05, *, viscous=True, write=True, failure=False, warning=False, filename="polar.txt"):
    return (
        (".OPERva c>\n" if viscous else ".OPERia c>\n")
        + ("MRCHDU: Convergence failed at 74 side 2\n" if warning else "")
        + f" a = {alpha} CL = {cl}\n Cm = {cm} CD = {cd} => CDf = 0.005\n"
        + ("VISCAL: Convergence failed\n" if failure else "")
        + (f"Point added to stored polar 1\nPoint written to save file {filename}\n" if write else "")
        + ".OPERva c>\n.OPERv c>\nXFOIL c>\n"
    )


def _validate(settings, polar=None, log=None, **kwargs):
    result = validate_polar_text(
        _polar() if polar is None else polar, _log() if log is None else log,
        kwargs.pop("stderr_text", ""), settings=settings,
        process_status=kwargs.pop("process_status", "completed"),
        returncode=kwargs.pop("returncode", 0), **kwargs,
    )
    json.dumps(result.to_dict(), allow_nan=False)
    if result.status == "failed":
        assert result.aerodynamics is None and not result.usable_for_training
    return result


@pytest.mark.parametrize("padding", [0, 2, 30])
def test_header_positions_are_not_fixed(settings, padding):
    result = _validate(settings, _polar(padding=padding))
    assert result.usable_for_training and result.aerodynamics.cl_error == 0
    assert result.aerodynamics.cdp is None and result.aerodynamics.top_xtr is None
    assert result.aerodynamics.bot_xtr is None


def test_reordered_columns_extra_column_and_scalar_metadata(settings):
    text = "alpha = 2.0\n" + _polar(("-0.05 0.006 0.5 2.0 -1.1",), "CM CD CL alpha Cpmin")
    result = _validate(settings, text)
    assert result.usable_for_training
    assert result.aerodynamics.extra_columns == (("cpmin", -1.1),)


def test_fortran_d_exponents_and_signed_decimals(settings):
    result = _validate(settings, _polar(("+2.0D+00 5.0d-1 6.0D-3 -.5E-1",)))
    assert result.usable_for_training and result.aerodynamics.cd == 0.006


def test_target_row_is_not_blindly_last(settings):
    result = _validate(settings, _polar(("2.0 .5 .006 -.05", "3.0 .6 .008 -.06")))
    assert result.usable_for_training and result.aerodynamics.cl == 0.5
    assert len(result.table.rows) == 2


@pytest.mark.parametrize("text,reason", [
    ("", "empty_polar"), ("  \n", "empty_polar"),
    ("no table header", "missing_column_header"),
    (_polar(header="alpha CL CD"), "missing_required_columns"),
    (_polar(header="alpha CL CD CM CL"), "duplicate_columns"),
    (_polar() + _polar(), "multiple_polar_tables"),
    (_polar() + "# alpha CL CD CM\n2 .5 .006 -.05\n", "multiple_polar_tables"),
    (_polar(()), "target_row_missing"),
    (_polar(("1 .2 .006 -.05",)), "target_row_missing"),
    (_polar(("2 .5 .006 -.05", "2 .5 .006 -.05")), "ambiguous_target_rows"),
    (_polar(("2 .5 0 -.05",)), "nonpositive_cd"),
    (_polar(("2 .5 -.001 -.05",)), "nonpositive_cd"),
    (_polar(("2 .5 .006",)), "damaged_target_or_unidentified_row"),
    (_polar(("2 .5 .006 -.05", "damaged unknown row")), "damaged_target_or_unidentified_row"),
])
def test_invalid_tables_are_rejected(settings, text, reason):
    result = _validate(settings, text)
    assert result.status == "failed" and result.failure_reason == reason


@pytest.mark.parametrize("token", ["NaN", "nan", "inf", "-Infinity", "*****", "1e999", "bad"])
@pytest.mark.parametrize("column", [0, 1, 2, 3])
def test_nonfinite_or_damaged_required_values_never_escape(settings, token, column):
    row = ["2.0", "0.5", "0.006", "-0.05"]
    row[column] = token
    result = _validate(settings, _polar((" ".join(row),)))
    assert result.status == "failed"
    assert result.failure_reason == "damaged_target_or_unidentified_row"


def test_known_damaged_nontarget_row_is_diagnostic_only(settings):
    result = _validate(settings, _polar(("1.0 .2 **** -.05", "2.0 .5 .006 -.05")))
    assert result.usable_for_training
    assert any("ignored_damaged_nontarget_row" in warning for warning in result.warnings)
    assert result.table.issues[0].cl == 0.2


@pytest.mark.parametrize("cl,expected", [(0.499, "success"), (0.501, "success"), (0.5011, "failed")])
def test_cl_tolerance_is_inclusive(settings, cl, expected):
    result = _validate(settings, _polar((f"2 {cl} .006 -.05",)), _log(cl=cl))
    assert result.status == expected


@pytest.mark.parametrize("status,returncode,reason", [
    ("timeout", -1, "process_timeout"), ("crashed", 7, "process_not_completed"),
    ("running", None, "process_not_completed"), ("solver_identity_mismatch", 0, "process_not_completed"),
    ("completed", 7, "process_returncode_invalid"), ("completed", None, "process_returncode_invalid"),
    ("completed", False, "process_returncode_invalid"),
])
def test_process_failure_overrides_valid_old_output(settings, status, returncode, reason):
    result = _validate(settings, process_status=status, returncode=returncode)
    assert result.status == "failed" and result.failure_reason == reason


@pytest.mark.parametrize("log,reason", [
    ("", "target_state_unconfirmed"),
    (_log(write=False), "target_polar_write_unconfirmed"),
    (_log(filename="other.txt"), "target_polar_write_unconfirmed"),
    (_log(viscous=False), "target_viscous_mode_unconfirmed"),
    (_log().replace(".OPERva c>\n", ""), "target_viscous_mode_unconfirmed"),
    (_log(failure=True, write=False), "target_convergence_failed"),
    (_log(failure=True, write=True), "target_convergence_failed"),
    (_log(cd=0.008), "polar_log_mismatch"),
    (_log(alpha=5), "polar_log_mismatch"),
    (_log(cm=-0.1), "polar_log_mismatch"),
    (_log(cl=0.4), "polar_log_mismatch"),
    (_log(alpha="NaN"), "target_log_numeric_error"),
    (_log() + "forrtl: severe (157): Program Exception\n", "fatal_solver_log"),
])
def test_target_evidence_is_required_and_bound_to_polar(settings, log, reason):
    result = _validate(settings, log=log)
    assert result.status == "failed" and result.failure_reason == reason


def test_early_continuation_failure_and_iteration_warnings_allow_final_success(settings):
    early = _log(alpha=1, cl=0.2, failure=True, write=False)
    result = _validate(settings, log=early + _log(warning=True))
    assert result.usable_for_training and result.evidence.iteration_warnings == 1
    assert "intermediate_boundary_layer_iteration_warnings" in result.warnings


def test_later_failed_target_cannot_reuse_earlier_saved_point(settings):
    result = _validate(settings, log=_log() + _log(failure=True, write=False))
    assert result.failure_reason == "target_convergence_failed"


def test_new_alpha_never_reuses_prior_iteration_drag(settings):
    log = ".OPERva c>\n a = 2 CL = .5\n Cm = -.05 CD = .006\n a = 2 CL = .5\nPoint written to save file polar.txt\n"
    result = _validate(settings, log=log)
    assert result.failure_reason == "target_state_unconfirmed"


def test_save_marker_must_follow_the_latest_solution(settings):
    log = ".OPERva c>\n a = 1 CL = .2\n Cm = -.05 CD = .007\nPoint written to save file polar.txt\n a = 2 CL = .5\n Cm = -.05 CD = .006\n"
    result = _validate(settings, log=log)
    assert result.failure_reason == "target_polar_write_unconfirmed"


@pytest.mark.parametrize("stderr,reason", [
    ("Segmentation fault", "fatal_solver_log"),
    ("VISCAL: Convergence failed", "unscoped_stderr_convergence_failure"),
    ("benign diagnostic\n", None),
])
def test_stderr_is_checked_without_rejecting_every_warning(settings, stderr, reason):
    result = _validate(settings, stderr_text=stderr)
    assert result.failure_reason == reason


@pytest.mark.parametrize("alpha,cd,status", [(9, .006, "needs_review"), (-9, .006, "needs_review"), (2, .05001, "needs_review"), (8, .05, "success")])
def test_review_flags_are_not_training_labels(settings, alpha, cd, status):
    result = _validate(settings, _polar((f"{alpha} .5 {cd} -.05",)), _log(alpha=alpha, cd=cd))
    assert result.status == status and result.aerodynamics is not None
    assert result.usable_for_training == (status == "success")
    assert bool(result.review_flags) == (status == "needs_review")


def test_optional_columns_and_transition_checks(settings):
    header = "alpha CL CD CM CDp Top_Xtr Bot_Xtr"
    good = _validate(settings, _polar(("2 .5 .006 -.05 .001 .5 1",), header))
    assert good.usable_for_training and good.aerodynamics.top_xtr == .5
    bad = _validate(settings, _polar(("2 .5 .006 -.05 .001 1.1 1",), header))
    assert bad.failure_reason == "invalid_transition_location"
    invalid = _validate(settings, _polar(("2 .5 .006 -.05 NaN .5 1",), header))
    assert invalid.status == "failed"


@pytest.mark.parametrize("field,value", [("target_cl", float("nan")), ("cl_tolerance", -1), ("cd_review_threshold", 0), ("abs_alpha_deg_review_threshold", True)])
def test_invalid_settings_fail(settings, field, value):
    with pytest.raises(PolarConfigurationError):
        _validate(replace(settings, **{field: value}))


def test_configuration_cannot_disable_acceptance_checks(config):
    config["xfoil"]["require_target_convergence"] = False
    with pytest.raises(PolarConfigurationError):
        PolarSettings.from_config(config)


@pytest.mark.parametrize("name,status", [("xfoil699_baseline", "success"), ("xfoil699_continuation", "success"), ("xfoil699_unconverged", "failed")])
def test_real_archived_fixtures(settings, name, status):
    directory = FIXTURES / name
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    for filename, expected in source["files_sha256"].items():
        assert sha256((directory / filename).read_bytes()).hexdigest() == expected
    result = parse_polar_file(directory / "polar.txt", directory / "stdout.log", directory / "stderr.log", settings=settings, process_status="completed", returncode=0)
    assert result.status == status
    json.dumps(result.to_dict(), allow_nan=False)
    if status == "success":
        assert result.aerodynamics.cl == .5 and result.aerodynamics.cd == .00599
        if name == "xfoil699_continuation":
            assert [row.cl for row in result.table.rows] == [.2, .3, .4, .5]
    else:
        assert result.failure_reason == "target_convergence_failed"
        assert result.aerodynamics is None


def _copy_attempt(lab):
    source = FIXTURES / "xfoil699_baseline"
    directory = lab / "attempt_1"
    shutil.copytree(source, directory)
    paths = {name: directory / filename for name, filename in (
        ("polar_path", "polar.txt"), ("stdout_path", "stdout.log"), ("stderr_path", "stderr.log"),
        ("input_path", "input.txt"), ("airfoil_path", "airfoil.dat"),
    )}
    return SimpleNamespace(
        status="completed", returncode=0, failure_reason=None, launched=True,
        remaining_pids=(), cleanup_errors=(), cl_sequence=(.5,), attempt_dir=directory,
        session_sha256=sha256(paths["input_path"].read_bytes()).hexdigest(),
        geometry_file_sha256=sha256(paths["airfoil_path"].read_bytes()).hexdigest(), **paths,
    )


def test_runner_adapter_binds_current_attempt_inputs(config, lab):
    attempt = _copy_attempt(lab)
    assert parse_attempt(attempt, config).usable_for_training
    attempt.input_path.write_text("modified session")
    assert parse_attempt(attempt, config).failure_reason == "attempt_input_hash_mismatch"


def test_runner_adapter_rejects_mixed_paths_and_target(config, lab):
    attempt = _copy_attempt(lab)
    attempt.polar_path = FIXTURES / "xfoil699_baseline/polar.txt"
    assert parse_attempt(attempt, config).failure_reason == "attempt_artifact_path_mismatch"
    attempt = _copy_attempt(lab / "other")
    attempt.cl_sequence = (.4,)
    assert parse_attempt(attempt, config).failure_reason == "attempt_target_mismatch"


def test_missing_files_and_failed_process_remain_distinct(settings, lab):
    missing = lab / "missing.txt"
    result = parse_polar_file(missing, lab / "stdout.log", lab / "stderr.log", settings=settings, process_status="completed", returncode=0)
    assert result.failure_reason == "polar_file_unreadable"
    timed_out = parse_polar_file(missing, missing, missing, settings=settings, process_status="timeout", returncode=-1)
    assert timed_out.failure_reason == "process_timeout"
