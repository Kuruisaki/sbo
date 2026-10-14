"""Geometry regressions; run with src on PYTHONPATH. No network or XFOIL required."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import yaml

from sbo.geometry import (
    GeometryError, check_geometry, generate_airfoil, generate_from_config,
    naca4_surfaces, write_airfoil,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "configs/naca2412.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def export_dir():
    # Use inherited Windows permissions; pytest's mode=0o700 tmpdir can prevent
    # access by a managed/sandboxed Windows process even inside the workspace.
    directory = ROOT / "runs" / f"geometry-export-{uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        assert directory.resolve().parent == (ROOT / "runs").resolve()
        (directory / "airfoil.dat").unlink(missing_ok=True)
        directory.rmdir()


@pytest.mark.parametrize("design", [None, (0.01, 0.25), (0.01, 0.60), (0.04, 0.25), (0.04, 0.60)])
def test_point_count_order_and_finite_te(config, design):
    result = generate_from_config(config, design)
    xy = result.coordinates
    assert xy.shape == (321, 2)
    assert np.isfinite(xy).all()
    np.testing.assert_array_equal(xy[160], [0.0, 0.0])
    assert np.count_nonzero(np.all(xy == [0.0, 0.0], axis=1)) == 1
    assert xy[0, 1] > xy[-1, 1]
    # Analytic finite TE gap for t=0.12, measured along the camber-line normal.
    assert np.linalg.norm(xy[0] - xy[-1]) == pytest.approx(0.00252, abs=1e-15)
    assert result.checks.valid and result.checks.signed_area > 0
    assert not result.checks.intersections


def test_no_clipping_or_false_monotonicity_rejection(config):
    result = generate_from_config(config)
    assert result.coordinates[:, 0].min() < 0.0
    upper_le_to_te = result.coordinates[:161][::-1]
    assert np.any(np.diff(upper_le_to_te[:, 0]) < 0.0)
    assert result.checks.valid


def test_analytic_camber_and_normal_offset():
    m, p, t = 0.02, 0.40, 0.12
    s = np.array([0.0, 0.2, p, 0.7, 1.0])
    upper, lower = naca4_surfaces(s, m, p, t, trailing_edge_coefficient=-0.1015)
    # Independent scalar camber probes rather than a second vectorized implementation.
    center = (upper + lower) / 2.0
    np.testing.assert_allclose(center[:, 0], s, atol=1e-15)
    np.testing.assert_allclose(center[:, 1], [0.0, 0.015, 0.02, 0.015, 0.0], atol=1e-15)
    assert upper[2, 0] == lower[2, 0] == p  # Camber slope is zero at its maximum.
    normal = upper - lower
    tangents = np.column_stack((np.ones(5), [0.1, 0.05, 0.0, -1.0 / 30.0, -1.0 / 15.0]))
    np.testing.assert_allclose(np.sum(normal * tangents, axis=1), 0.0, atol=1e-15)


def test_symmetric_limit(config):
    result = generate_airfoil(0.0, 0.4, 0.12, geometry=config["geometry"])
    upper = result.coordinates[:161]
    lower = result.coordinates[160:][::-1]
    np.testing.assert_allclose(upper[:, 0], lower[:, 0], atol=1e-15)
    np.testing.assert_allclose(upper[:, 1], -lower[:, 1], atol=1e-15)


@pytest.mark.parametrize("design", [(0.020000000001, 0.4), (0.02, 0.400000000001)])
def test_continuous_parameters_and_hash_identity(config, design):
    baseline = generate_from_config(config)
    perturbed = generate_from_config(config, design)
    # Even perturbations below the display-name precision remain distinct.
    assert baseline.name == perturbed.name
    assert not np.array_equal(baseline.coordinates, perturbed.coordinates)
    assert baseline.file_sha256 != perturbed.file_sha256
    assert baseline.geometry_hash != perturbed.geometry_hash
    repeated = generate_from_config(config, design)
    assert repeated.dat_bytes == perturbed.dat_bytes
    assert repeated.geometry_hash == perturbed.geometry_hash


def test_export_roundtrip_and_immutable_coordinates(config, export_dir):
    result = generate_from_config(config)
    destination = write_airfoil(result, export_dir / "airfoil.dat")
    raw = destination.read_bytes()
    assert raw == result.dat_bytes and b"\r" not in raw
    assert sha256(raw).hexdigest() == result.file_sha256
    assert raw.decode("ascii").splitlines()[0] == result.name
    np.testing.assert_array_equal(np.loadtxt(destination, skiprows=1), result.coordinates)
    with pytest.raises(ValueError):
        result.coordinates[0, 0] = 0.0
    with pytest.raises(ValueError):
        result.coordinates.setflags(write=True)
    with pytest.raises(FileExistsError):
        write_airfoil(result, destination)
    assert write_airfoil(result, destination, overwrite=True).read_bytes() == raw


@pytest.mark.parametrize("design", [(0.009, 0.4), (0.02, 0.61), (0.02,), (True, 0.4), (float("nan"), 0.4), (0.02, 0.0)])
def test_invalid_designs_fail(config, design):
    with pytest.raises(GeometryError):
        generate_from_config(config, design)


@pytest.mark.parametrize("key,value", [
    ("trailing_edge_coefficient", -0.1036), ("points_per_side", 161.5),
    ("points_per_side", True), ("points_per_side", 2),
    ("normalize_coordinates", True), ("clip_x_coordinates", True),
    ("generator_version", "unknown"), ("name_template", "NACA\n2412"),
    ("name_template", "中文翼型"), ("name_template", "NACA_{unknown}"),
    ("checks", ["finite_coordinates"]), ("extra_option", 1),
])
def test_invalid_or_unsupported_settings_fail(config, key, value):
    config["geometry"][key] = value
    with pytest.raises(GeometryError):
        generate_from_config(config)


def test_missing_settings_fail(config):
    del config["geometry"]["spacing"]
    with pytest.raises(GeometryError, match="missing"):
        generate_from_config(config)


@pytest.mark.parametrize("points,expected", [
    ([[0, 0], [1, 1], [0, 1], [1, 0]], "self_intersection"),
    ([[0, 0], [1, 0], [1, 0], [0, 1]], "adjacent_duplicate_points"),
    ([[0, 0], [1, 0], [0.5, 0], [1, 1], [0, 1]], "adjacent_segment_overlap"),
    ([[0, 0], [2, 0], [2, 2], [1, 0], [0, 2]], "self_intersection"),
    ([[0, 0], [1, 0], [1, float("nan")], [0, 1]], "nonfinite_coordinates"),
    ([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], "adjacent_duplicate_points"),
])
def test_invalid_contours_are_reported(points, expected):
    checks = check_geometry(points)
    assert not checks.valid
    assert expected in checks.errors


def test_nearby_separate_edges_are_not_intersections():
    # A thin but simple polygon exercises near-collinear disjoint segments.
    checks = check_geometry([[0, 0], [1, 0], [1, 1e-8], [0, 1e-8]])
    assert checks.valid and not checks.intersections


def test_implicit_closing_segment_is_checked():
    checks = check_geometry([[0, 0], [2, 0], [2, 2], [0, 2], [1, 3]])
    assert not checks.valid
    assert (2, 4) in checks.intersections


def test_seeded_interior_designs(config):
    bounds = np.asarray(config["design"]["bounds"])
    rng = np.random.default_rng(42)
    for design in rng.uniform(bounds[:, 0], bounds[:, 1], size=(12, 2)):
        assert generate_from_config(config, design).checks.valid


def test_uiuc_independent_reference(config):
    source = ROOT / "data/reference/naca2412-uiuc.dat"
    metadata = json.loads(source.with_suffix(".source.json").read_text(encoding="utf-8"))
    assert sha256(source.read_bytes()).hexdigest() == metadata["sha256"]
    reference = np.loadtxt(source, skiprows=1)
    assert check_geometry(reference).valid
    generated = generate_from_config(config).coordinates
    ref_le = int(np.argmin(reference[:, 0]))
    for measured, analytic in (
        (reference[:ref_le], generated[:161]), (reference[ref_le + 1:], generated[160:]),
    ):
        order = np.argsort(analytic[:, 0])
        prediction = np.interp(measured[:, 0], analytic[order, 0], analytic[order, 1])
        # UIUC ordinates have four decimals and a different sampling convention.
        # Compare y at matching surface x, rather than matching point indices.
        assert np.max(np.abs(prediction - measured[:, 1])) <= 0.0002


def test_variable_order_and_fixed_thickness(config):
    swapped = deepcopy(config)
    swapped["design"]["variables"] = ["p", "m"]
    with pytest.raises(GeometryError):
        generate_from_config(swapped)
    assert generate_from_config(config, [0.0237, 0.436]).t == config["design"]["t"]
