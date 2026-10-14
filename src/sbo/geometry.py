"""Continuous NACA four-digit geometry for the SBO experiment.

Inputs use unit-chord coordinates. The profile runs upper TE -> LE -> lower
TE, keeping a single LE and both finite-thickness TE endpoints. No model-number
rounding, coordinate clipping, or XFOIL re-paneling takes place here.

Formula reference: NASA TM-4741,
https://ntrs.nasa.gov/api/citations/19970008124/downloads/19970008124.pdf

Typical use (YAML loading belongs to the caller)::

    result = generate_from_config(config, design=(0.0237, 0.436))
    write_airfoil(result, attempt_dir / "airfoil.dat")

Only NumPy and the Python standard library are required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


GENERATOR_VERSION = "naca4-analytic-v1"
_CHECKS = ("finite_coordinates", "adjacent_duplicates", "self_intersection", "continuity")


class GeometryError(ValueError):
    """Invalid input, unsupported geometry settings, or invalid contour."""


def _finite_real(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise GeometryError(f"{label} must be a finite real number")
    number = float(value)
    if not np.isfinite(number):
        raise GeometryError(f"{label} must be a finite real number")
    return number


@dataclass(frozen=True)
class GeometrySettings:
    """Supported settings; construct from the YAML geometry mapping."""

    points_per_side: int
    trailing_edge_coefficient: float
    name_template: str
    generator_version: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> GeometrySettings:
        # Fixed settings define this generator version, rather than silent options.
        fixed = {
            "family": "naca4", "generator_version": GENERATOR_VERSION,
            "chord": 1.0, "spacing": "cosine", "surface_offset": "camber_normal",
            "derivative": "analytic", "point_order": "upper_te_le_lower_te",
            "merge_leading_edge": True, "merge_trailing_edge": False,
            "normalize_coordinates": False, "clip_x_coordinates": False,
            "trailing_edge_coefficient": -0.1015,
        }
        required = set(fixed) | {"points_per_side", "name_template", "checks"}
        if set(values) != required:
            raise GeometryError(
                f"geometry fields: missing={sorted(required - set(values))}, "
                f"unknown={sorted(set(values) - required)}"
            )
        for key, expected in fixed.items():
            actual = values[key]
            if isinstance(expected, bool):
                supported = actual is expected
            elif isinstance(expected, float):
                supported = _finite_real(actual, key) == expected
            else:
                supported = actual == expected
            if not supported:
                raise GeometryError(f"{key}={actual!r} is unsupported by {GENERATOR_VERSION}")
        checks = values["checks"]
        if not isinstance(checks, (list, tuple)) or sorted(checks) != sorted(_CHECKS):
            raise GeometryError(f"checks must contain exactly {list(_CHECKS)}")
        settings = cls(
            points_per_side=values["points_per_side"],
            trailing_edge_coefficient=float(values["trailing_edge_coefficient"]),
            name_template=values["name_template"],
            generator_version=values["generator_version"],
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        count = self.points_per_side
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 3:
            raise GeometryError("points_per_side must be an integer >= 3")
        if self.generator_version != GENERATOR_VERSION:
            raise GeometryError("unsupported generator_version")
        if _finite_real(self.trailing_edge_coefficient, "trailing_edge_coefficient") != -0.1015:
            raise GeometryError("this generator requires the finite-TE coefficient -0.1015")
        if not isinstance(self.name_template, str):
            raise GeometryError("name_template must be a string")


@dataclass(frozen=True)
class GeometryChecks:
    """Diagnostics use zero-based point and segment indices."""

    valid: bool
    errors: tuple[str, ...]
    adjacent_duplicates: tuple[int, ...] = ()
    intersections: tuple[tuple[int, int], ...] = ()
    signed_area: float | None = None
    trailing_edge_gap: float | None = None


@dataclass(frozen=True)
class GeometryResult:
    """Read-only coordinates, diagnostics, and reproducible XFOIL input bytes.

    file_sha256 hashes the exact bytes written to airfoil.dat. geometry_hash also
    covers full-precision parameters and generator settings; it is not a complete
    aerodynamic cache key (flow, paneling and solver identity belong to evaluator).
    """

    m: float
    p: float
    t: float
    settings: GeometrySettings
    name: str
    coordinates: NDArray[np.float64]
    checks: GeometryChecks
    dat_bytes: bytes
    file_sha256: str
    geometry_hash: str


def _parameters(m: Real, p: Real, t: Real) -> tuple[float, float, float]:
    m, p, t = (_finite_real(value, label) for value, label in ((m, "m"), (p, "p"), (t, "t")))
    if not 0.0 <= m < 1.0 or not 0.0 < p < 1.0 or not 0.0 < t < 1.0:
        raise GeometryError("require 0 <= m < 1, 0 < p < 1, and 0 < t < 1")
    return m, p, t


def naca4_surfaces(
    s: ArrayLike, m: Real, p: Real, t: Real, *, trailing_edge_coefficient: Real,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return upper/lower (n, 2) coordinates at arbitrary chord parameters s.

    The supplied s is the camber-line parameter, not the offset surface x.
    This low-level function permits analytic probes; generate_from_config also
    enforces the experiment's design bounds and fixed geometry convention.
    """
    m, p, t = _parameters(m, p, t)
    coefficient = _finite_real(trailing_edge_coefficient, "trailing_edge_coefficient")
    s = np.asarray(s, dtype=np.float64)
    if s.ndim != 1 or s.size == 0 or not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise GeometryError("s must be a nonempty finite 1D array in [0, 1]")
    forward = s <= p
    denominator = np.where(forward, p * p, (1.0 - p) ** 2)
    camber = m / denominator * np.where(forward, 2.0 * p * s - s * s, 1.0 - 2.0 * p + 2.0 * p * s - s * s)
    slope = 2.0 * m * (p - s) / denominator
    theta = np.arctan(slope)
    half_thickness = 5.0 * t * (
        0.2969 * np.sqrt(s) - 0.1260 * s - 0.3516 * s**2
        + 0.2843 * s**3 + coefficient * s**4
    )
    dx = half_thickness * np.sin(theta)
    dy = half_thickness * np.cos(theta)
    upper = np.column_stack((s - dx, camber + dy))
    lower = np.column_stack((s + dx, camber - dy))
    return upper, lower


def _cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def check_geometry(coordinates: ArrayLike) -> GeometryChecks:
    """Check a finite-TE contour, including its implicit closing TE segment.

    A continuous contour here means a simple, nondegenerate, counterclockwise
    polygon with nonzero neighboring edges. Nonadjacent touches and collinear
    overlaps count as intersections. Neighboring segments may only share their
    common endpoint. No x-monotonicity or [0, 1] clipping requirement is imposed.
    """
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        return GeometryChecks(False, ("expected_at_least_four_xy_points",))
    if not np.isfinite(points).all():
        return GeometryChecks(False, ("nonfinite_coordinates",))
    # Translate for stable area arithmetic. Tolerances track unit-chord geometry.
    q = points - points[0]
    ends = np.roll(q, -1, axis=0)
    edges = ends - q
    lengths = np.linalg.norm(edges, axis=1)
    scale = max(float(np.ptp(points, axis=0).max()), np.finfo(float).tiny)
    tolerance = 64.0 * np.finfo(float).eps * scale
    duplicates = tuple(int(i) for i in np.flatnonzero(lengths <= tolerance))
    area = float(0.5 * np.sum(_cross(q, ends)))
    gap = float(lengths[-1])
    errors: list[str] = []
    if duplicates:
        errors.append("adjacent_duplicate_points")
    if area <= tolerance * scale:
        errors.append("nonpositive_or_degenerate_signed_area")

    # Adjacent backtracking produces an overlap even without a nonadjacent crossing.
    previous = -np.roll(edges, 1, axis=0)
    overlap = (np.abs(_cross(previous, edges)) <= tolerance * (np.roll(lengths, 1) + lengths)) & (np.sum(previous * edges, axis=1) > tolerance**2)
    if overlap.any():
        errors.append("adjacent_segment_overlap")

    pairs: list[tuple[int, int]] = []
    count = len(q)
    for i in range(count - 2):
        j = np.arange(i + 2, count)
        if i == 0:
            j = j[j != count - 1]  # The closing TE segment neighbors segment 0.
        if not len(j):
            continue
        box_overlap = (
            np.maximum(np.minimum(q[i], ends[i]), np.minimum(q[j], ends[j]))
            <= np.minimum(np.maximum(q[i], ends[i]), np.maximum(q[j], ends[j])) + tolerance
        ).all(axis=1)
        j = j[box_overlap]
        if not len(j):
            continue
        o1 = _cross(edges[i], q[j] - q[i])
        o2 = _cross(edges[i], ends[j] - q[i])
        o3 = _cross(edges[j], q[i] - q[j])
        o4 = _cross(edges[j], ends[i] - q[j])
        signs = [np.where(np.abs(o) <= eps, 0, np.sign(o)) for o, eps in (
            (o1, tolerance * lengths[i]), (o2, tolerance * lengths[i]),
            (o3, tolerance * lengths[j]), (o4, tolerance * lengths[j]),
        )]
        hit = (signs[0] * signs[1] <= 0) & (signs[2] * signs[3] <= 0)
        pairs.extend((i, int(k)) for k in j[hit])
    if pairs:
        errors.append("self_intersection")
    return GeometryChecks(not errors, tuple(errors), duplicates, tuple(pairs), area, gap)


def _airfoil_name(template: str, m: float, p: float, t: float) -> str:
    try:
        name = template.format(m=m, p=p, t=t)
    except (KeyError, ValueError, IndexError, AttributeError) as exc:
        raise GeometryError("invalid name_template; supported parameters are m, p, t") from exc
    # A leading letter N avoids XFOIL's numeric/logical first-line ambiguity.
    if (not name or not name.startswith("N") or len(name) > 80
            or any(ord(c) < 32 or ord(c) > 126 for c in name)):
        raise GeometryError("airfoil name must start with N and contain 1..80 printable ASCII characters")
    return name


def generate_airfoil(
    m: Real, p: Real, t: Real, *, geometry: Mapping[str, Any] | GeometrySettings,
) -> GeometryResult:
    """Generate/check a contour using explicit geometry settings and parameters.

    This function does not enforce experiment-specific m/p bounds. Use
    generate_from_config for baseline/DOE/sequential experiment designs.
    """
    settings = geometry if isinstance(geometry, GeometrySettings) else GeometrySettings.from_mapping(geometry)
    settings.validate()
    # Accept NumPy numeric scalars without putting non-JSON scalar types in hashes.
    settings = GeometrySettings(
        int(settings.points_per_side), float(settings.trailing_edge_coefficient),
        settings.name_template, settings.generator_version,
    )
    m, p, t = _parameters(m, p, t)
    s = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, settings.points_per_side)))
    s[0], s[-1] = 0.0, 1.0
    upper, lower = naca4_surfaces(s, m, p, t, trailing_edge_coefficient=settings.trailing_edge_coefficient)
    joined = np.concatenate((upper[::-1], lower[1:]), axis=0)
    checks = check_geometry(joined)
    if not checks.valid:
        raise GeometryError(f"invalid NACA contour: {', '.join(checks.errors)}")
    # Immutable byte-backed storage prevents a changed array from invalidating hashes.
    coordinates = np.frombuffer(joined.tobytes(), dtype=np.float64).reshape(joined.shape)
    name = _airfoil_name(settings.name_template, m, p, t)
    dat_bytes = (name + "\n" + "".join(f"{x:.17g} {y:.17g}\n" for x, y in coordinates)).encode("ascii")
    file_hash = sha256(dat_bytes).hexdigest()
    identity = {
        "m": m.hex(), "p": p.hex(), "t": t.hex(),
        "settings": asdict(settings), "file_sha256": file_hash,
    }
    geometry_hash = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    return GeometryResult(m, p, t, settings, name, coordinates, checks, dat_bytes, file_hash, geometry_hash)


def generate_from_config(
    config: Mapping[str, Any], design: Sequence[Real] | None = None,
) -> GeometryResult:
    """Generate a baseline or [m, p] design from the loaded experiment YAML.

    Thickness is read only from design.t. Missing/unknown geometry settings,
    unsupported conventions and out-of-bounds designs are rejected explicitly.
    The caller owns loading and freezing the full experiment configuration.
    """
    try:
        definition = config["design"]
        geometry = config["geometry"]
        if list(definition["variables"]) != ["m", "p"]:
            raise GeometryError("design.variables must be [m, p]")
        values = definition["baseline"] if design is None else design
        if len(values) != 2:
            raise GeometryError("design must contain exactly [m, p]")
        m, p, t = _parameters(values[0], values[1], definition["t"])
        bounds = np.asarray(definition["bounds"], dtype=np.float64)
    except (KeyError, TypeError) as exc:
        raise GeometryError(f"missing or malformed design/geometry configuration: {exc}") from exc
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or (bounds[:, 0] >= bounds[:, 1]).any():
        raise GeometryError("design.bounds must be two finite increasing [lower, upper] pairs")
    if ((np.array([m, p]) < bounds[:, 0]) | (np.array([m, p]) > bounds[:, 1])).any():
        raise GeometryError("design is outside the configured m/p bounds")
    return generate_airfoil(m, p, t, geometry=geometry)


def write_airfoil(result: GeometryResult, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write exact ASCII/LF bytes; parent directory must exist (an attempt directory).

    Existing output is rejected by default, preventing accidental reuse of a dirty
    solver directory. Return the path; file_sha256 is available on the result.
    """
    destination = Path(path)
    with destination.open("wb" if overwrite else "xb") as stream:
        stream.write(result.dat_bytes)
    return destination
