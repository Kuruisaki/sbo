"""Minimization EI on original CD, Sobol/boundary pool and serial DE comparison.

This module cannot launch XFOIL. Proposals are unassessed coordinates, not labels.
Exclusions contain coordinates only; sealed holdout outcomes never enter a score.
"""
from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from .geometry import GeometryError, generate_from_config
from .sampling import denormalize, nearest_distances, sobol
from .surrogate import digest

ACQUISITION_VERSION = "ei-sobol-de-v2"
# Explicit finite search settings, recorded in every proposal (not aerodynamic fidelity).
DE_OPTIONS = {"maxiter": 128, "popsize": 15, "tol": 1e-6, "atol": 1e-12,
              "polish": True, "workers": 1, "updating": "immediate"}


def expected_improvement(mean, std, f_best: float, xi: float, *, sigma_zero_threshold: float = 1e-12) -> np.ndarray:
    """EI for minimizing CD; at tiny sigma use positive part of f_best - mu - xi."""
    mean, std = np.broadcast_arrays(np.asarray(mean, dtype=float), np.asarray(std, dtype=float))
    if (not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0)
            or not np.isfinite([f_best, xi, sigma_zero_threshold]).all() or xi < 0 or sigma_zero_threshold <= 0):
        raise ValueError("EI requires finite predictions, nonnegative std/xi and positive sigma threshold")
    improvement = f_best - mean - xi
    active = std > sigma_zero_threshold
    z = np.zeros_like(improvement)
    np.divide(improvement, std, out=z, where=active)
    with np.errstate(over="ignore", under="ignore"):
        smooth = improvement * ndtr(z) + std * np.exp(-.5 * z**2) / np.sqrt(2 * np.pi)
    result = np.maximum(np.where(active, smooth, improvement), 0)
    if not np.isfinite(result).all():
        raise ValueError("EI overflow/nonfinite value")
    return result


def validate_acquisition_settings(config):
    a = config["acquisition"]
    fixed = {"method": "expected_improvement", "optimizer": "differential_evolution",
             "sobol_seed_policy": "iteration_seed", "compare_optimizer_with_pool": True,
             "exploration_method": "maximin_distance_to_training", "sigma_zero_policy": "positive_part_of_improvement"}
    if any(a[name] != value for name, value in fixed.items()):
        raise ValueError("unsupported acquisition policy")
    if set(a["exclude"]) != {"successful", "failed", "reserved_holdout", "running"}:
        raise ValueError("acquisition must exclude all occupied coordinate categories")
    if set(a["fallback_on"]) != {"invalid_ei", "degenerate_ei", "duplicate_proposal"}:
        raise ValueError("unsupported acquisition fallback policy")
    if (a["xi"] < 0 or a["sigma_zero_threshold"] <= 0 or a["min_normalized_distance"] <= 0
            or a["exploration_every_successful_points"] <= 0 or a["boundary_points_per_edge"] < 2
            or a["seed_base"] < 0 or a["iteration_start"] < 1):
        raise ValueError("invalid acquisition numerical settings")
    count = a["sobol_pool_size"]
    if count <= 0 or count & (count - 1):
        raise ValueError("Sobol pool size must be a positive power of two")


def candidate_pool(config, iteration: int) -> np.ndarray:
    validate_acquisition_settings(config)
    a = config["acquisition"]
    if type(iteration) is not int or iteration < a["iteration_start"]:
        raise ValueError("invalid acquisition iteration")
    arrays = [sobol(a["sobol_pool_size"], a["seed_base"] + iteration, scramble=a["sobol_scramble"])]
    nodes = np.linspace(0, 1, a["boundary_points_per_edge"])
    if not a["include_corners"]:
        nodes = nodes[1:-1]
    if a["supplement_boundaries"] and len(nodes):
        arrays.extend([np.c_[nodes, np.zeros_like(nodes)], np.c_[nodes, np.ones_like(nodes)],
                       np.c_[np.zeros_like(nodes), nodes], np.c_[np.ones_like(nodes), nodes]])
    if a["include_corners"]:
        arrays.append(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    return np.unique(np.vstack(arrays), axis=0)


def _coordinates(points) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.empty((0, 2))
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise ValueError("exclusion/training coordinates must lie inside the unit square")
    return array


def propose(model, config, excluded, *, f_best: float, iteration: int = 1,
            successful_sequential_points: int = 0, poor_loo: bool = False) -> dict[str, Any]:
    validate_acquisition_settings(config)
    if type(successful_sequential_points) is not int or successful_sequential_points < 0:
        raise ValueError("successful sequential count must be a nonnegative integer")
    if not np.isfinite(f_best):
        raise ValueError("f_best must be finite and in original CD units")
    a = config["acquisition"]
    training = _coordinates(model.x_train)
    if not len(training):
        raise ValueError("acquisition requires training coordinates")
    occupied = _coordinates(excluded)
    # The DB and vectorized normalization can differ by a final floating-point
    # bit. Add missing training coordinates without counting those twice.
    for point in training:
        if nearest_distances([point], occupied)[0] > 1e-14:
            occupied = np.vstack([occupied, point])
    occupied = np.unique(occupied, axis=0)
    raw_pool = candidate_pool(config, iteration)
    distances = nearest_distances(raw_pool, occupied)
    pool = raw_pool[distances >= a["min_normalized_distance"]]
    if not len(pool):
        raise RuntimeError("no pool point satisfies normalized exclusion distance")
    seed = a["seed_base"] + iteration
    prediction_error, pool_ei = None, None
    de_record = {"attempted": False, "seed": seed, "options": DE_OPTIONS.copy(), "warnings": []}
    diagnostics = []
    with threadpool_limits(limits=1):
        try:
            mean, std = model.predict_normalized(pool)
            pool_ei = expected_improvement(mean, std, f_best, a["xi"], sigma_zero_threshold=a["sigma_zero_threshold"])
            if np.asarray(pool_ei).shape != (len(pool),):
                raise ValueError("predictor returned an incorrect pool shape")
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            prediction_error = f"{type(exc).__name__}: {exc}"
        max_pool_ei = float(np.max(pool_ei)) if pool_ei is not None else None
        reason = None
        if poor_loo and a["exploration_on_poor_loo"]:
            reason = "poor_loo_exploration"
        elif (successful_sequential_points + 1) % a["exploration_every_successful_points"] == 0:
            reason = "scheduled_exploration"
        elif pool_ei is None:
            reason = "invalid_ei"
        elif max_pool_ei <= 0:
            reason = "degenerate_ei"

        def geometry(point):
            if np.asarray(point).shape != (2,) or not np.isfinite(point).all() or np.any(point < 0) or np.any(point > 1):
                return None
            if nearest_distances([point], occupied)[0] < a["min_normalized_distance"]:
                return None
            try:
                return generate_from_config(config, denormalize([point], config["design"]["bounds"])[0])
            except GeometryError:
                return None

        winner, winner_geometry, source = None, None, None
        if reason is None:
            # Validate geometry of proposed winners, rather than generating thousands
            # of expensive contours during GP-only objective evaluations.
            for index in np.argsort(-pool_ei, kind="stable"):
                checked = geometry(pool[index])
                if checked is not None:
                    winner, winner_geometry, source = pool[index], checked, "pool"
                    break
            if winner is None:
                raise RuntimeError("no geometrically valid pool proposal")
            winner_ei = float(pool_ei[index])

            def objective(point):
                if nearest_distances([point], occupied)[0] < a["min_normalized_distance"]:
                    return np.inf
                try:
                    mu, sigma = model.predict_normalized(np.asarray(point)[None, :])
                    ei = expected_improvement(mu, sigma, f_best, a["xi"], sigma_zero_threshold=a["sigma_zero_threshold"])
                    return -float(ei[0])
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    return np.inf

            de_record["attempted"] = True
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    result = differential_evolution(objective, [(0., 1.), (0., 1.)], seed=seed, **DE_OPTIONS)
                    checked = geometry(result.x)
                    objective_value = objective(result.x)
                    de_record.update(converged=bool(result.success), message=str(result.message),
                                     function_evaluations=int(result.nfev), iterations=int(result.nit),
                                     u=np.asarray(result.x).tolist(), valid=bool(checked is not None and np.isfinite(objective_value)),
                                     ei=float(-objective_value) if np.isfinite(objective_value) else None)
                    if de_record["valid"] and -objective_value > winner_ei:
                        winner, winner_geometry, source = result.x, checked, "differential_evolution"
                    elif not de_record["valid"]:
                        diagnostics.append("de_candidate_discarded:invalid_geometry_or_exclusion")
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    de_record.update(error=f"{type(exc).__name__}: {exc}", valid=False)
                    diagnostics.append("de_failed:using_valid_pool")
                de_record["warnings"] = [{"category": item.category.__name__, "message": str(item.message)} for item in captured]
            reason = "maximum_expected_improvement"
        else:
            train_distances = nearest_distances(pool, training)
            for index in np.argsort(-train_distances, kind="stable"):
                checked = geometry(pool[index])
                if checked is not None:
                    winner, winner_geometry, source = pool[index], checked, "maximin_pool"
                    break
            if winner is None:
                raise RuntimeError("no geometrically valid maximin proposal")
        mu_cd, sigma_cd, ei_value = None, None, None
        try:
            mean, std = model.predict_normalized(np.asarray(winner)[None, :])
            value = expected_improvement(mean, std, f_best, a["xi"], sigma_zero_threshold=a["sigma_zero_threshold"])
            mu_cd, sigma_cd, ei_value = float(mean[0]), float(std[0]), float(value[0])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            prediction_error = f"{type(exc).__name__}: {exc}"
    winner = np.asarray(winner, dtype=float)
    return {"status": "proposed", "assessed": False, "acquisition_version": ACQUISITION_VERSION,
            "iteration": iteration, "seed": seed, "successful_sequential_points_before": successful_sequential_points,
            "u": np.asarray(winner).tolist(), "design": [winner_geometry.m, winner_geometry.p],
            "m": winner_geometry.m, "p": winner_geometry.p, "t": winner_geometry.t,
            "mu_cd": mu_cd, "sigma_cd": sigma_cd, "ei": ei_value, "f_best_cd": float(f_best), "xi_cd": a["xi"],
            "reason": reason, "source": source, "poor_loo": bool(poor_loo), "boundary_hit": bool(np.any((winner == 0) | (winner == 1))),
            "min_excluded_distance": float(nearest_distances([winner], occupied)[0]),
            "min_training_distance": float(nearest_distances([winner], training)[0]),
            "minimum_distance_threshold": a["min_normalized_distance"], "max_pool_ei": max_pool_ei,
            "pool_total_points": len(raw_pool), "pool_valid_distance_points": len(pool), "pool_hash": digest(raw_pool.tolist()),
            "exclusion_coordinates_hash": digest(occupied.tolist()), "exclusion_count": len(occupied),
            "de": de_record, "prediction_error": prediction_error, "diagnostics": diagnostics,
            "geometry_hash": winner_geometry.geometry_hash, "geometry_file_sha256": winner_geometry.file_sha256,
            "holdout_labels_used": False, "prediction_scale": "original_cd", "new_solver_launches": 0}
