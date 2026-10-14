"""Seeded unit-square sampling; distances use fixed bounds, never fitted scales.

SciPy 1.14.1 qmc.LatinHypercube/Sobol supply the initial designs. Replacement
points are maximin candidates and do not retain the strict Latin property.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc


def _points(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, 2))
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("points must be finite Nx2 coordinates")
    return array


def normalize(points, bounds) -> np.ndarray:
    limits = _points(bounds)
    if limits.shape != (2, 2) or np.any(limits[:, 1] <= limits[:, 0]):
        raise ValueError("two increasing variable bounds are required")
    array = _points(points)
    if np.any(array < limits[:, 0]) or np.any(array > limits[:, 1]):
        raise ValueError("design outside fixed bounds")
    return (array - limits[:, 0]) / (limits[:, 1] - limits[:, 0])


def denormalize(points, bounds) -> np.ndarray:
    limits = _points(bounds)
    if limits.shape != (2, 2) or np.any(limits[:, 1] <= limits[:, 0]):
        raise ValueError("two increasing variable bounds are required")
    array = _points(points)
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError("unit coordinates outside [0,1]")
    return limits[:, 0] + array * (limits[:, 1] - limits[:, 0])


def lhs(count: int, seed: int) -> np.ndarray:
    if type(count) is not int or count <= 0:
        raise ValueError("positive integer sample count required")
    return qmc.LatinHypercube(d=2, scramble=True, seed=seed).random(count)


def sobol(count: int, seed: int, *, scramble: bool = True) -> np.ndarray:
    if type(count) is not int or count <= 0 or count & (count - 1):
        raise ValueError("Sobol count must be a positive power of two")
    return qmc.Sobol(d=2, scramble=scramble, seed=seed).random_base2(count.bit_length() - 1)


def nearest_distances(points, excluded) -> np.ndarray:
    candidates, occupied = _points(points), _points(excluded)
    return cdist(candidates, occupied).min(axis=1) if len(occupied) else np.full(len(candidates), np.inf)


def distance_filter(points, excluded, minimum: float) -> np.ndarray:
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError("positive finite distance threshold required")
    accepted, occupied = [], _points(excluded).tolist()
    for point in _points(points):
        if nearest_distances([point], occupied)[0] >= minimum:
            accepted.append(point.tolist())
            occupied.append(point.tolist())
    return _points(accepted)


def maximin(excluded, rng: np.random.Generator, minimum: float, *, pool_size: int = 4096) -> list[float]:
    if pool_size <= 0 or not np.isfinite(minimum) or minimum <= 0:
        raise ValueError("positive pool size and distance threshold required")
    pool = rng.random((pool_size, 2))
    distance = nearest_distances(pool, excluded)
    index = int(np.argmax(distance))
    if distance[index] < minimum:
        raise RuntimeError("no replacement satisfies minimum distance")
    return pool[index].tolist()


def initial_plan(config, *, excluded=None) -> dict[str, Any]:
    """Reserve both coordinate sets before the first launch; no labels are stored."""
    settings, bounds = config["doe"], config["design"]["bounds"]
    if settings["method"] != "latin_hypercube" or settings["replacement_method"] != "maximin":
        raise ValueError("M2 supports latin_hypercube with maximin replacement")
    if not all(settings[name] for name in ("include_baseline", "reserve_holdout_coordinates",
        "seal_holdout_labels_until_final_verification", "reuse_accepted_preflight_baseline")):
        raise ValueError("M2 requires baseline reuse and sealed reserved holdout")
    if settings["initial_points"] <= 0 or settings["holdout_points"] <= 0:
        raise ValueError("positive training and holdout quotas required")
    if not np.isfinite(settings["min_normalized_distance"]) or settings["min_normalized_distance"] <= 0:
        raise ValueError("positive finite normalized distance required")
    rng = np.random.default_rng(settings["seed"])
    occupied = normalize([config["design"]["baseline"]], bounds).tolist()
    for point in _points([] if excluded is None else excluded):
        if np.any(point < 0) or np.any(point > 1):
            raise ValueError("excluded coordinates outside the unit square")
        if nearest_distances([point], occupied)[0] > 1e-14:
            occupied.append(point.tolist())
    raw = {"holdout": lhs(settings["holdout_points"], settings["holdout_seed"]),
           "train": lhs(settings["initial_points"], settings["seed"])}
    slots, strict = [], {"train": True, "holdout": True}
    # Reserve holdout first; both independent seeds remain fixed.
    for split in ("holdout", "train"):
        for index, proposed in enumerate(raw[split]):
            point = proposed.tolist()
            method = "lhs"
            if nearest_distances([point], occupied)[0] < settings["min_normalized_distance"]:
                point = maximin(occupied, rng, settings["min_normalized_distance"])
                method, strict[split] = "maximin", False
            occupied.append(point)
            slots.append({"slot_id": f"{split}_{index + 1:02d}", "split": split,
                          "u": point, "design": denormalize([point], bounds)[0].tolist(),
                          "method": method, "generation": 0, "eval_id": None, "status": "planned", "history": []})
    return {"slots": slots, "excluded": occupied, "rng_state": deepcopy(rng.bit_generator.state),
            "strict_lhs": strict, "seeds": {"train": settings["seed"], "holdout": settings["holdout_seed"]},
            "holdout_labels_sealed": True}


def replace_slot(plan: dict, slot: dict, config) -> None:
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(plan["rng_state"])
    point = maximin(plan["excluded"], rng, config["doe"]["min_normalized_distance"])
    slot["history"].append({key: deepcopy(slot[key]) for key in ("u", "design", "method", "eval_id", "status", "generation")})
    slot.update(u=point, design=denormalize([point], config["design"]["bounds"])[0].tolist(),
                method="maximin", generation=slot["generation"] + 1, eval_id=None, status="planned")
    plan["excluded"].append(point)
    plan["rng_state"] = deepcopy(rng.bit_generator.state)
    plan["strict_lhs"][slot["split"]] = False
