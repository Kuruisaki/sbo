"""Fixed-bound, original-CD Gaussian process and independently refitted LOO.

Only validated standard training records enter TrainingData. Synthetic functions
belong in tests, never in this module's production data path. sklearn 1.5 GPR
already reverses normalize_y for BOTH predictive mean and standard deviation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any, Callable
import warnings

import joblib
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from threadpoolctl import threadpool_limits

from .config import canonical_json, file_sha256, write_json
from .sampling import normalize

SURROGATE_VERSION = "gp-matern25-v1"


class SurrogateFitError(RuntimeError):
    status = "failed"
    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


def digest(value) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrainingData:
    records: tuple[dict[str, Any], ...]
    x: np.ndarray
    y: np.ndarray
    design_ids: tuple[str, ...]
    evaluation_ids: tuple[str, ...]
    data_hash: str
    config_hash: str


def prepare_training_data(records, config) -> TrainingData:
    rows = sorted((dict(row) for row in records), key=lambda row: row["design_id"])
    if len(rows) < 2:
        raise ValueError("at least two successful standard training designs are required")
    ids, evaluation_ids, physical, labels, selected = [], [], [], [], []
    keys = ("design_id", "eval_id", "m", "p", "t", "u_m", "u_p", "split", "status", "purpose", "config_hash", "panel_count", "cd", "cl")
    for row in rows:
        if (row["split"] != "train" or row["status"] != "success" or row["purpose"] not in {"baseline", "doe", "sequential"}
                or row["config_hash"] != config.config_hash or row["panel_count"] != config["xfoil"]["standard_n"]):
            raise ValueError("nontraining, unaccepted or mixed-configuration/fidelity record")
        if row["t"] != config["design"]["t"] or not np.isfinite(row["cd"]) or row["cd"] <= 0:
            raise ValueError("invalid thickness or CD label")
        if not np.isfinite(row["cl"]) or abs(row["cl"] - config["flow"]["target_cl"]) > config["flow"]["cl_tolerance"]:
            raise ValueError("training target CL is invalid")
        ids.append(row["design_id"])
        evaluation_ids.append(row["eval_id"])
        physical.append([row["m"], row["p"]])
        labels.append(row["cd"])
        selected.append({key: row[key] for key in keys})
    if len(set(ids)) != len(ids) or len(set(evaluation_ids)) != len(ids):
        raise ValueError("duplicate training identities")
    x = normalize(physical, config["design"]["bounds"])
    declared = np.asarray([[row["u_m"], row["u_p"]] for row in selected], dtype=float)
    if not np.isfinite(declared).all() or not np.allclose(declared, x, rtol=0, atol=1e-14):
        raise ValueError("training coordinates differ from fixed-bound normalization")
    if pdist(x).min() < config["doe"]["min_normalized_distance"]:
        raise ValueError("duplicate/too-close training coordinates; do not fix this with regularization")
    y = np.asarray(labels, dtype=float)
    x.setflags(write=False)
    y.setflags(write=False)
    return TrainingData(tuple(selected), x, y, tuple(ids), tuple(evaluation_ids),
                        digest({"config_hash": config.config_hash, "records": selected}), config.config_hash)


def validate_gp_settings(config) -> None:
    settings = config["gp"]
    required = {"model": "gaussian_process_regressor", "input_normalization": "fixed_design_bounds",
                "kernel": "constant_times_matern", "matern_nu": 2.5, "ard": True, "normalize_y": True,
                "white_kernel": False, "prediction_scale": "original_cd", "loo_refit_each_fold": True,
                "record_fit_warnings": True}
    if any(settings[key] != value for key, value in required.items()):
        raise ValueError("unsupported GP policy; requires normalized-y Constant*ARD Matern(2.5) with refitted LOO")
    lengths = np.asarray(settings["length_scale_initial"], dtype=float)
    length_bounds = np.asarray(settings["length_scale_bounds"], dtype=float)
    constant_bounds = np.asarray(settings["constant_value_bounds"], dtype=float)
    if lengths.shape != (2,) or length_bounds.shape != (2,) or constant_bounds.shape != (2,):
        raise ValueError("GP requires two ARD lengths and positive bound pairs")
    if not np.isfinite(np.r_[lengths, length_bounds, constant_bounds]).all():
        raise ValueError("GP hyperparameters must be finite")
    for bounds in (length_bounds, constant_bounds):
        if not 0 < bounds[0] < bounds[1]:
            raise ValueError("GP hyperparameter bounds must be positive and increasing")
    if np.any(lengths < length_bounds[0]) or np.any(lengths > length_bounds[1]):
        raise ValueError("initial lengths outside hyperparameter bounds")
    if not constant_bounds[0] <= settings["constant_value_initial"] <= constant_bounds[1]:
        raise ValueError("initial constant outside hyperparameter bounds")
    if settings["n_restarts_optimizer"] < 0 or settings["seed"] < 0:
        raise ValueError("GP restarts and seed must be nonnegative")


def library_versions() -> dict[str, str]:
    return {"python": platform.python_version(), **{name: version(name) for name in
            ("numpy", "scipy", "scikit-learn", "joblib")}}


@dataclass
class SurrogateModel:
    estimator: GaussianProcessRegressor
    bounds: tuple[tuple[float, float], ...]
    config_hash: str
    data_hash: str
    training_ids: tuple[str, ...]
    training_eval_ids: tuple[str, ...]
    fit_record: dict[str, Any]

    @property
    def x_train(self):
        return self.estimator.X_train_.copy()

    def predict_normalized(self, points) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("prediction coordinates must be nonempty finite Nx2")
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError("prediction outside fixed unit-square bounds")
        mean, std = self.estimator.predict(values, return_std=True)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
            raise ValueError("GP produced invalid original-CD mean/std")
        return mean, std  # sklearn has already restored the original CD scale.

    def predict(self, designs) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_normalized(normalize(designs, self.bounds))


def fit_gp(data: TrainingData, config) -> SurrogateModel:
    validate_gp_settings(config)
    # Revalidate records so a caller cannot mutate an input array/record after preparation.
    checked = prepare_training_data(data.records, config)
    if checked.data_hash != data.data_hash or not np.array_equal(checked.x, data.x) or not np.array_equal(checked.y, data.y):
        raise ValueError("training snapshot changed before GP fit")
    settings = config["gp"]
    attempts = []
    for alpha in settings["alpha_regularization_schedule"]:
        if alpha > settings["alpha_regularization_max"]:
            raise ValueError("regularization schedule exceeds declared maximum")
        kernel = ConstantKernel(settings["constant_value_initial"], tuple(settings["constant_value_bounds"])) * Matern(
            length_scale=list(settings["length_scale_initial"]), length_scale_bounds=tuple(settings["length_scale_bounds"]),
            nu=settings["matern_nu"])
        estimator = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                             n_restarts_optimizer=settings["n_restarts_optimizer"], random_state=settings["seed"])
        failure = None
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            try:
                estimator.fit(checked.x, checked.y)
            except np.linalg.LinAlgError as exc:
                failure = f"{type(exc).__name__}: {exc}"
        attempt = {"alpha_regularization": float(alpha), "failure": failure,
                   "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in captured]}
        attempts.append(attempt)
        if failure is not None:
            continue
        likelihood = float(estimator.log_marginal_likelihood_value_)
        if not np.isfinite(likelihood) or not np.isfinite(estimator.kernel_.theta).all():
            raise SurrogateFitError("nonfinite fitted kernel/likelihood; regularization cannot hide this", attempts)
        record = {"surrogate_version": SURROGATE_VERSION, "seed": settings["seed"],
                  "alpha_regularization": float(alpha), "regularization_space": "normalized_y",
                  "fit_attempts": attempts, "kernel": str(estimator.kernel_),
                  "constant_value": float(estimator.kernel_.k1.constant_value),
                  "length_scales": np.asarray(estimator.kernel_.k2.length_scale).tolist(), "matern_nu": settings["matern_nu"],
                  "log_marginal_likelihood": likelihood, "training_y_mean": float(np.mean(checked.y)),
                  "training_y_std": float(np.std(checked.y)), "versions": library_versions()}
        model = SurrogateModel(estimator, tuple(tuple(pair) for pair in config["design"]["bounds"]),
                               checked.config_hash, checked.data_hash, checked.design_ids, checked.evaluation_ids, record)
        model.predict_normalized(checked.x)
        return model
    raise SurrogateFitError("Cholesky failed through the maximum configured regularization", attempts)


def loo_cv(data: TrainingData, config, baseline_cd: float, *, completed_folds=(),
           on_fold: Callable[[dict], None] | None = None) -> dict:
    """Each fold owns a fresh GP, hyperparameter search and target normalization."""
    checked = prepare_training_data(data.records, config)
    if (checked.data_hash != data.data_hash or checked.design_ids != data.design_ids
            or checked.evaluation_ids != data.evaluation_ids or not np.array_equal(checked.x, data.x)
            or not np.array_equal(checked.y, data.y)):
        raise ValueError("training snapshot changed before LOO")
    if len(data.records) < 3 or not np.isfinite(baseline_cd) or baseline_cd <= 0:
        raise ValueError("LOO requires at least three records and a valid baseline CD")
    policy = config["quality"]["loo"]
    if (policy["metric"] != "mae" or policy["threshold_combination"] != "max" or
        policy["baseline_reference_n"] != config["xfoil"]["standard_n"] or policy["on_exceedance"] != "explore_next_iteration"):
        raise ValueError("unsupported LOO quality policy")
    if policy["absolute_cd_threshold"] < 0 or policy["relative_baseline_cd_threshold"] < 0:
        raise ValueError("LOO quality thresholds cannot be negative")
    saved = {fold["held_out_design_id"]: dict(fold) for fold in completed_folds}
    if len(saved) != len(completed_folds) or set(saved) - set(data.design_ids):
        raise ValueError("LOO checkpoint has duplicate or unknown folds")
    folds = []
    for index, row in enumerate(data.records):
        training = prepare_training_data([item for j, item in enumerate(data.records) if j != index], config)
        if row["design_id"] in saved:
            fold = saved[row["design_id"]]
            if (fold["training_data_hash"] != training.data_hash or fold["training_ids"] != list(training.design_ids)
                    or fold["observed_cd"] != row["cd"] or fold["held_out_eval_id"] != row["eval_id"]):
                raise ValueError("LOO checkpoint differs from current training snapshot")
            if not np.isfinite([fold["predicted_cd"], fold["sigma_cd"], fold["residual_cd"]]).all() or fold["sigma_cd"] < 0:
                raise ValueError("invalid LOO checkpoint prediction")
            if not np.isclose(fold["residual_cd"], fold["predicted_cd"] - fold["observed_cd"], rtol=0, atol=1e-15):
                raise ValueError("LOO checkpoint residual is inconsistent")
        else:
            fitted = fit_gp(training, config)
            mean, std = fitted.predict([[row["m"], row["p"]]])
            fold = {"held_out_design_id": row["design_id"], "held_out_eval_id": row["eval_id"],
                    "training_ids": list(training.design_ids), "training_data_hash": training.data_hash,
                    "observed_cd": row["cd"], "predicted_cd": float(mean[0]), "sigma_cd": float(std[0]),
                    "residual_cd": float(mean[0] - row["cd"]), "fit": fitted.fit_record}
            if on_fold is not None:
                on_fold(fold)
        folds.append(fold)
    residuals = np.asarray([fold["residual_cd"] for fold in folds])
    threshold = max(policy["absolute_cd_threshold"], policy["relative_baseline_cd_threshold"] * baseline_cd)
    mae = float(np.mean(np.abs(residuals)))
    return {"config_hash": data.config_hash, "training_data_hash": data.data_hash,
            "folds": folds, "mae_cd": mae, "rmse_cd": float(np.sqrt(np.mean(residuals**2))),
            "max_abs_error_cd": float(np.max(np.abs(residuals))), "threshold_cd": float(threshold),
            "baseline_reference_cd": float(baseline_cd), "quality_passed": bool(mae <= threshold),
            "explore_next_iteration": bool(mae > threshold), "method": "independent_refit_per_fold",
            "residual_sign": "predicted_minus_observed", "holdout_labels_used": False}


def save_model(model: SurrogateModel, directory: Path, config) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gp.joblib"
    temporary = directory / "gp.joblib.tmp"
    joblib.dump(model, temporary, compress=3)
    temporary.replace(path)
    metadata = {"model_id": directory.name, "config_hash": model.config_hash, "training_data_hash": model.data_hash,
                "training_ids": list(model.training_ids), "training_eval_ids": list(model.training_eval_ids),
                "prediction_scale": "original_cd", "input_normalization": "fixed_design_bounds",
                "bounds": [list(pair) for pair in model.bounds], "gp_settings": config.to_dict()["gp"],
                "fit": model.fit_record, "artifact": {"file": "gp.joblib", "sha256": file_sha256(path)}}
    write_json(directory / "metadata.json", metadata)
    return metadata


def load_model(directory: Path, metadata: dict, data: TrainingData, config) -> SurrogateModel:
    """Only load a locally created joblib after its DB-bound digest is verified."""
    if (metadata["config_hash"] != config.config_hash or metadata["training_data_hash"] != data.data_hash
            or metadata["training_ids"] != list(data.design_ids) or metadata["training_eval_ids"] != list(data.evaluation_ids)
            or metadata["fit"]["versions"] != library_versions() or metadata["fit"]["surrogate_version"] != SURROGATE_VERSION):
        raise ValueError("model metadata/configuration/training/library identity mismatch")
    if metadata["artifact"]["file"] != "gp.joblib" or file_sha256(directory / "gp.joblib") != metadata["artifact"]["sha256"]:
        raise ValueError("serialized model hash mismatch")
    if json.loads((directory / "metadata.json").read_text(encoding="utf-8")) != metadata:
        raise ValueError("model metadata file differs from the ledger")
    model = joblib.load(directory / "gp.joblib")
    if (not isinstance(model, SurrogateModel) or model.config_hash != data.config_hash or model.data_hash != data.data_hash
            or model.training_ids != data.design_ids or model.training_eval_ids != data.evaluation_ids
            or tuple(model.bounds) != tuple(tuple(pair) for pair in config["design"]["bounds"])
            or not np.array_equal(model.x_train, data.x)):
        raise ValueError("serialized model differs from verified training identity")
    model.predict_normalized(data.x)
    return model
