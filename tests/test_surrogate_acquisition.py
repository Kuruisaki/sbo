"""Known functions are algorithm tests only; never exported as aerodynamic data."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from test_evaluator import lab
import sbo.acquisition as acquisition
import sbo.surrogate as surrogate
from sbo.acquisition import candidate_pool, expected_improvement, propose
from sbo.config import FrozenConfig, load_config
from sbo.sampling import denormalize, nearest_distances
from sbo.surrogate import (SurrogateFitError, fit_gp, load_model, loo_cv,
                          prepare_training_data, save_model)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    values = load_config(ROOT / "configs/naca2412.yaml").to_dict()
    values["gp"]["n_restarts_optimizer"] = 0  # Explicit cheap test configuration.
    return FrozenConfig(values)


def records(config):
    unit = np.array([[x, y] for x in (.1, .5, .9) for y in (.1, .5, .9)])
    physical = denormalize(unit, config["design"]["bounds"])
    labels = .006 + .0005 * ((unit[:, 0] - .3)**2 + 2 * (unit[:, 1] - .4)**2)
    return [{"design_id": f"algorithm_only_{i}", "eval_id": f"algorithm_eval_{i}", "m": p[0], "p": p[1],
             "t": config["design"]["t"], "u_m": unit[i, 0], "u_p": unit[i, 1], "split": "train",
             "status": "success", "purpose": "doe", "config_hash": config.config_hash,
             "panel_count": config["xfoil"]["standard_n"], "cd": labels[i], "cl": config["flow"]["target_cl"]}
            for i, p in enumerate(physical)]


@pytest.mark.parametrize("field,value", [("split", "holdout"), ("status", "needs_review"), ("purpose", "verification"),
    ("config_hash", "wrong"), ("panel_count", 240), ("cd", float('nan')), ("cd", -1), ("cl", .7), ("t", .1), ("u_m", .99)])
def test_invalid_or_leaking_training_record_is_rejected(config, field, value):
    rows = records(config)
    rows[0][field] = value
    with pytest.raises(ValueError):
        prepare_training_data(rows, config)


def test_duplicate_identity_and_coordinates_are_rejected(config):
    rows = records(config)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_training_data(rows + [rows[0]], config)
    rows[1].update({key: rows[0][key] for key in ("m", "p", "u_m", "u_p")})
    with pytest.raises(ValueError, match="too-close"):
        prepare_training_data(rows, config)


def test_fixed_bounds_original_cd_mean_and_std(config):
    rows = records(config)
    data = prepare_training_data(rows, config)
    model = fit_gp(data, config)
    shifted = deepcopy(rows)
    for row in shifted:
        row["cd"] = row["cd"] * 4 + .01
    second = fit_gp(prepare_training_data(shifted, config), config)
    physical = [[.022, .43], [.036, .57]]
    mu, std = model.predict(physical)
    other_mu, other_std = second.predict(physical)
    np.testing.assert_allclose(other_mu, mu * 4 + .01, rtol=0, atol=2e-11)
    np.testing.assert_allclose(other_std, std * 4, rtol=0, atol=2e-11)
    assert model.bounds == tuple(tuple(pair) for pair in config["design"]["bounds"])
    assert max(std) < .001 and not data.x.flags.writeable
    with pytest.raises(ValueError):
        model.predict([[.08, .4]])


def test_cholesky_only_regularization_fallback_records_warnings(config, monkeypatch):
    original = surrogate.GaussianProcessRegressor.fit
    called = []

    def injected(self, x, y):
        called.append(self.alpha)
        if len(called) == 1:
            warnings.warn("injected numerical diagnostic", UserWarning)
            raise np.linalg.LinAlgError("injected Cholesky failure")
        return original(self, x, y)

    monkeypatch.setattr(surrogate.GaussianProcessRegressor, "fit", injected)
    model = fit_gp(prepare_training_data(records(config), config), config)
    assert called == [1e-6, 1e-5] and model.fit_record["alpha_regularization"] == 1e-5
    assert model.fit_record["fit_attempts"][0]["failure"]
    assert model.fit_record["fit_attempts"][0]["warnings"][0]["message"] == "injected numerical diagnostic"


def test_failed_regularization_schedule_and_no_fallback_for_other_errors(config, monkeypatch):
    calls = []

    def fail(self, x, y):
        calls.append(self.alpha)
        raise np.linalg.LinAlgError("injected")

    monkeypatch.setattr(surrogate.GaussianProcessRegressor, "fit", fail)
    with pytest.raises(SurrogateFitError) as error:
        fit_gp(prepare_training_data(records(config), config), config)
    assert calls == list(config["gp"]["alpha_regularization_schedule"])
    assert len(error.value.attempts) == 3

    def invalid(self, x, y):
        raise ValueError("invalid input cannot be hidden by regularization")

    monkeypatch.setattr(surrogate.GaussianProcessRegressor, "fit", invalid)
    with pytest.raises(ValueError, match="cannot be hidden"):
        fit_gp(prepare_training_data(records(config), config), config)


def test_every_loo_fold_refits_without_target_normalization_leak(config, monkeypatch):
    data = prepare_training_data(records(config), config)
    fitted = []
    actual = surrogate.fit_gp

    def track(training, cfg):
        fitted.append(training)
        return actual(training, cfg)

    monkeypatch.setattr(surrogate, "fit_gp", track)
    loo = loo_cv(data, config, .006)
    assert len(fitted) == 9
    for training, fold in zip(fitted, loo["folds"]):
        assert len(training.records) == 8 and fold["held_out_design_id"] not in fold["training_ids"]
        assert fold["fit"]["training_y_mean"] == pytest.approx(training.y.mean())
        assert fold["fit"]["training_y_std"] == pytest.approx(training.y.std())
    assert any(abs(fold["fit"]["training_y_mean"] - data.y.mean()) > 1e-10 for fold in loo["folds"])
    assert loo["threshold_cd"] == pytest.approx(.0003) and not loo["holdout_labels_used"]
    json.dumps(loo, allow_nan=False)
    assert loo["rmse_cd"] >= loo["mae_cd"] and loo["max_abs_error_cd"] >= loo["mae_cd"]


def test_loo_resume_reuses_only_matching_fold_snapshots(config, monkeypatch):
    data = prepare_training_data(records(config), config)
    loo = loo_cv(data, config, .006)
    actual = surrogate.fit_gp
    called = []

    def track(training, cfg):
        called.append(training.data_hash)
        return actual(training, cfg)

    monkeypatch.setattr(surrogate, "fit_gp", track)
    resumed = loo_cv(data, config, .006, completed_folds=loo["folds"][:3])
    assert len(called) == 6 and resumed == loo
    altered = deepcopy(loo["folds"][:3])
    altered[0]["training_ids"].append(altered[0]["held_out_design_id"])
    with pytest.raises(ValueError, match="checkpoint differs"):
        loo_cv(data, config, .006, completed_folds=altered)


def test_model_save_load_and_digest_before_unpickling(config, lab, monkeypatch):
    data = prepare_training_data(records(config), config)
    model = fit_gp(data, config)
    directory = lab / "iter_000"
    metadata = save_model(model, directory, config)
    loaded = load_model(directory, metadata, data, config)
    np.testing.assert_array_equal(loaded.predict_normalized(data.x)[0], model.predict_normalized(data.x)[0])
    (directory / "gp.joblib").write_bytes(b"corrupt model")
    monkeypatch.setattr(surrogate.joblib, "load", lambda *_: pytest.fail("must reject hash before unpickling"))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model(directory, metadata, data, config)


def test_minimizing_ei_direction_and_zero_sigma():
    best, xi = .006, 1e-5
    low, high = expected_improvement([.0058, .0062], [.00005, .00005], best, xi)
    assert low > high >= 0
    np.testing.assert_allclose(expected_improvement([.0058, .0062], [0, 1e-14], best, xi), [.00019, 0], atol=1e-15)
    assert float(expected_improvement(best - xi, .0001, best, xi)) == pytest.approx(.0001 / np.sqrt(2 * np.pi))
    with pytest.raises(ValueError):
        expected_improvement(.005, -.1, best, xi)
    with pytest.raises(ValueError):
        expected_improvement(float('nan'), .1, best, xi)


class ToyPredictor:
    x_train = np.array([[.1, .1], [.1, .9], [.9, .1], [.9, .9], [.5, .5]])

    def __init__(self, mode="normal"):
        self.mode = mode

    def predict_normalized(self, x):
        x = np.asarray(x)
        mean = .0058 + .001 * ((x[:, 0] - .75)**2 + (x[:, 1] - .15)**2)
        std = np.full(len(x), .00005)
        if self.mode == "invalid":
            mean[:] = np.nan
        elif self.mode == "degenerate":
            mean[:] = .007
            std[:] = 0
        return mean, std


def test_sobol_boundary_pool_fixed_iteration_seed(config):
    pool = candidate_pool(config, 1)
    assert len(pool) == 4096 + 256
    assert all(any(np.array_equal(point, corner) for point in pool) for corner in [[0, 0], [0, 1], [1, 0], [1, 1]])
    np.testing.assert_array_equal(pool, candidate_pool(config, 1))
    assert not np.array_equal(pool, candidate_pool(config, 2))


def test_pool_and_de_comparison_is_reproducible_and_unassessed(config):
    first = propose(ToyPredictor(), config, [[.2, .3]], f_best=.006)
    second = propose(ToyPredictor(), config, [[.2, .3]], f_best=.006)
    assert first == second and first["source"] == "differential_evolution"
    assert first["ei"] >= first["max_pool_ei"] and first["reason"] == "maximum_expected_improvement"
    assert first["min_excluded_distance"] >= .001 and first["status"] == "proposed"
    assert not first["assessed"] and first["new_solver_launches"] == 0
    assert first["seed"] == 1001
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize("mode,poor_loo,count,reason", [
    ("normal", True, 0, "poor_loo_exploration"), ("normal", False, 3, "scheduled_exploration"),
    ("invalid", False, 0, "invalid_ei"), ("degenerate", False, 0, "degenerate_ei")])
def test_exploration_fallbacks_ignore_labels_and_use_training_distance(config, mode, poor_loo, count, reason):
    predictor = ToyPredictor(mode)
    occupied = np.vstack([predictor.x_train, [[0, 1], [1, 1]]])
    result = propose(predictor, config, occupied, f_best=.006, poor_loo=poor_loo, successful_sequential_points=count)
    assert result["reason"] == reason and result["source"] == "maximin_pool" and not result["de"]["attempted"]
    pool = candidate_pool(config, 1)
    valid = pool[nearest_distances(pool, occupied) >= .001]
    assert result["min_training_distance"] == pytest.approx(nearest_distances(valid, predictor.x_train).max())
    json.dumps(result, allow_nan=False)
    if mode == "invalid":
        assert result["mu_cd"] is None and result["sigma_cd"] is None and result["ei"] is None


def test_de_duplicate_is_discarded_and_valid_pool_wins(config, monkeypatch):
    occupied = [[.75, .15]]  # Best prediction is explicitly a reserved holdout coordinate.
    monkeypatch.setattr(acquisition, "differential_evolution", lambda *args, **kwargs:
        SimpleNamespace(x=np.array(occupied[0]), success=False, message="injected duplicate", nfev=1, nit=1))
    result = propose(ToyPredictor(), config, occupied, f_best=.006)
    assert result["source"] == "pool" and result["min_excluded_distance"] >= .001
    assert not result["de"]["valid"] and result["diagnostics"]


def test_no_valid_distance_candidate_stops(config):
    values = config.to_dict()
    values["doe"]["min_normalized_distance"] = values["acquisition"]["min_normalized_distance"] = 2.
    changed = FrozenConfig(values)
    with pytest.raises(RuntimeError, match="no pool point"):
        propose(ToyPredictor(), changed, [], f_best=.006)


def test_mutated_training_snapshot_cannot_be_used_for_loo(config):
    data = prepare_training_data(records(config), config)
    data.records[0]["cd"] += .001
    with pytest.raises(ValueError, match="snapshot changed"):
        loo_cv(data, config, .006)
