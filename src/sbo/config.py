"""Strict, immutable configuration and run-directory creation for SBO."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import yaml

from .geometry import GeometrySettings, generate_from_config
from .polar import PolarSettings
from .xfoil_runner import XfoilSettings


class ConfigurationError(ValueError):
    """Unknown, missing, unsupported or inconsistent configuration."""


# Fixed v1 field/type contract; no defaults are inferred from an editable YAML.
_SCHEMA = {'design.variables': 'list',
 'design.bounds': 'list',
 'design.baseline': 'list',
 'design.t': 'float',
 'design.objective': 'str',
 'design.direction': 'str',
 'design.continuous_parameters': 'bool',
 'geometry.family': 'str',
 'geometry.generator_version': 'str',
 'geometry.chord': 'float',
 'geometry.points_per_side': 'int',
 'geometry.spacing': 'str',
 'geometry.trailing_edge_coefficient': 'float',
 'geometry.surface_offset': 'str',
 'geometry.derivative': 'str',
 'geometry.point_order': 'str',
 'geometry.merge_leading_edge': 'bool',
 'geometry.merge_trailing_edge': 'bool',
 'geometry.normalize_coordinates': 'bool',
 'geometry.clip_x_coordinates': 'bool',
 'geometry.name_template': 'str',
 'geometry.checks': 'list',
 'flow.reynolds': 'float',
 'flow.mach': 'float',
 'flow.target_cl': 'float',
 'flow.type': 'int',
 'flow.ncrit': 'float',
 'flow.xtr': 'list',
 'flow.cl_tolerance': 'float',
 'flow.viscous': 'bool',
 'xfoil.executable': 'str',
 'xfoil.expected_version': 'str',
 'xfoil.evaluation_policy_version': 'str',
 'xfoil.disable_graphics': 'bool',
 'xfoil.standard_n': 'int',
 'xfoil.ppar.p': 'float',
 'xfoil.ppar.t': 'float',
 'xfoil.ppar.r': 'float',
 'xfoil.ppar.xt': 'list',
 'xfoil.ppar.xb': 'list',
 'xfoil.max_attempts': 'int',
 'xfoil.initial_attempt.iteration_limit': 'int',
 'xfoil.initial_attempt.timeout_seconds': 'int',
 'xfoil.initial_attempt.cl_sequence': 'list',
 'xfoil.retry_attempt.iteration_limit': 'int',
 'xfoil.retry_attempt.timeout_seconds': 'int',
 'xfoil.retry_attempt.cl_sequence': 'list',
 'xfoil.fresh_process_per_attempt': 'bool',
 'xfoil.fresh_directory_per_attempt': 'bool',
 'xfoil.kill_process_tree_on_timeout': 'bool',
 'xfoil.accept_only_target_cl': 'bool',
 'xfoil.require_normal_exit': 'bool',
 'xfoil.require_target_convergence': 'bool',
 'xfoil.required_finite_columns': 'list',
 'xfoil.require_positive_cd': 'bool',
 'doe.method': 'str',
 'doe.initial_points': 'int',
 'doe.include_baseline': 'bool',
 'doe.seed': 'int',
 'doe.holdout_points': 'int',
 'doe.holdout_seed': 'int',
 'doe.min_normalized_distance': 'float',
 'doe.replacement_method': 'str',
 'doe.reserve_holdout_coordinates': 'bool',
 'doe.seal_holdout_labels_until_final_verification': 'bool',
 'doe.reuse_accepted_preflight_baseline': 'bool',
 'gp.model': 'str',
 'gp.input_normalization': 'str',
 'gp.kernel': 'str',
 'gp.matern_nu': 'float',
 'gp.ard': 'bool',
 'gp.length_scale_initial': 'list',
 'gp.length_scale_bounds': 'list',
 'gp.constant_value_initial': 'float',
 'gp.constant_value_bounds': 'list',
 'gp.normalize_y': 'bool',
 'gp.alpha_regularization': 'float',
 'gp.alpha_regularization_max': 'float',
 'gp.alpha_regularization_schedule': 'list',
 'gp.n_restarts_optimizer': 'int',
 'gp.seed': 'int',
 'gp.white_kernel': 'bool',
 'gp.prediction_scale': 'str',
 'gp.loo_refit_each_fold': 'bool',
 'gp.record_fit_warnings': 'bool',
 'acquisition.method': 'str',
 'acquisition.xi': 'float',
 'acquisition.sobol_pool_size': 'int',
 'acquisition.sobol_scramble': 'bool',
 'acquisition.supplement_boundaries': 'bool',
 'acquisition.include_corners': 'bool',
 'acquisition.boundary_points_per_edge': 'int',
 'acquisition.optimizer': 'str',
 'acquisition.seed_base': 'int',
 'acquisition.iteration_start': 'int',
 'acquisition.sobol_seed_policy': 'str',
 'acquisition.compare_optimizer_with_pool': 'bool',
 'acquisition.min_normalized_distance': 'float',
 'acquisition.exclude': 'list',
 'acquisition.exploration_every_successful_points': 'int',
 'acquisition.exploration_method': 'str',
 'acquisition.exploration_on_poor_loo': 'bool',
 'acquisition.fallback_on': 'list',
 'acquisition.sigma_zero_threshold': 'float',
 'acquisition.sigma_zero_policy': 'str',
 'budget.max_successful_sequential_points': 'int',
 'budget.max_planned_valid_standard_designs': 'int',
 'budget.max_distinct_standard_designs': 'int',
 'budget.max_process_launches': 'int',
 'budget.reserved_final_verification_launches': 'int',
 'budget.check_before_every_launch': 'bool',
 'budget.protect_verification_reserve': 'bool',
 'stopping.convergence.min_successful_sequential_points': 'int',
 'stopping.convergence.successful_points_window': 'int',
 'stopping.convergence.max_best_cd_improvement': 'float',
 'stopping.convergence.max_pool_ei': 'float',
 'stopping.convergence.combine_conditions': 'str',
 'stopping.convergence.improvement_comparison': 'str',
 'stopping.convergence.ei_comparison': 'str',
 'stopping.max_consecutive_failed_designs': 'int',
 'stopping.failure_rate_window': 'int',
 'stopping.max_failure_rate': 'float',
 'stopping.failure_rate_comparison': 'str',
 'stopping.require_full_failure_rate_window': 'bool',
 'stopping.failure_count_policy': 'str',
 'stopping.stop_on_budget_exhaustion': 'bool',
 'quality.baseline.panel_counts': 'list',
 'quality.baseline.repeat_n': 'int',
 'quality.baseline.max_repeat_cd_difference': 'float',
 'quality.baseline.require_all_target_points_valid': 'bool',
 'quality.baseline.require_geometry_transition_cp_review': 'bool',
 'quality.panel_convergence.compare_n': 'list',
 'quality.panel_convergence.absolute_cd_threshold': 'float',
 'quality.panel_convergence.relative_cd_threshold': 'float',
 'quality.panel_convergence.threshold_combination': 'str',
 'quality.panel_convergence.relative_reference_n': 'int',
 'quality.loo.metric': 'str',
 'quality.loo.absolute_cd_threshold': 'float',
 'quality.loo.relative_baseline_cd_threshold': 'float',
 'quality.loo.threshold_combination': 'str',
 'quality.loo.baseline_reference_n': 'int',
 'quality.loo.on_exceedance': 'str',
 'quality.holdout.metric': 'str',
 'quality.holdout.absolute_cd_threshold': 'float',
 'quality.holdout.relative_baseline_cd_threshold': 'float',
 'quality.holdout.threshold_combination': 'str',
 'quality.holdout.baseline_reference_n': 'int',
 'quality.holdout.on_exceedance': 'str',
 'quality.holdout.use_for_training_or_candidate_selection': 'bool',
 'quality.manual_review.abs_alpha_deg_above': 'float',
 'quality.manual_review.cd_above': 'float',
 'quality.manual_review.status': 'str',
 'quality.manual_review.exclude_from_training_until_accepted': 'bool',
 'verification.candidate_count': 'int',
 'verification.candidates_from': 'str',
 'verification.preliminary_rank_by': 'str',
 'verification.include_baseline': 'bool',
 'verification.panel_counts': 'list',
 'verification.repeat_n': 'int',
 'verification.bypass_cache': 'bool',
 'verification.save_cp': 'bool',
 'verification.save_transition_locations': 'bool',
 'verification.require_all_target_points_valid': 'bool',
 'verification.apply_panel_convergence_quality_gate': 'bool',
 'verification.max_repeat_cd_difference': 'float',
 'verification.numerical_uncertainty_floor': 'float',
 'verification.numerical_uncertainty_components': 'list',
 'verification.uncertainty_combination': 'str',
 'verification.final_rank_n': 'int',
 'verification.certify_improvement_above': 'str',
 'verification.report_standard_n_improvement': 'bool',
 'verification.recommendation_without_certified_improvement': 'str',
 'verification.recommendation_on_baseline_verification_failure': 'str',
 'verification.allow_verification_data_in_training': 'bool',
 'verification.holdout_metrics': 'list',
 'verification.holdout_interval_sigma': 'float',
 'runtime.config_version': 'str',
 'runtime.project_root': 'str',
 'runtime.runs_root': 'str',
 'runtime.reference_root': 'str',
 'runtime.execution': 'str',
 'runtime.workers': 'int',
 'runtime.log_level': 'str',
 'runtime.reject_unknown_fields': 'bool',
 'runtime.reject_missing_required_fields': 'bool',
 'runtime.freeze_config_at_run_creation': 'bool',
 'runtime.subsequent_stages_read_frozen_config': 'bool',
 'runtime.verify_frozen_config_hash': 'bool',
 'runtime.overwrite_existing_run': 'bool',
 'runtime.resume_budget_source': 'str',
 'runtime.persist_random_state': 'bool',
 'runtime.budget_counting.cache_hit_counts_as_launch': 'bool',
 'runtime.budget_counting.retry_counts_as_new_design': 'bool',
 'runtime.budget_counting.retry_counts_as_launch': 'bool',
 'runtime.budget_counting.verification_counts_as_standard_design': 'bool',
 'runtime.budget_counting.verification_counts_as_launch': 'bool',
 'runtime.budget_counting.preflight_counts_as_launch': 'bool',
 'runtime.needs_review.action': 'str',
 'runtime.needs_review.exit_nonzero': 'bool',
 'runtime.needs_review.require_explicit_decision_event': 'bool',
 'runtime.needs_review.allowed_decisions': 'list',
 'runtime.needs_review.auto_accept': 'bool'}


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigurationError("configuration keys must be strings")
        if key in mapping:
            raise ConfigurationError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _leaves(values: Mapping[str, Any], prefix="") -> dict[str, Any]:
    result = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ConfigurationError("configuration keys must be strings")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping) and value:
            result.update(_leaves(value, path))
        else:
            result[path] = value
    return result


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def validate_config(values: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise ConfigurationError("configuration must be a mapping")
    data = deepcopy(dict(values))
    leaves = _leaves(data)
    if set(leaves) != set(_SCHEMA):
        raise ConfigurationError(
            f"configuration fields: missing={sorted(set(_SCHEMA) - set(leaves))}, "
            f"unknown={sorted(set(leaves) - set(_SCHEMA))}"
        )
    for path, kind in _SCHEMA.items():
        value = leaves[path]
        valid = (kind == "bool" and type(value) is bool
                 or kind == "int" and type(value) is int
                 or kind == "str" and isinstance(value, str)
                 or kind == "list" and isinstance(value, (list, tuple))
                 or kind == "float" and not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value)))
        if not valid:
            raise ConfigurationError(f"{path} must have type {kind}")
        if kind == "float":
            parent = data
            parts = path.split(".")
            for part in parts[:-1]:
                parent = parent[part]
            parent[parts[-1]] = float(value)
    try:
        GeometrySettings.from_mapping(data["geometry"])
        generate_from_config(data)
        for number in (1, 2):
            XfoilSettings.from_config(data, attempt_number=number)
        PolarSettings.from_config(data)
        runtime = data["runtime"]
        if runtime["execution"] != "serial" or runtime["workers"] != 1:
            raise ConfigurationError("M1 supports serial execution with workers=1")
        for key in ("reject_unknown_fields", "reject_missing_required_fields", "freeze_config_at_run_creation",
                    "subsequent_stages_read_frozen_config", "verify_frozen_config_hash"):
            if runtime[key] is not True:
                raise ConfigurationError(f"runtime.{key} must be true")
        if runtime["overwrite_existing_run"] is not False or runtime["resume_budget_source"] != "database":
            raise ConfigurationError("existing runs cannot be overwritten; budgets must be read from the database")
        expected_counting = {
            "cache_hit_counts_as_launch": False, "retry_counts_as_new_design": False,
            "retry_counts_as_launch": True, "verification_counts_as_standard_design": False,
            "verification_counts_as_launch": True, "preflight_counts_as_launch": True,
        }
        if runtime["budget_counting"] != expected_counting:
            raise ConfigurationError("unsupported budget-counting policy")
        review = runtime["needs_review"]
        if (review["action"] != "save_and_stop" or review["exit_nonzero"] is not True
                or review["require_explicit_decision_event"] is not True or review["auto_accept"] is not False
                or review["allowed_decisions"] != ["accept", "reject"]):
            raise ConfigurationError("unsupported human-review policy")
        budget = data["budget"]
        for key in ("max_successful_sequential_points", "max_planned_valid_standard_designs",
                    "max_distinct_standard_designs", "max_process_launches"):
            if budget[key] <= 0:
                raise ConfigurationError(f"budget.{key} must be positive")
        reserve = budget["reserved_final_verification_launches"]
        if not 0 <= reserve < budget["max_process_launches"]:
            raise ConfigurationError("final verification reserve must fit the process-launch budget")
        if budget["check_before_every_launch"] is not True or budget["protect_verification_reserve"] is not True:
            raise ConfigurationError("every launch must enforce budgets and the final reserve")
        for name in ("project_root", "runs_root", "reference_root"):
            path = Path(runtime[name])
            if not path.is_absolute() or not str(path).isascii():
                raise ConfigurationError(f"runtime.{name} must be an absolute ASCII path")
            runtime[name] = str(path.resolve())
        project = Path(runtime["project_root"])
        for name in ("runs_root", "reference_root"):
            if not Path(runtime[name]).is_relative_to(project) or Path(runtime[name]) == project:
                raise ConfigurationError(f"runtime.{name} must be a child directory of project_root")
        if data["acquisition"]["min_normalized_distance"] != data["doe"]["min_normalized_distance"]:
            raise ConfigurationError("DOE and acquisition distance thresholds must agree")
        gp = data["gp"]
        if not (0 < gp["alpha_regularization"] <= gp["alpha_regularization_max"]):
            raise ConfigurationError("invalid GP regularization range")
        schedule = gp["alpha_regularization_schedule"]
        if (not schedule or schedule[0] != gp["alpha_regularization"]
                or schedule[-1] != gp["alpha_regularization_max"] or list(schedule) != sorted(set(schedule))):
            raise ConfigurationError("regularization schedule must be increasing and span the configured range")
        # All list values must also be JSON-exchangeable and finite.
        canonical_json(data)
    except (ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(str(exc)) from exc
    return data


class FrozenConfig(Mapping[str, Any]):
    def __init__(self, values: Mapping[str, Any]):
        self._data = validate_config(values)
        self._view = _freeze(self._data)
        self.config_hash = sha256(canonical_json(self._data).encode("utf-8")).hexdigest()

    def __getitem__(self, key):
        return self._view[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._view)

    def __len__(self):
        return len(self._view)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)


def load_config(path: str | Path) -> FrozenConfig:
    try:
        values = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML: {exc}") from exc
    return FrozenConfig(values)


def create_run(config: FrozenConfig, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ConfigurationError("run_id must be 1..64 ASCII letters/digits/underscores/hyphens")
    if run_id.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1,10)), *(f"LPT{i}" for i in range(1,10))}:
        raise ConfigurationError("run_id cannot be a Windows device name")
    directory = Path(config["runtime"]["runs_root"]) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    frozen_path = directory / "config.frozen.yaml"
    frozen_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(directory / "manifest.json", {
        "run_id": run_id, "config_hash": config.config_hash,
        "config_file_sha256": file_sha256(frozen_path), "status": "planned", "stage": "preflight",
    })
    return directory


def verify_run_config(run_dir: str | Path, config: FrozenConfig) -> None:
    directory = Path(run_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_hash"] != config.config_hash or file_sha256(directory / "config.frozen.yaml") != manifest["config_file_sha256"]:
        raise ConfigurationError("run's frozen configuration or its hash has changed")
    if load_config(directory / "config.frozen.yaml").config_hash != config.config_hash:
        raise ConfigurationError("frozen configuration does not match the requested evaluation configuration")
