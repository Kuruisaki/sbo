"""Shared immutable evaluation records and budget exception."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

class BudgetExhausted(RuntimeError):
    """No launch reservation is available; do not start a solver process."""


@dataclass(frozen=True)
class EvaluationResult:
    eval_id: str
    design_id: str
    config_hash: str
    cache_key: str
    split: str
    purpose: str
    fidelity: dict[str, Any]
    status: str
    failure_reason: str | None
    selected_attempt: int | None
    aerodynamics: dict[str, Any] | None
    attempts: tuple[dict[str, Any], ...]
    cache_hit: bool
    started_at_utc: str
    ended_at_utc: str
    evaluation_dir: Path
    result_path: Path
    geometry_file_sha256: str
    geometry_hash: str

    @property
    def usable_for_training(self) -> bool:
        return (self.status == "success" and self.split == "train"
                and self.purpose in {"baseline", "doe", "sequential"}
                and self.fidelity["panel_count"] == self.fidelity["standard_panel_count"]
                and self.aerodynamics is not None)

    def to_dict(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvaluationResult:
        values = dict(payload)
        for key in ("evaluation_dir", "result_path"):
            values[key] = Path(values[key])
        values["attempts"] = tuple(values["attempts"])
        return cls(**values)


