from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .ForwardCalibration import ForwardMotionModel
from .RotationModel import IdleResponseCurves, StateAwareRotationModel


class CalibrationSchemaError(ValueError):
    """A persisted mapper calibration is missing, unsupported, or inconsistent."""


@dataclass(frozen=True)
class MapperCalibration:
    """Validated mapper calibration artifact and its JSON schema boundary."""

    CURRENT_VERSION = 9

    version: int
    created_at: str
    source: str
    left_seconds_90: float
    right_seconds_90: float
    left_heading_sign: int
    right_heading_sign: int
    left_trials: list[dict[str, object]]
    right_trials: list[dict[str, object]]
    neutral_after_seconds: float
    neutral_timeout_trials: list[dict[str, object]]
    neutral_timeout_fit: dict[str, object]
    transition_trials: list[dict[str, object]]
    refinement_trials: list[dict[str, object]]
    rotation_model: StateAwareRotationModel
    forward_trials: list[dict[str, object]]
    forward_model: ForwardMotionModel

    def __post_init__(self) -> None:
        if self.version != self.CURRENT_VERSION:
            raise CalibrationSchemaError(
                f"unsupported mapper calibration version {self.version}; "
                f"expected {self.CURRENT_VERSION}"
            )
        if not self.created_at.strip():
            raise CalibrationSchemaError("created_at must not be empty")
        if not self.source.strip():
            raise CalibrationSchemaError("source must not be empty")
        if {self.left_heading_sign, self.right_heading_sign} != {-1, 1}:
            raise CalibrationSchemaError(
                "left/right heading signs must contain exactly -1 and 1"
            )
        if len(self.neutral_timeout_trials) < 4:
            raise CalibrationSchemaError(
                "validated turn-memory evidence requires at least four trials"
            )

        curve_data = self.neutral_timeout_fit.get("idle_response_curves")
        if not isinstance(curve_data, dict):
            raise CalibrationSchemaError(
                "neutral_timeout_fit is missing idle_response_curves"
            )
        neutral_curves = IdleResponseCurves.from_dict(
            cast(dict[str, object], curve_data)
        )
        if self.rotation_model.idle_response_curves != neutral_curves:
            raise CalibrationSchemaError(
                "rotation model and neutral-timeout fit contain different "
                "idle-response curves"
            )

        policy_data = self.neutral_timeout_fit.get("turn_memory_policy")
        if not isinstance(policy_data, dict):
            raise CalibrationSchemaError(
                "neutral_timeout_fit is missing turn_memory_policy"
            )
        if self.rotation_model.turn_memory_policy.to_dict() != policy_data:
            raise CalibrationSchemaError(
                "rotation model and neutral-timeout fit contain different "
                "turn-memory policies"
            )

    @classmethod
    def load(cls, path: Path) -> MapperCalibration:
        try:
            loaded: object = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationSchemaError(
                "mapper calibration could not be read as JSON"
            ) from error
        if not isinstance(loaded, dict):
            raise CalibrationSchemaError(
                "mapper calibration root must be a JSON object"
            )
        return cls.from_dict(cast(dict[str, object], loaded))

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MapperCalibration:
        version = _required_int(data, "version")
        if version != cls.CURRENT_VERSION:
            raise CalibrationSchemaError(
                f"unsupported mapper calibration version {version}; "
                f"expected {cls.CURRENT_VERSION}"
            )

        rotation_data = _required_dict(data, "rotation_model")
        forward_data = _required_dict(data, "forward_model")
        try:
            rotation_model = StateAwareRotationModel.from_dict(rotation_data)
            forward_model = ForwardMotionModel.from_dict(forward_data)
        except (KeyError, OverflowError, TypeError, ValueError) as error:
            raise CalibrationSchemaError(
                "rotation_model or forward_model is invalid"
            ) from error

        try:
            return cls(
                version=version,
                created_at=_required_str(data, "created_at"),
                source=_required_str(data, "source"),
                left_seconds_90=_required_float(data, "left_seconds_90"),
                right_seconds_90=_required_float(data, "right_seconds_90"),
                left_heading_sign=_required_int(data, "left_heading_sign"),
                right_heading_sign=_required_int(data, "right_heading_sign"),
                left_trials=_required_record_list(data, "left_trials"),
                right_trials=_required_record_list(data, "right_trials"),
                neutral_after_seconds=_required_float(data, "neutral_after_seconds"),
                neutral_timeout_trials=_required_record_list(
                    data, "neutral_timeout_trials"
                ),
                neutral_timeout_fit=_required_dict(data, "neutral_timeout_fit"),
                transition_trials=_required_record_list(data, "transition_trials"),
                refinement_trials=_required_record_list(data, "refinement_trials"),
                rotation_model=rotation_model,
                forward_trials=_required_record_list(data, "forward_trials"),
                forward_model=forward_model,
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, CalibrationSchemaError):
                raise
            raise CalibrationSchemaError(
                "mapper calibration contains invalid fields"
            ) from error

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "source": self.source,
            "left_seconds_90": self.left_seconds_90,
            "right_seconds_90": self.right_seconds_90,
            "left_heading_sign": self.left_heading_sign,
            "right_heading_sign": self.right_heading_sign,
            "left_trials": self.left_trials,
            "right_trials": self.right_trials,
            "neutral_after_seconds": self.neutral_after_seconds,
            "neutral_timeout_trials": self.neutral_timeout_trials,
            "neutral_timeout_fit": self.neutral_timeout_fit,
            "transition_trials": self.transition_trials,
            "refinement_trials": self.refinement_trials,
            "rotation_model": self.rotation_model.to_dict(),
            "forward_trials": self.forward_trials,
            "forward_model": self.forward_model.to_dict(),
        }


def _required_value(data: dict[str, object], field: str) -> object:
    if field not in data:
        raise CalibrationSchemaError(f"missing required field: {field}")
    return data[field]


def _required_str(data: dict[str, object], field: str) -> str:
    value = _required_value(data, field)
    if not isinstance(value, str):
        raise CalibrationSchemaError(f"{field} must be a string")
    return value


def _required_int(data: dict[str, object], field: str) -> int:
    value = _required_value(data, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationSchemaError(f"{field} must be an integer")
    return value


def _required_float(data: dict[str, object], field: str) -> float:
    value = _required_value(data, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationSchemaError(f"{field} must be numeric")
    return float(value)


def _required_dict(data: dict[str, object], field: str) -> dict[str, object]:
    value = _required_value(data, field)
    if not isinstance(value, dict):
        raise CalibrationSchemaError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _required_record_list(
    data: dict[str, object], field: str
) -> list[dict[str, object]]:
    value = _required_value(data, field)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise CalibrationSchemaError(f"{field} must be a list of objects")
    return [cast(dict[str, object], item) for item in value]
