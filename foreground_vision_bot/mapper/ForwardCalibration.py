from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    return float(value)


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


@dataclass(frozen=True)
class ForwardCalibrationTrial:
    """One settled forward pulse used to fit the fixed-camera travel scale."""

    requested_seconds: float
    actual_seconds: float
    distance_px: float
    confidence: float
    tracked_points: int


@dataclass(frozen=True)
class ForwardMotionModel:
    """
    Relative forward scale for the fixed game camera.

    One map cell is defined as the visual travel produced by
    ``nominal_seconds`` after removing the fitted stationary-flow baseline.
    This is deliberately a relative map unit, not a claim about game metres.
    """

    version: int
    nominal_seconds: float
    flow_rate_px_per_second: float
    baseline_flow_px: float
    dead_time_seconds: float
    pixels_per_cell: float
    rmse_px: float
    r_squared: float
    sample_count: int
    frame_width: int
    frame_height: int

    def __post_init__(self) -> None:
        finite_values = (
            self.nominal_seconds,
            self.flow_rate_px_per_second,
            self.baseline_flow_px,
            self.dead_time_seconds,
            self.pixels_per_cell,
            self.rmse_px,
            self.r_squared,
        )
        if not all(isfinite(value) for value in finite_values):
            raise ValueError("forward calibration values must be finite")
        if self.version != 1:
            raise ValueError(f"unsupported forward calibration version: {self.version}")
        if self.nominal_seconds <= 0.0:
            raise ValueError("nominal_seconds must be positive")
        if self.flow_rate_px_per_second <= 0.0:
            raise ValueError("flow_rate_px_per_second must be positive")
        if self.baseline_flow_px < 0.0 or self.dead_time_seconds < 0.0:
            raise ValueError("forward baseline/dead time cannot be negative")
        if self.pixels_per_cell <= 0.0:
            raise ValueError("pixels_per_cell must be positive")
        if self.rmse_px < 0.0:
            raise ValueError("rmse_px cannot be negative")
        if self.sample_count < 3:
            raise ValueError("forward calibration needs at least three samples")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("forward calibration frame size must be positive")

    def predicted_flow_px(self, actual_seconds: float) -> float:
        """Predict robust scene flow for one measured key hold."""
        duration = max(0.0, float(actual_seconds) - self.dead_time_seconds)
        return self.baseline_flow_px + self.flow_rate_px_per_second * duration

    def cells_for_flow(self, distance_px: float) -> float:
        """Convert observed robust scene flow to non-negative relative cells."""
        motion_px = max(0.0, float(distance_px) - self.baseline_flow_px)
        return motion_px / self.pixels_per_cell

    def matches_frame(self, frame) -> bool:
        """Reject use with a capture resolution different from calibration."""
        if frame is None or getattr(frame, "ndim", 0) < 2:
            return False
        height, width = frame.shape[:2]
        return width == self.frame_width and height == self.frame_height

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ForwardMotionModel:
        return cls(
            version=_as_int(data["version"], "version"),
            nominal_seconds=_as_float(
                data["nominal_seconds"],
                "nominal_seconds",
            ),
            flow_rate_px_per_second=_as_float(
                data["flow_rate_px_per_second"],
                "flow_rate_px_per_second",
            ),
            baseline_flow_px=_as_float(
                data.get("baseline_flow_px", 0.0),
                "baseline_flow_px",
            ),
            dead_time_seconds=_as_float(
                data.get("dead_time_seconds", 0.0),
                "dead_time_seconds",
            ),
            pixels_per_cell=_as_float(
                data["pixels_per_cell"],
                "pixels_per_cell",
            ),
            rmse_px=_as_float(data["rmse_px"], "rmse_px"),
            r_squared=_as_float(data["r_squared"], "r_squared"),
            sample_count=_as_int(data["sample_count"], "sample_count"),
            frame_width=_as_int(data["frame_width"], "frame_width"),
            frame_height=_as_int(data["frame_height"], "frame_height"),
        )


def fit_forward_motion_model(
    trials: Iterable[ForwardCalibrationTrial],
    *,
    nominal_seconds: float,
    frame_width: int,
    frame_height: int,
    minimum_confidence: float = 0.55,
    minimum_samples: int = 5,
    maximum_relative_rmse: float = 0.35,
    minimum_r_squared: float = 0.45,
) -> ForwardMotionModel:
    """
    Fit a robust fixed-camera forward response from multiple pulse durations.

    A weighted linear fit is trimmed by median absolute residual. A negative
    intercept is represented as command dead time; a positive intercept is
    retained as stationary/dynamic-scene baseline and subtracted at runtime.
    """
    nominal_seconds = float(nominal_seconds)
    if not isfinite(nominal_seconds) or nominal_seconds <= 0.0:
        raise ValueError("nominal_seconds must be finite and positive")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame size must be positive")
    if minimum_samples < 3:
        raise ValueError("minimum_samples must be at least three")

    accepted = [
        trial
        for trial in trials
        if _valid_trial(trial, minimum_confidence=minimum_confidence)
    ]
    if len(accepted) < minimum_samples:
        raise ValueError(
            "Not enough confident forward-motion trials "
            f"({len(accepted)}/{minimum_samples})."
        )

    distinct_durations = {round(trial.actual_seconds, 4) for trial in accepted}
    if len(distinct_durations) < 3:
        raise ValueError(
            "Forward calibration requires at least three distinct pulse durations."
        )

    durations = np.asarray(
        [trial.actual_seconds for trial in accepted],
        dtype=np.float64,
    )
    distances = np.asarray(
        [trial.distance_px for trial in accepted],
        dtype=np.float64,
    )
    weights = np.asarray(
        [max(0.05, trial.confidence) for trial in accepted],
        dtype=np.float64,
    )
    inliers = np.ones(len(accepted), dtype=bool)

    slope = float("nan")
    intercept = float("nan")
    for _ in range(3):
        slope, intercept = _weighted_line(
            durations[inliers],
            distances[inliers],
            weights[inliers],
        )
        residuals = distances - (slope * durations + intercept)
        center = float(np.median(residuals[inliers]))
        mad = float(np.median(np.abs(residuals[inliers] - center)))
        robust_sigma = 1.4826 * mad
        threshold = max(3.0 * robust_sigma, 0.75)
        candidate = np.abs(residuals - center) <= threshold
        if int(candidate.sum()) < minimum_samples or np.array_equal(candidate, inliers):
            break
        inliers = candidate

    if not isfinite(slope) or slope <= 0.0:
        raise ValueError("Forward travel did not increase with pulse duration.")

    shortest_duration = float(np.min(durations[inliers]))
    if intercept < 0.0:
        dead_time = min(-intercept / slope, shortest_duration * 0.90)
        baseline = 0.0
    else:
        dead_time = 0.0
        baseline = intercept

    effective_duration = nominal_seconds - dead_time
    if effective_duration <= 0.0:
        raise ValueError("Fitted forward dead time exceeds the nominal pulse.")
    pixels_per_cell = slope * effective_duration
    if pixels_per_cell <= 1.0:
        raise ValueError("Forward calibration produced too little visual travel.")

    predicted = baseline + slope * np.maximum(0.0, durations[inliers] - dead_time)
    residuals = distances[inliers] - predicted
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    relative_rmse = rmse / pixels_per_cell

    centered = distances[inliers] - float(np.mean(distances[inliers]))
    total_variation = float(np.sum(np.square(centered)))
    residual_variation = float(np.sum(np.square(residuals)))
    r_squared = (
        1.0 - residual_variation / total_variation if total_variation > 1e-9 else 0.0
    )

    if relative_rmse > maximum_relative_rmse:
        raise ValueError(
            "Forward calibration is too inconsistent "
            f"(relative RMSE {relative_rmse:.2f})."
        )
    if r_squared < minimum_r_squared:
        raise ValueError(
            "Forward travel is not sufficiently correlated with pulse duration "
            f"(R² {r_squared:.2f})."
        )

    return ForwardMotionModel(
        version=1,
        nominal_seconds=nominal_seconds,
        flow_rate_px_per_second=slope,
        baseline_flow_px=baseline,
        dead_time_seconds=dead_time,
        pixels_per_cell=pixels_per_cell,
        rmse_px=rmse,
        r_squared=r_squared,
        sample_count=int(inliers.sum()),
        frame_width=int(frame_width),
        frame_height=int(frame_height),
    )


def _valid_trial(
    trial: ForwardCalibrationTrial,
    *,
    minimum_confidence: float,
) -> bool:
    values = (
        trial.requested_seconds,
        trial.actual_seconds,
        trial.distance_px,
        trial.confidence,
    )
    return (
        all(isfinite(value) for value in values)
        and trial.requested_seconds > 0.0
        and trial.actual_seconds > 0.0
        and trial.distance_px > 0.5
        and trial.confidence >= minimum_confidence
        and trial.tracked_points >= 8
    )


def _weighted_line(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    design = np.column_stack((x, np.ones_like(x)))
    root_weights = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(
        design * root_weights[:, None],
        y * root_weights,
        rcond=None,
    )
    return float(coefficients[0]), float(coefficients[1])
