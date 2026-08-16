from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from math import isclose, isfinite

import numpy as np

MINIMUM_FORWARD_SAMPLES = 5
MINIMUM_FORWARD_R_SQUARED = 0.70
MAXIMUM_FORWARD_RELATIVE_RMSE = 0.15
MINIMUM_FORWARD_PIXELS_PER_CELL = 1.0
MAXIMUM_BASELINE_TO_CELL_FACTOR = 0.50
MAXIMUM_REPEAT_SPREAD_FACTOR = 0.20

# Mapping always requests the nominal pulse. A substantially different measured
# hold is outside the part of the response that this compact model establishes.
MINIMUM_RUNTIME_DURATION_FACTOR = 0.65
MAXIMUM_RUNTIME_DURATION_FACTOR = 1.50

# A short move is materially more dangerous than ordinary flow noise: treating
# it as a complete command can draw a path through an obstacle. Keep the
# under-travel gate tighter than the over-travel gate and fail closed rather
# than promoting either discrepancy to a complete map cell.
BASE_SHORTFALL_FACTOR = 0.12
MAXIMUM_SHORTFALL_FACTOR = 0.15
SHORTFALL_RMSE_FACTOR = 1.25
BASE_EXCESS_FACTOR = 0.12
MAXIMUM_EXCESS_FACTOR = 0.20
EXCESS_RMSE_FACTOR = 1.50
BASE_BLOCKED_MOTION_FACTOR = 0.04
MAXIMUM_BLOCKED_MOTION_FACTOR = 0.10
MINIMUM_FLOW_INLIER_RATIO = 0.60
MAXIMUM_DISPERSION_FACTOR = 0.65


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    return float(value)


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def forward_flow_coherence_error(
    *,
    reference_flow_px: float,
    dispersion_px: float,
    inlier_ratio: float,
) -> str | None:
    """Return why one flow field is unsafe, or ``None`` when it is coherent."""
    values = (reference_flow_px, dispersion_px, inlier_ratio)
    if any(not isfinite(value) for value in values):
        return "optical-flow coherence values are not finite"
    if reference_flow_px <= 0.0 or dispersion_px < 0.0:
        return "optical-flow magnitude or dispersion is invalid"
    if not 0.0 <= inlier_ratio <= 1.0:
        return "optical-flow inlier ratio is invalid"
    if inlier_ratio < MINIMUM_FLOW_INLIER_RATIO:
        return "optical flow is not coherent: too few consistent tracks"
    if dispersion_px > reference_flow_px * MAXIMUM_DISPERSION_FACTOR:
        return "optical flow is not coherent: vectors are too dispersed"
    return None


@dataclass(frozen=True)
class ForwardCalibrationTrial:
    """One settled forward pulse used to fit the fixed-camera travel scale."""

    requested_seconds: float
    actual_seconds: float
    distance_px: float
    confidence: float
    tracked_points: int
    dispersion_px: float
    inlier_ratio: float


@dataclass(frozen=True)
class ForwardObservationValidation:
    """Fail-closed validation of one calibrated forward observation."""

    reliable: bool
    blocked_candidate: bool
    distance_cells: float | None
    expected_flow_px: float | None
    observed_motion_px: float | None
    residual_px: float | None
    maximum_residual_px: float | None
    reason: str | None


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
        if self.dead_time_seconds > self.nominal_seconds * 0.90:
            raise ValueError("forward dead time must leave a measurable nominal pulse")
        if self.pixels_per_cell <= MINIMUM_FORWARD_PIXELS_PER_CELL:
            raise ValueError(
                "pixels_per_cell must exceed the minimum measurable travel"
            )
        if (
            self.baseline_flow_px
            > self.pixels_per_cell * MAXIMUM_BASELINE_TO_CELL_FACTOR
        ):
            raise ValueError(
                "forward baseline is too large relative to calibrated travel"
            )
        if self.rmse_px < 0.0:
            raise ValueError("rmse_px cannot be negative")
        if self.rmse_px / self.pixels_per_cell > MAXIMUM_FORWARD_RELATIVE_RMSE:
            raise ValueError("forward calibration RMSE is too large for its scale")
        if not MINIMUM_FORWARD_R_SQUARED <= self.r_squared <= 1.0:
            raise ValueError("forward calibration R-squared is outside safe bounds")
        if self.sample_count < MINIMUM_FORWARD_SAMPLES:
            raise ValueError(
                f"forward calibration needs at least {MINIMUM_FORWARD_SAMPLES} samples"
            )
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("forward calibration frame size must be positive")

        derived_pixels_per_cell = self.flow_rate_px_per_second * (
            self.nominal_seconds - self.dead_time_seconds
        )
        if not isclose(
            self.pixels_per_cell,
            derived_pixels_per_cell,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "pixels_per_cell is inconsistent with the fitted forward response"
            )

    def predicted_flow_px(self, actual_seconds: float) -> float:
        """Predict robust scene flow for one measured key hold."""
        duration = max(0.0, float(actual_seconds) - self.dead_time_seconds)
        return self.baseline_flow_px + self.flow_rate_px_per_second * duration

    def validate_observation(
        self,
        *,
        actual_seconds: float | None,
        distance_px: float,
        dispersion_px: float,
        inlier_ratio: float,
    ) -> ForwardObservationValidation:
        """
        Validate a runtime flow measurement against this calibrated response.

        A measurement is integrable only when the measured hold is close to the
        nominal mapping pulse, the flow agrees with the calibrated prediction,
        and tracked features describe one coherent scene translation. Low flow
        is reported separately so the tracker may classify a visually static
        scene as blocked without treating it as distance.
        """
        values = (distance_px, dispersion_px)
        if any(not isfinite(value) or value < 0.0 for value in values):
            return self._invalid_observation("forward flow values are invalid")
        if not isfinite(inlier_ratio) or not 0.0 <= inlier_ratio <= 1.0:
            return self._invalid_observation("forward flow inlier ratio is invalid")
        if actual_seconds is None or not isfinite(actual_seconds):
            return self._invalid_observation(
                "measured forward hold duration is unavailable"
            )
        actual_seconds = float(actual_seconds)
        minimum_duration = self.nominal_seconds * MINIMUM_RUNTIME_DURATION_FACTOR
        maximum_duration = self.nominal_seconds * MAXIMUM_RUNTIME_DURATION_FACTOR
        if not minimum_duration <= actual_seconds <= maximum_duration:
            return self._invalid_observation(
                "measured forward hold duration is outside the calibrated range"
            )

        expected_flow = self.predicted_flow_px(actual_seconds)
        expected_motion = max(0.0, expected_flow - self.baseline_flow_px)
        observed_motion = max(0.0, distance_px - self.baseline_flow_px)

        if expected_motion <= 0.0:
            return self._invalid_observation(
                "measured hold does not exceed calibrated forward dead time",
                expected_flow_px=expected_flow,
                observed_motion_px=observed_motion,
            )
        residual = abs(distance_px - expected_flow)
        shortfall = expected_flow - distance_px
        maximum_shortfall = min(
            expected_motion * MAXIMUM_SHORTFALL_FACTOR,
            max(
                expected_motion * BASE_SHORTFALL_FACTOR,
                self.rmse_px * SHORTFALL_RMSE_FACTOR,
            ),
        )
        excess = distance_px - expected_flow
        maximum_excess = min(
            expected_motion * MAXIMUM_EXCESS_FACTOR,
            max(
                expected_motion * BASE_EXCESS_FACTOR,
                self.rmse_px * EXCESS_RMSE_FACTOR,
            ),
        )
        coherence_error = forward_flow_coherence_error(
            reference_flow_px=expected_flow,
            dispersion_px=dispersion_px,
            inlier_ratio=inlier_ratio,
        )
        if coherence_error is not None:
            return self._invalid_observation(
                coherence_error,
                expected_flow_px=expected_flow,
                observed_motion_px=observed_motion,
                residual_px=residual,
                maximum_residual_px=(
                    maximum_shortfall if shortfall >= 0.0 else maximum_excess
                ),
            )
        if shortfall > maximum_shortfall:
            blocked_motion_ceiling = min(
                expected_motion * MAXIMUM_BLOCKED_MOTION_FACTOR,
                max(
                    expected_motion * BASE_BLOCKED_MOTION_FACTOR,
                    self.rmse_px,
                ),
            )
            blocked_candidate = observed_motion <= blocked_motion_ceiling
            return ForwardObservationValidation(
                reliable=False,
                blocked_candidate=blocked_candidate,
                distance_cells=None,
                expected_flow_px=expected_flow,
                observed_motion_px=observed_motion,
                residual_px=residual,
                maximum_residual_px=maximum_shortfall,
                reason=(
                    "observed flow is consistent with no forward travel"
                    if blocked_candidate
                    else (
                        "observed flow indicates partial forward movement; "
                        "the pose cannot be integrated safely"
                    )
                ),
            )
        if excess > maximum_excess:
            return self._invalid_observation(
                "observed flow exceeds the calibrated forward response",
                expected_flow_px=expected_flow,
                observed_motion_px=observed_motion,
                residual_px=residual,
                maximum_residual_px=maximum_excess,
            )

        # Optical flow is the validation signal, not the odometry ruler. The
        # same physical key hold produced highly texture-dependent flow in the
        # saved mapping logs. Once the observation agrees with calibration,
        # integrate the duration-derived relative travel instead of converting
        # that noisy per-frame magnitude directly into position.
        commanded_cells = expected_motion / self.pixels_per_cell
        return ForwardObservationValidation(
            reliable=True,
            blocked_candidate=False,
            distance_cells=commanded_cells,
            expected_flow_px=expected_flow,
            observed_motion_px=observed_motion,
            residual_px=residual,
            maximum_residual_px=(
                maximum_shortfall if shortfall >= 0.0 else maximum_excess
            ),
            reason=None,
        )

    @staticmethod
    def _invalid_observation(
        reason: str,
        *,
        expected_flow_px: float | None = None,
        observed_motion_px: float | None = None,
        residual_px: float | None = None,
        maximum_residual_px: float | None = None,
    ) -> ForwardObservationValidation:
        return ForwardObservationValidation(
            reliable=False,
            blocked_candidate=False,
            distance_cells=None,
            expected_flow_px=expected_flow_px,
            observed_motion_px=observed_motion_px,
            residual_px=residual_px,
            maximum_residual_px=maximum_residual_px,
            reason=reason,
        )

    def matches_frame(self, frame: np.ndarray | None) -> bool:
        """Reject use with a capture resolution different from calibration."""
        if frame is None or frame.ndim < 2:
            return False
        height, width = frame.shape[:2]
        return width == self.frame_width and height == self.frame_height

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ForwardMotionModel:
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
    maximum_relative_rmse: float = MAXIMUM_FORWARD_RELATIVE_RMSE,
    minimum_r_squared: float = MINIMUM_FORWARD_R_SQUARED,
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

    requested_groups = {round(trial.requested_seconds, 4) for trial in accepted}
    if len(requested_groups) < 3:
        raise ValueError(
            "Forward calibration requires at least three distinct pulse durations."
        )
    if any(
        sum(round(trial.requested_seconds, 4) == requested for trial in accepted) < 2
        for requested in requested_groups
    ):
        raise ValueError(
            "Every forward pulse duration must have at least two independent "
            "measurements."
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
    requested = np.asarray(
        [round(trial.requested_seconds, 4) for trial in accepted],
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

    if any(
        int(np.count_nonzero(inliers & np.isclose(requested, group))) < 2
        for group in requested_groups
    ):
        raise ValueError(
            "Robust forward fitting left a pulse duration without two "
            "repeatable measurements."
        )

    inlier_durations = durations[inliers]
    if not (
        float(np.min(inlier_durations))
        <= nominal_seconds
        <= float(np.max(inlier_durations))
    ):
        raise ValueError(
            "The nominal forward pulse must be bracketed by calibration trials."
        )

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
    if pixels_per_cell <= MINIMUM_FORWARD_PIXELS_PER_CELL:
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

    maximum_relative_rmse = min(
        float(maximum_relative_rmse),
        MAXIMUM_FORWARD_RELATIVE_RMSE,
    )
    if relative_rmse > maximum_relative_rmse:
        raise ValueError(
            "Forward calibration is too inconsistent "
            f"(relative RMSE {relative_rmse:.2f})."
        )
    minimum_r_squared = max(
        float(minimum_r_squared),
        MINIMUM_FORWARD_R_SQUARED,
    )
    if r_squared < minimum_r_squared:
        raise ValueError(
            "Forward travel is not sufficiently correlated with pulse duration "
            f"(R² {r_squared:.2f})."
        )

    maximum_repeat_spread = max(
        0.75,
        pixels_per_cell * MAXIMUM_REPEAT_SPREAD_FACTOR,
    )
    for group in requested_groups:
        group_distances = distances[inliers & np.isclose(requested, group)]
        repeat_spread = float(np.max(group_distances) - np.min(group_distances))
        if repeat_spread > maximum_repeat_spread:
            raise ValueError(
                "Repeated forward pulses were not visually repeatable "
                f"({group:.3f}s spread {repeat_spread:.2f}px, limit "
                f"{maximum_repeat_spread:.2f}px)."
            )

    model = ForwardMotionModel(
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
    for trial_index, (trial, retained) in enumerate(
        zip(accepted, inliers, strict=True),
        start=1,
    ):
        if not retained:
            continue
        validation = model.validate_observation(
            actual_seconds=trial.actual_seconds,
            distance_px=trial.distance_px,
            dispersion_px=trial.dispersion_px,
            inlier_ratio=trial.inlier_ratio,
        )
        if not validation.reliable:
            raise ValueError(
                "Fitted forward model rejects one of its retained supporting "
                f"trials ({trial_index}: {validation.reason})."
            )
    return model


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
        trial.dispersion_px,
        trial.inlier_ratio,
    )
    return (
        all(isfinite(value) for value in values)
        and trial.requested_seconds > 0.0
        and trial.actual_seconds > 0.0
        and 0.5 <= trial.actual_seconds / trial.requested_seconds <= 1.5
        and trial.distance_px > 0.5
        and trial.confidence >= minimum_confidence
        and trial.tracked_points >= 8
        and forward_flow_coherence_error(
            reference_flow_px=trial.distance_px,
            dispersion_px=trial.dispersion_px,
            inlier_ratio=trial.inlier_ratio,
        )
        is None
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
