from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from statistics import median
from typing import cast


class TurnDirection(StrEnum):
    LEFT = "left"
    RIGHT = "right"

    @property
    def opposite(self) -> TurnDirection:
        if self is TurnDirection.LEFT:
            return TurnDirection.RIGHT
        return TurnDirection.LEFT


class TurnTransition(StrEnum):
    """The rotation state immediately before a turn pulse."""

    NEUTRAL = "neutral"
    SAME_DIRECTION = "same_direction"
    REVERSAL = "reversal"


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    return float(value)


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _as_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


@dataclass(frozen=True)
class RotationSample:
    """One observed turn pulse and its settled minimap displacement."""

    direction: TurnDirection
    transition: TurnTransition
    requested_seconds: float
    clamped_seconds: float
    held_seconds: float
    measured_degrees: float
    confidence: float
    idle_seconds: float | None = None


@dataclass(frozen=True)
class TurnPulseResult:
    """Timing and state recorded for one issued rotation pulse."""

    direction: TurnDirection
    transition: TurnTransition
    requested_seconds: float
    clamped_seconds: float
    held_seconds: float
    elapsed_seconds: float
    idle_seconds: float | None

    def as_sample(
        self,
        *,
        measured_degrees: float,
        confidence: float,
    ) -> RotationSample:
        return RotationSample(
            direction=self.direction,
            transition=self.transition,
            requested_seconds=self.requested_seconds,
            clamped_seconds=self.clamped_seconds,
            held_seconds=self.held_seconds,
            measured_degrees=float(measured_degrees),
            confidence=float(confidence),
            idle_seconds=self.idle_seconds,
        )


class TurnTransitionTracker:
    """
    Classify a pulse from the previous completed turn.

    Releasing a key does not erase direction history. The game can retain a
    direction-dependent response after release, so history becomes neutral
    only after a configured idle period or an explicit reset.
    """

    def __init__(self, neutral_after_seconds: float = 2.0) -> None:
        neutral_after_seconds = float(neutral_after_seconds)
        if not isfinite(neutral_after_seconds) or neutral_after_seconds <= 0.0:
            raise ValueError("neutral_after_seconds must be finite and positive")
        self.neutral_after_seconds: float = neutral_after_seconds
        self._last_direction: TurnDirection | None = None
        self._last_completed_at: float | None = None

    @property
    def last_direction(self) -> TurnDirection | None:
        return self._last_direction

    def set_neutral_after_seconds(self, seconds: float) -> None:
        seconds = float(seconds)
        if not isfinite(seconds) or seconds <= 0.0:
            raise ValueError("neutral_after_seconds must be finite and positive")
        self.neutral_after_seconds = seconds

    def idle_seconds(self, *, now: float) -> float | None:
        if not isfinite(now):
            raise ValueError("now must be finite")
        if self._last_completed_at is None:
            return None
        return max(0.0, float(now) - self._last_completed_at)

    def classify(
        self,
        direction: TurnDirection,
        *,
        now: float,
    ) -> tuple[TurnTransition, float | None]:
        if not isfinite(now):
            raise ValueError("now must be finite")

        previous = self._last_direction
        completed_at = self._last_completed_at
        if previous is None or completed_at is None:
            return TurnTransition.NEUTRAL, None

        idle_seconds = self.idle_seconds(now=now)
        assert idle_seconds is not None
        if idle_seconds >= self.neutral_after_seconds:
            return TurnTransition.NEUTRAL, idle_seconds
        if previous is direction:
            return TurnTransition.SAME_DIRECTION, idle_seconds
        return TurnTransition.REVERSAL, idle_seconds

    def record(self, direction: TurnDirection, *, completed_at: float) -> None:
        if not isfinite(completed_at):
            raise ValueError("completed_at must be finite")
        self._last_direction = direction
        self._last_completed_at = float(completed_at)

    def reset(self) -> None:
        self._last_direction = None
        self._last_completed_at = None


@dataclass(frozen=True)
class RotationTiming:
    """Fitted response for one direction/transition combination."""

    rate_degrees_per_second: float
    dead_time_seconds: float
    sample_count: int
    median_error_degrees: float
    is_fallback: bool = False

    def __post_init__(self) -> None:
        if (
            not isfinite(self.rate_degrees_per_second)
            or self.rate_degrees_per_second <= 0.0
        ):
            raise ValueError("rotation rate must be finite and positive")
        if not isfinite(self.dead_time_seconds) or self.dead_time_seconds < 0.0:
            raise ValueError("rotation dead time must be finite and non-negative")
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if not isfinite(self.median_error_degrees) or self.median_error_degrees < 0.0:
            raise ValueError("median_error_degrees must be finite and non-negative")

    @property
    def seconds_90(self) -> float:
        return self.seconds_for(90.0)

    def seconds_for(self, degrees: float) -> float:
        degrees = float(degrees)
        if not isfinite(degrees):
            raise ValueError("degrees must be finite")
        if degrees <= 0.0:
            return 0.0
        return self.dead_time_seconds + degrees / self.rate_degrees_per_second

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "rate_degrees_per_second": round(
                self.rate_degrees_per_second,
                6,
            ),
            "dead_time_seconds": round(self.dead_time_seconds, 6),
            "seconds_90": round(self.seconds_90, 6),
            "sample_count": self.sample_count,
            "median_error_degrees": round(self.median_error_degrees, 6),
            "is_fallback": self.is_fallback,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RotationTiming:
        rate = _as_float(
            data["rate_degrees_per_second"],
            "rate_degrees_per_second",
        )
        dead_time = _as_float(
            data.get("dead_time_seconds", 0.0),
            "dead_time_seconds",
        )
        if not isfinite(rate) or rate <= 0.0:
            raise ValueError("rotation rate must be finite and positive")
        if not isfinite(dead_time) or dead_time < 0.0:
            raise ValueError("rotation dead time must be finite and non-negative")
        return cls(
            rate_degrees_per_second=rate,
            dead_time_seconds=dead_time,
            sample_count=_as_int(
                data.get("sample_count", 0),
                "sample_count",
            ),
            median_error_degrees=_as_float(
                data.get("median_error_degrees", 0.0),
                "median_error_degrees",
            ),
            is_fallback=_as_bool(
                data.get("is_fallback", False),
                "is_fallback",
            ),
        )


@dataclass(frozen=True)
class DirectionRotationProfile:
    neutral: RotationTiming
    same_direction: RotationTiming
    reversal: RotationTiming

    def timing_for(self, transition: TurnTransition) -> RotationTiming:
        if transition is TurnTransition.NEUTRAL:
            return self.neutral
        if transition is TurnTransition.SAME_DIRECTION:
            return self.same_direction
        return self.reversal

    def to_dict(self) -> dict[str, dict[str, float | int | bool]]:
        return {
            TurnTransition.NEUTRAL.value: self.neutral.to_dict(),
            TurnTransition.SAME_DIRECTION.value: self.same_direction.to_dict(),
            TurnTransition.REVERSAL.value: self.reversal.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> DirectionRotationProfile:
        def timing(name: TurnTransition) -> RotationTiming:
            value = data[name.value]
            if not isinstance(value, dict):
                raise TypeError(f"{name.value} rotation timing must be an object")
            return RotationTiming.from_dict(cast(dict[str, object], value))

        return cls(
            neutral=timing(TurnTransition.NEUTRAL),
            same_direction=timing(TurnTransition.SAME_DIRECTION),
            reversal=timing(TurnTransition.REVERSAL),
        )


@dataclass(frozen=True)
class DirectionIdleResponseCurve:
    """
    Monotonic progress from retained turn state to the neutral response.

    ``response_progress`` is zero for a fully state-dependent pulse and one
    for a neutral pulse. The curve is fitted from fixed-duration reversal
    probes, then used to blend either SAME_DIRECTION or REVERSAL timing toward
    the neutral timing without waiting at runtime.
    """

    idle_seconds: tuple[float, ...]
    response_progress: tuple[float, ...]
    source_sample_count: int
    stateful_response_degrees: float
    neutral_response_degrees: float
    maximum_monotonic_adjustment_degrees: float

    def __post_init__(self) -> None:
        if len(self.idle_seconds) != len(self.response_progress):
            raise ValueError("idle-response curve axes must have the same length")
        if len(self.idle_seconds) < 3:
            raise ValueError("idle-response curve requires at least three knots")
        if self.source_sample_count < 3:
            raise ValueError("idle-response curve requires at least three samples")
        if any(
            not isfinite(value)
            for value in (
                *self.idle_seconds,
                *self.response_progress,
                self.stateful_response_degrees,
                self.neutral_response_degrees,
                self.maximum_monotonic_adjustment_degrees,
            )
        ):
            raise ValueError("idle-response curve values must be finite")
        if self.idle_seconds[0] != 0.0:
            raise ValueError("idle-response curve must start at zero seconds")
        if self.response_progress[0] != 0.0:
            raise ValueError("idle-response curve must start at zero progress")
        if self.response_progress[-1] != 1.0:
            raise ValueError("idle-response curve must end at neutral progress")
        if any(
            second <= first
            for first, second in zip(
                self.idle_seconds,
                self.idle_seconds[1:],
            )
        ):
            raise ValueError("idle-response curve delays must be strictly increasing")
        if any(progress < 0.0 or progress > 1.0 for progress in self.response_progress):
            raise ValueError("idle-response progress must stay in [0, 1]")
        if any(
            second < first
            for first, second in zip(
                self.response_progress,
                self.response_progress[1:],
            )
        ):
            raise ValueError("idle-response progress must be monotonic")
        if self.stateful_response_degrees <= 0.0:
            raise ValueError("stateful response must be positive")
        if self.neutral_response_degrees <= self.stateful_response_degrees:
            raise ValueError("neutral response must exceed the stateful response")
        if self.maximum_monotonic_adjustment_degrees < 0.0:
            raise ValueError("monotonic adjustment must be non-negative")

    @property
    def neutral_after_seconds(self) -> float:
        return self.idle_seconds[-1]

    def progress_at(self, idle_seconds: float) -> float:
        idle_seconds = float(idle_seconds)
        if not isfinite(idle_seconds):
            raise ValueError("idle_seconds must be finite")
        if idle_seconds <= 0.0:
            return 0.0
        if idle_seconds >= self.idle_seconds[-1]:
            return 1.0

        for index in range(1, len(self.idle_seconds)):
            upper_idle = self.idle_seconds[index]
            if idle_seconds > upper_idle:
                continue
            lower_idle = self.idle_seconds[index - 1]
            lower_progress = self.response_progress[index - 1]
            upper_progress = self.response_progress[index]
            fraction = (idle_seconds - lower_idle) / (upper_idle - lower_idle)
            return lower_progress + fraction * (upper_progress - lower_progress)

        raise AssertionError("validated idle-response curve has no matching segment")

    def to_dict(self) -> dict[str, object]:
        return {
            "idle_seconds": [round(value, 6) for value in self.idle_seconds],
            "response_progress": [round(value, 6) for value in self.response_progress],
            "source_sample_count": self.source_sample_count,
            "stateful_response_degrees": round(
                self.stateful_response_degrees,
                6,
            ),
            "neutral_response_degrees": round(
                self.neutral_response_degrees,
                6,
            ),
            "maximum_monotonic_adjustment_degrees": round(
                self.maximum_monotonic_adjustment_degrees,
                6,
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> DirectionIdleResponseCurve:
        idle_values = data["idle_seconds"]
        progress_values = data["response_progress"]
        if not isinstance(idle_values, list) or not isinstance(progress_values, list):
            raise TypeError("idle-response curve axes must be arrays")
        idle_values = cast(list[object], idle_values)
        progress_values = cast(list[object], progress_values)
        return cls(
            idle_seconds=tuple(
                _as_float(value, "idle_seconds") for value in idle_values
            ),
            response_progress=tuple(
                _as_float(value, "response_progress") for value in progress_values
            ),
            source_sample_count=_as_int(
                data["source_sample_count"],
                "source_sample_count",
            ),
            stateful_response_degrees=_as_float(
                data["stateful_response_degrees"],
                "stateful_response_degrees",
            ),
            neutral_response_degrees=_as_float(
                data["neutral_response_degrees"],
                "neutral_response_degrees",
            ),
            maximum_monotonic_adjustment_degrees=_as_float(
                data["maximum_monotonic_adjustment_degrees"],
                "maximum_monotonic_adjustment_degrees",
            ),
        )


@dataclass(frozen=True)
class IdleResponseCurves:
    """Validated idle-response progress for both requested turn directions."""

    left: DirectionIdleResponseCurve
    right: DirectionIdleResponseCurve

    def curve_for(self, direction: TurnDirection) -> DirectionIdleResponseCurve:
        if direction is TurnDirection.LEFT:
            return self.left
        return self.right

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            TurnDirection.LEFT.value: self.left.to_dict(),
            TurnDirection.RIGHT.value: self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> IdleResponseCurves:
        version = _as_int(data["version"], "idle_response_curves.version")
        if version != 1:
            raise ValueError(
                f"unsupported idle-response curve version {version}; expected 1"
            )

        def curve(direction: TurnDirection) -> DirectionIdleResponseCurve:
            value = data[direction.value]
            if not isinstance(value, dict):
                raise TypeError(
                    f"{direction.value} idle-response curve must be an object"
                )
            return DirectionIdleResponseCurve.from_dict(cast(dict[str, object], value))

        return cls(
            left=curve(TurnDirection.LEFT),
            right=curve(TurnDirection.RIGHT),
        )


@dataclass(frozen=True)
class StateAwareRotationModel:
    """Calibrated pulse response split by direction and transition state."""

    left: DirectionRotationProfile
    right: DirectionRotationProfile
    neutral_after_seconds: float
    idle_response_curves: IdleResponseCurves | None = None

    def __post_init__(self) -> None:
        if (
            not isfinite(self.neutral_after_seconds)
            or self.neutral_after_seconds <= 0.0
        ):
            raise ValueError("neutral_after_seconds must be finite and positive")
        if self.idle_response_curves is not None:
            for direction in TurnDirection:
                curve = self.idle_response_curves.curve_for(direction)
                if abs(curve.neutral_after_seconds - self.neutral_after_seconds) > 1e-6:
                    raise ValueError(
                        f"{direction.value} idle-response curve endpoint must "
                        + "match neutral_after_seconds"
                    )

    def profile_for(self, direction: TurnDirection) -> DirectionRotationProfile:
        if direction is TurnDirection.LEFT:
            return self.left
        return self.right

    def timing_for(
        self,
        direction: TurnDirection,
        transition: TurnTransition,
    ) -> RotationTiming:
        return self.profile_for(direction).timing_for(transition)

    def seconds_for(
        self,
        direction: TurnDirection,
        transition: TurnTransition,
        degrees: float,
        *,
        idle_seconds: float | None = None,
    ) -> float:
        state_seconds = self.timing_for(direction, transition).seconds_for(degrees)
        if (
            transition is TurnTransition.NEUTRAL
            or idle_seconds is None
            or self.idle_response_curves is None
        ):
            return state_seconds

        idle_seconds = float(idle_seconds)
        if not isfinite(idle_seconds) or idle_seconds < 0.0:
            raise ValueError("idle_seconds must be finite and non-negative")
        progress = self.idle_response_curves.curve_for(direction).progress_at(
            idle_seconds
        )
        neutral_seconds = self.profile_for(direction).neutral.seconds_for(degrees)
        return state_seconds + progress * (neutral_seconds - state_seconds)

    def seconds_for_degrees(
        self,
        direction: TurnDirection | str,
        degrees: float,
        previous_direction: TurnDirection | str | None,
        *,
        idle_seconds: float | None = None,
    ) -> float:
        """
        Return a state-aware duration for callers that track only direction.

        When prior state is unknown, use the longest fitted transition for the
        requested angle. That avoids applying the faster same-direction timing
        to a possible reversal, which is the observed under-turn failure mode.
        """

        selected = TurnDirection(direction)
        profile = self.profile_for(selected)
        if previous_direction is None:
            return max(
                profile.neutral.seconds_for(degrees),
                profile.same_direction.seconds_for(degrees),
                profile.reversal.seconds_for(degrees),
            )

        previous = TurnDirection(previous_direction)
        transition = (
            TurnTransition.SAME_DIRECTION
            if previous is selected
            else TurnTransition.REVERSAL
        )
        return self.seconds_for(
            selected,
            transition,
            degrees,
            idle_seconds=idle_seconds,
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "version": 2,
            "neutral_after_seconds": round(self.neutral_after_seconds, 6),
            TurnDirection.LEFT.value: self.left.to_dict(),
            TurnDirection.RIGHT.value: self.right.to_dict(),
        }
        if self.idle_response_curves is not None:
            data["idle_response_curves"] = self.idle_response_curves.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> StateAwareRotationModel:
        version = _as_int(data["version"], "rotation_model.version")
        if version != 2:
            raise ValueError(
                f"unsupported rotation model version {version}; expected 2"
            )

        def profile(name: TurnDirection) -> DirectionRotationProfile:
            value = data[name.value]
            if not isinstance(value, dict):
                raise TypeError(f"{name.value} rotation profile must be an object")
            return DirectionRotationProfile.from_dict(cast(dict[str, object], value))

        curve_data = data["idle_response_curves"]
        if not isinstance(curve_data, dict):
            raise TypeError("idle_response_curves must be an object")
        neutral_after_seconds = _as_float(
            data["neutral_after_seconds"],
            "neutral_after_seconds",
        )
        if not isfinite(neutral_after_seconds) or neutral_after_seconds <= 0.0:
            raise ValueError("neutral_after_seconds must be finite and positive")
        return cls(
            left=profile(TurnDirection.LEFT),
            right=profile(TurnDirection.RIGHT),
            neutral_after_seconds=neutral_after_seconds,
            idle_response_curves=IdleResponseCurves.from_dict(
                cast(dict[str, object], curve_data)
            ),
        )


@dataclass(frozen=True)
class NeutralTimeoutSample:
    """One fixed-size turn pulse issued after a measured idle delay."""

    direction: TurnDirection
    requested_idle_seconds: float
    observed_idle_seconds: float
    measured_degrees: float
    uncertainty_degrees: float
    confidence: float
    requested_seconds: float = 1.0
    clamped_seconds: float = 1.0
    held_seconds: float = 1.0

    def as_neutral_rotation_sample(self) -> RotationSample:
        """Reuse a physically neutral timeout probe in the rotation fit."""
        return RotationSample(
            direction=self.direction,
            transition=TurnTransition.NEUTRAL,
            requested_seconds=self.requested_seconds,
            clamped_seconds=self.clamped_seconds,
            held_seconds=self.held_seconds,
            measured_degrees=self.measured_degrees,
            confidence=self.confidence,
            idle_seconds=self.observed_idle_seconds,
        )


@dataclass(frozen=True)
class DirectionNeutralTimeoutFit:
    """Validated change-point evidence for one requested turn direction."""

    last_stateful_seconds: float
    first_neutral_seconds: float
    neutral_response_degrees: float
    neutral_tolerance_degrees: float
    neutral_uncertainty_degrees: float
    stateful_sample_count: int
    neutral_sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "last_stateful_seconds": round(self.last_stateful_seconds, 6),
            "first_neutral_seconds": round(self.first_neutral_seconds, 6),
            "neutral_response_degrees": round(
                self.neutral_response_degrees,
                6,
            ),
            "neutral_tolerance_degrees": round(
                self.neutral_tolerance_degrees,
                6,
            ),
            "neutral_uncertainty_degrees": round(
                self.neutral_uncertainty_degrees,
                6,
            ),
            "stateful_sample_count": self.stateful_sample_count,
            "neutral_sample_count": self.neutral_sample_count,
        }


@dataclass(frozen=True)
class NeutralTimeoutFit:
    """Conservative cross-direction timeout selected from bounded probes."""

    neutral_after_seconds: float
    safety_margin_seconds: float
    left: DirectionNeutralTimeoutFit
    right: DirectionNeutralTimeoutFit
    idle_response_curves: IdleResponseCurves

    def fit_for(
        self,
        direction: TurnDirection,
    ) -> DirectionNeutralTimeoutFit:
        if direction is TurnDirection.LEFT:
            return self.left
        return self.right

    def to_dict(self) -> dict[str, object]:
        return {
            "neutral_after_seconds": round(
                self.neutral_after_seconds,
                6,
            ),
            "safety_margin_seconds": round(
                self.safety_margin_seconds,
                6,
            ),
            TurnDirection.LEFT.value: self.left.to_dict(),
            TurnDirection.RIGHT.value: self.right.to_dict(),
            "idle_response_curves": self.idle_response_curves.to_dict(),
        }


def fit_neutral_timeout(
    samples: Iterable[NeutralTimeoutSample],
    *,
    safety_margin_seconds: float = 0.25,
    maximum_idle_seconds: float = 6.0,
    minimum_confidence: float = 0.52,
    minimum_samples_per_direction: int = 6,
    neutral_tail_samples: int = 3,
    minimum_stateful_samples: int = 2,
    minimum_neutral_tolerance_degrees: float = 1.5,
    maximum_neutral_tail_spread_degrees: float = 2.5,
    maximum_sample_uncertainty_degrees: float = 3.0,
    minimum_response_span_degrees: float = 3.0,
    minimum_monotonic_tolerance_degrees: float = 1.0,
) -> NeutralTimeoutFit:
    """
    Find the reversal-to-neutral response change point in both directions.

    Samples must form one conservative pattern per direction: at least two
    confidently stateful responses followed by at least three mutually
    consistent neutral responses. Ambiguous or non-monotonic scans are rejected
    instead of silently assigning a timeout.
    """

    safety_margin_seconds = float(safety_margin_seconds)
    maximum_idle_seconds = float(maximum_idle_seconds)
    if not isfinite(safety_margin_seconds) or safety_margin_seconds <= 0.0:
        raise ValueError("neutral timeout safety margin must be finite and positive")
    if not isfinite(maximum_idle_seconds) or maximum_idle_seconds <= 0.0:
        raise ValueError("maximum idle delay must be finite and positive")
    if neutral_tail_samples < 3:
        raise ValueError("at least three neutral tail samples are required")
    if minimum_stateful_samples < 2:
        raise ValueError("at least two stateful samples are required")
    if (
        not isfinite(minimum_response_span_degrees)
        or minimum_response_span_degrees <= 0.0
    ):
        raise ValueError("minimum response span must be finite and positive")
    if (
        not isfinite(minimum_monotonic_tolerance_degrees)
        or minimum_monotonic_tolerance_degrees < 0.0
    ):
        raise ValueError("minimum monotonic tolerance must be finite and non-negative")

    grouped: dict[TurnDirection, list[NeutralTimeoutSample]] = defaultdict(list)
    for sample in samples:
        if _neutral_timeout_sample_is_valid(
            sample,
            minimum_confidence=minimum_confidence,
        ):
            grouped[sample.direction].append(sample)

    fits: dict[TurnDirection, DirectionNeutralTimeoutFit] = {}
    for direction in TurnDirection:
        ordered = sorted(
            grouped[direction],
            key=lambda sample: sample.observed_idle_seconds,
        )
        if len(ordered) < minimum_samples_per_direction:
            raise ValueError(
                f"{direction.value} neutral-timeout scan has only "
                + f"{len(ordered)} valid samples; "
                + f"{minimum_samples_per_direction} are required"
            )
        if len({round(sample.observed_idle_seconds, 3) for sample in ordered}) != len(
            ordered
        ):
            raise ValueError(
                f"{direction.value} neutral-timeout scan has duplicate idle delays"
            )

        tail = ordered[-neutral_tail_samples:]
        tail_spread = max(sample.measured_degrees for sample in tail) - min(
            sample.measured_degrees for sample in tail
        )
        if tail_spread > maximum_neutral_tail_spread_degrees:
            raise ValueError(
                f"{direction.value} neutral tail spans {tail_spread:.2f} "
                + "degrees; the bounded scan did not reach a stable plateau"
            )
        if any(
            sample.uncertainty_degrees > maximum_sample_uncertainty_degrees
            for sample in ordered
        ):
            raise ValueError(
                f"{direction.value} neutral-timeout scan contains heading "
                + "uncertainty too large to resolve the state change"
            )
        neutral_response = float(median(sample.measured_degrees for sample in tail))
        neutral_uncertainty = float(
            median(sample.uncertainty_degrees for sample in tail)
        )
        neutral_tolerance = max(
            float(minimum_neutral_tolerance_degrees),
            max(abs(sample.measured_degrees - neutral_response) for sample in tail),
        )
        neutral_tolerance = min(
            neutral_tolerance,
            float(maximum_neutral_tail_spread_degrees),
        )

        classifications: list[str] = []
        for sample in ordered:
            difference = sample.measured_degrees - neutral_response
            allowed = neutral_tolerance + max(
                sample.uncertainty_degrees,
                neutral_uncertainty,
            )
            if abs(difference) <= allowed:
                classifications.append("neutral")
            elif difference < -allowed:
                classifications.append("stateful")
            else:
                raise ValueError(
                    f"{direction.value} idle scan exceeded its neutral response "
                    + f"at {sample.observed_idle_seconds:.3f}s; the timeout "
                    + "change point is not reliable"
                )

        try:
            first_neutral_index = classifications.index("neutral")
        except ValueError as error:
            raise ValueError(
                f"{direction.value} idle scan never reached a stable neutral response"
            ) from error

        if any(
            state != "stateful" for state in classifications[:first_neutral_index]
        ) or any(state != "neutral" for state in classifications[first_neutral_index:]):
            raise ValueError(
                f"{direction.value} idle scan was non-monotonic; "
                + "the timeout was not fitted"
            )
        if first_neutral_index < minimum_stateful_samples:
            raise ValueError(
                f"{direction.value} idle scan did not establish enough "
                + "state-dependent response before neutrality"
            )
        neutral_count = len(ordered) - first_neutral_index
        if neutral_count < neutral_tail_samples:
            raise ValueError(
                f"{direction.value} idle scan did not validate a long enough "
                + "neutral tail"
            )

        first_neutral = ordered[first_neutral_index].observed_idle_seconds
        if first_neutral >= maximum_idle_seconds:
            raise ValueError(
                f"{direction.value} first neutral response was not reached "
                + "inside the bounded probe window"
            )

        fits[direction] = DirectionNeutralTimeoutFit(
            last_stateful_seconds=ordered[
                first_neutral_index - 1
            ].observed_idle_seconds,
            first_neutral_seconds=first_neutral,
            neutral_response_degrees=neutral_response,
            neutral_tolerance_degrees=neutral_tolerance,
            neutral_uncertainty_degrees=neutral_uncertainty,
            stateful_sample_count=first_neutral_index,
            neutral_sample_count=neutral_count,
        )

    neutral_after_seconds = (
        max(fit.first_neutral_seconds for fit in fits.values()) + safety_margin_seconds
    )
    if neutral_after_seconds > maximum_idle_seconds:
        raise ValueError(
            "the conservative neutral timeout exceeds the bounded probe window"
        )

    curves = IdleResponseCurves(
        left=_fit_direction_idle_response_curve(
            grouped[TurnDirection.LEFT],
            direction_fit=fits[TurnDirection.LEFT],
            neutral_after_seconds=neutral_after_seconds,
            minimum_response_span_degrees=minimum_response_span_degrees,
            minimum_monotonic_tolerance_degrees=(minimum_monotonic_tolerance_degrees),
        ),
        right=_fit_direction_idle_response_curve(
            grouped[TurnDirection.RIGHT],
            direction_fit=fits[TurnDirection.RIGHT],
            neutral_after_seconds=neutral_after_seconds,
            minimum_response_span_degrees=minimum_response_span_degrees,
            minimum_monotonic_tolerance_degrees=(minimum_monotonic_tolerance_degrees),
        ),
    )

    return NeutralTimeoutFit(
        neutral_after_seconds=neutral_after_seconds,
        safety_margin_seconds=safety_margin_seconds,
        left=fits[TurnDirection.LEFT],
        right=fits[TurnDirection.RIGHT],
        idle_response_curves=curves,
    )


def _fit_direction_idle_response_curve(
    samples: list[NeutralTimeoutSample],
    *,
    direction_fit: DirectionNeutralTimeoutFit,
    neutral_after_seconds: float,
    minimum_response_span_degrees: float,
    minimum_monotonic_tolerance_degrees: float,
) -> DirectionIdleResponseCurve:
    """
    Fit a bounded monotonic piecewise-linear response-progress curve.

    Small downward differences inside the strict heading uncertainty envelope
    are flattened. A decrease larger than that envelope, or a curve requiring
    a large monotonic correction, rejects the calibration.
    """
    ordered = sorted(samples, key=lambda sample: sample.observed_idle_seconds)
    if len(ordered) < 3:
        raise ValueError("idle-response curve requires at least three samples")

    reference_held_seconds = float(median(sample.held_seconds for sample in ordered))
    normalized_responses = [
        sample.measured_degrees * reference_held_seconds / sample.held_seconds
        for sample in ordered
    ]
    stateful_response = float(normalized_responses[0])
    neutral_response = float(median(normalized_responses[-3:]))
    response_span = neutral_response - stateful_response
    required_span = max(
        float(minimum_response_span_degrees),
        ordered[0].uncertainty_degrees + direction_fit.neutral_uncertainty_degrees,
    )
    if response_span <= required_span:
        raise ValueError(
            f"{ordered[0].direction.value} idle-response span is only "
            + f"{response_span:.2f} degrees; it cannot resolve continuous "
            + "turn-state progress"
        )

    peak_index = 0
    for index, sample in enumerate(ordered[1:], start=1):
        peak = ordered[peak_index]
        allowed_drop = max(
            float(minimum_monotonic_tolerance_degrees),
            peak.uncertainty_degrees,
            sample.uncertainty_degrees,
        )
        drop = normalized_responses[peak_index] - normalized_responses[index]
        if drop > allowed_drop:
            raise ValueError(
                f"{sample.direction.value} idle response decreased by "
                + f"{drop:.2f} degrees at {sample.observed_idle_seconds:.3f}s; "
                + "the response-progress curve is non-monotonic"
            )
        if normalized_responses[index] > normalized_responses[peak_index]:
            peak_index = index

    knot_idle: list[float] = [0.0]
    knot_progress: list[float] = [0.0]
    maximum_adjustment = 0.0
    previous_progress = 0.0
    for sample, response in zip(ordered, normalized_responses, strict=True):
        if sample.observed_idle_seconds >= neutral_after_seconds:
            continue
        raw_progress = min(
            1.0,
            max(
                0.0,
                (response - stateful_response) / response_span,
            ),
        )
        progress = max(previous_progress, raw_progress)
        maximum_adjustment = max(
            maximum_adjustment,
            (progress - raw_progress) * response_span,
        )
        knot_idle.append(float(sample.observed_idle_seconds))
        knot_progress.append(float(progress))
        previous_progress = progress

    maximum_allowed_adjustment = max(
        float(minimum_monotonic_tolerance_degrees),
        float(median(sample.uncertainty_degrees for sample in ordered)),
    )
    if maximum_adjustment > maximum_allowed_adjustment:
        raise ValueError(
            f"{ordered[0].direction.value} idle-response curve required a "
            + f"{maximum_adjustment:.2f}-degree monotonic correction; the scan "
            + "is too noisy"
        )

    endpoint = float(neutral_after_seconds)
    if knot_idle[-1] >= endpoint:
        raise AssertionError("idle-response endpoint must follow measured knots")
    knot_idle.append(endpoint)
    knot_progress.append(1.0)

    distinct_progress = {round(progress, 3) for progress in knot_progress}
    if len(distinct_progress) < 3:
        raise ValueError(
            f"{ordered[0].direction.value} idle-response scan did not resolve "
            + "an intermediate response state"
        )

    return DirectionIdleResponseCurve(
        idle_seconds=tuple(knot_idle),
        response_progress=tuple(knot_progress),
        source_sample_count=len(ordered),
        stateful_response_degrees=stateful_response,
        neutral_response_degrees=neutral_response,
        maximum_monotonic_adjustment_degrees=maximum_adjustment,
    )


def validate_neutral_timeout(
    fit: NeutralTimeoutFit,
    samples: Iterable[NeutralTimeoutSample],
    *,
    minimum_samples_per_direction: int = 2,
    maximum_idle_overshoot_seconds: float = 0.18,
    minimum_confidence: float = 0.52,
    maximum_sample_uncertainty_degrees: float = 3.0,
) -> None:
    """Require repeated neutral-like responses at the fitted threshold."""

    grouped: dict[TurnDirection, list[NeutralTimeoutSample]] = defaultdict(list)
    for sample in samples:
        if _neutral_timeout_sample_is_valid(
            sample,
            minimum_confidence=minimum_confidence,
        ):
            grouped[sample.direction].append(sample)

    for direction in TurnDirection:
        direction_samples = grouped[direction]
        if len(direction_samples) < minimum_samples_per_direction:
            raise ValueError(
                f"{direction.value} neutral-timeout validation has only "
                + f"{len(direction_samples)} valid samples; "
                + f"{minimum_samples_per_direction} are required"
            )
        if any(
            sample.uncertainty_degrees > maximum_sample_uncertainty_degrees
            for sample in direction_samples
        ):
            raise ValueError(
                f"{direction.value} neutral-timeout validation contains "
                + "heading uncertainty too large to confirm the threshold"
            )
        direction_fit = fit.fit_for(direction)
        for sample in direction_samples:
            if abs(sample.requested_idle_seconds - fit.neutral_after_seconds) > 0.02:
                raise ValueError(
                    f"{direction.value} validation was not requested at the "
                    + "fitted neutral timeout"
                )
            overshoot = sample.observed_idle_seconds - fit.neutral_after_seconds
            if overshoot < -0.01 or overshoot > maximum_idle_overshoot_seconds:
                raise ValueError(
                    f"{direction.value} validation pulse missed the fitted "
                    + f"idle delay by {overshoot:+.3f}s"
                )
            allowed = direction_fit.neutral_tolerance_degrees + max(
                sample.uncertainty_degrees,
                direction_fit.neutral_uncertainty_degrees,
            )
            if (
                abs(sample.measured_degrees - direction_fit.neutral_response_degrees)
                > allowed
            ):
                raise ValueError(
                    f"{direction.value} response at the fitted timeout did not "
                    + "match the validated neutral response"
                )


def _neutral_timeout_sample_is_valid(
    sample: NeutralTimeoutSample,
    *,
    minimum_confidence: float,
) -> bool:
    values = (
        sample.requested_idle_seconds,
        sample.observed_idle_seconds,
        sample.measured_degrees,
        sample.uncertainty_degrees,
        sample.confidence,
        sample.requested_seconds,
        sample.clamped_seconds,
        sample.held_seconds,
    )
    return (
        all(isfinite(value) for value in values)
        and sample.requested_idle_seconds > 0.0
        and sample.observed_idle_seconds > 0.0
        and sample.measured_degrees > 0.0
        and sample.uncertainty_degrees >= 0.0
        and sample.confidence >= minimum_confidence
        and sample.requested_seconds > 0.0
        and sample.clamped_seconds > 0.0
        and sample.held_seconds > 0.0
    )


def fit_rotation_model(
    samples: Iterable[RotationSample],
    *,
    fallback_left_seconds_90: float,
    fallback_right_seconds_90: float,
    neutral_after_seconds: float = 2.0,
    idle_response_curves: IdleResponseCurves | None = None,
    minimum_confidence: float = 0.52,
) -> StateAwareRotationModel:
    """
    Robustly fit ``angle = rate * max(0, held - dead_time)``.

    Theil-Sen slopes make the fit resistant to isolated heading outliers.
    Sparse transition groups fall back to the robust aggregate for that
    direction, while retaining a sample count and an explicit fallback flag.
    """

    valid_samples = [
        sample
        for sample in samples
        if _sample_is_valid(sample, minimum_confidence=minimum_confidence)
    ]
    grouped: dict[
        tuple[TurnDirection, TurnTransition],
        list[RotationSample],
    ] = defaultdict(list)
    by_direction: dict[TurnDirection, list[RotationSample]] = defaultdict(list)
    for sample in valid_samples:
        grouped[(sample.direction, sample.transition)].append(sample)
        by_direction[sample.direction].append(sample)

    fallback_seconds = {
        TurnDirection.LEFT: _validated_fallback(fallback_left_seconds_90),
        TurnDirection.RIGHT: _validated_fallback(fallback_right_seconds_90),
    }

    profiles: dict[TurnDirection, DirectionRotationProfile] = {}
    for direction in TurnDirection:
        base = _aggregate_timing(
            by_direction[direction],
            fallback_seconds_90=fallback_seconds[direction],
        )
        timings = {
            transition: _fit_transition_timing(
                grouped[(direction, transition)],
                fallback=base,
            )
            for transition in TurnTransition
        }
        neutral = timings[TurnTransition.NEUTRAL]
        if neutral.is_fallback:
            candidates = (
                timings[TurnTransition.SAME_DIRECTION],
                timings[TurnTransition.REVERSAL],
                base,
            )
            neutral = RotationTiming(
                rate_degrees_per_second=min(
                    timing.rate_degrees_per_second for timing in candidates
                ),
                dead_time_seconds=max(
                    timing.dead_time_seconds for timing in candidates
                ),
                sample_count=neutral.sample_count,
                median_error_degrees=max(
                    timing.median_error_degrees for timing in candidates
                ),
                is_fallback=True,
            )
        profiles[direction] = DirectionRotationProfile(
            neutral=neutral,
            same_direction=timings[TurnTransition.SAME_DIRECTION],
            reversal=timings[TurnTransition.REVERSAL],
        )

    return StateAwareRotationModel(
        left=profiles[TurnDirection.LEFT],
        right=profiles[TurnDirection.RIGHT],
        neutral_after_seconds=float(neutral_after_seconds),
        idle_response_curves=idle_response_curves,
    )


def _sample_is_valid(
    sample: RotationSample,
    *,
    minimum_confidence: float,
) -> bool:
    values = (
        sample.clamped_seconds,
        sample.held_seconds,
        sample.measured_degrees,
        sample.confidence,
    )
    return (
        all(isfinite(value) for value in values)
        and sample.clamped_seconds > 0.0
        and sample.held_seconds > 0.0
        and 1.5 <= sample.measured_degrees <= 120.0
        and sample.confidence >= minimum_confidence
    )


def _validated_fallback(seconds_90: float) -> float:
    seconds_90 = float(seconds_90)
    if not isfinite(seconds_90) or seconds_90 <= 0.0:
        raise ValueError("fallback 90-degree timing must be finite and positive")
    return seconds_90


def _aggregate_timing(
    samples: list[RotationSample],
    *,
    fallback_seconds_90: float,
) -> RotationTiming:
    if not samples:
        return RotationTiming(
            rate_degrees_per_second=90.0 / fallback_seconds_90,
            dead_time_seconds=0.0,
            sample_count=0,
            median_error_degrees=0.0,
            is_fallback=True,
        )

    equivalent_seconds = [
        sample.held_seconds * 90.0 / sample.measured_degrees for sample in samples
    ]
    center = float(median(equivalent_seconds))
    deviations = [abs(value - center) for value in equivalent_seconds]
    mad = float(median(deviations)) if deviations else 0.0
    threshold = max(3.0 * mad, 0.015)
    filtered = [
        value for value in equivalent_seconds if abs(value - center) <= threshold
    ]
    seconds_90 = float(median(filtered or equivalent_seconds))
    rate = 90.0 / seconds_90
    errors = [
        abs(sample.measured_degrees - rate * sample.held_seconds) for sample in samples
    ]
    return RotationTiming(
        rate_degrees_per_second=rate,
        dead_time_seconds=0.0,
        sample_count=len(samples),
        median_error_degrees=float(median(errors)),
    )


def _fit_transition_timing(
    samples: list[RotationSample],
    *,
    fallback: RotationTiming,
) -> RotationTiming:
    if len(samples) < 2:
        return RotationTiming(
            rate_degrees_per_second=fallback.rate_degrees_per_second,
            dead_time_seconds=fallback.dead_time_seconds,
            sample_count=len(samples),
            median_error_degrees=fallback.median_error_degrees,
            is_fallback=True,
        )

    # Distinct commanded durations identify an intentional multi-duration
    # experiment. Scheduler jitter in repeated fixed-duration probes must not
    # be mistaken for enough leverage to fit a dead-time intercept.
    distinct_durations = {round(sample.requested_seconds, 5) for sample in samples}
    if len(samples) < 3 or len(distinct_durations) < 3:
        seconds_90 = float(
            median(
                sample.held_seconds * 90.0 / sample.measured_degrees
                for sample in samples
            )
        )
        rate = 90.0 / seconds_90
        errors = [
            abs(sample.measured_degrees - rate * sample.held_seconds)
            for sample in samples
        ]
        return RotationTiming(
            rate_degrees_per_second=rate,
            dead_time_seconds=0.0,
            sample_count=len(samples),
            median_error_degrees=float(median(errors)),
        )

    rate, intercept = _theil_sen(samples)
    if not isfinite(rate) or rate <= 0.0:
        return RotationTiming(
            rate_degrees_per_second=fallback.rate_degrees_per_second,
            dead_time_seconds=fallback.dead_time_seconds,
            sample_count=len(samples),
            median_error_degrees=fallback.median_error_degrees,
            is_fallback=True,
        )

    dead_time = max(0.0, -intercept / rate)
    shortest = min(sample.held_seconds for sample in samples)
    dead_time = min(dead_time, shortest * 0.90)

    errors = [
        abs(sample.measured_degrees - rate * max(0.0, sample.held_seconds - dead_time))
        for sample in samples
    ]
    error_center = float(median(errors))
    error_deviations = [abs(error - error_center) for error in errors]
    error_mad = float(median(error_deviations))
    inlier_limit = max(error_center + 3.0 * error_mad, 2.0)
    inliers = [
        sample
        for sample, error in zip(samples, errors, strict=True)
        if error <= inlier_limit
    ]
    if 3 <= len(inliers) < len(samples):
        refined_rate, refined_intercept = _theil_sen(inliers)
        if isfinite(refined_rate) and refined_rate > 0.0:
            rate = refined_rate
            dead_time = max(0.0, -refined_intercept / rate)
            shortest = min(sample.held_seconds for sample in inliers)
            dead_time = min(dead_time, shortest * 0.90)
            samples = inliers
            errors = [
                abs(
                    sample.measured_degrees
                    - rate * max(0.0, sample.held_seconds - dead_time)
                )
                for sample in samples
            ]

    return RotationTiming(
        rate_degrees_per_second=rate,
        dead_time_seconds=dead_time,
        sample_count=len(samples),
        median_error_degrees=float(median(errors)),
    )


def _theil_sen(samples: list[RotationSample]) -> tuple[float, float]:
    slopes: list[float] = []
    for index, first in enumerate(samples):
        for second in samples[index + 1 :]:
            duration_delta = second.held_seconds - first.held_seconds
            if abs(duration_delta) < 1e-6:
                continue
            slope = (second.measured_degrees - first.measured_degrees) / duration_delta
            if isfinite(slope) and slope > 0.0:
                slopes.append(slope)

    if not slopes:
        return float("nan"), float("nan")

    rate = float(median(slopes))
    intercept = float(
        median(
            sample.measured_degrees - rate * sample.held_seconds for sample in samples
        )
    )
    return rate, intercept
