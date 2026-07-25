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


class TurnMemoryMode(StrEnum):
    """How long direction-dependent turn response remains observable."""

    DECAYS_TO_NEUTRAL = "decays_to_neutral"
    PERSISTENT_OBSERVED = "persistent_observed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class TurnMemoryPolicy:
    """Runtime policy derived from the calibrated turn-memory evidence."""

    mode: TurnMemoryMode
    observed_horizon_seconds: float
    neutral_after_seconds: float | None = None

    def __post_init__(self) -> None:
        if (
            not isfinite(self.observed_horizon_seconds)
            or self.observed_horizon_seconds <= 0
        ):
            raise ValueError("observed_horizon_seconds must be finite and positive")
        if self.mode is TurnMemoryMode.DECAYS_TO_NEUTRAL:
            if self.neutral_after_seconds is None:
                raise ValueError("neutral decay requires neutral_after_seconds")
            if (
                not isfinite(self.neutral_after_seconds)
                or self.neutral_after_seconds <= 0
            ):
                raise ValueError("neutral_after_seconds must be finite and positive")
        elif self.neutral_after_seconds is not None:
            raise ValueError("only neutral-decay mode may define neutral_after_seconds")
        if self.mode is TurnMemoryMode.INSUFFICIENT_EVIDENCE:
            raise ValueError(
                "insufficient turn-memory evidence cannot be used at runtime"
            )

    def becomes_neutral(self, idle_seconds: float) -> bool:
        if not isfinite(idle_seconds) or idle_seconds < 0:
            raise ValueError("idle_seconds must be finite and non-negative")
        return bool(
            self.mode is TurnMemoryMode.DECAYS_TO_NEUTRAL
            and self.neutral_after_seconds is not None
            and idle_seconds >= self.neutral_after_seconds
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "observed_horizon_seconds": round(self.observed_horizon_seconds, 6),
            "neutral_after_seconds": (
                round(self.neutral_after_seconds, 6)
                if self.neutral_after_seconds is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TurnMemoryPolicy:
        raw_neutral = data.get("neutral_after_seconds")
        neutral = (
            None
            if raw_neutral is None
            else _as_float(raw_neutral, "neutral_after_seconds")
        )
        return cls(
            mode=TurnMemoryMode(str(data["mode"])),
            observed_horizon_seconds=_as_float(
                data["observed_horizon_seconds"], "observed_horizon_seconds"
            ),
            neutral_after_seconds=neutral,
        )


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
    """Classify pulses without inventing a neutral timeout."""

    def __init__(
        self,
        policy: TurnMemoryPolicy | float = 2.0,
    ) -> None:
        if isinstance(policy, (int, float)) and not isinstance(policy, bool):
            seconds = float(policy)
            policy = TurnMemoryPolicy(
                mode=TurnMemoryMode.DECAYS_TO_NEUTRAL,
                observed_horizon_seconds=seconds,
                neutral_after_seconds=seconds,
            )
        self._policy = policy
        self._last_direction: TurnDirection | None = None
        self._last_completed_at: float | None = None

    @property
    def policy(self) -> TurnMemoryPolicy:
        return self._policy

    @property
    def neutral_after_seconds(self) -> float:
        return (
            self._policy.neutral_after_seconds
            if self._policy.neutral_after_seconds is not None
            else self._policy.observed_horizon_seconds
        )

    @property
    def last_direction(self) -> TurnDirection | None:
        return self._last_direction

    def set_policy(
        self,
        policy: TurnMemoryPolicy,
        *,
        reset_history: bool = True,
    ) -> None:
        self._policy = policy
        if reset_history:
            self.reset()

    def set_neutral_after_seconds(self, seconds: float) -> None:
        seconds = float(seconds)
        self.set_policy(
            TurnMemoryPolicy(
                mode=TurnMemoryMode.DECAYS_TO_NEUTRAL,
                observed_horizon_seconds=seconds,
                neutral_after_seconds=seconds,
            ),
            reset_history=False,
        )

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
        previous = self._last_direction
        if previous is None or self._last_completed_at is None:
            return TurnTransition.NEUTRAL, None
        idle_seconds = self.idle_seconds(now=now)
        assert idle_seconds is not None
        if self._policy.becomes_neutral(idle_seconds):
            return TurnTransition.NEUTRAL, idle_seconds
        return (
            TurnTransition.SAME_DIRECTION
            if previous is direction
            else TurnTransition.REVERSAL,
            idle_seconds,
        )

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
    """Observed monotonic decay; persistent curves need not reach neutral."""

    mode: TurnMemoryMode
    idle_seconds: tuple[float, ...]
    response_progress: tuple[float, ...]
    source_sample_count: int
    observed_horizon_seconds: float
    stateful_response_degrees: float
    reference_response_degrees: float
    maximum_monotonic_adjustment_degrees: float

    def __post_init__(self) -> None:
        if len(self.idle_seconds) != len(self.response_progress):
            raise ValueError("idle-response curve axes must have the same length")
        if len(self.idle_seconds) < 2 or self.source_sample_count < 2:
            raise ValueError("idle-response curve requires at least two samples")
        values = (
            *self.idle_seconds,
            *self.response_progress,
            self.observed_horizon_seconds,
            self.stateful_response_degrees,
            self.reference_response_degrees,
            self.maximum_monotonic_adjustment_degrees,
        )
        if not all(isfinite(v) for v in values):
            raise ValueError("idle-response curve values must be finite")
        if self.idle_seconds[0] != 0.0 or self.response_progress[0] != 0.0:
            raise ValueError("idle-response curve must start at zero")
        if any(b <= a for a, b in zip(self.idle_seconds, self.idle_seconds[1:])):
            raise ValueError("idle-response curve delays must be strictly increasing")
        if any(v < 0.0 or v > 1.0 for v in self.response_progress):
            raise ValueError("idle-response progress must stay in [0, 1]")
        if any(
            b < a for a, b in zip(self.response_progress, self.response_progress[1:])
        ):
            raise ValueError("idle-response progress must be monotonic")
        if abs(self.idle_seconds[-1] - self.observed_horizon_seconds) > 1e-6:
            raise ValueError("curve endpoint must match observed horizon")
        if (
            self.mode is TurnMemoryMode.DECAYS_TO_NEUTRAL
            and self.response_progress[-1] != 1.0
        ):
            raise ValueError("neutral-decay curve must end at full progress")
        if (
            self.mode is TurnMemoryMode.PERSISTENT_OBSERVED
            and self.response_progress[-1] >= 1.0
        ):
            raise ValueError("persistent curve must not claim neutral convergence")
        if self.mode is TurnMemoryMode.INSUFFICIENT_EVIDENCE:
            raise ValueError("insufficient evidence cannot create a runtime curve")

    @property
    def neutral_after_seconds(self) -> float | None:
        return (
            self.observed_horizon_seconds
            if self.mode is TurnMemoryMode.DECAYS_TO_NEUTRAL
            else None
        )

    @property
    def neutral_response_degrees(self) -> float:
        return self.reference_response_degrees

    def progress_at(self, idle_seconds: float) -> float:
        idle_seconds = float(idle_seconds)
        if not isfinite(idle_seconds):
            raise ValueError("idle_seconds must be finite")
        if idle_seconds <= 0.0:
            return 0.0
        if idle_seconds >= self.idle_seconds[-1]:
            return self.response_progress[-1]
        for i in range(1, len(self.idle_seconds)):
            if idle_seconds <= self.idle_seconds[i]:
                lo_t, hi_t = self.idle_seconds[i - 1], self.idle_seconds[i]
                lo_p, hi_p = self.response_progress[i - 1], self.response_progress[i]
                return lo_p + (idle_seconds - lo_t) / (hi_t - lo_t) * (hi_p - lo_p)
        raise AssertionError("validated curve has no matching segment")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "idle_seconds": [round(v, 6) for v in self.idle_seconds],
            "response_progress": [round(v, 6) for v in self.response_progress],
            "source_sample_count": self.source_sample_count,
            "observed_horizon_seconds": round(self.observed_horizon_seconds, 6),
            "stateful_response_degrees": round(self.stateful_response_degrees, 6),
            "reference_response_degrees": round(self.reference_response_degrees, 6),
            "maximum_monotonic_adjustment_degrees": round(
                self.maximum_monotonic_adjustment_degrees, 6
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> DirectionIdleResponseCurve:
        idle = cast(list[object], data["idle_seconds"])
        progress = cast(list[object], data["response_progress"])
        return cls(
            mode=TurnMemoryMode(str(data["mode"])),
            idle_seconds=tuple(_as_float(v, "idle_seconds") for v in idle),
            response_progress=tuple(
                _as_float(v, "response_progress") for v in progress
            ),
            source_sample_count=_as_int(
                data["source_sample_count"], "source_sample_count"
            ),
            observed_horizon_seconds=_as_float(
                data["observed_horizon_seconds"], "observed_horizon_seconds"
            ),
            stateful_response_degrees=_as_float(
                data["stateful_response_degrees"], "stateful_response_degrees"
            ),
            reference_response_degrees=_as_float(
                data["reference_response_degrees"], "reference_response_degrees"
            ),
            maximum_monotonic_adjustment_degrees=_as_float(
                data["maximum_monotonic_adjustment_degrees"],
                "maximum_monotonic_adjustment_degrees",
            ),
        )


@dataclass(frozen=True)
class IdleResponseCurves:
    left: DirectionIdleResponseCurve
    right: DirectionIdleResponseCurve

    def curve_for(self, direction: TurnDirection) -> DirectionIdleResponseCurve:
        return self.left if direction is TurnDirection.LEFT else self.right

    @property
    def observed_horizon_seconds(self) -> float:
        return max(
            self.left.observed_horizon_seconds, self.right.observed_horizon_seconds
        )

    @property
    def policy(self) -> TurnMemoryPolicy:
        modes = {self.left.mode, self.right.mode}
        if TurnMemoryMode.PERSISTENT_OBSERVED in modes:
            return TurnMemoryPolicy(
                TurnMemoryMode.PERSISTENT_OBSERVED, self.observed_horizon_seconds
            )
        neutral = max(
            self.left.observed_horizon_seconds, self.right.observed_horizon_seconds
        )
        return TurnMemoryPolicy(
            TurnMemoryMode.DECAYS_TO_NEUTRAL, self.observed_horizon_seconds, neutral
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> IdleResponseCurves:
        version = _as_int(data["version"], "idle_response_curves.version")
        if version != 2:
            raise ValueError(
                f"unsupported idle-response curve version {version}; expected 2"
            )
        return cls(
            left=DirectionIdleResponseCurve.from_dict(
                cast(dict[str, object], data["left"])
            ),
            right=DirectionIdleResponseCurve.from_dict(
                cast(dict[str, object], data["right"])
            ),
        )


@dataclass(frozen=True, init=False)
class StateAwareRotationModel:
    left: DirectionRotationProfile
    right: DirectionRotationProfile
    turn_memory_policy: TurnMemoryPolicy
    idle_response_curves: IdleResponseCurves | None = None

    def __init__(
        self,
        left: DirectionRotationProfile,
        right: DirectionRotationProfile,
        turn_memory_policy: TurnMemoryPolicy | float | None = None,
        idle_response_curves: IdleResponseCurves | None = None,
        *,
        neutral_after_seconds: float | None = None,
    ) -> None:
        """Create a model while preserving the pre-v9 timeout constructor API."""
        if isinstance(turn_memory_policy, (int, float)) and not isinstance(
            turn_memory_policy, bool
        ):
            if neutral_after_seconds is not None:
                raise TypeError(
                    "neutral_after_seconds was provided both positionally and "
                    "by keyword"
                )
            neutral_after_seconds = float(turn_memory_policy)
            turn_memory_policy = None

        if turn_memory_policy is not None and neutral_after_seconds is not None:
            raise TypeError(
                "provide turn_memory_policy or neutral_after_seconds, not both"
            )
        if turn_memory_policy is None:
            if neutral_after_seconds is not None:
                seconds = float(neutral_after_seconds)
                turn_memory_policy = TurnMemoryPolicy(
                    TurnMemoryMode.DECAYS_TO_NEUTRAL,
                    seconds,
                    seconds,
                )
            elif idle_response_curves is not None:
                turn_memory_policy = idle_response_curves.policy
            else:
                raise TypeError(
                    "turn_memory_policy or neutral_after_seconds is required"
                )

        if (
            idle_response_curves is not None
            and idle_response_curves.policy != turn_memory_policy
        ):
            raise ValueError("turn-memory policy does not match idle-response curves")

        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "turn_memory_policy", turn_memory_policy)
        object.__setattr__(self, "idle_response_curves", idle_response_curves)

    @property
    def neutral_after_seconds(self) -> float:
        return (
            self.turn_memory_policy.neutral_after_seconds
            or self.turn_memory_policy.observed_horizon_seconds
        )

    def profile_for(self, direction: TurnDirection) -> DirectionRotationProfile:
        return self.left if direction is TurnDirection.LEFT else self.right

    def timing_for(
        self, direction: TurnDirection, transition: TurnTransition
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
        progress = self.idle_response_curves.curve_for(direction).progress_at(
            float(idle_seconds)
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
            selected, transition, degrees, idle_seconds=idle_seconds
        )

    def to_dict(self) -> dict[str, object]:
        data = {
            "version": 3,
            "turn_memory_policy": self.turn_memory_policy.to_dict(),
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }
        if self.idle_response_curves is not None:
            data["idle_response_curves"] = self.idle_response_curves.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> StateAwareRotationModel:
        version = _as_int(data["version"], "rotation_model.version")
        if version != 3:
            raise ValueError(
                f"unsupported rotation model version {version}; expected 3"
            )

        def profile(direction: TurnDirection) -> DirectionRotationProfile:
            return DirectionRotationProfile.from_dict(
                cast(dict[str, object], data[direction.value])
            )

        curves_raw = data.get("idle_response_curves")
        curves = (
            None
            if curves_raw is None
            else IdleResponseCurves.from_dict(cast(dict[str, object], curves_raw))
        )
        policy = TurnMemoryPolicy.from_dict(
            cast(dict[str, object], data["turn_memory_policy"])
        )
        if curves is not None and curves.policy != policy:
            raise ValueError("turn-memory policy does not match idle-response curves")
        return cls(
            profile(TurnDirection.LEFT), profile(TurnDirection.RIGHT), policy, curves
        )


@dataclass(frozen=True)
class NeutralTimeoutSample:
    direction: TurnDirection
    requested_idle_seconds: float
    observed_idle_seconds: float
    measured_degrees: float
    uncertainty_degrees: float
    confidence: float
    conditioning_transition: TurnTransition = TurnTransition.REVERSAL
    requested_seconds: float = 1.0
    clamped_seconds: float = 1.0
    held_seconds: float = 1.0

    def as_neutral_rotation_sample(self) -> RotationSample:
        return RotationSample(
            self.direction,
            TurnTransition.NEUTRAL,
            self.requested_seconds,
            self.clamped_seconds,
            self.held_seconds,
            self.measured_degrees,
            self.confidence,
            self.observed_idle_seconds,
        )


@dataclass(frozen=True)
class DirectionNeutralTimeoutFit:
    mode: TurnMemoryMode
    observed_horizon_seconds: float
    last_stateful_seconds: float
    first_neutral_seconds: float | None
    neutral_response_degrees: float | None
    neutral_tolerance_degrees: float
    neutral_uncertainty_degrees: float
    stateful_sample_count: int
    neutral_sample_count: int
    terminal_same_response_degrees: float
    terminal_reversal_response_degrees: float

    def to_dict(self) -> dict[str, object]:
        return {
            k: (v.value if isinstance(v, StrEnum) else v)
            for k, v in self.__dict__.items()
        }


@dataclass(frozen=True)
class NeutralTimeoutFit:
    turn_memory_policy: TurnMemoryPolicy
    safety_margin_seconds: float
    left: DirectionNeutralTimeoutFit
    right: DirectionNeutralTimeoutFit
    idle_response_curves: IdleResponseCurves

    @property
    def neutral_after_seconds(self) -> float:
        return (
            self.turn_memory_policy.neutral_after_seconds
            or self.turn_memory_policy.observed_horizon_seconds
        )

    def fit_for(self, direction: TurnDirection) -> DirectionNeutralTimeoutFit:
        return self.left if direction is TurnDirection.LEFT else self.right

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_memory_policy": self.turn_memory_policy.to_dict(),
            "neutral_after_seconds": self.turn_memory_policy.neutral_after_seconds,
            "safety_margin_seconds": self.safety_margin_seconds,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "idle_response_curves": self.idle_response_curves.to_dict(),
        }


def _isotonic(values: list[float]) -> list[float]:
    out = values[:]
    changed = True
    while changed:
        changed = False
        for i in range(1, len(out)):
            if out[i] < out[i - 1]:
                avg = (out[i] + out[i - 1]) / 2.0
                out[i - 1] = out[i] = avg
                changed = True
    return out


def fit_neutral_timeout(
    samples: Iterable[NeutralTimeoutSample],
    *,
    safety_margin_seconds: float = 0.25,
    maximum_idle_seconds: float = 6.0,
    minimum_confidence: float = 0.52,
    minimum_samples_per_direction: int = 5,
    neutral_tail_samples: int = 2,
    minimum_stateful_samples: int = 1,
    minimum_neutral_tolerance_degrees: float = 1.5,
    maximum_neutral_tail_spread_degrees: float = 2.5,
    maximum_sample_uncertainty_degrees: float = 3.0,
    minimum_response_span_degrees: float = 3.0,
    minimum_monotonic_tolerance_degrees: float = 1.0,
) -> NeutralTimeoutFit:
    del minimum_monotonic_tolerance_degrees
    valid = [
        s
        for s in samples
        if _neutral_timeout_sample_is_valid(s, minimum_confidence=minimum_confidence)
        and s.uncertainty_degrees <= maximum_sample_uncertainty_degrees
    ]
    fits: dict[TurnDirection, DirectionNeutralTimeoutFit] = {}
    curves: dict[TurnDirection, DirectionIdleResponseCurve] = {}
    for direction in TurnDirection:
        ds = sorted(
            [s for s in valid if s.direction is direction],
            key=lambda x: x.observed_idle_seconds,
        )
        if len(ds) < minimum_samples_per_direction:
            raise ValueError(f"{direction.value} has insufficient turn-memory samples")
        same = [
            s for s in ds if s.conditioning_transition is TurnTransition.SAME_DIRECTION
        ]
        reversal = [
            s for s in ds if s.conditioning_transition is TurnTransition.REVERSAL
        ]
        if len(reversal) < minimum_stateful_samples:
            raise ValueError(f"{direction.value} lacks state-dependent response")
        horizon = min(max(s.observed_idle_seconds for s in ds), maximum_idle_seconds)
        same_tail = (
            same[-neutral_tail_samples:] if len(same) >= neutral_tail_samples else []
        )
        rev_tail = (
            reversal[-neutral_tail_samples:]
            if len(reversal) >= neutral_tail_samples
            else []
        )
        converged = False
        same_terminal = (
            float(median([s.measured_degrees for s in same_tail]))
            if same_tail
            else float("nan")
        )
        rev_terminal = (
            float(median([s.measured_degrees for s in rev_tail]))
            if rev_tail
            else float(
                median([s.measured_degrees for s in reversal[-neutral_tail_samples:]])
            )
        )
        tolerance = minimum_neutral_tolerance_degrees
        if same_tail and rev_tail:
            tolerance = max(
                tolerance,
                median([s.uncertainty_degrees for s in same_tail])
                + median([s.uncertainty_degrees for s in rev_tail]),
            )
            spread = max(
                max(s.measured_degrees for s in same_tail + rev_tail)
                - min(s.measured_degrees for s in same_tail + rev_tail),
                0.0,
            )
            converged = (
                abs(same_terminal - rev_terminal) <= tolerance
                and spread <= maximum_neutral_tail_spread_degrees
            )
        stateful = float(
            median(
                [
                    s.measured_degrees
                    for s in reversal[: max(minimum_stateful_samples, 2)]
                ]
            )
        )
        reference = (
            (same_terminal + rev_terminal) / 2.0
            if converged
            else max(
                same_terminal if isfinite(same_terminal) else stateful, rev_terminal
            )
        )
        span = reference - stateful
        if span < minimum_response_span_degrees:
            raise ValueError(
                f"{direction.value} probes did not establish a state-dependent response"
            )
        mode = (
            TurnMemoryMode.DECAYS_TO_NEUTRAL
            if converged
            else TurnMemoryMode.PERSISTENT_OBSERVED
        )
        ordered_rev = sorted(reversal, key=lambda x: x.observed_idle_seconds)
        times = [0.0] + [min(s.observed_idle_seconds, horizon) for s in ordered_rev]
        raw = [0.0] + [
            max(0.0, min(1.0, (s.measured_degrees - stateful) / span))
            for s in ordered_rev
        ]
        progress = _isotonic(raw)
        if mode is TurnMemoryMode.DECAYS_TO_NEUTRAL:
            progress[-1] = 1.0
        elif progress[-1] >= 1.0:
            progress[-1] = 0.999
        # coalesce duplicate times
        compact_t = []
        compact_p = []
        for t, pv in zip(times, progress, strict=True):
            if compact_t and abs(t - compact_t[-1]) < 1e-6:
                compact_p[-1] = max(compact_p[-1], pv)
            else:
                compact_t.append(t)
                compact_p.append(pv)
        if compact_t[-1] < horizon:
            compact_t.append(horizon)
            compact_p.append(compact_p[-1])
        curve = DirectionIdleResponseCurve(
            mode,
            tuple(compact_t),
            tuple(compact_p),
            len(ds),
            horizon,
            stateful,
            reference,
            0.0,
        )
        first_neutral = horizon + safety_margin_seconds if converged else None
        fits[direction] = DirectionNeutralTimeoutFit(
            mode,
            horizon,
            max(
                0.0,
                ordered_rev[
                    min(len(ordered_rev) - 1, max(minimum_stateful_samples - 1, 0))
                ].observed_idle_seconds,
            ),
            first_neutral,
            reference if converged else None,
            tolerance,
            max(s.uncertainty_degrees for s in ds),
            len(reversal),
            len(same_tail) + len(rev_tail),
            same_terminal,
            rev_terminal,
        )
        curves[direction] = curve
    idle_curves = IdleResponseCurves(
        curves[TurnDirection.LEFT], curves[TurnDirection.RIGHT]
    )
    policy = idle_curves.policy
    if policy.mode is TurnMemoryMode.DECAYS_TO_NEUTRAL:
        neutral = max(
            f.first_neutral_seconds or f.observed_horizon_seconds for f in fits.values()
        )
        policy = TurnMemoryPolicy(
            TurnMemoryMode.DECAYS_TO_NEUTRAL,
            max(f.observed_horizon_seconds for f in fits.values()),
            neutral,
        )
        # extend curves to policy endpoint
        ext = {}
        for d, c in curves.items():
            if c.observed_horizon_seconds < neutral:
                ext[d] = DirectionIdleResponseCurve(
                    c.mode,
                    c.idle_seconds + (neutral,),
                    c.response_progress[:-1] + (1.0, 1.0),
                    c.source_sample_count,
                    neutral,
                    c.stateful_response_degrees,
                    c.reference_response_degrees,
                    c.maximum_monotonic_adjustment_degrees,
                )
            else:
                ext[d] = c
        idle_curves = IdleResponseCurves(
            ext[TurnDirection.LEFT], ext[TurnDirection.RIGHT]
        )
    return NeutralTimeoutFit(
        policy,
        safety_margin_seconds,
        fits[TurnDirection.LEFT],
        fits[TurnDirection.RIGHT],
        idle_curves,
    )


def validate_neutral_timeout(
    fit: NeutralTimeoutFit,
    samples: Iterable[NeutralTimeoutSample],
    *,
    minimum_confidence: float = 0.52,
    **_ignored: object,
) -> None:
    valid = [
        s
        for s in samples
        if _neutral_timeout_sample_is_valid(s, minimum_confidence=minimum_confidence)
    ]
    for direction in TurnDirection:
        ds = [s for s in valid if s.direction is direction]
        if len(ds) < 2:
            raise ValueError(
                f"{direction.value} validation needs both same and reversal samples"
            )
        same = [
            s for s in ds if s.conditioning_transition is TurnTransition.SAME_DIRECTION
        ]
        rev = [s for s in ds if s.conditioning_transition is TurnTransition.REVERSAL]
        fitted = fit.fit_for(direction)
        if fitted.mode is TurnMemoryMode.DECAYS_TO_NEUTRAL and (
            not same
            or not rev
            or abs(
                median([s.measured_degrees for s in same])
                - median([s.measured_degrees for s in rev])
            )
            > fitted.neutral_tolerance_degrees
        ):
            raise ValueError(
                f"{direction.value} response at the fitted timeout did not match validated neutral response"
            )


def _neutral_timeout_sample_is_valid(
    sample: NeutralTimeoutSample, *, minimum_confidence: float
) -> bool:
    vals = (
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
        all(isfinite(v) for v in vals)
        and sample.requested_idle_seconds > 0
        and sample.observed_idle_seconds > 0
        and sample.measured_degrees > 0
        and sample.uncertainty_degrees >= 0
        and sample.confidence >= minimum_confidence
        and sample.requested_seconds > 0
        and sample.clamped_seconds > 0
        and sample.held_seconds > 0
    )


def fit_rotation_model(
    samples: Iterable[RotationSample],
    *,
    fallback_left_seconds_90: float,
    fallback_right_seconds_90: float,
    neutral_after_seconds: float = 2.0,
    idle_response_curves: IdleResponseCurves | None = None,
    turn_memory_policy: TurnMemoryPolicy | None = None,
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

    policy = turn_memory_policy
    if policy is None:
        if idle_response_curves is not None:
            policy = idle_response_curves.policy
        else:
            policy = TurnMemoryPolicy(
                TurnMemoryMode.DECAYS_TO_NEUTRAL,
                float(neutral_after_seconds),
                float(neutral_after_seconds),
            )
    return StateAwareRotationModel(
        left=profiles[TurnDirection.LEFT],
        right=profiles[TurnDirection.RIGHT],
        turn_memory_policy=policy,
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
