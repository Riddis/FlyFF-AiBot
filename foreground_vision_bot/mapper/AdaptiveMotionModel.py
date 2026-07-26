from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from .AdaptiveMotionTracker import DirectionalFlow


class TurnDirection(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class AdaptiveForwardOutcome(StrEnum):
    MOVED = "moved"
    BLOCKED = "blocked"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ForwardAssessment:
    outcome: AdaptiveForwardOutcome
    reliable: bool
    distance_cells: float | None
    confidence: float
    expected_flow_px: float | None
    observed_flow_px: float
    flow_ratio: float | None
    reason: str


@dataclass
class AdaptiveMotionModel:
    """
    Small online model learned during ordinary mapping.

    The model intentionally avoids pretending that optical-flow pixels are game
    metres. One successful nominal forward pulse defines one map cell. Optical
    flow is used to verify that the command produced a coherent, ordinary move.
    Turn timing is updated from measured minimap heading changes after each
    bounded turn pulse.
    """

    version: int = 1
    forward_seconds: float = 0.120
    left_seconds_per_degree: float = 0.00386
    right_seconds_per_degree: float = 0.00382
    left_turn_samples: int = 0
    right_turn_samples: int = 0
    forward_flow_px: float | None = None
    forward_flow_deviation_px: float | None = None
    forward_samples: int = 0
    rejected_turn_samples: int = 0
    uncertain_forward_samples: int = 0
    updated_at: str = ""

    MIN_SECONDS_PER_DEGREE = 0.0015
    MAX_SECONDS_PER_DEGREE = 0.0080
    MIN_FORWARD_FLOW_PX = 1.35
    STATIC_FLOW_PX = 1.10
    STATIC_CHANGE_SCORE = 0.018
    MIN_TRACKED_POINTS = 8
    MIN_INLIER_RATIO = 0.45
    MIN_FLOW_CONFIDENCE = 0.35

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"Unsupported adaptive motion model version: {self.version}")
        finite_positive = (
            self.forward_seconds,
            self.left_seconds_per_degree,
            self.right_seconds_per_degree,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in finite_positive):
            raise ValueError("Adaptive motion timing values must be finite and positive")
        if not 0.03 <= self.forward_seconds <= 0.50:
            raise ValueError("forward_seconds is outside the safe mapper range")
        for value in (self.left_seconds_per_degree, self.right_seconds_per_degree):
            if not self.MIN_SECONDS_PER_DEGREE <= value <= self.MAX_SECONDS_PER_DEGREE:
                raise ValueError("turn timing is outside the safe mapper range")
        for count in (
            self.left_turn_samples,
            self.right_turn_samples,
            self.forward_samples,
            self.rejected_turn_samples,
            self.uncertain_forward_samples,
        ):
            if isinstance(count, bool) or int(count) < 0:
                raise ValueError("Adaptive motion sample counts cannot be negative")
        if self.forward_flow_px is not None:
            if not math.isfinite(self.forward_flow_px) or self.forward_flow_px <= 0.0:
                raise ValueError("forward_flow_px must be finite and positive")
        if self.forward_flow_deviation_px is not None:
            if (
                not math.isfinite(self.forward_flow_deviation_px)
                or self.forward_flow_deviation_px < 0.0
            ):
                raise ValueError("forward_flow_deviation_px cannot be negative")
        if not self.updated_at:
            self.updated_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def load_or_default(
        cls,
        path: Path,
        *,
        forward_seconds: float = 0.120,
    ) -> tuple[AdaptiveMotionModel, str | None]:
        if not path.exists():
            return cls(forward_seconds=float(forward_seconds)), None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("adaptive motion file must contain an object")
            return cls.from_dict(payload), None
        except Exception as error:  # noqa: BLE001 - recover from local state damage.
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            invalid_path = path.with_name(f"{path.stem}.invalid_{timestamp}{path.suffix}")
            try:
                path.replace(invalid_path)
                detail = f" Invalid file moved to {invalid_path.name}."
            except OSError:
                detail = ""
            warning = (
                "Adaptive motion state could not be loaded; conservative defaults "
                f"will be used. Cause: {error}.{detail}"
            )
            return cls(forward_seconds=float(forward_seconds)), warning

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdaptiveMotionModel:
        allowed = {
            "version",
            "forward_seconds",
            "left_seconds_per_degree",
            "right_seconds_per_degree",
            "left_turn_samples",
            "right_turn_samples",
            "forward_flow_px",
            "forward_flow_deviation_px",
            "forward_samples",
            "rejected_turn_samples",
            "uncertain_forward_samples",
            "updated_at",
        }
        filtered = {key: value for key, value in data.items() if key in allowed}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def save_snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def seconds_per_degree(self, direction: TurnDirection) -> float:
        if direction is TurnDirection.LEFT:
            return float(self.left_seconds_per_degree)
        return float(self.right_seconds_per_degree)

    def seconds_for_turn(
        self,
        direction: TurnDirection,
        degrees: float,
        *,
        minimum_seconds: float = 0.015,
        maximum_seconds: float = 0.140,
    ) -> float:
        requested_degrees = max(0.0, float(degrees))
        seconds = requested_degrees * self.seconds_per_degree(direction)
        return min(float(maximum_seconds), max(float(minimum_seconds), seconds))

    def observe_turn(
        self,
        direction: TurnDirection,
        *,
        held_seconds: float,
        signed_motion_degrees: float,
        uncertainty_degrees: float,
    ) -> bool:
        """Update turn timing from one physically plausible observed pulse."""
        values = (held_seconds, signed_motion_degrees, uncertainty_degrees)
        if not all(math.isfinite(float(value)) for value in values):
            self.rejected_turn_samples += 1
            return False

        sign = -1.0 if direction is TurnDirection.LEFT else 1.0
        directed_motion = float(signed_motion_degrees) * sign
        uncertainty = max(0.0, float(uncertainty_degrees))
        minimum_motion = max(1.5, uncertainty * 0.65)
        if not 0.015 <= held_seconds <= 0.25:
            self.rejected_turn_samples += 1
            return False
        if not minimum_motion <= directed_motion <= 65.0:
            self.rejected_turn_samples += 1
            return False

        candidate = float(held_seconds) / directed_motion
        if not self.MIN_SECONDS_PER_DEGREE <= candidate <= self.MAX_SECONDS_PER_DEGREE:
            self.rejected_turn_samples += 1
            return False

        current = self.seconds_per_degree(direction)
        # Reject a likely heading alias or one-off input stall before it can
        # destabilise future commands. Genuine rate changes are still learned
        # gradually from repeated accepted observations.
        if not current * 0.45 <= candidate <= current * 1.85:
            self.rejected_turn_samples += 1
            return False

        sample_count = (
            self.left_turn_samples
            if direction is TurnDirection.LEFT
            else self.right_turn_samples
        )
        alpha = max(0.08, min(0.35, 1.0 / max(2, sample_count + 1)))
        updated = current * (1.0 - alpha) + candidate * alpha
        updated = min(
            self.MAX_SECONDS_PER_DEGREE,
            max(self.MIN_SECONDS_PER_DEGREE, updated),
        )
        if direction is TurnDirection.LEFT:
            self.left_seconds_per_degree = updated
            self.left_turn_samples += 1
        else:
            self.right_seconds_per_degree = updated
            self.right_turn_samples += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def assess_forward(
        self,
        flow: DirectionalFlow,
        *,
        change_score: float,
        held_seconds: float,
    ) -> ForwardAssessment:
        """Classify one nominal forward command without a pre-run calibration."""
        observed = max(0.0, float(flow.magnitude_px))
        expected = self.forward_flow_px
        ratio = observed / expected if expected is not None and expected > 0.0 else None

        if not math.isfinite(change_score) or not math.isfinite(held_seconds):
            return self._uncertain(observed, expected, ratio, "invalid forward evidence")
        if not 0.03 <= held_seconds <= 0.50:
            return self._uncertain(
                observed,
                expected,
                ratio,
                "measured forward hold is outside the safe range",
            )

        static_limit = self.STATIC_FLOW_PX
        if expected is not None:
            static_limit = min(static_limit, max(0.65, expected * 0.18))
        if observed <= static_limit and change_score <= self.STATIC_CHANGE_SCORE:
            confidence = float(
                max(
                    0.55,
                    min(
                        0.95,
                        1.0
                        - 0.35 * observed / max(static_limit, 0.1)
                        - 0.20 * change_score / self.STATIC_CHANGE_SCORE,
                    ),
                )
            )
            return ForwardAssessment(
                outcome=AdaptiveForwardOutcome.BLOCKED,
                reliable=True,
                distance_cells=0.0,
                confidence=confidence,
                expected_flow_px=expected,
                observed_flow_px=observed,
                flow_ratio=ratio,
                reason="two-frame view is effectively static after forward input",
            )

        coherent = (
            flow.tracked_points >= self.MIN_TRACKED_POINTS
            and flow.inlier_ratio >= self.MIN_INLIER_RATIO
            and flow.confidence >= self.MIN_FLOW_CONFIDENCE
            and flow.dispersion_px <= max(3.0, observed * 0.90)
        )
        if not coherent:
            return self._uncertain(
                observed,
                expected,
                ratio,
                "optical flow is not coherent enough to prove a complete step",
            )
        if observed < self.MIN_FORWARD_FLOW_PX:
            return self._uncertain(
                observed,
                expected,
                ratio,
                "coherent flow is too small to prove a complete step",
            )

        consistency = 0.70
        # Scene texture changes optical-flow magnitude dramatically. During the
        # first few successful steps the learned envelope is diagnostic only.
        # Once established, reject only extreme shortfall/excess rather than
        # pretending flow pixels are a metric distance ruler.
        if ratio is not None and self.forward_samples >= 3:
            lower = 0.20
            upper = 4.00
            if ratio < lower:
                return self._uncertain(
                    observed,
                    expected,
                    ratio,
                    "forward flow is well below the learned response; partial travel is possible",
                )
            if ratio > upper:
                return self._uncertain(
                    observed,
                    expected,
                    ratio,
                    "forward flow is well above the learned response",
                )
            consistency = math.exp(-abs(math.log(max(ratio, 1e-6))) / 1.20)

        duration_ratio = held_seconds / self.forward_seconds
        if not 0.70 <= duration_ratio <= 1.35:
            return self._uncertain(
                observed,
                expected,
                ratio,
                "forward key hold differed too much from the nominal mapper pulse",
            )

        confidence = float(
            max(
                0.0,
                min(
                    1.0,
                    0.55 * flow.confidence
                    + 0.25 * min(1.0, flow.inlier_ratio)
                    + 0.20 * consistency,
                ),
            )
        )
        distance_cells = min(1.20, max(0.80, duration_ratio))
        return ForwardAssessment(
            outcome=AdaptiveForwardOutcome.MOVED,
            reliable=True,
            distance_cells=distance_cells,
            confidence=confidence,
            expected_flow_px=expected,
            observed_flow_px=observed,
            flow_ratio=ratio,
            reason="coherent visual travel confirmed",
        )

    def observe_forward(self, flow: DirectionalFlow) -> bool:
        """Update the ordinary-flow envelope after an accepted complete step."""
        observed = float(flow.magnitude_px)
        if not math.isfinite(observed) or observed < self.MIN_FORWARD_FLOW_PX:
            return False
        if self.forward_flow_px is None:
            self.forward_flow_px = observed
            self.forward_flow_deviation_px = max(0.5, float(flow.dispersion_px))
            self.forward_samples = 1
            self.updated_at = datetime.now(timezone.utc).isoformat()
            return True

        current = float(self.forward_flow_px)
        ratio = observed / max(current, 1e-6)
        if not 0.15 <= ratio <= 5.00:
            self.uncertain_forward_samples += 1
            return False

        sample_count = self.forward_samples
        base_alpha = max(0.08, min(0.30, 1.0 / max(3, sample_count + 1)))
        # Large flow is often just richer nearby texture. Learn downward faster
        # than upward so one unusually large first observation does not make
        # ordinary later steps look like partial movement.
        alpha = min(0.40, base_alpha * 1.35) if observed < current else max(0.06, base_alpha * 0.50)
        residual = abs(observed - current)
        self.forward_flow_px = current * (1.0 - alpha) + observed * alpha
        deviation = self.forward_flow_deviation_px
        if deviation is None:
            deviation = residual
        self.forward_flow_deviation_px = deviation * (1.0 - alpha) + residual * alpha
        self.forward_samples += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def record_uncertain_forward(self) -> None:
        self.uncertain_forward_samples += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()

    @property
    def turn_confidence(self) -> float:
        samples = min(self.left_turn_samples, self.right_turn_samples)
        return min(1.0, samples / 10.0)

    @property
    def forward_confidence(self) -> float:
        return min(1.0, self.forward_samples / 10.0)

    def summary(self) -> str:
        left_90 = self.left_seconds_per_degree * 90.0
        right_90 = self.right_seconds_per_degree * 90.0
        flow = (
            f"{self.forward_flow_px:.2f}px"
            if self.forward_flow_px is not None
            else "bootstrap"
        )
        return (
            f"turn90 L={left_90:.3f}s/{self.left_turn_samples} "
            f"R={right_90:.3f}s/{self.right_turn_samples}; "
            f"forward={flow}/{self.forward_samples}"
        )

    def _uncertain(
        self,
        observed: float,
        expected: float | None,
        ratio: float | None,
        reason: str,
    ) -> ForwardAssessment:
        return ForwardAssessment(
            outcome=AdaptiveForwardOutcome.UNCERTAIN,
            reliable=False,
            distance_cells=None,
            confidence=0.0,
            expected_flow_px=expected,
            observed_flow_px=observed,
            flow_ratio=ratio,
            reason=reason,
        )
