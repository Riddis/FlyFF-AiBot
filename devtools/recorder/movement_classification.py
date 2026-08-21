from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .keyboard import FORWARD_BIT


@dataclass(frozen=True, slots=True)
class MovementControlReport:
    scheme: str
    confidence: float
    moving_intervals: int
    keyboard_explained_intervals: int
    unexplained_intervals: int
    total_distance_native: float
    keyboard_explained_distance_native: float
    unexplained_distance_native: float
    keyboard_explained_distance_ratio: float
    minimum_evidence_met: bool

    @property
    def direct_movement_labels_allowed(self) -> bool:
        return self.scheme == "keyboard_wasd" and self.minimum_evidence_met

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["direct_movement_labels_allowed"] = self.direct_movement_labels_allowed
        return payload


class MovementControlClassifier:
    """Infer keyboard-vs-click movement from displacement and recorded key state.

    The recorder intentionally has no user-facing role selector. A movement
    interval is considered keyboard-explained when the physical forward key was
    held at either edge of the interval. This also covers W/Z held through EVA,
    because the key mask preserves simultaneous movement state even when the
    exported action for that frame is CAST_EVA.
    """

    def __init__(
        self,
        *,
        minimum_distance_native: float = 5.0,
        minimum_moving_intervals: int = 20,
        movement_epsilon_native: float = 0.10,
        keyboard_threshold: float = 0.75,
        click_threshold: float = 0.20,
    ) -> None:
        if minimum_distance_native <= 0.0:
            raise ValueError("minimum_distance_native must be positive")
        if minimum_moving_intervals < 1:
            raise ValueError("minimum_moving_intervals must be positive")
        if movement_epsilon_native <= 0.0:
            raise ValueError("movement_epsilon_native must be positive")
        if not 0.0 <= click_threshold < keyboard_threshold <= 1.0:
            raise ValueError("movement classification thresholds are invalid")
        self.minimum_distance_native = float(minimum_distance_native)
        self.minimum_moving_intervals = int(minimum_moving_intervals)
        self.movement_epsilon_native = float(movement_epsilon_native)
        self.keyboard_threshold = float(keyboard_threshold)
        self.click_threshold = float(click_threshold)
        self._previous_position: tuple[float, float] | None = None
        self._previous_mask = 0
        self._previous_focused = False
        self._moving_intervals = 0
        self._keyboard_intervals = 0
        self._unexplained_intervals = 0
        self._total_distance = 0.0
        self._keyboard_distance = 0.0
        self._unexplained_distance = 0.0

    def observe(
        self,
        *,
        x: float,
        z: float,
        focused: bool,
        key_mask: int,
    ) -> None:
        current = (float(x), float(z))
        if not all(math.isfinite(value) for value in current):
            self._previous_position = None
            self._previous_mask = int(key_mask)
            self._previous_focused = bool(focused)
            return
        previous = self._previous_position
        previous_mask = self._previous_mask
        previous_focused = self._previous_focused
        self._previous_position = current
        self._previous_mask = int(key_mask)
        self._previous_focused = bool(focused)
        if previous is None or not (focused and previous_focused):
            return
        distance = math.hypot(current[0] - previous[0], current[1] - previous[1])
        if not math.isfinite(distance) or distance < self.movement_epsilon_native:
            return
        self._moving_intervals += 1
        self._total_distance += distance
        explained = bool((int(key_mask) | int(previous_mask)) & FORWARD_BIT)
        if explained:
            self._keyboard_intervals += 1
            self._keyboard_distance += distance
        else:
            self._unexplained_intervals += 1
            self._unexplained_distance += distance

    def report(self) -> MovementControlReport:
        minimum_evidence = bool(
            self._moving_intervals >= self.minimum_moving_intervals
            and self._total_distance >= self.minimum_distance_native
        )
        ratio = self._keyboard_distance / max(1.0e-9, self._total_distance)
        if not minimum_evidence:
            scheme = "unknown"
            confidence = 0.0
        elif ratio >= self.keyboard_threshold:
            scheme = "keyboard_wasd"
            confidence = min(1.0, (ratio - self.keyboard_threshold) / max(1.0e-9, 1.0 - self.keyboard_threshold))
        elif ratio <= self.click_threshold:
            scheme = "click_to_move"
            confidence = min(1.0, (self.click_threshold - ratio) / max(1.0e-9, self.click_threshold))
        else:
            scheme = "mixed"
            midpoint = 0.5 * (self.click_threshold + self.keyboard_threshold)
            half_span = 0.5 * (self.keyboard_threshold - self.click_threshold)
            confidence = max(0.0, 1.0 - abs(ratio - midpoint) / max(1.0e-9, half_span))
        return MovementControlReport(
            scheme=scheme,
            confidence=float(confidence),
            moving_intervals=self._moving_intervals,
            keyboard_explained_intervals=self._keyboard_intervals,
            unexplained_intervals=self._unexplained_intervals,
            total_distance_native=float(self._total_distance),
            keyboard_explained_distance_native=float(self._keyboard_distance),
            unexplained_distance_native=float(self._unexplained_distance),
            keyboard_explained_distance_ratio=float(ratio),
            minimum_evidence_met=minimum_evidence,
        )
