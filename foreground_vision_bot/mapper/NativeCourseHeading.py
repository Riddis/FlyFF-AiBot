from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from position import PlayerPose


@dataclass(frozen=True, slots=True)
class NativeCourseReading:
    angle_deg: float
    displacement: float
    path_length: float
    straightness: float
    sample_count: int


class NativeCourseHeadingTracker:
    """Derive a trustworthy facing hint from native X/Z movement.

    This is deliberately conservative. A course is emitted only while the
    player has translated far enough along a nearly straight path. Stationary,
    turning-in-place, curved, and teleport-like samples produce no reference,
    so they cannot incorrectly invalidate a valid minimap arrow reading.
    """

    VERSION = "1.0-straight-native-course"

    def __init__(
        self,
        *,
        maximum_age_seconds: float = 1.25,
        minimum_displacement_units: float = 3.2,
        minimum_path_straightness: float = 0.92,
        maximum_segment_units: float = 30.0,
        maximum_vertical_units: float = 8.0,
        maximum_samples: int = 24,
    ) -> None:
        self.maximum_age_seconds = float(maximum_age_seconds)
        self.minimum_displacement_units = float(minimum_displacement_units)
        self.minimum_path_straightness = float(minimum_path_straightness)
        self.maximum_segment_units = float(maximum_segment_units)
        self.maximum_vertical_units = float(maximum_vertical_units)
        self._samples: deque[PlayerPose] = deque(maxlen=max(3, int(maximum_samples)))

    def reset(self) -> None:
        self._samples.clear()

    def update(self, pose: PlayerPose | None) -> NativeCourseReading | None:
        if pose is None:
            return None
        if not all(math.isfinite(value) for value in (pose.x, pose.y, pose.z, pose.timestamp)):
            self.reset()
            return None

        if self._samples:
            previous = self._samples[-1]
            dt = float(pose.timestamp - previous.timestamp)
            horizontal = math.hypot(pose.x - previous.x, pose.z - previous.z)
            vertical = abs(pose.y - previous.y)
            if (
                dt <= 0.0
                or dt > self.maximum_age_seconds * 1.5
                or horizontal > self.maximum_segment_units
                or vertical > self.maximum_vertical_units
            ):
                self.reset()

        self._samples.append(pose)
        cutoff = float(pose.timestamp - self.maximum_age_seconds)
        while len(self._samples) > 2 and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

        if len(self._samples) < 3:
            return None

        first = self._samples[0]
        delta_x = float(pose.x - first.x)
        delta_z = float(pose.z - first.z)
        displacement = math.hypot(delta_x, delta_z)
        if displacement < self.minimum_displacement_units:
            return None

        path_length = 0.0
        for before, after in zip(self._samples, list(self._samples)[1:]):
            path_length += math.hypot(after.x - before.x, after.z - before.z)
        if path_length <= 1e-9:
            return None
        straightness = displacement / path_length
        if straightness < self.minimum_path_straightness:
            return None

        # Project convention: 0° = +Z/north, 90° = +X/east.
        angle = math.degrees(math.atan2(delta_x, delta_z)) % 360.0
        return NativeCourseReading(
            angle_deg=float(angle),
            displacement=float(displacement),
            path_length=float(path_length),
            straightness=float(straightness),
            sample_count=len(self._samples),
        )
