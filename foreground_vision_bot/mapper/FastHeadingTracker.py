from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from mapper.MinimapHeading import (
    HeadingReading,
    MinimapHeadingDetector,
)


@dataclass(frozen=True)
class FastHeadingState:
    reading: HeadingReading | None
    timestamp: float
    consecutive_misses: int

    @property
    def usable(self) -> bool:
        return (
            self.reading is not None
            and self.reading.confidence >= 0.30
            and self.consecutive_misses <= 3
        )


class FastHeadingTracker:
    """
    Thin reusable wrapper for the farming/training loop.

    Call update(frame) once per newly captured frame. It performs no sleeping
    and no internal frame capture.
    """

    def __init__(
        self,
        detector: MinimapHeadingDetector | None = None,
    ) -> None:
        self.detector = detector or MinimapHeadingDetector()
        self.consecutive_misses = 0
        self.last_state = FastHeadingState(
            reading=None,
            timestamp=perf_counter(),
            consecutive_misses=0,
        )

    def reset(self) -> None:
        self.detector.reset_fast()
        self.consecutive_misses = 0
        self.last_state = FastHeadingState(
            reading=None,
            timestamp=perf_counter(),
            consecutive_misses=0,
        )

    def update(self, frame) -> FastHeadingState:
        reading = self.detector.read_fast(frame)
        if reading is None or reading.is_stale:
            self.consecutive_misses += 1
        else:
            self.consecutive_misses = 0

        self.last_state = FastHeadingState(
            reading=reading,
            timestamp=perf_counter(),
            consecutive_misses=self.consecutive_misses,
        )
        return self.last_state
