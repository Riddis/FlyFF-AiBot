from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class PangDetection:
    visible: bool
    score: float
    center: tuple[int, int] | None


class PangDetector:
    def __init__(self, template_path: Path, threshold: float = 0.82) -> None:
        self.threshold = float(threshold)
        self.template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
        if self.template is None:
            raise FileNotFoundError(f"Missing Pang template: {template_path}")

    def detect(self, frame: np.ndarray) -> PangDetection:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        th, tw = self.template.shape[:2]
        if gray.shape[0] < th or gray.shape[1] < tw:
            return PangDetection(False, 0.0, None)
        result = cv.matchTemplate(gray, self.template, cv.TM_CCOEFF_NORMED)
        _, score, _, location = cv.minMaxLoc(result)
        if score < self.threshold:
            return PangDetection(False, float(score), None)
        return PangDetection(
            True,
            float(score),
            (location[0] + tw // 2, location[1] + th // 2),
        )
