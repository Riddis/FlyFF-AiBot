from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2 as cv
import numpy as np
from libs.DigitReader import DigitReader


@dataclass(frozen=True, slots=True)
class PlayerStatusAnchor:
    frame_width: int
    frame_height: int
    panel_x: int
    panel_y: int
    scale: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PlayerHealthReading:
    current_hp: int
    maximum_hp: int
    anchor: PlayerStatusAnchor


class _Template(NamedTuple):
    image: np.ndarray
    mask: np.ndarray


class DynamicPlayerStatusReader:
    """Locate FlyFF's player-status panel and OCR current/maximum HP."""

    VERSION = "1.0-event-driven-status-panel"
    TEMPLATE_THRESHOLD = 0.82
    LOCAL_SEARCH_RADIUS_PX = 6
    RELOCATION_MISS_THRESHOLD = 3
    SCAN_RETRY_FRAMES = 10
    PRIMARY_SCALES = (1.0,)
    FALLBACK_SCALES = (0.80, 0.90, 1.10, 1.20)

    # Canonical geometry in the supplied 218x110 current-client panel image.
    _CURRENT_HP_BOX = (101, 26, 153, 41)
    _MAXIMUM_HP_BOX = (153, 26, 209, 41)

    def __init__(
        self,
        *,
        asset_root: Path | None = None,
        digit_reader: DigitReader | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        root = asset_root or (project_root / "assets" / "ui")
        self._panel = self._load_template(root / "player_status.png")
        self._digit_reader = digit_reader or DigitReader(
            project_root / "assets" / "digits",
            threshold=0.85,
        )
        self._anchors: dict[tuple[int, int], PlayerStatusAnchor] = {}
        self._misses: dict[tuple[int, int], int] = {}
        self._last_shape: tuple[int, int] | None = None
        self._sequence = 0
        self._last_scan_sequence: dict[tuple[int, int], int] = {}
        self.full_scan_count = 0

    @staticmethod
    def _load_template(path: Path) -> _Template:
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load player status template: {path}")
        height, width = image.shape[:2]
        if width < 200 or height < 90:
            raise ValueError(f"Player status template is unexpectedly small: {path}")

        # Preserve the panel chrome and stable HP/MP/EXP labels while excluding
        # the character portrait, name/level, changing bars/numbers, and Lock XP.
        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[3:24, 4:185] = 0
        mask[22:80, 3:70] = 0
        mask[23:75, 94:width] = 0
        mask[78:101, 72:154] = 0
        if int(np.count_nonzero(mask)) < 500:
            raise ValueError(f"Player status template mask is empty: {path}")
        return _Template(image=image, mask=mask)

    @staticmethod
    def _as_bgr(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        raise ValueError(f"Unsupported player-status frame shape: {frame.shape}")

    @staticmethod
    def _scaled(template: _Template, scale: float) -> _Template:
        height, width = template.image.shape[:2]
        target = (
            max(20, int(round(width * scale))),
            max(20, int(round(height * scale))),
        )
        return _Template(
            image=cv.resize(template.image, target, interpolation=cv.INTER_LINEAR),
            mask=cv.resize(template.mask, target, interpolation=cv.INTER_NEAREST),
        )

    @staticmethod
    def _masked_match(frame: np.ndarray, template: _Template) -> np.ndarray:
        result = cv.matchTemplate(
            frame,
            template.image,
            cv.TM_CCORR_NORMED,
            mask=template.mask,
        )
        return np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)

    @staticmethod
    def _best_local_match(
        frame: np.ndarray,
        template: _Template,
        *,
        expected_x: int,
        expected_y: int,
        radius: int,
    ) -> tuple[float, tuple[int, int]]:
        frame_height, frame_width = frame.shape[:2]
        template_height, template_width = template.image.shape[:2]
        left = max(0, expected_x - radius)
        top = max(0, expected_y - radius)
        right = min(frame_width, expected_x + radius + template_width + 1)
        bottom = min(frame_height, expected_y + radius + template_height + 1)
        if right - left < template_width or bottom - top < template_height:
            return -1.0, (expected_x, expected_y)
        result = DynamicPlayerStatusReader._masked_match(
            frame[top:bottom, left:right],
            template,
        )
        _minimum, maximum, _minimum_location, maximum_location = cv.minMaxLoc(result)
        return float(maximum), (
            int(left + maximum_location[0]),
            int(top + maximum_location[1]),
        )

    def _validate_anchor(
        self,
        frame: np.ndarray,
        anchor: PlayerStatusAnchor,
    ) -> PlayerStatusAnchor | None:
        template = self._scaled(self._panel, anchor.scale)
        score, location = self._best_local_match(
            frame,
            template,
            expected_x=anchor.panel_x,
            expected_y=anchor.panel_y,
            radius=max(3, int(round(self.LOCAL_SEARCH_RADIUS_PX * anchor.scale))),
        )
        if score < self.TEMPLATE_THRESHOLD:
            return None
        return PlayerStatusAnchor(
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            panel_x=location[0],
            panel_y=location[1],
            scale=anchor.scale,
            confidence=score,
        )

    def _scan_scales(
        self,
        frame: np.ndarray,
        scales: tuple[float, ...],
    ) -> PlayerStatusAnchor | None:
        best: tuple[float, PlayerStatusAnchor] | None = None
        for scale in scales:
            template = self._scaled(self._panel, scale)
            if (
                frame.shape[0] < template.image.shape[0]
                or frame.shape[1] < template.image.shape[1]
            ):
                continue
            result = self._masked_match(frame, template)
            _minimum, score, _minimum_location, location = cv.minMaxLoc(result)
            if score < self.TEMPLATE_THRESHOLD:
                continue
            anchor = PlayerStatusAnchor(
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                panel_x=int(location[0]),
                panel_y=int(location[1]),
                scale=float(scale),
                confidence=float(score),
            )
            if best is None or score > best[0]:
                best = (float(score), anchor)
        return None if best is None else best[1]

    def _scan(self, frame: np.ndarray) -> PlayerStatusAnchor | None:
        self.full_scan_count += 1
        exact = self._scan_scales(frame, self.PRIMARY_SCALES)
        if exact is not None:
            return exact
        return self._scan_scales(frame, self.FALLBACK_SCALES)

    def locate(self, frame: np.ndarray) -> PlayerStatusAnchor | None:
        if frame is None or frame.size == 0:
            return None
        frame = self._as_bgr(frame)
        self._sequence += 1
        shape = (int(frame.shape[1]), int(frame.shape[0]))
        resized = self._last_shape is not None and shape != self._last_shape
        self._last_shape = shape

        anchor = self._anchors.get(shape)
        if anchor is not None and not resized:
            validated = self._validate_anchor(frame, anchor)
            if validated is not None:
                self._anchors[shape] = validated
                self._misses[shape] = 0
                return validated
            misses = self._misses.get(shape, 0) + 1
            self._misses[shape] = misses
            if misses < self.RELOCATION_MISS_THRESHOLD:
                return anchor

        previous_scan = self._last_scan_sequence.get(shape)
        if (
            not resized
            and previous_scan is not None
            and self._sequence - previous_scan < self.SCAN_RETRY_FRAMES
        ):
            return anchor
        self._last_scan_sequence[shape] = self._sequence
        discovered = self._scan(frame)
        if discovered is not None:
            self._anchors[shape] = discovered
            self._misses[shape] = 0
        return discovered

    @staticmethod
    def _crop_canonical(
        frame: np.ndarray,
        anchor: PlayerStatusAnchor,
        box: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        left, top, right, bottom = box
        scale = float(anchor.scale)
        left = int(round(anchor.panel_x + left * scale))
        top = int(round(anchor.panel_y + top * scale))
        right = int(round(anchor.panel_x + right * scale))
        bottom = int(round(anchor.panel_y + bottom * scale))
        frame_height, frame_width = frame.shape[:2]
        left, right = max(0, left), min(frame_width, right)
        top, bottom = max(0, top), min(frame_height, bottom)
        if right <= left or bottom <= top:
            return None
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        if abs(scale - 1.0) > 0.025:
            crop = cv.resize(
                crop,
                (
                    max(1, int(round((right - left) / scale))),
                    max(1, int(round((bottom - top) / scale))),
                ),
                interpolation=cv.INTER_LINEAR,
            )
        return crop

    def read(self, frame: np.ndarray) -> PlayerHealthReading | None:
        if frame is None or frame.size == 0:
            return None
        frame = self._as_bgr(frame)
        anchor = self.locate(frame)
        if anchor is None:
            return None
        current_crop = self._crop_canonical(frame, anchor, self._CURRENT_HP_BOX)
        maximum_crop = self._crop_canonical(frame, anchor, self._MAXIMUM_HP_BOX)
        if current_crop is None or maximum_crop is None:
            return None
        current_hp = self._digit_reader.read_segmented_number(current_crop)
        maximum_hp = self._digit_reader.read_segmented_number(maximum_crop)
        if (
            current_hp is None
            or maximum_hp is None
            or current_hp <= 0
            or maximum_hp <= 0
            or current_hp > maximum_hp
        ):
            return None
        return PlayerHealthReading(
            current_hp=int(current_hp),
            maximum_hp=int(maximum_hp),
            anchor=anchor,
        )

    def invalidate(self) -> None:
        self._anchors.clear()
        self._misses.clear()
        self._last_scan_sequence.clear()
        self._last_shape = None
