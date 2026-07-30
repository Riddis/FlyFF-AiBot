from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2 as cv
import numpy as np

from libs.DigitReader import DigitReader


@dataclass(frozen=True, slots=True)
class KillCounterAnchor:
    """Top-left of the trophy icon and the detected FlyFF UI scale."""

    frame_width: int
    frame_height: int
    trophy_x: int
    trophy_y: int
    scale: float
    confidence: float


@dataclass(frozen=True, slots=True)
class KillCounterReading:
    kills: int | None
    penya: int | None
    anchor: KillCounterAnchor


class _Template(NamedTuple):
    image: np.ndarray
    mask: np.ndarray


class DynamicKillCounterReader:
    """Locate and read FlyFF's movable kill/Penya tracker.

    The expensive whole-frame search is intentionally event driven. It runs
    only for a new frame size, initial acquisition, or after the cached icon
    pair has genuinely disappeared for several frames. Normal reads use two
    tiny cached icon checks and fixed geometry relative to the trophy icon.

    We locate the stable trophy/coin glyph pair rather than OCRing the words
    "Monsters killed" and "Penya" on every frame. That is substantially more
    robust against the transparent and hover panel backgrounds while still
    making the counter position resolution- and placement-independent.
    """

    VERSION = "1.0-event-driven-icon-pair"
    RELOCATION_MISS_THRESHOLD = 4
    SCAN_RETRY_FRAMES = 15
    LOCAL_SEARCH_RADIUS_PX = 8
    TEMPLATE_THRESHOLD = 0.78
    PAIR_THRESHOLD = 0.80
    PRIMARY_SCALES = (1.0,)
    FALLBACK_SCALES = (0.70, 0.80, 0.90, 1.10, 1.20, 1.35, 1.50)

    # Geometry in pixels relative to the canonical 26x26 trophy template.
    _COIN_DX = 0
    _COIN_DY = 26
    _KILLS_BOX = (115, -4, 215, 31)
    _PENYA_BOX = (70, 22, 215, 58)
    _TRACKING_BOX = (-5, -7, 224, 62)

    def __init__(
        self,
        *,
        asset_root: Path | None = None,
        digit_reader: DigitReader | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        root = asset_root or (project_root / "assets" / "ui")
        self._trophy = self._load_template(root / "kill_counter_trophy.png")
        self._coin = self._load_template(root / "kill_counter_penya.png")
        self._digit_reader = digit_reader or DigitReader(
            project_root / "assets" / "digits",
            threshold=0.85,
        )
        self._anchors: dict[tuple[int, int], KillCounterAnchor] = {}
        self._misses: dict[tuple[int, int], int] = {}
        self._last_shape: tuple[int, int] | None = None
        self._sequence = 0
        self._last_scan_sequence: dict[tuple[int, int], int] = {}
        self.full_scan_count = 0

    @staticmethod
    def _load_template(path: Path) -> _Template:
        raw = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not load counter anchor template: {path}")
        if raw.ndim != 3 or raw.shape[2] not in (3, 4):
            raise ValueError(f"Counter template must be BGR/BGRA: {path}")
        image = raw[:, :, :3]
        if raw.shape[2] == 4:
            mask = raw[:, :, 3]
        else:
            mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        if int(np.count_nonzero(mask)) < 8:
            raise ValueError(f"Counter template mask is empty: {path}")
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
        raise ValueError(f"Unsupported counter frame shape: {frame.shape}")

    @staticmethod
    def _scaled(template: _Template, scale: float) -> _Template:
        height, width = template.image.shape[:2]
        target = (
            max(6, int(round(width * scale))),
            max(6, int(round(height * scale))),
        )
        image = cv.resize(template.image, target, interpolation=cv.INTER_LINEAR)
        mask = cv.resize(template.mask, target, interpolation=cv.INTER_NEAREST)
        return _Template(image=image, mask=mask)

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
        height, width = frame.shape[:2]
        template_height, template_width = template.image.shape[:2]
        left = max(0, expected_x - radius)
        top = max(0, expected_y - radius)
        right = min(width, expected_x + radius + template_width + 1)
        bottom = min(height, expected_y + radius + template_height + 1)
        if right - left < template_width or bottom - top < template_height:
            return -1.0, (expected_x, expected_y)
        local = frame[top:bottom, left:right]
        result = DynamicKillCounterReader._masked_match(local, template)
        _minimum, maximum, _minimum_location, maximum_location = cv.minMaxLoc(result)
        return float(maximum), (
            int(left + maximum_location[0]),
            int(top + maximum_location[1]),
        )

    @staticmethod
    def _bright_text_density(
        frame: np.ndarray,
        *,
        trophy_x: int,
        trophy_y: int,
        scale: float,
    ) -> float:
        """Cheap label-shape check to reject unrelated trophy-like icons."""
        height, width = frame.shape[:2]
        x0 = int(round(trophy_x + 29 * scale))
        x1 = int(round(trophy_x + 155 * scale))
        y0 = int(round(trophy_y - 2 * scale))
        y1 = int(round(trophy_y + 21 * scale))
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        crop = frame[y0:y1, x0:x1]
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        # White label glyphs have a small but consistent footprint. The hover
        # state may recolour them, so this is a ranking bonus rather than a hard
        # requirement.
        return float(np.count_nonzero(gray >= 145)) / float(gray.size)

    def _validate_anchor(
        self,
        frame: np.ndarray,
        anchor: KillCounterAnchor,
    ) -> KillCounterAnchor | None:
        scale = float(anchor.scale)
        trophy = self._scaled(self._trophy, scale)
        coin = self._scaled(self._coin, scale)
        radius = max(3, int(round(self.LOCAL_SEARCH_RADIUS_PX * scale)))
        trophy_score, trophy_location = self._best_local_match(
            frame,
            trophy,
            expected_x=int(anchor.trophy_x),
            expected_y=int(anchor.trophy_y),
            radius=radius,
        )
        if trophy_score < self.TEMPLATE_THRESHOLD:
            return None
        expected_coin_x = int(round(trophy_location[0] + self._COIN_DX * scale))
        expected_coin_y = int(round(trophy_location[1] + self._COIN_DY * scale))
        coin_score, _coin_location = self._best_local_match(
            frame,
            coin,
            expected_x=expected_coin_x,
            expected_y=expected_coin_y,
            radius=radius,
        )
        pair_score = min(trophy_score, coin_score)
        if pair_score < self.PAIR_THRESHOLD:
            return None
        return KillCounterAnchor(
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            trophy_x=int(trophy_location[0]),
            trophy_y=int(trophy_location[1]),
            scale=scale,
            confidence=float(0.5 * trophy_score + 0.5 * coin_score),
        )

    def _scan_scales(
        self,
        frame: np.ndarray,
        scales: tuple[float, ...],
    ) -> KillCounterAnchor | None:
        frame_height, frame_width = frame.shape[:2]
        best: tuple[float, KillCounterAnchor] | None = None

        for scale in scales:
            trophy = self._scaled(self._trophy, scale)
            coin = self._scaled(self._coin, scale)
            if (
                frame_height < trophy.image.shape[0]
                or frame_width < trophy.image.shape[1]
            ):
                continue
            result = self._masked_match(frame, trophy)
            # Only inspect a small number of strongest, spatially distinct
            # trophy candidates at each scale.
            working = result.copy()
            for _ in range(12):
                _minimum, trophy_score, _minimum_location, trophy_location = (
                    cv.minMaxLoc(working)
                )
                if trophy_score < self.TEMPLATE_THRESHOLD:
                    break
                tx, ty = int(trophy_location[0]), int(trophy_location[1])
                suppress = max(8, int(round(18 * scale)))
                sx0 = max(0, tx - suppress)
                sy0 = max(0, ty - suppress)
                sx1 = min(working.shape[1], tx + suppress + 1)
                sy1 = min(working.shape[0], ty + suppress + 1)
                working[sy0:sy1, sx0:sx1] = -1.0

                expected_coin_x = int(round(tx + self._COIN_DX * scale))
                expected_coin_y = int(round(ty + self._COIN_DY * scale))
                radius = max(4, int(round(7 * scale)))
                coin_score, _coin_location = self._best_local_match(
                    frame,
                    coin,
                    expected_x=expected_coin_x,
                    expected_y=expected_coin_y,
                    radius=radius,
                )
                pair_score = min(float(trophy_score), float(coin_score))
                if pair_score < self.PAIR_THRESHOLD:
                    continue
                text_density = self._bright_text_density(
                    frame,
                    trophy_x=tx,
                    trophy_y=ty,
                    scale=scale,
                )
                score = (
                    0.46 * float(trophy_score)
                    + 0.46 * float(coin_score)
                    + 0.08 * min(1.0, text_density / 0.08)
                )
                anchor = KillCounterAnchor(
                    frame_width=frame_width,
                    frame_height=frame_height,
                    trophy_x=tx,
                    trophy_y=ty,
                    scale=float(scale),
                    confidence=float(score),
                )
                if best is None or score > best[0]:
                    best = (score, anchor)

        return None if best is None else best[1]

    def _scan(self, frame: np.ndarray) -> KillCounterAnchor | None:
        self.full_scan_count += 1
        # FlyFF normally keeps the UI at one pixel scale when only the client
        # resolution changes. Try that common case first; multi-scale matching
        # is a rare fallback for DPI/UI-scale changes.
        exact = self._scan_scales(frame, self.PRIMARY_SCALES)
        if exact is not None:
            return exact
        return self._scan_scales(frame, self.FALLBACK_SCALES)

    def _scan_is_due(self, shape: tuple[int, int]) -> bool:
        previous = self._last_scan_sequence.get(shape)
        return (
            previous is None
            or self._sequence - previous >= self.SCAN_RETRY_FRAMES
        )

    def locate(self, frame: np.ndarray) -> KillCounterAnchor | None:
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

        if not resized and not self._scan_is_due(shape):
            return anchor

        self._last_scan_sequence[shape] = self._sequence
        discovered = self._scan(frame)
        if discovered is None:
            return None
        self._anchors[shape] = discovered
        self._misses[shape] = 0
        return discovered

    @staticmethod
    def _scaled_box(
        anchor: KillCounterAnchor,
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = box
        scale = float(anchor.scale)
        return (
            int(round(anchor.trophy_x + left * scale)),
            int(round(anchor.trophy_y + top * scale)),
            int(round(anchor.trophy_x + right * scale)),
            int(round(anchor.trophy_y + bottom * scale)),
        )

    @classmethod
    def tracking_bounds(
        cls,
        anchor: KillCounterAnchor,
        *,
        frame_shape: tuple[int, int] | None = None,
    ) -> tuple[int, int, int, int]:
        """Return the full compact tracker rectangle for Bot Vision."""
        left, top, right, bottom = cls._scaled_box(anchor, cls._TRACKING_BOX)
        if frame_shape is not None:
            height, width = int(frame_shape[0]), int(frame_shape[1])
            left = max(0, min(width - 1, left))
            right = max(left, min(width - 1, right))
            top = max(0, min(height - 1, top))
            bottom = max(top, min(height - 1, bottom))
        return left, top, right, bottom

    @staticmethod
    def _crop_canonical(
        frame: np.ndarray,
        anchor: KillCounterAnchor,
        box: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        left, top, right, bottom = DynamicKillCounterReader._scaled_box(anchor, box)
        height, width = frame.shape[:2]
        left, right = max(0, left), min(width, right)
        top, bottom = max(0, top), min(height, bottom)
        if right <= left or bottom <= top:
            return None
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        if abs(anchor.scale - 1.0) > 0.025:
            canonical_width = max(1, int(round((right - left) / anchor.scale)))
            canonical_height = max(1, int(round((bottom - top) / anchor.scale)))
            crop = cv.resize(
                crop,
                (canonical_width, canonical_height),
                interpolation=cv.INTER_LINEAR,
            )
        return crop

    def read(self, frame: np.ndarray) -> KillCounterReading | None:
        if frame is None or frame.size == 0:
            return None
        frame = self._as_bgr(frame)
        anchor = self.locate(frame)
        if anchor is None:
            return None
        kills_crop = self._crop_canonical(frame, anchor, self._KILLS_BOX)
        penya_crop = self._crop_canonical(frame, anchor, self._PENYA_BOX)
        kills = (
            None
            if kills_crop is None
            else self._digit_reader.read_number(kills_crop)
        )
        penya = (
            None
            if penya_crop is None
            else self._digit_reader.read_number(penya_crop)
        )
        return KillCounterReading(
            kills=None if kills is None else max(0, int(kills)),
            penya=None if penya is None else max(0, int(penya)),
            anchor=anchor,
        )

    def invalidate(self) -> None:
        """Forget cached positions, forcing acquisition on the next frame."""
        self._anchors.clear()
        self._misses.clear()
        self._last_scan_sequence.clear()
        self._last_shape = None
