from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import time
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np
from assets.Assets import MobInfo
from libs.ActionExecutor import ActionExecutor, BotAction
from libs.ComputerVision import ComputerVision as CV
from libs.DigitReader import DigitReader
from libs.HumanKeyboard import HumanKeyboard

if TYPE_CHECKING:
    from capture_service import CaptureService
    from runtime_bus import RuntimeBus


Point = tuple[int, int]


@dataclass(frozen=True)
class MobTemplate:
    image: np.ndarray
    height_offset: int


class Bot:
    """
    RL-facing Flyff game adapter.

    The old rule-based farming loop has intentionally been removed. This class
    now owns only the low-level game interfaces needed by reinforcement
    learning:

      - background frame capture
      - anonymous visible-mob positions
      - kill-counter reading
      - keyboard action execution

    Mob species are used only internally to locate their name templates. The
    observation returned to the RL system contains positions only.
    """

    def __init__(self) -> None:
        print(f"[CV PREVIEW] Loaded Bot module: {Path(__file__).resolve()}")
        print("[CV PREVIEW] Diagnostic overlay version: v5")

        self.config = {
            "show_frames": False,
            "show_mobs_pos_boxes": False,
            "show_mobs_pos_markers": True,
            "show_matches_text": False,
            "mob_pos_match_threshold": 0.7,
            "mob_dedup_distance_px": 20.0,
            "selected_mobs": [],
            "kill_counter_crop": (168, 198, 1360, 1415),
            "show_kill_counter_crop": False,
        }

        self.runtime_bus: RuntimeBus | None = None
        self.capture_service: CaptureService | None = None
        self.keyboard: HumanKeyboard | None = None
        self.action_executor: ActionExecutor | None = None

        self._frame_lock = Lock()
        self._rl_enabled = False
        self._last_overlay_publish_at = 0.0
        self._latest_mob_points: list[Point] = []
        self._latest_mob_points_at = 0.0

        self.digit_reader = DigitReader(
            digits_dir=Path(__file__).parent / "assets" / "digits",
            threshold=0.85,
        )

        self._mob_templates: list[MobTemplate] = []
        self._reload_mob_templates()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare_window(
        self,
        window_handler: int,
        runtime_bus: RuntimeBus,
        capture_service: CaptureService,
    ) -> None:
        self.runtime_bus = runtime_bus
        self.capture_service = capture_service
        self.keyboard = HumanKeyboard(window_handler)
        self.action_executor = ActionExecutor(self.keyboard)
        self._emit("msg_green", "RL bot is ready.")

    def start(self) -> None:
        """
        Enable RL control.

        This no longer starts an internal farming thread. The Gymnasium
        environment controls actions by calling ActionExecutor.
        """
        self._require_setup()
        self._rl_enabled = True
        self._emit("msg_green", "RL control enabled.")

    def stop(self) -> None:
        self._rl_enabled = False

        if self.action_executor is not None:
            self.action_executor.stop_movement()

        self._emit("msg_yellow", "RL control stopped.")

    def close(self) -> None:
        self.stop()
        if self.capture_service is not None:
            self.capture_service.stop(5.0)
        self.release_input()
        self.runtime_bus = None
        cv.destroyAllWindows()

    def release_input(self) -> None:
        self.stop_movement()
        self.keyboard = None
        self.action_executor = None

    def set_config(self, **options) -> None:
        reload_templates = False

        for key, value in options.items():
            # Keep accepting obsolete GUI settings so the old GUI does not
            # crash while it is being replaced.
            self.config[key] = value

            if key == "selected_mobs":
                reload_templates = True

        if reload_templates:
            self._reload_mob_templates()

    def get_all_mobs(self):
        return MobInfo.get_all_mobs()

    @property
    def is_ready(self) -> bool:
        return (
            self.capture_service is not None
            and self.capture_service.active
            and self.keyboard is not None
            and self.action_executor is not None
            and self.get_frame() is not None
        )

    @property
    def rl_enabled(self) -> bool:
        return self._rl_enabled

    # ------------------------------------------------------------------
    # Public RL API
    # ------------------------------------------------------------------

    def get_visible_mobs(self) -> list[Point]:
        """
        Detect all registered mob-name templates and return anonymous positions.

        Species information is deliberately discarded. Nearby duplicate
        detections are merged because different templates or overlapping
        matches can report the same on-screen mob more than once.
        """
        frame, debug_frame = self._frame_snapshot()

        if frame is None:
            return []

        raw_points: list[Point] = []

        for template in self._mob_templates:
            try:
                matches, drawn_frame = CV.match_template_multi(
                    frame=frame,
                    crop_area=(50, -50, 50, -50),
                    template=template.image,
                    threshold=float(self.config["mob_pos_match_threshold"]),
                    box_offset=(0, template.height_offset),
                    frame_to_draw=(debug_frame if self.config["show_frames"] else None),
                    draw_rect=bool(self.config["show_mobs_pos_boxes"]),
                    draw_marker=bool(self.config["show_mobs_pos_markers"]),
                    draw_text=bool(self.config["show_matches_text"]),
                )
            except (cv.error, ValueError, TypeError) as error:
                print(f"Mob template match failed: {error}")
                continue

            raw_points.extend((int(point[0]), int(point[1])) for point in matches)

            if drawn_frame is not None:
                debug_frame = drawn_frame

        points = self._deduplicate_points(
            raw_points,
            max_distance=float(self.config["mob_dedup_distance_px"]),
        )

        with self._frame_lock:
            self._latest_mob_points = list(points)
            self._latest_mob_points_at = time()

        if self.config["show_frames"] and debug_frame is not None:
            self._publish_debug_frame(debug_frame, len(points))

        return points

    def read_kill_count(self) -> int | None:
        """
        Read the current total from the in-game kill counter.

        Delta calculation belongs to FlyffEnv, so this method is stateless.
        """
        frame, _ = self._frame_snapshot()

        if frame is None:
            return None

        top, bottom, left, right = self.config["kill_counter_crop"]

        height, width = frame.shape[:2]
        top = int(np.clip(top, 0, height))
        bottom = int(np.clip(bottom, 0, height))
        left = int(np.clip(left, 0, width))
        right = int(np.clip(right, 0, width))

        if bottom <= top or right <= left:
            return None

        crop = frame[top:bottom, left:right]

        if crop.size == 0:
            return None

        if self.config["show_kill_counter_crop"]:
            cv.imshow("Kill Counter Crop", crop)
            cv.waitKey(1)

        kills = self.digit_reader.read_number(crop)

        if kills is None:
            return None

        return max(0, int(kills))

    def execute_action(
        self,
        action: BotAction | int,
        duration: float | None = None,
    ) -> None:
        self._require_setup()

        if not self._rl_enabled:
            raise RuntimeError("RL control is disabled. Call bot.start() first.")

        assert self.action_executor is not None
        self.action_executor.execute(action, duration=duration)

    def stop_movement(self) -> None:
        if self.action_executor is not None:
            self.action_executor.stop_movement()

    def get_frame_shape(self) -> tuple[int, int] | None:
        frame, _ = self._frame_snapshot()

        if frame is None:
            return None

        height, width = frame.shape[:2]
        return height, width

    def get_frame(self) -> np.ndarray | None:
        """Return a thread-safe copy of the latest grayscale frame."""
        frame, _ = self._frame_snapshot()
        return frame

    def get_debug_frame(self) -> np.ndarray | None:
        """Return a thread-safe copy of the latest BGR frame."""
        _, debug_frame = self._frame_snapshot()
        return debug_frame

    # ------------------------------------------------------------------
    # Capture and detection internals
    # ------------------------------------------------------------------

    def _frame_snapshot(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.capture_service is not None:
            debug, frame = self.capture_service.snapshot()
            return frame, debug
        return None, None

    def build_preview(self, frame: np.ndarray) -> np.ndarray:
        self._draw_cached_mob_overlay(frame, time())
        return frame

    def _reload_mob_templates(self) -> None:
        entries = self.config.get("selected_mobs") or []

        # An empty selection means every registered mob. The detector still
        # uses individual name templates internally but discards species.
        if not entries:
            entries = MobInfo.get_all_mobs()

        if isinstance(entries, dict):
            entries = list(entries.values())

        templates: list[MobTemplate] = []

        for mob in entries:
            if not isinstance(mob, dict):
                continue

            name = mob.get("name")
            height_offset = mob.get("height_offset")

            if not name or height_offset is None:
                continue

            image_path = Path(__file__).parent / "assets" / "names" / f"{name}.png"
            image = cv.imread(
                str(image_path),
                cv.IMREAD_GRAYSCALE,
            )

            if image is None:
                print(f"Skipping missing mob template: {image_path}")
                continue

            templates.append(
                MobTemplate(
                    image=image,
                    height_offset=int(height_offset),
                )
            )

        self._mob_templates = templates
        print(f"Loaded {len(templates)} mob templates.")

    @staticmethod
    def _deduplicate_points(
        points: Iterable[Point],
        max_distance: float,
    ) -> list[Point]:
        """
        Merge nearby detections without retaining any species information.
        """
        max_distance_squared = max(0.0, max_distance) ** 2
        groups: list[list[Point]] = []

        for point in points:
            best_group: list[Point] | None = None
            best_distance: float | None = None

            for group in groups:
                center_x = sum(p[0] for p in group) / len(group)
                center_y = sum(p[1] for p in group) / len(group)
                dx = point[0] - center_x
                dy = point[1] - center_y
                distance_squared = dx * dx + dy * dy

                if distance_squared <= max_distance_squared and (
                    best_distance is None or distance_squared < best_distance
                ):
                    best_group = group
                    best_distance = distance_squared

            if best_group is None:
                groups.append([point])
            else:
                best_group.append(point)

        return [
            (
                round(sum(p[0] for p in group) / len(group)),
                round(sum(p[1] for p in group) / len(group)),
            )
            for group in groups
        ]

    def _draw_cached_mob_overlay(
        self,
        frame: np.ndarray,
        now: float,
    ) -> None:
        """Draw a diagnostic overlay and the latest detected mob positions."""
        height, width = frame.shape[:2]

        # This watermark proves that the displayed image is coming through this
        # exact overlay function, independently of mob detection.
        cv.rectangle(frame, (5, 5), (360, 72), (0, 0, 0), -1)
        cv.putText(
            frame,
            "CV OVERLAY ACTIVE - v5",
            (15, 32),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

        # Permanent center marker to verify that OpenCV drawing survives the
        # GUI resize/encoding path.
        center = (width // 2, height // 2)
        cv.drawMarker(
            frame,
            center,
            (255, 255, 0),
            markerType=cv.MARKER_CROSS,
            markerSize=50,
            thickness=4,
            line_type=cv.LINE_AA,
        )

        with self._frame_lock:
            points = list(self._latest_mob_points)
            detected_at = self._latest_mob_points_at

        age = now - detected_at if detected_at else float("inf")
        cv.putText(
            frame,
            f"cached mobs={len(points)} age={age:.2f}s",
            (15, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )

        if age > 5.0:
            return

        for index, point in enumerate(points, start=1):
            try:
                raw_x, raw_y = point
                raw_x = int(raw_x)
                raw_y = int(raw_y)
            except (TypeError, ValueError):
                continue

            # Template matching uses a 50 px crop on the left/top. Some CV
            # implementations return crop-relative coordinates, so compensate
            # when the point appears to be in cropped-frame coordinates.
            x = raw_x
            y = raw_y
            if 0 <= raw_x < max(1, width - 100):
                x = raw_x + 50
            if 0 <= raw_y < max(1, height - 100):
                y = raw_y + 50

            # Always clip into the visible frame so malformed/out-of-range
            # coordinates still produce a visible diagnostic marker.
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))

            cv.drawMarker(
                frame,
                (x, y),
                (0, 0, 255),
                markerType=cv.MARKER_CROSS,
                markerSize=42,
                thickness=4,
                line_type=cv.LINE_AA,
            )
            cv.circle(frame, (x, y), 18, (0, 255, 255), 3, cv.LINE_AA)
            cv.rectangle(
                frame,
                (max(0, x - 24), max(0, y - 24)),
                (min(width - 1, x + 24), min(height - 1, y + 24)),
                (255, 0, 255),
                2,
            )
            cv.putText(
                frame,
                f"{index}:{raw_x},{raw_y}",
                (min(width - 1, x + 25), max(20, y - 12)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )

    def _overlay_enabled(self) -> bool:
        """Return whether the preview expects detector annotations."""
        return bool(
            self.config.get("show_mobs_pos_boxes")
            or self.config.get("show_mobs_pos_markers")
            or self.config.get("show_matches_text")
        )

    def _publish_debug_frame(
        self,
        debug_frame: np.ndarray,
        mob_count: int,
    ) -> None:
        self._last_overlay_publish_at = time()

        cv.putText(
            debug_frame,
            f"Visible mobs: {mob_count}",
            (20, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )
        self._emit("debug_frame", debug_frame)

    def _require_setup(self) -> None:
        if (
            self.capture_service is None
            or not self.capture_service.active
            or self.keyboard is None
            or self.action_executor is None
        ):
            raise RuntimeError("Attach the Flyff window before controlling the bot.")

    def _emit(self, event: str, value) -> None:
        bus = self.runtime_bus
        if bus is not None:
            if event == "debug_frame":
                bus.publish_latest("debug_frame", value)
            elif event == "video_fps":
                bus.publish_latest("video_fps", value)
            elif event.startswith("msg"):
                bus.log(str(value), event)
            else:
                bus.publish_latest(event, value)
            return
