from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from typing import Callable
import json

from mapper.Explorer import Explorer
from mapper.MapLogger import MapLogger
from mapper.Calibration import RotationCalibrator
from mapper.MappingController import MappingController
from mapper.MinimapHeading import (
    MinimapHeadingDetector,
    signed_angle_delta,
)
from mapper.MotionTracker import MotionTracker
from mapper.OccupancyGrid import OccupancyGrid
from mapper.PangDetector import PangDetector


StatusCallback = Callable[[str], None]
FrameCallback = Callable[[object], None]


@dataclass(frozen=True)
class MapperConfig:
    forward_seconds: float = 0.12
    turn_left_seconds_90: float = 0.30
    turn_right_seconds_90: float = 0.30
    left_heading_sign: int = -1
    right_heading_sign: int = 1
    settle_seconds: float = 0.10
    teleport_confirmations: int = 2
    pang_threshold: float = 0.82
    save_every_steps: int = 5


class Mapper:
    def __init__(
        self,
        bot,
        status_callback: StatusCallback | None = None,
        frame_callback: FrameCallback | None = None,
        config: MapperConfig | None = None,
    ) -> None:
        if bot.keyboard is None:
            raise RuntimeError("Attach the Flyff window first.")

        self.bot = bot
        self.status_callback = status_callback or print
        self.frame_callback = frame_callback
        self.config = config or self._load_config()
        self.controller = MappingController(bot.keyboard)
        self.tracker = MotionTracker()
        self.heading_detector = MinimapHeadingDetector()
        self.heading_error_streak = 0
        self.fast_heading_miss_streak = 0
        self.grid = OccupancyGrid()
        self.explorer = Explorer()
        self.stop_event = Event()
        self._last_map_publish_at = 0.0
        self._map_publish_interval = 0.25

        template_path = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "names"
            / "Pang.png"
        )
        self.pang = PangDetector(
            template_path,
            threshold=self.config.pang_threshold,
        )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            Path(__file__).resolve().parent
            / "mapping_runs"
            / run_id
        )
        self.logger = MapLogger(self.output_dir / "mapping_steps.csv")

    def _publish_map_preview(self, force: bool = False) -> None:
        if self.frame_callback is None:
            return
        now = monotonic()
        if (
            not force
            and now - self._last_map_publish_at
            < self._map_publish_interval
        ):
            return
        self.frame_callback(self.grid.render())
        self._last_map_publish_at = now

    def stop(self) -> None:
        self.stop_event.set()
        self.controller.stop()

    def run(self) -> Path:
        self.status_callback(
            "Mapper starts in 5 seconds. Manually enter the dungeon, stand "
            "at the known spawn, and face the normal starting direction."
        )
        for remaining in range(5, 0, -1):
            if self.stop_event.wait(1.0):
                return self.output_dir
            self.status_callback(f"Mapper starting in {remaining}...")

        teleport_streak = 0
        step = 0
        initial_heading = self.heading_detector.read_stable(
            self.bot.get_frame,
            samples=7,
            delay=0.018,
        )
        if initial_heading is None:
            raise RuntimeError(
                "Could not read the minimap heading before mapping."
            )
        expected_heading = initial_heading.angle_deg
        self.heading_detector.reset_fast()

        try:
            while not self.stop_event.is_set():
                before = self._wait_for_frame()
                decision = self.explorer.decide(self.grid)

                if decision.action == "FORWARD":
                    self.controller.forward(self.config.forward_seconds)
                    commanded_forward = True
                elif decision.action == "TURN_LEFT":
                    self.controller.turn_left(
                        self.config.turn_left_seconds_90
                    )
                    self.grid.pose.heading_index = (
                        self.grid.pose.heading_index + 1
                    ) % 4
                    expected_heading = (
                        expected_heading
                        + 90.0 * self.config.left_heading_sign
                    ) % 360.0
                    commanded_forward = False
                else:
                    self.controller.turn_right(
                        self.config.turn_right_seconds_90
                    )
                    self.grid.pose.heading_index = (
                        self.grid.pose.heading_index - 1
                    ) % 4
                    expected_heading = (
                        expected_heading
                        + 90.0 * self.config.right_heading_sign
                    ) % 360.0
                    commanded_forward = False

                self.controller.settle(self.config.settle_seconds)
                after = self._wait_for_frame()

                # Single-frame, non-blocking heading update. This is the same
                # path intended for future farming/training control.
                fast_heading = self.heading_detector.read_fast(after)
                if fast_heading is None:
                    self.fast_heading_miss_streak += 1
                elif fast_heading.is_stale:
                    self.fast_heading_miss_streak += 1
                else:
                    self.fast_heading_miss_streak = 0

                motion = self.tracker.compare(
                    before,
                    after,
                    commanded_forward=commanded_forward,
                )

                if not commanded_forward:
                    actual_heading = self.heading_detector.read_stable(
                        self.bot.get_frame,
                        samples=5,
                        delay=0.015,
                    )
                    if actual_heading is None:
                        self.heading_error_streak += 1
                    else:
                        heading_error = abs(
                            signed_angle_delta(
                                expected_heading,
                                actual_heading.angle_deg,
                            )
                        )
                        if heading_error > 12.0:
                            self.heading_error_streak += 1
                            self.status_callback(
                                f"Heading mismatch: expected "
                                f"{expected_heading:.1f}°, observed "
                                f"{actual_heading.angle_deg:.1f}° "
                                f"(error {heading_error:.1f}°)."
                            )
                        else:
                            self.heading_error_streak = 0
                            expected_heading = actual_heading.angle_deg
                            self.heading_detector.reset_fast()
                            self.heading_detector.read_fast(after)

                    if self.heading_error_streak >= 2:
                        self.status_callback(
                            "Repeated minimap heading mismatch; starting "
                            "automatic turn recalibration."
                        )
                        calibrator = RotationCalibrator(
                            self.bot,
                            status_callback=self.status_callback,
                            frame_callback=self.frame_callback,
                        )
                        calibration_path = calibrator.run(manual=False)
                        data = json.loads(
                            calibration_path.read_text(encoding="utf-8")
                        )
                        self.config = MapperConfig(
                            forward_seconds=self.config.forward_seconds,
                            turn_left_seconds_90=float(
                                data["left_seconds_90"]
                            ),
                            turn_right_seconds_90=float(
                                data["right_seconds_90"]
                            ),
                            left_heading_sign=int(
                                data.get("left_heading_sign", -1)
                            ),
                            right_heading_sign=int(
                                data.get("right_heading_sign", 1)
                            ),
                            settle_seconds=self.config.settle_seconds,
                            teleport_confirmations=(
                                self.config.teleport_confirmations
                            ),
                            pang_threshold=self.config.pang_threshold,
                            save_every_steps=self.config.save_every_steps,
                        )
                        refreshed = self.heading_detector.read_stable(
                            self.bot.get_frame,
                            samples=7,
                            delay=0.018,
                        )
                        if refreshed is None:
                            raise RuntimeError(
                                "Heading unavailable after recalibration."
                            )
                        expected_heading = refreshed.angle_deg
                        self.heading_error_streak = 0

                if commanded_forward:
                    dx, dy = self.grid.DIRECTIONS[
                        self.grid.pose.heading_index
                    ]
                    next_x = self.grid.pose.x + dx
                    next_y = self.grid.pose.y + dy
                    if motion.collision_likely:
                        self.grid.mark_blocked(next_x, next_y)
                    else:
                        self.grid.pose.x = next_x
                        self.grid.pose.y = next_y
                        self.grid.mark_free(next_x, next_y)

                pang = self.pang.detect(after)
                if pang.visible:
                    self.grid.add_pang_sighting(
                        self.grid.pose.x,
                        self.grid.pose.y,
                        pang.score,
                    )

                teleport_streak = (
                    teleport_streak + 1
                    if motion.teleport_likely
                    else 0
                )
                teleport = (
                    teleport_streak >= self.config.teleport_confirmations
                )

                step += 1
                now = datetime.now().isoformat(timespec="milliseconds")
                self.logger.write(
                    {
                        "timestamp": now,
                        "step": step,
                        "x": self.grid.pose.x,
                        "y": self.grid.pose.y,
                        "heading": self.grid.pose.heading_index,
                        "action": decision.action,
                        "reason": decision.reason,
                        "change_score": round(motion.change_score, 5),
                        "median_flow_px": round(motion.median_flow_px, 3),
                        "tracked_points": motion.tracked_points,
                        "collision": motion.collision_likely,
                        "pang_visible": pang.visible,
                        "pang_score": round(pang.score, 4),
                        "teleport_suspected": teleport,
                        "fast_heading": (
                            round(fast_heading.angle_deg, 3)
                            if fast_heading is not None
                            else ""
                        ),
                        "fast_heading_confidence": (
                            round(fast_heading.confidence, 3)
                            if fast_heading is not None
                            else ""
                        ),
                        "fast_heading_stale": (
                            fast_heading.is_stale
                            if fast_heading is not None
                            else True
                        ),
                    }
                )

                self.status_callback(
                    f"map step={step} pose=({self.grid.pose.x},"
                    f"{self.grid.pose.y}) heading={self.grid.pose.heading_index} "
                    f"action={decision.action} collision="
                    f"{motion.collision_likely} pang={pang.visible} "
                    f"heading_fast="
                    f"{fast_heading.angle_deg:.1f}°/"
                    f"{fast_heading.confidence:.2f}"
                    if fast_heading is not None
                    else
                    f"map step={step} pose=({self.grid.pose.x},"
                    f"{self.grid.pose.y}) heading={self.grid.pose.heading_index} "
                    f"action={decision.action} collision="
                    f"{motion.collision_likely} pang={pang.visible} "
                    "heading_fast=unavailable"
                )

                self._publish_map_preview()

                if step % self.config.save_every_steps == 0:
                    self.grid.save(self.output_dir)

                if teleport:
                    self.grid.mark_forbidden(
                        self.grid.pose.x,
                        self.grid.pose.y,
                        radius=5,
                    )
                    self.grid.save(self.output_dir)
                    self.status_callback(
                        "Probable teleport detected. Mapper stopped at "
                        f"estimated pose ({self.grid.pose.x}, "
                        f"{self.grid.pose.y})."
                    )
                    break
        finally:
            self.controller.stop()
            self.grid.save(self.output_dir)
            self.logger.close()

        return self.output_dir

    def _load_config(self) -> MapperConfig:
        path = Path(__file__).resolve().parent / "calibration.json"
        if not path.exists():
            raise RuntimeError(
                "No mapper calibration found. Run Calibrate Mapper first."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return MapperConfig(
            turn_left_seconds_90=float(data["left_seconds_90"]),
            turn_right_seconds_90=float(data["right_seconds_90"]),
            left_heading_sign=int(data.get("left_heading_sign", -1)),
            right_heading_sign=int(data.get("right_heading_sign", 1)),
        )

    def _wait_for_frame(self):
        for _ in range(50):
            if self.stop_event.is_set():
                raise RuntimeError("Mapper stopped.")
            frame = self.bot.get_frame()
            if frame is not None:
                return frame
            sleep(0.05)
        raise RuntimeError("No game frame is available.")
