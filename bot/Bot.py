from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from time import time
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np
from assets.Assets import MobInfo
from libs.ActionExecutor import ActionExecutor, BotAction
from libs.ComputerVision import ComputerVision as CV
from libs.DigitReader import DigitReader
from libs.HumanKeyboard import VKEY, HumanKeyboard
from libs.KillCounterPanel import DynamicKillCounterReader
from libs.PlayerStatusPanel import DynamicPlayerStatusReader
from mapper.NativeCourseHeading import NativeCourseHeadingTracker
from position import (
    NativeActor,
    NativeFlyffMonsterProvider,
    NativeProcessService,
    NativeProviderAttachment,
    PlayerPose,
    PositionProvider,
    create_native_provider_attachment,
)

if TYPE_CHECKING:
    from runtime.capture_service import CaptureService, FrameSample
    from runtime.runtime_bus import RuntimeBus


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
            "show_detected_ui_elements": True,
            "show_mobs_pos_markers": True,
            # Legacy preview keys remain accepted by set_config but no longer
            # own separate GUI controls.
            "show_mobs_pos_boxes": False,
            "show_matches_text": False,
            "mob_pos_match_threshold": 0.7,
            "mob_dedup_distance_px": 20.0,
            "selected_mobs": [],
            "kill_counter_crop": (168, 198, 1360, 1415),
            "dynamic_kill_counter": True,
            "show_kill_counter_crop": False,
            "selected_map_name": None,
            "eva_hotkey": "F1",
            "show_native_monsters_on_map": True,
            "native_monster_map_refresh_seconds": 0.5,
            "native_monster_local_radius_cells": 50,
        }

        self.runtime_bus: RuntimeBus | None = None
        self.capture_service: CaptureService | None = None
        self.keyboard: HumanKeyboard | None = None
        self.action_executor: ActionExecutor | None = None
        self.position_provider: PositionProvider | None = None
        self.monster_provider: NativeFlyffMonsterProvider | None = None
        self.native_process_service: NativeProcessService | None = None
        self.native_provider_attachment: NativeProviderAttachment | None = None

        self._frame_lock = Lock()
        self._heading_lock = RLock()
        self._kill_counter_lock = RLock()
        self._latest_counter_reading = None
        self._latest_counter_reading_at = 0.0
        self._player_status_lock = RLock()
        self._rl_enabled = False
        self._latest_mob_points: list[Point] = []
        self._latest_mob_points_at = 0.0
        self._preview_detection_interval = 0.25
        self._heading_preview_detector = None
        self._latest_heading_reading = None
        self._latest_heading_reading_at = 0.0
        self._last_heading_error_at = 0.0
        self._native_course_tracker = NativeCourseHeadingTracker()
        self._native_map_overlay = None
        self._native_map_overlay_name: str | None = None
        self._last_native_map_publish_at = 0.0
        self._last_native_map_error_at = 0.0

        self.digit_reader = DigitReader(
            digits_dir=Path(__file__).resolve().parents[1] / "assets" / "digits",
            threshold=0.85,
        )
        self.kill_counter_reader = DynamicKillCounterReader(
            digit_reader=self.digit_reader,
        )
        self.player_status_reader = DynamicPlayerStatusReader(
            digit_reader=self.digit_reader,
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
        self._close_native_attachment()
        self._close_position_provider()
        self._close_monster_provider()
        self._close_native_process_service()
        native_attachment = None
        try:
            native_attachment = create_native_provider_attachment(window_handler)
            keyboard = HumanKeyboard(window_handler)
            action_executor = ActionExecutor(keyboard)
        except Exception:
            if native_attachment is not None:
                native_attachment.close()
            raise
        position_provider = native_attachment.position_provider
        monster_provider = native_attachment.monster_provider
        self.keyboard = keyboard
        self.action_executor = action_executor
        self.native_provider_attachment = native_attachment
        self.native_process_service = native_attachment.service
        self.position_provider = position_provider
        self.monster_provider = monster_provider
        if monster_provider is not None:
            self._emit(
                "msg_green",
                "Native monster reader attached. Actor slots and local species "
                "positions will be discovered dynamically.",
            )

        if position_provider is not None:
            resolver = position_provider.config.resolver
            if resolver == "module_pointer":
                pointer_storage = position_provider.pointer_storage_address
                detail = (
                    ""
                    if pointer_storage is None
                    else f" Pointer slot: 0x{pointer_storage:X}."
                )
                message = (
                    "Native player-position reader attached with module-pointer "
                    "resolution."
                )
            else:
                addresses = getattr(position_provider, "resolved_addresses", ())
                address_text = ", ".join(f"0x{value:X}" for value in addresses)
                detail = f" Resolved: {address_text}." if address_text else ""
                mode = "consensus" if resolver == "module_offsets" else "direct"
                message = f"Native player-position reader attached in {mode} mode."
            self._emit("msg_green", message + detail)
        # Heading continuity belongs to one capture attachment. Reusing an
        # old-window angle can make the fast jump guard pin Bot Vision after a
        # reattach.
        with self._heading_lock:
            self._heading_preview_detector = None
            self._latest_heading_reading = None
            self._latest_heading_reading_at = 0.0
        self._native_course_tracker.reset()
        with self._kill_counter_lock:
            self.kill_counter_reader.invalidate()
            self._latest_counter_reading = None
            self._latest_counter_reading_at = 0.0
        with self._player_status_lock:
            self.player_status_reader.invalidate()
        self._emit(
            "msg_yellow",
            "Background skill hotkeys are available, but this FlyFF client reads "
            "movement only while focused. The mapper pauses when focus is lost so "
            "unfocused input cannot create false walls.",
        )
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
        self.stop_movement()
        self._emit("msg_yellow", "RL control stopped.")

    def close(self) -> None:
        self.stop()
        if self.capture_service is not None:
            self.capture_service.stop(5.0)
        self.release_input()
        self.runtime_bus = None
        cv.destroyAllWindows()

    def release_input(self) -> None:
        first_error: Exception | None = None
        try:
            self.stop_movement()
        except Exception as error:  # noqa: BLE001 - still close all handles.
            first_error = error
        try:
            self._close_native_attachment()
        except Exception as error:  # noqa: BLE001 - finish input cleanup first.
            if first_error is None:
                first_error = error
        try:
            self._close_position_provider()
        except Exception as error:  # noqa: BLE001 - finish input cleanup first.
            if first_error is None:
                first_error = error
        try:
            self._close_monster_provider()
        except Exception as error:  # noqa: BLE001 - finish input cleanup first.
            if first_error is None:
                first_error = error
        try:
            self._close_native_process_service()
        except Exception as error:  # noqa: BLE001 - finish input cleanup first.
            if first_error is None:
                first_error = error
        finally:
            keyboard = self.keyboard
            self.keyboard = None
            self.action_executor = None
            if keyboard is not None:
                try:
                    keyboard.close()
                except Exception as error:  # noqa: BLE001 - report after cleanup.
                    if first_error is None:
                        first_error = error

        if first_error is not None:
            raise first_error

    def _close_position_provider(self) -> None:
        provider = getattr(self, "position_provider", None)
        self.position_provider = None
        if provider is not None:
            provider.close()

    def _close_monster_provider(self) -> None:
        provider = getattr(self, "monster_provider", None)
        self.monster_provider = None
        if provider is not None:
            provider.close()

    def _close_native_attachment(self) -> None:
        attachment = getattr(self, "native_provider_attachment", None)
        self.native_provider_attachment = None
        if attachment is None:
            return
        self.position_provider = None
        self.monster_provider = None
        self.native_process_service = None
        attachment.close()

    def _close_native_process_service(self) -> None:
        service = getattr(self, "native_process_service", None)
        self.native_process_service = None
        if service is not None:
            service.close()

    def set_config(self, **options) -> None:
        unknown = set(options) - set(self.config)
        if unknown:
            raise ValueError(
                "Unknown bot config key(s): " + ", ".join(sorted(unknown))
            )

        reload_templates = False
        reset_native_map = False

        for key, value in options.items():
            self.config[key] = value

            if key == "selected_mobs":
                reload_templates = True
            if key in {
                "selected_map_name",
                "native_monster_local_radius_cells",
            }:
                reset_native_map = True

        if reload_templates:
            self._reload_mob_templates()
        if reset_native_map:
            self._native_map_overlay = None
            self._native_map_overlay_name = None

    def get_all_mobs(self):
        return MobInfo.get_all_mobs()

    def capture_selected_monster(self) -> NativeActor:
        provider = self.monster_provider
        if provider is None:
            raise RuntimeError(
                "Native monster reader is disabled or the game window is not attached"
            )
        return provider.capture_selected_actor()

    def get_native_monsters(
        self,
        *,
        vision_radius_native: float | None = None,
        force_rediscovery: bool = False,
    ) -> list[NativeActor]:
        """Return selected registered species inside the local native radius."""
        provider = self.monster_provider
        if provider is None:
            return []

        species_ids: set[int] = set()
        for mob in self.config.get("selected_mobs") or []:
            if not isinstance(mob, dict):
                continue
            value = mob.get("species_id")
            if isinstance(value, bool):
                continue
            try:
                species_id = int(value)
            except (TypeError, ValueError):
                continue
            if species_id >= 0:
                species_ids.add(species_id)

        # An empty memory-species selection is deliberately not interpreted as
        # "all actors" because players, pets and helper entities share this
        # layout. Register/capture at least one monster species first.
        if not species_ids:
            return []

        return provider.read_active_actors(
            allowed_species_ids=species_ids,
            vision_radius_native=vision_radius_native,
            force_rediscovery=force_rediscovery,
        )

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

    def _cv_monster_detection_enabled(self) -> bool:
        """Return whether name-template mob CV is useful for this map.

        AoE farming consumes the dynamically recovered native actor set. The
        old name-template detector remains available for non-AoE maps and
        future single-target modes, but it is deliberately skipped for map
        profiles whose name identifies them as AoE.
        """

        map_name = str(self.config.get("selected_map_name") or "").strip()
        return "aoe" not in map_name.casefold()

    def get_visible_mobs(self) -> list[Point]:
        """
        Detect all registered mob-name templates and return anonymous positions.

        Species information is deliberately discarded. Nearby duplicate
        detections are merged because different templates or overlapping
        matches can report the same on-screen mob more than once.
        """
        if not self._cv_monster_detection_enabled():
            self._cache_mob_points([])
            return []

        frame, _debug_frame = self._frame_snapshot()

        if frame is None:
            return []

        points = self._detect_visible_mobs(frame)
        self._cache_mob_points(points)
        return points

    def _detect_visible_mobs(
        self,
        frame: np.ndarray,
        cancellation=None,
    ) -> list[Point]:
        """Detect and deduplicate mob centers without drawing annotations."""
        raw_points: list[Point] = []

        for template in self._mob_templates:
            if cancellation is not None and cancellation.cancelled:
                break
            try:
                matches, _drawn_frame = CV.match_template_multi(
                    frame=frame,
                    crop_area=(50, -50, 50, -50),
                    template=template.image,
                    threshold=float(self.config["mob_pos_match_threshold"]),
                    box_offset=(0, template.height_offset),
                )
            except (cv.error, ValueError, TypeError) as error:
                print(f"Mob template match failed: {error}")
                continue

            raw_points.extend((int(point[0]), int(point[1])) for point in matches)

        return self._deduplicate_points(
            raw_points,
            max_distance=float(self.config["mob_dedup_distance_px"]),
        )

    def _cache_mob_points(self, points: list[Point]) -> None:
        with self._frame_lock:
            self._latest_mob_points = list(points)
            self._latest_mob_points_at = time()

    def read_kill_count(self) -> int | None:
        """
        Read the current total from the in-game kill counter.

        Session delta calculation belongs to the canonical farming environment,
        so this capture helper remains stateless.
        """
        frame, _ = self._frame_snapshot()

        if frame is None:
            return None

        if bool(self.config.get("dynamic_kill_counter", True)):
            reader = getattr(self, "kill_counter_reader", None)
            if reader is None:
                reader = DynamicKillCounterReader(digit_reader=self.digit_reader)
                self.kill_counter_reader = reader
            with self._kill_counter_lock:
                reading = reader.read(frame)
                self._latest_counter_reading = reading
                self._latest_counter_reading_at = time()
            if reading is not None and reading.kills is not None:
                if self.config["show_kill_counter_crop"]:
                    anchor = reading.anchor
                    left, top, right, bottom = reader._scaled_box(
                        anchor,
                        reader._KILLS_BOX,
                    )
                    preview = frame[
                        max(0, top) : min(frame.shape[0], bottom),
                        max(0, left) : min(frame.shape[1], right),
                    ]
                    if preview.size:
                        cv.imshow("Kill Counter Crop", preview)
                        cv.waitKey(1)
                return max(0, int(reading.kills))

        # The panel anchor may be valid while one transparent/animated frame
        # makes digit OCR fail. Do not turn that transient miss into a permanent
        # None result; try the existing configured crop before giving up.
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

    def read_penya_count(self) -> int | None:
        """Read the Penya total from the same dynamically located tracker.

        The farming environment already samples the kill counter every step.
        Reuse that paired kill/Penya OCR result briefly so session statistics do
        not run the same digit crops twice for one captured frame.
        """

        now = time()
        with self._kill_counter_lock:
            cached = self._latest_counter_reading
            if cached is not None and now - self._latest_counter_reading_at <= 0.35:
                return cached.penya

        frame, _ = self._frame_snapshot()
        if frame is None:
            return None
        reader = getattr(self, "kill_counter_reader", None)
        if reader is None:
            reader = DynamicKillCounterReader(digit_reader=self.digit_reader)
            self.kill_counter_reader = reader
        with self._kill_counter_lock:
            reading = reader.read(frame)
            self._latest_counter_reading = reading
            self._latest_counter_reading_at = now
        return None if reading is None else reading.penya

    def read_player_health(self) -> tuple[int, int] | None:
        """OCR current/maximum HP from the dynamically located status panel."""

        gray_frame, color_frame = self._frame_snapshot()
        frame = color_frame if color_frame is not None else gray_frame
        if frame is None:
            return None
        reader = getattr(self, "player_status_reader", None)
        if reader is None:
            reader = DynamicPlayerStatusReader(digit_reader=self.digit_reader)
            self.player_status_reader = reader
        lock = getattr(self, "_player_status_lock", None)
        if lock is None:
            lock = RLock()
            self._player_status_lock = lock
        with lock:
            reading = reader.read(frame)
        if reading is None:
            return None
        return int(reading.current_hp), int(reading.maximum_hp)

    def redetect_ui_elements(self) -> dict[str, object]:
        """Forget and immediately reacquire the OCR panel anchors.

        The operation is deliberately limited to the kill/Penya tracker and
        player-status panel. It does not alter native pointers, actor discovery,
        or monster-template CV. Reader locks make the button safe while the
        farming worker is sampling the same panels.
        """

        gray_frame, color_frame = self._frame_snapshot()
        frame = color_frame if color_frame is not None else gray_frame
        if frame is None:
            return {
                "kill_counter_found": False,
                "player_status_found": False,
                "reason": "no captured frame is available",
            }

        kill_reader = getattr(self, "kill_counter_reader", None)
        if kill_reader is None:
            kill_reader = DynamicKillCounterReader(digit_reader=self.digit_reader)
            self.kill_counter_reader = kill_reader
        player_reader = getattr(self, "player_status_reader", None)
        if player_reader is None:
            player_reader = DynamicPlayerStatusReader(digit_reader=self.digit_reader)
            self.player_status_reader = player_reader

        with self._kill_counter_lock:
            kill_reader.invalidate()
            kill_reading = kill_reader.read(frame)
            self._latest_counter_reading = kill_reading
            self._latest_counter_reading_at = time()

        lock = getattr(self, "_player_status_lock", None)
        if lock is None:
            lock = RLock()
            self._player_status_lock = lock
        with lock:
            player_reader.invalidate()
            player_reading = player_reader.read(frame)

        return {
            "kill_counter_found": kill_reading is not None,
            "kill_count": (
                None if kill_reading is None else kill_reading.kills
            ),
            "penya": None if kill_reading is None else kill_reading.penya,
            "player_status_found": player_reading is not None,
            "current_hp": (
                None if player_reading is None else player_reading.current_hp
            ),
            "maximum_hp": (
                None if player_reading is None else player_reading.maximum_hp
            ),
            "reason": None,
        }

    def execute_action(
        self,
        action: BotAction | int,
    ) -> None:
        self._require_setup()

        if not self._rl_enabled:
            raise RuntimeError("RL control is disabled. Call bot.start() first.")

        assert self.action_executor is not None
        self.action_executor.execute(action)

    def stop_movement(self) -> None:
        first_error: Exception | None = None
        if self.action_executor is not None:
            try:
                self.action_executor.stop_movement()
            except Exception as error:  # noqa: BLE001 - still release raw keys.
                first_error = error

        # Mapper/calibration pulses use the shared HumanKeyboard directly, so
        # ActionExecutor's local held-key set cannot see them. Always emit
        # KEYUP for the complete mapper movement keymap on an external Stop.
        if self.keyboard is not None:
            try:
                self.keyboard.release_keys(
                    (
                        VKEY["z"],
                        VKEY["q"],
                        VKEY["d"],
                    )
                )
            except Exception as error:  # noqa: BLE001 - report after all attempts.
                if first_error is None:
                    first_error = error

        if first_error is not None:
            raise first_error

    @property
    def native_position_available(self) -> bool:
        return self.position_provider is not None

    def get_player_pose(self) -> PlayerPose | None:
        """Return a native world-pose sample when the provider is enabled."""
        provider = self.position_provider
        if provider is None:
            return None
        return provider.read_pose()

    def get_navigation_pose(
        self,
        *,
        max_heading_age_seconds: float = 1.0,
    ) -> PlayerPose | None:
        """Combine native X/Y/Z with the latest validated minimap heading."""
        pose = self.get_player_pose()
        if pose is None:
            return None
        if pose.heading_degrees is not None:
            return pose

        now = time()
        reading = None
        with self._heading_lock:
            if (
                self._latest_heading_reading is not None
                and now - self._latest_heading_reading_at
                <= max(0.05, float(max_heading_age_seconds))
            ):
                reading = self._latest_heading_reading
            else:
                _gray, frame = self._frame_snapshot()
                if frame is not None:
                    if self._heading_preview_detector is None:
                        from mapper import MinimapHeadingDetector

                        self._heading_preview_detector = MinimapHeadingDetector()
                    reading = self._heading_preview_detector.read_fast(frame)
                    if reading is not None:
                        self._latest_heading_reading = reading
                        self._latest_heading_reading_at = now
        if reading is None:
            return pose
        return PlayerPose(
            x=pose.x,
            y=pose.y,
            z=pose.z,
            heading_degrees=float(reading.angle_deg),
            timestamp=pose.timestamp,
        )

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

    def get_frame_sample(self) -> FrameSample | None:
        """
        Return the latest grayscale frame with capture freshness metadata.

        Calibration and mapping can use the generation/sequence identity to
        ensure that multi-frame vision reads do not count one capture more than
        once. The existing ``get_frame()`` API remains unchanged for low-cost
        consumers that do not need freshness guarantees.
        """
        if self.capture_service is None:
            return None
        return self.capture_service.sample(grayscale=True)

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

    def build_preview(self, frame: np.ndarray, cancellation=None) -> np.ndarray:
        now = time()
        show_mobs = bool(self.config.get("show_mobs_pos_markers", True))
        cv_mobs_enabled = self._cv_monster_detection_enabled()
        if show_mobs and cv_mobs_enabled:
            with self._frame_lock:
                detected_at = self._latest_mob_points_at
            if now - detected_at >= self._preview_detection_interval:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                self._cache_mob_points(
                    self._detect_visible_mobs(gray, cancellation=cancellation)
                )
        elif not cv_mobs_enabled:
            # Do not leave stale name-template markers visible after switching
            # to an AoE map. Native actor markers continue on the map overlay.
            self._cache_mob_points([])

        if cancellation is not None and cancellation.cancelled:
            return frame

        if show_mobs and cv_mobs_enabled:
            self._draw_cached_mob_overlay(frame, now)
        if bool(self.config.get("show_detected_ui_elements", True)):
            self._draw_heading_overlay(frame, now)
            self._draw_kill_counter_overlay(frame, now)
            self._draw_player_status_overlay(frame)
        self._publish_native_monster_map(now)
        return frame

    def build_preview_cancellable(self, frame: np.ndarray, cancellation) -> np.ndarray:
        """Build a preview that can stop between expensive overlay stages."""

        return self.build_preview(frame, cancellation=cancellation)

    def _publish_native_monster_map(self, now: float) -> None:
        """Publish the persistent map with native monster markers.

        This runs on the droppable preview worker, not Tk and not the RL
        control loop. The expensive global slot discovery is cached by the
        provider; ordinary refreshes only poll known actor addresses.
        """
        if not self.config.get("show_native_monsters_on_map", True):
            return
        map_name = self.config.get("selected_map_name")
        if not map_name or self.runtime_bus is None:
            return
        if self.monster_provider is None or self.position_provider is None:
            return
        service = getattr(self, "native_process_service", None)
        if service is not None and bool(
            getattr(service, "recovery_active", False)
        ):
            # Pointer/profile recovery deliberately invalidates ordinary player
            # reads while it rebuilds the independent reader. Suppress the
            # expected overlay error instead of flooding the console.
            return

        refresh = max(
            0.1,
            float(self.config.get("native_monster_map_refresh_seconds", 0.5)),
        )
        if now - self._last_native_map_publish_at < refresh:
            return
        self._last_native_map_publish_at = now

        try:
            if (
                self._native_map_overlay is None
                or self._native_map_overlay_name != str(map_name)
            ):
                # Import the class from its defining module. Importing it from
                # the package namespace is unsafe once Python has loaded the
                # identically named submodule: ``from mapper import
                # NativeMonsterMapOverlay`` can then resolve to the module
                # object instead of the class.
                from mapper.NativeMonsterMapOverlay import (
                    NativeMonsterMapOverlay,
                )

                self._native_map_overlay = NativeMonsterMapOverlay.load(
                    str(map_name),
                    local_radius_cells=int(
                        self.config.get("native_monster_local_radius_cells", 50)
                    ),
                )
                self._native_map_overlay_name = str(map_name)

            player_pose = self.get_player_pose()
            if player_pose is None:
                return
            actors = self.get_native_monsters()
            dashboard = self._native_map_overlay.render(player_pose, actors)
            self.runtime_bus.publish_latest("map_frame", dashboard)
        except Exception as error:  # noqa: BLE001 - optional preview boundary.
            if now - self._last_native_map_error_at >= 15.0:
                self._emit("msg_red", f"Native monster map overlay failed: {error}")
                self._last_native_map_error_at = now

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

            image_path = Path(__file__).resolve().parents[1] / "assets" / "names" / f"{name}.png"
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
        """Draw only EVA-relevant green boxes around cached mob positions."""
        height, width = frame.shape[:2]

        with self._frame_lock:
            points = list(self._latest_mob_points)
            detected_at = self._latest_mob_points_at

        age = now - detected_at if detected_at else float("inf")
        if age > 1.0:
            return

        for point in points:
            try:
                x, y = (int(point[0]), int(point[1]))
            except (TypeError, ValueError):
                continue

            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))

            cv.rectangle(
                frame,
                (max(0, x - 24), max(0, y - 24)),
                (min(width - 1, x + 24), min(height - 1, y + 24)),
                (0, 255, 0),
                2,
                cv.LINE_4,
            )

    def _draw_heading_overlay(self, frame: np.ndarray, now: float) -> None:
        """Draw and cache the fast minimap heading from the preview worker."""
        try:
            heading_lock = getattr(self, "_heading_lock", None)
            if heading_lock is None:
                heading_lock = RLock()
                self._heading_lock = heading_lock
            with heading_lock:
                if self._heading_preview_detector is None:
                    from mapper import MinimapHeadingDetector

                    self._heading_preview_detector = MinimapHeadingDetector()

                reading = self._heading_preview_detector.read_fast(frame)
                reference = self._native_course_heading_reference()
                observe_reference = getattr(
                    self._heading_preview_detector,
                    "observe_reference_heading",
                    None,
                )
                if (
                    reference is not None
                    and callable(observe_reference)
                    and observe_reference(reference, allow_opposite=True)
                ):
                    reading = self._heading_preview_detector.read_fast(frame)
                if reading is None:
                    return
                self._latest_heading_reading = reading
                self._latest_heading_reading_at = float(now)

            center_x, center_y = reading.center
            angle = math.radians(reading.angle_deg)
            length = 44
            endpoint = (
                round(center_x + math.sin(angle) * length),
                round(center_y - math.cos(angle) * length),
            )
            color = (0, 140, 255) if reading.is_stale else (0, 215, 255)
            cv.circle(frame, (center_x, center_y), 6, color, 2)
            cv.arrowedLine(
                frame,
                (center_x, center_y),
                endpoint,
                color,
                3,
                tipLength=0.28,
            )
            cv.putText(
                frame,
                f"{reading.angle_deg:.0f} deg {reading.confidence:.2f}",
                (max(4, center_x - 78), max(18, center_y - 18)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv.LINE_AA,
            )
        except Exception as error:  # noqa: BLE001 - optional preview boundary.
            if now - self._last_heading_error_at >= 15.0:
                self._emit("msg_red", f"Heading preview failed: {error}")
                self._last_heading_error_at = now

    def _draw_kill_counter_overlay(self, frame: np.ndarray, now: float) -> None:
        """Mark the dynamically tracked kill/Penya panel in Bot Vision."""
        del now
        if not bool(self.config.get("dynamic_kill_counter", True)):
            return
        try:
            counter_lock = getattr(self, "_kill_counter_lock", None)
            if counter_lock is None:
                counter_lock = RLock()
                self._kill_counter_lock = counter_lock
            with counter_lock:
                reader = self.kill_counter_reader
                anchor = reader.locate(frame)
                if anchor is None:
                    return
                left, top, right, bottom = reader.tracking_bounds(
                    anchor,
                    frame_shape=frame.shape[:2],
                )
            cv.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                frame,
                "Kill/Penya tracked",
                (left, max(16, top - 6)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )
        except Exception:
            # This diagnostic overlay must never interrupt capture or control.
            return

    def _draw_player_status_overlay(self, frame: np.ndarray) -> None:
        """Mark the OCR-validated player-status panel in Bot Vision."""

        try:
            status_lock = getattr(self, "_player_status_lock", None)
            if status_lock is None:
                status_lock = RLock()
                self._player_status_lock = status_lock
            with status_lock:
                reader = self.player_status_reader
                reading = reader.read(frame)
                if reading is None:
                    return
                left, top, right, bottom = reader.tracking_bounds(
                    reading.anchor,
                    frame_shape=frame.shape[:2],
                )
            color = (0, 255, 0)
            cv.rectangle(
                frame,
                (left, top),
                (right, bottom),
                color,
                2,
                cv.LINE_AA,
            )
            cv.putText(
                frame,
                f"Player HP {reading.current_hp}/{reading.maximum_hp}",
                (left, min(frame.shape[0] - 4, max(16, bottom + 18))),
                cv.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv.LINE_AA,
            )
        except Exception:
            # This diagnostic overlay must never interrupt capture or control.
            return

    def _native_course_heading_reference(self) -> float | None:
        provider = getattr(self, "position_provider", None)
        if provider is None:
            return None
        tracker = getattr(self, "_native_course_tracker", None)
        if tracker is None:
            tracker = NativeCourseHeadingTracker()
            self._native_course_tracker = tracker
        try:
            pose = provider.read_pose()
        except Exception:  # noqa: BLE001 - optional validation signal.
            return None
        course = tracker.update(pose)
        return None if course is None else float(course.angle_deg)

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
