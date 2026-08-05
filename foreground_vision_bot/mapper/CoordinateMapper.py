from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from capture_service import FrameSample
from libs.HumanKeyboard import VKEY, HumanKeyboard, KeyPressTiming
from position import PlayerPose
from worker_manager import CancellationToken

from .AdaptiveMappingController import AdaptiveMappingController
from .CompletionGuard import CompletionGuard, CompletionReport
from .CoordinateFrame import CoordinateFrame
from .CoordinateMapLogger import CoordinateMapLogger
from .Explorer import Explorer, ExplorerDecision
from .FreeSpaceInference import FreeSpaceInference
from .MapCatalog import MapCatalog, MapProfile
from .MinimapHeading import HeadingReading, MinimapHeadingDetector, signed_angle_delta
from .OccupancyGrid import OccupancyGrid
from .WallInference import WallInference

StatusCallback = Callable[[str], None]
FrameCallback = Callable[[object], None]
RecoveryCallback = Callable[[str, str, bool, bool], str | None]
DEFAULT_COORDINATE_MAPPER_CONFIG_PATH = Path(__file__).with_name("coordinate_mapper.json")


class CoordinateMapperBot(Protocol):
    keyboard: HumanKeyboard | None

    def get_player_pose(self) -> PlayerPose | None: ...

    def get_frame_sample(self) -> FrameSample | None: ...


@dataclass(frozen=True)
class MapperConfig:
    """Runtime settings for the native-coordinate mapper."""

    forward_seconds: float = 0.12
    backward_seconds: float = 0.18
    backward_retry_seconds: float = 0.28
    turn_left_90_seconds: float = 0.38
    turn_right_90_seconds: float = 0.335
    settle_seconds: float = 0.12
    startup_countdown_seconds: int = 3
    native_units_per_cell: float = 1.6
    grid_size: int = 1001
    blocked_distance_units: float = 0.10
    partial_distance_units: float = 0.75
    backward_blocked_distance_units: float = 0.05
    backward_partial_distance_units: float = 0.35
    backward_blocked_confirmations: int = 2
    minimap_heading_enabled: bool = True
    minimap_heading_samples: int = 9
    minimap_heading_min_confidence: float = 0.52
    minimap_heading_max_uncertainty_deg: float = 4.0
    minimap_heading_max_ambiguity: float = 0.70
    heading_mismatch_recheck_degrees: float = 30.0
    heading_collision_guard_degrees: float = 18.0
    heading_motion_min_distance_units: float = 0.30
    teleport_distance_units: float = 50.0
    teleport_vertical_distance_units: float = 25.0
    blocked_confirmations: int = 2
    heading_acquisition_attempts: int = 4
    eva_interval_steps: int = 20
    eva_settle_seconds: float = 0.40
    eva_retry_settle_seconds: float = 0.25
    map_publish_interval_seconds: float = 0.20
    slide_min_lateral_units: float = 0.30
    slide_min_forward_alignment: float = 0.70
    pause_when_unfocused: bool = True
    unfocused_poll_seconds: float = 0.25
    local_map_radius_cells: int = 50
    trap_score_threshold: int = 6
    partial_trap_score: int = 2
    blocked_trap_score: int = 3
    escape_backward_steps: int = 8
    escape_target_distance_units: float = 3.2
    escape_wiggle_degrees: float = 35.0
    escape_wiggle_after_blocked_attempts: int = 2
    completion_margin_cells: int = 4
    completion_min_free_cells: int = 100
    completion_stable_checks: int = 3
    wall_autofill_enabled: bool = True
    wall_autofill_max_gap_cells: int = 2
    wall_autofill_min_support_cells: int = 4
    wall_autofill_support_radius_cells: int = 6
    wall_autofill_max_line_error_cells: float = 0.75
    free_space_autofill_enabled: bool = True
    free_space_autofill_max_enclosed_area_cells: int = 12
    free_space_autofill_max_enclosed_span_cells: int = 4
    free_space_autofill_max_line_length_cells: int = 40
    free_space_autofill_interval_steps: int = 5

    def __post_init__(self) -> None:
        numeric_positive = {
            "forward_seconds": self.forward_seconds,
            "backward_seconds": self.backward_seconds,
            "backward_retry_seconds": self.backward_retry_seconds,
            "turn_left_90_seconds": self.turn_left_90_seconds,
            "turn_right_90_seconds": self.turn_right_90_seconds,
            "native_units_per_cell": self.native_units_per_cell,
            "teleport_distance_units": self.teleport_distance_units,
        }
        for name, value in numeric_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.settle_seconds) or self.settle_seconds < 0.0:
            raise ValueError("settle_seconds must be finite and non-negative")
        if not math.isfinite(self.blocked_distance_units) or self.blocked_distance_units < 0.0:
            raise ValueError("blocked_distance_units must be finite and non-negative")
        if self.partial_distance_units <= self.blocked_distance_units:
            raise ValueError("partial_distance_units must exceed blocked_distance_units")
        if (
            not math.isfinite(self.backward_blocked_distance_units)
            or self.backward_blocked_distance_units < 0.0
        ):
            raise ValueError(
                "backward_blocked_distance_units must be finite and non-negative"
            )
        if self.backward_partial_distance_units <= self.backward_blocked_distance_units:
            raise ValueError(
                "backward_partial_distance_units must exceed "
                "backward_blocked_distance_units"
            )
        if self.backward_retry_seconds < self.backward_seconds:
            raise ValueError(
                "backward_retry_seconds must be at least backward_seconds"
            )
        if self.backward_blocked_confirmations < 1:
            raise ValueError("backward_blocked_confirmations must be at least one")
        if self.minimap_heading_samples < 5:
            raise ValueError("minimap_heading_samples must be at least five")
        if not 0.0 < self.minimap_heading_min_confidence <= 1.0:
            raise ValueError("minimap_heading_min_confidence must be in (0, 1]")
        if (
            not math.isfinite(self.minimap_heading_max_uncertainty_deg)
            or self.minimap_heading_max_uncertainty_deg <= 0.0
        ):
            raise ValueError(
                "minimap_heading_max_uncertainty_deg must be finite and positive"
            )
        if not 0.0 <= self.minimap_heading_max_ambiguity <= 1.0:
            raise ValueError("minimap_heading_max_ambiguity must be in [0, 1]")
        if (
            not math.isfinite(self.heading_mismatch_recheck_degrees)
            or not 5.0 <= self.heading_mismatch_recheck_degrees <= 90.0
        ):
            raise ValueError(
                "heading_mismatch_recheck_degrees must be between 5 and 90"
            )
        if (
            not math.isfinite(self.heading_collision_guard_degrees)
            or not 1.0 <= self.heading_collision_guard_degrees <= 45.0
        ):
            raise ValueError(
                "heading_collision_guard_degrees must be between 1 and 45"
            )
        if (
            not math.isfinite(self.heading_motion_min_distance_units)
            or self.heading_motion_min_distance_units <= 0.0
        ):
            raise ValueError(
                "heading_motion_min_distance_units must be finite and positive"
            )
        if self.teleport_distance_units <= self.partial_distance_units:
            raise ValueError("teleport_distance_units must exceed partial_distance_units")
        if (
            not math.isfinite(self.teleport_vertical_distance_units)
            or self.teleport_vertical_distance_units <= 0.0
        ):
            raise ValueError(
                "teleport_vertical_distance_units must be finite and positive"
            )
        if self.grid_size < 101 or self.grid_size % 2 == 0:
            raise ValueError("grid_size must be an odd integer of at least 101")
        if self.blocked_confirmations < 1:
            raise ValueError("blocked_confirmations must be at least one")
        if self.heading_acquisition_attempts < 1:
            raise ValueError("heading_acquisition_attempts must be at least one")
        if self.startup_countdown_seconds < 0:
            raise ValueError("startup_countdown_seconds cannot be negative")
        if self.local_map_radius_cells < 3 or self.local_map_radius_cells > 250:
            raise ValueError("local_map_radius_cells must be between 3 and 250")
        if self.eva_interval_steps < 0:
            raise ValueError("eva_interval_steps cannot be negative")
        if self.eva_settle_seconds < 0.0:
            raise ValueError("eva_settle_seconds cannot be negative")
        if self.eva_retry_settle_seconds < 0.0:
            raise ValueError("eva_retry_settle_seconds cannot be negative")
        if (
            not math.isfinite(self.slide_min_lateral_units)
            or self.slide_min_lateral_units <= 0.0
        ):
            raise ValueError("slide_min_lateral_units must be finite and positive")
        if (
            not math.isfinite(self.slide_min_forward_alignment)
            or not 0.0 < self.slide_min_forward_alignment < 1.0
        ):
            raise ValueError("slide_min_forward_alignment must be in (0, 1)")
        if (
            not math.isfinite(self.unfocused_poll_seconds)
            or self.unfocused_poll_seconds <= 0.0
        ):
            raise ValueError("unfocused_poll_seconds must be finite and positive")
        if self.trap_score_threshold < 1:
            raise ValueError("trap_score_threshold must be at least one")
        if self.partial_trap_score < 1 or self.blocked_trap_score < 1:
            raise ValueError("trap score increments must be at least one")
        if self.escape_backward_steps < 1:
            raise ValueError("escape_backward_steps must be at least one")
        if not math.isfinite(self.escape_target_distance_units) or self.escape_target_distance_units <= 0.0:
            raise ValueError("escape_target_distance_units must be finite and positive")
        if not math.isfinite(self.escape_wiggle_degrees) or not 0.0 < self.escape_wiggle_degrees <= 90.0:
            raise ValueError("escape_wiggle_degrees must be in (0, 90]")
        if self.escape_wiggle_after_blocked_attempts < 1:
            raise ValueError("escape_wiggle_after_blocked_attempts must be at least one")
        if self.completion_margin_cells < 1:
            raise ValueError("completion_margin_cells must be at least one")
        if self.completion_min_free_cells < 1:
            raise ValueError("completion_min_free_cells must be at least one")
        if self.completion_stable_checks < 1:
            raise ValueError("completion_stable_checks must be at least one")
        if self.wall_autofill_max_gap_cells < 1 or self.wall_autofill_max_gap_cells > 6:
            raise ValueError("wall_autofill_max_gap_cells must be between 1 and 6")
        if self.wall_autofill_min_support_cells < 4:
            raise ValueError("wall_autofill_min_support_cells must be at least four")
        if (
            self.wall_autofill_support_radius_cells
            < self.wall_autofill_max_gap_cells + 2
        ):
            raise ValueError(
                "wall_autofill_support_radius_cells must exceed the gap by at least one"
            )
        if (
            not math.isfinite(self.wall_autofill_max_line_error_cells)
            or not 0.0 < self.wall_autofill_max_line_error_cells <= 1.5
        ):
            raise ValueError(
                "wall_autofill_max_line_error_cells must be in (0, 1.5]"
            )
        if self.free_space_autofill_max_enclosed_area_cells < 1:
            raise ValueError(
                "free_space_autofill_max_enclosed_area_cells must be at least one"
            )
        if self.free_space_autofill_max_enclosed_span_cells < 1:
            raise ValueError(
                "free_space_autofill_max_enclosed_span_cells must be at least one"
            )
        if self.free_space_autofill_max_line_length_cells < 2:
            raise ValueError(
                "free_space_autofill_max_line_length_cells must be at least two"
            )
        if self.free_space_autofill_interval_steps < 1:
            raise ValueError("free_space_autofill_interval_steps must be at least one")


def load_mapper_config(
    path: Path = DEFAULT_COORDINATE_MAPPER_CONFIG_PATH,
) -> MapperConfig:
    if not path.is_file():
        return MapperConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Coordinate mapper config root must be an object")
    # Use dataclasses.fields() rather than the implementation-detail
    # __dataclass_fields__ mapping.  This keeps the JSON schema tied to the
    # actual constructor fields and prevents packaged config/code mismatches
    # from rejecting a valid newly-added setting.
    allowed = {field.name for field in fields(MapperConfig) if field.init}
    unknown = sorted(set(payload) - allowed - {"version"})
    if unknown:
        raise ValueError(
            "Unknown coordinate mapper config fields: " + ", ".join(unknown)
        )
    version = int(payload.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported coordinate mapper config version: {version}")
    values = {key: payload[key] for key in allowed if key in payload}
    return MapperConfig(**values)


@dataclass(frozen=True)
class NativeMotion:
    before: PlayerPose
    after: PlayerPose
    delta_x: float
    delta_y: float
    delta_z: float
    horizontal_distance: float
    vertical_distance: float
    outcome: str
    forward_progress: float = 0.0
    lateral_distance: float = 0.0
    forward_alignment: float = 1.0


class MapTransitionDetected(RuntimeError):
    """Raised when native coordinates jump farther than physical movement allows."""

    def __init__(
        self,
        *,
        before: PlayerPose,
        after: PlayerPose,
        horizontal_distance: float,
        vertical_distance: float,
        context: str,
    ) -> None:
        self.before = before
        self.after = after
        self.horizontal_distance = float(horizontal_distance)
        self.vertical_distance = float(vertical_distance)
        self.context = str(context)
        super().__init__(
            f"native coordinate discontinuity during {self.context}: "
            f"horizontal={self.horizontal_distance:.3f}u, "
            f"vertical={self.vertical_distance:.3f}u"
        )


class MapProfileLocationMismatch(RuntimeError):
    """Raised when the current native coordinates cannot belong to this map grid."""

    def __init__(
        self,
        *,
        pose: PlayerPose,
        local_x: float,
        local_y: float,
        map_name: str,
    ) -> None:
        self.pose = pose
        self.local_x = float(local_x)
        self.local_y = float(local_y)
        self.map_name = str(map_name)
        super().__init__(
            f"current native pose maps to ({self.local_x:.1f}, {self.local_y:.1f}) "
            f"outside the '{self.map_name}' occupancy grid"
        )


class CoordinateMapper:
    """Map FlyFF directly from native X/Z coordinates.

    Position deltas are authoritative for precise travel and heading. The
    minimap arrow provides an absolute heading reference at startup, after
    turns, and whenever coordinate motion disagrees with the expected command.
    Optical flow and floor/not-floor labelling are not used.
    """

    VERSION = "5.0.9-heading-fusion-backpedal-collisions-free-autofill"
    FRAME_FILE = "coordinate_frame.json"

    def __init__(
        self,
        bot: CoordinateMapperBot,
        status_callback: StatusCallback | None = None,
        frame_callback: FrameCallback | None = None,
        config: MapperConfig | None = None,
        cancellation: CancellationToken | None = None,
        map_name: str | None = None,
        recovery_callback: RecoveryCallback | None = None,
        rl_shadow_enabled: bool = False,
        rl_policy_path: Path | None = None,
    ) -> None:
        del rl_policy_path
        if bot.keyboard is None:
            raise RuntimeError("Attach the Flyff window first.")
        if bot.get_player_pose() is None:
            raise RuntimeError(
                "Native player coordinates are unavailable. Check "
                "position/native_position.json and attach Neuz.exe again."
            )

        self.bot = bot
        self.config = config or load_mapper_config()
        self.status_callback = status_callback or print
        self.frame_callback = frame_callback
        self.cancellation = cancellation or CancellationToken()
        self.recovery_callback = recovery_callback
        self.rl_shadow_enabled = bool(rl_shadow_enabled)

        self.map_catalog = MapCatalog()
        self.map_profile: MapProfile = self.map_catalog.get(map_name)
        self.map_dir = self.map_catalog.map_directory(self.map_profile.name)
        self.frame_path = self.map_dir / self.FRAME_FILE

        self.controller = AdaptiveMappingController(bot.keyboard)
        self.heading_detector = MinimapHeadingDetector()
        self.explorer = Explorer()
        self.grid = OccupancyGrid(size=self.config.grid_size)
        self.coordinate_frame: CoordinateFrame | None = None
        self.completion_guard = CompletionGuard(
            margin_cells=self.config.completion_margin_cells,
            minimum_free_cells=self.config.completion_min_free_cells,
        )
        self.wall_inference = WallInference(
            max_gap_cells=self.config.wall_autofill_max_gap_cells,
            minimum_support_cells=self.config.wall_autofill_min_support_cells,
            support_radius_cells=self.config.wall_autofill_support_radius_cells,
            maximum_line_error_cells=self.config.wall_autofill_max_line_error_cells,
        )
        self.free_space_inference = FreeSpaceInference(
            maximum_enclosed_area_cells=(
                self.config.free_space_autofill_max_enclosed_area_cells
            ),
            maximum_enclosed_span_cells=(
                self.config.free_space_autofill_max_enclosed_span_cells
            ),
            maximum_line_length_cells=(
                self.config.free_space_autofill_max_line_length_cells
            ),
        )

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.map_catalog.run_directory(self.map_profile.name) / run_id
        self.logger = CoordinateMapLogger(self.output_dir / "coordinate_steps.csv")

        self._step = 0
        self._last_publish_at = 0.0
        self._heading_source = "unknown"
        self._blocked_counts: dict[tuple[int, int, int, int], int] = {}
        self._last_pose: PlayerPose | None = None
        self._trusted_pose: PlayerPose | None = None
        self._trap_score = 0
        self._trap_entry_edge: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._recent_traversed_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self._completion_streak = 0
        self._stall_probe_turn_next = True
        self._save_failure_streak = 0
        self._post_eva_guard_pending = False
        self._focus_pause_announced = False
        self._heading_warning_announced = False
        self._last_free_autofill_step = -10_000

    def stop(self) -> None:
        self.cancellation.cancel()
        self.controller.stop()

    def run(self) -> Path:
        primary_error: BaseException | None = None
        try:
            return self._run()
        except MapTransitionDetected as transition:
            self.controller.stop()
            self._record_map_transition(transition)
            return self.output_dir
        except MapProfileLocationMismatch as mismatch:
            self.controller.stop()
            self._record_profile_mismatch(mismatch)
            return self.output_dir
        except BaseException as error:
            primary_error = error
            try:
                self.grid.metadata.termination_reason = f"{type(error).__name__}: {error}"
            except Exception as metadata_error:  # noqa: BLE001
                error.add_note(f"Could not record mapper termination reason: {metadata_error}")
            raise
        finally:
            cleanup_error: Exception | None = None
            for label, action in (
                ("release movement keys", self.controller.stop),
                ("save coordinate map", self._save_state),
                ("close coordinate log", self.logger.close),
            ):
                try:
                    action()
                except Exception as error:  # noqa: BLE001
                    if primary_error is not None:
                        primary_error.add_note(f"Could not {label}: {error}")
                    elif cleanup_error is None:
                        cleanup_error = error
            if primary_error is None and cleanup_error is not None:
                raise cleanup_error

    def _run(self) -> Path:
        current = self._read_pose()
        self._load_or_create_coordinate_map(current)
        self._apply_wall_inference(context="loaded map")
        self._apply_free_space_inference(context="loaded map", force=True)
        self._validate_start_location(current)
        self._trusted_pose = current
        self._localize(current, heading_deg=self.grid.continuous_pose.heading_deg)
        self.grid.set_pose_reliability(
            position_known=True,
            heading_known=False,
            note="Native position is known; heading will be learned from movement.",
        )
        self.grid.metadata.map_name = self.map_profile.name
        self.grid.metadata.run_count += 1
        self.grid.metadata.termination_reason = None

        self.status_callback(
            f"Coordinate mapper {self.VERSION} starts in "
            f"'{self.map_profile.name}'. Native X/Z is authoritative for travel; "
            "the minimap arrow supplies absolute heading at startup, after turns, "
            "and during heading-safety recovery. No camera position, optical flow, "
            "floor model, or label popup is used. "
            f"The live map includes a +/-{self.config.local_map_radius_cells}-cell "
            "player-centred panel. "
            + (
                "Conservative short-gap wall autofill is enabled."
                if self.config.wall_autofill_enabled
                else "Wall autofill is disabled."
            )
            + (
                " Tiny enclosed and one-cell-wide unknown gaps are auto-filled "
                "as reversible free space."
                if self.config.free_space_autofill_enabled
                else " Free-space autofill is disabled."
            )
        )
        if self.rl_shadow_enabled:
            self.status_callback(
                "Mapper RL shadow was requested but is intentionally disabled in "
                "coordinate-map mode; deterministic frontier exploration is active."
            )
        self._publish_map(force=True)
        for remaining in range(self.config.startup_countdown_seconds, 0, -1):
            self.status_callback(f"Coordinate mapper starting in {remaining}...")
            if self.cancellation.wait(1.0):
                self.cancellation.raise_if_cancelled()

        self._acquire_heading()
        self._save_state()

        while not self.cancellation.cancelled:
            self.cancellation.raise_if_cancelled()
            self._wait_for_game_focus()
            self._verify_pose_continuity("between mapper actions")
            decision = self.explorer.decide(self.grid)
            if decision.action == "STOP":
                decision = self._handle_no_frontier()
                if decision is None:
                    break

            self._step += 1
            self._maybe_cast_eva()
            self._execute_decision(decision)
            self._publish_map(force=True)
            self._save_state()

        self.status_callback(f"Coordinate mapping run saved to {self.output_dir}.")
        return self.output_dir

    def _load_or_create_coordinate_map(self, current: PlayerPose) -> None:
        if self.frame_path.is_file():
            self.coordinate_frame = CoordinateFrame.load(self.frame_path)
            self.grid, warning = OccupancyGrid.load(self.map_dir)
            if warning is not None:
                self.status_callback(warning)
            self.status_callback(
                "Loaded coordinate map at native origin "
                f"({self.coordinate_frame.origin_native_x:.3f}, "
                f"{self.coordinate_frame.origin_native_z:.3f}); "
                f"{self.grid.known_cell_count()} known cells."
            )
            return

        archived = self._archive_visual_map_if_present()
        self.grid = OccupancyGrid(size=self.config.grid_size)
        self.coordinate_frame = CoordinateFrame(
            origin_native_x=current.x,
            origin_native_z=current.z,
            native_units_per_cell=self.config.native_units_per_cell,
        )
        self.coordinate_frame.save(self.frame_path)
        if archived is not None:
            self.status_callback(
                "Archived the incompatible visual-odometry map to "
                f"{archived.name} and started a clean native-coordinate map."
            )
        else:
            self.status_callback(
                "Created a new native-coordinate map with the current position as origin."
            )

    def _archive_visual_map_if_present(self) -> Path | None:
        legacy_names = ("map.json", "occupancy.npy", "visits.npy", "map_preview.png")
        existing = [self.map_dir / name for name in legacy_names if (self.map_dir / name).exists()]
        if not existing:
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive = self.map_dir / f"legacy_visual_map_{stamp}"
        archive.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.move(str(path), str(archive / path.name))
        return archive

    def _read_pose(self) -> PlayerPose:
        pose = self.bot.get_player_pose()
        if pose is None:
            raise RuntimeError("Native player position became unavailable")
        self._last_pose = pose
        return pose

    def _verify_pose_continuity(self, context: str) -> PlayerPose:
        """Read a pose and compare it with the last trusted native position.

        The old transition check only compared the two samples surrounding a
        forward keypress. A trigger can teleport the player after a keypress,
        during settling, or while turning. In that case the next action starts
        at the destination and its own before/after delta looks perfectly
        normal. This cross-action anchor catches that discontinuity.
        """
        pose = self._read_pose()
        reference = self._trusted_pose
        if reference is None:
            self._trusted_pose = pose
            return pose

        horizontal = math.hypot(pose.x - reference.x, pose.z - reference.z)
        vertical = abs(pose.y - reference.y)
        if self._is_transition_distance(horizontal, vertical):
            raise MapTransitionDetected(
                before=reference,
                after=pose,
                horizontal_distance=horizontal,
                vertical_distance=vertical,
                context=context,
            )

        self._trusted_pose = pose
        return pose

    def _is_transition_distance(self, horizontal: float, vertical: float) -> bool:
        return (
            horizontal >= self.config.teleport_distance_units
            or vertical >= self.config.teleport_vertical_distance_units
        )

    def _record_map_transition(self, transition: MapTransitionDetected) -> None:
        """Persist a transition marker without drawing the destination into the map."""
        frame = self._require_frame()
        source_local = frame.to_local_cells(
            transition.before.x,
            transition.before.z,
        )
        source_cell = self.grid.world_point_to_grid_cell(*source_local)
        direction = self.grid.DIRECTIONS[self.grid.pose.heading_index]
        attempted_cell = (
            source_cell[0] + direction[0],
            source_cell[1] + direction[1],
        )
        reason = (
            f"{transition.context}; impossible native jump "
            f"horizontal={transition.horizontal_distance:.3f}u, "
            f"vertical={transition.vertical_distance:.3f}u; destination native=("
            f"{transition.after.x:.3f}, {transition.after.y:.3f}, "
            f"{transition.after.z:.3f})"
        )
        self.grid.add_suspected_transition(
            from_x=source_cell[0],
            from_y=source_cell[1],
            attempted_x=attempted_cell[0],
            attempted_y=attempted_cell[1],
            heading_deg=self.grid.continuous_pose.heading_deg,
            reason=reason,
        )
        message = (
            "Teleport/map transition detected from impossible native-coordinate "
            f"movement ({transition.horizontal_distance:.1f} horizontal units, "
            f"{transition.vertical_distance:.1f} vertical units) during "
            f"{transition.context}. Movement keys were released and this map was "
            "saved without integrating the destination coordinates."
        )
        self.grid.metadata.termination_reason = message
        self.grid.set_pose_reliability(
            position_known=False,
            heading_known=False,
            note=(
                "Map transition detected; the last trusted map pose was preserved "
                "and destination coordinates were not integrated."
            ),
        )
        timing = KeyPressTiming(
            requested_seconds=0.0,
            clamped_seconds=0.0,
            held_seconds=0.0,
            elapsed_seconds=0.0,
        )
        motion = NativeMotion(
            before=transition.before,
            after=transition.after,
            delta_x=transition.after.x - transition.before.x,
            delta_y=transition.after.y - transition.before.y,
            delta_z=transition.after.z - transition.before.z,
            horizontal_distance=transition.horizontal_distance,
            vertical_distance=transition.vertical_distance,
            outcome="teleport",
        )
        self._log_motion(
            motion,
            action="MAP_TRANSITION",
            reason=transition.context,
            attempt=1,
            timing=timing,
            note="transition detected across trusted-pose boundary; destination not integrated",
        )
        self.status_callback(message)

    def _record_profile_mismatch(self, mismatch: MapProfileLocationMismatch) -> None:
        message = (
            f"Current native coordinates belong outside the '{mismatch.map_name}' "
            f"map frame (local=({mismatch.local_x:.1f}, {mismatch.local_y:.1f})). "
            "The mapper stopped safely before changing the saved occupancy map. "
            "Return to this map or select a separate map profile for the destination."
        )
        self.grid.metadata.termination_reason = message
        self.grid.set_pose_reliability(
            position_known=False,
            heading_known=False,
            note="Current client coordinates do not belong to this map profile.",
        )
        self.status_callback(message)

    def _localize(self, pose: PlayerPose, *, heading_deg: float) -> tuple[float, float]:
        frame = self._require_frame()
        local_x, local_y = frame.to_local_cells(pose.x, pose.z)
        self.grid.set_continuous_pose(local_x, local_y, heading_deg)
        return local_x, local_y

    def _validate_start_location(self, pose: PlayerPose) -> None:
        frame = self._require_frame()
        local_x, local_y = frame.to_local_cells(pose.x, pose.z)
        cell_x, cell_y = self.grid.world_point_to_grid_cell(local_x, local_y)
        if not self.grid.in_bounds(cell_x, cell_y):
            raise MapProfileLocationMismatch(
                pose=pose,
                local_x=local_x,
                local_y=local_y,
                map_name=self.map_profile.name,
            )

    def _heading_frame_sample(self) -> FrameSample | None:
        self.cancellation.raise_if_cancelled()
        supplier = getattr(self.bot, "get_frame_sample", None)
        if not callable(supplier):
            return None
        return supplier()

    def _read_minimap_heading(self, context: str) -> HeadingReading | None:
        if not self.config.minimap_heading_enabled:
            return None
        supplier = getattr(self.bot, "get_frame_sample", None)
        if not callable(supplier):
            if not getattr(self, "_heading_warning_announced", False):
                self.status_callback(
                    "Minimap heading fusion is enabled, but the attached bot does not "
                    "provide fresh frame samples. Native displacement fallback remains "
                    "active."
                )
                self._heading_warning_announced = True
            return None
        try:
            reading = self.heading_detector.read_strict(
                self._heading_frame_sample,
                samples=self.config.minimap_heading_samples,
                delay=0.015,
                fresh=True,
                require_distinct_frames=True,
                maximum_uncertainty_deg=(
                    self.config.minimap_heading_max_uncertainty_deg
                ),
                maximum_ambiguity=self.config.minimap_heading_max_ambiguity,
            )
        except Exception as error:  # noqa: BLE001 - vision is a safety aid.
            self.status_callback(
                f"Minimap heading read failed during {context}: {error}. "
                "No collision evidence will rely on an unverified commanded heading."
            )
            return None
        if (
            reading is None
            or reading.confidence < self.config.minimap_heading_min_confidence
        ):
            return None
        self._heading_warning_announced = False
        return reading

    def _set_heading_from_minimap(
        self,
        reading: HeadingReading,
        *,
        context: str,
    ) -> None:
        self.grid.set_heading_degrees(reading.angle_deg)
        self._heading_source = "minimap"
        uncertainty = reading.angular_uncertainty_deg
        self.grid.metadata.heading_uncertainty_deg = (
            None if uncertainty is None else float(uncertainty)
        )
        self.grid.set_pose_reliability(
            position_known=True,
            heading_known=True,
            note=f"Absolute heading verified from minimap arrow during {context}.",
        )

    def _mark_heading_uncertain(self, note: str) -> None:
        self._heading_source = "uncertain"
        self.grid.set_pose_reliability(
            position_known=True,
            heading_known=False,
            note=note,
        )

    def _collision_heading_is_trusted(self, *, context: str) -> bool:
        if self._heading_source in {"movement", "minimap", "position-only"}:
            # position-only preserves a previously verified heading while
            # integrating partial/sliding travel.
            return True

        prior = self.grid.continuous_pose.heading_deg % 360.0
        reading = self._read_minimap_heading(context)
        if reading is None:
            self._mark_heading_uncertain(
                f"{context}; blocker evidence is suspended until heading is verified"
            )
            return False
        correction = abs(signed_angle_delta(reading.angle_deg, prior))
        self._set_heading_from_minimap(reading, context=context)
        if correction >= self.config.heading_collision_guard_degrees:
            self.status_callback(
                f"map step={self._step}: heading was corrected by {correction:.1f}° "
                f"before accepting {context}; the current collision attempt is "
                "discarded and will be retried in the corrected direction."
            )
            return False
        return True

    def _acquire_heading(self) -> None:
        reading = self._read_minimap_heading("initial heading")
        if reading is not None:
            self._set_heading_from_minimap(reading, context="startup")
            self.status_callback(
                f"Initial heading acquired from the minimap arrow at "
                f"{reading.angle_deg:.1f}°. The next reliable native movement will "
                "refine it to the precise coordinate-derived heading."
            )
            self._publish_map(force=True)
            return

        self.status_callback(
            "Minimap heading was unavailable, so the mapper is falling back to a "
            "native displacement probe. A blocked probe turns right once and retries."
        )
        for attempt in range(1, self.config.heading_acquisition_attempts + 1):
            self.cancellation.raise_if_cancelled()
            self._wait_for_game_focus()
            motion, timing = self._measure_forward()
            self._step += 1
            self._log_motion(
                motion,
                action="HEADING_PROBE",
                reason="native heading acquisition",
                attempt=attempt,
                timing=timing,
                note="",
            )
            if motion.outcome == "teleport":
                raise MapTransitionDetected(
                    before=motion.before,
                    after=motion.after,
                    horizontal_distance=motion.horizontal_distance,
                    vertical_distance=motion.vertical_distance,
                    context="heading acquisition",
                )
            if motion.outcome != "blocked":
                infer_heading = self._motion_can_refine_heading(motion)
                heading = self._integrate_motion(
                    motion,
                    infer_heading=infer_heading,
                )
                self._heading_source = "movement"
                self.grid.set_pose_reliability(
                    position_known=True,
                    heading_known=True,
                    note="Heading measured from native X/Z displacement.",
                )
                self.status_callback(
                    f"Heading acquired at {heading:.1f}° from "
                    f"{motion.horizontal_distance:.3f} native units of travel."
                )
                self._publish_map(force=True)
                return

            if attempt < self.config.heading_acquisition_attempts:
                self._wait_for_game_focus()
                timing_turn = self.controller.turn_right(
                    self.config.turn_right_90_seconds
                )
                self.controller.settle(self.config.settle_seconds)
                self._verify_pose_continuity("after heading-acquisition turn")
                self._log_turn(
                    action="TURN_RIGHT",
                    reason="heading probe was blocked",
                    timing=timing_turn,
                    note="Absolute heading still unknown.",
                )

        raise RuntimeError(
            "Could not acquire heading: every native forward probe was blocked. "
            "Move the character to a less enclosed starting point and retry."
        )

    def _handle_no_frontier(self) -> ExplorerDecision | None:
        """Apply enclosure proof before allowing the mapper to terminate."""
        inferred_free = self._apply_free_space_inference(
            context="planner reported no frontier",
            force=True,
        )
        if inferred_free:
            retry = self.explorer.decide(self.grid)
            if retry.action != "STOP":
                return ExplorerDecision(
                    retry.action,
                    "free-space autofill removed tiny redundant frontier gaps",
                )
        report = self.completion_guard.analyze(self.grid)

        if report.candidate_complete:
            self._completion_streak += 1
            self.status_callback(
                "Completion safeguard passed "
                f"{self._completion_streak}/{self.config.completion_stable_checks}: "
                f"{report.reason}; outer_wall_cells={report.blocked_perimeter_cells}, "
                f"free_cells={report.free_cells}, obstacle_void_unknown="
                f"{report.enclosed_obstacle_void_cells}."
            )
            if self._completion_streak >= self.config.completion_stable_checks:
                message = (
                    "Mapping completed: a closed blocked outer perimeter was proven "
                    "and no explorable unknown interior cells remain."
                )
                self.grid.metadata.termination_reason = message
                self.status_callback(message)
                return None
            return self._persistent_probe_decision(
                "completion verification probe before another topology check"
            )

        self._completion_streak = 0
        self.status_callback(
            "Completion safeguard rejected planner stop: "
            f"{report.reason}; free={report.free_cells}, "
            f"outer_wall={report.blocked_perimeter_cells}, "
            f"unknown_inside={report.unresolved_unknown_cells}. Continuing."
        )

        cleared = self.grid.clear_temporary_avoidances()
        inferred_cleared = self.grid.clear_inferred_walls()
        softened = self.grid.remove_free_to_free_contact_boundaries()
        if cleared or softened or inferred_cleared:
            self.status_callback(
                "Planner stall cleanup removed "
                f"{cleared} temporary pocket avoidance(s) and "
                f"{softened} stale free-to-free collision edge(s), and released "
                f"{inferred_cleared} inferred wall cell(s) for direct reprobe."
            )
            retry = self.explorer.decide(self.grid)
            if retry.action != "STOP":
                return retry

        retry = self.explorer.decide(
            self.grid,
            ignore_contact_boundaries=True,
        )
        if retry.action != "STOP":
            return ExplorerDecision(
                retry.action,
                "completion safeguard retry ignoring edge-only avoidance",
            )

        patrol_path = self.grid.least_visited_free_path(
            ignore_contact_boundaries=True,
        )
        if patrol_path:
            return self._decision_toward(
                patrol_path[0],
                "persistent coverage patrol after incomplete-map safeguard",
            )

        return self._persistent_probe_decision(
            "no route to a frontier; bounded persistent directional probe"
        )

    def _persistent_probe_decision(self, reason: str) -> ExplorerDecision:
        # Alternate a cardinal turn with a measured forward attempt. The action
        # sequence is unbounded at run level, while each key press remains bounded.
        if self._stall_probe_turn_next:
            self._stall_probe_turn_next = False
            return ExplorerDecision("TURN_RIGHT", reason)
        self._stall_probe_turn_next = True
        return ExplorerDecision("FORWARD", reason)

    def _decision_toward(
        self,
        target: tuple[int, int],
        reason: str,
    ) -> ExplorerDecision:
        pose = self.grid.pose
        dx = target[0] - pose.x
        dy = target[1] - pose.y
        desired = Explorer._direction_index(dx, dy)
        return Explorer._turn_or_forward(pose.heading_index, desired, reason)

    def _execute_decision(self, decision: ExplorerDecision) -> None:
        if decision.action == "FORWARD":
            self._execute_forward(decision.reason)
            return
        if decision.action == "TURN_LEFT":
            self._execute_turn(left=True, reason=decision.reason)
            return
        if decision.action == "TURN_RIGHT":
            self._execute_turn(left=False, reason=decision.reason)
            return
        raise ValueError(f"Unsupported mapper action: {decision.action}")

    def _execute_turn(self, *, left: bool, reason: str) -> None:
        self._wait_for_game_focus()
        old_index = self.grid.pose.heading_index
        current_heading = self.grid.continuous_pose.heading_deg % 360.0
        if left:
            new_index = (old_index + 1) % 4
            target_heading = self.grid.heading_degrees_from_index(new_index)
            turn_degrees = (current_heading - target_heading) % 360.0
            if turn_degrees > 180.0:
                turn_degrees = 360.0 - turn_degrees
            requested = self.config.turn_left_90_seconds * turn_degrees / 90.0
            timing = self.controller.turn_left(max(0.015, requested))
            action = "TURN_LEFT"
        else:
            new_index = (old_index - 1) % 4
            target_heading = self.grid.heading_degrees_from_index(new_index)
            turn_degrees = (target_heading - current_heading) % 360.0
            if turn_degrees > 180.0:
                turn_degrees = 360.0 - turn_degrees
            requested = self.config.turn_right_90_seconds * turn_degrees / 90.0
            timing = self.controller.turn_right(max(0.015, requested))
            action = "TURN_RIGHT"
        self.controller.settle(self.config.settle_seconds)
        self._verify_pose_continuity(f"after {action}")
        self.grid.set_heading_degrees(target_heading)
        self._heading_source = "commanded-unverified"
        self.grid.set_pose_reliability(
            position_known=True,
            heading_known=False,
            note=(
                "Heading estimated from a scaled one-shot turn; minimap verification "
                "is pending. Collision evidence is suppressed until verified."
            ),
        )
        self._log_turn(
            action=action,
            reason=reason,
            timing=timing,
            note=f"scaled one-shot turn ({turn_degrees:.1f} degrees)",
        )
        self.status_callback(
            f"map step={self._step} action={action} one-shot={timing.held_seconds:.3f}s "
            f"requested_turn={turn_degrees:.1f}° estimated_heading={target_heading:.1f}°; "
            "checking the minimap arrow for absolute correction."
        )
        reading = self._read_minimap_heading(f"after {action}")
        if reading is not None:
            correction = abs(signed_angle_delta(reading.angle_deg, target_heading))
            self._set_heading_from_minimap(reading, context=f"after {action}")
            self.status_callback(
                f"map step={self._step}: minimap heading after {action} is "
                f"{reading.angle_deg:.1f}° (command estimate error {correction:.1f}°)."
            )
        else:
            self.status_callback(
                f"map step={self._step}: minimap heading could not be verified after "
                f"{action}; movement may continue, but no wall will be written until "
                "heading is verified or refined by aligned native movement."
            )

    def _execute_forward(self, reason: str) -> None:
        final_motion: NativeMotion | None = None
        final_count = 0
        final_confirmed = False
        fixed_edge: tuple[tuple[int, int], tuple[int, int]] | None = None
        evidence_attempts = 0
        measurement_attempt = 0

        while evidence_attempts < self.config.blocked_confirmations:
            self.cancellation.raise_if_cancelled()
            measurement_attempt += 1
            motion, timing = self._measure_forward()
            final_motion = motion

            if motion.outcome == "teleport":
                self._log_motion(
                    motion,
                    action="FORWARD",
                    reason=reason,
                    attempt=measurement_attempt,
                    timing=timing,
                    note="teleport-sized displacement",
                )
                raise MapTransitionDetected(
                    before=motion.before,
                    after=motion.after,
                    horizontal_distance=motion.horizontal_distance,
                    vertical_distance=motion.vertical_distance,
                    context="forward movement",
                )

            # The first movement immediately after EVA is not trusted as wall
            # evidence.  The cast lock can intermittently outlive the normal
            # settle delay.  Preserve any actual displacement, wait once more,
            # and repeat the same intended action.
            if getattr(self, "_post_eva_guard_pending", False):
                self._post_eva_guard_pending = False
                if motion.outcome in {"blocked", "partial", "sliding"}:
                    if motion.outcome != "blocked":
                        self._integrate_motion(motion, infer_heading=False)
                    self._log_motion(
                        motion,
                        action="FORWARD",
                        reason=reason,
                        attempt=measurement_attempt,
                        timing=timing,
                        note="post-EVA movement result ignored as collision evidence",
                    )
                    self.status_callback(
                        f"map step={self._step}: post-EVA {motion.outcome} result "
                        "was not recorded as a wall; waiting once more and retrying."
                    )
                    self.controller.settle(self.config.eva_retry_settle_seconds)
                    continue

            evidence_attempts += 1

            if motion.outcome == "sliding":
                motion, heading_recovered = self._resolve_suspicious_motion(
                    motion,
                    forward=True,
                    context="forward movement disagreed with expected heading",
                )
                if motion.outcome != "sliding":
                    self._log_motion(
                        motion,
                        action="FORWARD",
                        reason=reason,
                        attempt=measurement_attempt,
                        timing=timing,
                        note=(
                            "minimap verified/corrected heading; movement was "
                            "reclassified without blocker evidence"
                        ),
                    )
                    infer_heading = self._motion_can_refine_heading(motion)
                    heading = self._integrate_motion(
                        motion,
                        infer_heading=infer_heading,
                    )
                    correction_note = (
                        "corrected stale heading"
                        if heading_recovered
                        else "verified the commanded heading"
                    )
                    self.status_callback(
                        f"map step={self._step}: minimap {correction_note}; apparent "
                        f"obstacle slide was reclassified as {motion.outcome}, so no "
                        f"blocker was added. Native={motion.horizontal_distance:.4f}u "
                        f"heading={heading:.1f}°."
                    )
                    self._clear_trap_evidence()
                    return
                if self._heading_source == "uncertain":
                    self._integrate_motion(motion, infer_heading=False)
                    self._mark_heading_uncertain(
                        "Suspicious lateral movement was integrated, but absolute "
                        "heading still requires minimap verification."
                    )
                    self._log_motion(
                        motion,
                        action="FORWARD",
                        reason=reason,
                        attempt=measurement_attempt,
                        timing=timing,
                        note=(
                            "suspicious lateral movement integrated, but blocker "
                            "suppressed because absolute heading could not be verified"
                        ),
                    )
                    self.status_callback(
                        f"map step={self._step}: suspicious lateral movement was kept "
                        "as free travel, but no wall was written because heading could "
                        "not be verified."
                    )
                    return
                fixed_edge = fixed_edge or self._intended_edge(motion.before)
                heading = self._integrate_motion(motion, infer_heading=False)
                final_count, final_confirmed = self._record_blocked_boundary(
                    fixed_edge,
                    evidence=self.config.blocked_confirmations,
                    force_target=True,
                )
                self._log_motion(
                    motion,
                    action="FORWARD",
                    reason=reason,
                    attempt=measurement_attempt,
                    timing=timing,
                    note=(
                        "lateral slide integrated as free travel; intended boundary "
                        "recorded as blocked"
                    ),
                )
                self.status_callback(
                    f"map step={self._step} action=FORWARD native=sliding/"
                    f"{motion.horizontal_distance:.4f}u forward="
                    f"{motion.forward_progress:.4f}u lateral="
                    f"{motion.lateral_distance:.4f}u boundary="
                    f"{fixed_edge[0]}->{fixed_edge[1]} confirmations={final_count}; "
                    f"actual slide path integrated, heading preserved at {heading:.1f}°."
                )
                should_escape = self._register_trap_evidence(
                    self.config.blocked_trap_score,
                    fallback_edge=fixed_edge,
                )
                if should_escape:
                    self._recover_from_trap("repeated obstacle sliding")
                return

            if motion.outcome == "blocked":
                if not self._collision_heading_is_trusted(
                    context="forward blocked result"
                ):
                    self._log_motion(
                        motion,
                        action="FORWARD",
                        reason=reason,
                        attempt=measurement_attempt,
                        timing=timing,
                        note=(
                            "blocked result ignored because absolute heading was not "
                            "verified"
                        ),
                    )
                    self.status_callback(
                        f"map step={self._step}: near-zero forward movement was not "
                        "written as a wall because heading is unverified."
                    )
                    return
                if fixed_edge is None:
                    fixed_edge = self._intended_edge(motion.before)
                final_count, final_confirmed = self._record_blocked_boundary(
                    fixed_edge
                )
                note = (
                    "near-zero displacement; confirming same attempted boundary"
                    if not final_confirmed
                    else "blocked boundary confirmed from native displacement"
                )
            else:
                note = ""

            self._log_motion(
                motion,
                action="FORWARD",
                reason=reason,
                attempt=measurement_attempt,
                timing=timing,
                note=note,
            )

            if motion.outcome != "blocked":
                infer_heading = self._motion_can_refine_heading(motion)
                heading = self._integrate_motion(
                    motion,
                    infer_heading=infer_heading,
                )
                if motion.outcome == "partial":
                    should_escape = self._register_trap_evidence(
                        self.config.partial_trap_score,
                        fallback_edge=fixed_edge,
                    )
                else:
                    self._clear_trap_evidence()
                    should_escape = False
                self.status_callback(
                    f"map step={self._step} action=FORWARD native={motion.outcome}/"
                    f"{motion.horizontal_distance:.4f}u forward="
                    f"{motion.forward_progress:.4f}u lateral="
                    f"{motion.lateral_distance:.4f}u pose=("
                    f"{self.grid.continuous_pose.x:.2f},"
                    f"{self.grid.continuous_pose.y:.2f}) heading={heading:.1f}°."
                )
                if should_escape:
                    self._recover_from_trap("repeated partial movement")
                return

            if not final_confirmed:
                self.status_callback(
                    f"map step={self._step} forward displacement "
                    f"{motion.horizontal_distance:.4f}u; confirming the same boundary."
                )
                self.controller.settle(self.config.settle_seconds)

        if final_motion is None or fixed_edge is None:
            raise RuntimeError("Forward block confirmation ended without a measurement")
        if not final_confirmed:
            final_count, final_confirmed = self._record_blocked_boundary(fixed_edge)

        self.status_callback(
            f"map step={self._step} action=FORWARD native=blocked/"
            f"{final_motion.horizontal_distance:.4f}u boundary="
            f"{fixed_edge[0]}->{fixed_edge[1]} confirmations={final_count}; replanning."
        )
        should_escape = self._register_trap_evidence(
            self.config.blocked_trap_score,
            fallback_edge=fixed_edge,
        )
        if should_escape:
            self._recover_from_trap("multiple blocked directions in a confined area")

    def _measure_forward(self) -> tuple[NativeMotion, KeyPressTiming]:
        return self._measure_linear(forward=True)

    def _measure_backward(
        self,
        seconds: float | None = None,
    ) -> tuple[NativeMotion, KeyPressTiming]:
        return self._measure_linear(forward=False, seconds=seconds)

    def _measure_linear(
        self,
        *,
        forward: bool,
        seconds: float | None = None,
    ) -> tuple[NativeMotion, KeyPressTiming]:
        self._wait_for_game_focus()
        before = self._verify_pose_continuity(
            "before forward movement" if forward else "before backward movement"
        )
        duration = (
            self.config.forward_seconds
            if forward
            else self.config.backward_seconds
        )
        if seconds is not None:
            duration = float(seconds)
        if forward:
            timing = self.controller.forward(duration)
        else:
            timing = self.controller.backward(duration)
        self.controller.settle(self.config.settle_seconds)
        after = self._read_pose()
        delta_x = after.x - before.x
        delta_y = after.y - before.y
        delta_z = after.z - before.z
        horizontal = math.hypot(delta_x, delta_z)
        vertical = abs(delta_y)

        if getattr(self, "_heading_source", "unknown") == "unknown":
            # Heading acquisition cannot judge alignment against a direction we
            # do not know yet. Use total displacement until the first reliable
            # native movement establishes the heading.
            forward_progress = horizontal
            lateral = 0.0
            alignment = 1.0 if horizontal > 1e-9 else 0.0
        else:
            heading_radians = math.radians(self.grid.continuous_pose.heading_deg)
            intended_x = math.sin(heading_radians)
            intended_z = math.cos(heading_radians)
            if not forward:
                intended_x = -intended_x
                intended_z = -intended_z
            forward_progress = delta_x * intended_x + delta_z * intended_z
            lateral = abs(delta_x * intended_z - delta_z * intended_x)
            alignment = forward_progress / horizontal if horizontal > 1e-9 else 0.0

        outcome = self._classify_linear_values(
            horizontal=horizontal,
            vertical=vertical,
            forward_progress=forward_progress,
            lateral=lateral,
            alignment=alignment,
            forward=forward,
        )
        if outcome != "teleport":
            self._trusted_pose = after
        return (
            NativeMotion(
                before=before,
                after=after,
                delta_x=delta_x,
                delta_y=delta_y,
                delta_z=delta_z,
                horizontal_distance=horizontal,
                vertical_distance=vertical,
                outcome=outcome,
                forward_progress=forward_progress,
                lateral_distance=lateral,
                forward_alignment=alignment,
            ),
            timing,
        )

    def _classify_linear_values(
        self,
        *,
        horizontal: float,
        vertical: float,
        forward_progress: float,
        lateral: float,
        alignment: float,
        forward: bool,
    ) -> str:
        blocked_threshold = (
            self.config.blocked_distance_units
            if forward
            else self.config.backward_blocked_distance_units
        )
        partial_threshold = (
            self.config.partial_distance_units
            if forward
            else self.config.backward_partial_distance_units
        )
        if self._is_transition_distance(horizontal, vertical):
            return "teleport"
        if horizontal <= blocked_threshold:
            return "blocked"
        if (
            lateral >= self.config.slide_min_lateral_units
            and alignment < self.config.slide_min_forward_alignment
        ):
            return "sliding"
        if forward_progress < partial_threshold:
            return "partial"
        return "moved"

    def _reclassify_motion_for_heading(
        self,
        motion: NativeMotion,
        *,
        forward: bool,
        heading_deg: float,
    ) -> NativeMotion:
        heading_radians = math.radians(float(heading_deg) % 360.0)
        intended_x = math.sin(heading_radians)
        intended_z = math.cos(heading_radians)
        if not forward:
            intended_x = -intended_x
            intended_z = -intended_z
        forward_progress = motion.delta_x * intended_x + motion.delta_z * intended_z
        lateral = abs(motion.delta_x * intended_z - motion.delta_z * intended_x)
        alignment = (
            forward_progress / motion.horizontal_distance
            if motion.horizontal_distance > 1e-9
            else 0.0
        )
        outcome = self._classify_linear_values(
            horizontal=motion.horizontal_distance,
            vertical=motion.vertical_distance,
            forward_progress=forward_progress,
            lateral=lateral,
            alignment=alignment,
            forward=forward,
        )
        return NativeMotion(
            before=motion.before,
            after=motion.after,
            delta_x=motion.delta_x,
            delta_y=motion.delta_y,
            delta_z=motion.delta_z,
            horizontal_distance=motion.horizontal_distance,
            vertical_distance=motion.vertical_distance,
            outcome=outcome,
            forward_progress=forward_progress,
            lateral_distance=lateral,
            forward_alignment=alignment,
        )

    def _motion_can_refine_heading(self, motion: NativeMotion) -> bool:
        return (
            motion.outcome == "moved"
            and motion.horizontal_distance
            >= self.config.heading_motion_min_distance_units
            and motion.forward_alignment >= self.config.slide_min_forward_alignment
        )

    def _resolve_suspicious_motion(
        self,
        motion: NativeMotion,
        *,
        forward: bool,
        context: str,
    ) -> tuple[NativeMotion, bool]:
        """Use the minimap to distinguish a real slide from stale heading.

        Returns ``(motion, heading_recovered)``. If the minimap is unavailable,
        the heading is marked uncertain and the caller must suppress blocker
        evidence for this attempt.
        """
        if motion.horizontal_distance < self.config.heading_motion_min_distance_units:
            return motion, False

        expected_heading = self.grid.continuous_pose.heading_deg % 360.0
        if not forward:
            expected_heading = (expected_heading + 180.0) % 360.0
        measured_heading = self.heading_from_delta(motion.delta_x, motion.delta_z)
        mismatch = abs(signed_angle_delta(measured_heading, expected_heading))
        if mismatch < self.config.heading_mismatch_recheck_degrees:
            return motion, False

        reading = self._read_minimap_heading(context)
        if reading is None:
            self._mark_heading_uncertain(
                f"{context}; minimap verification was unavailable"
            )
            return motion, False

        prior_heading = self.grid.continuous_pose.heading_deg % 360.0
        self._set_heading_from_minimap(reading, context=context)
        reclassified = self._reclassify_motion_for_heading(
            motion,
            forward=forward,
            heading_deg=reading.angle_deg,
        )
        corrected = (
            abs(signed_angle_delta(reading.angle_deg, prior_heading))
            >= self.config.heading_collision_guard_degrees
        )
        return reclassified, corrected

    def _integrate_motion(
        self,
        motion: NativeMotion,
        *,
        infer_heading: bool = True,
    ) -> float:
        frame = self._require_frame()
        before_x, before_y = frame.to_local_cells(motion.before.x, motion.before.z)
        after_x, after_y = frame.to_local_cells(motion.after.x, motion.after.z)
        if infer_heading:
            heading = self.heading_from_delta(motion.delta_x, motion.delta_z)
            # Native displacement is authoritative while the character is
            # translating forward. Repeated large disagreement invalidates a
            # cached minimap centre so the next visual turn read reacquires the
            # Navigator instead of trusting a wrong circular UI element.
            detector = getattr(self, "heading_detector", None)
            if detector is not None:
                detector.observe_reference_heading(heading)
        else:
            heading = self.grid.continuous_pose.heading_deg

        traversed = self.grid.rasterize_segment(
            before_x,
            before_y,
            after_x,
            after_y,
        )
        for cell_x, cell_y in traversed:
            self.grid.mark_free(cell_x, cell_y)
        for first, second in zip(traversed, traversed[1:]):
            if abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1:
                self.grid.remove_contact_boundary(*first, *second)
                self.grid.remove_temporary_avoidance(*first, *second)
                self._recent_traversed_edges.append((first, second))
        if len(self._recent_traversed_edges) > 64:
            del self._recent_traversed_edges[:-64]

        previous_heading_source = getattr(self, "_heading_source", "unknown")
        self.grid.set_continuous_pose(after_x, after_y, heading)
        if infer_heading:
            self._heading_source = "movement"
            heading_known = True
            note = "Position and heading measured from native coordinate displacement."
        elif previous_heading_source in {"movement", "minimap", "position-only"}:
            self._heading_source = "position-only"
            heading_known = True
            note = "Position measured from native coordinates; verified facing preserved."
        else:
            # A partial or lateral move does not magically validate a heading
            # that was already uncertain after a turn or vision failure.
            self._heading_source = "uncertain"
            heading_known = False
            note = (
                "Position measured from native coordinates; facing remains "
                "unverified and collision evidence is suspended."
            )
        self.grid.set_pose_reliability(
            position_known=True,
            heading_known=heading_known,
            note=note,
        )

        current_cell = self.grid.world_point_to_grid_cell(before_x, before_y)
        for key in tuple(self._blocked_counts):
            if key[:2] == current_cell:
                self._blocked_counts.pop(key, None)
        self._apply_free_space_inference(context="measured traversal")
        return heading

    def _intended_edge(
        self,
        before: PlayerPose,
        *,
        forward: bool = True,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        frame = self._require_frame()
        local_x, local_y = frame.to_local_cells(before.x, before.z)
        from_cell = self.grid.world_point_to_grid_cell(local_x, local_y)
        direction = self.grid.DIRECTIONS[self.grid.pose.heading_index]
        if not forward:
            direction = (-direction[0], -direction[1])
        return from_cell, (from_cell[0] + direction[0], from_cell[1] + direction[1])

    def _record_blocked_boundary(
        self,
        edge: tuple[tuple[int, int], tuple[int, int]],
        *,
        evidence: int = 1,
        force_target: bool = False,
        required_confirmations: int | None = None,
        attempted_heading_deg: float | None = None,
    ) -> tuple[int, bool]:
        from_cell, to_cell = edge
        key = (*from_cell, *to_cell)
        count = self._blocked_counts.get(key, 0) + max(1, int(evidence))
        self._blocked_counts[key] = count
        threshold = (
            self.config.blocked_confirmations
            if required_confirmations is None
            else max(1, int(required_confirmations))
        )
        confirmed = count >= threshold
        if confirmed:
            self.grid.add_contact_boundary(
                from_x=from_cell[0],
                from_y=from_cell[1],
                to_x=to_cell[0],
                to_y=to_cell[1],
                heading_deg=(
                    self.grid.continuous_pose.heading_deg
                    if attempted_heading_deg is None
                    else float(attempted_heading_deg) % 360.0
                ),
                confirmations=count,
            )
            if force_target:
                self.grid.set_authoritative_blocked(*to_cell)
            else:
                self.grid.mark_blocked(*to_cell)
            self._apply_wall_inference(context=f"confirmed boundary at {to_cell}")
        return count, confirmed

    def _clear_blocked_evidence(
        self,
        edge: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        self._blocked_counts.pop((*edge[0], *edge[1]), None)

    def _apply_wall_inference(self, *, context: str) -> int:
        if not self.config.wall_autofill_enabled:
            return 0
        inference = getattr(self, "wall_inference", None)
        if inference is None:
            # Some focused tests construct a mapper without running __init__.
            # Lazily building the helper also keeps this method robust for old
            # serialized/controller integrations that predate wall autofill.
            inference = WallInference(
                max_gap_cells=self.config.wall_autofill_max_gap_cells,
                minimum_support_cells=self.config.wall_autofill_min_support_cells,
                support_radius_cells=self.config.wall_autofill_support_radius_cells,
                maximum_line_error_cells=(
                    self.config.wall_autofill_max_line_error_cells
                ),
            )
            self.wall_inference = inference
        result = inference.infer(self.grid)
        if result.added_count:
            self.status_callback(
                f"map step={self._step}: wall autofill added "
                f"{result.added_count} reversible cell(s) after {context}; "
                "they guide planning but do not count as completion proof."
            )
        return result.added_count

    def _apply_free_space_inference(
        self,
        *,
        context: str,
        force: bool = False,
    ) -> int:
        if not self.config.free_space_autofill_enabled:
            return 0
        current_step = getattr(self, "_step", 0)
        if (
            not force
            and current_step - getattr(self, "_last_free_autofill_step", -10_000)
            < self.config.free_space_autofill_interval_steps
        ):
            return 0
        inference = getattr(self, "free_space_inference", None)
        if inference is None:
            inference = FreeSpaceInference(
                maximum_enclosed_area_cells=(
                    self.config.free_space_autofill_max_enclosed_area_cells
                ),
                maximum_enclosed_span_cells=(
                    self.config.free_space_autofill_max_enclosed_span_cells
                ),
                maximum_line_length_cells=(
                    self.config.free_space_autofill_max_line_length_cells
                ),
            )
            self.free_space_inference = inference
        self._last_free_autofill_step = current_step
        result = inference.infer(self.grid)
        if result.added_count:
            self.status_callback(
                f"map step={current_step}: free-space autofill marked "
                f"{result.added_count} tiny unknown cell(s) free after {context} "
                f"({result.enclosed_hole_cells} enclosed-hole candidates, "
                f"{result.narrow_line_cells} one-cell-line candidates). "
                "They are reversible if later collision evidence disagrees."
            )
        return result.added_count

    def _register_trap_evidence(
        self,
        score: int,
        *,
        fallback_edge: tuple[tuple[int, int], tuple[int, int]] | None,
    ) -> bool:
        if self._trap_entry_edge is None:
            if self._recent_traversed_edges:
                self._trap_entry_edge = self._recent_traversed_edges[-1]
            elif fallback_edge is not None:
                self._trap_entry_edge = fallback_edge
        self._trap_score += max(1, int(score))
        return self._trap_score >= self.config.trap_score_threshold

    def _clear_trap_evidence(self) -> None:
        self._trap_score = 0
        self._trap_entry_edge = None

    def _recover_from_trap(self, reason: str) -> None:
        entry_edge = self._trap_entry_edge
        self.status_callback(
            f"map step={self._step}: confined-space recovery triggered ({reason}); "
            "reversing along the measured path before trying more frontiers."
        )
        total_retreat = 0.0
        blocked_attempts = 0
        successful_steps = 0

        for attempt in range(1, self.config.escape_backward_steps + 1):
            self.cancellation.raise_if_cancelled()
            motion, timing = self._measure_backward()
            if motion.outcome == "teleport":
                raise MapTransitionDetected(
                    before=motion.before,
                    after=motion.after,
                    horizontal_distance=motion.horizontal_distance,
                    vertical_distance=motion.vertical_distance,
                    context="trap recovery",
                )

            rear_edge = self._intended_edge(motion.before, forward=False)
            self._log_motion(
                motion,
                action="BACKWARD_ESCAPE",
                reason=reason,
                attempt=attempt,
                timing=timing,
                note=(
                    "native reverse-path escape using slower backpedal thresholds; "
                    "rear collisions require an extended confirmation"
                ),
            )

            # A lateral reverse movement can mean either a genuine slide along a
            # wall behind the player or a stale heading estimate. Re-read the
            # minimap before writing any blocker.
            if motion.outcome == "sliding":
                motion, _heading_recovered = self._resolve_suspicious_motion(
                    motion,
                    forward=False,
                    context="backpedal direction disagreed with expected heading",
                )
                if motion.outcome != "sliding":
                    self._clear_blocked_evidence(rear_edge)
                    self._integrate_motion(motion, infer_heading=False)
                    successful_steps += 1
                    retreat = max(0.0, motion.forward_progress)
                    total_retreat += retreat
                    self.status_callback(
                        f"map step={self._step}: minimap reclassified apparent "
                        f"backpedal sliding as {motion.outcome}; no rear blocker was "
                        f"added. Reverse progress={retreat:.4f}u."
                    )
                    if total_retreat >= self.config.escape_target_distance_units:
                        break
                    continue

                self._integrate_motion(motion, infer_heading=False)
                successful_steps += 1
                if self._heading_source == "uncertain":
                    self.status_callback(
                        f"map step={self._step}: backpedal moved laterally, but the "
                        "heading could not be verified; the actual path was kept free "
                        "and no rear wall was written."
                    )
                    continue

                attempted_heading = (
                    self.grid.continuous_pose.heading_deg + 180.0
                ) % 360.0
                count, confirmed = self._record_blocked_boundary(
                    rear_edge,
                    evidence=self.config.backward_blocked_confirmations,
                    force_target=True,
                    required_confirmations=(
                        self.config.backward_blocked_confirmations
                    ),
                    attempted_heading_deg=attempted_heading,
                )
                blocked_attempts += 1
                self.status_callback(
                    f"map step={self._step}: confirmed backpedal slide; actual "
                    f"lateral path was marked free and rear boundary "
                    f"{rear_edge[0]}->{rear_edge[1]} was marked blocked "
                    f"(confirmations={count}, confirmed={confirmed})."
                )
                if (
                    blocked_attempts
                    % self.config.escape_wiggle_after_blocked_attempts
                    == 0
                ):
                    self._escape_wiggle(
                        right=(
                            blocked_attempts
                            // self.config.escape_wiggle_after_blocked_attempts
                        )
                        % 2
                        == 1
                    )
                continue

            if motion.outcome == "blocked":
                self.status_callback(
                    f"map step={self._step}: near-zero backpedal "
                    f"({motion.horizontal_distance:.4f}u); retrying with a longer "
                    "pulse before accepting a rear collision."
                )
                retry_motion, retry_timing = self._measure_backward(
                    self.config.backward_retry_seconds
                )
                if retry_motion.outcome == "teleport":
                    raise MapTransitionDetected(
                        before=retry_motion.before,
                        after=retry_motion.after,
                        horizontal_distance=retry_motion.horizontal_distance,
                        vertical_distance=retry_motion.vertical_distance,
                        context="trap recovery backward recheck",
                    )
                self._log_motion(
                    retry_motion,
                    action="BACKWARD_ESCAPE_RECHECK",
                    reason=reason,
                    attempt=attempt,
                    timing=retry_timing,
                    note=(
                        "extended backpedal confirmation; only two aligned near-zero "
                        "attempts may create rear collision evidence"
                    ),
                )
                motion = retry_motion

                if motion.outcome == "sliding":
                    motion, _heading_recovered = self._resolve_suspicious_motion(
                        motion,
                        forward=False,
                        context="extended backpedal slid laterally",
                    )
                    if motion.outcome != "sliding":
                        self._clear_blocked_evidence(rear_edge)
                        self._integrate_motion(motion, infer_heading=False)
                        successful_steps += 1
                        retreat = max(0.0, motion.forward_progress)
                        total_retreat += retreat
                        self.status_callback(
                            f"map step={self._step}: extended reverse recheck moved "
                            f"normally after heading verification; no rear blocker was "
                            f"added. Reverse progress={retreat:.4f}u."
                        )
                        if total_retreat >= self.config.escape_target_distance_units:
                            break
                        continue

                    self._integrate_motion(motion, infer_heading=False)
                    successful_steps += 1
                    if self._heading_source != "uncertain":
                        attempted_heading = (
                            self.grid.continuous_pose.heading_deg + 180.0
                        ) % 360.0
                        count, confirmed = self._record_blocked_boundary(
                            rear_edge,
                            evidence=self.config.backward_blocked_confirmations,
                            force_target=True,
                            required_confirmations=(
                                self.config.backward_blocked_confirmations
                            ),
                            attempted_heading_deg=attempted_heading,
                        )
                        self.status_callback(
                            f"map step={self._step}: extended reverse recheck "
                            f"confirmed a slide; rear boundary {rear_edge[0]}->"
                            f"{rear_edge[1]} is blocked (confirmations={count}, "
                            f"confirmed={confirmed})."
                        )
                    else:
                        self.status_callback(
                            f"map step={self._step}: extended reverse recheck slid, "
                            "but heading remained uncertain; no rear blocker was added."
                        )
                    blocked_attempts += 1
                    continue

            if motion.outcome == "blocked":
                blocked_attempts += 1
                if self._collision_heading_is_trusted(
                    context="confirmed backpedal collision"
                ):
                    attempted_heading = (
                        self.grid.continuous_pose.heading_deg + 180.0
                    ) % 360.0
                    # The short pulse and extended recheck are two independent
                    # near-zero observations of the same fixed rear boundary.
                    count, confirmed = self._record_blocked_boundary(
                        rear_edge,
                        evidence=2,
                        force_target=True,
                        required_confirmations=(
                            self.config.backward_blocked_confirmations
                        ),
                        attempted_heading_deg=attempted_heading,
                    )
                    self.status_callback(
                        f"map step={self._step}: backpedal remained near-zero after "
                        f"the extended retry ({motion.horizontal_distance:.4f}u); "
                        f"rear boundary {rear_edge[0]}->{rear_edge[1]} recorded "
                        f"blocked (confirmations={count}, confirmed={confirmed})."
                    )
                else:
                    self.status_callback(
                        f"map step={self._step}: confirmed near-zero backpedal was "
                        "not written as a wall because absolute heading could not be "
                        "verified."
                    )
                if (
                    blocked_attempts
                    % self.config.escape_wiggle_after_blocked_attempts
                    == 0
                ):
                    self._escape_wiggle(
                        right=(
                            blocked_attempts
                            // self.config.escape_wiggle_after_blocked_attempts
                        )
                        % 2
                        == 1
                    )
                continue

            # Slow but aligned reverse movement is valid progress. Clear any
            # unconfirmed rear evidence from an earlier short pulse and retain
            # the actual measured path as free.
            blocked_attempts = 0
            self._clear_blocked_evidence(rear_edge)
            self._integrate_motion(motion, infer_heading=False)
            successful_steps += 1
            retreat = max(0.0, motion.forward_progress)
            total_retreat += retreat
            self.status_callback(
                f"map step={self._step} action=BACKWARD_ESCAPE native={motion.outcome}/"
                f"{motion.horizontal_distance:.4f}u reverse_progress={retreat:.4f}u "
                f"cumulative={total_retreat:.4f}u."
            )
            if total_retreat >= self.config.escape_target_distance_units:
                break

        if successful_steps == 0:
            escaped = self._radial_escape_probe(reason)
        else:
            escaped = True

        if entry_edge is not None:
            first, second = entry_edge
            self.grid.add_temporary_avoidance(*first, *second)
            self.status_callback(
                f"map step={self._step}: pocket entrance {first}->{second} is now "
                "temporarily avoided. It is not wall evidence and cannot make the "
                "map appear complete."
            )

        self._clear_trap_evidence()
        if escaped:
            self.status_callback(
                f"map step={self._step}: confined-space recovery completed; replanning."
            )
        else:
            self.status_callback(
                f"map step={self._step}: bounded recovery found no clear movement; "
                "the pocket entrance was blacklisted and mapping will replan without crashing."
            )

    def _escape_wiggle(self, *, right: bool) -> None:
        self._wait_for_game_focus()
        degrees = self.config.escape_wiggle_degrees
        current = self.grid.continuous_pose.heading_deg
        if right:
            seconds = self.config.turn_right_90_seconds * degrees / 90.0
            timing = self.controller.turn_right(seconds)
            heading = (current + degrees) % 360.0
            action = "ESCAPE_TURN_RIGHT"
        else:
            seconds = self.config.turn_left_90_seconds * degrees / 90.0
            timing = self.controller.turn_left(seconds)
            heading = (current - degrees) % 360.0
            action = "ESCAPE_TURN_LEFT"
        self.controller.settle(self.config.settle_seconds)
        self._verify_pose_continuity(f"after {action}")
        self.grid.set_heading_degrees(heading)
        self._heading_source = "escape-commanded"
        self._log_turn(
            action=action,
            reason="backward escape was blocked",
            timing=timing,
            note=f"bounded {degrees:.1f}-degree escape wiggle",
        )

    def _radial_escape_probe(self, reason: str) -> bool:
        self.status_callback(
            f"map step={self._step}: reverse path is blocked; scanning bounded "
            "45-degree escape directions using native movement checks."
        )
        for attempt in range(1, 9):
            self.cancellation.raise_if_cancelled()
            self._wait_for_game_focus()
            degrees = 45.0
            timing_turn = self.controller.turn_right(
                self.config.turn_right_90_seconds * degrees / 90.0
            )
            self.controller.settle(self.config.settle_seconds)
            self._verify_pose_continuity("after radial escape turn")
            heading = (self.grid.continuous_pose.heading_deg + degrees) % 360.0
            self.grid.set_heading_degrees(heading)
            self._heading_source = "escape-commanded"
            self._log_turn(
                action="ESCAPE_SCAN_TURN_RIGHT",
                reason=reason,
                timing=timing_turn,
                note="bounded radial escape scan",
            )

            motion, timing = self._measure_forward()
            self._log_motion(
                motion,
                action="ESCAPE_PROBE",
                reason=reason,
                attempt=attempt,
                timing=timing,
                note="bounded radial escape scan",
            )
            if motion.outcome == "teleport":
                raise MapTransitionDetected(
                    before=motion.before,
                    after=motion.after,
                    horizontal_distance=motion.horizontal_distance,
                    vertical_distance=motion.vertical_distance,
                    context="radial escape probe",
                )
            if motion.outcome in {"partial", "moved"}:
                self._integrate_motion(
                    motion,
                    infer_heading=motion.outcome == "moved",
                )
                if motion.outcome == "moved":
                    self.status_callback(
                        f"map step={self._step}: radial escape found an open direction "
                        f"after {attempt} probes ({motion.horizontal_distance:.4f}u)."
                    )
                    return True
        return False

    def _wait_for_game_focus(self) -> None:
        """Pause movement while this client requires foreground DirectInput.

        EVA can respond to posted messages in the background, but movement in
        this client does not.  Waiting here prevents an unfocused client from
        being recorded as a wall.
        """
        if not self.config.pause_when_unfocused:
            return
        keyboard = self.bot.keyboard
        is_foreground = getattr(keyboard, "is_target_foreground", lambda: True)
        if keyboard is None or is_foreground():
            if getattr(self, "_focus_pause_announced", False):
                self.status_callback(
                    "FlyFF regained focus; coordinate mapping is resuming."
                )
                self._focus_pause_announced = False
            return
        if not getattr(self, "_focus_pause_announced", False):
            self.status_callback(
                "FlyFF is not focused. This client ignores background movement "
                "input, so mapping is paused to avoid false blocked cells. "
                "EVA background hotkeys are a separate input path."
            )
            self._focus_pause_announced = True
        while not is_foreground():
            if self.cancellation.wait(self.config.unfocused_poll_seconds):
                self.cancellation.raise_if_cancelled()
        self.status_callback("FlyFF regained focus; coordinate mapping is resuming.")
        self._focus_pause_announced = False
        self.controller.settle(self.config.settle_seconds)

    def _maybe_cast_eva(self) -> None:
        interval = self.config.eva_interval_steps
        if interval <= 0 or self._step <= 0 or self._step % interval != 0:
            return
        self.controller.stop()
        assert self.bot.keyboard is not None
        eva_hotkey = str(self.bot.config.get("eva_hotkey", "F1")).upper()
        eva_vkey = VKEY.get(eva_hotkey)
        if eva_vkey is None or not eva_hotkey.startswith("F"):
            eva_hotkey = "F1"
            eva_vkey = VKEY[eva_hotkey]
        self.bot.keyboard.press_key(eva_vkey, press_time=0.03)
        self.controller.settle(self.config.eva_settle_seconds)
        self._verify_pose_continuity("after EVA cast")
        self.status_callback(
            f"map step={self._step}: cast EVA ({eva_hotkey})."
        )

    def _publish_map(self, *, force: bool = False) -> None:
        if self.frame_callback is None:
            return
        now = datetime.now(timezone.utc).timestamp()
        if not force and now - self._last_publish_at < self.config.map_publish_interval_seconds:
            return
        self._last_publish_at = now
        self.frame_callback(
            self.grid.render_dashboard(
                local_radius_cells=self.config.local_map_radius_cells,
            )
        )

    def _save_state(self) -> None:
        """Best-effort checkpointing that never stops active mapping on a lock.

        Windows may transiently lock JSON, NumPy, or preview files while they
        are inspected by antivirus, indexing, Explorer, an editor, or another
        reader. Each destination is saved independently. A failed checkpoint
        remains in memory and is retried on the next mapper step.
        """
        if self.coordinate_frame is None:
            return

        failures: list[tuple[str, OSError]] = []
        operations = (
            ("persistent coordinate frame", lambda: self.coordinate_frame.save(self.frame_path)),
            (
                "persistent map",
                lambda: self.grid.save(
                    self.map_dir,
                    preview_local_radius_cells=self.config.local_map_radius_cells,
                ),
            ),
            (
                "run snapshot",
                lambda: self.grid.save(
                    self.output_dir,
                    preview_local_radius_cells=self.config.local_map_radius_cells,
                ),
            ),
            (
                "run coordinate frame",
                lambda: self.coordinate_frame.save(self.output_dir / self.FRAME_FILE),
            ),
        )
        for label, operation in operations:
            try:
                operation()
            except OSError as error:
                failures.append((label, error))

        if failures:
            self._save_failure_streak = getattr(self, "_save_failure_streak", 0) + 1
            if self._should_report_save_failure(self._save_failure_streak):
                summary = "; ".join(
                    f"{label}: {type(error).__name__} ({error})"
                    for label, error in failures
                )
                self.status_callback(
                    "Mapper checkpoint was temporarily blocked by Windows; "
                    "mapping continues and the next step will retry. " + summary
                )
            return

        previous_streak = getattr(self, "_save_failure_streak", 0)
        self._save_failure_streak = 0
        if previous_streak:
            self.status_callback(
                "Mapper checkpoint recovered after "
                f"{previous_streak} deferred save attempt(s)."
            )

    @staticmethod
    def _should_report_save_failure(streak: int) -> bool:
        return streak in {1, 5, 20} or streak % 100 == 0

    def _log_turn(
        self,
        *,
        action: str,
        reason: str,
        timing: KeyPressTiming,
        note: str,
    ) -> None:
        pose = self._last_pose
        self.logger.write(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "map_name": self.map_profile.name,
                "step": self._step,
                "action": action,
                "reason": reason,
                "after_native_x": round(pose.x, 6) if pose is not None else "",
                "after_native_y": round(pose.y, 6) if pose is not None else "",
                "after_native_z": round(pose.z, 6) if pose is not None else "",
                "local_x_cells": round(self.grid.continuous_pose.x, 6),
                "local_y_cells": round(self.grid.continuous_pose.y, 6),
                "heading_deg": round(self.grid.continuous_pose.heading_deg, 6),
                "heading_index": self.grid.pose.heading_index,
                "heading_source": self._heading_source,
                "requested_seconds": round(timing.requested_seconds, 6),
                "held_seconds": round(timing.held_seconds, 6),
                "note": note,
            }
        )

    def _log_motion(
        self,
        motion: NativeMotion,
        *,
        action: str,
        reason: str,
        attempt: int,
        timing: KeyPressTiming,
        note: str,
    ) -> None:
        frame = self._require_frame()
        local_x, local_y = frame.to_local_cells(motion.after.x, motion.after.z)
        from_local = frame.to_local_cells(motion.before.x, motion.before.z)
        from_cell = self.grid.world_point_to_grid_cell(*from_local)
        direction = self.grid.DIRECTIONS[self.grid.pose.heading_index]
        to_cell = (from_cell[0] + direction[0], from_cell[1] + direction[1])
        edge_key = (*from_cell, *to_cell)
        count = self._blocked_counts.get(edge_key, 0)
        self.logger.write(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "map_name": self.map_profile.name,
                "step": self._step,
                "action": action,
                "reason": reason,
                "attempt": attempt,
                "before_native_x": round(motion.before.x, 6),
                "before_native_y": round(motion.before.y, 6),
                "before_native_z": round(motion.before.z, 6),
                "after_native_x": round(motion.after.x, 6),
                "after_native_y": round(motion.after.y, 6),
                "after_native_z": round(motion.after.z, 6),
                "delta_native_x": round(motion.delta_x, 6),
                "delta_native_y": round(motion.delta_y, 6),
                "delta_native_z": round(motion.delta_z, 6),
                "horizontal_distance_units": round(motion.horizontal_distance, 6),
                "vertical_distance_units": round(motion.vertical_distance, 6),
                "forward_progress_units": round(motion.forward_progress, 6),
                "lateral_distance_units": round(motion.lateral_distance, 6),
                "forward_alignment": round(motion.forward_alignment, 6),
                "motion_outcome": motion.outcome,
                "local_x_cells": round(local_x, 6),
                "local_y_cells": round(local_y, 6),
                "heading_deg": round(self.grid.continuous_pose.heading_deg, 6),
                "heading_index": self.grid.pose.heading_index,
                "heading_source": self._heading_source,
                "from_cell_x": from_cell[0],
                "from_cell_y": from_cell[1],
                "to_cell_x": to_cell[0],
                "to_cell_y": to_cell[1],
                "boundary_confirmations": count,
                "boundary_confirmed": count >= self.config.blocked_confirmations,
                "requested_seconds": round(timing.requested_seconds, 6),
                "held_seconds": round(timing.held_seconds, 6),
                "note": note,
            }
        )

    def _require_frame(self) -> CoordinateFrame:
        if self.coordinate_frame is None:
            raise RuntimeError("Coordinate frame has not been initialized")
        return self.coordinate_frame

    @staticmethod
    def heading_from_delta(delta_x: float, delta_z: float) -> float:
        if not math.isfinite(delta_x) or not math.isfinite(delta_z):
            raise ValueError("Movement delta must be finite")
        if math.hypot(delta_x, delta_z) <= 0.0:
            raise ValueError("Movement delta must be non-zero")
        # OccupancyGrid convention: 0° = +Z/north, 90° = +X/east.
        return math.degrees(math.atan2(delta_x, delta_z)) % 360.0


Mapper = CoordinateMapper
