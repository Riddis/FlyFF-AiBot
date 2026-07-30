from __future__ import annotations

import math
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Protocol

from position import PlayerPose
from worker_manager import CancellationToken, WorkerCancelled

from .CoordinateFrame import CoordinateFrame
from .CoordinateMapLogger import CoordinateMapLogger
from .CoordinateMapper import MapperConfig, load_mapper_config
from .MapCatalog import MapCatalog, MapProfile
from .OccupancyGrid import FORBIDDEN, FREE, OccupancyGrid

StatusCallback = Callable[[str], None]
FrameCallback = Callable[[object], None]


class ManualDriveBot(Protocol):
    def get_player_pose(self) -> PlayerPose | None: ...


class ManualDriveTransition(RuntimeError):
    def __init__(
        self,
        *,
        before: PlayerPose,
        after: PlayerPose,
        horizontal_distance: float,
        vertical_distance: float,
    ) -> None:
        self.before = before
        self.after = after
        self.horizontal_distance = float(horizontal_distance)
        self.vertical_distance = float(vertical_distance)
        super().__init__(
            "impossible native-coordinate jump while manual mapping: "
            f"horizontal={self.horizontal_distance:.3f}u, "
            f"vertical={self.vertical_distance:.3f}u"
        )


class ManualDriveTeleportAreaEntered(RuntimeError):
    def __init__(self, cell: tuple[int, int]) -> None:
        self.cell = (int(cell[0]), int(cell[1]))
        super().__init__(f"manual route entered teleport cell {self.cell}")


class ManualDriveMapper:
    """Track user-controlled movement and write traversed cells as FREE.

    This mode never sends keyboard input. Native coordinates are sampled at a
    short interval and each segment is rasterised through all crossed occupancy
    cells. Marked teleport cells and impossible coordinate jumps terminate the
    run before destination coordinates are integrated.
    """

    VERSION = "0.6.0-manual-drive-native-trace"
    FRAME_FILE = "coordinate_frame.json"
    SAMPLE_SECONDS = 0.05
    CHECKPOINT_SECONDS = 1.0
    STATUS_SECONDS = 2.0
    HEADING_MIN_NATIVE_DISTANCE = 0.05

    def __init__(
        self,
        bot: ManualDriveBot,
        *,
        status_callback: StatusCallback | None = None,
        frame_callback: FrameCallback | None = None,
        cancellation: CancellationToken | None = None,
        map_name: str | None = None,
        config: MapperConfig | None = None,
    ) -> None:
        initial = bot.get_player_pose()
        if initial is None:
            raise RuntimeError(
                "Native player coordinates are unavailable. Attach Neuz.exe and "
                "verify position/native_position.json first."
            )

        self.bot = bot
        self.status_callback = status_callback or print
        self.frame_callback = frame_callback
        self.cancellation = cancellation or CancellationToken()
        self.config = config or load_mapper_config()

        self.map_catalog = MapCatalog()
        self.map_profile: MapProfile = self.map_catalog.get(map_name)
        self.map_dir = self.map_catalog.map_directory(self.map_profile.name)
        self.frame_path = self.map_dir / self.FRAME_FILE

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_manual")
        self.output_dir = self.map_catalog.run_directory(self.map_profile.name) / run_id
        self.logger = CoordinateMapLogger(self.output_dir / "coordinate_steps.csv")

        self.grid = OccupancyGrid(size=self.config.grid_size)
        self.coordinate_frame: CoordinateFrame | None = None
        self._previous_pose = initial
        self._previous_local: tuple[float, float] | None = None
        self._step = 0
        self._new_free_cells = 0
        self._distance_native = 0.0
        self._last_publish_at = 0.0
        self._last_checkpoint_at = 0.0
        self._last_status_at = 0.0
        self._save_failure_streak = 0

    def stop(self) -> None:
        self.cancellation.cancel()

    def run(self) -> Path:
        primary_error: BaseException | None = None
        try:
            return self._run()
        except WorkerCancelled:
            self.grid.metadata.termination_reason = "Manual mapping stopped by user."
            self.status_callback(
                f"Manual mapping stopped. Run saved to {self.output_dir}."
            )
            raise
        except ManualDriveTransition as transition:
            self._record_transition(transition)
            return self.output_dir
        except ManualDriveTeleportAreaEntered as entered:
            message = (
                f"Manual mapping reached marked teleport area at cell {entered.cell}. "
                "Tracking stopped before that cell was changed."
            )
            self.grid.metadata.termination_reason = message
            self.status_callback(message)
            return self.output_dir
        except BaseException as error:
            primary_error = error
            self.grid.metadata.termination_reason = f"{type(error).__name__}: {error}"
            raise
        finally:
            cleanup_error: Exception | None = None
            for label, action in (
                ("save manual map", self._save_state),
                ("close manual coordinate log", self.logger.close),
            ):
                try:
                    action()
                except Exception as error:  # noqa: BLE001 - preserve original failure.
                    if primary_error is not None:
                        primary_error.add_note(f"Could not {label}: {error}")
                    elif cleanup_error is None:
                        cleanup_error = error
            if primary_error is None and cleanup_error is not None:
                raise cleanup_error

    def _run(self) -> Path:
        current = self._previous_pose
        self._load_or_create_map(current)
        local = self._local_from_pose(current)
        start_cell = self.grid.world_point_to_grid_cell(*local)
        if not self.grid.in_bounds(*start_cell):
            raise RuntimeError(
                f"Current position maps outside '{self.map_profile.name}' at {start_cell}."
            )
        if self.grid.value(*start_cell) == FORBIDDEN:
            raise ManualDriveTeleportAreaEntered(start_cell)

        self.grid.set_continuous_pose(
            local[0],
            local[1],
            self.grid.continuous_pose.heading_deg,
            mark_cell_free=True,
        )
        self.grid.set_pose_reliability(
            position_known=True,
            heading_known=True,
            note=(
                "Manual-drive position is native-coordinate authoritative; heading "
                "is refined from each meaningful user-controlled movement segment."
            ),
        )
        self.grid.metadata.map_name = self.map_profile.name
        self.grid.metadata.run_count += 1
        self.grid.metadata.termination_reason = None
        self._previous_local = local

        self.status_callback(
            f"Manual mapping {self.VERSION} started for '{self.map_profile.name}'. "
            "You control the character; the bot sends no movement, turn, or EVA keys. "
            "Every native-coordinate path segment is marked explored/free. Use Edit "
            "Map Cells afterward to paint blockers or red teleport cells."
        )
        self._publish(force=True)
        self._save_state()

        while not self.cancellation.cancelled:
            if self.cancellation.wait(self.SAMPLE_SECONDS):
                self.cancellation.raise_if_cancelled()
            current = self._read_pose()
            self._process_pose(self._previous_pose, current)
            self._previous_pose = current
            self._step += 1

            now = monotonic()
            if now - self._last_publish_at >= self.config.map_publish_interval_seconds:
                self._publish(force=True)
            if now - self._last_checkpoint_at >= self.CHECKPOINT_SECONDS:
                self._save_state()
            if now - self._last_status_at >= self.STATUS_SECONDS:
                self.status_callback(
                    f"Manual mapping active: {self._new_free_cells} newly explored cells, "
                    f"{self._distance_native:.1f} native units traced; player cell="
                    f"({self.grid.pose.x}, {self.grid.pose.y})."
                )
                self._last_status_at = now

        self.grid.metadata.termination_reason = "Manual mapping stopped by user."
        self.status_callback(
            f"Manual mapping stopped. Run saved to {self.output_dir}."
        )
        return self.output_dir

    def _load_or_create_map(self, current: PlayerPose) -> None:
        if self.frame_path.is_file():
            self.coordinate_frame = CoordinateFrame.load(self.frame_path)
            self.grid, warning = OccupancyGrid.load(self.map_dir)
            if warning is not None:
                self.status_callback(warning)
            self.status_callback(
                f"Loaded coordinate map at native origin "
                f"({self.coordinate_frame.origin_native_x:.3f}, "
                f"{self.coordinate_frame.origin_native_z:.3f}); "
                f"{self.grid.known_cell_count()} known cells."
            )
            return

        archived = self._archive_incompatible_map_if_present()
        self.coordinate_frame = CoordinateFrame(
            origin_native_x=float(current.x),
            origin_native_z=float(current.z),
            native_units_per_cell=self.config.native_units_per_cell,
        )
        self.grid = OccupancyGrid(size=self.config.grid_size)
        self.grid.metadata.map_name = self.map_profile.name
        if archived is None:
            self.status_callback(
                "Created a new coordinate map with the current manual-drive position "
                "as its native origin."
            )
        else:
            self.status_callback(
                f"Archived incompatible map files to {archived.name} and created a "
                "new native-coordinate map."
            )

    def _archive_incompatible_map_if_present(self) -> Path | None:
        names = ("map.json", "occupancy.npy", "visits.npy", "map_preview.png")
        existing = [self.map_dir / name for name in names if (self.map_dir / name).exists()]
        if not existing:
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive = self.map_dir / f"legacy_map_before_manual_{stamp}"
        archive.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.move(str(path), str(archive / path.name))
        return archive

    def _read_pose(self) -> PlayerPose:
        pose = self.bot.get_player_pose()
        if pose is None:
            raise RuntimeError("Native player position became unavailable")
        return pose

    def _local_from_pose(self, pose: PlayerPose) -> tuple[float, float]:
        if self.coordinate_frame is None:
            raise RuntimeError("Coordinate frame is not initialized")
        return self.coordinate_frame.to_local_cells(pose.x, pose.z)

    def _process_pose(self, before: PlayerPose, after: PlayerPose) -> None:
        horizontal = math.hypot(after.x - before.x, after.z - before.z)
        vertical = abs(after.y - before.y)
        if (
            horizontal >= self.config.teleport_distance_units
            or vertical >= self.config.teleport_vertical_distance_units
        ):
            raise ManualDriveTransition(
                before=before,
                after=after,
                horizontal_distance=horizontal,
                vertical_distance=vertical,
            )

        start_local = self._previous_local or self._local_from_pose(before)
        end_local = self._local_from_pose(after)
        end_cell = self.grid.world_point_to_grid_cell(*end_local)
        if not self.grid.in_bounds(*end_cell):
            raise RuntimeError(
                f"Manual route left the '{self.map_profile.name}' occupancy grid at "
                f"local=({end_local[0]:.1f}, {end_local[1]:.1f})."
            )

        traversed = self.grid.rasterize_segment(
            start_local[0], start_local[1], end_local[0], end_local[1]
        )
        for cell in traversed:
            if self.grid.value(*cell) == FORBIDDEN:
                raise ManualDriveTeleportAreaEntered(cell)

        heading = self.grid.continuous_pose.heading_deg
        if horizontal >= self.HEADING_MIN_NATIVE_DISTANCE:
            heading = math.degrees(math.atan2(after.x - before.x, after.z - before.z)) % 360.0

        newly_free = 0
        for cell in traversed:
            if self.grid.value(*cell) != FREE:
                newly_free += 1
            self.grid.mark_free(*cell)
        self.grid.set_continuous_pose(
            end_local[0], end_local[1], heading, mark_cell_free=True
        )
        self._previous_local = end_local
        self._distance_native += horizontal
        self._new_free_cells += newly_free

        if horizontal > 0.001 or newly_free:
            self._log_segment(
                before=before,
                after=after,
                horizontal=horizontal,
                vertical=vertical,
                traversed=traversed,
                newly_free=newly_free,
            )

    def _record_transition(self, transition: ManualDriveTransition) -> None:
        source_local = self._local_from_pose(transition.before)
        source = self.grid.world_point_to_grid_cell(*source_local)
        attempted = source
        reason = (
            "manual-drive impossible coordinate jump; destination native=("
            f"{transition.after.x:.3f}, {transition.after.y:.3f}, "
            f"{transition.after.z:.3f})"
        )
        self.grid.add_suspected_transition(
            from_x=source[0],
            from_y=source[1],
            attempted_x=attempted[0],
            attempted_y=attempted[1],
            heading_deg=self.grid.continuous_pose.heading_deg,
            reason=reason,
        )
        message = (
            "Teleport/map transition detected during manual mapping from an impossible "
            f"native-coordinate jump ({transition.horizontal_distance:.1f} horizontal, "
            f"{transition.vertical_distance:.1f} vertical units). The destination was "
            "not written into this map."
        )
        self.grid.metadata.termination_reason = message
        self.status_callback(message)

    def _log_segment(
        self,
        *,
        before: PlayerPose,
        after: PlayerPose,
        horizontal: float,
        vertical: float,
        traversed: tuple[tuple[int, int], ...],
        newly_free: int,
    ) -> None:
        end_cell = traversed[-1]
        start_cell = traversed[0]
        self.logger.write(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "map_name": self.map_profile.name,
                "step": self._step,
                "action": "MANUAL_DRIVE",
                "reason": "user-controlled native-coordinate trace",
                "attempt": 1,
                "before_native_x": round(before.x, 6),
                "before_native_y": round(before.y, 6),
                "before_native_z": round(before.z, 6),
                "after_native_x": round(after.x, 6),
                "after_native_y": round(after.y, 6),
                "after_native_z": round(after.z, 6),
                "delta_native_x": round(after.x - before.x, 6),
                "delta_native_y": round(after.y - before.y, 6),
                "delta_native_z": round(after.z - before.z, 6),
                "horizontal_distance_units": round(horizontal, 6),
                "vertical_distance_units": round(vertical, 6),
                "motion_outcome": "manual_trace",
                "local_x_cells": round(self.grid.continuous_pose.x, 6),
                "local_y_cells": round(self.grid.continuous_pose.y, 6),
                "heading_deg": round(self.grid.continuous_pose.heading_deg, 3),
                "heading_index": self.grid.pose.heading_index,
                "heading_source": "native_manual_displacement",
                "from_cell_x": start_cell[0],
                "from_cell_y": start_cell[1],
                "to_cell_x": end_cell[0],
                "to_cell_y": end_cell[1],
                "note": f"traversed_cells={len(traversed)}; newly_free={newly_free}",
            }
        )

    def _publish(self, *, force: bool = False) -> None:
        if self.frame_callback is None:
            return
        now = monotonic()
        if not force and now - self._last_publish_at < self.config.map_publish_interval_seconds:
            return
        self._last_publish_at = now
        self.frame_callback(
            self.grid.render_dashboard(
                local_radius_cells=self.config.local_map_radius_cells
            )
        )

    def _save_state(self) -> None:
        if self.coordinate_frame is None:
            return
        failures: list[tuple[str, OSError]] = []
        operations = (
            ("persistent frame", lambda: self.coordinate_frame.save(self.frame_path)),
            (
                "persistent map",
                lambda: self.grid.save(
                    self.map_dir,
                    preview_local_radius_cells=self.config.local_map_radius_cells,
                ),
            ),
            (
                "manual run snapshot",
                lambda: self.grid.save(
                    self.output_dir,
                    preview_local_radius_cells=self.config.local_map_radius_cells,
                ),
            ),
            (
                "manual run frame",
                lambda: self.coordinate_frame.save(self.output_dir / self.FRAME_FILE),
            ),
        )
        for label, operation in operations:
            try:
                operation()
            except OSError as error:
                failures.append((label, error))

        self._last_checkpoint_at = monotonic()
        if failures:
            self._save_failure_streak += 1
            if self._save_failure_streak in {1, 5, 20} or self._save_failure_streak % 100 == 0:
                details = "; ".join(
                    f"{label}: {type(error).__name__} ({error})"
                    for label, error in failures
                )
                self.status_callback(
                    "Manual-map checkpoint was temporarily blocked by Windows; "
                    "tracking continues and will retry. " + details
                )
            return
        if self._save_failure_streak:
            self.status_callback(
                "Manual-map checkpoint recovered after "
                f"{self._save_failure_streak} deferred save attempt(s)."
            )
        self._save_failure_streak = 0
