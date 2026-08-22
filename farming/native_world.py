from __future__ import annotations

# pyright: reportImplicitRelativeImport=false
from dataclasses import dataclass
from math import inf
from typing import Mapping, Protocol

from position.native_process_service import NativePointerSnapshot
from position.NativeFlyffMonsterProvider import (
    ActorCacheOutcome,
    ActorCacheRefreshResult,
    CachedActorReadResult,
    NativeActor,
)
from position.PositionProvider import PlayerPose

from .map_context import FarmingMapContext
from .observation import ActorObservation


class PointerSnapshotReader(Protocol):
    def read_pointer_snapshot(self) -> NativePointerSnapshot: ...


class SnapshotPoseReader(Protocol):
    def read_pose(
        self,
        *,
        pointer_snapshot: NativePointerSnapshot | None = None,
    ) -> PlayerPose: ...


class CachedActorReader(Protocol):
    def refresh_slot_cache(
        self,
        pointer_snapshot: NativePointerSnapshot,
        *,
        cancellation: object | None = None,
        deadline: float | None = None,
        force: bool = False,
    ) -> ActorCacheRefreshResult: ...

    def read_cached_active_actors(
        self,
        pointer_snapshot: NativePointerSnapshot,
        player_pose: PlayerPose,
        *,
        allowed_species_ids: set[int] | None = None,
        vision_radius_native: float | None = None,
    ) -> CachedActorReadResult: ...

    def read_actor_hp_states(
        self,
        candidates: tuple[tuple[int, int], ...],
    ) -> Mapping[tuple[int, int], int]: ...


class NativeWorldUnavailable(RuntimeError):
    def __init__(self, outcome: ActorCacheOutcome, message: str) -> None:
        super().__init__(message)
        self.outcome = outcome


@dataclass(frozen=True, slots=True)
class NativeWorldFrame:
    pointer_snapshot: NativePointerSnapshot
    player_pose: PlayerPose
    actors: tuple[NativeActor, ...]
    tracked_actors: tuple[NativeActor, ...] = ()

    @property
    def world_base(self) -> int:
        return self.pointer_snapshot.world_base

    @property
    def generation(self) -> int:
        return self.pointer_snapshot.generation


class NativeWorldReader:
    """Acquire position and cached actors from exactly one pointer snapshot."""

    def __init__(
        self,
        service: PointerSnapshotReader,
        position_reader: SnapshotPoseReader,
        actor_reader: CachedActorReader,
        *,
        allowed_species_ids: set[int] | None,
        vision_radius_native: float,
    ) -> None:
        if vision_radius_native <= 0.0:
            raise ValueError("vision_radius_native must be positive")
        self.service = service
        self.position_reader = position_reader
        self.actor_reader = actor_reader
        self.allowed_species_ids = (
            None
            if allowed_species_ids is None
            else {int(value) for value in allowed_species_ids}
        )
        self.vision_radius_native = float(vision_radius_native)

    def refresh_actor_cache(
        self,
        *,
        cancellation: object | None = None,
        deadline: float | None = None,
        force: bool = False,
    ) -> ActorCacheRefreshResult:
        snapshot = self.service.read_pointer_snapshot()
        return self.actor_reader.refresh_slot_cache(
            snapshot,
            cancellation=cancellation,
            deadline=deadline,
            force=force,
        )

    def read_actor_hp_states(
        self,
        candidates: tuple[tuple[int, int], ...],
    ) -> Mapping[tuple[int, int], int]:
        """Read post-cast HP directly without rebuilding a world frame."""

        return self.actor_reader.read_actor_hp_states(candidates)

    def read_frame(self) -> NativeWorldFrame:
        snapshot = self.service.read_pointer_snapshot()
        pose = self.position_reader.read_pose(pointer_snapshot=snapshot)
        actor_result = self.actor_reader.read_cached_active_actors(
            snapshot,
            pose,
            allowed_species_ids=self.allowed_species_ids,
            vision_radius_native=self.vision_radius_native,
        )
        if not actor_result.ready:
            raise NativeWorldUnavailable(
                actor_result.outcome,
                actor_result.message or "Cached native actors are unavailable",
            )
        return NativeWorldFrame(
            pointer_snapshot=snapshot,
            player_pose=pose,
            actors=actor_result.actors,
            tracked_actors=(
                actor_result.tracked_actors
                if actor_result.tracked_actors
                else actor_result.actors
            ),
        )


def build_actor_observations(
    frame: NativeWorldFrame,
    map_context: FarmingMapContext,
    *,
    maximum_geodesic_distance_cells: float | None = None,
) -> tuple[ActorObservation, ...]:
    """Build both legacy-layout and direct-native actor coordinate frames."""

    player_layout = map_context.native_to_layout_cells(
        frame.player_pose.x,
        frame.player_pose.z,
    )
    player_cell = map_context.native_to_layout_cell(
        frame.player_pose.x,
        frame.player_pose.z,
    )
    actor_layouts = [
        map_context.native_to_layout_cells(actor.x, actor.z) for actor in frame.actors
    ]
    actor_cells = [
        map_context.native_to_layout_cell(actor.x, actor.z) for actor in frame.actors
    ]

    distances = [inf] * len(frame.actors)
    if player_cell is not None:
        valid = [
            (index, cell) for index, cell in enumerate(actor_cells) if cell is not None
        ]
        if valid:
            queried = map_context.features.geodesic_distances(
                player_cell,
                (cell for _index, cell in valid),
                maximum_distance_cells=maximum_geodesic_distance_cells,
            )
            for (index, _cell), distance in zip(valid, queried, strict=True):
                distances[index] = distance

    units = map_context.native_units_per_cell
    return tuple(
        ActorObservation(
            actor_id=actor.base_address,
            legacy_dx_cells=layout[0] - player_layout[0],
            legacy_dy_cells=layout[1] - player_layout[1],
            direct_dx_cells=(actor.x - frame.player_pose.x) / units,
            direct_dz_cells=(actor.z - frame.player_pose.z) / units,
            geodesic_cells=distances[index],
            direct_path=map_context.features.direct_path_state(
                player_cell,
                actor_cells[index],
            ),
            alive=True,
        )
        for index, (actor, layout) in enumerate(
            zip(frame.actors, actor_layouts, strict=True)
        )
    )
