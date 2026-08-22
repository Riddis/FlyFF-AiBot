from __future__ import annotations

# pyright: reportImplicitRelativeImport=false
from pathlib import Path

import numpy as np
from farming.config import FarmingRuntimeConfig
from farming.map_context import FarmingMapContext
from farming.map_features import FarmingMapFeatures
from farming.trainer import build_live_farming_runtime
from mapper.CoordinateFrame import CoordinateFrame
from position.native_process_service import NativePointerSnapshot
from position.NativeFlyffMonsterProvider import (
    ActorCacheOutcome,
    ActorCacheRefreshResult,
    CachedActorReadResult,
)
from position.PositionProvider import PlayerPose
from runtime.worker_manager import CancellationToken


class FakeKeyboard:
    def __init__(self) -> None:
        self.trace: list[tuple[str, int]] = []

    def key_down(self, key: int) -> None:
        self.trace.append(("down", key))

    def key_up(self, key: int) -> None:
        self.trace.append(("up", key))

    def is_target_foreground(self) -> bool:
        return True

    def focus_target_window(self) -> bool:
        return True


def _snapshot() -> NativePointerSnapshot:
    return NativePointerSnapshot(
        player_pointer_address=1,
        world_pointer_address=2,
        player_base=3,
        world_base=4,
        generation=5,
        captured_at=6.0,
    )


class FakeService:
    def read_pointer_snapshot(self) -> NativePointerSnapshot:
        return _snapshot()


class FakePosition:
    def read_pose(
        self,
        *,
        pointer_snapshot: NativePointerSnapshot | None = None,
    ) -> PlayerPose:
        assert pointer_snapshot == _snapshot()
        return PlayerPose(0.0, 0.0, 0.0, 90.0, 6.0)


class FakeActors:
    def refresh_slot_cache(
        self,
        pointer_snapshot: NativePointerSnapshot,
        *,
        cancellation: object | None = None,
        deadline: float | None = None,
        force: bool = False,
    ) -> ActorCacheRefreshResult:
        del cancellation, force
        assert pointer_snapshot == _snapshot()
        assert deadline is not None
        return ActorCacheRefreshResult(
            ActorCacheOutcome.REFRESHED,
            world_base=4,
            generation=5,
            slot_count=12,
        )

    def read_cached_active_actors(
        self,
        pointer_snapshot: NativePointerSnapshot,
        player_pose: PlayerPose,
        *,
        allowed_species_ids: set[int] | None = None,
        vision_radius_native: float | None = None,
    ) -> CachedActorReadResult:
        del player_pose
        assert pointer_snapshot == _snapshot()
        assert allowed_species_ids == {101}
        assert vision_radius_native == 50.0
        return CachedActorReadResult(
            ActorCacheOutcome.READY,
            world_base=4,
            generation=5,
        )


class FakeBot:
    def __init__(self) -> None:
        self.config: dict[str, object] = {
            "selected_map_name": "Tower AoE",
            "selected_mobs": [{"species_id": 101}],
        }
        self.keyboard = FakeKeyboard()
        self.position_provider = FakePosition()
        self.monster_provider = FakeActors()
        self.native_process_service = FakeService()
        self.rl_enabled = False
        self.is_ready = True
        self.start_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        self.rl_enabled = True

    def read_kill_count(self) -> int | None:
        return 0


def _map_context(
    _name: str,
    **_kwargs: object,
) -> FarmingMapContext:
    traversable = np.ones((9, 9), dtype=np.bool_)
    forbidden = np.zeros_like(traversable)
    forbidden[8, 8] = True
    return FarmingMapContext(
        map_name="Tower AoE",
        map_directory=Path("."),
        coordinate_frame=CoordinateFrame(native_units_per_cell=1.0),
        grid_origin=4,
        source_bounds=(0, 0, 8, 8),
        features=FarmingMapFeatures(
            traversable=traversable,
            forbidden=forbidden,
            safe_traversable=traversable & ~forbidden,
            teleport_buffer_radius_cells=2.0,
        ),
        content_hash="PREFLIGHT",
    )


def test_runtime_preflight_reads_map_pointer_and_cache_before_any_input() -> None:
    bot = FakeBot()
    runtime = build_live_farming_runtime(
        bot,
        FarmingRuntimeConfig(),
        CancellationToken(),
        map_context_loader=_map_context,
    )

    assert bot.start_calls == 0
    assert bot.keyboard.trace == []
    assert runtime.preflight.actor_cache_outcome == "refreshed"
    assert runtime.preflight.actor_slots == 12
    assert runtime.preflight.pointer_generation == 5
    assert runtime.preflight.initial_map_cell == (4, 4)
    runtime.close()
