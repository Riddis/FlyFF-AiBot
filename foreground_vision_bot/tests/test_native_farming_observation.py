from __future__ import annotations

from pathlib import Path

import numpy as np

from libs.NativeFarmingObservation import (
    NativeFarmingObservationBuilder,
    NativeFarmingObservationConfig,
)
from libs.NativeMapContext import NativeMapContext
from mapper.CoordinateFrame import CoordinateFrame
from mapper.rl.ProceduralDungeon import DungeonLayout
from mapper.rl.TravelCost import build_safe_travel_cost_field
from position import NativeActor, PlayerPose


def _context() -> NativeMapContext:
    traversable = np.ones((31, 31), dtype=np.bool_)
    layout = DungeonLayout(traversable=traversable, spawn=(15, 15))
    return NativeMapContext(
        map_name="test",
        map_directory=Path("."),
        coordinate_frame=CoordinateFrame(
            origin_native_x=0.0,
            origin_native_z=0.0,
            native_units_per_cell=1.0,
        ),
        grid_origin=0,
        source_bounds=(0, -30, 30, 0),
        layout=layout,
        safe_traversable=traversable.copy(),
    )


def _actor(base: int, x: float, z: float, species: int = 944) -> NativeActor:
    return NativeActor(
        base_address=base,
        species_id=species,
        hp=100,
        x=x,
        y=0.0,
        z=z,
        distance_native=float(np.hypot(x, z)),
        active_species_id=species,
    )


def test_native_farming_observation_keeps_raw_targets_and_fixed_shape() -> None:
    context = _context()
    config = NativeFarmingObservationConfig(
        max_targets=4,
        vision_radius_cells=20.0,
        eva_radius_cells=3.0,
    )
    builder = NativeFarmingObservationBuilder(context, config)
    pose = PlayerPose(15.0, 0.0, 15.0, 0.0, 1.0)
    origin = context.nearest_safe_cell(context.native_to_layout_cells(15.0, 15.0))
    assert origin is not None
    travel = build_safe_travel_cost_field(
        context.layout,
        origin,
        obstacle_buffer_radius_cells=0,
        teleport_buffer_radius_cells=0,
    )
    actors = [
        _actor(0x1000, 16.0, 15.0),
        _actor(0x2000, 17.0, 15.0),
        _actor(0x3000, 25.0, 15.0),
    ]

    snapshot = builder.build(
        player_pose=pose,
        actors=actors,
        travel_cost=travel,
        eva_cooldown_fraction=0.5,
    )

    assert snapshot.vector.shape == (builder.observation_size,)
    assert snapshot.vector.dtype == np.float32
    assert len(snapshot.targets) == 3
    assert {target.actor.base_address for target in snapshot.targets} == {
        0x1000,
        0x2000,
        0x3000,
    }
    assert snapshot.player_eva_count == 2
    assert max(target.nearby_count for target in snapshot.targets) == 2


def test_native_map_context_round_trips_tower_spawn() -> None:
    context = NativeMapContext.load("Tower AoE")
    assert context.native_to_world_cells(253.0, 86.0) == (0.0, 0.0)
    layout = context.native_to_layout_cells(253.0, 86.0)
    assert context.nearest_safe_cell(layout) is not None
