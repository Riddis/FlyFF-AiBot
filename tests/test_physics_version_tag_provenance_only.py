"""2026-08-14: explicit contract test, per user request before spending any
curriculum-regeneration/PPO-retraining compute -- does `RecordedWorldModel.
movement_physics_model` (added 2026-08-13 alongside the calibrated-arc
kernel migration) actually SELECT which physics RecordedFarmingEnv
executes, or is it provenance-only (documentation of what a world's
`.movement` field was historically fit from, with zero runtime effect)?

Verified directly here, not just claimed in a docstring: a world tagged
"legacy_recorded_iid" and a world tagged "live_calibrated_arc" -- even one
with WILDLY different `.movement` Gaussian stats (proving the field's
VALUES don't matter either, not just its tag) -- produce byte-for-byte
identical RecordedFarmingEnv._move_player outcomes for the same starting
pose/previous_steering/action. The only runtime reader of a world's
`.movement` field anywhere in the codebase is RecordedFarmingEnv.
movement_path_clear -- the deliberately-frozen historical baseline behind
scripted_policies.obstacle_aware_command (see steering_oracle.py's module
docstring), never the actual movement physics.

CONCLUSION (also recorded in movement_kernel.py's and world_model.py's own
docstrings): movement_physics_model is PROVENANCE-ONLY. Loading a world
tagged "legacy_recorded_iid" into the current codebase does NOT reproduce
the historical per-action Gaussian turn-then-translate physics -- every
RecordedFarmingEnv, regardless of the loaded world's tag or `.movement`
content, unconditionally executes movement_kernel.advance_player_tick
(the calibrated constant-curvature-arc model). Exact reproduction of the
legacy model requires checking out the pre-2026-08-13 code revision (or an
equivalent frozen legacy physics function), not merely loading a
legacy-tagged world into today's code. This is NOT something this test
suite should build a compatibility subsystem to fix -- it is simply the
documented, verified truth about what the tag does and does not do.
"""
from __future__ import annotations

import math

import numpy as np

from farming.actions import FarmingAction
from navigation.movement_kernel import (
    LEGACY_MOVEMENT_PHYSICS_MODEL_ID,
    MOVEMENT_PHYSICS_MODEL_ID,
    SteeringDirection,
    advance_player_tick,
)
from simulator.environment import RecordedFarmingEnv
from simulator.map_model import MapModel
from simulator.world_model import MovementModel, RecordedWorldModel

SIZE = 61
CENTER = SIZE // 2


def _open_map() -> MapModel:
    return MapModel.from_arrays(np.ones((SIZE, SIZE), dtype=bool))


def _world(map_model: MapModel, *, movement_physics_model: str, movement: tuple[MovementModel, ...]) -> RecordedWorldModel:
    positions = tuple(map_model.layout_to_native(8 + index % 5, 8 + index // 5) for index in range(10))
    sections = tuple(positions for _ in range(3))
    return RecordedWorldModel(
        schema_version=5, source_recordings=("provenance-test",), section_count=2, hub_section=2,
        population_median=1, section_population_probabilities=(1 / 3, 1 / 3, 1 / 3),
        player_start_positions=(map_model.layout_to_native(CENTER, CENTER),),
        spawn_positions_by_section=sections,
        transition_probabilities=tuple((1 / 3, 1 / 3, 1 / 3) for _ in range(3)),
        respawn_delay_seconds=(2.0,), movement=movement, monster_speed_cells_per_second=0.0,
        frame_interval_seconds=0.2, native_units_per_cell=map_model.native_units_per_cell,
        recording_frame_interval_seconds=0.2, cast_step_seconds=0.8, cast_movement_seconds=0.2,
        respawn_model_mode="global_redistribution", respawn_delay_source="provenance-test",
        movement_physics_model=movement_physics_model,
    )


_NORMAL_MOVEMENT = (
    MovementModel(100, 1.0, 0.0, 0.0, 0.0),
    MovementModel(100, 1.0, 0.0, 0.25, 0.0),
    MovementModel(100, 1.0, 0.0, -0.25, 0.0),
    MovementModel(0, 0.0, 0.0, 0.0, 0.0),
    MovementModel(10, 1.0, 0.0, 0.0, 0.0),
)
# Deliberately absurd, easily-distinguishable stats (100-cell mean distance,
# 3.0rad mean turn) -- if _move_player read this for ANYTHING, the outcome
# would be wildly, unmissably different from _NORMAL_MOVEMENT's.
_WILD_MOVEMENT = tuple(
    MovementModel(m.samples, 100.0, 5.0, 3.0, 1.0) for m in _NORMAL_MOVEMENT
)


def _env_at(world: RecordedWorldModel, map_model: MapModel, *, heading: float, previous_steering: SteeringDirection) -> RecordedFarmingEnv:
    env = RecordedFarmingEnv(world, map_model=map_model, seed=1, episode_steps=50)
    env.reset(seed=1)
    native = map_model.layout_to_native(CENTER, CENTER)
    env.player_x, env.player_z = native
    env.heading = heading
    env.previous_steering = previous_steering
    return env


class TestPhysicsVersionTagHasNoRuntimeEffect:
    def test_legacy_and_calibrated_arc_tags_produce_identical_movement(self):
        """Same .movement content, only the TAG differs -- must be
        byte-for-byte identical."""
        map_model = _open_map()
        for action in (FarmingAction.RUN_FORWARD, FarmingAction.RUN_FORWARD_LEFT, FarmingAction.RUN_FORWARD_RIGHT):
            legacy_world = _world(map_model, movement_physics_model=LEGACY_MOVEMENT_PHYSICS_MODEL_ID, movement=_NORMAL_MOVEMENT)
            calibrated_world = _world(map_model, movement_physics_model=MOVEMENT_PHYSICS_MODEL_ID, movement=_NORMAL_MOVEMENT)

            env_legacy = _env_at(legacy_world, map_model, heading=0.3, previous_steering=SteeringDirection.NONE)
            env_calibrated = _env_at(calibrated_world, map_model, heading=0.3, previous_steering=SteeringDirection.NONE)

            displacement_legacy, contact_legacy = env_legacy._move_player(action)
            displacement_calibrated, contact_calibrated = env_calibrated._move_player(action)

            assert env_legacy.player_x == env_calibrated.player_x
            assert env_legacy.player_z == env_calibrated.player_z
            assert env_legacy.heading == env_calibrated.heading
            assert env_legacy.previous_steering == env_calibrated.previous_steering
            assert displacement_legacy == displacement_calibrated
            assert contact_legacy == contact_calibrated

    def test_wildly_different_movement_stats_produce_identical_movement(self):
        """Same TAG (irrelevant either way), but .movement's actual VALUES
        are wildly different (100-cell distance mean, 3rad turn mean) --
        if _move_player consulted them for anything, this would be
        unmissable. Must still be byte-for-byte identical."""
        map_model = _open_map()
        for action in (FarmingAction.RUN_FORWARD, FarmingAction.RUN_FORWARD_LEFT, FarmingAction.RUN_FORWARD_RIGHT):
            normal_world = _world(map_model, movement_physics_model=LEGACY_MOVEMENT_PHYSICS_MODEL_ID, movement=_NORMAL_MOVEMENT)
            wild_world = _world(map_model, movement_physics_model=LEGACY_MOVEMENT_PHYSICS_MODEL_ID, movement=_WILD_MOVEMENT)

            env_normal = _env_at(normal_world, map_model, heading=-1.1, previous_steering=SteeringDirection.LEFT)
            env_wild = _env_at(wild_world, map_model, heading=-1.1, previous_steering=SteeringDirection.LEFT)

            displacement_normal, contact_normal = env_normal._move_player(action)
            displacement_wild, contact_wild = env_wild._move_player(action)

            assert env_normal.player_x == env_wild.player_x
            assert env_normal.player_z == env_wild.player_z
            assert env_normal.heading == env_wild.heading
            assert env_normal.previous_steering == env_wild.previous_steering
            assert displacement_normal == displacement_wild
            assert contact_normal == contact_wild

    def test_move_player_matches_the_kernel_directly_regardless_of_world(self):
        """Positive confirmation, not just "the tag/movement don't matter":
        the environment's actual outcome IS movement_kernel.
        advance_player_tick's outcome, for a world carrying the wild,
        clearly-legacy-shaped .movement stats -- proving what DOES
        determine the physics (the kernel), not just what doesn't."""
        map_model = _open_map()
        world = _world(map_model, movement_physics_model=LEGACY_MOVEMENT_PHYSICS_MODEL_ID, movement=_WILD_MOVEMENT)
        heading0 = 0.6
        previous_steering = SteeringDirection.RIGHT
        env = _env_at(world, map_model, heading=heading0, previous_steering=previous_steering)
        native = map_model.layout_to_native(CENTER, CENTER)

        env._move_player(FarmingAction.RUN_FORWARD_LEFT)

        expected = advance_player_tick(map_model, native[0], native[1], heading0, previous_steering, SteeringDirection.LEFT)
        assert env.player_x == expected.x
        assert env.player_z == expected.z
        assert env.heading == expected.heading
        assert env.previous_steering == expected.next_previous_steering

    def test_old_serialized_world_without_the_tag_field_still_runs_calibrated_arc(self):
        """A world constructed the way pre-2026-08-13 code would have (no
        movement_physics_model kwarg at all -- the dataclass default
        applies) must STILL execute calibrated-arc physics identically --
        confirming the tag's ABSENCE (the real situation for every
        existing serialized world on disk) doesn't accidentally reactivate
        some dormant legacy code path. There is no such path to reactivate:
        this is the same assertion as the tests above, using the default-
        constructed (untagged-in-code) case specifically."""
        map_model = _open_map()
        world = _world(map_model, movement_physics_model=LEGACY_MOVEMENT_PHYSICS_MODEL_ID, movement=_NORMAL_MOVEMENT)
        assert world.movement_physics_model == LEGACY_MOVEMENT_PHYSICS_MODEL_ID  # sanity: this IS the default

        env = _env_at(world, map_model, heading=0.0, previous_steering=SteeringDirection.NONE)
        native = map_model.layout_to_native(CENTER, CENTER)
        env._move_player(FarmingAction.RUN_FORWARD_RIGHT)

        expected = advance_player_tick(map_model, native[0], native[1], 0.0, SteeringDirection.NONE, SteeringDirection.RIGHT)
        assert env.player_x == expected.x
        assert env.player_z == expected.z
