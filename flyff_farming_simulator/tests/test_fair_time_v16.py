from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from farming.actions import FarmingAction
from simulator.environment import RecordedFarmingEnv
from simulator.fair_time_cli import build_parser
from simulator.map_model import MapModel
from simulator.synthetic import generate_synthetic_curriculum, iter_variant_environments
from simulator.world_model import MovementModel, RecordedWorldModel


def _model(map_model: MapModel) -> RecordedWorldModel:
    rng = np.random.default_rng(42)
    positions = tuple(
        map_model.layout_to_native(*map_model.random_safe_cell(rng))
        for _ in range(24)
    )
    sections = tuple(tuple(positions) for _ in range(7))
    transition = tuple(tuple(1.0 / 7.0 for _ in range(7)) for _ in range(7))
    movement = (
        MovementModel(10, 1.0, 0.0, 0.0, 0.0),
        MovementModel(10, 1.0, 0.0, 0.3, 0.0),
        MovementModel(10, 1.0, 0.0, -0.3, 0.0),
        MovementModel(0, 0.0, 0.0, 0.0, 0.0),
        MovementModel(10, 1.0, 0.0, 0.0, 0.0),
    )
    return RecordedWorldModel(
        schema_version=4,
        source_recordings=("test",),
        section_count=6,
        hub_section=6,
        population_median=24,
        section_population_probabilities=tuple(1.0 / 7.0 for _ in range(7)),
        player_start_positions=(positions[0],),
        spawn_positions_by_section=sections,
        transition_probabilities=transition,
        respawn_delay_seconds=(0.2, 0.4),
        movement=movement,
        monster_speed_cells_per_second=0.0,
        frame_interval_seconds=0.2,
        native_units_per_cell=map_model.native_units_per_cell,
        recording_frame_interval_seconds=0.5,
        cast_step_seconds=0.8,
        cast_movement_seconds=0.2,
        respawn_model_mode="global_redistribution",
        respawn_delay_source="same_slot_aggregate_provisional",
    )


def test_invalid_eva_uses_control_interval_and_restarts_cooldown() -> None:
    map_model = MapModel.load()
    env = RecordedFarmingEnv(
        _model(map_model),
        map_model=map_model,
        seed=1,
        episode_steps=20,
        episode_seconds=10.0,
    )
    env.reset(seed=1)

    _obs, _reward, _terminated, _truncated, first = env.step(
        int(FarmingAction.CAST_EVA)
    )
    assert first["valid_eva_casts"] == 1
    elapsed_before = env.elapsed
    cooldown_anchor_before = env.last_eva_at

    _obs, _reward, _terminated, _truncated, second = env.step(
        int(FarmingAction.CAST_EVA)
    )
    assert math.isclose(env.elapsed - elapsed_before, env.movement_dt)
    assert env.last_eva_at > cooldown_anchor_before
    assert second["eva_attempts"] == 2
    assert second["valid_eva_casts"] == 1
    assert second["invalid_eva_attempts"] == 1
    assert not env.eva_available()


def test_successful_eva_uses_cast_duration() -> None:
    map_model = MapModel.load()
    env = RecordedFarmingEnv(
        _model(map_model),
        map_model=map_model,
        seed=2,
        episode_steps=20,
        episode_seconds=10.0,
    )
    env.reset(seed=2)
    before = env.elapsed
    _obs, _reward, _terminated, _truncated, info = env.step(
        int(FarmingAction.CAST_EVA)
    )
    assert math.isclose(env.elapsed - before, env.cast_dt)
    assert info["valid_eva_casts"] == 1
    assert info["invalid_eva_attempts"] == 0


def test_episode_ends_by_simulated_time_with_action_cap_as_fallback() -> None:
    map_model = MapModel.load()
    env = RecordedFarmingEnv(
        _model(map_model),
        map_model=map_model,
        seed=3,
        episode_steps=100,
        episode_seconds=1.0,
    )
    env.reset(seed=3)
    truncated = False
    last = {}
    while not truncated:
        _obs, _reward, _terminated, truncated, last = env.step(
            int(FarmingAction.RUN_FORWARD)
        )
    assert float(last["elapsed_seconds"]) >= 1.0
    assert env.steps == 5

    capped = RecordedFarmingEnv(
        _model(map_model),
        map_model=map_model,
        seed=4,
        episode_steps=2,
        episode_seconds=100.0,
    )
    capped.reset(seed=4)
    _obs, _reward, _terminated, truncated, _info = capped.step(
        int(FarmingAction.RUN_FORWARD)
    )
    assert not truncated
    _obs, _reward, _terminated, truncated, _info = capped.step(
        int(FarmingAction.RUN_FORWARD)
    )
    assert truncated


def test_variant_filter_and_fixed_time_propagation(tmp_path: Path) -> None:
    map_model = MapModel.load()
    manifest = generate_synthetic_curriculum(
        tmp_path / "curriculum",
        count=3,
        seed=123,
        reference_model=_model(map_model),
        overwrite=True,
    )
    all_variants = list(iter_variant_environments(manifest, episode_steps=10))
    first_name = all_variants[0][0].name
    for _entry, env in all_variants:
        env.close()

    selected = list(
        iter_variant_environments(
            manifest,
            variant_name=first_name,
            episode_steps=20,
            episode_seconds=2.0,
        )
    )
    assert len(selected) == 1
    entry, env = selected[0]
    assert entry.name == first_name
    assert env.episode_seconds == 2.0
    observation, info = env.reset(seed=1)
    assert observation.shape == (923,)
    assert info["episode_seconds_target"] == 2.0
    env.close()


def test_fair_time_cli_fast_defaults() -> None:
    args = build_parser().parse_args(
        ["evaluate-synthetic", "curriculum.json", "policy.zip"]
    )
    assert args.episodes_per_layout == 1
    assert args.episode_seconds == 60.0
    assert args.max_actions == 400
    assert args.torch_threads == 1
