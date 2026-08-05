from __future__ import annotations

import math

import numpy as np

from farming.actions import FarmingAction
from farming.map_features import MapCellRisk
from simulator.environment import RecordedFarmingEnv
from simulator.fair_time_cli import build_parser
from simulator.map_model import MapModel
from simulator.reward_model import (
    REWARD_CONTRACT_ID,
    SimulatorRewardCalculator,
    SimulatorRewardConfig,
    SimulatorRewardEvidence,
)
from simulator.world_model import MovementModel, RecordedWorldModel


def _world(map_model: MapModel) -> RecordedWorldModel:
    rng = np.random.default_rng(71)
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
        source_recordings=("reward-test",),
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


def test_concave_kill_reward_bounds_large_group_spikes() -> None:
    calculator = SimulatorRewardCalculator()
    one = calculator.calculate(SimulatorRewardEvidence(native_kill_delta=1))
    hundred = calculator.calculate(SimulatorRewardEvidence(native_kill_delta=100))
    assert math.isclose(one.components.kill, 1.0)
    assert math.isclose(hundred.components.kill, 10.0)
    assert hundred.components.kill < 100.0


def test_approach_is_linear_and_contact_applies_during_eva() -> None:
    calculator = SimulatorRewardCalculator()
    result = calculator.calculate(
        SimulatorRewardEvidence(
            approach_progress_cells=100.0,
            eva_attempted=True,
            eva_available=True,
            contact=True,
            map_cell_risk=MapCellRisk.SAFE,
        )
    )
    assert math.isclose(result.components.approach, 3.0)
    assert math.isclose(result.components.contact, -0.035)
    assert math.isclose(result.components.eva_miss, -0.05)


def test_jump_flair_remains_enabled() -> None:
    calculator = SimulatorRewardCalculator()
    result = calculator.calculate(
        SimulatorRewardEvidence(
            jump_performed=True,
            map_cell_risk=MapCellRisk.SAFE,
        )
    )
    assert math.isclose(result.components.jump_flair, 0.001)


def test_environment_reports_cumulative_reward_components() -> None:
    map_model = MapModel.load()
    env = RecordedFarmingEnv(
        _world(map_model),
        map_model=map_model,
        seed=8,
        episode_steps=10,
        episode_seconds=10.0,
    )
    env.reset(seed=8)
    _obs, reward, _terminated, _truncated, info = env.step(
        int(FarmingAction.RUN_FORWARD_JUMP)
    )
    totals = info["reward_component_totals"]
    assert info["reward_contract"]["reward_contract"] == REWARD_CONTRACT_ID
    assert math.isclose(totals["jump_flair"], 0.001)
    assert math.isclose(sum(totals.values()), reward)
    env.close()


def test_monsters_remain_visible_beyond_collision_and_eva_ranges() -> None:
    map_model = MapModel.load()
    env = RecordedFarmingEnv(
        _world(map_model),
        map_model=map_model,
        seed=9,
        episode_steps=10,
        episode_seconds=10.0,
    )
    env.reset(seed=9)
    player_cell = map_model.native_to_layout_cell(env.player_x, env.player_z)
    field = map_model.features.bounded_geodesic_field(
        player_cell, maximum_distance_cells=45.0
    )
    distant_cell, distance = next(
        (cell, value) for cell, value in field.items() if 25.0 <= value <= 35.0
    )
    x, z = map_model.layout_to_native(*distant_cell)
    for actor in env.actors:
        actor.alive = False
    env.actors[0].alive = True
    env.actors[0].x = x
    env.actors[0].z = z
    candidates = env._visible_candidates()
    assert distance > env.eva_radius_cells
    assert distance > 5.0
    assert len(candidates) == 1
    assert candidates[0][0] <= env.vision_radius_cells
    env.close()


def test_reward_audit_cli_defaults() -> None:
    args = build_parser().parse_args(["audit-rewards", "curriculum.json"])
    assert args.stage == "all"
    assert args.episodes_per_layout == 3
    assert args.episode_seconds == 10.0
    assert args.max_actions == 80
    assert args.layout_limit == 0
    assert not args.require_sanity


def test_reward_config_keeps_jump_flair() -> None:
    config = SimulatorRewardConfig()
    assert config.jump_flair_reward == 0.001
    assert config.as_dict()["reward_contract"] == REWARD_CONTRACT_ID


def test_missed_eva_opportunity_penalty_only_applies_when_ready_and_grouped() -> None:
    calculator = SimulatorRewardCalculator()
    missed = calculator.calculate(
        SimulatorRewardEvidence(
            eva_attempted=False,
            eva_available=True,
            eva_target_count_before_action=3,
            map_cell_risk=MapCellRisk.SAFE,
        )
    )
    assert math.isclose(missed.components.missed_eva_opportunity, -0.04)

    unavailable = calculator.calculate(
        SimulatorRewardEvidence(
            eva_attempted=False,
            eva_available=False,
            eva_target_count_before_action=20,
            map_cell_risk=MapCellRisk.SAFE,
        )
    )
    too_small = calculator.calculate(
        SimulatorRewardEvidence(
            eva_attempted=False,
            eva_available=True,
            eva_target_count_before_action=2,
            map_cell_risk=MapCellRisk.SAFE,
        )
    )
    cast = calculator.calculate(
        SimulatorRewardEvidence(
            eva_attempted=True,
            eva_available=True,
            eva_target_count_before_action=3,
            native_kill_delta=3,
            map_cell_risk=MapCellRisk.SAFE,
        )
    )
    assert unavailable.components.missed_eva_opportunity == 0.0
    assert too_small.components.missed_eva_opportunity == 0.0
    assert cast.components.missed_eva_opportunity == 0.0


def test_environment_counts_missed_eva_opportunity() -> None:
    map_model = MapModel.load()
    env = RecordedFarmingEnv(
        _world(map_model),
        map_model=map_model,
        seed=91,
        episode_steps=10,
        episode_seconds=10.0,
    )
    env.reset(seed=91)
    for actor in env.actors:
        actor.alive = False
    for actor in env.actors[:3]:
        actor.alive = True
        actor.x = env.player_x
        actor.z = env.player_z
    _obs, _reward, _terminated, _truncated, info = env.step(
        int(FarmingAction.RUN_FORWARD)
    )
    assert info["missed_eva_opportunities"] == 1
    assert math.isclose(
        info["reward_components"]["missed_eva_opportunity"], -0.04
    )
    env.close()


def test_train_cli_exposes_teacher_bootstrap_options() -> None:
    args = build_parser().parse_args(
        [
            "train-synthetic",
            "curriculum.json",
            "--output",
            "pilot",
            "--teacher-bootstrap-samples",
            "4000",
            "--teacher-bootstrap-policy",
            "obstacle_aware",
        ]
    )
    assert args.teacher_bootstrap_samples == 4000
    assert args.teacher_bootstrap_epochs == 8
    assert args.teacher_bootstrap_policy == "obstacle_aware"
