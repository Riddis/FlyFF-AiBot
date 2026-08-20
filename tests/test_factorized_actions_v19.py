from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from farming.actions import (
    FarmingAction,
    FarmingCommand,
    FarmingEvent,
    SteeringAction,
    coerce_farming_command,
)
from farming.model_contract import (
    ModelContractError,
    ModelContractMetadata,
    ModelSpaceSignature,
    validate_model_contract,
)
from simulator.scripted_policies import obstacle_aware_command
from simulator.synthetic import iter_variant_environments


def test_factorized_command_maps_steering_and_event_independently() -> None:
    command = FarmingCommand(SteeringAction.LEFT, FarmingEvent.CAST_EVA)

    assert command.movement_action is FarmingAction.RUN_FORWARD_LEFT
    assert command.legacy_action is FarmingAction.CAST_EVA
    assert command.as_array() == (1, 1)


def test_numpy_multidiscrete_action_is_accepted() -> None:
    command = coerce_farming_command(np.asarray([2, 2], dtype=np.int64))

    assert command.steering is SteeringAction.RIGHT
    assert command.event is FarmingEvent.JUMP


def test_legacy_eva_preserves_supplied_steering_for_diagnostics() -> None:
    command = coerce_farming_command(
        FarmingAction.CAST_EVA,
        legacy_event_steering=SteeringAction.RIGHT,
    )

    assert command == FarmingCommand(SteeringAction.RIGHT, FarmingEvent.CAST_EVA)


def test_factorized_model_contract_rejects_scalar_discrete_space() -> None:
    observation_space = SimpleNamespace(shape=(923,), dtype=np.dtype("float32"))
    with pytest.raises(ModelContractError, match="scalar Discrete"):
        ModelSpaceSignature.from_spaces(
            observation_space,
            SimpleNamespace(n=5, start=0),
        )


def test_factorized_model_contract_accepts_multidiscrete_signature() -> None:
    signature = ModelSpaceSignature.from_spaces(
        SimpleNamespace(shape=(923,), dtype=np.dtype("float32")),
        SimpleNamespace(
            nvec=np.asarray([3, 3], dtype=np.int64),
            start=np.asarray([0, 0], dtype=np.int64),
        ),
    )

    validation = validate_model_contract(
        signature,
        metadata=ModelContractMetadata.current(),
    )
    assert validation.contract_hash == ModelContractMetadata.current().contract_hash


def test_environment_applies_steering_during_eva_and_jump() -> None:
    entry, env = next(
        iter(
            iter_variant_environments(
                "curricula/synthetic_curriculum/curriculum.json",
                stage="early",
                episode_steps=20,
                episode_seconds=3.0,
            )
        )
    )
    del entry
    try:
        observation, _ = env.reset(seed=7)
        assert observation.shape == (923,)

        _, _, _, _, eva_info = env.step([1, 1])
        assert eva_info["factorized_action"] == [1, 1]
        assert eva_info["steering"] == "LEFT"
        assert eva_info["event"] == "CAST_EVA"
        assert env.held_movement is FarmingAction.RUN_FORWARD_LEFT

        _, _, _, _, jump_info = env.step([2, 2])
        assert jump_info["factorized_action"] == [2, 2]
        assert jump_info["steering"] == "RIGHT"
        assert jump_info["event"] == "JUMP"
        assert env.held_movement is FarmingAction.RUN_FORWARD_RIGHT
    finally:
        env.close()


def test_obstacle_aware_teacher_prefers_eva_over_jump_when_boxed_in() -> None:
    """EVA does not require movement, so it must win even when every steering
    direction is blocked. A prior regression returned JUMP unconditionally in
    the fully-blocked branch and never reached the EVA check, which silently
    discarded the majority of EVA-eligible teacher labels near obstacles."""

    fake_env = SimpleNamespace(
        best_group_relative_angle=lambda: 0.0,
        nearest_reachable_relative_angle=lambda: None,
        movement_path_clear=lambda action: False,
        eva_available=lambda: True,
        eva_target_count=lambda: 2,
    )

    command = obstacle_aware_command(fake_env)

    assert command.event is FarmingEvent.CAST_EVA


def test_obstacle_aware_teacher_never_jumps_to_escape_a_blocked_cell() -> None:
    """Jump is idle flair, never an obstacle-recovery mechanic. Being boxed in
    on every steering direction must not by itself trigger a jump label."""

    fake_env = SimpleNamespace(
        best_group_relative_angle=lambda: 0.0,
        nearest_reachable_relative_angle=lambda: None,
        movement_path_clear=lambda action: False,
        eva_available=lambda: False,
        eva_target_count=lambda: 0,
        jump_available=lambda: True,
        rng=SimpleNamespace(random=lambda: 0.999),
    )

    command = obstacle_aware_command(fake_env)

    assert command.event is FarmingEvent.NONE


def test_obstacle_aware_teacher_flair_jump_is_independent_of_obstacles() -> None:
    """A rare flair jump can still occur while cruising normally (not
    blocked, no EVA target), driven only by the low idle-jump roll."""

    fake_env = SimpleNamespace(
        best_group_relative_angle=lambda: 0.0,
        nearest_reachable_relative_angle=lambda: None,
        movement_path_clear=lambda action: True,
        eva_available=lambda: False,
        eva_target_count=lambda: 0,
        jump_available=lambda: True,
        rng=SimpleNamespace(random=lambda: 0.0),
    )

    command = obstacle_aware_command(fake_env)

    assert command.event is FarmingEvent.JUMP


def test_obstacle_aware_teacher_steers_toward_reachable_target_not_unreachable_best() -> None:
    """A visible 'best' group across an obstacle cannot actually be walked to.
    Chasing it left the player wedged against the obstacle for the rest of an
    episode in practice. When a genuinely reachable target exists, it must be
    preferred over an unreachable higher-scoring one."""

    fake_env = SimpleNamespace(
        best_group_relative_angle=lambda: 0.0,  # unreachable target: dead ahead
        nearest_reachable_relative_angle=lambda: -0.5,  # reachable target: to the right
        movement_path_clear=lambda action: True,
        eva_available=lambda: False,
        eva_target_count=lambda: 0,
        jump_available=lambda: True,
        rng=SimpleNamespace(random=lambda: 0.999),
    )

    command = obstacle_aware_command(fake_env)

    assert command.steering is SteeringAction.RIGHT


def test_obstacle_aware_teacher_returns_factorized_command() -> None:
    _entry, env = next(
        iter(
            iter_variant_environments(
                "curricula/synthetic_curriculum/curriculum.json",
                stage="early",
                episode_steps=20,
                episode_seconds=3.0,
            )
        )
    )
    try:
        env.reset(seed=3)
        command = obstacle_aware_command(env)
        assert isinstance(command, FarmingCommand)
        assert 0 <= int(command.steering) < 3
        assert 0 <= int(command.event) < 3
    finally:
        env.close()
