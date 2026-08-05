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
                "synthetic_curriculum/curriculum.json",
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


def test_obstacle_aware_teacher_returns_factorized_command() -> None:
    _entry, env = next(
        iter(
            iter_variant_environments(
                "synthetic_curriculum/curriculum.json",
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
