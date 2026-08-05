from __future__ import annotations

from farming.actions import FarmingAction

from .environment import RecordedFarmingEnv


def nearest_group_action(env: RecordedFarmingEnv) -> int:
    if env.eva_available() and env.eva_target_count() > 0:
        return int(FarmingAction.CAST_EVA)
    angle = env.best_group_relative_angle()
    if angle is None or abs(angle) <= 0.18:
        return int(FarmingAction.RUN_FORWARD)
    return int(
        FarmingAction.RUN_FORWARD_LEFT
        if angle > 0.0
        else FarmingAction.RUN_FORWARD_RIGHT
    )


def nearest_reachable_action(env: RecordedFarmingEnv) -> int:
    if env.eva_available() and env.eva_target_count() > 0:
        return int(FarmingAction.CAST_EVA)
    angle = env.nearest_reachable_relative_angle()
    if angle is None or abs(angle) <= 0.18:
        return int(FarmingAction.RUN_FORWARD)
    return int(
        FarmingAction.RUN_FORWARD_LEFT
        if angle > 0.0
        else FarmingAction.RUN_FORWARD_RIGHT
    )


def obstacle_aware_action(env: RecordedFarmingEnv) -> int:
    if env.eva_available() and env.eva_target_count() > 0:
        return int(FarmingAction.CAST_EVA)
    angle = env.best_group_relative_angle()
    desired = (
        FarmingAction.RUN_FORWARD
        if angle is None or abs(angle) <= 0.18
        else (
            FarmingAction.RUN_FORWARD_LEFT
            if angle > 0.0
            else FarmingAction.RUN_FORWARD_RIGHT
        )
    )
    if env.movement_path_clear(desired):
        return int(desired)
    for alternative in (
        FarmingAction.RUN_FORWARD_LEFT,
        FarmingAction.RUN_FORWARD_RIGHT,
        FarmingAction.RUN_FORWARD,
    ):
        if env.movement_path_clear(alternative):
            return int(alternative)
    return int(FarmingAction.RUN_FORWARD_JUMP)


def scripted_action(name: str, env: RecordedFarmingEnv) -> int:
    normalized = str(name).strip().lower()
    if normalized == "nearest_group":
        return nearest_group_action(env)
    if normalized == "nearest_reachable":
        return nearest_reachable_action(env)
    if normalized == "obstacle_aware":
        return obstacle_aware_action(env)
    raise ValueError(f"Unsupported scripted policy: {name}")

# Factorized-policy helpers -------------------------------------------------
from farming.actions import FarmingCommand, FarmingEvent, SteeringAction


def _steering_for_angle(angle: float | None) -> SteeringAction:
    if angle is None or abs(angle) <= 0.18:
        return SteeringAction.STRAIGHT
    return SteeringAction.LEFT if angle > 0.0 else SteeringAction.RIGHT


def nearest_group_command(env: RecordedFarmingEnv) -> FarmingCommand:
    steering = _steering_for_angle(env.best_group_relative_angle())
    event = (
        FarmingEvent.CAST_EVA
        if env.eva_available() and env.eva_target_count() > 0
        else FarmingEvent.NONE
    )
    return FarmingCommand(steering, event)


def nearest_reachable_command(env: RecordedFarmingEnv) -> FarmingCommand:
    steering = _steering_for_angle(env.nearest_reachable_relative_angle())
    event = (
        FarmingEvent.CAST_EVA
        if env.eva_available() and env.eva_target_count() > 0
        else FarmingEvent.NONE
    )
    return FarmingCommand(steering, event)


def obstacle_aware_command(env: RecordedFarmingEnv) -> FarmingCommand:
    steering = _steering_for_angle(env.best_group_relative_angle())
    movement = steering.legacy_movement
    if not env.movement_path_clear(movement):
        for alternative in (
            SteeringAction.LEFT,
            SteeringAction.RIGHT,
            SteeringAction.STRAIGHT,
        ):
            if env.movement_path_clear(alternative.legacy_movement):
                steering = alternative
                break
        else:
            return FarmingCommand(steering, FarmingEvent.JUMP)
    event = (
        FarmingEvent.CAST_EVA
        if env.eva_available() and env.eva_target_count() > 0
        else FarmingEvent.NONE
    )
    return FarmingCommand(steering, event)


def scripted_command(name: str, env: RecordedFarmingEnv) -> FarmingCommand:
    normalized = str(name).strip().lower()
    if normalized == "nearest_group":
        return nearest_group_command(env)
    if normalized == "nearest_reachable":
        return nearest_reachable_command(env)
    if normalized == "obstacle_aware":
        return obstacle_aware_command(env)
    raise ValueError(f"Unsupported scripted policy: {name}")
