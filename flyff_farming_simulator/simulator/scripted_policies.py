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


# Jump is never an obstacle-recovery or "unstuck" mechanic: getting boxed in
# is a steering/positioning problem the policy must learn to avoid on its own
# (via contact/obstacle penalties), not something a jump resolves. Jump is
# purely idle flair -- an occasional human-like tap while cruising with
# nothing better to do. The rate below matches the ~0.25% jump frequency
# measured in the real direct-WASD Riddims recording (3 of 1,196 samples), so
# the teacher reproduces a human cadence rather than an invented one.
FLAIR_JUMP_PROBABILITY = 0.0025


def _event_for(env: RecordedFarmingEnv) -> FarmingEvent:
    if env.eva_available() and env.eva_target_count() > 0:
        return FarmingEvent.CAST_EVA
    if env.jump_available() and env.rng.random() < FLAIR_JUMP_PROBABILITY:
        return FarmingEvent.JUMP
    return FarmingEvent.NONE


def nearest_group_command(env: RecordedFarmingEnv) -> FarmingCommand:
    steering = _steering_for_angle(env.best_group_relative_angle())
    return FarmingCommand(steering, _event_for(env))


def nearest_reachable_command(env: RecordedFarmingEnv) -> FarmingCommand:
    steering = _steering_for_angle(env.nearest_reachable_relative_angle())
    return FarmingCommand(steering, _event_for(env))


def _obstacle_aware_target_angle(env: RecordedFarmingEnv) -> float | None:
    """Prefer a geodesically reachable target over the highest-scoring
    visible one. ``best_group_relative_angle`` can point at a group that is
    visible but separated by an obstacle; chasing it left the player wedged
    against the obstacle for the rest of the episode, since the local
    left/right/straight dodge below never resolves an unreachable heading.
    Falling back to the best visible angle only when nothing is reachable at
    all still orients the player toward distant activity instead of idling."""

    reachable = env.nearest_reachable_relative_angle()
    if reachable is not None:
        return reachable
    return env.best_group_relative_angle()


def obstacle_aware_command(env: RecordedFarmingEnv) -> FarmingCommand:
    steering = _steering_for_angle(_obstacle_aware_target_angle(env))
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
        # If no alternative is clear either, keep the originally desired
        # steering rather than substituting a jump. EVA below still fires
        # regardless of movement, since it never required a clear path.
    return FarmingCommand(steering, _event_for(env))


def scripted_command(name: str, env: RecordedFarmingEnv) -> FarmingCommand:
    normalized = str(name).strip().lower()
    if normalized == "nearest_group":
        return nearest_group_command(env)
    if normalized == "nearest_reachable":
        return nearest_reachable_command(env)
    if normalized == "obstacle_aware":
        return obstacle_aware_command(env)
    raise ValueError(f"Unsupported scripted policy: {name}")
