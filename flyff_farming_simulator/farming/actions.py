from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique
from numbers import Integral
from typing import Sequence


@unique
class FarmingAction(IntEnum):
    """Legacy recorded action labels.

    Recorder archives before the factorized policy contract persist these five
    labels. They remain stable for ingestion and for the unchanged 923-value
    observation semantics, but they are no longer the policy action space.
    """

    RUN_FORWARD = 0
    RUN_FORWARD_LEFT = 1
    RUN_FORWARD_RIGHT = 2
    CAST_EVA = 3
    RUN_FORWARD_JUMP = 4

    @property
    def is_movement(self) -> bool:
        return self is not FarmingAction.CAST_EVA


@unique
class SteeringAction(IntEnum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2

    @property
    def legacy_movement(self) -> FarmingAction:
        return {
            SteeringAction.STRAIGHT: FarmingAction.RUN_FORWARD,
            SteeringAction.LEFT: FarmingAction.RUN_FORWARD_LEFT,
            SteeringAction.RIGHT: FarmingAction.RUN_FORWARD_RIGHT,
        }[self]


@unique
class FarmingEvent(IntEnum):
    NONE = 0
    CAST_EVA = 1
    JUMP = 2


@dataclass(frozen=True, slots=True)
class FarmingCommand:
    """One factorized policy command.

    W/Z is always held while farming control is active. ``steering`` controls
    only the additional left/right key and ``event`` independently taps EVA or
    jump. This is represented by Gymnasium MultiDiscrete([3, 3]).
    """

    steering: SteeringAction
    event: FarmingEvent = FarmingEvent.NONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "steering", coerce_steering(self.steering))
        object.__setattr__(self, "event", coerce_event(self.event))

    @property
    def movement_action(self) -> FarmingAction:
        return self.steering.legacy_movement

    @property
    def legacy_action(self) -> FarmingAction:
        if self.event is FarmingEvent.CAST_EVA:
            return FarmingAction.CAST_EVA
        if self.event is FarmingEvent.JUMP:
            return FarmingAction.RUN_FORWARD_JUMP
        return self.movement_action

    @property
    def eva_requested(self) -> bool:
        return self.event is FarmingEvent.CAST_EVA

    @property
    def jump_requested(self) -> bool:
        return self.event is FarmingEvent.JUMP

    def as_array(self) -> tuple[int, int]:
        return int(self.steering), int(self.event)


ACTION_NAMES: tuple[str, ...] = tuple(action.name for action in FarmingAction)
ACTION_COUNT = len(FarmingAction)
STEERING_NAMES: tuple[str, ...] = tuple(action.name for action in SteeringAction)
EVENT_NAMES: tuple[str, ...] = tuple(action.name for action in FarmingEvent)
POLICY_ACTION_NVECS: tuple[int, int] = (len(SteeringAction), len(FarmingEvent))
POLICY_ACTION_HEAD_NAMES: tuple[str, str] = ("steering", "event")


def coerce_farming_action(value: FarmingAction | int) -> FarmingAction:
    """Coerce a legacy recording/observation action label."""

    if isinstance(value, FarmingAction):
        return value
    if isinstance(value, bool) or isinstance(value, IntEnum) or not isinstance(value, Integral):
        raise ValueError(f"A unified farming action must be an integer action index, not {value!r}")
    try:
        return FarmingAction(int(value))
    except ValueError as error:
        allowed = ", ".join(f"{action.value}={action.name}" for action in FarmingAction)
        raise ValueError(f"Invalid unified farming action {value!r}; expected {allowed}") from error


def coerce_steering(value: SteeringAction | int) -> SteeringAction:
    if isinstance(value, SteeringAction):
        return value
    if isinstance(value, bool) or isinstance(value, IntEnum) or not isinstance(value, Integral):
        raise ValueError(f"Steering must be an integer index, not {value!r}")
    try:
        return SteeringAction(int(value))
    except ValueError as error:
        raise ValueError("Steering must be 0=STRAIGHT, 1=LEFT, or 2=RIGHT") from error


def coerce_event(value: FarmingEvent | int) -> FarmingEvent:
    if isinstance(value, FarmingEvent):
        return value
    if isinstance(value, bool) or isinstance(value, IntEnum) or not isinstance(value, Integral):
        raise ValueError(f"Farming event must be an integer index, not {value!r}")
    try:
        return FarmingEvent(int(value))
    except ValueError as error:
        raise ValueError("Farming event must be 0=NONE, 1=CAST_EVA, or 2=JUMP") from error


def command_from_legacy_action(
    value: FarmingAction | int,
    *,
    steering: SteeringAction = SteeringAction.STRAIGHT,
) -> FarmingCommand:
    """Translate an archive-era action to the factorized policy contract."""

    action = coerce_farming_action(value)
    if action is FarmingAction.RUN_FORWARD:
        return FarmingCommand(SteeringAction.STRAIGHT, FarmingEvent.NONE)
    if action is FarmingAction.RUN_FORWARD_LEFT:
        return FarmingCommand(SteeringAction.LEFT, FarmingEvent.NONE)
    if action is FarmingAction.RUN_FORWARD_RIGHT:
        return FarmingCommand(SteeringAction.RIGHT, FarmingEvent.NONE)
    if action is FarmingAction.CAST_EVA:
        return FarmingCommand(steering, FarmingEvent.CAST_EVA)
    return FarmingCommand(SteeringAction.STRAIGHT, FarmingEvent.JUMP)


def coerce_farming_command(
    value: FarmingCommand | Sequence[int] | FarmingAction | int,
    *,
    legacy_event_steering: SteeringAction = SteeringAction.STRAIGHT,
) -> FarmingCommand:
    """Coerce a MultiDiscrete command.

    Scalar legacy actions remain accepted only for recorder conversion, tests,
    and diagnostic scripts. Policy checkpoints are contract-validated as
    MultiDiscrete([3, 3]) and therefore cannot emit the old scalar vocabulary.
    """

    if isinstance(value, FarmingCommand):
        return value
    if isinstance(value, FarmingAction) or (
        isinstance(value, Integral) and not isinstance(value, bool)
    ):
        return command_from_legacy_action(value, steering=legacy_event_steering)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        # NumPy arrays do not register as Sequence on every supported version.
        try:
            items = list(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(
                "A factorized farming command must contain [steering, event]"
            ) from error
    else:
        items = list(value)
    if len(items) != 2:
        raise ValueError(
            f"A factorized farming command must contain exactly 2 values, got {items!r}"
        )
    return FarmingCommand(coerce_steering(items[0]), coerce_event(items[1]))
