from __future__ import annotations

from enum import IntEnum, unique
from numbers import Integral


@unique
class FarmingAction(IntEnum):
    """The complete action vocabulary of the unified farming policy.

    Values are persisted in PPO checkpoints.  Reordering or extending this
    enum is a model-contract change, even if the member names stay the same.
    """

    RUN_FORWARD = 0
    RUN_FORWARD_LEFT = 1
    RUN_FORWARD_RIGHT = 2
    CAST_EVA = 3

    @property
    def is_movement(self) -> bool:
        return self is not FarmingAction.CAST_EVA


ACTION_NAMES: tuple[str, ...] = tuple(action.name for action in FarmingAction)
ACTION_COUNT = len(FarmingAction)


def coerce_farming_action(value: FarmingAction | int) -> FarmingAction:
    """Return a strict unified action or raise a user-facing ``ValueError``."""

    if isinstance(value, FarmingAction):
        return value
    if (
        isinstance(value, bool)
        or isinstance(value, IntEnum)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            f"A unified farming action must be an integer action index, not {value!r}"
        )
    try:
        return FarmingAction(int(value))
    except ValueError as error:
        allowed = ", ".join(f"{action.value}={action.name}" for action in FarmingAction)
        raise ValueError(
            f"Invalid unified farming action {value!r}; expected {allowed}"
        ) from error
