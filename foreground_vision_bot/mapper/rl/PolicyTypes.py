from __future__ import annotations

from enum import IntEnum


class MapperAction(IntEnum):
    """High-level actions available to the exploration policy."""

    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    WAIT = 3
    REACQUIRE_HEADING = 4
    BACKTRACK = 5


class ObservationQuality(IntEnum):
    """Quality of the latest live/simulated observation."""

    VALID = 0
    CONTACT = 1
    CAMERA_OBSCURED = 2
    HEADING_UNAVAILABLE = 3
    UNRESOLVED = 4


class MotionOutcome(IntEnum):
    """Outcome of the most recent high-level action."""

    NONE = 0
    MOVED = 1
    BLOCKED = 2
    CONTACT_SLIDE = 3
    TURNED = 4
    RECOVERED = 5
    INVALID_OBSERVATION = 6


ACTION_NAMES: tuple[str, ...] = tuple(action.name for action in MapperAction)
QUALITY_NAMES: tuple[str, ...] = tuple(item.name for item in ObservationQuality)
OUTCOME_NAMES: tuple[str, ...] = tuple(item.name for item in MotionOutcome)


def action_from_name(name: str) -> MapperAction:
    normalized = str(name).strip().upper()
    aliases = {
        "REACQUIRE": "REACQUIRE_HEADING",
        "REOBSERVE": "WAIT",
        "STOP": "WAIT",
    }
    normalized = aliases.get(normalized, normalized)
    return MapperAction[normalized]
