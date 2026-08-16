from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PlayerDiscrimination(str, Enum):
    """Validated strategies for excluding monster anchors from player proof."""

    LEGACY_SPECIES_ACTIVE = "legacy_species_active"
    EXACT_MONSTER_ANCHORS = "exact_monster_anchors"


@dataclass(frozen=True, slots=True)
class AttachPolicy:
    """Application-selected behavior passed into the shared position mechanism."""

    name: str
    player_discrimination: PlayerDiscrimination
    activate_presence_sampling_on_attach: bool
    allow_longitudinal_presence_profiling: bool


LIVE_ATTACH_POLICY = AttachPolicy(
    name="live",
    player_discrimination=PlayerDiscrimination.LEGACY_SPECIES_ACTIVE,
    activate_presence_sampling_on_attach=True,
    allow_longitudinal_presence_profiling=False,
)

RECORDING_ATTACH_POLICY = AttachPolicy(
    name="recording",
    player_discrimination=PlayerDiscrimination.EXACT_MONSTER_ANCHORS,
    activate_presence_sampling_on_attach=False,
    allow_longitudinal_presence_profiling=True,
)
