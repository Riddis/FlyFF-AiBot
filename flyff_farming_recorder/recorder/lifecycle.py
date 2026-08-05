from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from position.IndependentNativeReader import IndependentActorSlotRead


STATE_CODES = {
    "empty": 0,
    "living": 1,
    "dead": 2,
    "other_species": 3,
    "invalid_hp": 4,
    "unreadable": 5,
}


@dataclass(slots=True)
class LifecycleTracker:
    """Classify actor-slot transitions without treating slot addresses as identities."""

    kill_radius: float
    previous: dict[int, IndependentActorSlotRead] = field(default_factory=dict, init=False)
    last_death: dict[tuple[int, int], tuple[int, int, int]] = field(default_factory=dict, init=False)
    deaths: int = field(default=0, init=False)
    target_appearances: int = field(default=0, init=False)
    target_disappearances: int = field(default=0, init=False)
    respawn_candidates: int = field(default=0, init=False)
    probable_kills: int = field(default=0, init=False)
    reuses: int = field(default=0, init=False)
    deaths_by_species: dict[int, int] = field(default_factory=dict, init=False)
    probable_kills_by_species: dict[int, int] = field(default_factory=dict, init=False)
    appearances_by_species: dict[int, int] = field(default_factory=dict, init=False)
    disappearances_by_species: dict[int, int] = field(default_factory=dict, init=False)

    def update(
        self,
        states: Iterable[IndependentActorSlotRead],
        *,
        elapsed_ms: int,
        player_x_q: int,
        player_z_q: int,
        phase: int,
        quantize,
    ) -> list[list[object]]:
        current = {int(state.base): state for state in states}
        events: list[list[object]] = []
        for base in sorted(set(self.previous) & set(current)):
            before = self.previous[base]
            after = current[base]
            if before.species > 0 and after.species > 0 and before.species != after.species:
                self.reuses += 1
                events.append([
                    "reuse", elapsed_ms, base, before.species, after.species,
                    before.hp, after.hp, phase,
                ])

            before_live_target = bool(before.target_species and before.hp > 0)
            after_live_target = bool(after.target_species and after.hp > 0)

            if (
                before.target_species
                and after.target_species
                and before.species == after.species
                and before.hp > 0
                and after.hp == 0
            ):
                distance_q = quantize(before.distance_native)
                probable = bool(before.distance_native <= self.kill_radius)
                self.deaths += 1
                self.probable_kills += int(probable)
                species = int(before.species)
                self.deaths_by_species[species] = self.deaths_by_species.get(species, 0) + 1
                if probable:
                    self.probable_kills_by_species[species] = (
                        self.probable_kills_by_species.get(species, 0) + 1
                    )
                death_position = (quantize(before.x), quantize(before.z))
                self.last_death[(base, before.species)] = (
                    elapsed_ms,
                    death_position[0],
                    death_position[1],
                )
                events.append([
                    "death", elapsed_ms, base, before.species, before.hp,
                    death_position[0], death_position[1], player_x_q, player_z_q,
                    distance_q, probable, phase,
                ])

            # A living target that turns into another species/unreadable object is a
            # stream/reallocation event, not a death or a proven despawn.
            if before_live_target and not after_live_target and not (
                after.target_species and after.hp == 0
            ):
                self.target_disappearances += 1
                species = int(before.species)
                self.disappearances_by_species[species] = (
                    self.disappearances_by_species.get(species, 0) + 1
                )
                events.append([
                    "target_disappearance", elapsed_ms, base, before.species, before.hp,
                    quantize(before.x), quantize(before.z), after.species, after.hp,
                    int(STATE_CODES.get(after.state, STATE_CODES["unreadable"])), phase,
                ])

            if after_live_target and not before_live_target:
                prior = self.last_death.pop((base, after.species), None)
                if prior is None:
                    self.target_appearances += 1
                    species = int(after.species)
                    self.appearances_by_species[species] = (
                        self.appearances_by_species.get(species, 0) + 1
                    )
                    events.append([
                        "target_appearance", elapsed_ms, base, after.species, after.hp,
                        quantize(after.x), quantize(after.z), before.species, before.hp,
                        int(STATE_CODES.get(before.state, STATE_CODES["unreadable"])), phase,
                    ])
                else:
                    self.respawn_candidates += 1
                    delay_ms = max(0, elapsed_ms - prior[0])
                    events.append([
                        "respawn_candidate", elapsed_ms, base, after.species, after.hp,
                        quantize(after.x), quantize(after.z), before.species, before.hp,
                        prior[0], prior[1], prior[2], delay_ms, phase,
                    ])
        self.previous = current
        return events
