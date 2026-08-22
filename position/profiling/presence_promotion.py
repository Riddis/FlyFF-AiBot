from __future__ import annotations

import struct
from collections.abc import Iterable, Mapping
from typing import Protocol

from ..IndependentNativeReader import IndependentNativeReader


class ProcessMemoryReader(Protocol):
    def read(self, address: int, size: int) -> bytes: ...


def evidence_supports_presence_promotion(evidence: Mapping[str, object]) -> bool:
    """Apply the frozen longitudinal evidence gate without touching Layer 1."""

    live_samples = int(evidence.get("live_samples", 0))
    live_matches = int(evidence.get("live_matches", 0))
    dormant_samples = int(evidence.get("dormant_samples", 0))
    dormant_matches = int(evidence.get("dormant_matches", 0))
    inactive_samples = int(evidence.get("inactive_samples", 0))
    inactive_matches = int(evidence.get("inactive_matches", 0))
    unique_bases = int(evidence.get("unique_bases", 0))
    species_values = evidence.get("species", ())
    species_count = (
        len(species_values)
        if isinstance(species_values, (list, tuple, set))
        else 0
    )
    return bool(
        live_samples >= 32
        and live_matches * 10 >= live_samples * 9
        and dormant_samples >= 8
        and dormant_matches * 10 <= dormant_samples
        and (inactive_samples < 4 or inactive_matches * 10 <= inactive_samples)
        and unique_bases >= 16
        and (species_count >= 2 or live_samples >= 64)
    )


def promote_validated_presence_offset(
    reader: IndependentNativeReader,
    memory: ProcessMemoryReader,
    offset: int,
    *,
    evidence: Mapping[str, object],
    selected_species_ids: Iterable[int],
    source: str = "session_longitudinal_profiler",
) -> bool:
    """Revalidate a profiled candidate in the current process, then install it."""

    if not evidence_supports_presence_promotion(evidence):
        return False
    resolved = int(offset)
    selected = {int(value) for value in selected_species_ids if int(value) > 0}
    if not selected or not reader.monster_targets:
        return False
    layout = reader.monster_targets[0]
    current_samples = 0
    current_matches = 0
    for base in reader.actor_slots[:256]:
        try:
            species = struct.unpack(
                "<i", memory.read(int(base) + int(layout.species_offset), 4)
            )[0]
            if species not in selected:
                continue
            value = struct.unpack("<i", memory.read(int(base) + resolved, 4))[0]
        except Exception:
            continue
        current_samples += 1
        current_matches += int(value == species)
        if current_samples >= 64:
            break
    if current_samples < 8 or current_matches * 5 < current_samples * 4:
        return False
    return reader.install_validated_presence_offset(resolved, source=source)
