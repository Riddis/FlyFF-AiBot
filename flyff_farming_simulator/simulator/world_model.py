from __future__ import annotations

import gzip
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np
from farming.actions import ACTION_NAMES

from .map_model import MapModel
from .schema import (
    RecordingArchive,
    has_validated_presence,
    unique_recording_paths,
    validate_recording_contract,
)

_FORWARD_BIT = 1 << 0
_LEFT_BIT = 1 << 1
_RIGHT_BIT = 1 << 2
_JUMP_BIT = 1 << 3


@dataclass(frozen=True, slots=True)
class MovementModel:
    samples: int
    distance_mean_cells: float
    distance_std_cells: float
    turn_mean_radians: float
    turn_std_radians: float


@dataclass(frozen=True, slots=True)
class RecordedWorldModel:
    schema_version: int
    source_recordings: tuple[str, ...]
    section_count: int
    hub_section: int
    population_median: int
    section_population_probabilities: tuple[float, ...]
    player_start_positions: tuple[tuple[float, float], ...]
    spawn_positions_by_section: tuple[tuple[tuple[float, float], ...], ...]
    transition_probabilities: tuple[tuple[float, ...], ...]
    respawn_delay_seconds: tuple[float, ...]
    movement: tuple[MovementModel, ...]
    monster_speed_cells_per_second: float
    frame_interval_seconds: float
    native_units_per_cell: float
    respawn_candidate_count: int = 0
    unmatched_appearance_count: int = 0
    legacy_recording_count: int = 0
    respawn_position_sample_counts: tuple[int, ...] = ()
    recording_frame_interval_seconds: float = 0.2
    cast_step_seconds: float = 0.8
    cast_movement_seconds: float = 0.2
    respawn_model_mode: str = "global_redistribution"
    respawn_delay_source: str = "same_slot_aggregate_provisional"
    human_action_probabilities: tuple[float, ...] = (0.60, 0.14, 0.14, 0.10, 0.02)
    fit_warnings: tuple[str, ...] = field(default_factory=tuple)

    def save(self, path: str | Path) -> Path:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        with gzip.open(resolved, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(payload, handle, separators=(",", ":"))
        return resolved

    @classmethod
    def load(cls, path: str | Path) -> "RecordedWorldModel":
        with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["source_recordings"] = tuple(payload["source_recordings"])
        if "section_population_probabilities" not in payload:
            section_total = int(payload["section_count"]) + 1
            payload["section_population_probabilities"] = [1.0 / section_total] * section_total
        payload["section_population_probabilities"] = tuple(
            float(value) for value in payload["section_population_probabilities"]
        )
        payload["player_start_positions"] = tuple(tuple(item) for item in payload["player_start_positions"])
        payload["spawn_positions_by_section"] = tuple(
            tuple(tuple(position) for position in section)
            for section in payload["spawn_positions_by_section"]
        )
        payload["transition_probabilities"] = tuple(tuple(row) for row in payload["transition_probabilities"])
        payload["respawn_delay_seconds"] = tuple(float(value) for value in payload["respawn_delay_seconds"])
        payload["movement"] = tuple(MovementModel(**item) for item in payload["movement"])
        payload["respawn_position_sample_counts"] = tuple(
            int(value) for value in payload.get("respawn_position_sample_counts", ())
        )
        payload["human_action_probabilities"] = tuple(
            float(value)
            for value in payload.get(
                "human_action_probabilities", (0.60, 0.14, 0.14, 0.10, 0.02)
            )
        )
        payload["fit_warnings"] = tuple(str(value) for value in payload.get("fit_warnings", ()))
        payload.setdefault("recording_frame_interval_seconds", float(payload.get("frame_interval_seconds", 0.2)))
        payload.setdefault("cast_step_seconds", 0.8)
        payload.setdefault("cast_movement_seconds", float(payload.get("frame_interval_seconds", 0.2)))
        payload.setdefault("respawn_model_mode", "legacy_same_slot")
        payload.setdefault("respawn_delay_source", "legacy_same_slot")
        return cls(**payload)


def movement_action_from_mask(mask: int) -> int | None:
    value = int(mask)
    forward = bool(value & _FORWARD_BIT)
    left = bool(value & _LEFT_BIT)
    right = bool(value & _RIGHT_BIT)
    jump = bool(value & _JUMP_BIT)
    if forward and jump:
        return 4
    if forward and left and not right:
        return 1
    if forward and right and not left:
        return 2
    if forward and not left and not right:
        return 0
    return None


def steering_action_from_mask(mask: int) -> int | None:
    """Steering alone (0=STRAIGHT, 1=LEFT, 2=RIGHT), independent of the jump
    bit. ``movement_action_from_mask`` collapses any forward+jump combination
    to the single legacy action 4 regardless of concurrent left/right bits,
    which is correct for the legacy 5-way scalar contract but would silently
    erase steering for a factorized (steering, event) conversion -- a jump
    tap must never replace the concurrently-held steering key. Left and
    right held together are ambiguous and intentionally return STRAIGHT,
    matching movement_action_from_mask's own tie-breaking.
    """

    value = int(mask)
    if not (value & _FORWARD_BIT):
        return None
    left = bool(value & _LEFT_BIT)
    right = bool(value & _RIGHT_BIT)
    if left and not right:
        return 1
    if right and not left:
        return 2
    return 0


def _robust_delay_samples(values: list[float]) -> list[float]:
    valid = np.asarray(
        [value for value in values if math.isfinite(value) and 0.1 <= value <= 120.0],
        dtype=np.float64,
    )
    if valid.size == 0:
        return [12.0]
    if valid.size < 20:
        return [float(value) for value in valid]
    center = float(np.median(valid))
    absolute = np.abs(valid - center)
    mad = float(np.median(absolute))
    if mad > 1.0e-9:
        lower = max(0.1, center - 4.0 * mad)
        upper = min(120.0, center + 4.0 * mad)
    else:
        lower = max(0.1, float(np.quantile(valid, 0.01)))
        upper = min(120.0, float(np.quantile(valid, 0.99)))
    clipped = valid[(valid >= lower) & (valid <= upper)]
    if clipped.size < max(10, int(valid.size * 0.50)):
        clipped = valid[(valid >= np.quantile(valid, 0.02)) & (valid <= np.quantile(valid, 0.95))]
    return [float(value) for value in clipped]


def fit_world_model(
    recording_paths: Iterable[str | Path],
    *,
    map_model: MapModel | None = None,
    section_count: int = 6,
    seed: int = 0,
    maximum_positions_per_section: int = 20_000,
    control_interval_seconds: float = 0.20,
    cast_step_seconds: float = 0.80,
    cast_movement_seconds: float = 0.20,
    population_scale: float = 1.0,
    allow_unvalidated_presence: bool = False,
) -> RecordedWorldModel:
    paths = unique_recording_paths(recording_paths)
    if not paths:
        raise ValueError("At least one recording ZIP is required")
    if section_count < 2:
        raise ValueError("section_count must be at least two")
    if not math.isfinite(control_interval_seconds) or control_interval_seconds <= 0.0:
        raise ValueError("control_interval_seconds must be finite and positive")
    if not math.isfinite(cast_step_seconds) or cast_step_seconds <= 0.0:
        raise ValueError("cast_step_seconds must be finite and positive")
    if not math.isfinite(cast_movement_seconds) or cast_movement_seconds < 0.0:
        raise ValueError("cast_movement_seconds must be finite and non-negative")
    if not math.isfinite(population_scale) or population_scale <= 0.0:
        raise ValueError("population_scale must be finite and positive")

    map_data = map_model or MapModel.load()
    rng = np.random.default_rng(seed)
    section_total = section_count + 1
    observed_positions: list[list[tuple[float, float]]] = [[] for _ in range(section_total)]
    observed_position_samples = np.zeros(section_total, dtype=np.int64)
    section_population_totals = np.ones(section_total, dtype=np.float64)

    def add_position(section: int, value: tuple[float, float]) -> None:
        index = min(max(0, int(section)), section_total - 1)
        observed_position_samples[index] += 1
        seen = int(observed_position_samples[index])
        values = observed_positions[index]
        if len(values) < maximum_positions_per_section:
            values.append(value)
            return
        replacement = int(rng.integers(0, seen))
        if replacement < maximum_positions_per_section:
            values[replacement] = value

    movement_speeds: list[list[float]] = [[] for _ in range(5)]
    movement_turn_rates: list[list[float]] = [[] for _ in range(5)]
    movement_samples = np.zeros(5, dtype=np.int64)
    populations: list[int] = []
    player_starts: list[tuple[float, float]] = []
    frame_intervals: list[float] = []
    action_counts = np.ones(5, dtype=np.float64)
    raw_respawn_delays: list[float] = []
    respawn_candidate_count = 0
    unmatched_appearance_count = 0
    legacy_recording_count = 0
    recorder_versions: list[str] = []
    contract_warnings: list[str] = []
    unvalidated_presence_paths: list[Path] = []

    for path in paths:
        archive = RecordingArchive(path)
        if not has_validated_presence(archive.manifest):
            if not allow_unvalidated_presence:
                raise ValueError(
                    f"{path}: authoritative world-model fitting requires a dynamically "
                    "validated presence field; re-record with recorder 1.10+ or use the "
                    "legacy override only for an evidence-backed diagnostic fit"
                )
            unvalidated_presence_paths.append(path)
        contract_warnings.extend(
            validate_recording_contract(
                archive,
                action_names=ACTION_NAMES,
                origin_native_x=map_data.origin_native_x,
                origin_native_z=map_data.origin_native_z,
                native_units_per_cell=map_data.native_units_per_cell,
            )
        )
        recorder_versions.append(str(archive.manifest.get("recorder_version", "unknown")))
        saw_legacy_spawn = False
        for event in archive.events():
            if event.kind == "respawn_candidate":
                respawn_candidate_count += 1
                if len(event.values) > 11:
                    delay_ms = int(event.values[11])
                    if delay_ms >= 0:
                        raw_respawn_delays.append(delay_ms / 1000.0)
            elif event.kind == "target_appearance":
                unmatched_appearance_count += 1
            elif event.kind == "spawn":
                saw_legacy_spawn = True
                delay_ms = int(event.values[8]) if len(event.values) > 8 else -1
                if delay_ms >= 0:
                    respawn_candidate_count += 1
                    raw_respawn_delays.append(delay_ms / 1000.0)
                else:
                    unmatched_appearance_count += 1
        legacy_recording_count += int(saw_legacy_spawn)

        previous = None
        captured_start = False
        for frame in archive.frames():
            if frame.phase == 1 and frame.focused:
                populations.append(int(frame.living_monsters))
                if not captured_start:
                    player_starts.append((frame.player_x, frame.player_z))
                    captured_start = True
                if 0 <= frame.action < 5:
                    action_counts[frame.action] += 1.0
                for actor in frame.actors:
                    if not actor.living:
                        continue
                    section = map_data.section(actor.x, actor.z, section_count=section_count)
                    section_population_totals[section] += 1.0
                    add_position(section, (actor.x, actor.z))

            if previous is not None:
                dt = (frame.elapsed_ms - previous.elapsed_ms) / 1000.0
                if 0.02 <= dt <= 2.0:
                    frame_intervals.append(float(dt))
                before_action = movement_action_from_mask(previous.key_mask)
                after_action = movement_action_from_mask(frame.key_mask)
                if (
                    previous.phase == 1
                    and frame.phase == 1
                    and previous.focused
                    and frame.focused
                    and before_action is not None
                    and before_action == after_action
                    and 0.10 <= dt <= 1.50
                ):
                    distance = math.hypot(
                        frame.player_x - previous.player_x,
                        frame.player_z - previous.player_z,
                    ) / map_data.native_units_per_cell
                    heading_delta = math.atan2(
                        math.sin(frame.heading_radians - previous.heading_radians),
                        math.cos(frame.heading_radians - previous.heading_radians),
                    )
                    speed = distance / dt
                    turn_rate = heading_delta / dt
                    if 0.0 <= speed <= 30.0 and math.isfinite(turn_rate):
                        movement_samples[before_action] += 1
                        movement_speeds[before_action].append(float(speed))
                        movement_turn_rates[before_action].append(float(turn_rate))
            previous = frame

    section_probabilities = section_population_totals / section_population_totals.sum()
    transition = np.tile(section_probabilities, (section_total, 1))
    respawn_delays = _robust_delay_samples(raw_respawn_delays)

    default_distances = [2.2, 1.7, 1.7, 0.0, 2.2]
    default_turns = [0.0, 0.10, -0.10, 0.0, 0.0]
    movement_models: list[MovementModel] = []
    for action in range(5):
        if action == 3:
            movement_models.append(MovementModel(0, 0.0, 0.0, 0.0, 0.0))
            continue
        speeds = np.asarray(movement_speeds[action], dtype=np.float64)
        rates = np.asarray(movement_turn_rates[action], dtype=np.float64)
        if speeds.size:
            speeds = speeds[(speeds >= 0.0) & (speeds <= 30.0)]
        distances = speeds * float(control_interval_seconds)
        if action == 1:
            expected_rates = rates[rates > 0.05]
        elif action == 2:
            expected_rates = rates[rates < -0.05]
        else:
            expected_rates = np.asarray([], dtype=np.float64)
        turns = expected_rates * float(control_interval_seconds)
        if action == 4 and distances.size < 5:
            source = np.asarray(movement_speeds[0], dtype=np.float64)
            distances = source * float(control_interval_seconds)
        mean_distance = float(np.median(distances)) if distances.size else default_distances[action]
        std_distance = float(np.std(distances)) if distances.size > 1 else max(0.10, mean_distance * 0.15)
        if action in (1, 2):
            mean_turn = float(np.median(turns)) if turns.size else default_turns[action]
            std_turn = float(np.std(turns)) if turns.size > 1 else max(0.02, abs(mean_turn) * 0.25)
        else:
            mean_turn = 0.0
            std_turn = 0.01
        movement_models.append(
            MovementModel(
                samples=int(movement_samples[action]),
                distance_mean_cells=max(0.0, mean_distance),
                distance_std_cells=max(0.01, std_distance),
                turn_mean_radians=mean_turn,
                turn_std_radians=max(0.005, std_turn),
            )
        )

    if not player_starts:
        cell = map_data.random_safe_cell(rng)
        player_starts.append(map_data.layout_to_native(*cell))
    all_positions = [position for section in observed_positions for position in section]
    if not all_positions:
        for _ in range(1_000):
            all_positions.append(map_data.layout_to_native(*map_data.random_safe_cell(rng)))
    for section in range(section_total):
        if not observed_positions[section]:
            observed_positions[section] = list(all_positions[: min(len(all_positions), 1_000)])

    population = int(round(median(populations) * population_scale)) if populations else min(500, len(all_positions))
    population = max(1, min(5_000, population))
    recording_frame_interval = float(np.median(frame_intervals)) if frame_intervals else control_interval_seconds
    human_actions = action_counts / action_counts.sum()

    warnings: list[str] = []
    warnings.extend(dict.fromkeys(contract_warnings))
    if unvalidated_presence_paths:
        warnings.append(
            "One or more archives lacked a dynamically validated presence field and "
            "were admitted only by explicit legacy override. Population, density, "
            "disappearance, and respawn estimates from this fit are not authoritative."
        )
    if len(paths) < 3:
        warnings.append(
            "The world model is fitted from fewer than three representative sessions; "
            "section density and movement distributions are provisional."
        )
    warnings.append(
        "Recorder archives do not contain a persistent monster identity. Same-slot "
        "reappearances are used only as an aggregate respawn-delay hint, never as a "
        "death-to-spawn pairing."
    )
    warnings.append(
        "Respawn destinations use the observed global section distribution. This "
        "models population redistribution without claiming a measured source-to-target map."
    )
    warnings.append(
        "Simulator resets use only the first focused farming position from each recording, "
        "because real sessions begin at the map spawn rather than at arbitrary route positions."
    )
    if any(version.startswith("1.7") or version.startswith("1.6") for version in recorder_versions):
        warnings.append(
            "At least one archive predates spawned-field filtering in frame snapshots. "
            "Its population count is an approximate map-wide actor population, not an "
            "exact simultaneously instantiated count."
        )

    return RecordedWorldModel(
        schema_version=4,
        source_recordings=tuple(str(path.resolve()) for path in paths),
        section_count=section_count,
        hub_section=section_count,
        population_median=population,
        section_population_probabilities=tuple(float(value) for value in section_probabilities),
        player_start_positions=tuple(player_starts[:10_000]),
        spawn_positions_by_section=tuple(tuple(section) for section in observed_positions),
        transition_probabilities=tuple(tuple(float(value) for value in row) for row in transition),
        respawn_delay_seconds=tuple(respawn_delays),
        movement=tuple(movement_models),
        monster_speed_cells_per_second=0.0,
        frame_interval_seconds=float(control_interval_seconds),
        native_units_per_cell=map_data.native_units_per_cell,
        respawn_candidate_count=respawn_candidate_count,
        unmatched_appearance_count=unmatched_appearance_count,
        legacy_recording_count=legacy_recording_count,
        respawn_position_sample_counts=tuple(int(value) for value in observed_position_samples),
        recording_frame_interval_seconds=recording_frame_interval,
        cast_step_seconds=float(cast_step_seconds),
        cast_movement_seconds=float(cast_movement_seconds),
        respawn_model_mode="global_redistribution",
        respawn_delay_source="same_slot_aggregate_provisional",
        human_action_probabilities=tuple(float(value) for value in human_actions),
        fit_warnings=tuple(warnings),
    )
