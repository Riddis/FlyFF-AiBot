from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from json import dumps
from math import cos, hypot, isfinite, sin
from numbers import Integral
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .actions import FarmingAction, coerce_farming_action
from .map_features import (
    DEFAULT_MAXIMUM_GEODESIC_EXPANSIONS,
    DEFAULT_TELEPORT_BUFFER_RADIUS_CELLS,
    DirectPathState,
)

FloatArray = NDArray[np.float32]

OBSERVATION_SCHEMA_ID: Final = "native-unified-482-v1"
LEGACY_ACTOR_SLOTS: Final = 32
ACTOR_FEATURES: Final = 7
LEGACY_AGGREGATE_FEATURES: Final = 5
UNIFIED_STATE_FEATURES: Final = 16
LOCAL_MAP_SIDE: Final = 11
DIRECT_ACTOR_SLOTS: Final = 12
LEGACY_NEAREST_QUOTA: Final = 8
LEGACY_UTILITY_DISTANCE_WEIGHT: Final = 0.045

LEGACY_ACTOR_START: Final = 0
LEGACY_ACTOR_STOP: Final = 224
LEGACY_MASK_START: Final = 224
LEGACY_MASK_STOP: Final = 256
LEGACY_AGGREGATE_START: Final = 256
LEGACY_AGGREGATE_STOP: Final = 261
UNIFIED_STATE_START: Final = 261
UNIFIED_STATE_STOP: Final = 277
LOCAL_MAP_START: Final = 277
LOCAL_MAP_STOP: Final = 398
DIRECT_ACTOR_START: Final = 398
DIRECT_ACTOR_STOP: Final = 482
OBSERVATION_SIZE: Final = DIRECT_ACTOR_STOP

_LEGACY_ACTOR_FIELD_NAMES: Final = (
    "dx_over_vision",
    "dy_over_vision",
    "euclidean_over_vision",
    "geodesic_over_vision",
    "nearby_over_density_scale",
    "within_eva",
    "active",
)
_LEGACY_AGGREGATE_FIELD_NAMES: Final = (
    "visible_over_scale",
    "eva_count_over_scale",
    "best_nearby_over_scale",
    "eva_cooldown_fraction",
    "has_selected_actor",
)
_UNIFIED_STATE_FIELD_NAMES: Final = (
    "player_normalized_x",
    "player_normalized_z",
    "heading_sin",
    "heading_cos",
    "eva_cooldown_bipolar",
    "eva_actor_count_bipolar",
    "visible_actor_count_bipolar",
    "displacement_bipolar",
    "contact_bipolar",
    "held_forward_bipolar",
    "held_forward_left_bipolar",
    "held_forward_right_bipolar",
    "last_action_cast_bipolar",
    "time_since_cast_bipolar",
    "direct_clear_fraction_bipolar",
    "map_available_bipolar",
)
_DIRECT_ACTOR_FIELD_NAMES: Final = (
    "dx_over_vision",
    "dz_over_vision",
    "distance_bipolar",
    "active",
    "within_eva_bipolar",
    "direct_path_state",
    "pack_density_bipolar",
)


@dataclass(frozen=True, slots=True)
class ObservationScales:
    """Normalization constants that are part of the semantic model contract."""

    vision_radius_cells: float = 50.0
    eva_radius_cells: float = 8.0
    legacy_density_count_scale: float = 40.0
    legacy_visible_count_scale: float = 200.0
    maximum_eva_count: float = 32.0
    maximum_visible_count: float = 256.0
    maximum_pack_density: float = 24.0
    maximum_displacement_cells: float = 4.0
    blocked_actor_maximum_distance_cells: float = 20.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            numeric = float(value)
            if not isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class ActorObservation:
    """One actor with both coordinate frames required by the active model.

    Legacy features use layout-space deltas, whose second axis follows the map
    array row direction.  Direct features preserve native X/Z deltas after
    conversion to cells.  Keeping both prevents a non-zero native Z delta from
    being silently reused with the layout-space sign convention.
    """

    actor_id: int
    legacy_dx_cells: float
    legacy_dy_cells: float
    direct_dx_cells: float
    direct_dz_cells: float
    geodesic_cells: float = float("inf")
    direct_path: DirectPathState = DirectPathState.UNKNOWN
    alive: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.actor_id, bool) or not isinstance(self.actor_id, Integral):
            raise ValueError("actor_id must be an integer and cannot be boolean")
        coordinate_values = {
            "legacy_dx_cells": self.legacy_dx_cells,
            "legacy_dy_cells": self.legacy_dy_cells,
            "direct_dx_cells": self.direct_dx_cells,
            "direct_dz_cells": self.direct_dz_cells,
        }
        for name, value in coordinate_values.items():
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        geodesic = float(self.geodesic_cells)
        if np.isnan(geodesic) or geodesic < 0.0:
            raise ValueError("geodesic_cells must be non-negative or infinity")
        if isinstance(self.direct_path, bool):
            raise ValueError("direct_path cannot be boolean")
        try:
            direct_path = DirectPathState(self.direct_path)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"direct_path must be a DirectPathState, got {self.direct_path!r}"
            ) from error
        object.__setattr__(self, "actor_id", int(self.actor_id))
        for name, value in coordinate_values.items():
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, "geodesic_cells", geodesic)
        object.__setattr__(self, "direct_path", direct_path)

    @property
    def legacy_euclidean_cells(self) -> float:
        return hypot(self.legacy_dx_cells, self.legacy_dy_cells)

    @property
    def direct_euclidean_cells(self) -> float:
        return hypot(self.direct_dx_cells, self.direct_dz_cells)


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    """Player/control state used by the 16 unified state fields."""

    normalized_x: float
    normalized_z: float
    heading_radians: float
    eva_cooldown_fraction: float
    displacement_cells: float
    contact: bool
    held_movement: FarmingAction | None
    last_policy_action: FarmingAction
    time_since_cast_fraction: float
    map_available: bool

    def __post_init__(self) -> None:
        finite_values = {
            "normalized_x": self.normalized_x,
            "normalized_z": self.normalized_z,
            "heading_radians": self.heading_radians,
            "eva_cooldown_fraction": self.eva_cooldown_fraction,
            "displacement_cells": self.displacement_cells,
            "time_since_cast_fraction": self.time_since_cast_fraction,
        }
        for name, value in finite_values.items():
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.displacement_cells < 0.0:
            raise ValueError("displacement_cells cannot be negative")

        held = self.held_movement
        if held is not None:
            held = coerce_farming_action(held)
            if not held.is_movement:
                raise ValueError("held_movement cannot be CAST_EVA")
        object.__setattr__(self, "held_movement", held)
        object.__setattr__(
            self,
            "last_policy_action",
            coerce_farming_action(self.last_policy_action),
        )


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    """All inputs for one vector, acquired from one coherent step snapshot."""

    player: PlayerObservation
    actors: tuple[ActorObservation, ...]
    local_map: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "actors", tuple(self.actors))


@dataclass(frozen=True, slots=True)
class BuiltObservation:
    vector: FloatArray
    legacy_actor_ids: tuple[int, ...]
    direct_actor_ids: tuple[int, ...]
    visible_actor_count: int
    eva_actor_count: int
    direct_clear_fraction: float


def _build_observation_fields() -> tuple[str, ...]:
    fields: list[str] = []
    for slot in range(LEGACY_ACTOR_SLOTS):
        fields.extend(
            f"legacy_actor[{slot:02d}].{name}" for name in _LEGACY_ACTOR_FIELD_NAMES
        )
    fields.extend(f"legacy_mask[{slot:02d}]" for slot in range(LEGACY_ACTOR_SLOTS))
    fields.extend(f"legacy_aggregate.{name}" for name in _LEGACY_AGGREGATE_FIELD_NAMES)
    fields.extend(f"unified_state.{name}" for name in _UNIFIED_STATE_FIELD_NAMES)
    radius = LOCAL_MAP_SIDE // 2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            fields.append(f"local_map[dy={dy:+d},dx={dx:+d}]")
    for slot in range(DIRECT_ACTOR_SLOTS):
        fields.extend(
            f"direct_actor[{slot:02d}].{name}" for name in _DIRECT_ACTOR_FIELD_NAMES
        )
    if len(fields) != OBSERVATION_SIZE:
        raise RuntimeError(
            f"Observation field contract has {len(fields)} fields, "
            f"expected {OBSERVATION_SIZE}"
        )
    return tuple(fields)


OBSERVATION_FIELDS: Final = _build_observation_fields()


def observation_schema_descriptor(
    scales: ObservationScales | None = None,
) -> dict[str, object]:
    selected_scales = scales or ObservationScales()
    return {
        "schema_id": OBSERVATION_SCHEMA_ID,
        "dtype": "float32",
        "size": OBSERVATION_SIZE,
        "fields": OBSERVATION_FIELDS,
        "scales": asdict(selected_scales),
        "coordinate_provenance": {
            "player_position": {
                "fields": ("player_normalized_x", "player_normalized_z"),
                "source": "player layout cell normalized independently to [-1,1]",
                "second_axis": "layout array y/row, not raw native Z",
            },
            "legacy_actor_offsets": {
                "source": "actor layout cell minus player layout cell",
                "axes": ("layout_x", "layout_y"),
                "distance": "hypot(legacy_dx_cells,legacy_dy_cells)",
                "consumers": (
                    "legacy actor dx/dy/euclidean/within_eva",
                    "legacy nearby density",
                    "legacy eva aggregate",
                ),
            },
            "direct_actor_offsets": {
                "source": (
                    "(actor native X/Z minus player native X/Z) divided by "
                    "native_units_per_cell"
                ),
                "axes": ("native_x", "native_z"),
                "distance": "hypot(direct_dx_cells,direct_dz_cells)",
                "consumers": (
                    "direct actor dx/dz/distance/within_eva",
                    "distant-blocked eligibility",
                    "direct pack density",
                ),
            },
            "legacy_geodesic": {
                "source": "safe-traversable layout cells",
                "connectivity": 8,
                "diagonal_corner_cutting": False,
                "default_maximum_expansions": DEFAULT_MAXIMUM_GEODESIC_EXPANSIONS,
                "non_finite_effect": "excluded from legacy selection only",
            },
            "direct_path": {
                "source": "Bresenham player-to-actor layout-cell segment",
                "inspected_cells": "interior only; player and actor endpoints ignored",
                "blocked": "any interior cell outside safe_traversable",
                "unknown": "missing/outside endpoint or unknown interior evidence",
            },
        },
        "actor_populations": {
            "live": "actors with alive=True; actor_id is a unique integer",
            "legacy_density": (
                "all live actors, including unreachable actors and the actor itself"
            ),
            "legacy_selection": "live actors with finite geodesic_cells",
            "direct_eligible": (
                "all live actors except BLOCKED actors whose direct-native distance "
                "is greater than blocked_actor_maximum_distance_cells"
            ),
            "direct_density": (
                "all direct-eligible actors, including unselected actors and self"
            ),
            "visible_count": "all live actors",
            "eva_count": "all live actors within inclusive legacy-layout EVA radius",
        },
        "actor_selection": {
            "legacy_slots": LEGACY_ACTOR_SLOTS,
            "legacy_nearest_quota": LEGACY_NEAREST_QUOTA,
            "legacy_nearest_order": (
                "geodesic_cells ascending",
                "nearby_count descending",
                "actor_id ascending",
            ),
            "legacy_utility": (
                "nearby_count - legacy_utility_distance_weight * geodesic_cells"
            ),
            "legacy_utility_distance_weight": LEGACY_UTILITY_DISTANCE_WEIGHT,
            "legacy_utility_order": (
                "utility descending",
                "geodesic_cells ascending",
                "actor_id ascending",
            ),
            "legacy_merge": "nearest quota first, then utility order, de-duplicated",
            "direct_slots": DIRECT_ACTOR_SLOTS,
            "direct_path_rank": (
                DirectPathState.CLEAR.name,
                DirectPathState.UNKNOWN.name,
                DirectPathState.BLOCKED.name,
            ),
            "direct_order": (
                "direct_path_rank ascending",
                "direct-native Euclidean distance ascending",
                "actor_id ascending",
            ),
            "direct_clear_fraction": (
                "CLEAR selected count divided by max(1, selected count)"
            ),
        },
        "density_semantics": {
            "neighbor_boundary": "inclusive <= eva_radius_cells",
            "self_counted": True,
            "legacy_coordinates": "legacy layout offsets",
            "legacy_population": "legacy_density",
            "direct_coordinates": "direct native X/Z offsets in cells",
            "direct_population": "direct_density",
        },
        "local_map_contract": {
            "side": LOCAL_MAP_SIDE,
            "flattening": "row-major; dy outer from -5 to +5, dx inner from -5 to +5",
            "center": "player layout cell",
            "provider": "FarmingMapFeatures.local_crop",
            "default_teleport_buffer_radius_cells": (
                DEFAULT_TELEPORT_BUFFER_RADIUS_CELLS
            ),
            "precedence": (
                "outside_or_unknown",
                "exact forbidden trigger",
                "non-trigger cell within inclusive teleport buffer",
                "ordinary blocked",
                "safe traversable",
            ),
            "input_validation": "finite float32-compatible values already in [-1,1]",
        },
        "numeric_semantics": {
            "normalization_clipping": "unit [0,1], bipolar [-1,1]",
            "inactive_actor_records": "all zero",
            "legacy_mask": "1 selected, 0 inactive",
            "actor_distance_boundaries": "inclusive",
            "heading": "sin then cos of coherent native heading radians",
            "held_movement": "three bipolar one-hot fields; CAST_EVA holds prior movement",
            "booleans": "bipolar except legacy within_eva/active and legacy mask",
        },
        "encodings": {
            "legacy_inactive_record": 0.0,
            "direct_inactive_record": 0.0,
            "bipolar_false": -1.0,
            "bipolar_true": 1.0,
            "direct_path_blocked": int(DirectPathState.BLOCKED),
            "direct_path_unknown": int(DirectPathState.UNKNOWN),
            "direct_path_clear": int(DirectPathState.CLEAR),
            "local_map": {
                "safe": -1.0,
                "outside_or_unknown": 0.0,
                "ordinary_blocked": 0.25,
                "teleport_buffer": 0.75,
                "teleport_trigger": 1.0,
            },
        },
    }


def observation_schema_hash(scales: ObservationScales | None = None) -> str:
    payload = dumps(
        observation_schema_descriptor(scales),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest().upper()


OBSERVATION_SCHEMA_HASH: Final = (
    "7B8E1FC27E67CD5ECF200382DD1644DF9B253FCB654AA9F11413694F97C3DC15"
)


class ObservationBuilder:
    """Build the exact ``native-unified-482-v1`` vector from typed inputs."""

    def __init__(self, scales: ObservationScales | None = None) -> None:
        self.scales = scales or ObservationScales()

    @property
    def observation_size(self) -> int:
        return OBSERVATION_SIZE

    @property
    def schema_id(self) -> str:
        return OBSERVATION_SCHEMA_ID

    @property
    def schema_hash(self) -> str:
        return observation_schema_hash(self.scales)

    def build(self, frame: ObservationFrame) -> BuiltObservation:
        actors = tuple(actor for actor in frame.actors if actor.alive)
        actor_ids = [actor.actor_id for actor in actors]
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("actor_id values must be unique within one observation")

        local_map = np.asarray(frame.local_map, dtype=np.float32).reshape(-1)
        expected_local_size = LOCAL_MAP_SIDE * LOCAL_MAP_SIDE
        if local_map.size != expected_local_size:
            raise ValueError(
                f"local_map must contain {expected_local_size} values, "
                f"got {local_map.size}"
            )
        if not np.all(np.isfinite(local_map)):
            raise ValueError("local_map values must be finite")
        if np.any((local_map < -1.0) | (local_map > 1.0)):
            raise ValueError("local_map values must already be within [-1, 1]")
        local_map = np.ascontiguousarray(local_map, dtype=np.float32)

        legacy_nearby_counts = self._legacy_nearby_counts(actors)
        direct_eligible = self._direct_eligible_actors(actors)
        direct_nearby_counts = self._direct_nearby_counts(direct_eligible)
        legacy = self._select_legacy_actors(actors, legacy_nearby_counts)
        direct = self._select_direct_actors(direct_eligible)
        eva_count = sum(
            actor.legacy_euclidean_cells <= self.scales.eva_radius_cells
            for actor in actors
        )
        direct_clear_count = sum(
            actor.direct_path is DirectPathState.CLEAR for actor in direct
        )
        direct_clear_fraction = direct_clear_count / max(1, len(direct))

        legacy_values = self._legacy_actor_values(legacy, legacy_nearby_counts)
        legacy_mask = np.zeros(LEGACY_ACTOR_SLOTS, dtype=np.float32)
        legacy_mask[: len(legacy)] = 1.0
        best_nearby = max(
            (legacy_nearby_counts[actor.actor_id] for actor in legacy),
            default=0,
        )
        legacy_aggregate = np.asarray(
            (
                _unit(len(actors), self.scales.legacy_visible_count_scale),
                _unit(eva_count, self.scales.legacy_density_count_scale),
                _unit(best_nearby, self.scales.legacy_density_count_scale),
                _clip_unit(frame.player.eva_cooldown_fraction),
                1.0 if legacy else 0.0,
            ),
            dtype=np.float32,
        )
        state = self._unified_state(
            frame.player,
            visible_count=len(actors),
            eva_count=eva_count,
            direct_clear_fraction=direct_clear_fraction,
        )
        direct_values = self._direct_actor_values(direct, direct_nearby_counts)

        vector = np.concatenate(
            (
                legacy_values.reshape(-1),
                legacy_mask,
                legacy_aggregate,
                state,
                local_map,
                direct_values.reshape(-1),
            )
        ).astype(np.float32, copy=False)
        if vector.shape != (OBSERVATION_SIZE,):
            raise RuntimeError(
                f"Observation shape {vector.shape} does not match "
                f"{OBSERVATION_SCHEMA_ID} ({OBSERVATION_SIZE},)"
            )
        if not np.all(np.isfinite(vector)):
            raise RuntimeError("Observation contains a non-finite value")
        return BuiltObservation(
            vector=np.ascontiguousarray(vector),
            legacy_actor_ids=tuple(actor.actor_id for actor in legacy),
            direct_actor_ids=tuple(actor.actor_id for actor in direct),
            visible_actor_count=len(actors),
            eva_actor_count=int(eva_count),
            direct_clear_fraction=float(direct_clear_fraction),
        )

    def build_vector(self, frame: ObservationFrame) -> FloatArray:
        return self.build(frame).vector

    def _legacy_nearby_counts(
        self,
        actors: tuple[ActorObservation, ...],
    ) -> dict[int, int]:
        positions = {
            actor.actor_id: (actor.legacy_dx_cells, actor.legacy_dy_cells)
            for actor in actors
        }
        return self._nearby_counts(positions)

    def _direct_nearby_counts(
        self,
        actors: tuple[ActorObservation, ...],
    ) -> dict[int, int]:
        positions = {
            actor.actor_id: (actor.direct_dx_cells, actor.direct_dz_cells)
            for actor in actors
        }
        return self._nearby_counts(positions)

    def _nearby_counts(
        self,
        positions: dict[int, tuple[float, float]],
    ) -> dict[int, int]:
        radius = self.scales.eva_radius_cells
        return {
            actor_id: sum(
                hypot(other_x - actor_x, other_y - actor_y) <= radius
                for other_x, other_y in positions.values()
            )
            for actor_id, (actor_x, actor_y) in positions.items()
        }

    def _select_legacy_actors(
        self,
        actors: tuple[ActorObservation, ...],
        nearby_counts: dict[int, int],
    ) -> tuple[ActorObservation, ...]:
        reachable = tuple(actor for actor in actors if isfinite(actor.geodesic_cells))
        by_distance = sorted(
            reachable,
            key=lambda actor: (
                actor.geodesic_cells,
                -nearby_counts[actor.actor_id],
                actor.actor_id,
            ),
        )
        by_utility = sorted(
            reachable,
            key=lambda actor: (
                -(
                    nearby_counts[actor.actor_id]
                    - LEGACY_UTILITY_DISTANCE_WEIGHT * actor.geodesic_cells
                ),
                actor.geodesic_cells,
                actor.actor_id,
            ),
        )
        selected: list[ActorObservation] = []
        seen: set[int] = set()
        for source in (by_distance[:LEGACY_NEAREST_QUOTA], by_utility):
            for actor in source:
                if actor.actor_id in seen:
                    continue
                selected.append(actor)
                seen.add(actor.actor_id)
                if len(selected) >= LEGACY_ACTOR_SLOTS:
                    return tuple(selected)
        return tuple(selected)

    def _direct_eligible_actors(
        self,
        actors: tuple[ActorObservation, ...],
    ) -> tuple[ActorObservation, ...]:
        return tuple(
            actor
            for actor in actors
            if not (
                actor.direct_path is DirectPathState.BLOCKED
                and actor.direct_euclidean_cells
                > self.scales.blocked_actor_maximum_distance_cells
            )
        )

    def _select_direct_actors(
        self,
        eligible: tuple[ActorObservation, ...],
    ) -> tuple[ActorObservation, ...]:
        ranks = {
            DirectPathState.CLEAR: 0,
            DirectPathState.UNKNOWN: 1,
            DirectPathState.BLOCKED: 2,
        }
        return tuple(
            sorted(
                eligible,
                key=lambda actor: (
                    ranks[actor.direct_path],
                    actor.direct_euclidean_cells,
                    actor.actor_id,
                ),
            )[:DIRECT_ACTOR_SLOTS]
        )

    def _legacy_actor_values(
        self,
        actors: tuple[ActorObservation, ...],
        nearby_counts: dict[int, int],
    ) -> FloatArray:
        values = np.zeros(
            (LEGACY_ACTOR_SLOTS, ACTOR_FEATURES),
            dtype=np.float32,
        )
        vision = self.scales.vision_radius_cells
        density = self.scales.legacy_density_count_scale
        eva = self.scales.eva_radius_cells
        for slot, actor in enumerate(actors):
            values[slot] = (
                np.clip(actor.legacy_dx_cells / vision, -1.0, 1.0),
                np.clip(actor.legacy_dy_cells / vision, -1.0, 1.0),
                _unit(actor.legacy_euclidean_cells, vision),
                _unit(actor.geodesic_cells, vision),
                _unit(nearby_counts[actor.actor_id], density),
                1.0 if actor.legacy_euclidean_cells <= eva else 0.0,
                1.0,
            )
        return values

    def _unified_state(
        self,
        player: PlayerObservation,
        *,
        visible_count: int,
        eva_count: int,
        direct_clear_fraction: float,
    ) -> FloatArray:
        held = (
            1.0 if player.held_movement is FarmingAction.RUN_FORWARD else -1.0,
            1.0 if player.held_movement is FarmingAction.RUN_FORWARD_LEFT else -1.0,
            1.0 if player.held_movement is FarmingAction.RUN_FORWARD_RIGHT else -1.0,
        )
        return np.asarray(
            (
                np.clip(player.normalized_x, -1.0, 1.0),
                np.clip(player.normalized_z, -1.0, 1.0),
                sin(player.heading_radians),
                cos(player.heading_radians),
                _bipolar_fraction(player.eva_cooldown_fraction),
                _bipolar_unit(eva_count, self.scales.maximum_eva_count),
                _bipolar_unit(
                    visible_count,
                    self.scales.maximum_visible_count,
                ),
                _bipolar_unit(
                    player.displacement_cells,
                    self.scales.maximum_displacement_cells,
                ),
                1.0 if player.contact else -1.0,
                held[0],
                held[1],
                held[2],
                1.0 if player.last_policy_action is FarmingAction.CAST_EVA else -1.0,
                _bipolar_fraction(player.time_since_cast_fraction),
                _bipolar_fraction(direct_clear_fraction),
                1.0 if player.map_available else -1.0,
            ),
            dtype=np.float32,
        )

    def _direct_actor_values(
        self,
        actors: tuple[ActorObservation, ...],
        nearby_counts: dict[int, int],
    ) -> FloatArray:
        values = np.zeros(
            (DIRECT_ACTOR_SLOTS, ACTOR_FEATURES),
            dtype=np.float32,
        )
        vision = self.scales.vision_radius_cells
        eva = self.scales.eva_radius_cells
        density = self.scales.maximum_pack_density
        for slot, actor in enumerate(actors):
            values[slot] = (
                np.clip(actor.direct_dx_cells / vision, -1.0, 1.0),
                np.clip(actor.direct_dz_cells / vision, -1.0, 1.0),
                _bipolar_unit(actor.direct_euclidean_cells, vision),
                1.0,
                1.0 if actor.direct_euclidean_cells <= eva else -1.0,
                float(actor.direct_path),
                _bipolar_unit(nearby_counts[actor.actor_id], density),
            )
        return values


def _clip_unit(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _unit(value: float | int, scale: float) -> float:
    return float(np.clip(float(value) / float(scale), 0.0, 1.0))


def _bipolar_fraction(value: float) -> float:
    return _clip_unit(value) * 2.0 - 1.0


def _bipolar_unit(value: float | int, scale: float) -> float:
    return _unit(value, scale) * 2.0 - 1.0
