from __future__ import annotations

from hashlib import sha256
from math import pi

import numpy as np
import pytest
from farming.actions import FarmingAction
from farming.map_features import DirectPathState
from farming.observation import (
    CONTEXT_MAP_START,
    DIRECT_ACTOR_START,
    LEGACY_AGGREGATE_START,
    LEGACY_MASK_START,
    LOCAL_MAP_START,
    OBSERVATION_FIELDS,
    OBSERVATION_SCHEMA_HASH,
    OBSERVATION_SCHEMA_ID,
    OBSERVATION_SIZE,
    UNIFIED_STATE_START,
    ActorObservation,
    ObservationBuilder,
    ObservationFrame,
    ObservationScales,
    PlayerObservation,
    observation_schema_descriptor,
    observation_schema_hash,
)


def _frame() -> ObservationFrame:
    actors = (
        ActorObservation(
            actor_id=10,
            legacy_dx_cells=5.0,
            legacy_dy_cells=0.0,
            direct_dx_cells=5.0,
            direct_dz_cells=0.0,
            geodesic_cells=5.0,
            direct_path=DirectPathState.CLEAR,
        ),
        ActorObservation(
            actor_id=20,
            legacy_dx_cells=30.0,
            legacy_dy_cells=0.0,
            direct_dx_cells=30.0,
            direct_dz_cells=0.0,
            geodesic_cells=6.0,
            direct_path=DirectPathState.BLOCKED,
        ),
        ActorObservation(
            actor_id=30,
            legacy_dx_cells=3.0,
            legacy_dy_cells=0.0,
            direct_dx_cells=3.0,
            direct_dz_cells=0.0,
            direct_path=DirectPathState.UNKNOWN,
        ),
    )
    local_map = np.linspace(-1.0, 1.0, 121, dtype=np.float32).reshape(11, 11)
    context_map = np.linspace(-1.0, 1.0, 441, dtype=np.float32).reshape(21, 21)
    player = PlayerObservation(
        normalized_x=0.25,
        normalized_z=-0.5,
        heading_radians=pi / 2.0,
        eva_cooldown_fraction=0.5,
        displacement_cells=2.0,
        contact=True,
        held_movement=FarmingAction.RUN_FORWARD_LEFT,
        last_policy_action=FarmingAction.CAST_EVA,
        jump_cooldown_fraction=0.25,
        map_available=True,
    )
    return ObservationFrame(
        player=player,
        actors=actors,
        local_map=local_map,
        context_map=context_map,
    )


def test_observation_schema_freezes_every_segment_and_field_index() -> None:
    assert OBSERVATION_SCHEMA_ID == "native-unified-923-v4"
    assert len(OBSERVATION_FIELDS) == OBSERVATION_SIZE == 923
    assert OBSERVATION_FIELDS[0] == "legacy_actor[00].dx_over_vision"
    assert OBSERVATION_FIELDS[223] == "legacy_actor[31].active"
    assert OBSERVATION_FIELDS[224] == "legacy_mask[00]"
    assert OBSERVATION_FIELDS[255] == "legacy_mask[31]"
    assert OBSERVATION_FIELDS[256] == "legacy_aggregate.visible_over_scale"
    assert OBSERVATION_FIELDS[260] == "legacy_aggregate.has_selected_actor"
    assert OBSERVATION_FIELDS[261] == "unified_state.player_normalized_x"
    assert OBSERVATION_FIELDS[274] == "unified_state.jump_cooldown_bipolar"
    assert OBSERVATION_FIELDS[276] == "unified_state.map_available_bipolar"
    assert OBSERVATION_FIELDS[277] == "local_map[dy=-5,dx=-5]"
    assert OBSERVATION_FIELDS[397] == "local_map[dy=+5,dx=+5]"
    assert OBSERVATION_FIELDS[398] == "context_map[gy=-10,gx=-10]"
    assert OBSERVATION_FIELDS[838] == "context_map[gy=+10,gx=+10]"
    assert OBSERVATION_FIELDS[839] == "direct_actor[00].dx_over_vision"
    assert OBSERVATION_FIELDS[840] == "direct_actor[00].dz_over_vision"
    assert OBSERVATION_FIELDS[922] == "direct_actor[11].pack_density_bipolar"


def test_schema_hash_covers_field_order_and_normalization_scales() -> None:
    assert len(OBSERVATION_SCHEMA_HASH) == 64
    assert OBSERVATION_SCHEMA_HASH == observation_schema_hash()
    assert OBSERVATION_SCHEMA_HASH != observation_schema_hash(
        ObservationScales(vision_radius_cells=60.0)
    )
    descriptor = observation_schema_descriptor()
    assert descriptor["coordinate_provenance"]["legacy_actor_offsets"][  # type: ignore[index]
        "axes"
    ] == ("layout_x", "layout_y")
    assert descriptor["coordinate_provenance"]["direct_actor_offsets"][  # type: ignore[index]
        "axes"
    ] == ("native_x", "native_z")
    assert descriptor["actor_populations"]["direct_density"] == (  # type: ignore[index]
        "all direct-eligible actors, including unselected actors and self"
    )
    assert descriptor["encodings"]["local_map"] == {  # type: ignore[index]
        "safe": -1.0,
        "obstacle_buffer": -0.25,
        "outside_or_unknown": 0.0,
        "obstacle": 0.5,
        "teleport_buffer": 0.75,
        "teleport_trigger": 1.0,
    }
    assert descriptor["context_map_contract"]["side"] == 21  # type: ignore[index]
    assert descriptor["context_map_contract"]["radius_cells"] == 50  # type: ignore[index]


def test_builder_emits_exact_923_vector_from_one_typed_frame() -> None:
    built = ObservationBuilder().build(_frame())
    vector = built.vector

    assert vector.shape == (923,)
    assert vector.dtype == np.float32
    assert built.visible_actor_count == 3
    assert built.eva_actor_count == 2
    assert built.legacy_actor_ids == (10, 20)
    assert built.direct_actor_ids == (10, 30)
    assert built.direct_clear_fraction == pytest.approx(0.5)

    assert vector[0:7] == pytest.approx((0.1, 0.0, 0.1, 0.1, 0.05, 1.0, 1.0))
    assert vector[LEGACY_MASK_START : LEGACY_MASK_START + 3] == pytest.approx(
        (1.0, 1.0, 0.0)
    )
    assert vector[LEGACY_AGGREGATE_START : LEGACY_AGGREGATE_START + 5] == pytest.approx(
        (3 / 200, 2 / 40, 2 / 40, 0.5, 1.0)
    )

    state = vector[UNIFIED_STATE_START : UNIFIED_STATE_START + 16]
    assert state == pytest.approx(
        (
            0.25,
            -0.5,
            1.0,
            0.0,
            0.0,
            -0.875,
            3 / 256 * 2 - 1,
            0.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -0.5,
            0.0,
            1.0,
        ),
        abs=1.0e-6,
    )
    assert vector[LOCAL_MAP_START : LOCAL_MAP_START + 121] == pytest.approx(
        _frame().local_map.reshape(-1)
    )
    assert vector[CONTEXT_MAP_START : CONTEXT_MAP_START + 441] == pytest.approx(
        _frame().context_map.reshape(-1)
    )
    assert vector[DIRECT_ACTOR_START : DIRECT_ACTOR_START + 7] == pytest.approx(
        (
            0.1,
            0.0,
            -0.8,
            1.0,
            1.0,
            1.0,
            2 / 24 * 2 - 1,
        )
    )
    assert np.all(vector[DIRECT_ACTOR_START + 14 : DIRECT_ACTOR_START + 21] == 0.0)


def test_builder_rejects_ambiguous_actor_identity_and_wrong_local_crop() -> None:
    frame = _frame()
    duplicate = ObservationFrame(
        player=frame.player,
        actors=(frame.actors[0], frame.actors[0]),
        local_map=frame.local_map,
        context_map=frame.context_map,
    )
    with pytest.raises(ValueError, match="unique"):
        ObservationBuilder().build(duplicate)

    malformed = ObservationFrame(
        player=frame.player,
        actors=frame.actors,
        local_map=np.zeros((3, 3), dtype=np.float32),
        context_map=frame.context_map,
    )
    with pytest.raises(ValueError, match="121"):
        ObservationBuilder().build(malformed)

    out_of_contract = ObservationFrame(
        player=frame.player,
        actors=frame.actors,
        local_map=np.full((11, 11), 1.01, dtype=np.float32),
        context_map=frame.context_map,
    )
    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        ObservationBuilder().build(out_of_contract)

    malformed_context = ObservationFrame(
        player=frame.player,
        actors=frame.actors,
        local_map=frame.local_map,
        context_map=np.zeros((3, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="441"):
        ObservationBuilder().build(malformed_context)


@pytest.mark.parametrize("actor_id", [True, 1.5])
def test_actor_identity_rejects_boolean_and_fractional_values(actor_id: object) -> None:
    with pytest.raises(ValueError, match="actor_id must be an integer"):
        ActorObservation(
            actor_id=actor_id,  # type: ignore[arg-type]
            legacy_dx_cells=0.0,
            legacy_dy_cells=0.0,
            direct_dx_cells=0.0,
            direct_dz_cells=0.0,
        )


def test_direct_density_uses_all_eligible_actors_after_blocked_filtering() -> None:
    actors = tuple(
        ActorObservation(
            actor_id=index,
            legacy_dx_cells=0.0,
            legacy_dy_cells=0.0,
            direct_dx_cells=25.0,
            direct_dz_cells=0.0,
            geodesic_cells=1.0,
            direct_path=DirectPathState.CLEAR,
        )
        for index in range(13)
    ) + (
        ActorObservation(
            actor_id=99,
            legacy_dx_cells=0.0,
            legacy_dy_cells=0.0,
            direct_dx_cells=26.0,
            direct_dz_cells=0.0,
            geodesic_cells=1.0,
            direct_path=DirectPathState.BLOCKED,
        ),
    )
    frame = ObservationFrame(
        player=_frame().player,
        actors=actors,
        local_map=np.zeros((11, 11), dtype=np.float32),
        context_map=np.zeros((21, 21), dtype=np.float32),
    )

    built = ObservationBuilder().build(frame)

    assert built.direct_actor_ids == tuple(range(12))
    assert 99 not in built.direct_actor_ids
    assert built.vector[4] == pytest.approx(14 / 40)
    assert built.vector[DIRECT_ACTOR_START + 6] == pytest.approx(13 / 24 * 2 - 1)


def test_nonzero_native_z_uses_distinct_layout_and_direct_offsets_full_golden() -> None:
    local_map = np.resize(
        np.asarray((-1.0, -0.25, 0.50, 0.75, 1.0, 0.0), dtype=np.float32),
        121,
    ).reshape(11, 11)
    actor = ActorObservation(
        actor_id=77,
        legacy_dx_cells=6.0,
        legacy_dy_cells=-8.0,
        direct_dx_cells=3.0,
        direct_dz_cells=4.0,
        geodesic_cells=12.5,
        direct_path=DirectPathState.CLEAR,
    )
    player = PlayerObservation(
        normalized_x=-0.25,
        normalized_z=0.5,
        heading_radians=0.0,
        eva_cooldown_fraction=0.25,
        displacement_cells=1.0,
        contact=False,
        held_movement=FarmingAction.RUN_FORWARD_RIGHT,
        last_policy_action=FarmingAction.RUN_FORWARD_RIGHT,
        jump_cooldown_fraction=0.75,
        map_available=True,
    )
    context_map = np.resize(
        np.asarray((-1.0, -0.25, 0.50, 0.75, 1.0, 0.0), dtype=np.float32),
        441,
    ).reshape(21, 21)
    vector = ObservationBuilder().build_vector(
        ObservationFrame(
            player=player,
            actors=(actor,),
            local_map=local_map,
            context_map=context_map,
        )
    )

    golden = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    golden[0:7] = (0.12, -0.16, 0.2, 0.25, 0.025, 0.0, 1.0)
    golden[LEGACY_MASK_START] = 1.0
    golden[LEGACY_AGGREGATE_START : LEGACY_AGGREGATE_START + 5] = (
        1 / 200,
        0.0,
        1 / 40,
        0.25,
        1.0,
    )
    golden[UNIFIED_STATE_START : UNIFIED_STATE_START + 16] = (
        -0.25,
        0.5,
        0.0,
        1.0,
        -0.5,
        -1.0,
        1 / 256 * 2 - 1,
        -0.5,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        0.5,
        1.0,
        1.0,
    )
    golden[LOCAL_MAP_START : LOCAL_MAP_START + 121] = local_map.reshape(-1)
    golden[CONTEXT_MAP_START : CONTEXT_MAP_START + 441] = context_map.reshape(-1)
    golden[DIRECT_ACTOR_START : DIRECT_ACTOR_START + 7] = (
        0.06,
        0.08,
        -0.8,
        1.0,
        1.0,
        1.0,
        1 / 24 * 2 - 1,
    )

    np.testing.assert_array_equal(vector, golden)
    assert sha256(vector.tobytes()).hexdigest().upper() == (
        "F89B24EBBDD3B723276A810B9E30BA7044A12E6EEEE884473D9A55F2622AEBD0"
    )
