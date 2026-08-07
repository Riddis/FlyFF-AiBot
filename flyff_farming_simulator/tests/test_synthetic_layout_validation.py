from __future__ import annotations

import math

import numpy as np

from simulator.map_model import MapModel
from simulator.synthetic import (
    _boundary_safe_positions,
    _generate_validated_layout,
    _layout_escapability_reasons,
    _regains_movement_within,
    _STAGE_ESCAPE_TICKS,
)
from simulator.world_model import MovementModel

_MOVEMENT = (
    MovementModel(100, 1.5, 0.0, 0.0, 0.0),
    MovementModel(100, 1.5, 0.0, 0.25, 0.0),
    MovementModel(100, 1.5, 0.0, -0.25, 0.0),
    MovementModel(0, 0.0, 0.0, 0.0, 0.0),
    MovementModel(10, 1.5, 0.0, 0.0, 0.0),
)


def _cul_de_sac(width: int) -> MapModel:
    """A corridor of the given width, closed at the top (row < 5), open into
    a large field at the bottom (row >= 30). ``width=1`` leaves no room to
    turn inside the corridor; escaping requires reversing heading almost
    entirely, which needs many more ticks than a single glancing turn."""

    size = 41
    traversable = np.zeros((size, size), dtype=bool)
    center = 20
    half = width // 2
    traversable[5:36, center - half : center - half + width] = True
    traversable[30:41, :] = True
    return MapModel.from_arrays(traversable, obstacle_radius_cells=0)


def test_open_field_position_is_immediately_escapable() -> None:
    open_map = MapModel.from_arrays(np.ones((41, 41), dtype=bool), obstacle_radius_cells=0)
    x, z = open_map.layout_to_native(20, 20)
    assert _regains_movement_within(open_map, _MOVEMENT, x, z, 0.0, max_ticks=1)


def test_dead_end_facing_the_closed_wall_is_not_escapable_within_a_tight_budget() -> None:
    """Reproduces the mechanism behind a real stuck teacher episode: facing
    directly into a dead end's closed wall from right up against it takes
    real turning ticks to correct, not an instant recovery.

    Pinned to a literal tick count (measured: fails through 12 ticks,
    succeeds by 16) rather than _STAGE_ESCAPE_TICKS["early"]. That constant
    was recalibrated after obstacle_radius_cells -> 0 to cover the worst
    case actually found in generated boundary-safe positions (24 ticks,
    concave multi-wall corners), which turns out to need MORE ticks than a
    clean single-wall corridor reversal like this one (16 ticks) -- so this
    fixture is no longer a proxy for the early-stage difficulty ceiling,
    just a fixed regression pin on the search itself."""

    cul_de_sac = _cul_de_sac(width=3)
    x, z = cul_de_sac.layout_to_native(20, 5)
    heading_into_wall = math.pi / 2.0

    assert not _regains_movement_within(
        cul_de_sac, _MOVEMENT, x, z, heading_into_wall, max_ticks=12
    )


def test_dead_end_is_escapable_given_the_advanced_stage_budget() -> None:
    """The same dead end must still be provably escapable with the same
    controls given enough ticks -- never trust an unrecoverable trap even at
    the most lenient stage."""

    cul_de_sac = _cul_de_sac(width=3)
    x, z = cul_de_sac.layout_to_native(20, 5)
    heading_into_wall = math.pi / 2.0

    assert _regains_movement_within(
        cul_de_sac, _MOVEMENT, x, z, heading_into_wall, max_ticks=_STAGE_ESCAPE_TICKS["advanced"]
    )


def test_moving_deeper_into_a_dead_end_does_not_count_as_escaping() -> None:
    """A sliding move that is still blocked, or that shuffles the player
    sideways or further into the pocket, must not be mistaken for having
    regained real movement -- only a genuinely uncontacted step counts."""

    cul_de_sac = _cul_de_sac(width=1)
    x, z = cul_de_sac.layout_to_native(20, 5)
    heading_into_wall = math.pi / 2.0

    assert not _regains_movement_within(
        cul_de_sac, _MOVEMENT, x, z, heading_into_wall, max_ticks=3
    )


def test_boundary_safe_positions_excludes_interior_cells() -> None:
    traversable = np.ones((41, 41), dtype=bool)
    traversable[20, 20] = False  # a single obstacle cell carved out of open ground
    map_model = MapModel.from_arrays(traversable, obstacle_radius_cells=0)

    boundary = {tuple(cell) for cell in _boundary_safe_positions(map_model)}

    assert (19, 20) in boundary or (21, 20) in boundary or (20, 19) in boundary or (20, 21) in boundary
    assert (5, 5) not in boundary  # far from both the carved obstacle and the map edge


def test_layout_escapability_reasons_flags_an_unescapable_spawn() -> None:
    """A fully sealed single-cell pocket -- not merely a narrow corridor --
    so the assertion holds by construction (zero room to move in any
    direction) rather than depending on a specific tick budget. A width=1
    open-ended corridor no longer works for this: under the recalibrated
    early-stage budget (see _STAGE_ESCAPE_TICKS' comment) it turns out to be
    escapable from every sampled heading well within budget."""

    size = 41
    traversable = np.zeros((size, size), dtype=bool)
    traversable[20, 20] = True
    sealed_box = MapModel.from_arrays(traversable, obstacle_radius_cells=0)
    spawn_native = sealed_box.layout_to_native(20, 20)

    reasons = _layout_escapability_reasons(
        sealed_box,
        _MOVEMENT,
        np.random.default_rng(0),
        stage="early",
        spawn_native=spawn_native,
    )

    assert any("spawn" in reason for reason in reasons)


def test_generate_validated_layout_produces_an_escapable_open_field() -> None:
    map_model, spawn_cell, metadata, _rng = _generate_validated_layout(
        "open_field",
        base_seed=42,
        obstacle_level=1,
        stage="early",
        movement=_MOVEMENT,
    )

    assert metadata["escapability_validated"] is True
    spawn_native = map_model.layout_to_native(*spawn_cell)
    assert _regains_movement_within(
        map_model,
        _MOVEMENT,
        spawn_native[0],
        spawn_native[1],
        0.0,
        max_ticks=_STAGE_ESCAPE_TICKS["early"],
    )
