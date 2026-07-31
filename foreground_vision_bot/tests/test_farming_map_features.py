from __future__ import annotations

from math import inf, sqrt

import numpy as np
import pytest
from farming.map_features import (
    DirectPathState,
    FarmingMapFeatures,
    bresenham_cells,
)


def _map_features() -> FarmingMapFeatures:
    traversable = np.ones((9, 9), dtype=np.bool_)
    traversable[1, 1] = False
    forbidden = np.zeros_like(traversable)
    forbidden[4, 4] = True
    safe = traversable & ~forbidden
    return FarmingMapFeatures(
        traversable=traversable,
        forbidden=forbidden,
        safe_traversable=safe,
        teleport_buffer_radius_cells=2.0,
    )


def test_local_map_crop_distinguishes_trigger_buffer_wall_and_outside() -> None:
    features = _map_features()
    crop = features.local_crop((4, 4), side=7).reshape(7, 7)

    assert crop[3, 3] == pytest.approx(1.0)
    assert crop[3, 4] == pytest.approx(0.75)
    wall_crop = features.local_crop((2, 2), side=5).reshape(5, 5)
    assert wall_crop[1, 1] == pytest.approx(0.25)
    outside_crop = features.local_crop((0, 0), side=3).reshape(3, 3)
    assert outside_crop[0, 0] == pytest.approx(0.0)
    assert outside_crop[1, 1] == pytest.approx(-1.0)


def test_forbidden_distance_field_is_exact_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _map_features()

    calls = 0
    original = FarmingMapFeatures._get_forbidden_distance_field

    def counted(instance: FarmingMapFeatures) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(instance)

    monkeypatch.setattr(FarmingMapFeatures, "_get_forbidden_distance_field", counted)
    assert features.forbidden_distance((4, 4)) == pytest.approx(0.0)
    assert features.forbidden_distance((5, 4)) == pytest.approx(1.0)
    assert features.forbidden_distance((5, 5)) == pytest.approx(sqrt(2.0))
    assert features.forbidden_distance((-1, 0)) is None
    assert calls == 3

    fresh = _map_features()
    calls = 0
    monkeypatch.setattr(
        np,
        "any",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local_crop rescanned the full forbidden map")
        ),
    )
    fresh.local_crop((4, 4), side=7)
    assert calls == 1


def test_direct_path_reports_blocked_clear_and_unknown_without_long_detours() -> None:
    features = _map_features()

    assert features.direct_path_state((0, 4), (8, 4)) is DirectPathState.BLOCKED
    assert features.direct_path_state((0, 0), (0, 8)) is DirectPathState.CLEAR
    assert features.direct_path_state(None, (0, 8)) is DirectPathState.UNKNOWN
    assert features.segment_crosses_forbidden((0, 4), (8, 4))


def test_geodesic_queries_are_bounded_cached_and_prevent_corner_cutting() -> None:
    open_cells = np.ones((5, 5), dtype=np.bool_)
    forbidden = np.zeros_like(open_cells)
    features = FarmingMapFeatures(
        traversable=open_cells,
        forbidden=forbidden,
        geodesic_cache_size=2,
    )

    assert features.geodesic_distance((0, 0), (2, 2)) == pytest.approx(2.0 * sqrt(2.0))
    assert features.geodesic_distance((2, 2), (0, 0)) == pytest.approx(2.0 * sqrt(2.0))
    assert features.geodesic_cache_entries == 2
    assert (
        features.geodesic_distance(
            (0, 0),
            (4, 4),
            maximum_distance_cells=2.0,
        )
        == inf
    )

    isolated = open_cells.copy()
    isolated[0, 1] = False
    isolated[1, 0] = False
    corner_features = FarmingMapFeatures(
        traversable=isolated,
        forbidden=forbidden,
        safe_traversable=isolated,
    )
    assert corner_features.geodesic_distance((0, 0), (1, 1)) == inf


def test_map_arrays_are_immutable_snapshots_and_bresenham_includes_endpoints() -> None:
    traversable = np.ones((3, 3), dtype=np.bool_)
    forbidden = np.zeros_like(traversable)
    features = FarmingMapFeatures(
        traversable=traversable,
        forbidden=forbidden,
    )
    traversable[0, 0] = False

    assert bool(features.traversable[0, 0])
    with pytest.raises(ValueError):
        features.traversable[0, 0] = False
    exposed = features.traversable
    with pytest.raises(ValueError):
        exposed.setflags(write=True)
    assert isinstance(exposed.base, np.ndarray)
    with pytest.raises(ValueError):
        exposed.base.setflags(write=True)
    assert bool(features.traversable[0, 0])
    assert tuple(bresenham_cells((0, 0), (2, 1))) == (
        (0, 0),
        (1, 1),
        (2, 1),
    )


def test_explicit_safe_cells_must_be_a_traversable_non_forbidden_subset() -> None:
    traversable = np.ones((3, 3), dtype=np.bool_)
    traversable[0, 0] = False
    forbidden = np.zeros_like(traversable)
    unsafe_safe = np.ones_like(traversable)

    with pytest.raises(ValueError, match="subset of traversable"):
        FarmingMapFeatures(
            traversable=traversable,
            forbidden=forbidden,
            safe_traversable=unsafe_safe,
        )
