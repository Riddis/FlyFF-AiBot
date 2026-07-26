from __future__ import annotations

import pytest

from mapper.MapLogger import MapLogger


def test_canonical_row_matches_shared_logger_schema_order() -> None:
    row = MapLogger.canonical_row({"timestamp": "now", "continuous_y": 2.0})

    assert tuple(row) == MapLogger.FIELDS
    assert row["timestamp"] == "now"
    assert row["continuous_y"] == 2.0
    assert row["map_name"] == ""


def test_canonical_row_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="outside MapLogger.FIELDS"):
        MapLogger.canonical_row({"not_a_field": 1})
