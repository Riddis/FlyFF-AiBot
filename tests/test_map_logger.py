from __future__ import annotations

import csv

from mapper.MapLogger import MapLogger


def test_map_logger_accepts_complete_mapper_step(tmp_path) -> None:
    path = tmp_path / "mapping_steps.csv"
    logger = MapLogger(path)
    row = {field: field for field in MapLogger.FIELDS}

    logger.write(row)
    logger.close()

    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))

    assert records == [row]
    assert records[0]["fast_heading"] == "fast_heading"
    assert records[0]["fast_heading_confidence"] == "fast_heading_confidence"
    assert records[0]["fast_heading_stale"] == "fast_heading_stale"
