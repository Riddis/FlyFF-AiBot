from __future__ import annotations

import csv
from time import monotonic

import numpy as np
from capture_service import FrameSample
from mapper.Explorer import ExplorerDecision
from mapper.MapLogger import MapLogger
from mapper.Mapper import Mapper, _StepResult
from mapper.OccupancyGrid import OccupancyGrid
from mapper.PangDetector import PangDetection


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


def test_mapper_step_row_exactly_matches_logger_schema() -> None:
    captured_rows: list[dict[str, object]] = []
    mapper = Mapper.__new__(Mapper)
    mapper.grid = OccupancyGrid(size=21)
    mapper._position_known = True
    mapper._heading_known = True
    mapper.logger = type(
        "CaptureLogger",
        (),
        {"write": lambda _self, row: captured_rows.append(row)},
    )()
    sample = FrameSample(
        frame=np.zeros((20, 20), dtype=np.uint8),
        generation=1,
        sequence=2,
        captured_at=monotonic(),
    )
    result = _StepResult(
        frame_sample=sample,
        fast_heading=None,
        strict_heading=None,
        motion=None,
        key_timing=None,
        distance_cells=None,
        integration=None,
        pose_known=True,
    )

    mapper._write_step(
        step=1,
        decision=ExplorerDecision("TURN_LEFT", "test"),
        result=result,
        pang=PangDetection(False, 0.0, None),
    )

    assert len(captured_rows) == 1
    assert tuple(captured_rows[0]) == MapLogger.FIELDS
