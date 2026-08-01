from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CoordinateMapLogger:
    """Compact CSV logger for the native-coordinate mapper."""

    FIELDS = (
        "timestamp",
        "map_name",
        "step",
        "action",
        "reason",
        "attempt",
        "before_native_x",
        "before_native_y",
        "before_native_z",
        "after_native_x",
        "after_native_y",
        "after_native_z",
        "delta_native_x",
        "delta_native_y",
        "delta_native_z",
        "horizontal_distance_units",
        "vertical_distance_units",
        "forward_progress_units",
        "lateral_distance_units",
        "forward_alignment",
        "motion_outcome",
        "local_x_cells",
        "local_y_cells",
        "heading_deg",
        "heading_index",
        "heading_source",
        "from_cell_x",
        "from_cell_y",
        "to_cell_x",
        "to_cell_y",
        "boundary_confirmations",
        "boundary_confirmed",
        "requested_seconds",
        "held_seconds",
        "note",
    )

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.handle.flush()

    @classmethod
    def canonical_row(cls, row: dict[str, Any]) -> dict[str, Any]:
        unknown = tuple(field for field in row if field not in cls.FIELDS)
        if unknown:
            raise ValueError(
                "Coordinate map row contains unknown fields: " + ", ".join(unknown)
            )
        return {field: row.get(field, "") for field in cls.FIELDS}

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(self.canonical_row(row))
        self.handle.flush()

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()
