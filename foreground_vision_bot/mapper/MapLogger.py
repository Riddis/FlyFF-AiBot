from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class MapLogger:
    FIELDS = (
        "timestamp",
        "step",
        "x",
        "y",
        "heading",
        "action",
        "reason",
        "change_score",
        "median_flow_px",
        "tracked_points",
        "collision",
        "pang_visible",
        "pang_score",
        "teleport_suspected",
    )

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()
