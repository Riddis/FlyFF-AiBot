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
        "continuous_x",
        "continuous_y",
        "position_known",
        "heading_known",
        "pose_known",
        "heading_index",
        "heading_deg",
        "action",
        "reason",
        "frame_sequence",
        "requested_seconds",
        "held_seconds",
        "change_score",
        "flow_dx_px",
        "flow_dy_px",
        "median_flow_px",
        "flow_dispersion_px",
        "flow_confidence",
        "flow_inlier_ratio",
        "tracked_points",
        "flow_detected_points",
        "flow_valid_tracks",
        "flow_moving_points",
        "flow_moving_ratio",
        "flow_spatial_coverage",
        "flow_occupied_regions",
        "flow_translation_coherence",
        "flow_expansion_coherence",
        "flow_camera_model",
        "motion_debug_path",
        "motion_outcome",
        "distance_cells",
        "expected_flow_px",
        "observed_motion_px",
        "flow_residual_px",
        "maximum_flow_residual_px",
        "flow_validation_reason",
        "odometry_integrated",
        "collision",
        "pang_visible",
        "pang_score",
        "teleport_suspected",
        "fast_heading",
        "fast_heading_confidence",
        "fast_heading_uncertainty",
        "fast_heading_stale",
        "strict_heading",
        "strict_heading_confidence",
        "strict_heading_uncertainty",
        "stop_reason",
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
