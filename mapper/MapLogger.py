from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class MapLogger:
    FIELDS = (
        "timestamp",
        "map_name",
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
        "rl_shadow_enabled",
        "rl_shadow_action",
        "rl_shadow_agrees",
        "rl_shadow_status",
        "frame_sequence",
        "requested_seconds",
        "held_seconds",
        "native_shadow_enabled",
        "native_before_x",
        "native_before_y",
        "native_before_z",
        "native_after_x",
        "native_after_y",
        "native_after_z",
        "native_horizontal_distance",
        "native_vertical_delta",
        "native_motion_outcome",
        "native_motion_error",
        "visual_native_agree",
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
        "camera_obscured",
        "camera_recovery_attempted",
        "camera_recovered",
        "camera_recovery_turns",
        "camera_recovery_reason",
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
        "recovery_reason",
        "recovery_requires_spawn_reset",
        "stop_reason",
    )

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.handle.flush()

    @classmethod
    def canonical_row(cls, row: dict[str, Any]) -> dict[str, Any]:
        """Return one row in the exact persisted CSV schema order.

        Mapper implementations evolved at different times and may omit fields
        that are irrelevant to a particular runtime.  Canonicalising at the
        boundary keeps legacy and adaptive mappers compatible with one shared
        logger without allowing dictionary insertion order to drift from the
        CSV header.
        """

        unknown = tuple(field for field in row if field not in cls.FIELDS)
        if unknown:
            raise ValueError(
                "Mapper log row contains fields outside MapLogger.FIELDS: "
                + ", ".join(unknown)
            )
        return {field: row.get(field, "") for field in cls.FIELDS}

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(self.canonical_row(row))
        self.handle.flush()

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()
