"""Physical-clearance steering features, derived statelessly from the
existing 923-value observation's cached `local_map` slice.

This is a DIFFERENT consumer of the same map data than
`local_clearance.sample_heading_relative_clearance`, with a different goal.
`local_clearance` feeds the Phase 1 emergency-escape controller and is
deliberately conservative -- it treats `OBSTACLE_BUFFER` as near-blocked
(weight 0.15), because a controller that's already decided the policy is
stuck should err toward caution when picking a blind escape direction.

This module feeds the LEARNED steering branch instead, whose whole purpose
is to let the model use available space confidently and stop right at
actual contact rather than give every wall a wide, inefficient berth (see
the approved Phase 2 plan's Step 0 finding: `OBSTACLE_BUFFER` is a software
dilation that is NOT physically blocking -- `movement_kinematics.
_BLOCKING_RISKS` excludes it). So the weighting here represents *physical
usable space*, not the escape controller's safety-zone concept, and
`OBSTACLE_BUFFER` defaults to being treated as traversable.

Reads only the first 923 raw observation values -- never the navigation
history sidecar -- so this composes cleanly with
`simulator.navigation_history.NavigationHistoryWrapper` regardless of
whether the sidecar is present.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from farming.map_features import (
    LOCAL_MAP_OBSTACLE,
    LOCAL_MAP_OBSTACLE_BUFFER,
    LOCAL_MAP_OUTSIDE_OR_UNKNOWN,
    LOCAL_MAP_SAFE,
    LOCAL_MAP_TELEPORT_BUFFER,
    LOCAL_MAP_TELEPORT_TRIGGER,
)

LOCAL_MAP_START = 277
LOCAL_MAP_STOP = 398
LOCAL_MAP_SIDE = 11
LOCAL_MAP_RADIUS = LOCAL_MAP_SIDE // 2  # 5
HEADING_SIN_INDEX = 263
HEADING_COS_INDEX = 264

CLEARANCE_FEATURE_SIZE = 3

# Sample distances in cells along each sector's heading. Extended to the
# full local_map crop radius based on real calibration data (see
# evaluations/navigation_calibration_results.json): with only (1,2,3),
# clearance read as fully-clear (1.0) on the immediately-preceding tick for
# 10% of real contacts (p90=1.0 of `clearance_before_contact`), i.e. not
# enough lead time for a meaningful fraction of real collisions -- not a
# hypothetical, an observed gap. Extending toward LOCAL_MAP_RADIUS per the
# Phase 2 plan's explicit lead-time validation requirement.
SAMPLE_DISTANCES_CELLS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)

SECTORS: dict[str, float] = {"left": math.pi / 4, "forward": 0.0, "right": -math.pi / 4}

# Physical-usable-space weighting -- deliberately NOT local_clearance.py's
# _RISK_WEIGHT. OBSTACLE_BUFFER defaults to traversable (matches
# movement_kinematics' own blocking definition); TELEPORT_BUFFER/TRIGGER
# stay cautious regardless, per Step 0's explicit carve-out that teleport
# safety is a separate hazard not affected by the obstacle-margin decision.
_CLEARANCE_WEIGHT: dict[float, float] = {
    LOCAL_MAP_SAFE: 1.0,
    LOCAL_MAP_OBSTACLE_BUFFER: 1.0,
    LOCAL_MAP_OUTSIDE_OR_UNKNOWN: 0.5,
    LOCAL_MAP_OBSTACLE: 0.0,
    LOCAL_MAP_TELEPORT_BUFFER: 0.1,
    LOCAL_MAP_TELEPORT_TRIGGER: 0.0,
}
_ENCODED_VALUES = np.array(list(_CLEARANCE_WEIGHT.keys()), dtype=np.float32)
_ENCODED_SCORES = np.array(list(_CLEARANCE_WEIGHT.values()), dtype=np.float32)


def _nearest_weight_numpy(encoded: np.ndarray) -> np.ndarray:
    # Snap each encoded local_map value to its nearest known constant (guards
    # against float round-trip noise) then look up its physical-clearance score.
    diffs = np.abs(encoded[..., None] - _ENCODED_VALUES)
    nearest = np.argmin(diffs, axis=-1)
    return _ENCODED_SCORES[nearest]


def derive_physical_clearance_features(observations: np.ndarray) -> np.ndarray:
    """Compute the 3 clearance features (left, forward, right) for one or
    many 923-value observations. Accepts shape (923,) or (N, 923)."""

    single = observations.ndim == 1
    obs = np.atleast_2d(observations).astype(np.float32)
    n = obs.shape[0]

    heading = np.arctan2(obs[:, HEADING_SIN_INDEX], obs[:, HEADING_COS_INDEX])
    local_map = obs[:, LOCAL_MAP_START:LOCAL_MAP_STOP].reshape(n, LOCAL_MAP_SIDE, LOCAL_MAP_SIDE)

    features = np.zeros((n, CLEARANCE_FEATURE_SIZE), dtype=np.float32)
    for feature_index, sector_offset in enumerate(SECTORS.values()):
        sector_angle = heading + sector_offset
        direction_x = np.cos(sector_angle)
        direction_z = np.sin(sector_angle)
        weights = []
        for distance in SAMPLE_DISTANCES_CELLS:
            # Native +x -> layout +dx (direct); native +z -> layout -dy
            # (flipped), per map_model.native_to_layout_cells. Sampling
            # directly in cell space against the cached crop, not native
            # units, since local_map is already indexed in layout-cell
            # offsets from the player's own cell.
            dx = np.clip(np.round(direction_x * distance), -LOCAL_MAP_RADIUS, LOCAL_MAP_RADIUS).astype(np.int64)
            dy = np.clip(np.round(-direction_z * distance), -LOCAL_MAP_RADIUS, LOCAL_MAP_RADIUS).astype(np.int64)
            rows = dy + LOCAL_MAP_RADIUS
            cols = dx + LOCAL_MAP_RADIUS
            encoded = local_map[np.arange(n), rows, cols]
            weights.append(_nearest_weight_numpy(encoded))
        features[:, feature_index] = np.mean(np.stack(weights, axis=0), axis=0)

    return features[0] if single else features


def derive_physical_clearance_features_torch(observations: torch.Tensor) -> torch.Tensor:
    """Torch-native mirror of derive_physical_clearance_features, for use
    inside the policy's forward pass (batched, no host round-trip)."""

    n = observations.shape[0]
    device = observations.device
    heading = torch.atan2(observations[:, HEADING_SIN_INDEX], observations[:, HEADING_COS_INDEX])
    local_map = observations[:, LOCAL_MAP_START:LOCAL_MAP_STOP].reshape(n, LOCAL_MAP_SIDE, LOCAL_MAP_SIDE)

    encoded_values = torch.as_tensor(_ENCODED_VALUES, device=device, dtype=observations.dtype)
    encoded_scores = torch.as_tensor(_ENCODED_SCORES, device=device, dtype=observations.dtype)
    rows_idx = torch.arange(n, device=device)

    feature_columns = []
    for sector_offset in SECTORS.values():
        sector_angle = heading + sector_offset
        direction_x = torch.cos(sector_angle)
        direction_z = torch.sin(sector_angle)
        weights = []
        for distance in SAMPLE_DISTANCES_CELLS:
            dx = torch.clamp(torch.round(direction_x * distance), -LOCAL_MAP_RADIUS, LOCAL_MAP_RADIUS).long()
            dy = torch.clamp(torch.round(-direction_z * distance), -LOCAL_MAP_RADIUS, LOCAL_MAP_RADIUS).long()
            row = dy + LOCAL_MAP_RADIUS
            col = dx + LOCAL_MAP_RADIUS
            encoded = local_map[rows_idx, row, col]
            diffs = torch.abs(encoded.unsqueeze(-1) - encoded_values)
            nearest = torch.argmin(diffs, dim=-1)
            weights.append(encoded_scores[nearest])
        feature_columns.append(torch.stack(weights, dim=0).mean(dim=0))

    return torch.stack(feature_columns, dim=1)
