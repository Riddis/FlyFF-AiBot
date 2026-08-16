"""B1 compatibility re-exports for canonical map features."""

# BRIDGE B1 — removed in Phase 7
from flyff_farming_simulator.farming.map_features import (
    DEFAULT_CONTEXT_MAP_RADIUS_CELLS,
    DEFAULT_CONTEXT_MAP_SIDE,
    DEFAULT_GEODESIC_CACHE_SIZE,
    DEFAULT_MAXIMUM_GEODESIC_EXPANSIONS,
    DEFAULT_TELEPORT_BUFFER_RADIUS_CELLS,
    LOCAL_MAP_OBSTACLE,
    LOCAL_MAP_OBSTACLE_BUFFER,
    LOCAL_MAP_OUTSIDE_OR_UNKNOWN,
    LOCAL_MAP_SAFE,
    LOCAL_MAP_TELEPORT_BUFFER,
    LOCAL_MAP_TELEPORT_TRIGGER,
    BoolArray,
    Cell,
    DirectPathState,
    FarmingMapFeatures,
    FloatArray,
    MapCellRisk,
    bresenham_cells,
)

__all__ = [
    "DEFAULT_CONTEXT_MAP_RADIUS_CELLS",
    "DEFAULT_CONTEXT_MAP_SIDE",
    "DEFAULT_GEODESIC_CACHE_SIZE",
    "DEFAULT_MAXIMUM_GEODESIC_EXPANSIONS",
    "DEFAULT_TELEPORT_BUFFER_RADIUS_CELLS",
    "LOCAL_MAP_OBSTACLE",
    "LOCAL_MAP_OBSTACLE_BUFFER",
    "LOCAL_MAP_OUTSIDE_OR_UNKNOWN",
    "LOCAL_MAP_SAFE",
    "LOCAL_MAP_TELEPORT_BUFFER",
    "LOCAL_MAP_TELEPORT_TRIGGER",
    "BoolArray",
    "Cell",
    "DirectPathState",
    "FarmingMapFeatures",
    "FloatArray",
    "MapCellRisk",
    "bresenham_cells",
]
