"""B1 compatibility re-exports for canonical map-mask helpers."""

# BRIDGE B1 — removed in Phase 7
from flyff_farming_simulator.farming.map_masks import (
    InflatedMapMasks,
    dilate_mask,
    inflate_map_masks,
)

__all__ = ["InflatedMapMasks", "dilate_mask", "inflate_map_masks"]
