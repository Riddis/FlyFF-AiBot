"""The smallest structural map protocol the current shared navigation
algorithms actually need, derived mechanically from `movement_kinematics.py`'s
real access expressions (`map_model.native_units_per_cell`,
`map_model.features.cell_risk(cell)`, `map_model.native_to_layout_cell(x, z)`)
-- not copied from `simulator.map_model.MapModel`'s full surface.

`simulator.map_model.MapModel` already satisfies this protocol structurally,
with zero numerical change (see `tests/test_navigation_dependency_boundary.py`).
This module intentionally contains no algorithms and does not import
`simulator.map_model` (that would recreate the exact coupling this protocol
exists to remove, and simulator already imports `navigation`, so the reverse
import would cycle).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from farming.map_features import FarmingMapFeatures

Cell = tuple[int, int]


@runtime_checkable
class NavigationMapProtocol(Protocol):
    native_units_per_cell: float
    features: FarmingMapFeatures

    def native_to_layout_cell(self, x: float, z: float) -> Cell | None: ...
