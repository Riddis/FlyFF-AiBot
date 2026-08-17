"""Behavior-free compatibility shim. Zero movement-kernel implementation is
defined here, or ever should be -- the canonical implementation is
navigation/movement_kernel.py.

Same rationale as simulator/kinodynamic_route_planner.py (see that file's
docstring for the full account): `AdvanceResult.__module__` is pinned to
this path to preserve the frozen G7/G8c typed-encoding fixtures, and this
file exists solely so that pickle's global-object lookup
(`getattr(sys.modules["simulator.movement_kernel"], "AdvanceResult")`) has
a real, importable module to resolve against -- confirmed necessary via a
fresh-subprocess pickle.dumps()/loads() round-trip probe that failed with
`PicklingError: Can't pickle <class 'simulator.movement_kernel.
AdvanceResult'>: No module named 'simulator.movement_kernel'` before this
file existed.

Re-exports the exact same class object (not a copy) from
navigation/movement_kernel.py. Registered as a permanent
(`removal_gate = "NEVER"`) compatibility shim in CANONICAL_OWNERS.toml's
`[[shim]]` registry so R7c does not flag this re-export."""

from __future__ import annotations

from navigation.movement_kernel import AdvanceResult

__all__ = ["AdvanceResult"]
