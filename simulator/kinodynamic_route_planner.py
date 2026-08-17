"""Behavior-free compatibility shim. Zero routing/planning implementation
is defined here, or ever should be -- the canonical implementation is
navigation/kinodynamic_route_planner.py.

Why this file exists (2026-08-17, post-Phase-9 pickle-compatibility
hardening): Phase 9 physically moved this module to navigation/ and, to
preserve the frozen G7/G8c typed-encoding fixtures
(tests/fixtures/migration/router_kernel.json embeds
f"{type(value).__module__}.{type(value).__qualname__}" for every decoded
KinoState/RouteEdgeInfo), pinned `KinoState.__module__` and
`RouteEdgeInfo.__module__` back to this path string. That override is
sufficient for the frozen fixtures -- they only ever read
`__module__.__qualname__` as a string -- but it is NOT sufficient for
`pickle.dumps()`/`pickle.loads()` of a live instance: pickle's global-object
lookup imports `obj.__module__` and asserts
`getattr(that_module, obj.__qualname__) is obj` before it will write (or
resolve) a class reference, and without a real, importable module at this
exact path, that lookup previously failed with
`PicklingError: Can't pickle <class 'simulator.kinodynamic_route_planner.
KinoState'>: No module named 'simulator.kinodynamic_route_planner'` --
confirmed via a fresh-subprocess round-trip probe before this file existed.

This file makes that lookup succeed again by re-exporting the exact same
class objects (not copies) from navigation/kinodynamic_route_planner.py, so
`getattr(simulator.kinodynamic_route_planner, "KinoState") is
navigation.kinodynamic_route_planner.KinoState` holds. Registered as a
permanent (`removal_gate = "NEVER"`) compatibility shim in
CANONICAL_OWNERS.toml's `[[shim]]` registry so R7c does not flag this
re-export, and R6/R7a never see it as a competing owner in the first place
(it contains no `class`/`def`/top-level-assignment definitions for these
names, only an import)."""

from __future__ import annotations

from navigation.kinodynamic_route_planner import KinoState, RouteEdgeInfo

__all__ = ["KinoState", "RouteEdgeInfo"]
