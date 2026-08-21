"""Phase-9 pickle-compatibility hardening (2026-08-17): tests for the two
narrow, behavior-free compatibility shims
(simulator/kinodynamic_route_planner.py, simulator/movement_kernel.py)
created after a fresh-subprocess probe proved that pinning
KinoState.__module__/RouteEdgeInfo.__module__/AdvanceResult.__module__ back
to their pre-Phase-9 path strings (for frozen G7/G8c fixture compatibility)
was not sufficient for actual pickle.dumps()/pickle.loads() of a live
instance -- only for the frozen fixtures' own string-based
__module__.__qualname__ encoding.

Covers, in order: canonical implementation origin, legacy-import identity,
in-process and cold-subprocess pickle round-trips, absence of duplicate
behavioral definitions, and the historical guard's fail-closed
classification shift these shims cause (MISSING -> hash mismatch, still a
refusal, never a pass)."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

PICKLE_PROBE = r"""
import pickle
import sys
sys.path.insert(0, {repo!r})

from navigation.kinodynamic_route_planner import KinoState, RouteEdgeInfo
from navigation.movement_kernel import AdvanceResult, SteeringDirection

cases = {{
    "KinoState": KinoState(x=1.0, z=2.0, heading=0.5, previous_steering=SteeringDirection.LEFT),
    "RouteEdgeInfo": RouteEdgeInfo(action="advance", distance_cells=3.0, heading_change_radians=0.1, robust_clearance_cells=2.5),
    "AdvanceResult": AdvanceResult(x=1.0, z=2.0, heading=0.5, contact=False, next_previous_steering=SteeringDirection.RIGHT),
}}

results = {{}}
for name, obj in cases.items():
    blob = pickle.dumps(obj)
    restored = pickle.loads(blob)
    results[name] = (
        type(restored) is type(obj),
        restored == obj,
        type(restored).__module__,
        type(restored).__qualname__,
    )

for name, (same_class, equal, module, qualname) in results.items():
    print(f"{{name}}|{{same_class}}|{{equal}}|{{module}}|{{qualname}}")
"""


def test_canonical_implementation_origin_remains_navigation() -> None:
    """KinoState/RouteEdgeInfo/AdvanceResult must still be actually
    DEFINED (class statement) only in navigation/*, never re-defined
    (only re-exported) anywhere else -- this is exactly what R6/R7a's
    `definition_owners` (AST class/def scan) also enforces, checked here
    directly against source rather than through the ruler."""
    router_tree = ast.parse((REPO / "navigation/kinodynamic_route_planner.py").read_text(encoding="utf-8"))
    kernel_tree = ast.parse((REPO / "navigation/movement_kernel.py").read_text(encoding="utf-8"))
    router_classes = {node.name for node in router_tree.body if isinstance(node, ast.ClassDef)}
    kernel_classes = {node.name for node in kernel_tree.body if isinstance(node, ast.ClassDef)}
    assert {"KinoState", "RouteEdgeInfo"} <= router_classes
    assert "AdvanceResult" in kernel_classes


def test_legacy_import_resolves_to_the_same_class_objects() -> None:
    import navigation.kinodynamic_route_planner as canonical_router
    import navigation.movement_kernel as canonical_kernel
    import simulator.kinodynamic_route_planner as legacy_router
    import simulator.movement_kernel as legacy_kernel

    assert legacy_router.KinoState is canonical_router.KinoState
    assert legacy_router.RouteEdgeInfo is canonical_router.RouteEdgeInfo
    assert legacy_kernel.AdvanceResult is canonical_kernel.AdvanceResult


def test_pickle_round_trip_succeeds_in_process() -> None:
    from navigation.kinodynamic_route_planner import KinoState, RouteEdgeInfo
    from navigation.movement_kernel import AdvanceResult, SteeringDirection

    import pickle

    kino = KinoState(x=1.0, z=2.0, heading=0.5, previous_steering=SteeringDirection.LEFT)
    edge = RouteEdgeInfo(action="advance", distance_cells=3.0, heading_change_radians=0.1, robust_clearance_cells=2.5)
    advance = AdvanceResult(x=1.0, z=2.0, heading=0.5, contact=False, next_previous_steering=SteeringDirection.RIGHT)

    for original in (kino, edge, advance):
        restored = pickle.loads(pickle.dumps(original))
        assert type(restored) is type(original)
        assert restored == original


def test_pickle_round_trip_succeeds_in_a_fresh_subprocess_with_only_repo_root_on_sys_path() -> None:
    """The rigorous form of the check above: a cold interpreter, only the
    collapsed repository root inserted onto sys.path (plus whatever -I
    isolated mode keeps for the stdlib/venv itself), proving this doesn't
    depend on some other module having already been imported first."""
    result = subprocess.run(
        [sys.executable, "-I", "-c", PICKLE_PROBE.format(repo=str(REPO))],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    parsed = {line.split("|")[0]: line.split("|")[1:] for line in lines}
    assert set(parsed) == {"KinoState", "RouteEdgeInfo", "AdvanceResult"}

    same_class, equal, module, qualname = parsed["KinoState"]
    assert same_class == "True" and equal == "True"
    assert module == "simulator.kinodynamic_route_planner" and qualname == "KinoState"

    same_class, equal, module, qualname = parsed["RouteEdgeInfo"]
    assert same_class == "True" and equal == "True"
    assert module == "simulator.kinodynamic_route_planner" and qualname == "RouteEdgeInfo"

    same_class, equal, module, qualname = parsed["AdvanceResult"]
    assert same_class == "True" and equal == "True"
    assert module == "simulator.movement_kernel" and qualname == "AdvanceResult"


def test_compat_shims_contain_no_duplicate_behavioral_definitions() -> None:
    """The shims must be pure re-export: only __future__ import, the
    navigation.* import, and an __all__ assignment. No class/function
    definitions, no other logic, ever."""
    for shim_path, forbidden_class_names in (
        (REPO / "simulator/kinodynamic_route_planner.py", {"KinoState", "RouteEdgeInfo"}),
        (REPO / "simulator/movement_kernel.py", {"AdvanceResult"}),
    ):
        tree = ast.parse(shim_path.read_text(encoding="utf-8"), filename=str(shim_path))
        offenders = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                offenders.append(f"{type(node).__name__}:{node.name}")
            elif isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if targets != ["__all__"]:
                    offenders.append(f"Assign:{targets}")
            elif isinstance(node, ast.ImportFrom):
                continue
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # module docstring
            else:
                offenders.append(f"{type(node).__name__}")
        assert offenders == [], f"{shim_path}: unexpected non-re-export content: {offenders}"
        assert forbidden_class_names <= set().union(*[
            {alias.name for alias in node.names}
            for node in tree.body if isinstance(node, ast.ImportFrom)
        ]), f"{shim_path}: expected class names not found among its imports"


def test_historical_guard_still_fails_closed_with_the_shims_present() -> None:
    """The shims now occupy the two paths scratchpad_historical_
    reproduction_guard.py's REQUIRED_FILES checks, so the guard's
    fail-closed REASON shifts from MISSING (Phase 9, no file there at all)
    to a hash mismatch (Phase-9-hardening, a real but different file is
    there) -- both are refusals; neither is a pass, and this is not a
    "fix" of the guard, just documentation of the new reason. No
    REQUIRED_FILES bytes were touched to produce this result."""
    import simulator.scratchpad.scratchpad_historical_reproduction_guard as guard

    with pytest.raises(RuntimeError) as excinfo:
        guard.verify_historical_snapshot()
    message = str(excinfo.value)
    assert "simulator/kinodynamic_route_planner.py" in message
    assert "simulator/movement_kernel.py" in message
    assert "MISSING" not in message, "expected a hash mismatch now that a real (shim) file exists at these paths"
    for rel in guard.REQUIRED_FILES:
        if rel in ("simulator/kinodynamic_route_planner.py", "simulator/movement_kernel.py"):
            continue
        assert rel not in message
