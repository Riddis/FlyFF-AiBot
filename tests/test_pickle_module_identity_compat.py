"""Post-migration compatibility purge (2026-08-21): the two pickle-compat
shims (simulator/kinodynamic_route_planner.py, simulator/movement_kernel.py)
created 2026-08-17 were removed. A static pickle disassembly of every
internal file inside models/generalized_waypoint_both_seed2_0051200.zip
(the checkpoint these shims were originally believed to protect) found
zero references to either shim module or to KinoState/RouteEdgeInfo/
AdvanceResult anywhere -- their __module__ pins existed solely for
tests/fixtures/migration/router_kernel.json, a Phase-3 G8c migration-
continuity fixture nothing in the current product or test suite reads
or validates. See docs/decisions/0002-preserve-abi-compatibility-shims.md's
Retirement section for the full evidence trail.

KinoState/RouteEdgeInfo/AdvanceResult now carry their natural
navigation.* module identity (no override). Covers, in order: canonical
implementation origin, in-process and cold-subprocess pickle round-trips
under that natural identity, and the historical guard's fail-closed
classification with the shims absent (still MISSING, still a refusal,
never a pass -- the guard already refused to run before this change,
since its own tracked files had drifted from their 2026-08-15 snapshot
hash back on 2026-08-17 when these shims were first introduced)."""

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


def test_no_module_identity_override_remains() -> None:
    """The classes must carry their natural module identity -- no
    __module__ assignment anywhere in their defining files."""
    for path, names in (
        (REPO / "navigation/kinodynamic_route_planner.py", ("KinoState", "RouteEdgeInfo")),
        (REPO / "navigation/movement_kernel.py", ("AdvanceResult",)),
    ):
        source = path.read_text(encoding="utf-8")
        for name in names:
            assert f"{name}.__module__" not in source, f"{path}: {name}.__module__ override still present"


def test_shim_modules_no_longer_exist() -> None:
    assert not (REPO / "simulator" / "kinodynamic_route_planner.py").exists()
    assert not (REPO / "simulator" / "movement_kernel.py").exists()


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
    depend on some other module having already been imported first, and
    that no simulator.* compatibility module is required for this to
    work."""
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
    assert module == "navigation.kinodynamic_route_planner" and qualname == "KinoState"

    same_class, equal, module, qualname = parsed["RouteEdgeInfo"]
    assert same_class == "True" and equal == "True"
    assert module == "navigation.kinodynamic_route_planner" and qualname == "RouteEdgeInfo"

    same_class, equal, module, qualname = parsed["AdvanceResult"]
    assert same_class == "True" and equal == "True"
    assert module == "navigation.movement_kernel" and qualname == "AdvanceResult"


def test_historical_guard_fails_closed_with_shims_removed() -> None:
    """The shims previously occupied the two paths scratchpad_historical_
    reproduction_guard.py's REQUIRED_FILES checks. With the shims
    removed, the guard's fail-closed REASON reverts to MISSING (its
    original Phase-9 state, before the 2026-08-17 hardening shims ever
    existed) -- still a refusal, never a pass. This is not a "fix" of
    the guard: it already refused to run before this change too, since
    its tracked files had already drifted from the 2026-08-15 snapshot
    hash the moment those shims first narrowed to pure re-exports."""
    import simulator.scratchpad.scratchpad_historical_reproduction_guard as guard

    with pytest.raises(RuntimeError) as excinfo:
        guard.verify_historical_snapshot()
    message = str(excinfo.value)
    assert "simulator/kinodynamic_route_planner.py" in message
    assert "simulator/movement_kernel.py" in message
    assert "MISSING" in message, "expected MISSING now that the shim files no longer exist"
    for rel in guard.REQUIRED_FILES:
        if rel in ("simulator/kinodynamic_route_planner.py", "simulator/movement_kernel.py"):
            continue
        assert rel not in message
