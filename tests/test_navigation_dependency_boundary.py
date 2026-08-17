"""Phase-9 architectural gate: proves the shared navigation core has no
training/runtime dependency leak, so the finished canonical development bot
(and, later, a standalone derivation from it) could consume this package
without dragging simulator/training internals into its runtime.

This is NOT a runtime integration test -- it does not build or validate a
standalone bot. It only proves the import closure of navigation.* stays
clean of the specific disallowed dependencies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

NAVIGATION_MODULES = (
    "navigation.kinodynamic_route_planner",
    "navigation.movement_kernel",
    "navigation.movement_kinematics",
    "navigation.navigation_evidence",
    "navigation.map_protocol",
)

DISALLOWED_MODULE_PREFIXES = (
    "gymnasium",
    "gym",
    "stable_baselines3",
    "torch",
    "recorder",
    "position",
    "runtime_bus",
    "win32api",
    "win32con",
    "win32gui",
    "win32ui",
    "simulator.environment",
    "simulator.navigation_history",
    "simulator.router_waypoint_env",
    "simulator.static_waypoint_env",
    "simulator.single_obstacle_env",
    "simulator.synthetic",
    "simulator.basic_training",
    "simulator.navigation_dataset",
    "simulator.split_branch_policy",
)

_PROBE = r"""
import sys
sys.path.insert(0, {repo!r})

before = set(sys.modules)
for name in {names!r}:
    __import__(name)
after = set(sys.modules)
newly_imported = sorted(after - before)
print("\n".join(newly_imported))
"""


def test_navigation_modules_import_without_disallowed_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE.format(repo=str(REPO), names=NAVIGATION_MODULES)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    imported = [line for line in result.stdout.splitlines() if line.strip()]
    violations = [
        name for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in DISALLOWED_MODULE_PREFIXES)
    ]
    assert violations == [], (
        f"navigation.* import closure pulled in disallowed dependencies: {violations}\n"
        f"full closure: {imported}"
    )


def test_navigation_modules_do_not_reference_disallowed_names_in_source() -> None:
    """Belt-and-braces source-level check (catches TYPE_CHECKING-guarded or
    lazily-imported references the runtime import-closure probe above would
    miss). AST-based -- only real import statements, never docstring/comment
    prose describing what must not be imported."""
    import ast

    navigation_dir = REPO / "navigation"
    offenders: list[str] = []
    for path in sorted(navigation_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            else:
                continue
            for root in roots:
                if any(root == prefix or f"{root}." == f"{prefix}." for prefix in DISALLOWED_MODULE_PREFIXES):
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno}: imports {root!r}")
    assert offenders == [], offenders


def test_simulator_map_model_satisfies_the_navigation_map_protocol() -> None:
    """Structural (not textual) proof that the minimal protocol
    navigation.movement_kinematics actually needs is satisfied by the real
    simulator.map_model.MapModel, with zero numerical change -- confirming
    future dev-bot implementability is structurally possible without
    writing that adapter now."""
    import numpy as np

    from navigation.map_protocol import NavigationMapProtocol
    from simulator.map_model import MapModel

    model = MapModel.from_arrays(np.ones((21, 21), dtype=bool))
    assert isinstance(model, NavigationMapProtocol)
