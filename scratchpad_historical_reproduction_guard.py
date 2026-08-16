"""2026-08-15: fail-closed guard for the historical 840M/820M A-vs-D
router reproduction scripts, per explicit user instruction.

Problem: `scratchpad_legacy_qualified_selector.py`'s `select_persistent_
waypoint_legacy_pre_v2` is itself a frozen function body, but it still
CALLS current production helpers (`annotate_route_edges`, `_direct_hop_
min_clearance`, `DESIRED_CLEARANCE_CELLS`, etc.) imported live from
`simulator/kinodynamic_route_planner.py`. If those helpers are
legitimately improved later, the "frozen" historical control A would
silently start behaving differently without anyone touching the
archival function. Symmetrically, historical D (`select_persistent_
waypoint_experimental_invalid_hop_guard`) is now an alias to production
`select_persistent_waypoint` -- if production navigation evolves later,
historical D silently follows it too.

Rather than duplicating the entire dependency closure (annotate_route_
edges -> _arc_edge_check -> movement_kernel arc math, etc.) into another
frozen copy, this module verifies -- BEFORE any historical reproduction
runs -- that every file the reproduction actually depends on still has
the EXACT byte content it had at promotion time (2026-08-15, commit
`203ffb8`/`df361a5`). If anything has changed, it refuses to run rather
than silently producing a result that no longer means what its own
saved JSON says it means.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FROZEN_SNAPSHOT_PATH = ROOT / "evaluations" / "router_v2_historical_reproduction_snapshot_20260815.json"

# The selector/helper/constant snapshot (kinodynamic_route_planner.py +
# its own movement_kernel.py dependency), the evaluation harness
# (run_episode_general_router's selector_fn wiring + eval_obstacle_
# manifest), and the checkpoint -- everything select_persistent_
# waypoint_legacy_pre_v2 and the promoted production selector both
# transitively depend on for a historical A-vs-D run.
REQUIRED_FILES: tuple[str, ...] = (
    "simulator/kinodynamic_route_planner.py",
    "simulator/movement_kernel.py",
    "scratchpad_general_router_episode.py",
    "scratchpad_beginner_navigation_mix_pools.py",
    "scratchpad_legacy_qualified_selector.py",
    "models/generalized_waypoint_both_seed2_0051200.zip",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_historical_snapshot(*, extra_files: tuple[str, ...] = ()) -> None:
    """Raises RuntimeError (refuses to run) if any required file's
    CURRENT hash doesn't match the frozen 2026-08-15 snapshot. Call this
    at the top of any script that reproduces the historical 840M/820M
    A-vs-D router comparison, before loading the model or evaluating
    anything."""
    if not FROZEN_SNAPSHOT_PATH.exists():
        raise RuntimeError(
            f"REFUSE TO RUN: frozen snapshot file missing ({FROZEN_SNAPSHOT_PATH}). "
            "Cannot verify this is a faithful historical reproduction."
        )
    frozen = json.loads(FROZEN_SNAPSHOT_PATH.read_text(encoding="utf-8"))

    mismatches: list[tuple[str, str, str]] = []
    for rel in (*REQUIRED_FILES, *extra_files):
        path = ROOT / rel
        current = _sha256_file(path) if path.exists() else "MISSING"
        expected = frozen.get(rel)
        if expected is None:
            mismatches.append((rel, "NOT_IN_SNAPSHOT", current))
        elif current != expected:
            mismatches.append((rel, expected, current))

    if mismatches:
        detail = "\n".join(f"  {rel}:\n    expected {exp}\n    got      {got}" for rel, exp, got in mismatches)
        raise RuntimeError(
            "REFUSE TO RUN: This historical qualification requires the 2026-08-15 frozen "
            "evaluation implementation. Current code has changed since that snapshot:\n"
            f"{detail}\n\n"
            "Use the archived snapshot/commit (203ffb8 / df361a5) instead of rerunning this "
            "script against current code -- checkout that commit, or read the already-saved "
            "result JSON, rather than trusting a fresh run under drifted dependencies. If you "
            "deliberately intend to establish a NEW historical reproduction baseline, update "
            f"{FROZEN_SNAPSHOT_PATH.name} explicitly (do not do this silently)."
        )
