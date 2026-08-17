"""Proves tests/helpers/router_qualification_harness.py's
build_multi_wall_world/GeneralRouterEpisodeResult/run_episode_general_router/
summarize_general_router are structurally identical to the frozen
scratchpad_general_router_episode.py they were copied from -- so the two can
never silently drift apart. Only the router import
(simulator.kinodynamic_route_planner -> navigation.kinodynamic_route_planner,
mechanically required by Phase 9's navigation extraction) is expected to
differ; this test does not compare import statements, only the copied
definitions' own bodies."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN_SOURCE = REPO / "scratchpad_general_router_episode.py"
HARNESS_SOURCE = REPO / "tests" / "helpers" / "router_qualification_harness.py"
EXPECTED_FROZEN_SHA256 = "dc7ce7ff6940c1f4e98fad5b66fecb7f58c1616b3d0693935db2d7b3f4576f39"
COPIED_DEFINITIONS = (
    "build_multi_wall_world", "GeneralRouterEpisodeResult", "run_episode_general_router",
    "summarize_general_router",
)


def _definitions(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in COPIED_DEFINITIONS
    }


def test_frozen_scratchpad_bytes_are_unchanged() -> None:
    """Guards this test's own premise: if this ever fails, the frozen
    historical-guard file itself changed (a STOP condition on its own,
    unrelated to Phase 9), not something this test should silently adapt to."""
    actual = hashlib.sha256(FROZEN_SOURCE.read_bytes()).hexdigest()
    assert actual == EXPECTED_FROZEN_SHA256, (
        f"scratchpad_general_router_episode.py changed unexpectedly: "
        f"expected {EXPECTED_FROZEN_SHA256}, got {actual}"
    )


def test_all_copied_definitions_are_present_in_both_files() -> None:
    frozen = _definitions(FROZEN_SOURCE)
    harness = _definitions(HARNESS_SOURCE)
    assert set(frozen) == set(COPIED_DEFINITIONS)
    assert set(harness) == set(COPIED_DEFINITIONS)


def test_copied_definitions_are_ast_identical_to_the_frozen_source() -> None:
    """The only permitted difference between the two files is the router
    import line, which lives outside every one of these three definitions
    -- so an AST-level (not text-level) comparison of just the definitions
    themselves, with line/column metadata stripped, proves the copy is
    behaviorally verbatim."""
    frozen = _definitions(FROZEN_SOURCE)
    harness = _definitions(HARNESS_SOURCE)
    for name in COPIED_DEFINITIONS:
        frozen_dump = ast.dump(frozen[name], annotate_fields=True, include_attributes=False)
        harness_dump = ast.dump(harness[name], annotate_fields=True, include_attributes=False)
        assert frozen_dump == harness_dump, f"{name} has drifted from the frozen source"
