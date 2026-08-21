"""Proves tests/helpers/beginner_navigation_mix_harness.py's copied
definitions (_final_native_for/_reconstruct_single_wall_world/
_reconstruct_two_wall_world/eval_obstacle_manifest/load_manifest) are
structurally identical to the frozen scratchpad_beginner_navigation_mix_pools.py
they were copied from -- so the two can never silently drift apart. That
frozen file is itself one of scratchpad_historical_reproduction_guard.py's
REQUIRED_FILES and must never be edited; this test does not compare import
statements, only the copied definitions' own bodies, since the harness
substitutes a small number of mechanically-necessary import sources
(tests.helpers.router_qualification_harness instead of the now-unimportable
scratchpad_general_router_episode)."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN_SOURCE = REPO / "scratchpad" / "scratchpad_beginner_navigation_mix_pools.py"
HARNESS_SOURCE = REPO / "tests" / "helpers" / "beginner_navigation_mix_harness.py"
EXPECTED_FROZEN_SHA256 = "dd9a4630c30059ce809ed8320c24b095eb9b3e4fe99b76a4e271a2404be84156"
COPIED_DEFINITIONS = (
    "_final_native_for", "_reconstruct_single_wall_world", "_reconstruct_two_wall_world",
    "eval_obstacle_manifest", "load_manifest",
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
        f"scratchpad_beginner_navigation_mix_pools.py changed unexpectedly: "
        f"expected {EXPECTED_FROZEN_SHA256}, got {actual}"
    )


def test_all_copied_definitions_are_present_in_both_files() -> None:
    frozen = _definitions(FROZEN_SOURCE)
    harness = _definitions(HARNESS_SOURCE)
    assert set(frozen) == set(COPIED_DEFINITIONS)
    assert set(harness) == set(COPIED_DEFINITIONS)


def test_copied_definitions_are_ast_identical_to_the_frozen_source() -> None:
    """The only permitted difference between the two files is which module
    build_multi_wall_world/run_episode_general_router/summarize_general_router/
    TwoWallSpec are imported from, which lives outside every one of these
    five definitions -- so an AST-level (not text-level) comparison of just
    the definitions themselves, with line/column metadata stripped, proves
    the copy is behaviorally verbatim."""
    frozen = _definitions(FROZEN_SOURCE)
    harness = _definitions(HARNESS_SOURCE)
    for name in COPIED_DEFINITIONS:
        frozen_dump = ast.dump(frozen[name], annotate_fields=True, include_attributes=False)
        harness_dump = ast.dump(harness[name], annotate_fields=True, include_attributes=False)
        assert frozen_dump == harness_dump, f"{name} has drifted from the frozen source"
