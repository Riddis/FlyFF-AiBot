"""Phase-9 section-14 requirement: proves the historical B4 tag
(``historical-reproduction-baseline-20260815``) still resolves to its exact
expected SHA, and that every ``scratchpad_historical_reproduction_guard.
REQUIRED_FILES`` member remains available there with byte-identical content
to the frozen 2026-08-15 snapshot -- even though Phase 9 legitimately made
those same paths unavailable at current HEAD (see
``test_navigation_dependency_boundary.py`` and the guard's own expected
fail-closed behavior, documented in
``docs/migration/PHASE9_NAVIGATION_OWNER_ANALYSIS.md`` section 6).
Historical reproduction is commit-addressed: this test is what makes that
claim checkable rather than merely asserted in prose.

Path note: ``REQUIRED_FILES`` lists CURRENT paths, which have moved twice
since the B4 tag. Phase 7's "mechanically collapse project roots" commit
(``bfc5c6d``, dated 2026-08-17) relocated these files without changing
their bytes; the 2026-08-21 repository cleanup then moved the three
``scratchpad_*.py`` entries again, from repository root into
``scratchpad/``, again without changing their bytes (see MISTAKES.md,
"repository hygiene / gitattributes-path drift"). The B4 tag (dated
2026-08-16, BEFORE either move) has them at their original nested path
under ``flyff_farming_simulator/``, with no ``scratchpad/`` segment
(confirmed via ``git ls-tree -r --name-only
historical-reproduction-baseline-20260815``). This test resolves that
original pre-collapse layout explicitly -- it is the content, not the
current path, that the frozen snapshot actually pins."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import simulator.scratchpad.scratchpad_historical_reproduction_guard as guard

REPO = Path(__file__).resolve().parents[1]
HISTORICAL_TAG = "historical-reproduction-baseline-20260815"
EXPECTED_TAG_SHA = "a90de59232b81753c1b2ea35b8990325c26674e5"

# The B4 tag predates the Phase-7 root collapse, the 2026-08-21
# scratchpad/ reorg, AND the later simulator/scratchpad/ nesting -- strip
# whichever scratchpad-prefix segment REQUIRED_FILES carries today (it
# didn't exist at that tag) before prepending the pre-collapse
# flyff_farming_simulator/ prefix.
def _strip_scratchpad_prefix(rel: str) -> str:
    for prefix in ("simulator/scratchpad/", "scratchpad/"):
        if rel.startswith(prefix):
            return rel[len(prefix):]
    return rel


PRE_COLLAPSE_PATH = {
    rel: f"flyff_farming_simulator/{_strip_scratchpad_prefix(rel)}" for rel in guard.REQUIRED_FILES
}


def _git_show(ref: str, path: str) -> bytes:
    result = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO, capture_output=True, check=True)
    return result.stdout


def test_historical_tag_resolves_to_the_exact_expected_sha() -> None:
    result = subprocess.run(["git", "rev-parse", HISTORICAL_TAG], cwd=REPO, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == EXPECTED_TAG_SHA


def test_required_historical_files_remain_available_at_the_tag_with_frozen_content() -> None:
    frozen = json.loads(guard.FROZEN_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    for rel in guard.REQUIRED_FILES:
        expected_hash = frozen[rel]
        content = _git_show(HISTORICAL_TAG, PRE_COLLAPSE_PATH[rel])
        actual_hash = hashlib.sha256(content).hexdigest()
        assert actual_hash == expected_hash, (
            f"{rel} at {HISTORICAL_TAG} ({PRE_COLLAPSE_PATH[rel]}) does not match the frozen snapshot: "
            f"expected {expected_hash}, got {actual_hash}"
        )


def test_frozen_snapshots_own_recorded_commit_is_an_ancestor_of_the_historical_tag() -> None:
    frozen = json.loads(guard.FROZEN_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    recorded_commit = frozen["git_commit"]
    result = subprocess.run(["git", "merge-base", "--is-ancestor", recorded_commit, HISTORICAL_TAG], cwd=REPO)
    assert result.returncode == 0, (
        f"frozen snapshot's recorded git_commit {recorded_commit} is not an ancestor of {HISTORICAL_TAG}"
    )


def test_current_head_guard_fails_closed_only_for_the_two_phase9_moved_files() -> None:
    """The expected-fail-closed case itself (Section 14): at current HEAD,
    the guard must refuse to run, and the ONLY reported mismatches must be
    the two files Phase 9 explicitly relocated. Any other mismatch is a
    STOP condition, not something this test should tolerate."""
    try:
        guard.verify_historical_snapshot()
    except RuntimeError as exc:
        message = str(exc)
        assert "simulator/kinodynamic_route_planner.py" in message
        assert "simulator/movement_kernel.py" in message
        for rel in guard.REQUIRED_FILES:
            if rel in ("simulator/kinodynamic_route_planner.py", "simulator/movement_kernel.py"):
                continue
            assert rel not in message, (
                f"unexpected additional historical-guard mismatch for {rel} -- "
                "only the two Phase-9-moved files should fail closed"
            )
    else:
        raise AssertionError(
            "expected the historical guard to fail closed at current HEAD "
            "(simulator/kinodynamic_route_planner.py and simulator/movement_kernel.py "
            "no longer exist there after Phase 9's move)"
        )
