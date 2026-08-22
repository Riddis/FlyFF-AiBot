"""Focused unit tests for the zero-collision hard graduation gate
(docs/PROJECT_GOALS.md section 2a) at Beginner/Intermediate/Advanced:
`total_collision_events` (`distinct_contact_events` -- genuine collision
EVENTS, not the raw `contacts_per_100_distance` tick-rate proxy) must be
exactly zero on the main/heldout bar for a round to pass, full stop,
regardless of every other metric. Exercises each script's own
`check_round_passes_absolute_bar`/`_check_bar` directly against synthetic
aggregate dicts -- no real rollout needed, since these are pure functions
of already-aggregated numbers.

Advanced in particular had this gate disabled entirely
(`AUTO_GRADUATION_ENABLED=False`) until this same change removed the
bypass -- these tests are the direct proof the restored gate actually
rejects a single collision, not just that the bypass flag is gone."""
from __future__ import annotations

import importlib

import pytest

BEGINNER = importlib.import_module("simulator.tools.RUN_CANONICAL_BEGINNER")
INTERMEDIATE = importlib.import_module("simulator.tools.RUN_CANONICAL_INTERMEDIATE")
ADVANCED = importlib.import_module("simulator.tools.RUN_CANONICAL_ADVANCED")


def _passing_agg(*, total_collision_events: int) -> dict:
    """An aggregate dict that clears every OTHER bar (stagnation, zero-kill,
    contacts/100, unique cells, kills/hour) -- isolates the assertions below
    to the collision-events gate specifically."""
    return {
        "total_collision_events": total_collision_events,
        "total_physical_stagnation_episodes": 0,
        "total_zero_kill_episodes": 0,
        "max_layout_contacts_per_100_distance": 1.0,
        "min_unique_cells_median": 1_000.0,
        "min_kills_per_hour_median": 5_000.0,
        "total_episodes": 10,
    }


def test_beginner_gate_passes_at_zero_collision_events():
    ok, reasons = BEGINNER.check_round_passes_absolute_bar(
        _passing_agg(total_collision_events=0), _passing_agg(total_collision_events=0), _passing_agg(total_collision_events=0),
    )
    assert ok, reasons
    assert reasons == []


def test_beginner_gate_fails_at_exactly_one_collision_event_on_heldout():
    ok, reasons = BEGINNER.check_round_passes_absolute_bar(
        _passing_agg(total_collision_events=1), _passing_agg(total_collision_events=0), _passing_agg(total_collision_events=0),
    )
    assert not ok
    assert any("heldout" in r and "total_collision_events=1" in r for r in reasons), reasons


def test_beginner_gate_fails_at_exactly_one_collision_event_on_unseen_templates():
    ok, reasons = BEGINNER.check_round_passes_absolute_bar(
        _passing_agg(total_collision_events=0), _passing_agg(total_collision_events=1), _passing_agg(total_collision_events=0),
    )
    assert not ok
    assert any("unseen_templates" in r and "total_collision_events=1" in r for r in reasons), reasons


def test_beginner_gate_allows_exactly_one_collision_event_on_challenge_only():
    """The challenge manifest's own explicitly looser bar
    (GRADUATION_MAX_COLLISION_EVENTS_CHALLENGE=1) is a real, documented
    exception to the otherwise-zero standard -- confirm it actually
    behaves looser, not just that the constant is set to 1."""
    ok, reasons = BEGINNER.check_round_passes_absolute_bar(
        _passing_agg(total_collision_events=0), _passing_agg(total_collision_events=0), _passing_agg(total_collision_events=1),
    )
    assert ok, reasons


def test_beginner_gate_fails_at_two_collision_events_on_challenge():
    ok, reasons = BEGINNER.check_round_passes_absolute_bar(
        _passing_agg(total_collision_events=0), _passing_agg(total_collision_events=0), _passing_agg(total_collision_events=2),
    )
    assert not ok
    assert any("challenge" in r and "total_collision_events=2" in r for r in reasons), reasons


@pytest.mark.parametrize("module", [INTERMEDIATE, ADVANCED])
def test_gate_passes_at_zero_collision_events(module):
    ok, reasons = module.check_round_passes_absolute_bar(_passing_agg(total_collision_events=0))
    assert ok, reasons
    assert reasons == []


@pytest.mark.parametrize("module", [INTERMEDIATE, ADVANCED])
def test_gate_fails_at_exactly_one_collision_event(module):
    ok, reasons = module.check_round_passes_absolute_bar(_passing_agg(total_collision_events=1))
    assert not ok
    assert any("total_collision_events=1" in r for r in reasons), reasons


def test_advanced_auto_graduation_bypass_flag_is_gone():
    """AUTO_GRADUATION_ENABLED was a hardcoded False bypass that skipped
    the collision gate entirely (disabled 2026-08-08 pending this exact
    fix) -- confirm the flag itself no longer exists, not merely that the
    gate happens to reject collisions when checked directly above."""
    assert not hasattr(ADVANCED, "AUTO_GRADUATION_ENABLED")
