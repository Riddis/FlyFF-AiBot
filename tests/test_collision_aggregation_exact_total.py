"""Proves the zero-collision hard gate's aggregate collision count is an
EXACT sum of raw per-episode `distinct_contact_events`, not a
`round(median * n_episodes)` reconstruction.

That reconstruction is mathematically invalid: for per-episode counts
[0, 0, 1], the median is 0, so `round(0 * 3) == 0` despite one real
collision. Whenever collisions are concentrated in a minority of episodes
below the median, the reconstructed "total" silently rounds down to 0 --
exactly defeating the "zero collisions across every role" hard gate
(docs/PROJECT_GOALS.md section 2a). See MISTAKES.md 2026-08-23.

Covers both layers: the evaluator's own aggregation
(`simulator.milestone_evaluator._summarize_episodes`) and every canonical
runner's `_aggregate()`/`check_round_passes_absolute_bar()` that consumes
it (Beginner heldout/unseen_templates/challenge, Intermediate, Advanced).
"""

from __future__ import annotations

import importlib

import pytest

from simulator.milestone_evaluator import _summarize_episodes

BEGINNER = importlib.import_module("simulator.tools.RUN_CANONICAL_BEGINNER")
INTERMEDIATE = importlib.import_module("simulator.tools.RUN_CANONICAL_INTERMEDIATE")
ADVANCED = importlib.import_module("simulator.tools.RUN_CANONICAL_ADVANCED")

# Each pattern is one layout's raw per-episode distinct_contact_events.
# median-reconstruction (round(median * n_episodes)) is shown alongside the
# true sum to make explicit which patterns the old bug silently passed.
PATTERNS = [
    pytest.param([0, 0, 0], 0, id="all_clean"),
    pytest.param([0, 0, 1], 1, id="single_outlier_below_median"),
    pytest.param([0, 1, 0, 0], 1, id="single_outlier_in_larger_group"),
    pytest.param([1, 1, 0], 2, id="majority_collision"),
]


def _episode(distinct_contact_events: int) -> dict:
    """A minimal per-episode result dict shaped like
    `milestone_evaluator.run_episode`'s real return value -- only the keys
    `_summarize_episodes` actually accesses without `.get()`."""
    return {
        "_eva_target_counts": [],
        "_teacher_events": [],
        "_policy_events": [],
        "steering_persistent": False,
        "physical_stagnation": False,
        "productive_sustained_turn": False,
        "zero_kill": False,
        "kills_per_simulated_hour": 5000.0,
        "distinct_contact_events": distinct_contact_events,
    }


def _layout_summary(pattern: list[int]) -> dict:
    results = [_episode(c) for c in pattern]
    return _summarize_episodes("layout", results, None)


@pytest.mark.parametrize("pattern,expected_total", PATTERNS)
def test_evaluator_emits_exact_sum_not_median_reconstruction(pattern, expected_total):
    summary = _layout_summary(pattern)
    assert summary["total_distinct_contact_events"] == expected_total
    assert summary["total_distinct_contact_events"] == sum(pattern)


def test_single_outlier_below_median_exposes_the_bug_directly():
    """The concrete counterexample from the blocker report: [0, 0, 1] has
    median 0, so round(median * n_episodes) == 0 even though one collision
    genuinely occurred."""
    summary = _layout_summary([0, 0, 1])
    assert summary["distinct_contact_events"]["median"] == 0.0
    median_reconstruction = round(summary["distinct_contact_events"]["median"] * summary["n_episodes"])
    assert median_reconstruction == 0, "median reconstruction should be the (wrong) value this test guards against"
    assert summary["total_distinct_contact_events"] == 1


@pytest.mark.parametrize("pattern,expected_total", PATTERNS)
def test_beginner_aggregate_heldout_uses_exact_total(pattern, expected_total):
    report = {"layouts": {"L1": _layout_summary(pattern)}}
    agg = BEGINNER._aggregate(report)
    assert agg["total_collision_events"] == expected_total


@pytest.mark.parametrize("pattern,expected_total", PATTERNS)
def test_beginner_aggregate_unseen_templates_uses_exact_total(pattern, expected_total):
    report = {"layouts": {"L1": _layout_summary(pattern)}}
    agg = BEGINNER._aggregate(report)
    assert agg["total_collision_events"] == expected_total


@pytest.mark.parametrize("pattern,expected_total", PATTERNS)
def test_beginner_aggregate_challenge_uses_exact_total(pattern, expected_total):
    report = {"challenge_family": {"C1": _layout_summary(pattern)}}
    agg = BEGINNER._aggregate(report)
    assert agg["total_collision_events"] == expected_total


@pytest.mark.parametrize("pattern,expected_total", PATTERNS)
def test_intermediate_aggregate_uses_exact_total(pattern, expected_total):
    report = {"layouts": {"L1": _layout_summary(pattern)}}
    agg = INTERMEDIATE._aggregate(report)
    assert agg["total_collision_events"] == expected_total


@pytest.mark.parametrize("pattern,expected_total", PATTERNS)
def test_advanced_aggregate_uses_exact_total(pattern, expected_total):
    report = {"layouts": {"L1": _layout_summary(pattern)}}
    agg = ADVANCED._aggregate(report)
    assert agg["total_collision_events"] == expected_total


def _passing_agg_except_collisions(total_collision_events: int) -> dict:
    return {
        "total_collision_events": total_collision_events,
        "total_physical_stagnation_episodes": 0,
        "total_zero_kill_episodes": 0,
        "max_layout_contacts_per_100_distance": 1.0,
        "min_unique_cells_median": 1_000.0,
        "min_kills_per_hour_median": 5_000.0,
        "total_episodes": 10,
    }


@pytest.mark.parametrize("pattern,expected_total", PATTERNS)
def test_beginner_end_to_end_graduation_gate_rejects_any_real_collision(pattern, expected_total):
    """Full evaluator -> _aggregate -> check_round_passes_absolute_bar
    chain for Beginner's heldout role, using a realistic per-episode
    pattern rather than a hand-built aggregate dict."""
    heldout_report = {"layouts": {"L1": _layout_summary(pattern)}}
    heldout_agg = BEGINNER._aggregate(heldout_report)
    unseen_agg = _passing_agg_except_collisions(0)
    challenge_agg = _passing_agg_except_collisions(0)
    ok, reasons = BEGINNER.check_round_passes_absolute_bar(heldout_agg, unseen_agg, challenge_agg)
    if expected_total == 0:
        assert ok, reasons
    else:
        assert not ok
        assert any(f"total_collision_events={expected_total}" in r for r in reasons), reasons


@pytest.mark.parametrize("module", [INTERMEDIATE, ADVANCED])
@pytest.mark.parametrize("pattern,expected_total", PATTERNS)
def test_intermediate_advanced_end_to_end_graduation_gate_rejects_any_real_collision(module, pattern, expected_total):
    heldout_report = {"layouts": {"L1": _layout_summary(pattern)}}
    heldout_agg = module._aggregate(heldout_report)
    ok, reasons = module.check_round_passes_absolute_bar(heldout_agg)
    if expected_total == 0:
        assert ok, reasons
    else:
        assert not ok
        assert any(f"total_collision_events={expected_total}" in r for r in reasons), reasons
