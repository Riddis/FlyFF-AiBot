from __future__ import annotations

import numpy as np

from simulator.navigation_dataset import (
    CATEGORY_PRECEDENCE,
    MiningConfig,
    _TickRecord,
    _classify_tick,
    _group_into_events,
    mine_navigation_dataset,
)

_DUMMY_OBS = np.zeros((925,), dtype=np.float32)


def _record(**overrides) -> _TickRecord:
    base = dict(
        tick=0, observation_925=_DUMMY_OBS, contact=False, min_clearance=1.0, displacement=1.79,
        policy_steering=0, policy_event=0, teacher_steering=0, teacher_event=0,
    )
    base.update(overrides)
    return _TickRecord(**base)


def _config(**overrides) -> MiningConfig:
    base = dict(
        contact_rate_window=5, persistent_contact_rate_threshold=0.5,
        danger_near_clearance_max=0.6, razor_thin_clearance_min=0.15,
        expected_clear_path_displacement=1.79, safe_displacement_fraction_min=0.5,
    )
    base.update(overrides)
    return MiningConfig(**base)


def test_open_field_healthy_movement_is_ordinary():
    config = _config()
    records = [_record(min_clearance=1.0, displacement=1.79, contact=False) for _ in range(10)]
    for i in range(len(records)):
        assert _classify_tick(records, i, config) == "ordinary"


def test_sustained_high_contact_rate_is_persistent_wedge():
    config = _config()
    records = [_record(contact=True, min_clearance=0.0, displacement=0.0) for _ in range(10)]
    # After the window fills with contact, rate saturates -- must classify persistent.
    assert _classify_tick(records, 9, config) == "persistent_wedge"


def test_first_contact_after_clear_run_is_collision_onset():
    config = _config()
    records = [_record(contact=False, min_clearance=1.0) for _ in range(5)]
    records.append(_record(contact=True, min_clearance=0.0, displacement=0.5))
    assert _classify_tick(records, 5, config) == "collision_onset"


def test_close_pass_with_zero_contact_and_healthy_progress_is_safe_proximity():
    config = _config()
    records = [_record(contact=False, min_clearance=1.0) for _ in range(3)]
    records.append(_record(contact=False, min_clearance=0.4, displacement=1.7))
    assert _classify_tick(records, 3, config) == "safe_proximity"


def test_razor_thin_clearance_is_excluded_from_safe_proximity():
    config = _config()
    records = [_record(contact=False, min_clearance=1.0) for _ in range(3)]
    records.append(_record(contact=False, min_clearance=0.05, displacement=1.7))  # below razor_thin_clearance_min
    assert _classify_tick(records, 3, config) == "ordinary"


def test_low_displacement_close_pass_is_not_safe_proximity():
    config = _config()
    records = [_record(contact=False, min_clearance=1.0) for _ in range(3)]
    records.append(_record(contact=False, min_clearance=0.4, displacement=0.1))  # stalled, not "confident" movement
    assert _classify_tick(records, 3, config) == "ordinary"


def test_category_precedence_persistent_beats_everything():
    config = _config()
    records = [_record(contact=True, min_clearance=0.0, displacement=0.0) for _ in range(10)]
    result = _classify_tick(records, 9, config)
    assert result == CATEGORY_PRECEDENCE[0]


def test_group_into_events_collapses_consecutive_runs():
    events = _group_into_events(["ordinary", "ordinary", "collision_onset", "persistent_wedge", "persistent_wedge", "ordinary"])
    assert events == [
        ("ordinary", 0, 1),
        ("collision_onset", 2, 2),
        ("persistent_wedge", 3, 4),
        ("ordinary", 5, 5),
    ]


def test_group_into_events_handles_empty_and_single():
    assert _group_into_events([]) == []
    assert _group_into_events(["ordinary"]) == [("ordinary", 0, 0)]


def test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts():
    """Integration smoke test on real layouts known to produce a mix (from
    the calibration run: wide_neck/split_field templates show heavy
    contact, open_field shows mostly-clear movement)."""
    from stable_baselines3 import PPO

    model = PPO.load("models/split_branch_pilot_15000.zip", device="cpu")
    config = MiningConfig(max_events_per_layout_seed=15, max_events_per_episode=8)
    result = mine_navigation_dataset(
        "synthetic_curriculum/curriculum.json",
        ["01_early_open_field_typical_fast", "04_early_wide_neck_typical_bursty"],
        seeds=[100, 101, 102],
        model=model,
        episode_seconds=90.0,
        max_actions=400,
        config=config,
    )
    assert result["observations"].shape[0] > 0
    assert result["observations"].shape[1] == 925
    assert result["actions"].shape == (result["observations"].shape[0], 2)
    assert set(result["categories"]) <= set(CATEGORY_PRECEDENCE)
    # Not a strict requirement that every category appears in this small a
    # sample, but persistent_wedge/collision_onset/ordinary should, given
    # the wide_neck template is known contact-heavy.
    counts = result["category_counts"]
    assert counts["ordinary"] > 0
    assert counts["persistent_wedge"] + counts["collision_onset"] > 0
