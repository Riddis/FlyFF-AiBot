"""Stratified DAgger dataset builder for Phase 2's steering fine-tune.

Four mutually-exclusive categories, precedence high -> low (first match
wins), mined from RAW-policy (no recovery) rollouts on TRAINING-side
layouts/seeds -- never on early_heldout/early_challenge/
early_generator_validation, which stay untouched graduation exams, and never
on the navigation_calibration pool trajectories used to derive the
thresholds below (see MiningConfig / evaluations/navigation_calibration_results.json).

  1. persistent_wedge  -- contact-rate over the window at/above the elevated
                           threshold, whether or not it resolves into full
                           stagnation. Teacher-labeled.
  2. collision_onset    -- the few ticks immediately before/at the FIRST
                           contact in an otherwise-clear run. Teacher-labeled
                           (shows the turn that should have happened
                           earlier).
  3. safe_proximity     -- contact-rate stays exactly 0 over the window,
                           clearance dips into the danger-near band (covers
                           both approach-and-clear and sustained wall-
                           parallel travel -- both are "used available space
                           confidently without touching it"), displacement
                           stays near the clear-path rate throughout, not a
                           razor-thin survival. Labeled from the POLICY's own
                           action, not an automatic teacher overwrite (the
                           teacher isn't necessarily efficiency-optimal and
                           blind substitution risks teaching unnecessarily
                           wide margins) -- teacher disagreement is recorded
                           (`teacher_agrees`) for manual review, not used to
                           exclude the sample.
  4. ordinary            -- everything else. Teacher-labeled, standard DAgger
                           practice.

Sampling discipline: consecutive same-category ticks are grouped into one
"event"; at most `max_samples_per_event` representative ticks are drawn from
each event (not every tick in a 400-tick wedge), with additional per-episode
and per-layout-seed caps -- so no single pathological trajectory can
dominate the dataset.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from farming.actions import FarmingEvent
from .local_navigation_features import derive_physical_clearance_features
from .navigation_history import POLICY_INPUT_SIZE, NavigationHistoryWrapper
from .scripted_policies import scripted_command
from .synthetic import iter_variant_environments


@dataclass(frozen=True)
class MiningConfig:
    """Frozen calibration outputs from the navigation_calibration pool (12
    layouts x 6 templates x 5 seeds, 15k raw policy; see
    evaluations/navigation_calibration_results.json), not guesses:

    - expected_clear_path_displacement=1.79: median per-tick displacement on
      fully-open-clearance ticks (n=23326, reproduced twice: 1.7898/1.7920).
    - danger_near_clearance_max=0.6: the 90th percentile of clearance
      immediately preceding a real contact, AFTER extending
      SAMPLE_DISTANCES_CELLS to (1..5) -- the original (1,2,3) sampling had
      p90=1.0 (fully "clear" the tick before 10% of real contacts, i.e. not
      enough lead time), confirmed fixed by direct re-measurement (p90 0.6
      with the extended lookahead).
    - history_window/contact_rate_window=15: matches
      NavigationHistoryWrapper's CALIBRATED_HISTORY_WINDOW, so the steering
      feature and the mining threshold read the same window.

    persistent_contact_rate_threshold, razor_thin_clearance_min, and
    safe_displacement_fraction_min are informed judgment calls (not
    mechanically derived from a single statistic) grounded in the same
    data: contact-rate windows conditioned on containing >=1 contact skew
    heavily toward full saturation (median 1.0 at every window size tested,
    p25 ~0.35-0.47) -- 0.5 sits just above that p25, reasonably separating
    onset from sustained scraping.
    """

    history_window: int = 15
    contact_rate_window: int = 15
    persistent_contact_rate_threshold: float = 0.5
    danger_near_clearance_max: float = 0.6
    razor_thin_clearance_min: float = 0.15
    expected_clear_path_displacement: float = 1.79
    safe_displacement_fraction_min: float = 0.5
    max_samples_per_event: int = 1
    max_events_per_episode: int = 6
    max_events_per_layout_seed: int = 20


@dataclass
class _TickRecord:
    tick: int
    observation_925: np.ndarray
    contact: bool
    min_clearance: float
    displacement: float
    policy_steering: int
    policy_event: int
    teacher_steering: int
    teacher_event: int


CATEGORY_PRECEDENCE: tuple[str, ...] = ("persistent_wedge", "collision_onset", "safe_proximity", "ordinary")


def _classify_tick(records: list[_TickRecord], index: int, config: MiningConfig) -> str:
    window = config.contact_rate_window
    start = max(0, index - window + 1)
    recent = records[start : index + 1]
    contact_rate = float(np.mean([r.contact for r in recent])) if recent else 0.0

    if contact_rate >= config.persistent_contact_rate_threshold:
        return "persistent_wedge"

    current = records[index]
    if current.contact:
        prior_start = max(0, index - window)
        prior = records[prior_start:index]
        prior_rate = float(np.mean([r.contact for r in prior])) if prior else 0.0
        if prior_rate == 0.0:
            return "collision_onset"
        return "persistent_wedge"  # contact recurring but window not yet saturated -- still not "clean"

    if contact_rate == 0.0 and config.razor_thin_clearance_min < current.min_clearance <= config.danger_near_clearance_max:
        if current.displacement >= config.safe_displacement_fraction_min * config.expected_clear_path_displacement:
            return "safe_proximity"

    return "ordinary"


def _group_into_events(categories: list[str]) -> list[tuple[str, int, int]]:
    """Collapse consecutive same-category ticks into (category, start, end) events."""
    events: list[tuple[str, int, int]] = []
    if not categories:
        return events
    start = 0
    current = categories[0]
    for i in range(1, len(categories)):
        if categories[i] != current:
            events.append((current, start, i - 1))
            start = i
            current = categories[i]
    events.append((current, start, len(categories) - 1))
    return events


def _policy_forward(net: Any, observation_925: np.ndarray) -> tuple[int, int]:
    raw = observation_925[:923]
    with torch.no_grad():
        obs_t = torch.as_tensor(raw[None, :], dtype=torch.float32, device=net.device)
        dist = net.get_distribution(obs_t).distribution
        s = dist[0].probs[0].cpu().numpy()
        e = dist[1].probs[0].cpu().numpy()
    return int(s.argmax()), int(e.argmax())


def _roll_episode(
    curriculum_path: str, layout_name: str, *, seed: int, model: Any, episode_seconds: float, max_actions: int,
    history_window: int, expected_clear_path_displacement: float,
) -> list[_TickRecord]:
    net = model.policy
    entry, base_env = next(iter(iter_variant_environments(
        curriculum_path, stage="early", seed=seed, episode_steps=max_actions,
        episode_seconds=episode_seconds, variant_name=layout_name,
    )))
    env = NavigationHistoryWrapper(
        base_env, window=history_window, expected_clear_path_displacement=expected_clear_path_displacement,
    )
    observation, _info = env.reset(seed=seed)

    records: list[_TickRecord] = []
    prev_distance = 0.0
    prev_contacts = 0
    for tick in range(int(max_actions)):
        raw = np.asarray(observation, dtype=np.float32)[:923]
        teacher_command = scripted_command("obstacle_aware", base_env)
        policy_steering, policy_event = _policy_forward(net, np.asarray(observation, dtype=np.float32))
        clearance = derive_physical_clearance_features(raw)

        next_observation, _r, term, trunc, info = env.step(np.asarray([policy_steering, policy_event], dtype=np.int64))

        total_distance = float(info.get("total_distance_cells", prev_distance))
        total_contacts = int(info.get("contacts", prev_contacts))
        displacement_this_tick = total_distance - prev_distance
        contact_this_tick = total_contacts > prev_contacts
        prev_distance = total_distance
        prev_contacts = total_contacts

        records.append(_TickRecord(
            tick=tick, observation_925=np.asarray(observation, dtype=np.float32),
            contact=contact_this_tick, min_clearance=float(np.min(clearance)),
            displacement=displacement_this_tick, policy_steering=policy_steering, policy_event=policy_event,
            teacher_steering=int(teacher_command.steering), teacher_event=int(teacher_command.event),
        ))
        observation = next_observation
        if term or trunc:
            break

    env.close()
    return records


def mine_navigation_dataset(
    curriculum_path: str,
    layout_names: list[str],
    *,
    seeds: list[int],
    model: Any,
    episode_seconds: float,
    max_actions: int,
    config: MiningConfig = MiningConfig(),
) -> dict[str, Any]:
    """Roll RAW policy (no recovery) episodes across layout_names x seeds,
    classify every tick, group into events, and emit a capped, stratified,
    labeled dataset. `layout_names`/`curriculum_path` must be TRAINING-side
    (not early_heldout/early_challenge/early_generator_validation, and not
    the navigation_calibration pool)."""

    observations: list[np.ndarray] = []
    actions: list[tuple[int, int]] = []
    categories: list[str] = []
    layout_ids: list[int] = []
    episode_ids: list[int] = []
    teacher_agrees_flags: list[bool] = []
    category_counts: dict[str, int] = {c: 0 for c in CATEGORY_PRECEDENCE}
    episode_counter = 0

    for layout_id, layout_name in enumerate(layout_names):
        layout_event_count = 0
        for seed in seeds:
            if layout_event_count >= config.max_events_per_layout_seed:
                break
            records = _roll_episode(
                curriculum_path, layout_name, seed=seed, model=model,
                episode_seconds=episode_seconds, max_actions=max_actions,
                history_window=config.history_window,
                expected_clear_path_displacement=config.expected_clear_path_displacement,
            )
            tick_categories = [_classify_tick(records, i, config) for i in range(len(records))]
            events = _group_into_events(tick_categories)

            episode_event_count = 0
            for category, start, end in events:
                if episode_event_count >= config.max_events_per_episode:
                    break
                if layout_event_count >= config.max_events_per_layout_seed:
                    break
                span = list(range(start, end + 1))
                sample_indices = span[-config.max_samples_per_event :]  # representative tick(s), end of the event
                for i in sample_indices:
                    record = records[i]
                    if category in ("persistent_wedge", "collision_onset", "ordinary"):
                        label = (record.teacher_steering, record.teacher_event)
                        agrees = (record.teacher_steering, record.teacher_event) == (
                            record.policy_steering, record.policy_event,
                        )
                    else:  # safe_proximity
                        label = (record.policy_steering, record.policy_event)
                        agrees = (record.teacher_steering, record.teacher_event) == (
                            record.policy_steering, record.policy_event,
                        )
                    observations.append(record.observation_925)
                    actions.append(label)
                    categories.append(category)
                    layout_ids.append(layout_id)
                    episode_ids.append(episode_counter)
                    teacher_agrees_flags.append(agrees)
                    category_counts[category] += 1
                episode_event_count += 1
                layout_event_count += 1
            episode_counter += 1

    return {
        "observations": np.asarray(observations, dtype=np.float32) if observations else np.zeros((0, POLICY_INPUT_SIZE), dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64) if actions else np.zeros((0, 2), dtype=np.int64),
        "categories": categories,
        "layout_index": np.asarray(layout_ids, dtype=np.int64),
        "episode_index": np.asarray(episode_ids, dtype=np.int64),
        "teacher_agrees": np.asarray(teacher_agrees_flags, dtype=bool),
        "category_counts": category_counts,
        "layout_names": list(layout_names),
    }
