"""Shared progress reporting for long-running training/collection loops.

Waiting on a background job with no visible progress is the specific
complaint this addresses: every long loop in the Basic/Beginner training
entrypoints should print a line at a bounded interval (never silent for
minutes), whether the job runs in the background (read the redirected
stdout file) or in a foreground console (see it live).
"""

from __future__ import annotations

from collections import Counter
import time

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class ProgressPrinter:
    """Manual progress reporter for plain Python loops (BC epochs, DAgger
    collection episodes) -- print a line at most every `min_interval_seconds`,
    always on the first and last call regardless of interval, with
    throughput and ETA."""

    def __init__(self, total: int, *, label: str, min_interval_seconds: float = 10.0) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.min_interval_seconds = float(min_interval_seconds)
        self.start_time = time.monotonic()
        self._last_print_time = 0.0
        self._done = 0

    def update(self, done: int, *, extra: str = "") -> None:
        self._done = done
        now = time.monotonic()
        is_last = done >= self.total
        if not is_last and (now - self._last_print_time) < self.min_interval_seconds:
            return
        self._last_print_time = now
        elapsed = now - self.start_time
        rate = done / elapsed if elapsed > 1e-6 else 0.0
        remaining = self.total - done
        eta_seconds = remaining / rate if rate > 1e-9 else float("inf")
        eta_text = f"{eta_seconds:6.0f}s" if eta_seconds < 36000 else "  n/a"
        extra_text = f" {extra}" if extra else ""
        print(
            f"[{self.label}] {done}/{self.total} ({100.0 * done / self.total:5.1f}%) "
            f"elapsed={elapsed:6.0f}s rate={rate:6.2f}/s eta={eta_text}{extra_text}",
            flush=True,
        )

    def finish(self, *, extra: str = "") -> None:
        self.update(self.total, extra=extra)


class SB3ProgressCallback(BaseCallback):
    """SB3 training callback: print a progress line at most every
    `min_interval_seconds`, always on the very first and very last call."""

    def __init__(self, total_timesteps: int, *, label: str, min_interval_seconds: float = 15.0) -> None:
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self.label = label
        self.min_interval_seconds = float(min_interval_seconds)
        self._start_time = 0.0
        self._last_print_time = 0.0
        self._printed_first = False
        self._printed_last = False
        self._start_num_timesteps = 0

    def _on_training_start(self) -> None:
        self._start_time = time.monotonic()
        self._last_print_time = self._start_time
        self._printed_first = False
        self._printed_last = False
        # learn(..., reset_num_timesteps=False) (every chunk after the first
        # in a resumed lineage) leaves self.num_timesteps holding the
        # model's LIFETIME cumulative count, not this chunk's -- captured
        # here so progress/ETA below are chunk-relative. Without this a
        # resumed round starting from e.g. 260112 lifetime steps against a
        # 10000 per-chunk total printed "260112/10000 (2601%)", a real bug
        # (not just cosmetic -- rate/eta were computed off the same wrong
        # numerator), found during the 2026-08-08 collision-metric review.
        self._start_num_timesteps = self.num_timesteps

    def _on_step(self) -> bool:
        now = time.monotonic()
        chunk_timesteps = self.num_timesteps - self._start_num_timesteps
        # SB3 collects a full n_steps rollout before checking the stopping
        # condition, so chunk_timesteps can overshoot total_timesteps for a
        # small requested total (n_steps > total_timesteps) -- print the
        # first overshoot call once (so completion is still visible), then
        # go back to normal throttling instead of printing every remaining
        # step in the batch.
        is_last = chunk_timesteps >= self.total_timesteps
        force_print = is_last and not self._printed_last
        if self._printed_first and not force_print and (now - self._last_print_time) < self.min_interval_seconds:
            return True
        self._last_print_time = now
        self._printed_first = True
        if is_last:
            self._printed_last = True
        requested_progress = min(chunk_timesteps, self.total_timesteps)
        elapsed = now - self._start_time
        rate = chunk_timesteps / elapsed if elapsed > 1e-6 else 0.0
        remaining = max(0, self.total_timesteps - requested_progress)
        eta_seconds = remaining / rate if rate > 1e-9 else float("inf")
        eta_text = f"{eta_seconds:6.0f}s" if eta_seconds < 36000 else "  n/a"
        recent_rewards = self._recent_episode_stat("r")
        recent_lengths = self._recent_episode_stat("l")
        stats = ""
        if recent_rewards is not None:
            stats = f" ep_reward_mean={recent_rewards:8.3f} ep_len_mean={recent_lengths:6.1f}"
        print(
            f"[{self.label}] requested={requested_progress}/{self.total_timesteps} "
            f"({100.0 * requested_progress / max(1, self.total_timesteps):5.1f}%) "
            f"elapsed={elapsed:6.0f}s rate={rate:7.1f}steps/s eta={eta_text}{stats} "
            f"(actual_rollout_aligned={chunk_timesteps} lifetime={self.num_timesteps})",
            flush=True,
        )
        return True

    def _recent_episode_stat(self, key: str) -> float | None:
        buffer = getattr(self.model, "ep_info_buffer", None)
        if not buffer:
            return None
        values = [ep[key] for ep in buffer if key in ep]
        if not values:
            return None
        return float(sum(values) / len(values))


class CurriculumRolloutMetricsCallback(BaseCallback):
    """Cheap per-rollout aggregates for the farming PPO composition.

    The wrapper already puts compact action/navigation facts in ``info``;
    this callback only counts them and emits one TensorBoard summary per
    rollout. It deliberately does not retain per-tick forensic traces.
    """

    def __init__(self) -> None:
        super().__init__()
        self._target_actions: Counter[int] = Counter()
        self._event_actions: Counter[int] = Counter()
        self._steps = 0
        self._invalid_targets = 0
        self._planner_failures = 0
        self._planner_attempts = 0
        self._distinct_contacts = 0
        self._in_contact: list[bool] = []
        self._previous_contacts: list[int] = []
        self._training_start_time = 0.0
        self._training_start_timesteps = 0

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.training_env, "num_envs", 1))
        self._in_contact = [False] * n_envs
        self._previous_contacts = [0] * n_envs
        self._training_start_time = time.monotonic()
        self._training_start_timesteps = int(self.num_timesteps)

    def _on_rollout_start(self) -> None:
        self._target_actions.clear()
        self._event_actions.clear()
        self._steps = 0
        self._invalid_targets = 0
        self._planner_failures = 0
        self._planner_attempts = 0
        self._distinct_contacts = 0

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals.get("actions", []))
        infos = self.locals.get("infos", [])
        dones = np.asarray(self.locals.get("dones", []), dtype=np.bool_).reshape(-1)
        if actions.ndim == 1 and actions.size == 2:
            actions = actions.reshape(1, 2)
        if actions.ndim >= 2 and actions.shape[-1] >= 2:
            for target_action, event_action in actions[:, :2]:
                self._target_actions[int(target_action)] += 1
                self._event_actions[int(event_action)] += 1

        for env_index, info in enumerate(infos):
            self._steps += 1
            self._invalid_targets += int(bool(info.get("invalid_target_selection", False)))
            attempted = bool(info.get("steering_plan_attempted", False))
            failed = bool(info.get("steering_planner_failure", False))
            self._planner_attempts += int(attempted)
            self._planner_failures += int(failed)

            contacts = int(info.get("contacts", 0) or 0)
            contact_this_tick = contacts > self._previous_contacts[env_index]
            if contact_this_tick and not self._in_contact[env_index]:
                self._distinct_contacts += 1
            self._in_contact[env_index] = contact_this_tick
            self._previous_contacts[env_index] = contacts
            if env_index < len(dones) and bool(dones[env_index]):
                self._in_contact[env_index] = False
                self._previous_contacts[env_index] = 0
        return True

    def _on_rollout_end(self) -> None:
        total_actions = max(1, sum(self._target_actions.values()))
        for action, count in sorted(self._target_actions.items()):
            self.logger.record(f"farming/target_action_{action}_count", count)
        for action, count in sorted(self._event_actions.items()):
            self.logger.record(f"farming/event_action_{action}_count", count)
        self.logger.record("farming/target_keep_rate", self._target_actions.get(0, 0) / total_actions)
        self.logger.record("farming/invalid_target_selection_rate", self._invalid_targets / max(1, self._steps))
        self.logger.record("navigation/planner_failure_count", self._planner_failures)
        self.logger.record(
            "navigation/planner_failure_rate", self._planner_failures / max(1, self._planner_attempts),
        )
        self.logger.record("navigation/distinct_contact_events", self._distinct_contacts)
        elapsed = max(1e-9, time.monotonic() - self._training_start_time)
        elapsed_steps = int(self.num_timesteps) - self._training_start_timesteps
        self.logger.record("diagnostics/fps", elapsed_steps / elapsed)
        self.logger.record("time/cumulative_timesteps", int(self.num_timesteps))
