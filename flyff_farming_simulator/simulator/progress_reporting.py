"""Shared progress reporting for long-running training/collection loops.

Waiting on a background job with no visible progress is the specific
complaint this addresses: every long loop in the Basic/Beginner training
entrypoints should print a line at a bounded interval (never silent for
minutes), whether the job runs in the background (read the redirected
stdout file) or in a foreground console (see it live).
"""

from __future__ import annotations

import sys
import time
from typing import Any

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

    def _on_training_start(self) -> None:
        self._start_time = time.monotonic()
        self._last_print_time = self._start_time
        self._printed_first = False
        self._printed_last = False

    def _on_step(self) -> bool:
        now = time.monotonic()
        # SB3 collects a full n_steps rollout before checking the stopping
        # condition, so num_timesteps can overshoot total_timesteps for a
        # small requested total (n_steps > total_timesteps) -- print the
        # first overshoot call once (so completion is still visible), then
        # go back to normal throttling instead of printing every remaining
        # step in the batch.
        is_last = self.num_timesteps >= self.total_timesteps
        force_print = is_last and not self._printed_last
        if self._printed_first and not force_print and (now - self._last_print_time) < self.min_interval_seconds:
            return True
        self._last_print_time = now
        self._printed_first = True
        if is_last:
            self._printed_last = True
        elapsed = now - self._start_time
        rate = self.num_timesteps / elapsed if elapsed > 1e-6 else 0.0
        remaining = max(0, self.total_timesteps - self.num_timesteps)
        eta_seconds = remaining / rate if rate > 1e-9 else float("inf")
        eta_text = f"{eta_seconds:6.0f}s" if eta_seconds < 36000 else "  n/a"
        recent_rewards = self._recent_episode_stat("r")
        recent_lengths = self._recent_episode_stat("l")
        stats = ""
        if recent_rewards is not None:
            stats = f" ep_reward_mean={recent_rewards:8.3f} ep_len_mean={recent_lengths:6.1f}"
        print(
            f"[{self.label}] {self.num_timesteps}/{self.total_timesteps} "
            f"({100.0 * self.num_timesteps / max(1, self.total_timesteps):5.1f}%) "
            f"elapsed={elapsed:6.0f}s rate={rate:7.1f}steps/s eta={eta_text}{stats}",
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
