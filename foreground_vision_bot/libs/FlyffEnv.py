from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import hypot
from time import monotonic, sleep
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from libs.ActionExecutor import ActionExecutor, BotAction
from libs.ObservationBuilder import ObservationBuilder

MobReader = Callable[[], Iterable[Sequence[int | float]]]
KillReader = Callable[[], int | None]


@dataclass(frozen=True)
class FlyffEnvConfig:
    observation_delay: float = 0.05
    eva_cooldown_seconds: float = 2.0

    max_episode_seconds: float = 300.0
    max_no_kill_seconds: float = 60.0

    base_kill_reward: float = 1.0
    group_bonus_per_extra_kill: float = 0.05
    group_multiplier_cap: float = 1.5
    max_kill_delta: int = 40

    time_penalty_per_second: float = 0.01
    invalid_eva_penalty: float = 0.10
    eva_miss_penalty: float = 0.20

    eva_check_timeout_seconds: float = 0.30
    eva_check_poll_seconds: float = 0.05

    density_reward_scale: float = 0.01
    max_density_reward: float = 0.20
    inner_density_weight: float = 3.0
    middle_density_weight: float = 2.0
    outer_density_weight: float = 1.0

    # Dead mob names can remain detectable briefly after a successful EVA.
    # Suppress detections matching mobs that were inside the EVA radius.
    despawn_filter_seconds: float = 1.20
    despawn_match_radius_px: float = 35.0


class FlyffEnv(gym.Env):
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        action_executor: ActionExecutor,
        observation_builder: ObservationBuilder,
        read_mobs: MobReader,
        read_kills: KillReader,
        config: FlyffEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.action_executor = action_executor
        self.observation_builder = observation_builder
        self.read_mobs = read_mobs
        self.read_kills = read_kills
        self.config = config or FlyffEnvConfig()

        self._allowed_actions = (
            BotAction.MOVE_FORWARD,
            BotAction.FORWARD_LEFT,
            BotAction.FORWARD_RIGHT,
            BotAction.CAST_EVA,
        )
        self.action_space = spaces.Discrete(len(self._allowed_actions))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_builder.observation_size,),
            dtype=np.float32,
        )

        self._episode_started_at = 0.0
        self._last_kill_at = 0.0
        self._last_cast_at: float | None = None
        self._previous_kills: int | None = None
        self._previous_mobs: list[tuple[float, float]] = []

        self._stale_mob_positions: list[tuple[float, float]] = []
        self._stale_mobs_until = 0.0

        self._episode_reward = 0.0
        self._episode_kills = 0
        self._episode_steps = 0
        self._episode_eva_casts = 0
        self._episode_eva_successes = 0
        self._episode_eva_misses = 0
        self._episode_eva_unknown = 0
        self._episode_zero_nearby_eva_casts = 0
        self._episode_invalid_eva_requests = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.action_executor.stop_movement()

        now = monotonic()
        self._episode_started_at = now
        self._last_kill_at = now
        self._last_cast_at = None
        self._stale_mob_positions = []
        self._stale_mobs_until = 0.0

        self._previous_kills = self._read_valid_kills()
        self._previous_mobs = self._read_mob_positions()

        self._episode_reward = 0.0
        self._episode_kills = 0
        self._episode_steps = 0
        self._episode_eva_casts = 0
        self._episode_eva_successes = 0
        self._episode_eva_misses = 0
        self._episode_eva_unknown = 0
        self._episode_zero_nearby_eva_casts = 0
        self._episode_invalid_eva_requests = 0

        cooldown_fraction = self._eva_cooldown_fraction()
        observation = self._build_observation(
            self._previous_mobs,
            cooldown_fraction,
        )

        return observation, {
            "kills": self._previous_kills,
            "kill_delta": 0,
            "visible_mobs": len(self._previous_mobs),
            "nearby_mobs": self._outer_nearby_count(self._previous_mobs),
            "density_score": self._density_score(self._previous_mobs),
            "eva_cooldown_fraction": cooldown_fraction,
            "reward_components": self._empty_reward_components(),
        }

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        step_started_at = monotonic()

        action_index = int(action)
        if not 0 <= action_index < len(self._allowed_actions):
            raise ValueError(f"Invalid action index: {action_index}")

        selected_action = self._allowed_actions[action_index]
        before_mobs = self._previous_mobs
        density_before = self._density_score(before_mobs)

        cooldown_before = self._eva_cooldown_fraction()
        eva_ready = cooldown_before >= 0.999
        invalid_eva = selected_action == BotAction.CAST_EVA and not eva_ready

        if invalid_eva:
            # Preserve currently held movement.
            self._episode_invalid_eva_requests += 1
        else:
            self.action_executor.execute(selected_action)
            if selected_action == BotAction.CAST_EVA:
                self._last_cast_at = monotonic()
                self._episode_eva_casts += 1
                if self._outer_nearby_count(before_mobs) == 0:
                    self._episode_zero_nearby_eva_casts += 1

        sleep(max(0.0, self.config.observation_delay))

        after_mobs = self._read_mob_positions()

        if selected_action == BotAction.CAST_EVA and not invalid_eva:
            kill_sample = self._read_eva_kill_sample()
        else:
            kill_sample = self._read_kill_sample()

        if kill_sample is None:
            current_kills = int(self._previous_kills or 0)
            raw_kill_delta = 0
        else:
            current_kills = kill_sample
            raw_kill_delta = self._calculate_kill_delta(current_kills)

        valid_kill_delta = int(raw_kill_delta)
        components = self._empty_reward_components()

        eva_success = False
        eva_miss = False
        eva_unknown = False
        eva_kills = 0

        if selected_action == BotAction.CAST_EVA and not invalid_eva:
            if kill_sample is None:
                eva_unknown = True
                self._episode_eva_unknown += 1
            elif valid_kill_delta > 0:
                eva_success = True
                eva_kills = valid_kill_delta
                self._episode_eva_successes += 1

                # Mark mobs that were inside EVA range as temporarily stale.
                self._begin_despawn_filter(before_mobs)
                after_mobs = self._filter_stale_mobs(after_mobs)
            else:
                eva_miss = True
                self._episode_eva_misses += 1
                components["eva_miss"] = -self.config.eva_miss_penalty

        components["kill"] = self._kill_reward(valid_kill_delta)

        if selected_action != BotAction.CAST_EVA:
            density_after = self._density_score(after_mobs)
            density_delta = density_after - density_before
            components["density"] = float(
                np.clip(
                    density_delta * self.config.density_reward_scale,
                    -self.config.max_density_reward,
                    self.config.max_density_reward,
                )
            )

        if invalid_eva:
            components["invalid_eva"] = -self.config.invalid_eva_penalty

        now = monotonic()
        elapsed_seconds = max(now - step_started_at, 0.0)
        components["time"] = -elapsed_seconds * self.config.time_penalty_per_second

        reward = float(sum(components.values()))

        if valid_kill_delta > 0:
            self._last_kill_at = now

        self._episode_steps += 1
        self._episode_kills += valid_kill_delta
        self._episode_reward += reward
        self._previous_mobs = after_mobs

        terminated = False
        truncated = (
            now - self._episode_started_at >= self.config.max_episode_seconds
            or now - self._last_kill_at >= self.config.max_no_kill_seconds
        )

        cooldown_fraction = self._eva_cooldown_fraction()
        observation = self._build_observation(after_mobs, cooldown_fraction)

        info = {
            "kills": current_kills,
            "kill_delta": valid_kill_delta,
            "raw_kill_delta": raw_kill_delta,
            "visible_mobs": len(after_mobs),
            "nearby_mobs": self._outer_nearby_count(after_mobs),
            "density_score": self._density_score(after_mobs),
            "eva_cooldown_fraction": cooldown_fraction,
            "eva_ready": cooldown_fraction >= 0.999,
            "invalid_eva": invalid_eva,
            "eva_success": eva_success,
            "eva_miss": eva_miss,
            "eva_unknown": eva_unknown,
            "eva_kills": eva_kills,
            "eva_outcome_pending": False,
            "action_index": action_index,
            "action_name": selected_action.name,
            "reward_components": components,
            "episode_reward_live": self._episode_reward,
            "episode_kills_live": self._episode_kills,
            "episode_steps_live": self._episode_steps,
            "episode_eva_casts_live": self._episode_eva_casts,
            "episode_eva_successes_live": self._episode_eva_successes,
            "episode_eva_misses_live": self._episode_eva_misses,
            "episode_eva_unknown_live": self._episode_eva_unknown,
            "episode_zero_nearby_eva_casts_live": (self._episode_zero_nearby_eva_casts),
            "episode_invalid_eva_requests_live": (self._episode_invalid_eva_requests),
            "episode_seconds_live": now - self._episode_started_at,
        }

        if truncated or terminated:
            info["episode_summary"] = {
                "reward": self._episode_reward,
                "kills": self._episode_kills,
                "steps": self._episode_steps,
                "seconds": now - self._episode_started_at,
                "eva_casts": self._episode_eva_casts,
                "eva_successes": self._episode_eva_successes,
                "eva_misses": self._episode_eva_misses,
                "eva_unknown": self._episode_eva_unknown,
                "zero_nearby_eva_casts": self._episode_zero_nearby_eva_casts,
                "invalid_eva_requests": self._episode_invalid_eva_requests,
            }

        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.action_executor.stop_movement()

    def _build_observation(
        self,
        mobs: list[tuple[float, float]],
        cooldown_fraction: float,
    ) -> np.ndarray:
        return self.observation_builder.build(
            mob_positions=mobs,
            eva_cooldown_fraction=cooldown_fraction,
        )

    def _read_mob_positions(self) -> list[tuple[float, float]]:
        positions: list[tuple[float, float]] = []
        for point in self.read_mobs():
            if len(point) != 2:
                continue
            positions.append((float(point[0]), float(point[1])))
        return self._filter_stale_mobs(positions)

    def _filter_stale_mobs(
        self,
        positions: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if monotonic() >= self._stale_mobs_until:
            self._stale_mob_positions = []
            return positions

        match_radius = max(self.config.despawn_match_radius_px, 0.0)
        filtered: list[tuple[float, float]] = []

        for position in positions:
            if any(
                hypot(
                    position[0] - stale[0],
                    position[1] - stale[1],
                )
                <= match_radius
                for stale in self._stale_mob_positions
            ):
                continue
            filtered.append(position)

        return filtered

    def _begin_despawn_filter(
        self,
        mobs_at_cast: list[tuple[float, float]],
    ) -> None:
        player_x = self.observation_builder.config.resolved_player_x
        player_y = self.observation_builder.config.resolved_player_y
        radius = (
            self._frame_diagonal()
            * self.observation_builder.config.outer_radius_fraction
        )

        self._stale_mob_positions = [
            (x, y)
            for x, y in mobs_at_cast
            if hypot(x - player_x, y - player_y) <= radius
        ]
        self._stale_mobs_until = monotonic() + max(
            self.config.despawn_filter_seconds, 0.0
        )

    def _read_kill_sample(self) -> int | None:
        kills = self.read_kills()
        if kills is None:
            return None
        return max(0, int(kills))

    def _read_eva_kill_sample(self) -> int | None:
        baseline = int(self._previous_kills or 0)
        timeout = max(0.0, self.config.eva_check_timeout_seconds)
        poll_delay = max(0.01, self.config.eva_check_poll_seconds)
        deadline = monotonic() + timeout
        last_valid: int | None = None

        while True:
            sample = self._read_kill_sample()
            if sample is not None:
                last_valid = sample
                delta = sample - baseline
                if 0 < delta <= self.config.max_kill_delta:
                    return sample
                if sample < baseline:
                    return sample

            if monotonic() >= deadline:
                return last_valid
            sleep(poll_delay)

    def _read_valid_kills(self) -> int:
        kills = self._read_kill_sample()
        if kills is None:
            return int(self._previous_kills or 0)
        return kills

    def _calculate_kill_delta(self, current_kills: int) -> int:
        if self._previous_kills is None:
            self._previous_kills = current_kills
            return 0

        if current_kills < self._previous_kills:
            self._previous_kills = current_kills
            return 0

        delta = current_kills - self._previous_kills
        if delta > self.config.max_kill_delta:
            return 0

        self._previous_kills = current_kills
        return delta

    def _kill_reward(self, kills: int) -> float:
        if kills <= 0:
            return 0.0

        multiplier = min(
            1.0 + self.config.group_bonus_per_extra_kill * max(kills - 1, 0),
            self.config.group_multiplier_cap,
        )
        return float(kills * self.config.base_kill_reward * multiplier)

    def _eva_cooldown_fraction(self) -> float:
        if self._last_cast_at is None:
            return 1.0
        cooldown = max(self.config.eva_cooldown_seconds, 0.001)
        elapsed = monotonic() - self._last_cast_at
        return float(np.clip(elapsed / cooldown, 0.0, 1.0))

    def _frame_diagonal(self) -> float:
        config = self.observation_builder.config
        return max(hypot(config.frame_width, config.frame_height), 1.0)

    def _distance_counts(
        self,
        mobs: list[tuple[float, float]],
    ) -> tuple[int, int, int]:
        config = self.observation_builder.config
        player_x = config.resolved_player_x
        player_y = config.resolved_player_y
        diagonal = self._frame_diagonal()

        inner_radius = diagonal * config.inner_radius_fraction
        middle_radius = diagonal * config.middle_radius_fraction
        outer_radius = diagonal * config.outer_radius_fraction

        distances = [hypot(x - player_x, y - player_y) for x, y in mobs]
        inner = sum(distance <= inner_radius for distance in distances)
        middle = sum(distance <= middle_radius for distance in distances)
        outer = sum(distance <= outer_radius for distance in distances)
        return inner, middle, outer

    def _outer_nearby_count(
        self,
        mobs: list[tuple[float, float]],
    ) -> int:
        return self._distance_counts(mobs)[2]

    def _density_score(
        self,
        mobs: list[tuple[float, float]],
    ) -> float:
        inner, middle_cumulative, outer_cumulative = self._distance_counts(mobs)
        middle_shell = max(middle_cumulative - inner, 0)
        outer_shell = max(outer_cumulative - middle_cumulative, 0)

        return float(
            inner * self.config.inner_density_weight
            + middle_shell * self.config.middle_density_weight
            + outer_shell * self.config.outer_density_weight
        )

    @staticmethod
    def _empty_reward_components() -> dict[str, float]:
        return {
            "kill": 0.0,
            "time": 0.0,
            "invalid_eva": 0.0,
            "eva_miss": 0.0,
            "density": 0.0,
        }
