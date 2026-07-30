from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from libs.NativeFarmingEnv import NativeFarmingEnv, NativeFarmingEnvConfig
from libs.NativeFarmingObservation import (
    NativeFarmingObservationBuilder,
    NativeFarmingObservationConfig,
)
from libs.NativeMapContext import NativeMapContext
from mapper.CoordinateFrame import CoordinateFrame
from mapper.rl.ProceduralDungeon import DungeonLayout
from position import NativeActor, PlayerPose


def _context() -> NativeMapContext:
    traversable = np.ones((41, 41), dtype=np.bool_)
    layout = DungeonLayout(traversable=traversable, spawn=(20, 20))
    return NativeMapContext(
        map_name="test",
        map_directory=Path("."),
        coordinate_frame=CoordinateFrame(native_units_per_cell=1.0),
        grid_origin=0,
        source_bounds=(0, -40, 40, 0),
        layout=layout,
        safe_traversable=traversable.copy(),
    )


def _actor(base: int, x: float, z: float) -> NativeActor:
    return NativeActor(base, 944, 100, x, 0.0, z, 0.0, 944)


class _Bot:
    def __init__(self) -> None:
        self.rl_enabled = True
        self.pose = PlayerPose(20.0, 0.0, 20.0, 0.0, 0.0)
        self.actors = [_actor(1, 22.0, 20.0), _actor(2, 23.0, 20.0)]
        self.kills = 0

    def get_navigation_pose(self, **_kwargs):
        return self.pose

    def get_native_monsters(self, **_kwargs):
        return list(self.actors)

    def read_kill_count(self):
        return self.kills


class _Navigator:
    def __init__(self) -> None:
        self.goals: list[tuple[int, int]] = []
        self.cast_count = 0

    def stop(self) -> None:
        return None

    def navigate_toward_cell(self, goal, *, duration_seconds=None):
        self.goals.append(tuple(goal))
        return SimpleNamespace(
            goal_cell=tuple(goal),
            actions=1,
            elapsed_seconds=float(duration_seconds or 0.0),
            initial_distance_cells=2.0,
            final_distance_cells=1.0,
            arrived=True,
            recovery_used=False,
            last_action="RUN_FORWARD",
        )

    def cast_eva(self) -> None:
        self.cast_count += 1


class _Sweep:
    def run(self, **_kwargs) -> bool:
        return False

    def should_run(self, **_kwargs) -> bool:
        return False


def _environment() -> tuple[NativeFarmingEnv, _Bot, _Navigator]:
    context = _context()
    bot = _Bot()
    navigator = _Navigator()
    builder = NativeFarmingObservationBuilder(
        context,
        NativeFarmingObservationConfig(
            max_targets=4,
            vision_radius_cells=20.0,
            eva_radius_cells=4.0,
        ),
    )
    env = NativeFarmingEnv(
        bot=bot,
        map_context=context,
        observation_builder=builder,
        navigator=navigator,
        camera_sweep=_Sweep(),
        config=NativeFarmingEnvConfig(
            observation_delay_seconds=0.0,
            navigation_burst_seconds=0.1,
            eva_cooldown_seconds=0.01,
            eva_result_timeout_seconds=0.01,
            eva_result_poll_seconds=0.001,
            episode_seconds=10.0,
            camera_sweep_on_first_reset=False,
            camera_sweep_when_empty=False,
        ),
    )
    return env, bot, navigator


def test_native_farming_target_action_hands_goal_to_frozen_navigator() -> None:
    env, _bot, navigator = _environment()
    observation, _info = env.reset()

    after, reward, terminated, truncated, info = env.step(0)

    assert observation.shape == after.shape
    assert navigator.goals
    assert not terminated
    assert not truncated
    assert info["action_name"] == "TARGET_0"
    assert info["navigation"]["last_action"] == "RUN_FORWARD"
    assert reward <= 0.25


def test_native_farming_eva_rewards_every_confirmed_kill() -> None:
    env, bot, navigator = _environment()
    env.reset()
    env._read_eva_result = lambda: 3

    _after, reward, _terminated, _truncated, info = env.step(env.cast_action)

    assert navigator.cast_count == 1
    assert info["kill_delta"] == 3
    assert info["eva_success"]
    assert reward > 2.8
    assert bot.kills == 0
