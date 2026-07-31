from __future__ import annotations

# The base reset is intentionally captured before importing the runtime patch
# composition module below.
# ruff: noqa: I001

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from libs.CameraDiscoverySweep import CameraDiscoverySweep
from libs.NativeFarmingEnv import NativeFarmingEnv
from position import PointerResolutionError

_BASE_NATIVE_FARMING_RESET = NativeFarmingEnv.reset
_BASE_NATIVE_FARMING_STEP = NativeFarmingEnv.step

from native_farming import (
    _CancellationGuardEnv,
    _preflight_native_startup,
    dry_run_native_farming,
)
from worker_manager import CancellationToken, WorkerCancelled


def test_importing_runtime_helpers_does_not_patch_base_environment() -> None:
    assert NativeFarmingEnv.reset is _BASE_NATIVE_FARMING_RESET
    assert NativeFarmingEnv.step is _BASE_NATIVE_FARMING_STEP


class _PositionProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.reads = 0

    def read_pose(self):
        self.reads += 1
        if self.fail:
            raise PointerResolutionError("native player pointer is unavailable")
        return SimpleNamespace(x=10.0, y=0.0, z=20.0)


class _MonsterProvider:
    def __init__(self) -> None:
        self.player_reads = 0
        self.world_reads = 0

    def read_player_base(self) -> int:
        self.player_reads += 1
        return 0x1000

    def read_world_base(self) -> int:
        self.world_reads += 1
        return 0x2000


class _StartupBot:
    def __init__(self, *, fail_pose: bool = False) -> None:
        self.position_provider = _PositionProvider(fail=fail_pose)
        self.monster_provider = _MonsterProvider()
        self.config = {
            "selected_mobs": [{"species_id": 944}],
            "selected_map_name": "Tower AoE",
        }
        self.is_ready = True
        self.start_calls = 0

    def get_player_pose(self):
        return self.position_provider.read_pose()

    def start(self) -> None:
        self.start_calls += 1


def test_startup_preflight_uses_existing_bounded_native_reader_apis() -> None:
    bot = _StartupBot()

    state = _preflight_native_startup(bot, CancellationToken())

    assert state.player_pose.x == pytest.approx(10.0)
    assert state.player_base == 0x1000
    assert state.world_base == 0x2000
    assert bot.position_provider.reads == 1
    assert bot.monster_provider.player_reads == 1
    assert bot.monster_provider.world_reads == 1


def test_failed_startup_preflight_never_enables_bot_control() -> None:
    bot = _StartupBot(fail_pose=True)
    messages: list[str] = []

    with pytest.raises(RuntimeError, match="startup preflight failed"):
        dry_run_native_farming(
            bot,
            status_callback=messages.append,
            cancellation=CancellationToken(),
        )

    assert bot.start_calls == 0
    assert any("preflight failed" in message.lower() for message in messages)


class _CountingEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(4)
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.reset_calls += 1
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        del action
        self.step_calls += 1
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


def test_training_guard_does_not_step_or_reset_after_user_cancellation() -> None:
    token = CancellationToken()
    wrapped = _CancellationGuardEnv(_CountingEnv(), token)
    token.cancel()

    with pytest.raises(WorkerCancelled):
        wrapped.step(0)
    with pytest.raises(WorkerCancelled):
        wrapped.reset()

    assert wrapped.env.step_calls == 0
    assert wrapped.env.reset_calls == 0


def test_cancelled_camera_sweep_does_not_touch_native_or_input() -> None:
    token = CancellationToken()
    token.cancel()

    class _FailIfUsed:
        def __getattr__(self, name):
            raise AssertionError(f"cancelled sweep accessed {name}")

    bot = SimpleNamespace(
        rl_enabled=True,
        keyboard=_FailIfUsed(),
        monster_provider=_FailIfUsed(),
        get_player_pose=lambda: pytest.fail("cancelled sweep read the player pose"),
    )
    sweep = CameraDiscoverySweep(bot, cancellation=token)

    assert not sweep.run(force=True)


def test_cancelled_reset_stops_before_native_ocr_or_snapshot_reads() -> None:
    token = CancellationToken()
    token.cancel()
    stop_calls: list[None] = []
    env = object.__new__(NativeFarmingEnv)
    env.navigator = SimpleNamespace(
        cancellation=token,
        stop=lambda: stop_calls.append(None),
    )
    env.bot = SimpleNamespace(
        rl_enabled=True,
        get_native_monsters=lambda: pytest.fail("cancelled reset read actors"),
    )

    with pytest.raises(WorkerCancelled):
        _BASE_NATIVE_FARMING_RESET(env)

    assert len(stop_calls) == 1


def test_camera_is_not_marked_warm_when_reset_is_cancelled_during_sweep() -> None:
    token = CancellationToken()

    class _Sweep:
        def run(self, **_kwargs) -> bool:
            token.cancel()
            return False

    env = object.__new__(NativeFarmingEnv)
    env.navigator = SimpleNamespace(cancellation=token, stop=lambda: None)
    env.bot = SimpleNamespace(rl_enabled=True, get_native_monsters=list)
    env.camera_sweep = _Sweep()
    env.config = SimpleNamespace(camera_sweep_on_first_reset=True)
    env._camera_warmed = False

    with pytest.raises(WorkerCancelled):
        _BASE_NATIVE_FARMING_RESET(env)

    assert not env._camera_warmed
