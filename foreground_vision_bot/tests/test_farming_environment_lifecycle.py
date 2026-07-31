from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from farming.config import FarmingRuntimeConfig
from farming.control import DirectFarmingControl, FarmingKeyMap
from farming.environment import (
    FarmingEnvironmentError,
    FarmingEnvironmentState,
    UnifiedFarmingEnv,
)
from farming.map_context import FarmingMapContext
from farming.map_features import FarmingMapFeatures
from farming.native_world import NativeWorldFrame, NativeWorldUnavailable
from farming.sb3_adapter import (
    ExternalSessionEnded,
    PolicyTerminalDelivered,
    UnifiedFarmingGymEnv,
)
from mapper.CoordinateFrame import CoordinateFrame
from position.native_process_service import NativePointerSnapshot
from position.NativeFlyffMonsterProvider import ActorCacheOutcome
from position.PositionProvider import PlayerPose


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeToken:
    cancelled = False

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock

    def wait(self, seconds: float) -> bool:
        self.clock.sleep(seconds)
        return self.cancelled


class FakeKeyboard:
    def __init__(self) -> None:
        self.trace: list[tuple[str, int]] = []

    def key_down(self, key: int) -> None:
        self.trace.append(("down", key))

    def key_up(self, key: int) -> None:
        self.trace.append(("up", key))

    def is_target_foreground(self) -> bool:
        return True

    def focus_target_window(self) -> bool:
        return True


class FakeWorldReader:
    def __init__(self, *items: NativeWorldFrame | Exception) -> None:
        self.items = list(items)
        self.calls = 0

    def read_frame(self) -> NativeWorldFrame:
        self.calls += 1
        if not self.items:
            raise AssertionError("Unexpected native frame read")
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _frame(x: float, z: float = 0.0) -> NativeWorldFrame:
    return NativeWorldFrame(
        pointer_snapshot=NativePointerSnapshot(
            player_pointer_address=1,
            world_pointer_address=2,
            player_base=3,
            world_base=4,
            generation=0,
            captured_at=x,
        ),
        player_pose=PlayerPose(x, 0.0, z, 90.0, x),
        actors=(),
    )


def _map_context() -> FarmingMapContext:
    traversable = np.ones((11, 11), dtype=np.bool_)
    forbidden = np.zeros_like(traversable)
    forbidden[5, 7] = True
    safe = traversable & ~forbidden
    return FarmingMapContext(
        map_name="Tower AoE",
        map_directory=Path("."),
        coordinate_frame=CoordinateFrame(native_units_per_cell=1.0),
        grid_origin=5,
        source_bounds=(0, 0, 10, 10),
        features=FarmingMapFeatures(
            traversable=traversable,
            forbidden=forbidden,
            safe_traversable=safe,
            teleport_buffer_radius_cells=2.0,
        ),
        content_hash="TEST",
    )


def _environment(
    world: FakeWorldReader,
) -> tuple[UnifiedFarmingEnv, FakeClock, FakeKeyboard]:
    clock = FakeClock()
    token = FakeToken(clock)
    keyboard = FakeKeyboard()
    control = DirectFarmingControl(
        keyboard,
        token,
        keymap=FarmingKeyMap.azerty(),
        sleeper=clock.sleep,
    )
    config = FarmingRuntimeConfig(
        episode_seconds=100.0,
        control_interval_seconds=0.20,
        pointer_grace_seconds=0.10,
        pointer_poll_seconds=0.05,
        cast_minimum_absence_seconds=0.05,
        cast_result_timeout_seconds=0.20,
        cast_poll_seconds=0.05,
    )
    ocr_reads = {"count": 0}

    def read_ocr() -> int:
        ocr_reads["count"] += 1
        return 10

    environment = UnifiedFarmingEnv(
        world,  # type: ignore[arg-type]
        _map_context(),
        control,
        token,
        config=config,
        read_ocr_kills=read_ocr,
        clock=clock,
        sleeper=clock.sleep,
    )
    environment._test_ocr_reads = ocr_reads  # type: ignore[attr-defined]
    return environment, clock, keyboard


def test_environment_has_one_live_reset_and_one_reward_calculation() -> None:
    environment, _clock, _keyboard = _environment(
        FakeWorldReader(_frame(0.0), _frame(1.0))
    )

    initial = environment.reset()
    step = environment.step(0)

    assert initial.observation.shape == (482,)
    assert environment.state is FarmingEnvironmentState.ACTIVE
    assert step.outcome.should_stop_session is False
    assert step.info["action_name"] == "RUN_FORWARD"
    assert environment.control.held_movement is not None
    assert environment._test_ocr_reads == {"count": 1}  # type: ignore[attr-defined]
    assert step.reward.total == pytest.approx(step.reward.components.total)
    with pytest.raises(FarmingEnvironmentError):
        environment.reset()


def test_policy_forbidden_terminal_is_delivered_once_then_sink_refuses_step() -> None:
    environment, _clock, keyboard = _environment(
        FakeWorldReader(_frame(0.0), _frame(2.0))
    )
    adapter = UnifiedFarmingGymEnv(environment)
    adapter.reset()

    observation, reward, terminated, truncated, info = adapter.step(0)

    assert observation.shape == (482,)
    assert reward < -40.0
    assert terminated is True
    assert truncated is False
    assert info["session_end_reason"] == "forbidden_zone_entered"
    assert environment.state is FarmingEnvironmentState.SEALED
    assert environment.control.held_keys == ()
    assert keyboard.trace[-1][0] == "up"

    sink, sink_info = adapter.reset()
    assert np.array_equal(sink, observation)
    assert sink_info == {"terminal_sink": True}
    with pytest.raises(PolicyTerminalDelivered):
        adapter.step(0)


def test_external_world_change_raises_before_gym_tuple_and_cannot_reset() -> None:
    environment, _clock, _keyboard = _environment(
        FakeWorldReader(
            _frame(0.0),
            NativeWorldUnavailable(ActorCacheOutcome.WORLD_MISMATCH, "changed"),
        )
    )
    adapter = UnifiedFarmingGymEnv(environment)
    adapter.reset()

    with pytest.raises(ExternalSessionEnded) as captured:
        adapter.step(0)

    assert captured.value.step_result.reward.total == pytest.approx(0.0)
    assert (
        captured.value.step_result.outcome.classification.value == "external_truncation"
    )
    assert environment.state is FarmingEnvironmentState.SEALED
    with pytest.raises(RuntimeError, match="cannot be auto-reset"):
        adapter.reset()
