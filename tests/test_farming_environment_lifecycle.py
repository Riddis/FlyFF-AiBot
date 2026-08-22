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
from farming.kills import CastCandidate, CastWindow, NativeKillResult
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
        self.on_wait = None

    def wait(self, seconds: float) -> bool:
        self.clock.sleep(seconds)
        if callable(self.on_wait):
            self.on_wait()
        return self.cancelled


class FakeKeyboard:
    def __init__(self) -> None:
        self.trace: list[tuple[str, int]] = []
        self.foreground = True
        self.focus_calls = 0

    def key_down(self, key: int) -> None:
        self.trace.append(("down", key))

    def key_up(self, key: int) -> None:
        self.trace.append(("up", key))

    def is_target_foreground(self) -> bool:
        return self.foreground

    def focus_target_window(self) -> bool:
        self.focus_calls += 1
        return self.foreground


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

    def read_actor_hp_states(self, _candidates):
        return {}


def _frame(
    x: float,
    z: float = 0.0,
    *,
    timestamp: float | None = None,
    player_base: int = 3,
    pointer_slot: int = 1,
) -> NativeWorldFrame:
    captured = x if timestamp is None else float(timestamp)
    return NativeWorldFrame(
        pointer_snapshot=NativePointerSnapshot(
            player_pointer_address=pointer_slot,
            world_pointer_address=2,
            player_base=player_base,
            world_base=4,
            generation=0,
            captured_at=captured,
        ),
        player_pose=PlayerPose(x, 0.0, z, 90.0, captured),
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


def _map_context_with_obstacle() -> FarmingMapContext:
    traversable = np.ones((11, 11), dtype=np.bool_)
    traversable[5, 6] = False
    forbidden = np.zeros_like(traversable)
    forbidden[5, 9] = True
    safe = traversable & ~forbidden
    safe[5, 5] = False
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
            teleport_buffer_radius_cells=1.0,
        ),
        content_hash="OBSTACLE-TEST",
    )


def _environment(
    world: FakeWorldReader,
    *,
    map_context: FarmingMapContext | None = None,
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
        _map_context() if map_context is None else map_context,
        control,
        token,
        config=config,
        read_ocr_kills=read_ocr,
        clock=clock,
        sleeper=clock.sleep,
    )
    environment._test_ocr_reads = ocr_reads  # type: ignore[attr-defined]
    return environment, clock, keyboard


def test_environment_applies_distinct_map_buffer_and_obstacle_penalties() -> None:
    context = _map_context_with_obstacle()
    buffer_env, _clock, _keyboard = _environment(
        FakeWorldReader(_frame(-1.0), _frame(0.0)),
        map_context=context,
    )
    buffer_env.reset()
    buffer_step = buffer_env.step(0)

    assert buffer_step.info["map_cell_risk"] == "obstacle_buffer"
    assert buffer_step.reward.components.obstacle_buffer == pytest.approx(-0.025)
    assert buffer_step.reward.components.obstacle_cell == 0.0

    obstacle_env, _clock, _keyboard = _environment(
        FakeWorldReader(_frame(0.0), _frame(1.0)),
        map_context=context,
    )
    obstacle_env.reset()
    obstacle_step = obstacle_env.step(0)

    assert obstacle_step.info["map_cell_risk"] == "obstacle"
    assert obstacle_step.reward.components.obstacle_buffer == 0.0
    assert obstacle_step.reward.components.obstacle_cell == pytest.approx(-0.75)


def test_focus_pause_refreshes_the_motion_baseline_without_refocusing() -> None:
    world = FakeWorldReader(
        _frame(0.0, timestamp=0.0),
        _frame(2.0, timestamp=10.0),
        _frame(3.0, timestamp=11.0),
    )
    environment, clock, keyboard = _environment(world)
    environment.control.focus.ensure_focused()
    environment.reset()
    statuses: list[str] = []
    environment.control.focus.status_callback = statuses.append
    keyboard.foreground = False
    environment.cancellation.on_wait = lambda: setattr(
        keyboard,
        "foreground",
        True,
    )

    step = environment.step(0)

    assert keyboard.focus_calls == 0
    assert world.calls == 3
    assert clock.value == pytest.approx(0.25)
    assert step.info["player_displacement_cells"] == pytest.approx(1.0)
    assert statuses == [
        "FlyFF lost focus; farming control paused. Focus FlyFF to resume.",
        "FlyFF regained focus; farming control resumed.",
    ]


def test_focus_loss_during_eva_discards_kill_and_transition() -> None:
    world = FakeWorldReader(_frame(0.0), _frame(2.0), _frame(3.0))
    environment, _clock, keyboard = _environment(world)
    environment.control.focus.ensure_focused()
    environment.reset()

    class FocusDroppingKillTracker:
        def begin_cast(self, _frame):
            # A real CastWindow, not a bare sentinel: _info() reads
            # cast_window.candidates unconditionally whenever cast_window
            # is not None (farming/environment.py), matching what the
            # real NativeKillTracker.begin_cast always returns.
            return CastWindow(0.0, ())

        def confirm_cast(self, *_args, **_kwargs):
            keyboard.foreground = False
            environment.cancellation.on_wait = lambda: setattr(
                keyboard, "foreground", True
            )
            return NativeKillResult(
                (CastCandidate(0x1000, 944, 100),),
                1,
                1,
                0,
                False,
                0.05,
            )

    environment.kill_tracker = FocusDroppingKillTracker()  # type: ignore[assignment]
    step = environment.step(3)

    assert step.info["native_kill_delta"] == 0
    assert step.reward.components.kill == 0.0
    assert world.calls == 3


def test_environment_has_one_live_reset_and_one_reward_calculation() -> None:
    environment, _clock, _keyboard = _environment(
        FakeWorldReader(_frame(0.0), _frame(1.0))
    )

    initial = environment.reset()
    step = environment.step(0)

    assert initial.observation.shape == (923,)
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

    assert observation.shape == (923,)
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


def test_teleport_guard_allows_large_displacement_after_long_step() -> None:
    environment, _clock, _keyboard = _environment(FakeWorldReader())
    before = _frame(0.0, timestamp=0.0)
    after = _frame(30.0, timestamp=2.0)

    selected, diagnostic = environment._confirm_teleport_sample(before, after)

    assert selected is after
    assert diagnostic.suspected is False
    assert diagnostic.confirmed is False
    assert diagnostic.reason == "below_threshold"
    assert diagnostic.effective_threshold_cells == pytest.approx(43.0)
    assert diagnostic.displacements_cells == pytest.approx((30.0,))


def test_teleport_guard_rejects_one_bad_coordinate_sample() -> None:
    environment, _clock, _keyboard = _environment(
        FakeWorldReader(
            _frame(0.2, timestamp=0.25),
            _frame(0.3, timestamp=0.30),
        )
    )
    before = _frame(0.0, timestamp=0.0)
    outlier = _frame(30.0, timestamp=0.20)

    selected, diagnostic = environment._confirm_teleport_sample(before, outlier)

    assert selected.player_pose.x == pytest.approx(0.3)
    assert diagnostic.suspected is True
    assert diagnostic.confirmed is False
    assert diagnostic.reason == "returned_below_threshold"
    assert diagnostic.selected_sample_index == 2
    assert diagnostic.displacements_cells == pytest.approx((30.0, 0.2, 0.3))
    assert diagnostic.native_positions == (
        (30.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.3, 0.0, 0.0),
    )
    assert diagnostic.before_native_position == (0.0, 0.0, 0.0)
    assert diagnostic.pose_timestamps == pytest.approx((0.20, 0.25, 0.30))


def test_teleport_guard_requires_stable_identity_and_destination() -> None:
    environment, _clock, _keyboard = _environment(
        FakeWorldReader(
            _frame(101.0, timestamp=0.25),
            _frame(101.5, timestamp=0.30),
        )
    )
    before = _frame(0.0, timestamp=0.0)
    first = _frame(100.0, timestamp=0.20)

    selected, diagnostic = environment._confirm_teleport_sample(before, first)

    assert selected.player_pose.x == pytest.approx(101.5)
    assert diagnostic.confirmed is True
    assert diagnostic.reason == "stable_repeated_discontinuity_same_player"
    assert diagnostic.destination_spread_cells == pytest.approx(1.5)
    assert diagnostic.player_bases == (3, 3, 3)


def test_teleport_guard_rejects_unstable_far_destination() -> None:
    environment, _clock, _keyboard = _environment(
        FakeWorldReader(
            _frame(-30.0, timestamp=0.25),
            _frame(31.0, timestamp=0.30),
        )
    )
    before = _frame(0.0, timestamp=0.0)
    first = _frame(30.0, timestamp=0.20)

    selected, diagnostic = environment._confirm_teleport_sample(before, first)

    assert diagnostic.confirmed is False
    assert diagnostic.reason == "destination_unstable"
    assert diagnostic.destination_spread_cells == pytest.approx(61.0)
    assert selected.player_pose.x == pytest.approx(30.0)


def test_teleport_guard_rejects_unstable_player_identity_samples() -> None:
    environment, _clock, _keyboard = _environment(
        FakeWorldReader(
            _frame(101.0, timestamp=0.25, player_base=9, pointer_slot=8),
            _frame(101.5, timestamp=0.30, player_base=10, pointer_slot=9),
        )
    )
    before = _frame(0.0, timestamp=0.0, player_base=3, pointer_slot=1)
    first = _frame(100.0, timestamp=0.20, player_base=9, pointer_slot=8)

    _selected, diagnostic = environment._confirm_teleport_sample(before, first)

    assert diagnostic.confirmed is False
    assert diagnostic.player_identity_stable is False
    assert diagnostic.reason == "player_identity_unstable"
    assert diagnostic.player_bases == (9, 9, 10)



def test_teleport_guard_confirms_stable_relocation_to_new_player_object() -> None:
    environment, _clock, _keyboard = _environment(
        FakeWorldReader(
            _frame(101.0, timestamp=0.25, player_base=9, pointer_slot=8),
            _frame(101.5, timestamp=0.30, player_base=9, pointer_slot=8),
        )
    )
    before = _frame(0.0, timestamp=0.0, player_base=3, pointer_slot=1)
    first = _frame(100.0, timestamp=0.20, player_base=9, pointer_slot=8)

    _selected, diagnostic = environment._confirm_teleport_sample(before, first)

    assert diagnostic.confirmed is True
    assert diagnostic.player_identity_stable is True
    assert diagnostic.player_identity_changed is True
    assert diagnostic.reason == "stable_repeated_discontinuity_new_player"
    assert diagnostic.player_bases == (9, 9, 9)
    assert diagnostic.pointer_slots == (8, 8, 8)


def test_confirmed_outside_map_teleport_ends_cleanly_without_building_bad_frame() -> None:
    environment, _clock, keyboard = _environment(
        FakeWorldReader(
            _frame(-4.0, timestamp=0.0),
            _frame(100.0, timestamp=0.20),
            _frame(101.0, timestamp=0.25),
            _frame(101.5, timestamp=0.30),
        )
    )
    adapter = UnifiedFarmingGymEnv(environment)
    adapter.reset()

    with pytest.raises(ExternalSessionEnded) as captured:
        adapter.step(0)

    result = captured.value.step_result
    assert result.outcome.reason.value == "external_teleport"
    assert result.info["teleport_confirmed"] is True
    assert result.info["teleport_selected_map_cell"] is None
    assert result.info["teleport_before_native_position"] == [-4.0, 0.0, 0.0]
    assert result.info["unexpected_teleport"] is True
    assert result.info["teleport_source_in_mapped_area"] is False
    assert result.info["teleport_recovery_pulse_attempted"] is True
    assert result.info["teleport_recovery_pulse_completed"] is True
    assert result.info["teleport_recovery_pulse_seconds"] == pytest.approx(0.3)
    assert result.info["teleport_recovery_pulse_error"] is None
    assert environment.control.held_keys == ()
    assert keyboard.trace[-3:] == [
        ("up", 0x5A),
        ("down", 0x5A),
        ("up", 0x5A),
    ]
    assert environment.state is FarmingEnvironmentState.SEALED


def test_confirmed_teleport_from_mapped_area_does_not_send_recovery_pulse() -> None:
    environment, _clock, keyboard = _environment(
        FakeWorldReader(
            _frame(0.0, timestamp=0.0),
            _frame(100.0, timestamp=0.20),
            _frame(101.0, timestamp=0.25),
            _frame(101.5, timestamp=0.30),
        )
    )
    adapter = UnifiedFarmingGymEnv(environment)
    adapter.reset()

    observation, reward, terminated, truncated, info = adapter.step(0)

    assert observation.shape == (923,)
    assert reward < -40.0
    assert terminated is True
    assert truncated is False
    assert info["session_end_reason"] == "forbidden_zone_entered"
    assert info["teleport_source_in_mapped_area"] is True
    assert info["unexpected_teleport"] is False
    assert info["teleport_recovery_pulse_attempted"] is False
    assert info["teleport_recovery_pulse_completed"] is False
    assert keyboard.trace == [("down", 0x5A), ("up", 0x5A)]


def test_incoherent_far_samples_stop_without_consuming_untrusted_position() -> None:
    environment, _clock, _keyboard = _environment(
        FakeWorldReader(
            _frame(0.0, timestamp=0.0),
            _frame(30.0, timestamp=0.20),
            _frame(-30.0, timestamp=0.25),
            _frame(31.0, timestamp=0.30),
        )
    )
    adapter = UnifiedFarmingGymEnv(environment)
    adapter.reset()

    with pytest.raises(ExternalSessionEnded) as captured:
        adapter.step(0)

    result = captured.value.step_result
    assert result.outcome.reason.value == "pointer_grace_exhausted"
    assert result.info["teleport_suspected"] is True
    assert result.info["teleport_confirmed"] is False
    assert result.info["teleport_usable_sample_found"] is False
    assert result.info["teleport_reason"] == "destination_unstable"
    assert result.observation.shape == (923,)
    assert environment.state is FarmingEnvironmentState.SEALED


def test_jump_action_always_executes_but_flair_reward_obeys_cooldown() -> None:
    environment, _clock, keyboard = _environment(
        FakeWorldReader(
            _frame(0.0, timestamp=0.0),
            _frame(0.2, timestamp=0.2),
            _frame(0.4, timestamp=0.4),
        )
    )
    environment.reset()

    first = environment.step(4)
    first_trace = tuple(keyboard.trace)
    repeated = environment.step(4)

    assert first.info["jump_requested"] is True
    assert first.info["jump_available"] is True
    assert first.info["jump_reward_available"] is True
    assert first.info["jump_rewarded"] is True
    assert first.info["jump_performed"] is True
    assert first.info["executed_action_name"] == "RUN_FORWARD_JUMP"
    assert first.reward.components.jump_flair == pytest.approx(0.001)
    assert first_trace == (
        ("down", 0x5A),
        ("down", 0x20),
        ("up", 0x20),
    )

    assert repeated.info["jump_requested"] is True
    assert repeated.info["jump_available"] is True
    assert repeated.info["jump_reward_available"] is False
    assert repeated.info["jump_rewarded"] is False
    assert repeated.info["jump_performed"] is True
    assert repeated.info["executed_action_name"] == "RUN_FORWARD_JUMP"
    assert repeated.reward.components.jump_flair == 0.0
    assert tuple(keyboard.trace)[len(first_trace) :] == (
        ("down", 0x20),
        ("up", 0x20),
    )
