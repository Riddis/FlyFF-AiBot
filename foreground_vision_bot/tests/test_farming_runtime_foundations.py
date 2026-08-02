from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from farming.config import FarmingRuntimeConfig
from farming.control import (
    DirectFarmingControl,
    FarmingControlCancelled,
    FarmingControlUnavailable,
    FarmingKeyMap,
    WindowFocusService,
)
from farming.map_context import FarmingMapContext
from farming.map_features import FarmingMapFeatures
from farming.native_world import (
    NativeWorldFrame,
    NativeWorldReader,
    NativeWorldUnavailable,
    build_actor_observations,
)
from mapper.CoordinateFrame import CoordinateFrame
from position.native_process_service import NativePointerSnapshot
from position.NativeFlyffMonsterProvider import (
    ActorCacheOutcome,
    ActorCacheRefreshResult,
    CachedActorReadResult,
    NativeActor,
)
from position.PositionProvider import PlayerPose


class FakeToken:
    def __init__(self) -> None:
        self.cancelled = False
        self.cancel_during_wait = False

    def wait(self, _seconds: float) -> bool:
        if self.cancel_during_wait:
            self.cancelled = True
        return self.cancelled


class FakeKeyboard:
    def __init__(self) -> None:
        self.trace: list[tuple[str, int | None]] = []
        self.foreground = True

    def key_down(self, key: int) -> None:
        self.trace.append(("down", key))

    def key_up(self, key: int) -> None:
        self.trace.append(("up", key))

    def is_target_foreground(self) -> bool:
        return self.foreground

    def focus_target_window(self) -> bool:
        self.trace.append(("focus", None))
        return self.foreground


def test_focus_service_autofocuses_then_allows_manual_grace() -> None:
    keyboard = FakeKeyboard()
    keyboard.foreground = False

    class Token(FakeToken):
        def wait(self, _seconds: float) -> bool:
            keyboard.foreground = True
            return False

    statuses: list[str] = []
    focus = WindowFocusService(
        keyboard,
        Token(),
        grace_seconds=0.1,
        poll_seconds=0.01,
        status_callback=statuses.append,
    )

    focus.ensure_focused()

    assert keyboard.trace == [("focus", None)]
    assert statuses == [
        "FlyFF did not accept automatic focus; focus it manually to continue."
    ]


def test_focus_service_wait_is_cancellable_and_control_releases_keys() -> None:
    keyboard = FakeKeyboard()
    token = FakeToken()
    keys = FarmingKeyMap.azerty()
    control = DirectFarmingControl(keyboard, token, keymap=keys)
    control.execute(0)
    keyboard.foreground = False
    token.cancel_during_wait = True

    with pytest.raises(FarmingControlCancelled, match="focus wait"):
        control.execute(1)

    assert control.held_keys == ()
    assert keyboard.trace[-2:] == [("focus", None), ("up", keys.forward)]




def test_emergency_forward_pulse_releases_existing_movement_and_stops() -> None:
    keyboard = FakeKeyboard()

    class RecordingToken(FakeToken):
        def __init__(self) -> None:
            super().__init__()
            self.waits: list[float] = []

        def wait(self, seconds: float) -> bool:
            self.waits.append(float(seconds))
            return False

    token = RecordingToken()
    keys = FarmingKeyMap.azerty()
    control = DirectFarmingControl(keyboard, token, keymap=keys)
    control.execute(1)

    control.pulse_forward(0.3)

    assert token.waits == [pytest.approx(0.3)]
    assert control.held_keys == ()
    assert control.held_movement is None
    assert keyboard.trace == [
        ("down", keys.forward),
        ("down", keys.left),
        ("up", keys.left),
        ("up", keys.forward),
        ("down", keys.forward),
        ("up", keys.forward),
    ]


def test_shipped_config_migrates_without_using_hierarchical_navigation() -> None:
    root = Path(__file__).parents[1]

    config = FarmingRuntimeConfig.load(root / "native_farming.json")

    assert config.control_interval_seconds == pytest.approx(0.2)
    assert config.pointer_grace_seconds == pytest.approx(3.0)
    assert config.keyboard_layout == "azerty"
    assert config.model_path.endswith("native_strategy_map_risk_ppo")
    assert config.checkpoint_frequency == 50_000
    assert not hasattr(config, "total_timesteps")
    assert not hasattr(config, "episode_seconds")
    assert config.jump_cooldown_seconds == pytest.approx(2.0)
    assert config.jump_flair_reward == pytest.approx(0.001)
    assert config.teleport_confirmation_samples == 3
    assert config.unexpected_teleport_forward_pulse_seconds == pytest.approx(0.3)
    assert config.obstacle_buffer_penalty == pytest.approx(0.025)
    assert config.obstacle_cell_penalty == pytest.approx(0.75)
    assert "movement_model_path" not in config.contract_payload()


def test_shipped_tower_map_loads_with_explicit_teleport_features() -> None:
    context = FarmingMapContext.load(
        "Tower AoE",
        teleport_buffer_radius_cells=2.0,
    )

    assert context.map_name == "Tower AoE"
    assert context.features.has_forbidden
    assert context.features.shape == (310, 294)
    assert context.native_units_per_cell == pytest.approx(1.6)
    assert len(context.content_hash) == 64


def test_config_rejects_unknown_or_boolean_numeric_values(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"mystery": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        FarmingRuntimeConfig.load(unknown)

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"checkpoint_frequency": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        FarmingRuntimeConfig.load(invalid)

    invalid.write_text(
        json.dumps({"control_interval_seconds": "0.2"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="real number"):
        FarmingRuntimeConfig.load(invalid)


def test_direct_control_persists_movement_and_eva_never_releases_it() -> None:
    keyboard = FakeKeyboard()
    token = FakeToken()
    keys = FarmingKeyMap.azerty()
    control = DirectFarmingControl(keyboard, token, keymap=keys)

    control.execute(0)
    control.execute(1)
    control.execute(2)
    before_eva = tuple(keyboard.trace)
    control.execute(3)

    assert control.cancellation is token
    assert control.held_keys == (keys.forward, keys.right)
    assert keyboard.trace[:4] == [
        ("down", keys.forward),
        ("down", keys.left),
        ("up", keys.left),
        ("down", keys.right),
    ]
    assert keyboard.trace[len(before_eva) :] == [
        ("down", keys.eva),
        ("up", keys.eva),
    ]
    control.close()
    after_close = tuple(keyboard.trace)
    control.close()
    assert tuple(keyboard.trace) == after_close
    assert keyboard.trace[-2:] == [("up", keys.right), ("up", keys.forward)]


def test_direct_control_forward_jump_keeps_forward_held_and_taps_space() -> None:
    keyboard = FakeKeyboard()
    token = FakeToken()
    keys = FarmingKeyMap.azerty()
    control = DirectFarmingControl(keyboard, token, keymap=keys)

    control.execute(0)
    before_jump = tuple(keyboard.trace)
    control.execute(4)

    assert control.held_keys == (keys.forward,)
    assert control.held_movement is not None
    assert keyboard.trace[len(before_jump) :] == [
        ("down", keys.jump),
        ("up", keys.jump),
    ]


def test_direct_control_releases_on_focus_loss_and_eva_cancellation() -> None:
    keyboard = FakeKeyboard()
    token = FakeToken()
    keys = FarmingKeyMap.azerty()
    control = DirectFarmingControl(
        keyboard,
        token,
        keymap=keys,
        focus_service=WindowFocusService(
            keyboard,
            token,
            autofocus=False,
            grace_seconds=0.001,
            poll_seconds=0.001,
        ),
    )
    control.execute(2)
    keyboard.foreground = False

    with pytest.raises(FarmingControlUnavailable):
        control.execute(0)
    assert control.held_keys == ()

    keyboard.foreground = True
    control.execute(1)
    token.cancel_during_wait = True
    with pytest.raises(FarmingControlCancelled):
        control.execute(3)
    assert control.held_keys == ()
    assert ("up", keys.eva) in keyboard.trace


def _snapshot() -> NativePointerSnapshot:
    return NativePointerSnapshot(
        player_pointer_address=0x10,
        world_pointer_address=0x20,
        player_base=0x1000,
        world_base=0x2000,
        generation=7,
        captured_at=1.0,
    )


def _actor() -> NativeActor:
    return NativeActor(
        base_address=0x3000,
        species_id=874,
        hp=100,
        x=1.6,
        y=0.0,
        z=1.6,
        distance_native=2.0,
        active_species_id=874,
    )


def test_native_world_uses_one_identical_pointer_snapshot_for_pose_and_actors() -> None:
    snapshot = _snapshot()
    pose = PlayerPose(0.0, 0.0, 0.0, 90.0, 1.0)

    class Service:
        calls = 0

        def read_pointer_snapshot(self):
            self.calls += 1
            return snapshot

    class Position:
        def read_pose(self, *, pointer_snapshot=None):
            assert pointer_snapshot is snapshot
            return pose

    class Actors:
        def read_cached_active_actors(
            self,
            pointer_snapshot,
            player_pose,
            *,
            allowed_species_ids=None,
            vision_radius_native=None,
        ):
            assert pointer_snapshot is snapshot
            assert player_pose is pose
            return CachedActorReadResult(
                ActorCacheOutcome.READY,
                snapshot.world_base,
                snapshot.generation,
                actors=(_actor(),),
            )

        def refresh_slot_cache(self, pointer_snapshot, **_kwargs):
            return ActorCacheRefreshResult(
                ActorCacheOutcome.CACHED,
                pointer_snapshot.world_base,
                pointer_snapshot.generation,
                slot_count=1,
            )

    service = Service()
    reader = NativeWorldReader(
        service,
        Position(),
        Actors(),
        allowed_species_ids={874},
        vision_radius_native=100.0,
    )

    frame = reader.read_frame()

    assert service.calls == 1
    assert frame.pointer_snapshot is snapshot
    assert frame.actors == (_actor(),)


def test_native_world_propagates_typed_cache_unavailability() -> None:
    snapshot = _snapshot()

    class Service:
        def read_pointer_snapshot(self):
            return snapshot

    class Position:
        def read_pose(self, *, pointer_snapshot=None):
            return PlayerPose(0.0, 0.0, 0.0, None, 1.0)

    class Actors:
        def read_cached_active_actors(self, *_args, **_kwargs):
            return CachedActorReadResult(
                ActorCacheOutcome.WORLD_MISMATCH,
                snapshot.world_base,
                snapshot.generation,
                message="changed",
            )

        def refresh_slot_cache(self, pointer_snapshot, **_kwargs):
            raise AssertionError("not used")

    reader = NativeWorldReader(
        Service(),
        Position(),
        Actors(),
        allowed_species_ids=None,
        vision_radius_native=100.0,
    )

    with pytest.raises(NativeWorldUnavailable) as captured:
        reader.read_frame()
    assert captured.value.outcome is ActorCacheOutcome.WORLD_MISMATCH


def test_actor_observation_keeps_layout_y_and_native_z_signs_distinct() -> None:
    traversable = np.ones((5, 5), dtype=np.bool_)
    features = FarmingMapFeatures(
        traversable=traversable,
        forbidden=np.zeros_like(traversable),
        safe_traversable=traversable,
        teleport_buffer_radius_cells=0.0,
    )
    context = FarmingMapContext(
        map_name="test",
        map_directory=Path("."),
        coordinate_frame=CoordinateFrame(native_units_per_cell=1.6),
        grid_origin=2,
        source_bounds=(0, 0, 4, 4),
        features=features,
        content_hash="TEST",
    )
    frame = NativeWorldFrame(
        pointer_snapshot=_snapshot(),
        player_pose=PlayerPose(0.0, 0.0, 0.0, None, 1.0),
        actors=(_actor(),),
    )

    observed = build_actor_observations(frame, context)[0]

    assert observed.legacy_dx_cells == pytest.approx(1.0)
    assert observed.legacy_dy_cells == pytest.approx(-1.0)
    assert observed.direct_dx_cells == pytest.approx(1.0)
    assert observed.direct_dz_cells == pytest.approx(1.0)
