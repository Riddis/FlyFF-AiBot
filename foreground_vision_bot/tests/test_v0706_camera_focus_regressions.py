from __future__ import annotations

from types import SimpleNamespace

import pytest

from libs.CameraDiscoverySweep import CameraDiscoveryConfig, CameraDiscoverySweep


class _Provider:
    def __init__(self) -> None:
        self.discovered_slot_bases: list[int] = []
        self.discovery_calls = 0

    def read_world_base(self) -> int:
        return 123

    def discover_slots(self, *, force: bool = False) -> None:
        assert force is True
        self.discovery_calls += 1
        self.discovered_slot_bases[:] = [1, 2, 3]


class _Keyboard:
    def __init__(self, *, focus_after_checks: int | None = None) -> None:
        self.foreground = False
        self.focus_after_checks = focus_after_checks
        self.focus_checks = 0
        self.focus_calls = 0
        self.events: list[tuple[str, int]] = []

    def is_target_foreground(self) -> bool:
        self.focus_checks += 1
        if (
            self.focus_after_checks is not None
            and self.focus_checks >= self.focus_after_checks
        ):
            self.foreground = True
        return self.foreground

    def focus_target_window(self) -> bool:
        self.focus_calls += 1
        self.foreground = True
        return True

    def key_down(self, key: int) -> None:
        self.events.append(("down", key))

    def key_up(self, key: int) -> None:
        self.events.append(("up", key))


class _RefusingKeyboard(_Keyboard):
    def focus_target_window(self) -> bool:
        self.focus_calls += 1
        return False


def _sweep(keyboard, messages: list[str], **config_overrides) -> CameraDiscoverySweep:
    provider = _Provider()
    bot = SimpleNamespace(
        keyboard=keyboard,
        monster_provider=provider,
        rl_enabled=True,
        get_player_pose=lambda: SimpleNamespace(x=10.0, z=20.0),
    )
    config_values = {
        "quarter_turn_seconds": 0.001,
        "quarter_settle_seconds": 0.0,
        "quarters": 1,
        "focus_timeout_seconds": 0.04,
        "focus_poll_seconds": 0.001,
        "focus_settle_seconds": 0.0,
    }
    config_values.update(config_overrides)
    config = CameraDiscoveryConfig(**config_values)
    return CameraDiscoverySweep(bot, config=config, status_callback=messages.append)


def test_camera_sweep_autofocuses_before_movement() -> None:
    messages: list[str] = []
    keyboard = _Keyboard()
    sweep = _sweep(keyboard, messages)

    assert sweep.run(force=True) is True
    assert keyboard.focus_calls == 1
    assert keyboard.events[0] == ("down", 0x44)
    assert any("focused automatically" in message for message in messages)
    assert sweep.bot.monster_provider.discovery_calls == 1


def test_camera_sweep_waits_for_manual_focus_when_windows_refuses() -> None:
    messages: list[str] = []
    keyboard = _RefusingKeyboard(focus_after_checks=4)
    sweep = _sweep(keyboard, messages)

    assert sweep.run(force=True) is True
    assert keyboard.focus_calls >= 1
    assert any("Click the FlyFF window now" in message for message in messages)
    assert any("focus detected" in message for message in messages)


def test_camera_sweep_times_out_with_actionable_error() -> None:
    messages: list[str] = []
    keyboard = _RefusingKeyboard()
    sweep = _sweep(keyboard, messages, focus_timeout_seconds=0.005)

    with pytest.raises(RuntimeError, match="did not gain focus"):
        sweep.run(force=True)
    assert keyboard.events == []
    assert any("Click the FlyFF window now" in message for message in messages)


def test_camera_focus_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="focus_timeout_seconds"):
        CameraDiscoveryConfig(focus_timeout_seconds=0.0)
    with pytest.raises(ValueError, match="focus_poll_seconds"):
        CameraDiscoveryConfig(focus_poll_seconds=0.0)
    with pytest.raises(ValueError, match="focus_settle_seconds"):
        CameraDiscoveryConfig(focus_settle_seconds=-0.1)
