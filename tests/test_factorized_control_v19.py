from __future__ import annotations

from farming.actions import FarmingCommand, FarmingEvent, SteeringAction
from farming.control import DirectFarmingControl, FarmingKeyMap


class _Token:
    cancelled = False

    def wait(self, _seconds: float) -> bool:
        return False


class _Keyboard:
    def __init__(self) -> None:
        self.trace: list[tuple[str, int]] = []
        self.foreground = True

    def key_down(self, key: int) -> None:
        self.trace.append(("down", key))

    def key_up(self, key: int) -> None:
        self.trace.append(("up", key))

    def is_target_foreground(self) -> bool:
        return self.foreground

    def focus_target_window(self) -> bool:
        self.foreground = True
        return True


def test_forward_stays_latched_while_steering_and_tapping_events() -> None:
    keyboard = _Keyboard()
    keys = FarmingKeyMap.qwerty()
    control = DirectFarmingControl(
        keyboard,
        _Token(),
        keymap=keys,
        sleeper=lambda _seconds: None,
    )

    control.execute(FarmingCommand(SteeringAction.STRAIGHT, FarmingEvent.NONE))
    control.execute(FarmingCommand(SteeringAction.LEFT, FarmingEvent.CAST_EVA))
    control.execute(FarmingCommand(SteeringAction.RIGHT, FarmingEvent.JUMP))

    assert control.held_keys == (keys.forward, keys.right)
    assert keyboard.trace.count(("down", keys.forward)) == 1
    assert ("up", keys.forward) not in keyboard.trace
    assert ("down", keys.eva) in keyboard.trace
    assert ("up", keys.eva) in keyboard.trace
    assert ("down", keys.jump) in keyboard.trace
    assert ("up", keys.jump) in keyboard.trace

    control.release()
    assert keyboard.trace[-2:] == [("up", keys.right), ("up", keys.forward)]


def test_focus_loss_releases_latched_forward() -> None:
    keyboard = _Keyboard()
    keys = FarmingKeyMap.qwerty()
    control = DirectFarmingControl(keyboard, _Token(), keymap=keys)
    control.execute(FarmingCommand(SteeringAction.LEFT, FarmingEvent.NONE))

    keyboard.foreground = False
    assert control.execute_prepared(
        FarmingCommand(SteeringAction.RIGHT, FarmingEvent.NONE)
    ) is False
    assert control.held_keys == ()
    assert keyboard.trace[-2:] == [("up", keys.left), ("up", keys.forward)]
