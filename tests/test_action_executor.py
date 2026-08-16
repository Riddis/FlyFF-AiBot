from __future__ import annotations

from libs.ActionExecutor import ActionExecutor, BotAction, MovementKeyMap


class FakeKeyboard:
    def __init__(self) -> None:
        self.down: list[int] = []
        self.up: list[int] = []
        self.pressed: list[int] = []

    def key_down(self, key: int) -> None:
        self.down.append(key)

    def key_up(self, key: int) -> None:
        self.up.append(key)

    def press_key(self, key: int, press_time: float = 0.03) -> None:
        self.pressed.append(key)


def test_stop_releases_all_persistent_azerty_movement() -> None:
    keyboard = FakeKeyboard()
    executor = ActionExecutor(
        keyboard,
        keymap=MovementKeyMap.azerty(),
    )

    executor.execute(BotAction.FORWARD_LEFT)
    executor.stop_movement()

    assert keyboard.down == [0x5A, 0x51]
    assert keyboard.up == [0x51, 0x5A]


def test_eva_releases_and_restores_movement() -> None:
    keyboard = FakeKeyboard()
    executor = ActionExecutor(
        keyboard,
        keymap=MovementKeyMap.azerty(),
    )

    executor.execute(BotAction.MOVE_FORWARD)
    executor.execute(BotAction.CAST_EVA)

    assert keyboard.up == [0x5A]
    assert keyboard.pressed == [0x70]
    assert keyboard.down == [0x5A, 0x5A]
