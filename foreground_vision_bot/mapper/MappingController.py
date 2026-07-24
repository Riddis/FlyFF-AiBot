from __future__ import annotations

from time import sleep

from libs.HumanKeyboard import VKEY, HumanKeyboard


class MappingController:
    """Mapper-only AZERTY movement controller."""

    def __init__(self, keyboard: HumanKeyboard) -> None:
        self.keyboard = keyboard
        self.forward_key = VKEY["z"]
        self.left_key = VKEY["q"]
        self.right_key = VKEY["d"]

    def stop(self) -> None:
        """
        Use HumanKeyboard's public release API.

        The previous mapper called key_up() directly. That bypassed the normal
        input path used by the working RL actions and could leave Flyff treating
        a turn key as held after the short calibration pulse.
        """
        self.keyboard.release_all()

    def _pulse(self, key: int, seconds: float) -> None:
        """
        Send one finite press through HumanKeyboard's tested public API.

        Do not compose a pulse from mapper-level key_down/sleep/key_up calls.
        HumanKeyboard owns the exact Windows message sequence required by the
        game.
        """
        duration = max(float(seconds), 0.015)
        self.stop()

        self.keyboard.press_key(key, press_time=duration)

    def forward(self, seconds: float) -> None:
        self._pulse(self.forward_key, seconds)

    def turn_left(self, seconds: float) -> None:
        self._pulse(self.left_key, seconds)

    def turn_right(self, seconds: float) -> None:
        self._pulse(self.right_key, seconds)

    def settle(self, seconds: float) -> None:
        self.stop()
        sleep(max(float(seconds), 0.0))
