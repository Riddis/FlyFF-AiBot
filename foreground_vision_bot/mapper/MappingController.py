from __future__ import annotations

from time import sleep

from libs.HumanKeyboard import HumanKeyboard, VKEY


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
        if hasattr(self.keyboard, "release_all"):
            self.keyboard.release_all()
            return

        for key in (self.forward_key, self.left_key, self.right_key):
            try:
                if hasattr(self.keyboard, "release_key"):
                    self.keyboard.release_key(key)
                else:
                    self.keyboard.key_up(key)
            except Exception:
                pass

    def _pulse(self, key: int, seconds: float) -> None:
        """
        Send one finite press through HumanKeyboard's tested public API.

        Do not compose a pulse from mapper-level key_down/sleep/key_up calls.
        HumanKeyboard owns the exact Windows message sequence required by the
        game.
        """
        duration = max(float(seconds), 0.015)
        self.stop()

        if hasattr(self.keyboard, "press_key"):
            try:
                self.keyboard.press_key(key, press_time=duration)
                return
            except TypeError:
                # Compatibility with older press_key(key) implementations.
                pass

        if hasattr(self.keyboard, "hold_key"):
            self.keyboard.hold_key(
                key,
                stop_when_w=False,
                press_time=duration,
            )
            return

        # Last-resort compatibility path.
        self.keyboard.key_down(key)
        try:
            sleep(duration)
        finally:
            self.keyboard.key_up(key)
            sleep(0.01)

    def begin_turn(self, direction: str) -> None:
        """
        Persistent turning is retained only for compatibility.

        Calibration does not use this method anymore.
        """
        self.stop()
        if direction == "left":
            self.keyboard.key_down(self.left_key)
        elif direction == "right":
            self.keyboard.key_down(self.right_key)
        else:
            raise ValueError(f"Unknown turn direction: {direction}")

    def repeat_turn(self, direction: str) -> None:
        if direction == "left":
            self.keyboard.key_down(self.left_key)
        elif direction == "right":
            self.keyboard.key_down(self.right_key)
        else:
            raise ValueError(f"Unknown turn direction: {direction}")

    def end_turn(self, direction: str) -> None:
        key = self.left_key if direction == "left" else self.right_key
        if hasattr(self.keyboard, "release_key"):
            self.keyboard.release_key(key)
        else:
            self.keyboard.key_up(key)

    def forward(self, seconds: float) -> None:
        self._pulse(self.forward_key, seconds)

    def turn_left(self, seconds: float) -> None:
        self._pulse(self.left_key, seconds)

    def turn_right(self, seconds: float) -> None:
        self._pulse(self.right_key, seconds)

    def settle(self, seconds: float) -> None:
        self.stop()
        sleep(max(float(seconds), 0.0))
