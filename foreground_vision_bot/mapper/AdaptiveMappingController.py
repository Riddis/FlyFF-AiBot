from __future__ import annotations

from time import sleep

from libs.HumanKeyboard import VKEY, HumanKeyboard, KeyPressTiming


class AdaptiveMappingController:
    """Small calibration-free AZERTY movement controller for mapping."""

    def __init__(self, keyboard: HumanKeyboard) -> None:
        self.keyboard = keyboard
        self.forward_key = VKEY["z"]
        self.left_key = VKEY["q"]
        self.right_key = VKEY["d"]

    def stop(self) -> None:
        first_error: Exception | None = None
        try:
            self.keyboard.release_keys(
                (self.forward_key, self.left_key, self.right_key)
            )
        except Exception as error:  # noqa: BLE001 - still release tracked keys.
            first_error = error
        try:
            self.keyboard.release_all()
        except Exception as error:  # noqa: BLE001 - report after both attempts.
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error

    def _pulse(self, key: int, seconds: float) -> KeyPressTiming:
        duration = max(float(seconds), 0.015)
        self.stop()
        try:
            return self.keyboard.press_key(key, press_time=duration)
        finally:
            self.stop()

    def forward(self, seconds: float) -> KeyPressTiming:
        return self._pulse(self.forward_key, seconds)

    def turn_left(self, seconds: float) -> KeyPressTiming:
        return self._pulse(self.left_key, seconds)

    def turn_right(self, seconds: float) -> KeyPressTiming:
        return self._pulse(self.right_key, seconds)

    def settle(self, seconds: float) -> None:
        self.stop()
        sleep(max(0.0, float(seconds)))
