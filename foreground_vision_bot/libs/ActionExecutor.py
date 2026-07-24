from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from libs.HumanKeyboard import HumanKeyboard, VKEY


class BotAction(IntEnum):
    MOVE_FORWARD = 0
    FORWARD_LEFT = 1
    FORWARD_RIGHT = 2
    CAST_EVA = 3


@dataclass(frozen=True)
class MovementKeyMap:
    forward: int
    left: int
    right: int

    @classmethod
    def azerty(cls) -> "MovementKeyMap":
        return cls(
            forward=VKEY["z"],
            left=VKEY["q"],
            right=VKEY["d"],
        )

    @classmethod
    def qwerty(cls) -> "MovementKeyMap":
        return cls(
            forward=VKEY["w"],
            left=VKEY["a"],
            right=VKEY["d"],
        )


class ActionExecutor:
    """
    Converts the four discrete PPO actions into persistent Flyff input.

    Movement keys remain held between environment steps. Keys are changed only
    when the requested movement direction changes. EVA briefly releases
    movement, presses the skill key, and restores the previous movement state.
    """

    def __init__(
        self,
        keyboard: HumanKeyboard,
        eva_key: int = VKEY["F1"],
        default_duration: float = 0.15,
        keymap: MovementKeyMap | None = None,
    ) -> None:
        self.keyboard = keyboard
        self.eva_key = int(eva_key)

        # Retained for compatibility with existing Bot/FlyffEnv calls. Movement
        # is now persistent, so execute() does not sleep for this duration.
        self.default_duration = float(default_duration)

        self.keymap = keymap or MovementKeyMap.azerty()
        self._held_movement: tuple[int, ...] = ()

    @property
    def movement_keys(self) -> tuple[int, int, int]:
        return (
            self.keymap.forward,
            self.keymap.left,
            self.keymap.right,
        )

    def execute(
        self,
        action: int | BotAction,
        duration: float | None = None,
    ) -> None:
        # duration is accepted for compatibility but persistent movement does
        # not use it.
        del duration

        selected = BotAction(action)

        action_keys = {
            BotAction.MOVE_FORWARD: (
                self.keymap.forward,
            ),
            BotAction.FORWARD_LEFT: (
                self.keymap.forward,
                self.keymap.left,
            ),
            BotAction.FORWARD_RIGHT: (
                self.keymap.forward,
                self.keymap.right,
            ),
        }

        if selected in action_keys:
            self._set_movement(action_keys[selected])
            return

        if selected == BotAction.CAST_EVA:
            previous_movement = self._held_movement
            self.stop_movement()

            self.keyboard.press_key(
                self.eva_key,
                press_time=0.03,
            )

            # Restoring immediately is intentional. During the game's cast
            # lock the held movement input is ignored; movement resumes as soon
            # as the character is allowed to move again.
            self._set_movement(previous_movement)
            return

        raise ValueError(f"Unsupported action: {selected}")

    def _set_movement(self, keys: tuple[int, ...]) -> None:
        desired = tuple(dict.fromkeys(int(key) for key in keys))

        if desired == self._held_movement:
            return

        current = set(self._held_movement)
        requested = set(desired)

        # Release keys that are no longer part of the requested movement.
        for key in reversed(self._held_movement):
            if key not in requested:
                self.keyboard.key_up(key)

        # Press only newly requested keys.
        for key in desired:
            if key not in current:
                self.keyboard.key_down(key)

        self._held_movement = desired

    def stop_movement(self) -> None:
        for key in reversed(self._held_movement):
            self.keyboard.key_up(key)

        self._held_movement = ()