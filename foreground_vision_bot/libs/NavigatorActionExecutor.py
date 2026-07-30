from __future__ import annotations

from mapper.rl.NavigatorCore import NavigatorAction

from libs.ActionExecutor import MovementKeyMap
from libs.HumanKeyboard import VKEY, HumanKeyboard


class NavigatorActionExecutor:
    """Translate navigator actions into persistent forward-running key states.

    W/Z remains held across policy decisions.  Steering changes only the side
    key, and jumping taps Space while forward is already held.  This is the
    live-control counterpart of the simulator's forward-only arc action space.
    """

    def __init__(
        self,
        keyboard: HumanKeyboard,
        *,
        keymap: MovementKeyMap | None = None,
        jump_key: int = VKEY["spacebar"],
    ) -> None:
        self.keyboard = keyboard
        self.keymap = keymap or MovementKeyMap.azerty()
        self.jump_key = int(jump_key)
        self._held: tuple[int, ...] = ()

    @property
    def held_keys(self) -> tuple[int, ...]:
        return self._held

    def execute(self, action: NavigatorAction | int) -> None:
        selected = NavigatorAction(int(action))
        if selected is NavigatorAction.RUN_FORWARD:
            self._set_movement((self.keymap.forward,))
            return
        if selected is NavigatorAction.RUN_FORWARD_LEFT:
            self._set_movement((self.keymap.forward, self.keymap.left))
            return
        if selected is NavigatorAction.RUN_FORWARD_RIGHT:
            self._set_movement((self.keymap.forward, self.keymap.right))
            return
        if selected is NavigatorAction.FORWARD_JUMP:
            self._set_movement((self.keymap.forward,))
            self.keyboard.press_key(self.jump_key, press_time=0.03)
            return
        raise ValueError(f"Unsupported navigator action: {selected}")

    def stop(self) -> None:
        for key in reversed(self._held):
            self.keyboard.key_up(key)
        self._held = ()

    def _set_movement(self, keys: tuple[int, ...]) -> None:
        desired = tuple(dict.fromkeys(int(key) for key in keys))
        if desired == self._held:
            return
        current = set(self._held)
        requested = set(desired)
        for key in reversed(self._held):
            if key not in requested:
                self.keyboard.key_up(key)
        for key in desired:
            if key not in current:
                self.keyboard.key_down(key)
        self._held = desired
