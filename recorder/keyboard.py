from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum


class RecordedAction(IntEnum):
    RUN_FORWARD = 0
    RUN_FORWARD_LEFT = 1
    RUN_FORWARD_RIGHT = 2
    CAST_EVA = 3
    RUN_FORWARD_JUMP = 4
    UNMAPPED = -1


FORWARD_BIT = 1 << 0
LEFT_BIT = 1 << 1
RIGHT_BIT = 1 << 2
JUMP_BIT = 1 << 3
EVA_BIT = 1 << 4

SUPPORTED_EVA_HOTKEYS = tuple(
    [*(f"F{i}" for i in range(1, 13)), *(str(i) for i in range(1, 10))]
)


@dataclass(frozen=True, slots=True)
class KeySnapshot:
    mask: int
    action: int
    eva_pressed_edge: bool


def virtual_key_for_hotkey(value: str) -> int:
    """Return the Windows virtual-key code for a displayed hotkey choice.

    Number choices intentionally use VK_1 through VK_9. Windows reports those
    physical top-row keys with the same VK codes on QWERTY and AZERTY layouts,
    even when the unshifted AZERTY character printed by the key is a symbol.
    """

    text = value.strip().upper()
    if text.startswith("F") and text[1:].isdigit():
        number = int(text[1:])
        if 1 <= number <= 12:
            return 0x70 + number - 1
    if len(text) == 1 and "1" <= text <= "9":
        return ord(text)
    raise ValueError("EVA hotkey must be F1 through F12 or 1 through 9")


def _user32():
    windll = getattr(ctypes, "windll", None)
    return None if windll is None else ctypes.windll.user32


def is_virtual_key_down(vk: int) -> bool:
    user32 = _user32()
    if user32 is None:  # non-Windows unit tests
        return False
    return bool(user32.GetAsyncKeyState(int(vk)) & 0x8000)


def is_hotkey_down(value: str) -> bool:
    return is_virtual_key_down(virtual_key_for_hotkey(value))


class KeyboardSampler:
    def __init__(self, *, layout: str, eva_hotkey: str) -> None:
        normalized = layout.strip().lower()
        if normalized not in {"azerty", "qwerty"}:
            raise ValueError("Keyboard layout must be azerty or qwerty")
        self.layout = normalized
        self.forward_vk = ord("Z") if normalized == "azerty" else ord("W")
        self.left_vk = ord("Q") if normalized == "azerty" else ord("A")
        self.right_vk = ord("D")
        self.jump_vk = 0x20
        self.eva_vk = virtual_key_for_hotkey(eva_hotkey)
        self._previous_eva = False

    @staticmethod
    def _down(vk: int) -> bool:
        return is_virtual_key_down(vk)

    def sample(self) -> KeySnapshot:
        forward = self._down(self.forward_vk)
        left = self._down(self.left_vk)
        right = self._down(self.right_vk)
        jump = self._down(self.jump_vk)
        eva = self._down(self.eva_vk)
        eva_edge = eva and not self._previous_eva
        self._previous_eva = eva
        mask = (
            (FORWARD_BIT if forward else 0)
            | (LEFT_BIT if left else 0)
            | (RIGHT_BIT if right else 0)
            | (JUMP_BIT if jump else 0)
            | (EVA_BIT if eva else 0)
        )
        if eva_edge:
            action = RecordedAction.CAST_EVA
        elif forward and jump:
            action = RecordedAction.RUN_FORWARD_JUMP
        elif forward and left and not right:
            action = RecordedAction.RUN_FORWARD_LEFT
        elif forward and right and not left:
            action = RecordedAction.RUN_FORWARD_RIGHT
        elif forward and not left and not right:
            action = RecordedAction.RUN_FORWARD
        else:
            action = RecordedAction.UNMAPPED
        return KeySnapshot(mask=mask, action=int(action), eva_pressed_edge=eva_edge)
