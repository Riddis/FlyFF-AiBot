from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClientWindow:
    hwnd: int
    title: str

    def __str__(self) -> str:
        return self.title


def find_client_windows(title_prefix: str) -> tuple[ClientWindow, ...]:
    try:
        import win32gui  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - Windows packaging path
        raise RuntimeError("pywin32 is required to find the FlyFF client") from error

    prefix = title_prefix.casefold()
    matches: list[ClientWindow] = []

    def visit(hwnd: int, _extra: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = str(win32gui.GetWindowText(hwnd)).strip()
        if title.casefold().startswith(prefix):
            matches.append(ClientWindow(int(hwnd), title))

    win32gui.EnumWindows(visit, None)
    return tuple(sorted(matches, key=lambda item: item.title.casefold()))


def foreground_window() -> int:
    try:
        import win32gui  # type: ignore[import-untyped]
    except ImportError:
        return 0
    return int(win32gui.GetForegroundWindow())
