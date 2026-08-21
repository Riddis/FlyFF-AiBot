from __future__ import annotations

from libs import helpers


def test_window_enumeration_accepts_unicode_titles(monkeypatch) -> None:
    titles = {
        101: "Spirit Of Madrigal",
        202: "你好世界",
        303: "",
    }

    def enumerate_windows(callback, context) -> None:
        for handle in titles:
            callback(handle, context)

    monkeypatch.setattr(helpers.win32gui, "EnumWindows", enumerate_windows)
    monkeypatch.setattr(helpers.win32gui, "IsWindowVisible", lambda _handle: True)
    monkeypatch.setattr(
        helpers.win32gui,
        "GetWindowText",
        lambda handle: titles[handle],
    )

    assert helpers.get_window_handlers() == {
        "Spirit Of Madrigal": 101,
        "你好世界": 202,
    }
