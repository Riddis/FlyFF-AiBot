from __future__ import annotations

import json
from pathlib import Path

from position.factory import create_native_position_provider


class ExplodingBackend:
    def get_window_process_id(self, _window_handle: int) -> int:
        raise AssertionError("disabled configuration must not touch Win32")

    def open_process(self, _pid: int, _access: int) -> int:
        raise AssertionError("disabled configuration must not touch Win32")

    def read_process_memory(self, _handle: int, _address: int, _size: int) -> bytes:
        raise AssertionError("disabled configuration must not touch Win32")

    def close_handle(self, _handle: int) -> None:
        raise AssertionError("disabled configuration must not touch Win32")


def test_factory_is_noop_when_native_position_is_disabled(tmp_path: Path) -> None:
    path = tmp_path / "native_position.json"
    path.write_text(json.dumps({"enabled": False}), encoding="utf-8")

    provider = create_native_position_provider(
        123,
        config_path=path,
        backend=ExplodingBackend(),
    )

    assert provider is None
