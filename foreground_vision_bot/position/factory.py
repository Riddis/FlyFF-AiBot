from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic

from .NativeFlyffPositionProvider import NativeFlyffPositionProvider
from .PositionConfig import (
    DEFAULT_POSITION_CONFIG_PATH,
    load_native_position_config,
)
from .Win32ProcessMemory import Win32MemoryBackend


def create_native_position_provider(
    window_handle: int,
    *,
    config_path: str | Path = DEFAULT_POSITION_CONFIG_PATH,
    backend: Win32MemoryBackend | None = None,
    clock: Callable[[], float] = monotonic,
) -> NativeFlyffPositionProvider | None:
    """Build the configured provider, or return ``None`` when disabled."""

    config = load_native_position_config(config_path)
    if not config.enabled:
        return None
    return NativeFlyffPositionProvider.from_window_handle(
        window_handle,
        config,
        backend=backend,
        clock=clock,
    )
