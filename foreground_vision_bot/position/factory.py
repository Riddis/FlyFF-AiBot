from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic

from .native_process_service import NativeProcessService
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
    native_service: NativeProcessService | None = None,
) -> NativeFlyffPositionProvider | None:
    """Build the configured provider, or return ``None`` when disabled."""

    config = load_native_position_config(config_path)
    if not config.enabled:
        return None
    if native_service is not None:
        return NativeFlyffPositionProvider.from_native_service(
            native_service,
            config,
            clock=clock,
        )
    return NativeFlyffPositionProvider.from_window_handle(
        window_handle,
        config,
        backend=backend,
        clock=clock,
    )
