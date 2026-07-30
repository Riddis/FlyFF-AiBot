from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Callable

from .MonsterConfig import (
    DEFAULT_MONSTER_CONFIG_PATH,
    load_native_monster_config,
)
from .NativeFlyffMonsterProvider import NativeFlyffMonsterProvider
from .Win32ProcessMemory import Win32MemoryBackend


def create_native_monster_provider(
    window_handle: int,
    *,
    config_path: str | Path = DEFAULT_MONSTER_CONFIG_PATH,
    backend: Win32MemoryBackend | None = None,
    clock: Callable[[], float] = monotonic,
) -> NativeFlyffMonsterProvider | None:
    config = load_native_monster_config(config_path)
    if not config.enabled:
        return None
    return NativeFlyffMonsterProvider.from_window_handle(
        window_handle,
        config,
        backend=backend,
        clock=clock,
    )


def create_native_monster_provider_from_process_id(
    process_id: int,
    *,
    config_path: str | Path = DEFAULT_MONSTER_CONFIG_PATH,
    backend: Win32MemoryBackend | None = None,
    clock: Callable[[], float] = monotonic,
) -> NativeFlyffMonsterProvider | None:
    config = load_native_monster_config(config_path)
    if not config.enabled:
        return None
    return NativeFlyffMonsterProvider.from_process_id(
        process_id,
        config,
        backend=backend,
        clock=clock,
    )
