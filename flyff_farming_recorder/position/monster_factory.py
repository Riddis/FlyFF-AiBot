from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Callable

from .MonsterConfig import (
    DEFAULT_MONSTER_CONFIG_PATH,
    load_native_monster_config,
)
from .native_process_service import NativeProcessService
from .NativeFlyffMonsterProvider import NativeFlyffMonsterProvider
from .NativePointerRecovery import recover_interrupted_pointer_persistence
from .PositionConfig import DEFAULT_POSITION_CONFIG_PATH
from .Win32ProcessMemory import Win32MemoryBackend


def create_native_monster_provider(
    window_handle: int,
    *,
    config_path: str | Path = DEFAULT_MONSTER_CONFIG_PATH,
    position_config_path: str | Path = DEFAULT_POSITION_CONFIG_PATH,
    backend: Win32MemoryBackend | None = None,
    clock: Callable[[], float] = monotonic,
    native_service: NativeProcessService | None = None,
) -> NativeFlyffMonsterProvider | None:
    recover_interrupted_pointer_persistence(
        position_config_path=position_config_path,
        monster_config_path=config_path,
    )
    config = load_native_monster_config(config_path)
    if not config.enabled:
        return None
    if native_service is not None:
        return NativeFlyffMonsterProvider.from_native_service(
            native_service,
            config,
            clock=clock,
        )
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
    position_config_path: str | Path = DEFAULT_POSITION_CONFIG_PATH,
    backend: Win32MemoryBackend | None = None,
    clock: Callable[[], float] = monotonic,
) -> NativeFlyffMonsterProvider | None:
    recover_interrupted_pointer_persistence(
        position_config_path=position_config_path,
        monster_config_path=config_path,
    )
    config = load_native_monster_config(config_path)
    if not config.enabled:
        return None
    return NativeFlyffMonsterProvider.from_process_id(
        process_id,
        config,
        backend=backend,
        clock=clock,
    )
