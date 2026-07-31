from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

from .MonsterConfig import (
    DEFAULT_MONSTER_CONFIG_PATH,
    load_native_monster_config,
)
from .native_process_service import NativeProcessService
from .NativeFlyffMonsterProvider import NativeFlyffMonsterProvider
from .NativeFlyffPositionProvider import NativeFlyffPositionProvider
from .PositionConfig import (
    DEFAULT_POSITION_CONFIG_PATH,
    load_native_position_config,
)
from .Win32ProcessMemory import Win32MemoryBackend


@dataclass(slots=True)
class NativeProviderAttachment:
    """Providers borrowing one deterministically owned native process service."""

    service: NativeProcessService | None
    position_provider: NativeFlyffPositionProvider | None
    monster_provider: NativeFlyffMonsterProvider | None
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        for provider in (self.position_provider, self.monster_provider):
            if provider is None:
                continue
            try:
                provider.close()
            except Exception as error:  # noqa: BLE001 - close the owner too.
                if first_error is None:
                    first_error = error
        if self.service is not None:
            try:
                self.service.close()
            except Exception as error:  # noqa: BLE001 - preserve first failure.
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def create_native_provider_attachment(
    window_handle: int,
    *,
    position_config_path: str | Path = DEFAULT_POSITION_CONFIG_PATH,
    monster_config_path: str | Path = DEFAULT_MONSTER_CONFIG_PATH,
    backend: Win32MemoryBackend | None = None,
    clock: Callable[[], float] = monotonic,
) -> NativeProviderAttachment:
    """Build enabled native providers over exactly one process attachment."""

    position_config = load_native_position_config(position_config_path)
    monster_config = load_native_monster_config(monster_config_path)
    if not position_config.enabled and not monster_config.enabled:
        return NativeProviderAttachment(None, None, None)

    service = NativeProcessService.from_window_handle(
        window_handle,
        monster_config,
        position_config=position_config,
        backend=backend,
        clock=clock,
    )
    position_provider: NativeFlyffPositionProvider | None = None
    monster_provider: NativeFlyffMonsterProvider | None = None
    try:
        if position_config.enabled:
            position_provider = NativeFlyffPositionProvider.from_native_service(
                service,
                position_config,
                clock=clock,
            )
        if monster_config.enabled:
            monster_provider = NativeFlyffMonsterProvider.from_native_service(
                service,
                monster_config,
                clock=clock,
            )
        return NativeProviderAttachment(
            service,
            position_provider,
            monster_provider,
        )
    except Exception:
        if position_provider is not None:
            position_provider.close()
        if monster_provider is not None:
            monster_provider.close()
        service.close()
        raise
