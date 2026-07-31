from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Protocol

from .MonsterConfig import NativeMonsterConfig
from .NativePointerRecovery import (
    DEFAULT_RECOVERY_TIMEOUT_SECONDS,
    PlayerPointerRecovery,
    PointerRecoveryMetrics,
    PointerRecoveryState,
    PointerRecoveryStatusCallback,
    recover_local_player_pointer,
)
from .PositionConfig import NativePositionConfig
from .PositionProvider import PositionProviderError
from .Win32ProcessMemory import Win32MemoryBackend, Win32ProcessMemory


class NativeProcessMemory(Protocol):
    pid: int

    def read(self, address: int, size: int) -> bytes: ...

    def module_base(self, module_name: str) -> int: ...

    def readable_regions(
        self,
        *,
        maximum_address: int = 0x7FFFFFFF,
        private_only: bool = True,
    ) -> tuple[object, ...]: ...

    def close(self) -> None: ...


class NativeProcessServiceError(PositionProviderError):
    """The shared native process attachment could not complete an operation."""


class NativePointerSnapshotError(NativeProcessServiceError):
    """Player/world pointer state was null, stale, or changed during a read."""


class NativeRecoveryOutcome(str, Enum):
    SUCCESS = "success"
    CACHE_HIT = "cache_hit"
    NOT_FOUND = "not_found"
    DEADLINE = "deadline"
    CANCELLED = "cancelled"
    NEGATIVE_CACHE = "negative_cache"
    ERROR = "error"
    SHARED_UNKNOWN = "shared_unknown"

    @classmethod
    def from_metrics(cls, metrics: PointerRecoveryMetrics) -> NativeRecoveryOutcome:
        try:
            return cls(metrics.outcome)
        except ValueError:
            return cls.ERROR


@dataclass(frozen=True, slots=True)
class NativePointerSnapshot:
    """One stable player/world pointer sample for a native observation step."""

    player_pointer_address: int
    world_pointer_address: int
    player_base: int
    world_base: int
    generation: int
    captured_at: float


@dataclass(frozen=True, slots=True)
class NativeRecoveryResult:
    """Typed result returned by a synchronous, explicitly requested recovery."""

    outcome: NativeRecoveryOutcome
    recovery: PlayerPointerRecovery | None
    metrics: PointerRecoveryMetrics
    applied: bool

    @property
    def succeeded(self) -> bool:
        return self.recovery is not None


class NativeProcessService:
    """Own one process handle and one coherent native pointer state.

    The service is deliberately synchronous. A runtime-managed worker may call
    :meth:`recover_pointers`, passing its cancellation token and deadline, but
    this class never creates a thread. Ordinary snapshots do only a bounded
    sequence of reads and never scan or recover.
    """

    def __init__(
        self,
        memory: NativeProcessMemory,
        monster_config: NativeMonsterConfig,
        *,
        position_config: NativePositionConfig | None = None,
        position_config_path: str | Path | None = None,
        monster_config_path: str | Path | None = None,
        owns_memory: bool = True,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._memory = memory
        self.monster_config = monster_config
        self.position_config = position_config
        self.position_config_path = position_config_path
        self.monster_config_path = monster_config_path
        self._owns_memory = bool(owns_memory)
        self._clock = clock
        self._lock = RLock()
        self._closed = False
        self._resources_closed = False
        self._active_recoveries = 0
        self._generation = 0
        self._recovery_state = PointerRecoveryState()
        self.last_recovery_result: NativeRecoveryResult | None = None

        module_name = monster_config.module_name
        configured_player_offset = monster_config.player_pointer_offset
        if (
            position_config is not None
            and position_config.enabled
            and position_config.resolver == "module_pointer"
        ):
            if position_config.module_name != module_name:
                raise NativeProcessServiceError(
                    "Position and monster readers must use the same module for "
                    "a shared native attachment"
                )
            if position_config.pointer_offset is None:
                raise NativeProcessServiceError(
                    "The shared module-pointer position reader needs a pointer "
                    "offset"
                )
            if position_config.pointer_offset != configured_player_offset:
                raise NativeProcessServiceError(
                    "Position and monster configs disagree on the local-player "
                    "pointer offset"
                )

        self._module_base = int(self._memory.module_base(module_name))
        self._configured_player_pointer_offset = int(configured_player_offset)
        self._player_pointer_address = (
            self._module_base + self._configured_player_pointer_offset
        )
        self._world_pointer_address = (
            self._module_base + monster_config.world_pointer_offset
        )

    @classmethod
    def from_window_handle(
        cls,
        window_handle: int,
        monster_config: NativeMonsterConfig,
        *,
        position_config: NativePositionConfig | None = None,
        position_config_path: str | Path | None = None,
        monster_config_path: str | Path | None = None,
        backend: Win32MemoryBackend | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> NativeProcessService:
        memory = Win32ProcessMemory.from_window_handle(
            window_handle,
            backend=backend,
        )
        try:
            return cls(
                memory,
                monster_config,
                position_config=position_config,
                position_config_path=position_config_path,
                monster_config_path=monster_config_path,
                owns_memory=True,
                clock=clock,
            )
        except Exception:
            memory.close()
            raise

    @property
    def memory(self) -> NativeProcessMemory:
        return self._memory

    @property
    def module_base(self) -> int:
        return self._module_base

    @property
    def player_pointer_address(self) -> int:
        with self._lock:
            return self._player_pointer_address

    @property
    def world_pointer_address(self) -> int:
        with self._lock:
            return self._world_pointer_address

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise NativeProcessServiceError("Native process attachment is closed")

    def _read_u32(self, address: int) -> int:
        return int(struct.unpack("<I", self._memory.read(address, 4))[0])

    def read_pointer_snapshot(self) -> NativePointerSnapshot:
        """Read one stable player/world state without scanning or recovery."""

        with self._lock:
            self._require_open()
            player_pointer_address = self._player_pointer_address
            world_pointer_address = self._world_pointer_address
            try:
                player = self._read_u32(player_pointer_address)
                world = self._read_u32(world_pointer_address)
                if player <= 0:
                    raise NativePointerSnapshotError(
                        "Local-player pointer is null; explicit pointer recovery "
                        "is required"
                    )
                if world <= 0:
                    raise NativePointerSnapshotError(
                        "Current-world pointer is null; explicit pointer recovery "
                        "is required"
                    )
                if (
                    self._read_u32(
                        player + self.monster_config.self_pointer_offset
                    )
                    != player
                ):
                    raise NativePointerSnapshotError(
                        "Local-player pointer is stale or unreadable; explicit "
                        "pointer recovery is required"
                    )
                if (
                    self._read_u32(player + self.monster_config.world_offset)
                    != world
                ):
                    raise NativePointerSnapshotError(
                        "Player and world pointers are inconsistent; explicit "
                        "pointer recovery is required"
                    )
                # A map/login transition may race the four reads above. A
                # second slot sample makes the returned pair coherent without
                # widening the ordinary path into discovery or recovery.
                if (
                    self._read_u32(player_pointer_address) != player
                    or self._read_u32(world_pointer_address) != world
                ):
                    raise NativePointerSnapshotError(
                        "Player/world pointers changed during the snapshot"
                    )
            except NativePointerSnapshotError:
                raise
            except Exception as error:
                raise NativePointerSnapshotError(
                    "Could not read a coherent native player/world snapshot"
                ) from error

            return NativePointerSnapshot(
                player_pointer_address=player_pointer_address,
                world_pointer_address=world_pointer_address,
                player_base=player,
                world_base=world,
                generation=self._generation,
                captured_at=float(self._clock()),
            )

    def recover_pointers(
        self,
        *,
        persist: bool = False,
        cancellation: object | None = None,
        deadline: float | None = None,
        timeout_seconds: float = DEFAULT_RECOVERY_TIMEOUT_SECONDS,
        status_callback: PointerRecoveryStatusCallback | None = None,
    ) -> NativeRecoveryResult:
        """Synchronously run explicit recovery under this attachment owner."""

        with self._lock:
            self._require_open()
            observed_generation = self._generation
            module_base = self._module_base
            configured_player_offset = self._configured_player_pointer_offset
            self._active_recoveries += 1

        try:
            # Recovery is intentionally outside the short ordinary-read lock.
            # The process handle remains open until every registered attempt
            # exits, even if close() is requested while this call is scanning.
            recovery = recover_local_player_pointer(
                self._memory,
                module_base=module_base,
                configured_player_pointer_offset=configured_player_offset,
                state=self._recovery_state,
                monster_config=self.monster_config,
                persist=persist,
                position_config_path=self.position_config_path,
                monster_config_path=self.monster_config_path,
                cancellation=cancellation,
                deadline=deadline,
                timeout_seconds=timeout_seconds,
                status_callback=status_callback,
                clock=self._clock,
            )
            metrics = self._recovery_state.metrics_for(
                int(getattr(self._memory, "pid", 0)),
                module_base,
            )
            if metrics is None:
                raise NativeProcessServiceError(
                    "Explicit pointer recovery completed without metrics"
                )

            with self._lock:
                applied = bool(
                    recovery is not None
                    and not self._closed
                    and self._generation == observed_generation
                )
                if applied and recovery is not None:
                    self._player_pointer_address = recovery.player_pointer_address
                    if recovery.world_pointer_address is not None:
                        self._world_pointer_address = recovery.world_pointer_address
                    self._generation += 1
                result = NativeRecoveryResult(
                    outcome=NativeRecoveryOutcome.from_metrics(metrics),
                    recovery=recovery,
                    metrics=metrics,
                    applied=applied,
                )
                if not self._closed:
                    self.last_recovery_result = result
                return result
        finally:
            self._finish_recovery()

    def _finish_recovery(self) -> None:
        close_memory = False
        with self._lock:
            self._active_recoveries -= 1
            if self._active_recoveries < 0:
                self._active_recoveries = 0
                raise NativeProcessServiceError(
                    "Native recovery lifecycle count became invalid"
                )
            close_memory = self._prepare_resource_close_locked()
        if close_memory:
            self._memory.close()

    def _prepare_resource_close_locked(self) -> bool:
        if (
            not self._closed
            or self._active_recoveries > 0
            or self._resources_closed
        ):
            return False
        self._resources_closed = True
        self._recovery_state.clear()
        return self._owns_memory

    def close(self) -> None:
        close_memory = False
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close_memory = self._prepare_resource_close_locked()
        if close_memory:
            self._memory.close()
