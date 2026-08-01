from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Protocol

from .AnchoredPointerDiscovery import PointerRecoveryHints
from .MonsterConfig import NativeMonsterConfig
from .IndependentNativeReader import IndependentNativeReader
from .NativePointerRecovery import (
    DEFAULT_RECOVERY_TIMEOUT_SECONDS,
    PlayerPointerRecovery,
    PointerRecoveryMetrics,
    PointerRecoveryProgress,
    PointerRecoveryState,
    PointerRecoveryStatusCallback,
    recover_local_player_pointer,
)
from .PositionConfig import NativePositionConfig
from .PositionProvider import PositionProviderError
from .Win32ProcessMemory import ModuleInfo, Win32MemoryBackend, Win32ProcessMemory


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
    ANCHOR_HINTS_REQUIRED = "anchor_hints_required"
    MONSTER_CONSENSUS_NOT_FOUND = "monster_consensus_not_found"
    ACTOR_LAYOUT_INCONCLUSIVE = "actor_layout_inconclusive"
    SPAWN_PLAYER_NOT_FOUND = "spawn_player_not_found"
    ANCHOR_AMBIGUOUS = "anchor_ambiguous"
    MOVEMENT_REQUIRED = "movement_required"
    MOVEMENT_NOT_OBSERVED = "movement_not_observed"
    MOVEMENT_CANDIDATE_STALE = "movement_candidate_stale"

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
    mode: str = "world_linked"


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
        allow_independent_recovery: bool = False,
    ) -> None:
        self._memory = memory
        self.monster_config = monster_config
        self.position_config = position_config
        self.position_config_path = position_config_path
        self.monster_config_path = monster_config_path
        self._owns_memory = bool(owns_memory)
        self._clock = clock
        self._allow_independent_recovery = bool(allow_independent_recovery)
        self._lock = RLock()
        self._closed = False
        self._resources_closed = False
        self._active_recoveries = 0
        self._generation = 0
        self._recovery_state = PointerRecoveryState()
        self.last_recovery_result: NativeRecoveryResult | None = None
        self._independent_reader: IndependentNativeReader | None = None

        module_name = monster_config.module_name
        configured_player_offset = monster_config.player_pointer_hint_offset
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
            if (
                position_config.pointer_chain_offsets
                != monster_config.player_pointer_chain_offsets
            ):
                raise NativeProcessServiceError(
                    "Position and monster configs disagree on the local-player "
                    "pointer chain"
                )

        module_info_fn = getattr(self._memory, "module_info", None)
        module_info: ModuleInfo | None = None
        if callable(module_info_fn):
            try:
                candidate_info = module_info_fn(module_name)
                if isinstance(candidate_info, ModuleInfo):
                    module_info = candidate_info
            except Exception:
                module_info = None
        self._module_base = int(
            module_info.base_address
            if module_info is not None
            else self._memory.module_base(module_name)
        )
        self._module_info = module_info
        self._module_name = module_name
        self._pointer_width_bytes = 4
        self._configured_player_pointer_offset = int(configured_player_offset)
        self._configured_world_pointer_offset = int(
            monster_config.world_pointer_hint_offset
        )
        self._player_pointer_address = (
            self._module_base + self._configured_player_pointer_offset
        )
        self._world_pointer_address = (
            self._module_base + monster_config.world_pointer_hint_offset
        )
        self._player_pointer_chain_offsets = tuple(
            monster_config.player_pointer_chain_offsets
        )
        self._world_pointer_chain_offsets = tuple(
            monster_config.world_pointer_chain_offsets
        )
        self._world_field_offset = int(monster_config.world_offset)
        self._world_vtable_offset = monster_config.world_vtable_offset
        self._world_vtable_field_offset = int(
            monster_config.world_vtable_field_offset
        )
        self._world_identity_kind = monster_config.world_identity_kind
        self._self_pointer_offset = int(monster_config.self_pointer_offset)
        self._species_offset = int(monster_config.species_offset)
        self._active_species_offset = int(monster_config.active_species_offset)
        self._hp_offset = int(monster_config.hp_offset)
        self._x_offset = int(monster_config.x_offset)
        self._y_offset = int(monster_config.y_offset)
        self._z_offset = int(monster_config.z_offset)

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
        allow_independent_recovery: bool = True,
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
                allow_independent_recovery=allow_independent_recovery,
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
    def module_name(self) -> str:
        return self._module_name

    @property
    def module_path(self) -> str | None:
        if self._module_info is None or not self._module_info.path:
            return None
        return self._module_info.path

    @property
    def module_size(self) -> int | None:
        if self._module_info is None or self._module_info.size <= 0:
            return None
        return self._module_info.size

    @property
    def pointer_width_bytes(self) -> int:
        return self._pointer_width_bytes

    @property
    def configured_player_pointer_offset(self) -> int:
        return self._configured_player_pointer_offset

    @property
    def configured_world_pointer_offset(self) -> int:
        return self._configured_world_pointer_offset

    @property
    def player_pointer_chain_offsets(self) -> tuple[int, ...]:
        with self._lock:
            return self._player_pointer_chain_offsets

    @property
    def world_pointer_chain_offsets(self) -> tuple[int, ...]:
        with self._lock:
            return self._world_pointer_chain_offsets

    @property
    def world_field_offset(self) -> int:
        with self._lock:
            return self._world_field_offset

    @property
    def world_vtable_offset(self) -> int | None:
        with self._lock:
            return self._world_vtable_offset

    @property
    def world_vtable_field_offset(self) -> int:
        with self._lock:
            return self._world_vtable_field_offset

    @property
    def world_identity_kind(self) -> str:
        with self._lock:
            return self._world_identity_kind

    @property
    def self_pointer_offset(self) -> int:
        with self._lock:
            return self._self_pointer_offset

    @property
    def species_offset(self) -> int:
        with self._lock:
            return self._species_offset

    @property
    def active_species_offset(self) -> int:
        with self._lock:
            return self._active_species_offset

    @property
    def hp_offset(self) -> int:
        with self._lock:
            reader = self._independent_reader
            return (
                self._hp_offset
                if reader is None
                else int(reader.monster_hp_offset)
            )

    @property
    def monster_hp_candidate_offsets(self) -> tuple[int, ...]:
        with self._lock:
            reader = self._independent_reader
            return () if reader is None else reader.monster_hp_candidate_offsets

    @property
    def monster_hp_offset_validated(self) -> bool:
        with self._lock:
            reader = self._independent_reader
            return False if reader is None else reader.monster_hp_offset_validated

    @property
    def monster_hp_transition_support(self) -> tuple[tuple[int, int], ...]:
        with self._lock:
            reader = self._independent_reader
            return () if reader is None else reader.monster_hp_transition_support

    @property
    def x_offset(self) -> int:
        with self._lock:
            return self._x_offset

    @property
    def y_offset(self) -> int:
        with self._lock:
            return self._y_offset

    @property
    def z_offset(self) -> int:
        with self._lock:
            return self._z_offset

    @property
    def independent_reader(self) -> IndependentNativeReader | None:
        with self._lock:
            return self._independent_reader

    @property
    def independent_mode(self) -> bool:
        with self._lock:
            return self._independent_reader is not None

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

    def _resolve_pointer(
        self,
        pointer_address: int,
        chain_offsets: tuple[int, ...],
    ) -> int:
        value = self._read_u32(pointer_address)
        for offset in chain_offsets:
            value = self._read_u32(value + offset)
        return value

    def read_pointer_snapshot(self) -> NativePointerSnapshot:
        """Read one stable player/world state without scanning or recovery."""

        with self._lock:
            self._require_open()
            independent_reader = self._independent_reader
            if independent_reader is not None:
                try:
                    first = independent_reader.read_player()
                    second = independent_reader.read_player()
                    if (first.base, first.pointer_slot) != (
                        second.base,
                        second.pointer_slot,
                    ):
                        raise NativePointerSnapshotError(
                            "Recovered independent player alias changed during the snapshot"
                        )
                except NativePointerSnapshotError:
                    raise
                except Exception as error:
                    raise NativePointerSnapshotError(
                        "Recovered independent player aliases are stale or unreadable; "
                        "explicit pointer recovery is required"
                    ) from error
                return NativePointerSnapshot(
                    player_pointer_address=second.pointer_slot,
                    world_pointer_address=0,
                    player_base=second.base,
                    world_base=0,
                    generation=self._generation,
                    captured_at=float(self._clock()),
                    mode="independent",
                )
            if self._configured_player_pointer_offset <= 0:
                raise NativePointerSnapshotError(
                    "No local-player pointer hint is configured; explicit pointer "
                    "recovery is required"
                )
            if self._configured_world_pointer_offset <= 0:
                raise NativePointerSnapshotError(
                    "No current-world pointer hint is configured; explicit pointer "
                    "recovery is required"
                )
            player_pointer_address = self._player_pointer_address
            world_pointer_address = self._world_pointer_address
            player_chain = self._player_pointer_chain_offsets
            world_chain = self._world_pointer_chain_offsets
            self_pointer_offset = self._self_pointer_offset
            world_field_offset = self._world_field_offset
            world_vtable_offset = self._world_vtable_offset
            world_vtable_field_offset = self._world_vtable_field_offset
            try:
                player = self._resolve_pointer(player_pointer_address, player_chain)
                world = self._resolve_pointer(world_pointer_address, world_chain)
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
                        player + self_pointer_offset
                    )
                    != player
                ):
                    raise NativePointerSnapshotError(
                        "Local-player pointer is stale or unreadable; explicit "
                        "pointer recovery is required"
                    )
                world_rooted_player = bool(
                    player_pointer_address == world_pointer_address
                    and not world_chain
                    and len(player_chain) == 1
                )
                if (
                    not world_rooted_player
                    and self._read_u32(player + world_field_offset) != world
                ):
                    raise NativePointerSnapshotError(
                        "Player and world pointers are inconsistent; explicit "
                        "pointer recovery is required"
                    )
                if (
                    world_vtable_offset is not None
                    and self._read_u32(world + world_vtable_field_offset)
                    != self._module_base + world_vtable_offset
                ):
                    raise NativePointerSnapshotError(
                        "Current-world object identity is stale or invalid; "
                        "explicit pointer recovery is required"
                    )
                # A map/login transition may race the four reads above. A
                # second slot sample makes the returned pair coherent without
                # widening the ordinary path into discovery or recovery.
                if (
                    self._resolve_pointer(player_pointer_address, player_chain)
                    != player
                    or self._resolve_pointer(world_pointer_address, world_chain)
                    != world
                ):
                    raise NativePointerSnapshotError(
                        "Player/world pointers changed during the snapshot"
                    )
                if (
                    world_vtable_offset is not None
                    and self._read_u32(world + world_vtable_field_offset)
                    != self._module_base + world_vtable_offset
                ):
                    raise NativePointerSnapshotError(
                        "Current-world object identity changed during the snapshot"
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
        hints: PointerRecoveryHints | None = None,
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
                configured_world_pointer_offset=(
                    self._configured_world_pointer_offset
                ),
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
                hints=hints,
                allow_independent=self._allow_independent_recovery,
            )
            metrics = self._recovery_state.metrics_for(
                int(getattr(self._memory, "pid", 0)),
                module_base,
            )
            if metrics is None:
                raise NativeProcessServiceError(
                    "Explicit pointer recovery completed without metrics"
                )

            independent_reader: IndependentNativeReader | None = None
            if recovery is not None and recovery.independent_discovery is not None:
                if self._module_info is None:
                    raise NativeProcessServiceError(
                        "Independent recovery requires the mapped module extent"
                    )
                try:
                    independent_reader = IndependentNativeReader(
                        self._memory,
                        self._module_info,
                        recovery.independent_discovery,
                        configured_player_offset=(
                            recovery.configured_player_pointer_offset
                        ),
                        monster_current_hp_offset=self.monster_config.hp_offset,
                        monster_active_species_offset=(
                            self.monster_config.active_species_offset
                        ),
                        expected_full_hp_by_species=dict(
                            recovery.independent_expected_full_hp_by_species
                        ),
                        slots_each_direction=31,
                    )
                    independent_reader.read_player()

                    selected_species = {
                        int(value)
                        for value in (() if hints is None else hints.known_species_ids)
                        if int(value) > 0
                    }
                    anchored_species = {
                        int(species)
                        for species, _hp in (
                            recovery.independent_expected_full_hp_by_species
                        )
                    }
                    additional_species = tuple(
                        sorted(selected_species - anchored_species)
                    )
                    if additional_species and status_callback is not None:
                        try:
                            status_callback(
                                PointerRecoveryProgress(
                                    phase="runtime_actor_slots_ready",
                                    message=(
                                        "Pointer and actor layout recovery is complete. "
                                        f"The independent reader will inspect {len(independent_reader.actor_slots)} "
                                        "bounded slab addresses directly; no timed slot promotion "
                                        "or active-field gate is used. Additional selected species "
                                        f"{additional_species} are included whenever instantiated; "
                                        "pointer recovery is not rescanned."
                                    ),
                                    metrics=metrics,
                                )
                            )
                        except Exception:
                            pass
                except Exception as error:
                    raise NativeProcessServiceError(
                        "Independent player/actor reader could not be activated"
                    ) from error

            with self._lock:
                applied = bool(
                    recovery is not None
                    and not self._closed
                    and self._generation == observed_generation
                )
                if applied and recovery is not None:
                    self._independent_reader = independent_reader
                    self._player_pointer_address = recovery.player_pointer_address
                    self._configured_player_pointer_offset = (
                        recovery.player_pointer_offset
                    )
                    if recovery.world_pointer_address is not None:
                        self._world_pointer_address = recovery.world_pointer_address
                    if recovery.world_pointer_offset is not None:
                        self._configured_world_pointer_offset = (
                            recovery.world_pointer_offset
                        )
                    self._player_pointer_chain_offsets = tuple(
                        recovery.player_pointer_chain_offsets
                    )
                    self._world_pointer_chain_offsets = tuple(
                        recovery.world_pointer_chain_offsets
                    )
                    if recovery.world_field_offset is not None:
                        self._world_field_offset = recovery.world_field_offset
                    if recovery.world_vtable_offset is not None:
                        self._world_vtable_offset = recovery.world_vtable_offset
                    if recovery.world_vtable_field_offset is not None:
                        self._world_vtable_field_offset = (
                            recovery.world_vtable_field_offset
                        )
                    if recovery.world_identity_kind is not None:
                        self._world_identity_kind = recovery.world_identity_kind
                    if recovery.self_pointer_offset is not None:
                        self._self_pointer_offset = recovery.self_pointer_offset
                    if recovery.species_offset is not None:
                        self._species_offset = recovery.species_offset
                    if recovery.active_species_offset is not None:
                        self._active_species_offset = recovery.active_species_offset
                    if recovery.hp_offset is not None:
                        self._hp_offset = recovery.hp_offset
                    if recovery.x_offset is not None:
                        self._x_offset = recovery.x_offset
                    if recovery.y_offset is not None:
                        self._y_offset = recovery.y_offset
                    if recovery.z_offset is not None:
                        self._z_offset = recovery.z_offset
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
