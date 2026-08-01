from __future__ import annotations

import math
import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from time import monotonic, perf_counter
from typing import Protocol, cast

from .MonsterConfig import NativeMonsterConfig
from .IndependentNativeReader import IndependentNativeReader
from .native_process_service import (
    NativePointerSnapshot,
    NativeProcessService,
)
from .NativePointerRecovery import (
    PlayerPointerRecovery,
)
from .PositionProvider import PlayerPose, PositionProviderError
from .Win32ProcessMemory import (
    MemorySearchCancelled,
    MemorySearchDeadline,
    MemorySearchDiagnostics,
    Win32MemoryBackend,
    Win32ProcessMemory,
)


class NativeMonsterReadError(PositionProviderError):
    """A native actor-pool read could not be completed."""


class ActorMemory(Protocol):
    last_search_diagnostics: MemorySearchDiagnostics

    def read(self, address: int, size: int) -> bytes: ...

    def module_base(self, module_name: str) -> int: ...

    def find_u32(
        self,
        value: int,
        *,
        maximum_address: int,
        private_only: bool,
        chunk_size: int,
        cancellation: object | None = None,
        deadline: float | None = None,
    ) -> tuple[int, ...]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class NativeActor:
    base_address: int
    species_id: int
    hp: int
    x: float
    y: float
    z: float
    distance_native: float
    active_species_id: int
    hp_offset: int | None = None
    hp_candidates: tuple[tuple[int, int], ...] = ()
    hp_offset_validated: bool = False


class ActorCacheOutcome(str, Enum):
    REFRESHED = "refreshed"
    CACHED = "cached"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"
    DEADLINE = "deadline"
    WORLD_MISMATCH = "world_mismatch"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ActorCacheRefreshResult:
    outcome: ActorCacheOutcome
    world_base: int
    generation: int
    slot_count: int = 0
    message: str = ""

    @property
    def ready(self) -> bool:
        return self.outcome in {
            ActorCacheOutcome.REFRESHED,
            ActorCacheOutcome.CACHED,
        }


@dataclass(frozen=True, slots=True)
class CachedActorReadResult:
    outcome: ActorCacheOutcome
    world_base: int
    generation: int
    actors: tuple[NativeActor, ...] = ()
    message: str = ""
    tracked_actors: tuple[NativeActor, ...] = ()

    @property
    def ready(self) -> bool:
        return self.outcome is ActorCacheOutcome.READY


@dataclass(frozen=True, slots=True)
class ActorPoolDiagnostics:
    player_base: int
    world_base: int
    discovered_slots: int
    discovery_regions_considered: int = 0
    discovery_regions_read: int = 0
    discovery_bytes_read: int = 0
    discovery_elapsed_seconds: float = 0.0
    discovery_read_failures: int = 0
    world_pointer_matches: int = 0
    rejected_invalid_actor: int = 0
    rejected_wrong_world: int = 0
    rejected_not_present: int = 0
    rejected_dead: int = 0
    rejected_species: int = 0
    rejected_distance: int = 0
    unreadable_cached_slots: int = 0
    runtime_promoted_slots: int = 0
    pending_actor_slot_probes: int = 0


class NativeFlyffMonsterProvider:
    """Read-only global FlyFF actor discovery and local monster observation.

    FlyFF stores monsters in several independent 32-slot slabs. The local
    player can live in a completely unrelated allocation, so discovery cannot
    walk outward from the player by the actor stride. Instead, an infrequent
    process-wide read-only scan searches committed readable memory for the
    current-world pointer at actor ``+0x16C`` and validates each candidate with
    the actor self pointer at ``+0x1EE0``.
    """

    def __init__(
        self,
        memory: ActorMemory,
        config: NativeMonsterConfig,
        *,
        clock: Callable[[], float] = monotonic,
        native_service: NativeProcessService | None = None,
        owns_memory: bool = True,
    ) -> None:
        self._memory = memory
        self._native_service = native_service
        self._owns_memory = bool(owns_memory)
        self._closed = False
        self.config = config
        self._clock = clock
        self._module_base = (
            native_service.module_base
            if native_service is not None
            else self._memory.module_base(config.module_name)
        )
        self._player_pointer_address = self._module_base + config.player_pointer_hint_offset
        self._world_pointer_address = self._module_base + config.world_pointer_hint_offset
        self._slot_bases: tuple[int, ...] = ()
        self._cached_world: int | None = None
        self._cached_generation: int | None = None
        self._last_discovery_at = -math.inf
        self._lock = RLock()
        self._refresh_active = False
        self._resources_closed = False
        self.last_diagnostics: ActorPoolDiagnostics | None = None
        self._last_pointer_recovery: PlayerPointerRecovery | None = None

    @classmethod
    def from_window_handle(
        cls,
        window_handle: int,
        config: NativeMonsterConfig,
        *,
        backend: Win32MemoryBackend | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> NativeFlyffMonsterProvider:
        memory = Win32ProcessMemory.from_window_handle(
            window_handle,
            backend=backend,
        )
        try:
            return cls(memory, config, clock=clock)
        except Exception:
            memory.close()
            raise

    @classmethod
    def from_process_id(
        cls,
        process_id: int,
        config: NativeMonsterConfig,
        *,
        backend: Win32MemoryBackend | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> NativeFlyffMonsterProvider:
        memory = Win32ProcessMemory(
            process_id,
            backend=backend,
        )
        try:
            return cls(memory, config, clock=clock)
        except Exception:
            memory.close()
            raise

    @classmethod
    def from_native_service(
        cls,
        native_service: NativeProcessService,
        config: NativeMonsterConfig,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> NativeFlyffMonsterProvider:
        return cls(
            cast(ActorMemory, cast(object, native_service.memory)),
            config,
            clock=clock,
            native_service=native_service,
            owns_memory=False,
        )

    @property
    def module_base(self) -> int:
        return self._module_base

    @property
    def player_pointer_address(self) -> int:
        if self._native_service is not None:
            return self._native_service.player_pointer_address
        return self._player_pointer_address

    @property
    def world_pointer_address(self) -> int:
        if self._native_service is not None:
            return self._native_service.world_pointer_address
        return self._world_pointer_address

    @property
    def discovered_slot_bases(self) -> tuple[int, ...]:
        reader = self._independent_reader()
        if reader is not None:
            return reader.actor_slots
        return self._slot_bases

    @property
    def last_pointer_recovery(self) -> PlayerPointerRecovery | None:
        if self._native_service is not None:
            result = self._native_service.last_recovery_result
            return None if result is None else result.recovery
        return self._last_pointer_recovery

    def _independent_reader(self) -> IndependentNativeReader | None:
        if self._native_service is None:
            return None
        return self._native_service.independent_reader

    @staticmethod
    def _native_actors_from_independent(monsters) -> tuple[NativeActor, ...]:
        return tuple(
            NativeActor(
                base_address=item.base,
                species_id=item.species,
                hp=item.hp,
                x=item.x,
                y=item.y,
                z=item.z,
                distance_native=item.distance_native,
                active_species_id=getattr(item, "active_species", item.species),
                hp_offset=int(getattr(item, "hp_offset", 0)) or None,
                hp_candidates=tuple(getattr(item, "hp_candidates", ())),
                hp_offset_validated=bool(
                    getattr(item, "hp_offset_validated", False)
                ),
            )
            for item in monsters
        )

    @staticmethod
    def _tracked_actors_from_independent(
        actor_states,
        *,
        vision_radius_native: float,
    ) -> tuple[NativeActor, ...]:
        tracked: list[NativeActor] = []
        for item in actor_states:
            if not bool(getattr(item, "target_species", False)):
                continue
            species = int(getattr(item, "species", 0))
            hp = int(getattr(item, "hp", -1))
            distance = float(getattr(item, "distance_native", math.inf))
            x = float(getattr(item, "x", math.nan))
            y = float(getattr(item, "y", math.nan))
            z = float(getattr(item, "z", math.nan))
            if (
                species <= 0
                or hp < 0
                or not all(math.isfinite(value) for value in (x, y, z, distance))
                or distance > float(vision_radius_native)
            ):
                continue
            tracked.append(
                NativeActor(
                    base_address=int(getattr(item, "base")),
                    species_id=species,
                    hp=hp,
                    x=x,
                    y=y,
                    z=z,
                    distance_native=distance,
                    active_species_id=int(
                        getattr(item, "active_species", species)
                    ),
                    hp_offset=(
                        int(getattr(item, "hp_offset", 0)) or None
                    ),
                    hp_candidates=tuple(getattr(item, "hp_candidates", ())),
                    hp_offset_validated=bool(
                        getattr(item, "hp_offset_validated", False)
                    ),
                )
            )
        tracked.sort(key=lambda actor: (actor.distance_native, actor.base_address))
        return tuple(tracked)

    def read_actor_hp_states(
        self,
        candidates: Iterable[tuple[int, int]],
    ) -> dict[tuple[int, int], int]:
        """Read species/HP directly for post-cast confirmation.

        Independent mode delegates to the exact recovered actor layout and does
        not apply map radius, active-state, or cache membership filters. The
        legacy world mode performs the equivalent direct field reads.
        """

        normalized = tuple(
            (int(base), int(species))
            for base, species in candidates
            if int(base) > 0x10000 and int(species) > 0
        )
        independent = self._independent_reader()
        if independent is not None:
            return independent.read_actor_hp_states(normalized)

        result: dict[tuple[int, int], int] = {}
        for base, expected_species in normalized:
            try:
                species = self._read_i32(base + self._species_field_offset)
                hp = self._read_i32(base + self._hp_field_offset)
            except Exception:
                continue
            if species == expected_species and hp >= 0:
                result[(base, expected_species)] = hp
        return result

    def _read_u32(self, address: int) -> int:
        return int(struct.unpack("<I", self._memory.read(address, 4))[0])

    def _read_i32(self, address: int) -> int:
        return int(struct.unpack("<i", self._memory.read(address, 4))[0])

    def _read_float(self, address: int) -> float:
        return float(struct.unpack("<f", self._memory.read(address, 4))[0])

    def _resolve_pointer(
        self,
        address: int,
        chain_offsets: tuple[int, ...],
    ) -> int:
        value = self._read_u32(address)
        for offset in chain_offsets:
            value = self._read_u32(value + offset)
        return value

    @property
    def _world_field_offset(self) -> int:
        if self._native_service is not None:
            return self._native_service.world_field_offset
        return self.config.world_offset

    @property
    def _self_field_offset(self) -> int:
        if self._native_service is not None:
            return self._native_service.self_pointer_offset
        return self.config.self_pointer_offset

    @property
    def _species_field_offset(self) -> int:
        if self._native_service is not None:
            return self._native_service.species_offset
        return self.config.species_offset

    @property
    def _active_species_field_offset(self) -> int:
        if self._native_service is not None:
            return self._native_service.active_species_offset
        return self.config.active_species_offset

    @property
    def _hp_field_offset(self) -> int:
        if self._native_service is not None:
            return self._native_service.hp_offset
        return self.config.hp_offset

    @property
    def _x_field_offset(self) -> int:
        if self._native_service is not None:
            return self._native_service.x_offset
        return self.config.x_offset

    @property
    def _y_field_offset(self) -> int:
        if self._native_service is not None:
            return self._native_service.y_offset
        return self.config.y_offset

    @property
    def _z_field_offset(self) -> int:
        if self._native_service is not None:
            return self._native_service.z_offset
        return self.config.z_offset

    def _shared_pointer_snapshot(
        self,
        pointer_snapshot: NativePointerSnapshot | None,
    ) -> NativePointerSnapshot | None:
        if self._native_service is None:
            return None
        if pointer_snapshot is not None:
            return pointer_snapshot
        try:
            return self._native_service.read_pointer_snapshot()
        except Exception as error:
            raise NativeMonsterReadError(str(error)) from error

    def read_player_base(
        self,
        *,
        pointer_snapshot: NativePointerSnapshot | None = None,
    ) -> int:
        shared = self._shared_pointer_snapshot(pointer_snapshot)
        if shared is not None:
            return shared.player_base
        base = self._resolve_pointer(
            self._player_pointer_address,
            self.config.player_pointer_chain_offsets,
        )
        if base <= 0:
            raise NativeMonsterReadError(
                "Local-player pointer is null; explicit pointer recovery is "
                "required after login/map transition checks complete"
            )
        if not self._is_self_valid(base):
            raise NativeMonsterReadError(
                "Local-player pointer is stale or unreadable; explicit pointer "
                "recovery is required"
            )
        return base

    def read_world_base(
        self,
        *,
        pointer_snapshot: NativePointerSnapshot | None = None,
    ) -> int:
        shared = self._shared_pointer_snapshot(pointer_snapshot)
        if shared is not None:
            return shared.world_base
        base = self._resolve_pointer(
            self._world_pointer_address,
            self.config.world_pointer_chain_offsets,
        )
        if base <= 0:
            raise NativeMonsterReadError(
                "Current-world pointer is null; explicit pointer recovery is "
                "required after login/map transition checks complete"
            )
        try:
            player = self._resolve_pointer(
                self._player_pointer_address,
                self.config.player_pointer_chain_offsets,
            )
            if player <= 0:
                raise NativeMonsterReadError(
                    "Local-player pointer is null while validating the current "
                    "world pointer"
                )
            world_rooted_player = bool(
                self._player_pointer_address == self._world_pointer_address
                and not self.config.world_pointer_chain_offsets
                and len(self.config.player_pointer_chain_offsets) == 1
            )
            player_world = (
                base
                if world_rooted_player
                else self._read_u32(player + self._world_field_offset)
            )
        except NativeMonsterReadError:
            raise
        except Exception as error:
            raise NativeMonsterReadError(
                "Current-world pointer could not be validated against the local player"
            ) from error
        if player_world != base:
            raise NativeMonsterReadError(
                "Current-world pointer is stale or inconsistent with the "
                "local player; explicit pointer recovery is required"
            )
        return base

    def _is_self_valid(self, base: int) -> bool:
        if base <= 0:
            return False
        try:
            return self._read_u32(base + self._self_field_offset) == base
        except Exception:  # A probe outside committed/readable memory is normal.
            return False

    def discover_slots(
        self,
        *,
        force: bool = False,
        pointer_snapshot: NativePointerSnapshot | None = None,
    ) -> tuple[int, ...]:
        with self._lock:
            now = self._clock()
            shared = self._shared_pointer_snapshot(pointer_snapshot)
            independent = self._independent_reader()
            if independent is not None:
                player = self.read_player_base(pointer_snapshot=shared)
                ordered = independent.actor_slots
                self._slot_bases = ordered
                self._cached_world = 0
                self._cached_generation = (
                    None if shared is None else shared.generation
                )
                self._last_discovery_at = now
                self.last_diagnostics = ActorPoolDiagnostics(
                    player_base=player,
                    world_base=0,
                    discovered_slots=len(ordered),
                )
                return ordered
            player = self.read_player_base(pointer_snapshot=shared)
            world = self.read_world_base(pointer_snapshot=shared)
            interval = self.config.discovery_interval_seconds
            cache_valid = (
                not force
                and self._slot_bases
                and self._cached_world == world
                and now - self._last_discovery_at < interval
            )
            if cache_valid:
                return self._slot_bases

            discovery_started_at = perf_counter()
            matches = self._memory.find_u32(
                world,
                maximum_address=self.config.maximum_scan_address,
                private_only=self.config.private_memory_only,
                chunk_size=self.config.discovery_chunk_bytes,
            )
            discovery_elapsed = max(0.0, perf_counter() - discovery_started_at)
            slots: set[int] = set()
            invalid = 0
            for world_field_address in matches:
                base = int(world_field_address) - self._world_field_offset
                if base <= 0 or not self._is_self_valid(base):
                    invalid += 1
                    continue
                try:
                    if self._read_u32(base + self._world_field_offset) != world:
                        invalid += 1
                        continue
                except Exception:
                    invalid += 1
                    continue
                slots.add(base)

            # The local player is an actor and serves as an important discovery
            # sanity check even when its allocation is separate from the mobs.
            if self._is_self_valid(player):
                slots.add(player)

            diagnostics = getattr(
                self._memory,
                "last_search_diagnostics",
                MemorySearchDiagnostics(matches=len(matches)),
            )
            ordered = tuple(sorted(slots))
            self._slot_bases = ordered
            self._cached_world = world
            self._last_discovery_at = now
            self.last_diagnostics = ActorPoolDiagnostics(
                player_base=player,
                world_base=world,
                discovered_slots=len(ordered),
                discovery_regions_considered=diagnostics.regions_considered,
                discovery_regions_read=diagnostics.regions_read,
                discovery_bytes_read=diagnostics.bytes_read,
                discovery_elapsed_seconds=discovery_elapsed,
                discovery_read_failures=diagnostics.read_failures,
                world_pointer_matches=diagnostics.matches,
                rejected_invalid_actor=invalid,
            )
            return ordered

    @staticmethod
    def _cancellation_requested(cancellation: object | None) -> bool:
        if cancellation is None:
            return False
        cancelled = getattr(cancellation, "cancelled", None)
        if cancelled is not None:
            return bool(cancelled() if callable(cancelled) else cancelled)
        is_set = getattr(cancellation, "is_set", None)
        return bool(is_set()) if callable(is_set) else False

    def _refresh_stop_outcome(
        self,
        cancellation: object | None,
        deadline: float | None,
    ) -> ActorCacheOutcome | None:
        if self._cancellation_requested(cancellation):
            return ActorCacheOutcome.CANCELLED
        if deadline is not None and self._clock() >= float(deadline):
            return ActorCacheOutcome.DEADLINE
        return None

    def _finish_slot_refresh(self) -> None:
        close_memory = False
        with self._lock:
            self._refresh_active = False
            if self._closed and self._owns_memory and not self._resources_closed:
                self._resources_closed = True
                close_memory = True
        if close_memory:
            self._memory.close()

    def refresh_slot_cache(
        self,
        pointer_snapshot: NativePointerSnapshot,
        *,
        cancellation: object | None = None,
        deadline: float | None = None,
        force: bool = False,
    ) -> ActorCacheRefreshResult:
        """Explicitly refresh actor slots without blocking ordinary reads.

        At most one refresh scans at a time. The scan runs outside the cache
        lock, cooperatively observes cancellation/deadline boundaries, and is
        published only if the player/world snapshot is still current.
        """

        if not isinstance(pointer_snapshot, NativePointerSnapshot):
            raise TypeError("pointer_snapshot must be a NativePointerSnapshot")
        now = self._clock()
        independent = self._independent_reader()
        if independent is not None:
            stopped = self._refresh_stop_outcome(cancellation, deadline)
            if stopped is not None:
                return ActorCacheRefreshResult(
                    stopped,
                    0,
                    pointer_snapshot.generation,
                )
            ordered = independent.actor_slots
            with self._lock:
                if self._closed:
                    return ActorCacheRefreshResult(
                        ActorCacheOutcome.UNAVAILABLE,
                        0,
                        pointer_snapshot.generation,
                        message="Native actor provider is closed",
                    )
                cached = bool(
                    not force
                    and self._slot_bases == ordered
                    and self._cached_world == 0
                    and self._cached_generation == pointer_snapshot.generation
                )
                self._slot_bases = ordered
                self._cached_world = 0
                self._cached_generation = pointer_snapshot.generation
                self._last_discovery_at = now
                self.last_diagnostics = ActorPoolDiagnostics(
                    player_base=pointer_snapshot.player_base,
                    world_base=0,
                    discovered_slots=len(ordered),
                )
            return ActorCacheRefreshResult(
                ActorCacheOutcome.CACHED if cached else ActorCacheOutcome.REFRESHED,
                0,
                pointer_snapshot.generation,
                slot_count=len(ordered),
                message="Independent anchored actor cache is ready",
            )
        with self._lock:
            if self._closed:
                return ActorCacheRefreshResult(
                    ActorCacheOutcome.UNAVAILABLE,
                    pointer_snapshot.world_base,
                    pointer_snapshot.generation,
                    message="Native actor provider is closed",
                )
            cache_valid = bool(
                not force
                and self._slot_bases
                and self._cached_world == pointer_snapshot.world_base
                and self._cached_generation == pointer_snapshot.generation
                and now - self._last_discovery_at
                < self.config.discovery_interval_seconds
            )
            if cache_valid:
                return ActorCacheRefreshResult(
                    ActorCacheOutcome.CACHED,
                    pointer_snapshot.world_base,
                    pointer_snapshot.generation,
                    slot_count=len(self._slot_bases),
                )
            if self._refresh_active:
                return ActorCacheRefreshResult(
                    ActorCacheOutcome.IN_PROGRESS,
                    pointer_snapshot.world_base,
                    pointer_snapshot.generation,
                    slot_count=len(self._slot_bases),
                    message="Another actor-slot refresh is already running",
                )
            self._refresh_active = True

        try:
            stopped = self._refresh_stop_outcome(cancellation, deadline)
            if stopped is not None:
                return ActorCacheRefreshResult(
                    stopped,
                    pointer_snapshot.world_base,
                    pointer_snapshot.generation,
                )

            discovery_started_at = perf_counter()
            matches = self._memory.find_u32(
                pointer_snapshot.world_base,
                maximum_address=self.config.maximum_scan_address,
                private_only=self.config.private_memory_only,
                chunk_size=self.config.discovery_chunk_bytes,
                cancellation=cancellation,
                deadline=deadline,
            )
            discovery_elapsed = max(0.0, perf_counter() - discovery_started_at)
            slots: set[int] = set()
            invalid = 0
            for world_field_address in matches:
                stopped = self._refresh_stop_outcome(cancellation, deadline)
                if stopped is not None:
                    return ActorCacheRefreshResult(
                        stopped,
                        pointer_snapshot.world_base,
                        pointer_snapshot.generation,
                    )
                base = int(world_field_address) - self._world_field_offset
                if base <= 0 or not self._is_self_valid(base):
                    invalid += 1
                    continue
                try:
                    if (
                        self._read_u32(base + self._world_field_offset)
                        != pointer_snapshot.world_base
                    ):
                        invalid += 1
                        continue
                except Exception:
                    invalid += 1
                    continue
                slots.add(base)
            if self._is_self_valid(pointer_snapshot.player_base):
                slots.add(pointer_snapshot.player_base)

            try:
                if self._native_service is not None:
                    current = self._native_service.read_pointer_snapshot()
                    current_matches = bool(
                        current.player_base == pointer_snapshot.player_base
                        and current.world_base == pointer_snapshot.world_base
                        and current.generation == pointer_snapshot.generation
                    )
                else:
                    current_matches = bool(
                        self.read_player_base() == pointer_snapshot.player_base
                        and self.read_world_base() == pointer_snapshot.world_base
                    )
            except Exception as error:
                return ActorCacheRefreshResult(
                    ActorCacheOutcome.UNAVAILABLE,
                    pointer_snapshot.world_base,
                    pointer_snapshot.generation,
                    message=f"Could not revalidate native world: {error}",
                )
            if not current_matches:
                return ActorCacheRefreshResult(
                    ActorCacheOutcome.WORLD_MISMATCH,
                    pointer_snapshot.world_base,
                    pointer_snapshot.generation,
                    message="Native player/world changed during actor discovery",
                )

            diagnostics = getattr(
                self._memory,
                "last_search_diagnostics",
                MemorySearchDiagnostics(matches=len(matches)),
            )
            ordered = tuple(sorted(slots))
            with self._lock:
                if self._closed:
                    return ActorCacheRefreshResult(
                        ActorCacheOutcome.UNAVAILABLE,
                        pointer_snapshot.world_base,
                        pointer_snapshot.generation,
                        message="Native actor provider closed during discovery",
                    )
                self._slot_bases = ordered
                self._cached_world = pointer_snapshot.world_base
                self._cached_generation = pointer_snapshot.generation
                self._last_discovery_at = now
                self.last_diagnostics = ActorPoolDiagnostics(
                    player_base=pointer_snapshot.player_base,
                    world_base=pointer_snapshot.world_base,
                    discovered_slots=len(ordered),
                    discovery_regions_considered=diagnostics.regions_considered,
                    discovery_regions_read=diagnostics.regions_read,
                    discovery_bytes_read=diagnostics.bytes_read,
                    discovery_elapsed_seconds=discovery_elapsed,
                    discovery_read_failures=diagnostics.read_failures,
                    world_pointer_matches=diagnostics.matches,
                    rejected_invalid_actor=invalid,
                )
            return ActorCacheRefreshResult(
                ActorCacheOutcome.REFRESHED,
                pointer_snapshot.world_base,
                pointer_snapshot.generation,
                slot_count=len(ordered),
            )
        except MemorySearchCancelled:
            return ActorCacheRefreshResult(
                ActorCacheOutcome.CANCELLED,
                pointer_snapshot.world_base,
                pointer_snapshot.generation,
            )
        except MemorySearchDeadline:
            return ActorCacheRefreshResult(
                ActorCacheOutcome.DEADLINE,
                pointer_snapshot.world_base,
                pointer_snapshot.generation,
            )
        except Exception as error:
            return ActorCacheRefreshResult(
                ActorCacheOutcome.UNAVAILABLE,
                pointer_snapshot.world_base,
                pointer_snapshot.generation,
                message=f"Actor-slot refresh failed: {type(error).__name__}: {error}",
            )
        finally:
            self._finish_slot_refresh()

    def read_cached_active_actors(
        self,
        pointer_snapshot: NativePointerSnapshot,
        player_pose: PlayerPose,
        *,
        allowed_species_ids: Iterable[int] | None = None,
        vision_radius_native: float | None = None,
    ) -> CachedActorReadResult:
        """Read only cached actor addresses; never discover or recover."""

        if not isinstance(pointer_snapshot, NativePointerSnapshot):
            raise TypeError("pointer_snapshot must be a NativePointerSnapshot")
        if not isinstance(player_pose, PlayerPose):
            raise TypeError("player_pose must be a PlayerPose")
        independent = self._independent_reader()
        if independent is not None:
            with self._lock:
                if self._closed:
                    return CachedActorReadResult(
                        ActorCacheOutcome.UNAVAILABLE,
                        0,
                        pointer_snapshot.generation,
                        message="Native actor provider is closed",
                    )
                if (
                    not self._slot_bases
                    or self._cached_world != 0
                    or self._cached_generation != pointer_snapshot.generation
                ):
                    return CachedActorReadResult(
                        ActorCacheOutcome.WORLD_MISMATCH,
                        0,
                        pointer_snapshot.generation,
                        message="Independent actor cache belongs to another recovery generation",
                    )
            allowed = (
                None
                if allowed_species_ids is None
                else {int(value) for value in allowed_species_ids}
            )
            radius = (
                self.config.vision_radius_native
                if vision_radius_native is None
                else float(vision_radius_native)
            )
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError("vision_radius_native must be finite and positive")
            try:
                native_snapshot = independent.snapshot(
                    allowed_species=allowed,
                    vision_radius_native=radius,
                )
            except Exception as error:
                return CachedActorReadResult(
                    ActorCacheOutcome.UNAVAILABLE,
                    0,
                    pointer_snapshot.generation,
                    message=(
                        "Independent actor read failed: "
                        f"{type(error).__name__}: {error}"
                    ),
                )
            actors = self._native_actors_from_independent(native_snapshot.monsters)
            tracked_actors = self._tracked_actors_from_independent(
                native_snapshot.actor_states,
                vision_radius_native=radius,
            )
            with self._lock:
                self.last_diagnostics = ActorPoolDiagnostics(
                    player_base=native_snapshot.player.base,
                    world_base=0,
                    discovered_slots=native_snapshot.cached_actor_slots,
                    rejected_not_present=native_snapshot.dead_or_dormant_slots,
                    rejected_dead=native_snapshot.zero_hp_monsters,
                    rejected_species=native_snapshot.other_species_slots,
                    rejected_distance=max(
                        0,
                        native_snapshot.living_monsters
                        - native_snapshot.visible_living_monsters,
                    ),
                    unreadable_cached_slots=native_snapshot.unreadable_slots,
                    runtime_promoted_slots=independent.runtime_promoted_slots,
                    pending_actor_slot_probes=independent.pending_actor_slots,
                )
            return CachedActorReadResult(
                ActorCacheOutcome.READY,
                0,
                pointer_snapshot.generation,
                actors=actors,
                tracked_actors=tracked_actors,
            )
        with self._lock:
            if self._closed:
                return CachedActorReadResult(
                    ActorCacheOutcome.UNAVAILABLE,
                    pointer_snapshot.world_base,
                    pointer_snapshot.generation,
                    message="Native actor provider is closed",
                )
            if not self._slot_bases or self._cached_world is None:
                return CachedActorReadResult(
                    ActorCacheOutcome.UNAVAILABLE,
                    pointer_snapshot.world_base,
                    pointer_snapshot.generation,
                    message="Actor-slot cache has not been refreshed",
                )
            if (
                self._cached_world != pointer_snapshot.world_base
                or self._cached_generation != pointer_snapshot.generation
            ):
                return CachedActorReadResult(
                    ActorCacheOutcome.WORLD_MISMATCH,
                    pointer_snapshot.world_base,
                    pointer_snapshot.generation,
                    message="Actor-slot cache belongs to another native world",
                )
            slot_bases = self._slot_bases

        allowed = (
            None
            if allowed_species_ids is None
            else {int(value) for value in allowed_species_ids}
        )
        radius = (
            self.config.vision_radius_native
            if vision_radius_native is None
            else float(vision_radius_native)
        )
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("vision_radius_native must be finite and positive")

        wrong_world = not_present = dead = wrong_species = too_far = unreadable = 0
        actors: list[NativeActor] = []
        tracked_actors: list[NativeActor] = []
        for base in slot_bases:
            if base == pointer_snapshot.player_base:
                continue
            try:
                (
                    actor_world,
                    species,
                    hp,
                    x,
                    y,
                    z,
                    distance,
                    active_species,
                ) = self._read_actor_fields(base, player_pose.x, player_pose.z)
            except Exception:
                unreadable += 1
                continue
            if actor_world != pointer_snapshot.world_base:
                wrong_world += 1
                continue
            if species <= 0:
                not_present += 1
                continue
            if allowed is not None and species not in allowed:
                wrong_species += 1
                continue
            if not all(math.isfinite(value) for value in (x, y, z, distance)):
                unreadable += 1
                continue
            limit = self.config.maximum_absolute_coordinate
            if any(abs(value) > limit for value in (x, y, z)):
                unreadable += 1
                continue
            if distance > radius:
                too_far += 1
                continue
            tracked = NativeActor(
                base_address=base,
                species_id=species,
                hp=hp,
                x=x,
                y=y,
                z=z,
                distance_native=distance,
                active_species_id=active_species,
            )
            if hp >= 0:
                tracked_actors.append(tracked)
            if active_species != species:
                not_present += 1
                continue
            if hp <= 0:
                dead += 1
                continue
            actors.append(tracked)

        actors.sort(key=lambda actor: (actor.distance_native, actor.base_address))
        tracked_actors.sort(
            key=lambda actor: (actor.distance_native, actor.base_address)
        )
        with self._lock:
            previous = self.last_diagnostics
            self.last_diagnostics = ActorPoolDiagnostics(
                player_base=pointer_snapshot.player_base,
                world_base=pointer_snapshot.world_base,
                discovered_slots=len(slot_bases),
                discovery_regions_considered=(
                    0 if previous is None else previous.discovery_regions_considered
                ),
                discovery_regions_read=(
                    0 if previous is None else previous.discovery_regions_read
                ),
                discovery_bytes_read=(
                    0 if previous is None else previous.discovery_bytes_read
                ),
                discovery_elapsed_seconds=(
                    0.0 if previous is None else previous.discovery_elapsed_seconds
                ),
                discovery_read_failures=(
                    0 if previous is None else previous.discovery_read_failures
                ),
                world_pointer_matches=(
                    0 if previous is None else previous.world_pointer_matches
                ),
                rejected_invalid_actor=(
                    0 if previous is None else previous.rejected_invalid_actor
                ),
                rejected_wrong_world=wrong_world,
                rejected_not_present=not_present,
                rejected_dead=dead,
                rejected_species=wrong_species,
                rejected_distance=too_far,
                unreadable_cached_slots=unreadable,
            )
        return CachedActorReadResult(
            ActorCacheOutcome.READY,
            pointer_snapshot.world_base,
            pointer_snapshot.generation,
            actors=tuple(actors),
            tracked_actors=tuple(tracked_actors),
        )

    def _read_actor_fields(
        self,
        base: int,
        player_x: float,
        player_z: float,
    ) -> tuple[int, int, int, float, float, float, float, int]:
        world = self._read_u32(base + self._world_field_offset)
        species = self._read_i32(base + self._species_field_offset)
        active_species = self._read_i32(base + self._active_species_field_offset)
        hp = self._read_i32(base + self._hp_field_offset)
        x = self._read_float(base + self._x_field_offset)
        y = self._read_float(base + self._y_field_offset)
        z = self._read_float(base + self._z_field_offset)
        distance = math.hypot(x - player_x, z - player_z)
        return world, species, hp, x, y, z, distance, active_species

    def read_active_actors(
        self,
        *,
        allowed_species_ids: Iterable[int] | None = None,
        vision_radius_native: float | None = None,
        force_rediscovery: bool = False,
        pointer_snapshot: NativePointerSnapshot | None = None,
    ) -> list[NativeActor]:
        independent = self._independent_reader()
        if independent is not None:
            allowed = (
                None
                if allowed_species_ids is None
                else {int(value) for value in allowed_species_ids}
            )
            radius = (
                self.config.vision_radius_native
                if vision_radius_native is None
                else float(vision_radius_native)
            )
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError("vision_radius_native must be finite and positive")
            try:
                snapshot = independent.snapshot(
                    allowed_species=allowed,
                    vision_radius_native=radius,
                )
            except Exception as error:
                raise NativeMonsterReadError(
                    f"Independent actor read failed: {type(error).__name__}: {error}"
                ) from error
            actors = self._native_actors_from_independent(snapshot.monsters)
            with self._lock:
                self._slot_bases = independent.actor_slots
                self._cached_world = 0
                if pointer_snapshot is not None:
                    self._cached_generation = pointer_snapshot.generation
                self.last_diagnostics = ActorPoolDiagnostics(
                    player_base=snapshot.player.base,
                    world_base=0,
                    discovered_slots=snapshot.cached_actor_slots,
                    rejected_not_present=snapshot.dead_or_dormant_slots,
                    rejected_dead=snapshot.zero_hp_monsters,
                    rejected_species=snapshot.other_species_slots,
                    rejected_distance=max(
                        0,
                        snapshot.living_monsters - snapshot.visible_living_monsters,
                    ),
                    unreadable_cached_slots=snapshot.unreadable_slots,
                    runtime_promoted_slots=independent.runtime_promoted_slots,
                    pending_actor_slot_probes=independent.pending_actor_slots,
                )
            return list(actors)
        with self._lock:
            shared = self._shared_pointer_snapshot(pointer_snapshot)
            player = self.read_player_base(pointer_snapshot=shared)
            world = self.read_world_base(pointer_snapshot=shared)
            player_x = self._read_float(player + self._x_field_offset)
            player_z = self._read_float(player + self._z_field_offset)
            if not math.isfinite(player_x) or not math.isfinite(player_z):
                raise NativeMonsterReadError("Player coordinates are not finite")

            allowed = (
                None
                if allowed_species_ids is None
                else {int(value) for value in allowed_species_ids}
            )
            radius = (
                self.config.vision_radius_native
                if vision_radius_native is None
                else float(vision_radius_native)
            )
            if radius <= 0.0:
                raise ValueError("vision_radius_native must be positive")

            wrong_world = 0
            not_present = 0
            dead = 0
            wrong_species = 0
            too_far = 0
            unreadable = 0
            actors: list[NativeActor] = []

            for base in self.discover_slots(
                force=force_rediscovery,
                pointer_snapshot=shared,
            ):
                if base == player:
                    continue
                try:
                    (
                        actor_world,
                        species,
                        hp,
                        x,
                        y,
                        z,
                        distance,
                        active_species,
                    ) = self._read_actor_fields(base, player_x, player_z)
                except Exception:
                    unreadable += 1
                    continue

                if actor_world != world:
                    wrong_world += 1
                    continue
                # +0x1DBC duplicates +0x174 only while the actor is actually
                # instantiated in the scene. Dormant reusable slots retain HP
                # and their ordinary species ID but clear this duplicate.
                if species <= 0 or active_species != species:
                    not_present += 1
                    continue
                if hp <= 0:
                    dead += 1
                    continue
                if allowed is not None and species not in allowed:
                    wrong_species += 1
                    continue
                if not all(math.isfinite(value) for value in (x, y, z, distance)):
                    unreadable += 1
                    continue
                limit = self.config.maximum_absolute_coordinate
                if any(abs(value) > limit for value in (x, y, z)):
                    unreadable += 1
                    continue
                if distance > radius:
                    too_far += 1
                    continue

                actors.append(
                    NativeActor(
                        base_address=base,
                        species_id=species,
                        hp=hp,
                        x=x,
                        y=y,
                        z=z,
                        distance_native=distance,
                        active_species_id=active_species,
                    )
                )

            actors.sort(key=lambda actor: (actor.distance_native, actor.base_address))
            previous = self.last_diagnostics
            self.last_diagnostics = ActorPoolDiagnostics(
                player_base=player,
                world_base=world,
                discovered_slots=len(self._slot_bases),
                discovery_regions_considered=(
                    0 if previous is None else previous.discovery_regions_considered
                ),
                discovery_regions_read=(
                    0 if previous is None else previous.discovery_regions_read
                ),
                discovery_bytes_read=(
                    0 if previous is None else previous.discovery_bytes_read
                ),
                discovery_elapsed_seconds=(
                    0.0 if previous is None else previous.discovery_elapsed_seconds
                ),
                discovery_read_failures=(
                    0 if previous is None else previous.discovery_read_failures
                ),
                world_pointer_matches=(
                    0 if previous is None else previous.world_pointer_matches
                ),
                rejected_invalid_actor=(
                    0 if previous is None else previous.rejected_invalid_actor
                ),
                rejected_wrong_world=wrong_world,
                rejected_not_present=not_present,
                rejected_dead=dead,
                rejected_species=wrong_species,
                rejected_distance=too_far,
                unreadable_cached_slots=unreadable,
            )
            return actors

    def capture_selected_actor(
        self,
        *,
        pointer_snapshot: NativePointerSnapshot | None = None,
    ) -> NativeActor:
        if self._independent_reader() is not None:
            raise NativeMonsterReadError(
                "Selected-actor capture is unavailable in independent pointer mode; "
                "use the GUI monster registry and selected species for training."
            )
        with self._lock:
            shared = self._shared_pointer_snapshot(pointer_snapshot)
            player = self.read_player_base(pointer_snapshot=shared)
            world = self.read_world_base(pointer_snapshot=shared)
            selected = self._read_u32(world + self.config.selected_actor_offset)
            if selected <= 0:
                raise NativeMonsterReadError("No actor is currently selected")
            if selected == player:
                raise NativeMonsterReadError("The selected actor is the local player")
            if not self._is_self_valid(selected):
                raise NativeMonsterReadError(
                    f"Selected actor 0x{selected:X} failed the self-pointer check"
                )

            player_x = self._read_float(player + self._x_field_offset)
            player_z = self._read_float(player + self._z_field_offset)
            (
                actor_world,
                species,
                hp,
                x,
                y,
                z,
                distance,
                active_species,
            ) = self._read_actor_fields(selected, player_x, player_z)

            if actor_world != world:
                raise NativeMonsterReadError(
                    "The selected actor does not belong to the current world"
                )
            if species <= 0 or active_species != species:
                raise NativeMonsterReadError(
                    "The selected actor is not currently instantiated in the scene"
                )
            if hp <= 0:
                raise NativeMonsterReadError("The selected actor is dead")
            if not all(math.isfinite(value) for value in (x, y, z, distance)):
                raise NativeMonsterReadError(
                    "The selected actor has invalid native coordinates"
                )

            return NativeActor(
                base_address=selected,
                species_id=species,
                hp=hp,
                x=x,
                y=y,
                z=z,
                distance_native=distance,
                active_species_id=active_species,
            )

    def close(self) -> None:
        close_memory = False
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if (
                self._owns_memory
                and not self._refresh_active
                and not self._resources_closed
            ):
                self._resources_closed = True
                close_memory = True
        if close_memory:
            self._memory.close()
