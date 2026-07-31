from __future__ import annotations

import math
import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import RLock
from time import monotonic, perf_counter
from typing import Protocol

from .MonsterConfig import NativeMonsterConfig
from .native_process_service import (
    NativePointerSnapshot,
    NativeProcessService,
)
from .NativePointerRecovery import (
    PlayerPointerRecovery,
)
from .PositionProvider import PositionProviderError
from .Win32ProcessMemory import (
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
        self._player_pointer_address = (
            self._module_base + config.player_pointer_offset
        )
        self._world_pointer_address = self._module_base + config.world_pointer_offset
        self._slot_bases: tuple[int, ...] = ()
        self._cached_world: int | None = None
        self._last_discovery_at = -math.inf
        self._lock = RLock()
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
            native_service.memory,  # type: ignore[arg-type]
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
        return self._slot_bases

    @property
    def last_pointer_recovery(self) -> PlayerPointerRecovery | None:
        if self._native_service is not None:
            result = self._native_service.last_recovery_result
            return None if result is None else result.recovery
        return self._last_pointer_recovery

    def _read_u32(self, address: int) -> int:
        return int(struct.unpack("<I", self._memory.read(address, 4))[0])

    def _read_i32(self, address: int) -> int:
        return int(struct.unpack("<i", self._memory.read(address, 4))[0])

    def _read_float(self, address: int) -> float:
        return float(struct.unpack("<f", self._memory.read(address, 4))[0])

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
        base = self._read_u32(self._player_pointer_address)
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
        base = self._read_u32(self._world_pointer_address)
        if base <= 0:
            raise NativeMonsterReadError(
                "Current-world pointer is null; explicit pointer recovery is "
                "required after login/map transition checks complete"
            )
        try:
            player = self._read_u32(self._player_pointer_address)
            if player <= 0:
                raise NativeMonsterReadError(
                    "Local-player pointer is null while validating the current "
                    "world pointer"
                )
            player_world = self._read_u32(player + self.config.world_offset)
        except NativeMonsterReadError:
            raise
        except Exception as error:
            raise NativeMonsterReadError(
                "Current-world pointer could not be validated against the "
                "local player"
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
            return self._read_u32(base + self.config.self_pointer_offset) == base
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
                base = int(world_field_address) - self.config.world_offset
                if base <= 0 or not self._is_self_valid(base):
                    invalid += 1
                    continue
                try:
                    if self._read_u32(base + self.config.world_offset) != world:
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

    def _read_actor_fields(
        self,
        base: int,
        player_x: float,
        player_z: float,
    ) -> tuple[int, int, int, float, float, float, float, int]:
        config = self.config
        world = self._read_u32(base + config.world_offset)
        species = self._read_i32(base + config.species_offset)
        active_species = self._read_i32(base + config.active_species_offset)
        hp = self._read_i32(base + config.hp_offset)
        x = self._read_float(base + config.x_offset)
        y = self._read_float(base + config.y_offset)
        z = self._read_float(base + config.z_offset)
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
        with self._lock:
            shared = self._shared_pointer_snapshot(pointer_snapshot)
            player = self.read_player_base(pointer_snapshot=shared)
            world = self.read_world_base(pointer_snapshot=shared)
            player_x = self._read_float(player + self.config.x_offset)
            player_z = self._read_float(player + self.config.z_offset)
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

            player_x = self._read_float(player + self.config.x_offset)
            player_z = self._read_float(player + self.config.z_offset)
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
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_memory:
                self._memory.close()
