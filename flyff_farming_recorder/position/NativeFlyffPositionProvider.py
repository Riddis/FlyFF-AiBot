from __future__ import annotations

import math
import struct
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from statistics import median
from time import monotonic
from typing import cast

from .native_process_service import (
    NativePointerSnapshot,
    NativeProcessService,
)
from .NativePointerRecovery import (
    PlayerPointerRecovery,
)
from .PositionConfig import NativePositionConfig
from .PositionProvider import PlayerPose, PositionProviderError
from .Win32ProcessMemory import Win32MemoryBackend, Win32ProcessMemory


class InvalidPlayerPoseError(PositionProviderError):
    """A configured transform returned values that cannot be a player pose."""


class PoseConsensusError(PositionProviderError):
    """The configured transform copies could not produce a valid consensus."""


class PointerResolutionError(PositionProviderError):
    """A configured module pointer was null or could not be read."""


@dataclass(frozen=True, slots=True)
class PositionReadDiagnostics:
    """Details from the latest native pose read, useful for shadow telemetry."""

    resolved_addresses: tuple[int, ...]
    successful_addresses: tuple[int, ...]
    consensus_addresses: tuple[int, ...]
    failed_addresses: tuple[int, ...]
    maximum_consensus_delta: float | None
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CandidatePose:
    address: int
    x: float
    y: float
    z: float
    heading_degrees: float | None


class NativeFlyffPositionProvider:
    """Resolve the local-player transform and return a validated native pose."""

    FLOAT_SIZE = 4

    def __init__(
        self,
        memory: Win32ProcessMemory,
        config: NativePositionConfig,
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
        self._module_base: int | None = None
        self._pointer_storage_address: int | None = None
        self._resolved_addresses = self._resolve_addresses()
        self.last_diagnostics: PositionReadDiagnostics | None = None
        self._last_pointer_recovery: PlayerPointerRecovery | None = None


    @classmethod
    def from_window_handle(
        cls,
        window_handle: int,
        config: NativePositionConfig,
        *,
        backend: Win32MemoryBackend | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> NativeFlyffPositionProvider:
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
    def from_native_service(
        cls,
        native_service: NativeProcessService,
        config: NativePositionConfig,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> NativeFlyffPositionProvider:
        return cls(
            cast(Win32ProcessMemory, cast(object, native_service.memory)),
            config,
            clock=clock,
            native_service=native_service,
            owns_memory=False,
        )

    @property
    def resolved_addresses(self) -> tuple[int, ...]:
        return self._resolved_addresses

    @property
    def module_base(self) -> int | None:
        return self._module_base

    @property
    def pointer_storage_address(self) -> int | None:
        if (
            self._native_service is not None
            and self.config.resolver == "module_pointer"
        ):
            return self._native_service.player_pointer_address
        return self._pointer_storage_address

    @property
    def last_pointer_recovery(self) -> PlayerPointerRecovery | None:
        if self._native_service is not None:
            result = self._native_service.last_recovery_result
            return None if result is None else result.recovery
        return self._last_pointer_recovery

    def _resolve_addresses(self) -> tuple[int, ...]:
        if self.config.resolver == "direct_address":
            if self.config.transform_address is None:
                raise ValueError("transform_address is required")
            return (int(self.config.transform_address),)

        module_name = self.config.module_name
        if module_name is None:
            raise ValueError("module_name is required")
        if (
            self._native_service is not None
            and self.config.resolver == "module_pointer"
        ):
            self._module_base = self._native_service.module_base
        else:
            self._module_base = self._memory.module_base(module_name)

        if self.config.resolver == "module_pointer":
            if self.config.pointer_offset is None:
                raise ValueError("pointer_offset is required")
            if self._native_service is not None:
                self._pointer_storage_address = (
                    self._native_service.player_pointer_address
                )
            else:
                self._pointer_storage_address = (
                    self._module_base + self.config.pointer_offset
                )
            # The actual transform base is read afresh on every sample.  The
            # local-player object may be recreated after login, respawn, or a
            # map transition without changing the module-relative pointer slot.
            return ()

        return tuple(
            self._module_base + offset for offset in self.config.transform_offsets
        )

    def _read_pointer_target(
        self,
        pointer_snapshot: NativePointerSnapshot | None = None,
    ) -> int:
        if self._native_service is not None:
            try:
                snapshot = (
                    pointer_snapshot
                    if pointer_snapshot is not None
                    else self._native_service.read_pointer_snapshot()
                )
            except Exception as error:
                raise PointerResolutionError(str(error)) from error
            self._pointer_storage_address = snapshot.player_pointer_address
            return snapshot.player_base

        pointer_storage = self._pointer_storage_address
        if pointer_storage is None:
            raise PointerResolutionError("module pointer storage is not configured")
        try:
            raw = self._memory.read(pointer_storage, 4)
            target = int(struct.unpack("<I", raw)[0])
            for offset in self.config.pointer_chain_offsets:
                raw = self._memory.read(target + offset, 4)
                target = int(struct.unpack("<I", raw)[0])
        except Exception as error:
            raise PointerResolutionError(
                f"Could not resolve 32-bit player pointer at 0x{pointer_storage:X}: "
                f"{type(error).__name__}: {error}"
            ) from error
        if target == 0:
            raise PointerResolutionError(
                f"Player pointer at 0x{pointer_storage:X} is null; "
                "explicit pointer recovery is required after login/map "
                "transition checks complete"
            )
        return target

    def _addresses_for_read(
        self,
        pointer_snapshot: NativePointerSnapshot | None = None,
    ) -> tuple[int, ...]:
        if self.config.resolver != "module_pointer":
            return self._resolved_addresses
        target = self._read_pointer_target(pointer_snapshot)
        self._resolved_addresses = (target,)
        return self._resolved_addresses

    def _read_candidate(self, base_address: int) -> _CandidatePose:
        x_offset = (
            self.config.x_offset
            if self._native_service is None
            else self._native_service.x_offset
        )
        y_offset = (
            self.config.y_offset
            if self._native_service is None
            else self._native_service.y_offset
        )
        z_offset = (
            self.config.z_offset
            if self._native_service is None
            else self._native_service.z_offset
        )
        offsets = [x_offset, y_offset, z_offset]
        if self.config.heading_offset is not None:
            offsets.append(self.config.heading_offset)
        block_start = min(offsets)
        block_size = max(offsets) - block_start + self.FLOAT_SIZE
        data = self._memory.read(
            base_address + block_start,
            block_size,
        )

        def read_float(offset: int) -> float:
            return float(struct.unpack_from("<f", data, offset - block_start)[0])

        x = read_float(x_offset)
        y = read_float(y_offset)
        z = read_float(z_offset)

        heading: float | None = None
        if self.config.heading_offset is not None:
            heading = read_float(self.config.heading_offset)
            if self.config.heading_unit == "radians":
                heading = math.degrees(heading)
            heading %= 360.0

        values = [x, y, z]
        if heading is not None:
            values.append(heading)
        if not all(math.isfinite(value) for value in values):
            raise InvalidPlayerPoseError(
                f"0x{base_address:X} returned non-finite values: {values!r}"
            )

        limit = self.config.maximum_absolute_coordinate
        if any(abs(value) > limit for value in (x, y, z)):
            raise InvalidPlayerPoseError(
                f"0x{base_address:X} returned coordinates outside the configured "
                f"+/-{limit:g} limit: {(x, y, z)!r}"
            )

        return _CandidatePose(
            address=base_address,
            x=x,
            y=y,
            z=z,
            heading_degrees=heading,
        )

    @staticmethod
    def _coordinate_delta(left: _CandidatePose, right: _CandidatePose) -> float:
        return max(
            abs(left.x - right.x),
            abs(left.y - right.y),
            abs(left.z - right.z),
        )

    @staticmethod
    def _mean_heading(candidates: list[_CandidatePose]) -> float | None:
        headings = [
            candidate.heading_degrees
            for candidate in candidates
            if candidate.heading_degrees is not None
        ]
        if len(headings) != len(candidates) or not headings:
            return None
        sin_total = sum(math.sin(math.radians(value)) for value in headings)
        cos_total = sum(math.cos(math.radians(value)) for value in headings)
        if abs(sin_total) < 1e-12 and abs(cos_total) < 1e-12:
            return float(headings[0]) % 360.0
        return math.degrees(math.atan2(sin_total, cos_total)) % 360.0

    def _select_consensus(
        self,
        candidates: list[_CandidatePose],
    ) -> tuple[list[_CandidatePose], float]:
        tolerance = self.config.consensus_tolerance
        best_cluster: list[_CandidatePose] = []
        best_delta = math.inf

        for size in range(len(candidates), 0, -1):
            for candidate_group in combinations(candidates, size):
                group = list(candidate_group)
                group_delta = max(
                    (
                        self._coordinate_delta(left, right)
                        for index, left in enumerate(group)
                        for right in group[index + 1 :]
                    ),
                    default=0.0,
                )
                if group_delta <= tolerance and group_delta < best_delta:
                    best_cluster = group
                    best_delta = group_delta
            if best_cluster:
                break

        required = min(
            self.config.minimum_consensus_sources,
            len(self._resolved_addresses),
        )
        if len(best_cluster) < required:
            values = ", ".join(
                f"0x{candidate.address:X}=({candidate.x:.6f},"
                f"{candidate.y:.6f},{candidate.z:.6f})"
                for candidate in candidates
            )
            raise PoseConsensusError(
                f"Native position consensus needs {required} source(s) within "
                f"{tolerance:g}, but the largest cluster had {len(best_cluster)}. "
                f"Valid reads: {values or 'none'}"
            )
        return best_cluster, best_delta

    def read_pose(
        self,
        *,
        pointer_snapshot: NativePointerSnapshot | None = None,
    ) -> PlayerPose:
        candidates: list[_CandidatePose] = []
        failed_addresses: list[int] = []
        errors: list[str] = []

        try:
            addresses = self._addresses_for_read(pointer_snapshot)
        except Exception as error:
            failed = (
                ()
                if self._pointer_storage_address is None
                else (self._pointer_storage_address,)
            )
            self.last_diagnostics = PositionReadDiagnostics(
                resolved_addresses=(),
                successful_addresses=(),
                consensus_addresses=(),
                failed_addresses=failed,
                maximum_consensus_delta=None,
                errors=(f"{type(error).__name__}: {error}",),
            )
            raise

        for address in addresses:
            try:
                candidates.append(self._read_candidate(address))
            except Exception as error:  # noqa: BLE001 - consensus tolerates one copy.
                failed_addresses.append(address)
                errors.append(f"0x{address:X}: {type(error).__name__}: {error}")
                if len(addresses) == 1:
                    self.last_diagnostics = PositionReadDiagnostics(
                        resolved_addresses=addresses,
                        successful_addresses=(),
                        consensus_addresses=(),
                        failed_addresses=tuple(failed_addresses),
                        maximum_consensus_delta=None,
                        errors=tuple(errors),
                    )
                    raise

        try:
            consensus, maximum_delta = self._select_consensus(candidates)
        except PoseConsensusError as error:
            self.last_diagnostics = PositionReadDiagnostics(
                resolved_addresses=addresses,
                successful_addresses=tuple(item.address for item in candidates),
                consensus_addresses=(),
                failed_addresses=tuple(failed_addresses),
                maximum_consensus_delta=None,
                errors=tuple([*errors, str(error)]),
            )
            raise

        consensus_addresses = tuple(item.address for item in consensus)
        self.last_diagnostics = PositionReadDiagnostics(
            resolved_addresses=addresses,
            successful_addresses=tuple(item.address for item in candidates),
            consensus_addresses=consensus_addresses,
            failed_addresses=tuple(failed_addresses),
            maximum_consensus_delta=maximum_delta,
            errors=tuple(errors),
        )

        return PlayerPose(
            x=float(median(item.x for item in consensus)),
            y=float(median(item.y for item in consensus)),
            z=float(median(item.z for item in consensus)),
            heading_degrees=self._mean_heading(consensus),
            timestamp=float(self._clock()),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_memory:
            self._memory.close()
