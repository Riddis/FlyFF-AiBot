from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from time import monotonic
from typing import Any, cast

from .AnchoredPointerDiscovery import PointerRecoveryHints
from .native_process_service import (
    NativePointerSnapshot,
    NativePointerSnapshotError,
    NativeRecoveryOutcome,
    NativeRecoveryResult,
)
from .NativeFlyffMonsterProvider import ActorPoolDiagnostics
from .NativeFlyffPositionProvider import PositionReadDiagnostics
from .NativePointerRecovery import PointerRecoveryMetrics, PointerRecoveryProgress


class NativeHealthStatus(str, Enum):
    """Health of the shared native attachment without initiating discovery."""

    DETACHED = "detached"
    CLOSED = "closed"
    HEALTHY = "healthy"
    POINTER_UNAVAILABLE = "pointer_unavailable"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class NativeProviderHealth:
    """Cached provider facts; no provider read or actor discovery is performed."""

    position_attached: bool
    position_resolver: str | None
    position_last_read: PositionReadDiagnostics | None
    monster_attached: bool
    cached_actor_slots: int
    monster_last_read: ActorPoolDiagnostics | None


@dataclass(frozen=True, slots=True)
class NativeRuntimeFacts:
    """Cached application facts plus one bounded, read-only focus probe."""

    selected_map_name: str | None
    map_overlay_loaded: bool
    native_player_position: tuple[float, float, float] | None
    map_coordinate_cell: tuple[float, float] | None
    coordinate_error: str | None
    ocr_enabled: bool
    ocr_anchor_cached: bool | None
    ocr_full_scan_count: int | None
    keyboard_attached: bool
    target_focused: bool | None
    focus_error: str | None


@dataclass(frozen=True, slots=True)
class NativeHealthSnapshot:
    """One bounded health sample from the shared native process service."""

    status: NativeHealthStatus
    captured_at: float
    process_id: int | None
    module_base: int | None
    player_pointer_address: int | None
    world_pointer_address: int | None
    player_base: int | None
    world_base: int | None
    pointer_generation: int | None
    error: str | None
    providers: NativeProviderHealth
    runtime: NativeRuntimeFacts
    module_name: str | None = None
    module_path: str | None = None
    module_size: int | None = None
    pointer_width_bytes: int | None = None
    configured_player_pointer_offset: int | None = None
    configured_world_pointer_offset: int | None = None
    player_pointer_chain_offsets: tuple[int, ...] = ()
    world_pointer_chain_offsets: tuple[int, ...] = ()
    world_field_offset: int | None = None
    self_pointer_offset: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly recursive dataclass representation."""

        return asdict(self)


class NativeDiagnosticOutcome(str, Enum):
    HEALTH_ONLY = "health_only"
    RECOVERY_UNAVAILABLE = "recovery_unavailable"
    RECOVERY_SUCCEEDED = "recovery_succeeded"
    RECOVERY_NOT_APPLIED = "recovery_not_applied"
    RECOVERY_NOT_FOUND = "recovery_not_found"
    RECOVERY_DEADLINE = "recovery_deadline"
    RECOVERY_CANCELLED = "recovery_cancelled"
    RECOVERY_HINTS_REQUIRED = "recovery_hints_required"
    RECOVERY_MOVEMENT_REQUIRED = "recovery_movement_required"
    RECOVERY_MOVEMENT_NOT_OBSERVED = "recovery_movement_not_observed"
    RECOVERY_ANCHOR_INCONCLUSIVE = "recovery_anchor_inconclusive"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class NativeDiagnosticProgress:
    """One health/recovery status suitable for a runtime status channel."""

    phase: str
    message: str
    metrics: PointerRecoveryMetrics | None = None


NativeDiagnosticStatusCallback = Callable[[NativeDiagnosticProgress], None]


@dataclass(frozen=True, slots=True)
class NativeDiagnosticReport:
    """Typed result of a managed health or explicit recovery request."""

    outcome: NativeDiagnosticOutcome
    recovery_requested: bool
    persistence_requested: bool
    before: NativeHealthSnapshot
    after: NativeHealthSnapshot
    recovery: NativeRecoveryResult | None
    progress_updates: int
    last_progress: NativeDiagnosticProgress | None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly recursive dataclass representation."""

        return asdict(self)


def collect_native_health(
    bot: Any,
    *,
    clock: Callable[[], float] = monotonic,
) -> NativeHealthSnapshot:
    """Collect bounded attachment health without scanning or recovering.

    The only process-memory operation is
    ``NativeProcessService.read_pointer_snapshot``: a fixed coherent pointer
    sample. Provider diagnostics, actor-slot counts, map, and OCR values come
    from already-cached application state. Focus is sampled with the keyboard
    service's constant-time, read-only foreground-window query.
    """

    providers = _provider_health(bot)
    service = getattr(bot, "native_process_service", None)
    captured_at = float(clock())
    if service is None:
        return NativeHealthSnapshot(
            status=NativeHealthStatus.DETACHED,
            captured_at=captured_at,
            process_id=None,
            module_base=None,
            player_pointer_address=None,
            world_pointer_address=None,
            player_base=None,
            world_base=None,
            pointer_generation=None,
            error=None,
            providers=providers,
            runtime=_runtime_facts(bot),
        )

    process_id = _optional_int(getattr(getattr(service, "memory", None), "pid", None))
    module_base = _optional_int(getattr(service, "module_base", None))
    module_name = getattr(service, "module_name", None)
    module_path = getattr(service, "module_path", None)
    module_size = _optional_int(getattr(service, "module_size", None))
    pointer_width_bytes = _optional_int(
        getattr(service, "pointer_width_bytes", None)
    )
    configured_player_offset = _optional_int(
        getattr(service, "configured_player_pointer_offset", None)
    )
    configured_world_offset = _optional_int(
        getattr(service, "configured_world_pointer_offset", None)
    )
    player_chain = tuple(getattr(service, "player_pointer_chain_offsets", ()))
    world_chain = tuple(getattr(service, "world_pointer_chain_offsets", ()))
    world_field_offset = _optional_int(getattr(service, "world_field_offset", None))
    self_pointer_offset = _optional_int(getattr(service, "self_pointer_offset", None))
    player_pointer_address = _optional_int(
        getattr(service, "player_pointer_address", None)
    )
    world_pointer_address = _optional_int(
        getattr(service, "world_pointer_address", None)
    )
    if bool(getattr(service, "is_closed", False)):
        return NativeHealthSnapshot(
            status=NativeHealthStatus.CLOSED,
            captured_at=captured_at,
            process_id=process_id,
            module_base=module_base,
            player_pointer_address=player_pointer_address,
            world_pointer_address=world_pointer_address,
            player_base=None,
            world_base=None,
            pointer_generation=None,
            error="Native process attachment is closed",
            providers=providers,
            runtime=_runtime_facts(bot),
            module_name=None if module_name is None else str(module_name),
            module_path=None if module_path is None else str(module_path),
            module_size=module_size,
            pointer_width_bytes=pointer_width_bytes,
            configured_player_pointer_offset=configured_player_offset,
            configured_world_pointer_offset=configured_world_offset,
            player_pointer_chain_offsets=player_chain,
            world_pointer_chain_offsets=world_chain,
            world_field_offset=world_field_offset,
            self_pointer_offset=self_pointer_offset,
        )

    pointer: NativePointerSnapshot | None = None
    try:
        pointer = service.read_pointer_snapshot()
    except NativePointerSnapshotError as error:
        status = NativeHealthStatus.POINTER_UNAVAILABLE
        error_text = str(error)
        player_base = None
        world_base = None
        generation = None
    except Exception as error:  # noqa: BLE001 - diagnostic boundary is typed.
        status = NativeHealthStatus.ERROR
        error_text = f"{type(error).__name__}: {error}"
        player_base = None
        world_base = None
        generation = None
    else:
        resolved_pointer = cast(NativePointerSnapshot, pointer)
        status = NativeHealthStatus.HEALTHY
        error_text = None
        player_base = int(resolved_pointer.player_base)
        world_base = int(resolved_pointer.world_base)
        generation = int(resolved_pointer.generation)
        player_pointer_address = int(resolved_pointer.player_pointer_address)
        world_pointer_address = int(resolved_pointer.world_pointer_address)

    runtime = _runtime_facts(
        bot,
        pointer_snapshot=pointer,
    )

    return NativeHealthSnapshot(
        status=status,
        captured_at=captured_at,
        process_id=process_id,
        module_base=module_base,
        player_pointer_address=player_pointer_address,
        world_pointer_address=world_pointer_address,
        player_base=player_base,
        world_base=world_base,
        pointer_generation=generation,
        error=error_text,
        providers=providers,
        runtime=runtime,
        module_name=None if module_name is None else str(module_name),
        module_path=None if module_path is None else str(module_path),
        module_size=module_size,
        pointer_width_bytes=pointer_width_bytes,
        configured_player_pointer_offset=configured_player_offset,
        configured_world_pointer_offset=configured_world_offset,
        player_pointer_chain_offsets=player_chain,
        world_pointer_chain_offsets=world_chain,
        world_field_offset=world_field_offset,
        self_pointer_offset=self_pointer_offset,
    )


def run_native_diagnostic(
    bot: Any,
    *,
    recover: bool,
    persist: bool = False,
    recovery_hints: PointerRecoveryHints | None = None,
    cancellation: object | None,
    deadline: float | None,
    timeout_seconds: float,
    status_callback: NativeDiagnosticStatusCallback | None = None,
    clock: Callable[[], float] = monotonic,
) -> NativeDiagnosticReport:
    """Run health-only diagnostics or one explicitly requested recovery.

    Recovery is synchronous here by design. The runtime controller must invoke
    this function only inside its managed DIAGNOSTIC worker. Persistence is
    permitted only when the caller explicitly requests recovery and persistence.
    """

    timeout = float(timeout_seconds)
    if not isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_seconds must be a finite positive value")
    bounded_deadline = float(clock()) + timeout
    if deadline is not None:
        candidate = float(deadline)
        if not isfinite(candidate):
            raise ValueError("deadline must be finite")
        bounded_deadline = min(bounded_deadline, candidate)

    updates = 0
    last_progress: NativeDiagnosticProgress | None = None

    def emit(progress: NativeDiagnosticProgress) -> None:
        nonlocal updates, last_progress
        updates += 1
        last_progress = progress
        if status_callback is not None:
            status_callback(progress)

    before = collect_native_health(bot, clock=clock)
    emit(
        NativeDiagnosticProgress(
            phase="health",
            message=f"Native health: {before.status.value}.",
        )
    )
    if not recover:
        return NativeDiagnosticReport(
            outcome=NativeDiagnosticOutcome.HEALTH_ONLY,
            recovery_requested=False,
            persistence_requested=False,
            before=before,
            after=before,
            recovery=None,
            progress_updates=updates,
            last_progress=last_progress,
        )

    service = getattr(bot, "native_process_service", None)
    if service is None or bool(getattr(service, "is_closed", False)):
        emit(
            NativeDiagnosticProgress(
                phase="unavailable",
                message="Explicit native recovery requires an open attachment.",
            )
        )
        return NativeDiagnosticReport(
            outcome=NativeDiagnosticOutcome.RECOVERY_UNAVAILABLE,
            recovery_requested=True,
            persistence_requested=bool(persist),
            before=before,
            after=before,
            recovery=None,
            progress_updates=updates,
            last_progress=last_progress,
            error="Native process attachment is unavailable",
        )

    def pointer_status(progress: PointerRecoveryProgress) -> None:
        emit(
            NativeDiagnosticProgress(
                phase=progress.phase,
                message=progress.message,
                metrics=progress.metrics,
            )
        )

    try:
        recovery = service.recover_pointers(
            persist=bool(persist),
            cancellation=cancellation,
            deadline=bounded_deadline,
            timeout_seconds=timeout,
            status_callback=pointer_status,
            hints=recovery_hints,
        )
    except Exception as error:  # noqa: BLE001 - return typed diagnostic output.
        after = collect_native_health(bot, clock=clock)
        error_text = f"{type(error).__name__}: {error}"
        emit(
            NativeDiagnosticProgress(
                phase="error",
                message=f"Explicit native recovery failed: {error_text}",
            )
        )
        return NativeDiagnosticReport(
            outcome=NativeDiagnosticOutcome.ERROR,
            recovery_requested=True,
            persistence_requested=bool(persist),
            before=before,
            after=after,
            recovery=None,
            progress_updates=updates,
            last_progress=last_progress,
            error=error_text,
        )

    after = collect_native_health(bot, clock=clock)
    outcome = _diagnostic_outcome(recovery)
    return NativeDiagnosticReport(
        outcome=outcome,
        recovery_requested=True,
        persistence_requested=bool(persist),
        before=before,
        after=after,
        recovery=recovery,
        progress_updates=updates,
        last_progress=last_progress,
    )


def _diagnostic_outcome(result: NativeRecoveryResult) -> NativeDiagnosticOutcome:
    if result.succeeded:
        if result.applied:
            return NativeDiagnosticOutcome.RECOVERY_SUCCEEDED
        return NativeDiagnosticOutcome.RECOVERY_NOT_APPLIED
    if result.outcome is NativeRecoveryOutcome.CANCELLED:
        return NativeDiagnosticOutcome.RECOVERY_CANCELLED
    if result.outcome is NativeRecoveryOutcome.DEADLINE:
        return NativeDiagnosticOutcome.RECOVERY_DEADLINE
    if result.outcome is NativeRecoveryOutcome.ANCHOR_HINTS_REQUIRED:
        return NativeDiagnosticOutcome.RECOVERY_HINTS_REQUIRED
    if result.outcome is NativeRecoveryOutcome.MOVEMENT_REQUIRED:
        return NativeDiagnosticOutcome.RECOVERY_MOVEMENT_REQUIRED
    if result.outcome is NativeRecoveryOutcome.MOVEMENT_NOT_OBSERVED:
        return NativeDiagnosticOutcome.RECOVERY_MOVEMENT_NOT_OBSERVED
    if result.outcome in {
        NativeRecoveryOutcome.MONSTER_CONSENSUS_NOT_FOUND,
        NativeRecoveryOutcome.ACTOR_LAYOUT_INCONCLUSIVE,
        NativeRecoveryOutcome.SPAWN_PLAYER_NOT_FOUND,
        NativeRecoveryOutcome.ANCHOR_AMBIGUOUS,
        NativeRecoveryOutcome.MOVEMENT_CANDIDATE_STALE,
    }:
        return NativeDiagnosticOutcome.RECOVERY_ANCHOR_INCONCLUSIVE
    if result.outcome in {
        NativeRecoveryOutcome.NOT_FOUND,
        NativeRecoveryOutcome.NEGATIVE_CACHE,
    }:
        return NativeDiagnosticOutcome.RECOVERY_NOT_FOUND
    return NativeDiagnosticOutcome.ERROR


def _provider_health(bot: Any) -> NativeProviderHealth:
    position = getattr(bot, "position_provider", None)
    monster = getattr(bot, "monster_provider", None)
    position_diagnostics = getattr(position, "last_diagnostics", None)
    if not isinstance(position_diagnostics, PositionReadDiagnostics):
        position_diagnostics = None
    monster_diagnostics = getattr(monster, "last_diagnostics", None)
    if not isinstance(monster_diagnostics, ActorPoolDiagnostics):
        monster_diagnostics = None
    config = getattr(position, "config", None)
    resolver_value = getattr(config, "resolver", None)
    resolver = None if resolver_value is None else str(resolver_value)
    cached_slots = getattr(monster, "discovered_slot_bases", ())
    try:
        cached_actor_slots = len(cached_slots)
    except TypeError:
        cached_actor_slots = 0
    return NativeProviderHealth(
        position_attached=position is not None,
        position_resolver=resolver,
        position_last_read=position_diagnostics,
        monster_attached=monster is not None,
        cached_actor_slots=max(0, int(cached_actor_slots)),
        monster_last_read=monster_diagnostics,
    )


def _runtime_facts(
    bot: Any,
    *,
    pointer_snapshot: object | None = None,
) -> NativeRuntimeFacts:
    config_value = getattr(bot, "config", {})
    config: Mapping[str, object] = (
        config_value if isinstance(config_value, Mapping) else {}
    )
    selected_value = config.get("selected_map_name")
    selected_map = None
    if selected_value is not None and str(selected_value).strip():
        selected_map = str(selected_value).strip()
    overlay_name = getattr(bot, "_native_map_overlay_name", None)
    overlay = getattr(bot, "_native_map_overlay", None)
    map_overlay_loaded = bool(
        selected_map
        and overlay_name == selected_map
        and overlay is not None
    )

    native_player_position: tuple[float, float, float] | None = None
    map_coordinate_cell: tuple[float, float] | None = None
    coordinate_error: str | None = None
    if pointer_snapshot is not None:
        position = getattr(bot, "position_provider", None)
        read_pose = getattr(position, "read_pose", None)
        coordinate_frame = getattr(overlay, "coordinate_frame", None)
        to_local_cells = getattr(coordinate_frame, "to_local_cells", None)
        if not callable(read_pose):
            coordinate_error = "Native position reader is unavailable"
        elif not callable(to_local_cells):
            coordinate_error = "Selected map coordinate frame is unavailable"
        else:
            try:
                pose = read_pose(pointer_snapshot=pointer_snapshot)
                x = float(getattr(pose, "x"))
                y = float(getattr(pose, "y"))
                z = float(getattr(pose, "z"))
                local_x, local_y = cast(Any, to_local_cells)(x, z)
                native_player_position = (x, y, z)
                map_coordinate_cell = (float(local_x), float(local_y))
            except Exception as error:  # noqa: BLE001 - typed health boundary.
                coordinate_error = f"{type(error).__name__}: {error}"

    reader = getattr(bot, "kill_counter_reader", None)
    anchors = getattr(reader, "_anchors", None)
    ocr_anchor_cached = (
        bool(anchors)
        if isinstance(anchors, Mapping)
        else None
    )
    full_scan_value = getattr(reader, "full_scan_count", None)
    ocr_full_scan_count = _optional_int(full_scan_value)

    keyboard = getattr(bot, "keyboard", None)
    target_focused: bool | None = None
    focus_error: str | None = None
    focus_check = getattr(keyboard, "is_target_foreground", None)
    if callable(focus_check):
        try:
            target_focused = bool(focus_check())
        except Exception as error:  # noqa: BLE001 - health must remain available.
            focus_error = f"{type(error).__name__}: {error}"

    return NativeRuntimeFacts(
        selected_map_name=selected_map,
        map_overlay_loaded=map_overlay_loaded,
        native_player_position=native_player_position,
        map_coordinate_cell=map_coordinate_cell,
        coordinate_error=coordinate_error,
        ocr_enabled=bool(config.get("dynamic_kill_counter", False)),
        ocr_anchor_cached=ocr_anchor_cached,
        ocr_full_scan_count=ocr_full_scan_count,
        keyboard_attached=keyboard is not None,
        target_focused=target_focused,
        focus_error=focus_error,
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None
