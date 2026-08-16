"""B2 compatibility exports for the canonical position implementation."""

# BRIDGE B2 — removed in Phase 7
from position.AutonomousPointerSelection import (
    AnchoredPlayerObservation,
    Callable,
    DirectPlayerSlotEvidence,
    Mapping,
    MemoryRegion,
    ModuleInfo,
    PassivePlayerProof,
    Path,
    PointerScanSnapshot,
    PointerWorkflowMemory,
    _PAGE_WRITABLE,
    _is_writable,
    _read_player_like,
    _region_for,
    dataclass,
    historical_direct_offset_counts,
    load_matching_snapshot_history,
    math,
    prove_player_and_rank_direct_slots,
    read_player_observation,
    sleep,
)
