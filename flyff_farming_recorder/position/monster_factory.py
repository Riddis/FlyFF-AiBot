"""B2 compatibility exports for the canonical position implementation."""

# BRIDGE B2 — removed in Phase 7
from position.monster_factory import (
    Callable,
    DEFAULT_MONSTER_CONFIG_PATH,
    DEFAULT_POSITION_CONFIG_PATH,
    NativeFlyffMonsterProvider,
    NativeProcessService,
    Path,
    Win32MemoryBackend,
    create_native_monster_provider,
    create_native_monster_provider_from_process_id,
    load_native_monster_config,
    monotonic,
    recover_interrupted_pointer_persistence,
)
