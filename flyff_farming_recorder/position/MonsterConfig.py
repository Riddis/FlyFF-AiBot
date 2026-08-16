"""B2 compatibility exports for the canonical position implementation."""

# BRIDGE B2 — removed in Phase 7
from position.MonsterConfig import (
    Any,
    DEFAULT_MONSTER_CONFIG_PATH,
    MonsterConfigurationError,
    NativeMonsterConfig,
    Path,
    PositionProviderError,
    _parse_int,
    _parse_offset_list,
    _parse_optional_int,
    dataclass,
    json,
    load_native_monster_config,
)
