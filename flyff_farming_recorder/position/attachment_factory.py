"""B2 compatibility exports for the canonical position implementation."""

from position.attachment_factory import (
    Callable,
    DEFAULT_MONSTER_CONFIG_PATH,
    DEFAULT_POSITION_CONFIG_PATH,
    NativeFlyffMonsterProvider,
    NativeFlyffPositionProvider,
    NativeProcessService,
    NativeProviderAttachment,
    Path,
    Win32MemoryBackend,
    create_native_provider_attachment,
    dataclass,
    field,
    load_native_monster_config,
    load_native_position_config,
    monotonic,
    recover_interrupted_pointer_persistence,
)
