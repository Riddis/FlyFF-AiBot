from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PATCH_VERSION = "0.7.0.8"


def _insert_once(source: str, marker: str, addition: str, *, label: str) -> str:
    if addition.strip() in source:
        return source
    if marker not in source:
        raise RuntimeError(f"Could not find {label} marker")
    return source.replace(marker, marker + addition, 1)


def _patch_position_provider(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    source = _insert_once(
        source,
        "from .PositionConfig import NativePositionConfig\n",
        "from .NativePointerRecovery import (\n"
        "    PlayerPointerRecovery,\n"
        "    recover_local_player_pointer,\n"
        ")\n",
        label="position-provider import",
    )
    source = _insert_once(
        source,
        "        self.last_diagnostics: PositionReadDiagnostics | None = None\n",
        "        self.last_pointer_recovery: PlayerPointerRecovery | None = None\n",
        label="position-provider diagnostics",
    )

    old = '''        if target == 0:\n            raise PointerResolutionError(\n                f"Player pointer at 0x{pointer_storage:X} is null; "\n                "the character may not be fully logged in yet"\n            )\n        return target\n'''
    new = '''        if target == 0:\n            module_base = self._module_base\n            configured_offset = self.config.pointer_offset\n            recovery = None\n            if module_base is not None and configured_offset is not None:\n                recovery = recover_local_player_pointer(\n                    self._memory,\n                    module_base=module_base,\n                    configured_player_pointer_offset=configured_offset,\n                )\n            if recovery is not None:\n                self.last_pointer_recovery = recovery\n                self._pointer_storage_address = recovery.player_pointer_address\n                pointer_storage = recovery.player_pointer_address\n                try:\n                    target = int(\n                        struct.unpack(\n                            "<I",\n                            self._memory.read(pointer_storage, 4),\n                        )[0]\n                    )\n                except Exception as error:\n                    raise PointerResolutionError(\n                        "Recovered player pointer could not be reread at "\n                        f"0x{pointer_storage:X}: {type(error).__name__}: {error}"\n                    ) from error\n            if target == 0:\n                raise PointerResolutionError(\n                    f"Player pointer at 0x{pointer_storage:X} is null; "\n                    "the character may not be fully logged in yet, or the "\n                    "client update changed the configured pointer offset"\n                )\n        return target\n'''
    if new not in source:
        if old not in source:
            raise RuntimeError("Could not find the null player-pointer block")
        source = source.replace(old, new, 1)
    path.write_text(source, encoding="utf-8")


def _patch_monster_provider(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    source = _insert_once(
        source,
        "from .MonsterConfig import NativeMonsterConfig\n",
        "from .NativePointerRecovery import (\n"
        "    PlayerPointerRecovery,\n"
        "    recover_local_player_pointer,\n"
        ")\n",
        label="monster-provider import",
    )
    source = _insert_once(
        source,
        "        self.last_diagnostics: ActorPoolDiagnostics | None = None\n",
        "        self.last_pointer_recovery: PlayerPointerRecovery | None = None\n",
        label="monster-provider diagnostics",
    )

    old = '''    def read_player_base(self) -> int:\n        base = self._read_u32(self._player_pointer_address)\n        if base <= 0:\n            raise NativeMonsterReadError("Local-player pointer is null")\n        return base\n\n    def read_world_base(self) -> int:\n        base = self._read_u32(self._world_pointer_address)\n        if base <= 0:\n            raise NativeMonsterReadError("Current-world pointer is null")\n        return base\n'''
    new = '''    def _recover_pointer_slots(self) -> PlayerPointerRecovery | None:\n        recovery = recover_local_player_pointer(\n            self._memory,\n            module_base=self._module_base,\n            configured_player_pointer_offset=self.config.player_pointer_offset,\n            monster_config=self.config,\n        )\n        if recovery is None:\n            return None\n        self.last_pointer_recovery = recovery\n        self._player_pointer_address = recovery.player_pointer_address\n        if recovery.world_pointer_address is not None:\n            self._world_pointer_address = recovery.world_pointer_address\n        return recovery\n\n    def read_player_base(self) -> int:\n        base = self._read_u32(self._player_pointer_address)\n        if base <= 0:\n            recovery = self._recover_pointer_slots()\n            if recovery is not None:\n                base = self._read_u32(self._player_pointer_address)\n        if base <= 0:\n            raise NativeMonsterReadError(\n                "Local-player pointer is null and automatic offset recovery "\n                "did not find a validated replacement"\n            )\n        return base\n\n    def read_world_base(self) -> int:\n        base = self._read_u32(self._world_pointer_address)\n        if base <= 0:\n            recovery = self._recover_pointer_slots()\n            if recovery is not None:\n                base = self._read_u32(self._world_pointer_address)\n        if base <= 0:\n            raise NativeMonsterReadError(\n                "Current-world pointer is null and automatic offset recovery "\n                "did not find a validated replacement"\n            )\n        return base\n'''
    if new not in source:
        if old not in source:
            raise RuntimeError("Could not find the native player/world read methods")
        source = source.replace(old, new, 1)
    path.write_text(source, encoding="utf-8")


def apply(project: Path, *, run_tests: bool) -> None:
    project = project.resolve()
    required = [
        project / "position" / "NativeFlyffPositionProvider.py",
        project / "position" / "NativeFlyffMonsterProvider.py",
        project / "position" / "native_position.json",
        project / "position" / "native_monsters.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing required files:\n" + "\n".join(missing))

    root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = project / ".patch_backups" / f"v0708_{timestamp}"
    touched = [
        project / "position" / "NativeFlyffPositionProvider.py",
        project / "position" / "NativeFlyffMonsterProvider.py",
        project / "position" / "NativePointerRecovery.py",
        project / "tests" / "test_v0708_pointer_recovery.py",
    ]
    existed = {path: path.exists() for path in touched}
    for path in touched:
        if path.exists():
            relative = path.relative_to(project)
            destination = backup_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    try:
        _patch_position_provider(project / "position" / "NativeFlyffPositionProvider.py")
        _patch_monster_provider(project / "position" / "NativeFlyffMonsterProvider.py")
        shutil.copy2(
            root / "payload" / "position" / "NativePointerRecovery.py",
            project / "position" / "NativePointerRecovery.py",
        )
        shutil.copy2(
            root / "payload" / "tests" / "test_v0708_pointer_recovery.py",
            project / "tests" / "test_v0708_pointer_recovery.py",
        )

        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "py_compile",
                str(project / "position" / "NativePointerRecovery.py"),
                str(project / "position" / "NativeFlyffPositionProvider.py"),
                str(project / "position" / "NativeFlyffMonsterProvider.py"),
            ],
            cwd=project,
            check=True,
        )
        if run_tests:
            command = [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                "-q",
                "tests/test_v0708_pointer_recovery.py",
                "tests/test_native_position_provider.py",
                "tests/test_native_monster_provider.py",
            ]
            print("Running focused tests:", " ".join(command))
            result = subprocess.run(command, cwd=project)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Focused tests failed with exit code {result.returncode}"
                )
    except Exception:
        for path in touched:
            relative = path.relative_to(project)
            backup = backup_root / relative
            if backup.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, path)
            elif not existed[path] and path.exists():
                path.unlink()
        raise

    print(f"Applied native pointer recovery patch v{PATCH_VERSION}.")
    print(f"Backups: {backup_root}")
    print(
        "The first null-pointer read will scan for the shifted player/world "
        "globals and persist validated offsets to both native JSON configs."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    try:
        apply(args.project, run_tests=args.run_tests)
    except Exception as error:
        print(f"PATCH FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
