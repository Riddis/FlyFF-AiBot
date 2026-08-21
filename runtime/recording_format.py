"""Generic gzip+msgpack packed-stream write primitives.

Moved out of `devtools/recorder/format.py` (which now re-exports from here
for backward compatibility) so the canonical dev app's own recording sink
(`recording_sink.py`) can reuse the exact same archive-writing code
without importing the `devtools.recorder` package itself -- `devtools.recorder`
stays outside the dev app's import closure (it is the *acquisition* package;
the dev app owns its own native reader and must never run a second
one). This module never imports anything native/recorder-specific --
stdlib plus `msgpack` only, the same as before the move.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgpack


FORMAT_VERSION = 1


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or "player"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class PackedStreamWriter:
    def __init__(self, path: Path, *, header: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._raw = gzip.open(path, "wb", compresslevel=6)
        self._packer = msgpack.Packer(use_bin_type=True)
        self._closed = False
        self.write({"type": "header", "format_version": FORMAT_VERSION, **header})

    def write(self, value: object) -> None:
        if self._closed:
            raise RuntimeError("Packed stream is closed")
        self._raw.write(self._packer.pack(value))

    def flush(self) -> None:
        if not self._closed:
            self._raw.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._raw.close()

    def __enter__(self) -> "PackedStreamWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def read_packed_stream(path: Path):
    with gzip.open(path, "rb") as handle:
        yield from msgpack.Unpacker(handle, raw=False, strict_map_key=False)


def package_session(session_directory: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(session_directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(session_directory))
    os.replace(temporary, destination)
    return destination


def remove_session_directory(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
