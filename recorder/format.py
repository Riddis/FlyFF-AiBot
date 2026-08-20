"""Re-exports the generic packed-stream write primitives from
`recording_format.py` (repository root) -- moved there so the dev app's
own recording sink can reuse the exact same archive-writing code
without importing `recorder` itself. This module stays for
`recorder/session.py` and anything else already importing
`recorder.format` -- one implementation, two import paths."""

from __future__ import annotations

from recording_format import (
    FORMAT_VERSION,
    PackedStreamWriter,
    atomic_json,
    package_session,
    read_packed_stream,
    remove_session_directory,
    safe_component,
    utc_timestamp,
)

__all__ = [
    "FORMAT_VERSION",
    "PackedStreamWriter",
    "atomic_json",
    "package_session",
    "read_packed_stream",
    "remove_session_directory",
    "safe_component",
    "utc_timestamp",
]
