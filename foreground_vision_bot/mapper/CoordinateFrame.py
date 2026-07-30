from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic_ns


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """Persistent transform from FlyFF world units to occupancy-grid cells."""

    version: int = 1
    origin_native_x: float = 0.0
    origin_native_z: float = 0.0
    native_units_per_cell: float = 1.6

    _REPLACE_ATTEMPTS = 8
    _INITIAL_RETRY_SECONDS = 0.02
    _MAX_RETRY_SECONDS = 0.25

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"Unsupported coordinate-frame version: {self.version}")
        if not all(
            math.isfinite(value)
            for value in (
                self.origin_native_x,
                self.origin_native_z,
                self.native_units_per_cell,
            )
        ):
            raise ValueError("Coordinate-frame values must be finite")
        if self.native_units_per_cell <= 0.0:
            raise ValueError("native_units_per_cell must be positive")

    def to_local_cells(self, native_x: float, native_z: float) -> tuple[float, float]:
        if not math.isfinite(native_x) or not math.isfinite(native_z):
            raise ValueError("Native coordinates must be finite")
        return (
            (float(native_x) - self.origin_native_x) / self.native_units_per_cell,
            (float(native_z) - self.origin_native_z) / self.native_units_per_cell,
        )

    def save(self, path: Path) -> None:
        """Persist the frame without exposing a partially written JSON file.

        Windows can temporarily deny ``os.replace`` while antivirus, indexing,
        a file preview, or another reader has the destination open without
        delete sharing. The coordinate frame is immutable, so an identical
        existing file needs no rewrite. New content uses a unique temporary
        file and bounded retries before surfacing a real persistence failure.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(asdict(self), indent=2) + "\n"

        if self._destination_already_matches(path, serialized):
            return

        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{monotonic_ns()}.tmp"
        )
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())

            delay = self._INITIAL_RETRY_SECONDS
            for attempt in range(1, self._REPLACE_ATTEMPTS + 1):
                try:
                    os.replace(temporary, path)
                    return
                except PermissionError:
                    # Another writer may have completed the same immutable
                    # save while this process was waiting for the lock.
                    if self._destination_already_matches(path, serialized):
                        return
                    if attempt >= self._REPLACE_ATTEMPTS:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2.0, self._MAX_RETRY_SECONDS)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _destination_already_matches(path: Path, serialized: str) -> bool:
        try:
            return path.is_file() and path.read_text(encoding="utf-8") == serialized
        except OSError:
            return False

    @classmethod
    def load(cls, path: Path) -> CoordinateFrame:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Coordinate-frame root must be an object")
        return cls(
            version=int(payload.get("version", 1)),
            origin_native_x=float(payload["origin_native_x"]),
            origin_native_z=float(payload["origin_native_z"]),
            native_units_per_cell=float(payload.get("native_units_per_cell", 1.6)),
        )
