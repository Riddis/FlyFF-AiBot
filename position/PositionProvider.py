from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class PositionProviderError(RuntimeError):
    """Base error raised by native position providers."""


@dataclass(frozen=True, slots=True)
class PlayerPose:
    """One read-only sample of the local player's native world pose."""

    x: float
    y: float
    z: float
    heading_degrees: float | None
    timestamp: float


@runtime_checkable
class PositionProvider(Protocol):
    """Minimal position interface consumed by the bot and mapper."""

    def read_pose(self) -> PlayerPose:
        """Read the latest native player pose."""

    def close(self) -> None:
        """Release any process handles owned by the provider."""
