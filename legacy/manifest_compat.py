"""Historical archive-manifest compatibility.

Recorder versions 1.7.0 and 1.9.0 wrote manifests with no ``policy_contract``,
``map_contract``, or embedded ``recording_provenance`` at all; recorder
1.11.0 is the first version that embeds all three (confirmed by directly
reading all 8 Phase-0 archives' manifests -- see
``docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md`` sections C and F). This
module isolates exactly those absence-driven compatibility decisions. It
takes the current contract values (role/scheme) as parameters rather than
importing them from ``simulator.schema``, so the dependency direction stays
canonical (``simulator.schema``) -> legacy, never the reverse.

A present-but-wrong contract is always a hard current-format validation
failure in ``simulator.schema`` itself -- this module is consulted only when
the relevant field is genuinely absent.

This intentionally lives at top-level ``legacy/``, not nested under an
``archives/`` package: the frozen Phase-3 G7 semantic contract encodes each
decoded record's fully-qualified class name
(``type(value).__module__.__qualname__``) as part of its typed hash, so
``RecordingArchive``/``RecordedFrame``/``RecordedActor``/``RecordedEvent``
cannot be relocated out of ``simulator.schema`` without changing that frozen
hash -- see ``docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md`` section F for
the full account of why the originally-planned ``archives/schema.py``
placement was abandoned.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DEFAULT_PROVENANCE_REGISTRY = Path(__file__).resolve().parents[1] / "recording_provenance.json"


def missing_policy_contract_warning(archive_name: str) -> str:
    return f"{archive_name}: legacy archive has no embedded policy contract"


def missing_map_contract_warning(archive_name: str) -> str:
    return f"{archive_name}: legacy archive has no embedded coordinate-frame contract"


@lru_cache(maxsize=8)
def _trusted_direct_hashes(required_role: str, required_scheme: str, registry_path: str) -> frozenset[str]:
    path = Path(registry_path)
    if not path.is_file():
        return frozenset()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Invalid recording provenance registry: {path}")
    recordings = payload.get("recordings")
    if not isinstance(recordings, dict):
        raise ValueError(f"Recording provenance registry has no recordings object: {path}")
    trusted: set[str] = set()
    for raw_hash, provenance in recordings.items():
        digest = str(raw_hash).upper()
        if len(digest) != 64 or any(character not in "0123456789ABCDEF" for character in digest):
            raise ValueError(f"Invalid SHA-256 in recording provenance registry: {raw_hash!r}")
        if not isinstance(provenance, dict):
            raise ValueError(f"Invalid provenance entry for {digest}")
        if (
            provenance.get("recording_role") == required_role
            and provenance.get("movement_control_scheme") == required_scheme
            and provenance.get("direct_movement_labels_allowed") is True
        ):
            trusted.add(digest)
    return frozenset(trusted)


def attested_by_registry(
    *,
    recording_hash: str | None,
    required_role: str,
    required_scheme: str,
    registry_path: str | Path = DEFAULT_PROVENANCE_REGISTRY,
) -> bool:
    """Legacy fallback: an archive predating embedded ``recording_provenance``
    can still be recognized as demonstration-eligible via an external,
    hash-keyed attestation registry (``recording_provenance.json``)."""

    return bool(
        recording_hash is not None
        and str(recording_hash).upper()
        in _trusted_direct_hashes(required_role, required_scheme, str(Path(registry_path).resolve()))
    )
