"""Classify every recording archive and report what it can be used for.

Runs entirely from already-recorded frame data (position, focus, key mask),
so it produces a real classification even for archives that predate
recorder 1.11's embedded recording_provenance block (1.7.0-1.9.0 wrote no
such field at all and would otherwise show up as "unknown"). This mirrors
exactly what the recorder's own MovementControlClassifier does live, just
replayed after the fact -- there is no separate/looser logic here.

Usage:
    python -m tools.inventory_recordings recordings/training recordings/eva_only
    python -m tools.inventory_recordings --output recordings/INDEX.json recordings/**/*.zip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from recorder.movement_classification import MovementControlClassifier
from simulator.schema import (
    RecordingArchive,
    allows_direct_movement_labels,
    direct_movement_provenance_source,
    has_validated_presence,
    recording_sha256,
)

_LEGACY_ACTION_NAMES = ("FORWARD", "LEFT", "RIGHT", "EVA", "JUMP")


def classify_recording(path: Path) -> dict[str, Any]:
    archive = RecordingArchive(path)
    manifest = archive.manifest

    classifier = MovementControlClassifier()
    legacy_action_counts = [0, 0, 0, 0, 0]
    total_frames = 0
    for frame in archive.frames():
        total_frames += 1
        classifier.observe(
            x=frame.player_x,
            z=frame.player_z,
            focused=frame.focused,
            key_mask=frame.key_mask,
        )
        if 0 <= frame.action < 5:
            legacy_action_counts[frame.action] += 1
    movement_report = classifier.report().to_dict()

    sha256 = recording_sha256(path)
    presence_validated = has_validated_presence(manifest)
    demo_eligible = allows_direct_movement_labels(manifest, recording_hash=sha256)
    provenance_source = direct_movement_provenance_source(manifest, recording_hash=sha256)
    eva_events = legacy_action_counts[3]
    sampling = manifest.get("sampling") or {}

    embedded_scheme = None
    embedded_provenance = manifest.get("recording_provenance")
    if isinstance(embedded_provenance, dict):
        embedded_classification = embedded_provenance.get("movement_classification")
        if isinstance(embedded_classification, dict):
            embedded_scheme = embedded_classification.get("scheme")

    return {
        "path": str(path),
        "filename": path.name,
        "sha256": sha256,
        "recorder_version": manifest.get("recorder_version"),
        "duration_seconds": manifest.get("duration_seconds"),
        "total_frames": total_frames,
        "embedded_movement_scheme": embedded_scheme,
        "retroactive_movement_scheme": movement_report["scheme"],
        "retroactive_movement_confidence": movement_report["confidence"],
        "retroactive_classification": movement_report,
        "legacy_action_counts": dict(zip(_LEGACY_ACTION_NAMES, legacy_action_counts, strict=True)),
        "eva_events": eva_events,
        "jump_events": legacy_action_counts[4],
        "presence_species_offset": sampling.get("presence_species_offset"),
        "presence_species_validated": presence_validated,
        "presence_validation_source": (manifest.get("native") or {}).get("presence_validation_source"),
        "direct_movement_provenance_source": provenance_source,
        "ready_for_demonstrations": demo_eligible,
        "ready_for_world_model": presence_validated,
        "eva_only_eligible": eva_events > 0,
        "usable_for": _usable_for(demo_eligible, presence_validated, eva_events),
    }


def _usable_for(demo_eligible: bool, world_eligible: bool, eva_events: int) -> list[str]:
    uses: list[str] = []
    if demo_eligible:
        uses.append("direct_steering_demonstration")
    if world_eligible:
        uses.append("world_model_fitting")
    if eva_events > 0:
        uses.append("eva_event_evidence")
    uses.append("pointer_recovery_diagnostics")  # every archive keeps this value; see recording_provenance.json notes
    return uses


def _iter_archives(patterns: list[str]) -> list[Path]:
    seen: dict[str, Path] = {}
    for raw in patterns:
        candidate = Path(raw)
        matches = [candidate] if candidate.is_file() else sorted(candidate.parent.glob(candidate.name)) if not candidate.is_dir() else sorted(candidate.glob("*.zip"))
        if candidate.is_dir():
            matches = sorted(candidate.glob("*.zip"))
        for match in matches:
            if match.suffix.lower() == ".zip" and match.is_file():
                seen[str(match.resolve())] = match
    return sorted(seen.values(), key=lambda item: item.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Recording archives, or directories to scan for *.zip")
    parser.add_argument("--output", type=Path, help="Write the full JSON report here")
    args = parser.parse_args()

    archives = _iter_archives(args.paths)
    if not archives:
        raise SystemExit("No recording archives matched the given paths")

    results = []
    for path in archives:
        try:
            results.append(classify_recording(path))
        except Exception as error:  # noqa: BLE001
            results.append({"path": str(path), "filename": path.name, "error": f"{type(error).__name__}: {error}"})

    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
