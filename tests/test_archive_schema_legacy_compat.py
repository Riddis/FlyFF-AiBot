"""Characterizes the exact legacy/historical manifest-compatibility rules
documented in docs/migration/PHASE8_ARCHIVE_OWNER_ANALYSIS.md sections C and
F, against synthetic minimal archives -- before and after the Phase-8
legacy/ isolation (simulator.schema.RecordingArchive/RecordedFrame/
RecordedActor/RecordedEvent stayed at their original location; only the
absence-driven compatibility logic moved into legacy/manifest_compat.py, so
these imports are unchanged by that split). Each rule is proven both
positive (legacy archive triggers the adapter) and negative (current-format
archive does not need it)."""

from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import msgpack


def _write_stream(path: Path, values: list[object]) -> None:
    packer = msgpack.Packer(use_bin_type=True)
    with gzip.open(path, "wb") as handle:
        for value in values:
            handle.write(packer.pack(value))


def _build_archive(tmp_path: Path, name: str, manifest: dict[str, object]) -> Path:
    session = tmp_path / name
    session.mkdir()
    (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for member in ("frames", "events", "inputs"):
        _write_stream(session / f"{member}.msgpack.gz", [{"type": "header"}])
    zip_path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in session.iterdir():
            archive.write(path, path.name)
    return zip_path


def _base_manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 2,
        "recorder_version": "1.9.0",
        "sampling": {"position_quantum_native": 0.05},
    }
    manifest.update(overrides)
    return manifest


CURRENT_ACTION_NAMES = ("RUN_FORWARD", "RUN_FORWARD_LEFT", "RUN_FORWARD_RIGHT", "CAST_EVA", "RUN_FORWARD_JUMP")
CURRENT_MAP_ORIGIN = (0.0, 0.0, 1.0)


def _validate(archive):
    from simulator.schema import validate_recording_contract

    return validate_recording_contract(
        archive,
        action_names=CURRENT_ACTION_NAMES,
        origin_native_x=CURRENT_MAP_ORIGIN[0],
        origin_native_z=CURRENT_MAP_ORIGIN[1],
        native_units_per_cell=CURRENT_MAP_ORIGIN[2],
    )


def test_legacy_archive_without_policy_contract_warns_but_does_not_raise(tmp_path: Path) -> None:
    from simulator.schema import RecordingArchive

    path = _build_archive(tmp_path, "legacy_no_policy", _base_manifest(
        map_contract={"origin_native_x": 0.0, "origin_native_z": 0.0, "native_units_per_cell": 1.0},
    ))
    warnings = _validate(RecordingArchive(path))
    assert any("no embedded policy contract" in warning for warning in warnings)


def test_legacy_archive_without_map_contract_warns_but_does_not_raise(tmp_path: Path) -> None:
    from simulator.schema import RecordingArchive

    path = _build_archive(tmp_path, "legacy_no_map", _base_manifest(
        policy_contract={"action_names": list(CURRENT_ACTION_NAMES)},
    ))
    warnings = _validate(RecordingArchive(path))
    assert any("no embedded coordinate-frame contract" in warning for warning in warnings)


def test_current_format_archive_with_both_contracts_has_no_legacy_warnings(tmp_path: Path) -> None:
    from simulator.schema import RecordingArchive

    path = _build_archive(tmp_path, "current_both_contracts", _base_manifest(
        recorder_version="1.11.0",
        policy_contract={"action_names": list(CURRENT_ACTION_NAMES)},
        map_contract={"origin_native_x": 0.0, "origin_native_z": 0.0, "native_units_per_cell": 1.0},
    ))
    warnings = _validate(RecordingArchive(path))
    assert warnings == ()


def test_mismatched_policy_contract_still_raises_not_warns(tmp_path: Path) -> None:
    """The legacy adapter only covers ABSENCE. A present-but-wrong contract
    must still fail loudly -- the adapter must not be applied where it does
    not belong."""
    from simulator.schema import RecordingArchive

    path = _build_archive(tmp_path, "wrong_policy", _base_manifest(
        recorder_version="1.11.0",
        policy_contract={"action_names": ["WRONG"]},
        map_contract={"origin_native_x": 0.0, "origin_native_z": 0.0, "native_units_per_cell": 1.0},
    ))
    try:
        _validate(RecordingArchive(path))
    except ValueError as error:
        assert "action contract mismatch" in str(error)
    else:
        raise AssertionError("expected ValueError for a present-but-mismatched policy contract")


def test_provenance_registry_fallback_accepts_an_attested_legacy_archive(tmp_path: Path) -> None:
    from simulator.schema import allows_direct_movement_labels, recording_sha256

    path = _build_archive(tmp_path, "legacy_attested", _base_manifest())
    digest = recording_sha256(path)
    registry_path = tmp_path / "recording_provenance.json"
    registry_path.write_text(json.dumps({
        "schema_version": 1,
        "recordings": {
            digest: {
                "recording_role": "direct_keyboard_demonstration",
                "movement_control_scheme": "keyboard_wasd",
                "direct_movement_labels_allowed": True,
            },
        },
    }), encoding="utf-8")
    manifest = json.loads(zipfile.ZipFile(path).read("manifest.json"))
    assert allows_direct_movement_labels(
        manifest, recording_hash=digest, registry_path=registry_path,
    )


def test_provenance_registry_fallback_rejects_an_unattested_legacy_archive(tmp_path: Path) -> None:
    from simulator.schema import allows_direct_movement_labels, recording_sha256

    path = _build_archive(tmp_path, "legacy_unattested", _base_manifest())
    digest = recording_sha256(path)
    registry_path = tmp_path / "recording_provenance.json"
    registry_path.write_text(json.dumps({"schema_version": 1, "recordings": {}}), encoding="utf-8")
    manifest = json.loads(zipfile.ZipFile(path).read("manifest.json"))
    assert not allows_direct_movement_labels(
        manifest, recording_hash=digest, registry_path=registry_path,
    )


def test_embedded_provenance_is_trusted_without_consulting_the_registry(tmp_path: Path) -> None:
    """Negative/current-format test: the legacy registry fallback must not
    be consulted (and must not matter) when the manifest already embeds a
    valid direct-movement-labels declaration."""
    from simulator.schema import allows_direct_movement_labels

    manifest = _base_manifest(
        recorder_version="1.11.0",
        recording_provenance={
            "recording_role": "direct_keyboard_demonstration",
            "movement_control_scheme": "keyboard_wasd",
            "direct_movement_labels_allowed": True,
        },
    )
    assert allows_direct_movement_labels(
        manifest, recording_hash="0" * 64, registry_path=Path("does/not/exist.json"),
    )
