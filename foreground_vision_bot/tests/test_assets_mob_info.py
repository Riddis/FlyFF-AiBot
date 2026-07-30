from __future__ import annotations

import json
from pathlib import Path

import assets.Assets as assets_module
from assets.Assets import MobInfo


def _redirect_assets(monkeypatch, tmp_path: Path) -> Path:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "names").mkdir()
    monkeypatch.setattr(assets_module, "__file__", str(assets_dir / "Assets.py"))
    return assets_dir


def test_add_new_mob_skips_copy_when_legacy_image_is_already_destination(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assets_dir = _redirect_assets(monkeypatch, tmp_path)
    stored_image = assets_dir / "names" / "Captain Asterius.png"
    stored_image.write_bytes(b"existing-image")

    MobInfo.add_new_mob(
        name="Captain Asterius",
        map_name="Tower AoE",
        image_path=str(stored_image),
        height_offset=50,
        element="water",
        species_id=944,
    )

    assert stored_image.read_bytes() == b"existing-image"
    saved = json.loads((assets_dir / "mobs_list.json").read_text())
    assert saved["Captain Asterius"]["species_id"] == 944


def test_add_new_mob_still_copies_a_distinct_legacy_image(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assets_dir = _redirect_assets(monkeypatch, tmp_path)
    source_image = tmp_path / "captured.png"
    source_image.write_bytes(b"new-image")

    MobInfo.add_new_mob(
        name="Captain Reinecke",
        map_name="Tower AoE",
        image_path=str(source_image),
        height_offset=40,
        element="fire",
        species_id=948,
    )

    copied = assets_dir / "names" / "Captain Reinecke.png"
    assert copied.read_bytes() == b"new-image"
    saved = json.loads((assets_dir / "mobs_list.json").read_text())
    assert saved["Captain Reinecke"]["species_id"] == 948
