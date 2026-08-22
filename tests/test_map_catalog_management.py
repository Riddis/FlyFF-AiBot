from __future__ import annotations

import json
from pathlib import Path

import pytest

from mapper.MapCatalog import MapCatalog


def _catalog(tmp_path: Path) -> MapCatalog:
    path = tmp_path / "map_profiles.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "default_map": "Tower AoE",
                "maps": [
                    {
                        "name": "Tower AoE",
                        "slug": "tower_aoe",
                        "mobs": ["Captain Asterius", "Captain Dantalian"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return MapCatalog(
        path,
        maps_root=tmp_path / "maps",
        runs_root=tmp_path / "runs",
    )


def test_create_reset_and_reload_map_profile(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    profile = catalog.create_map(
        "Volcane Depths",
        mobs=("Meteonyker", "Meteonyker"),
    )

    assert profile.slug == "volcane_depths"
    assert profile.mobs == ("Meteonyker",)
    assert not catalog.legacy_import_allowed(profile.name)

    map_dir = catalog.map_directory(profile.name)
    (map_dir / "map.json").write_text("{}", encoding="utf-8")
    run_dir = catalog.run_directory(profile.name) / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "mapping_steps.csv").write_text("step\n", encoding="utf-8")

    catalog.reset_map(profile.name)

    assert not (map_dir / "map.json").exists()
    assert (map_dir / catalog.SKIP_LEGACY_IMPORT_MARKER).is_file()
    assert run_dir.is_dir()

    reloaded = MapCatalog(
        catalog.path,
        maps_root=catalog.maps_root,
        runs_root=catalog.runs_root,
    )
    assert reloaded.get("Volcane Depths").mobs == ("Meteonyker",)


def test_delete_map_removes_profile_and_can_keep_run_history(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    profile = catalog.create_map("Second Map", mobs=())
    run_dir = catalog.run_directory(profile.name) / "run-1"
    run_dir.mkdir(parents=True)

    selected = catalog.delete_map(profile.name, delete_run_history=False)

    assert selected == "Tower AoE"
    assert catalog.names() == ("Tower AoE",)
    assert run_dir.is_dir()
    with pytest.raises(ValueError, match="At least one map"):
        catalog.delete_map("Tower AoE")


def test_rejects_duplicate_map_names_case_insensitively(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    with pytest.raises(ValueError, match="already exists"):
        catalog.create_map("tower aoe")
