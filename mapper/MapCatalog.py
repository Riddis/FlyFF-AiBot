from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MapProfile:
    name: str
    slug: str
    mobs: tuple[str, ...]


class MapCatalog:
    """Persistent catalog for named maps and their available mobs."""

    SKIP_LEGACY_IMPORT_MARKER = ".skip_legacy_import"

    def __init__(
        self,
        path: Path | None = None,
        *,
        maps_root: Path | None = None,
        runs_root: Path | None = None,
    ) -> None:
        module_root = Path(__file__).resolve().parent
        self.path = path or module_root / "map_profiles.json"
        self.maps_root = maps_root or module_root / "maps"
        self.runs_root = runs_root or module_root / "mapping_runs"
        self._profiles: dict[str, MapProfile] = {}
        self._default_name = ""
        self.reload()

    @property
    def default_name(self) -> str:
        return self._default_name

    def reload(self) -> None:
        self._profiles, self._default_name = self._load()

    def names(self) -> tuple[str, ...]:
        return tuple(profile.name for profile in self._profiles.values())

    def get(self, name: str | None = None) -> MapProfile:
        requested = str(name or self._default_name).strip()
        profile = self._profiles.get(requested)
        if profile is None:
            available = ", ".join(self.names())
            raise ValueError(f"Unknown map {requested!r}. Available maps: {available}")
        return profile

    def mobs_for(self, name: str | None = None) -> tuple[str, ...]:
        return self.get(name).mobs

    def map_directory(self, name: str | None = None) -> Path:
        return self.maps_root / self.get(name).slug

    def preview_path(self, name: str | None = None) -> Path:
        return self.map_directory(name) / "map_preview.png"

    def run_directory(self, name: str | None = None) -> Path:
        return self.runs_root / self.get(name).slug

    def create_map(
        self,
        name: str,
        *,
        mobs: Iterable[str] = (),
    ) -> MapProfile:
        clean_name = self._validate_new_name(name)
        slug = self.slugify(clean_name)
        used_slugs = {profile.slug.casefold() for profile in self._profiles.values()}
        if slug.casefold() in used_slugs:
            raise ValueError(
                f"Map name {clean_name!r} produces an existing map folder {slug!r}."
            )

        profile = MapProfile(
            name=clean_name,
            slug=slug,
            mobs=self._clean_mobs(mobs),
        )
        self._profiles[profile.name] = profile
        self._persist()

        # A newly created profile must never import an unrelated pre-v1.3 run.
        self._write_legacy_import_marker(profile.name)
        return profile

    def update_mobs(self, name: str, mobs: Iterable[str]) -> MapProfile:
        current = self.get(name)
        updated = MapProfile(
            name=current.name,
            slug=current.slug,
            mobs=self._clean_mobs(mobs),
        )
        self._profiles[current.name] = updated
        self._persist()
        return updated

    def reset_map(
        self,
        name: str,
        *,
        delete_run_history: bool = False,
    ) -> MapProfile:
        profile = self.get(name)
        self._safe_remove_tree(self.map_directory(profile.name), self.maps_root)
        self._write_legacy_import_marker(profile.name)
        if delete_run_history:
            self._safe_remove_tree(self.run_directory(profile.name), self.runs_root)
        return profile

    def delete_map(
        self,
        name: str,
        *,
        delete_run_history: bool = False,
    ) -> str:
        profile = self.get(name)
        if len(self._profiles) <= 1:
            raise ValueError("At least one map profile must remain.")

        del self._profiles[profile.name]
        if self._default_name == profile.name:
            self._default_name = next(iter(self._profiles))
        self._persist()

        self._safe_remove_tree(
            self.map_directory_for_profile(profile, self.maps_root),
            self.maps_root,
        )
        if delete_run_history:
            self._safe_remove_tree(
                self.run_directory_for_profile(profile, self.runs_root),
                self.runs_root,
            )
        return self._default_name

    def legacy_import_allowed(self, name: str | None = None) -> bool:
        marker = self.map_directory(name) / self.SKIP_LEGACY_IMPORT_MARKER
        return not marker.exists()

    def mark_legacy_import_complete(self, name: str | None = None) -> None:
        self._write_legacy_import_marker(self.get(name).name)

    def best_legacy_run(self) -> Path | None:
        """Find the richest pre-v1.3 run for one-time persistent-map import."""
        import numpy as np

        if not self.runs_root.is_dir():
            return None
        candidates: list[tuple[int, str, Path]] = []
        for directory in self.runs_root.iterdir():
            if not directory.is_dir():
                continue
            # v1.3+ map-specific directories contain run subdirectories and are
            # not legacy snapshots themselves.
            occupancy_path = directory / "occupancy.npy"
            if not (
                occupancy_path.is_file()
                and (directory / "visits.npy").is_file()
                and (directory / "map.json").is_file()
            ):
                continue
            try:
                cells = np.load(occupancy_path, allow_pickle=False)
                known = int(np.count_nonzero(cells))
            except Exception:  # noqa: BLE001 - ignore corrupt legacy runs.
                continue
            candidates.append((known, directory.name, directory))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2]

    @staticmethod
    def slugify(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
        return slug or "unnamed_map"

    @staticmethod
    def map_directory_for_profile(profile: MapProfile, root: Path | None = None) -> Path:
        base = root or Path(__file__).resolve().parent / "maps"
        return base / profile.slug

    @staticmethod
    def run_directory_for_profile(profile: MapProfile, root: Path | None = None) -> Path:
        base = root or Path(__file__).resolve().parent / "mapping_runs"
        return base / profile.slug

    def _validate_new_name(self, name: str) -> str:
        clean_name = " ".join(str(name).strip().split())
        if not clean_name:
            raise ValueError("Map name cannot be empty.")
        if len(clean_name) > 80:
            raise ValueError("Map name cannot exceed 80 characters.")
        existing = {profile.name.casefold() for profile in self._profiles.values()}
        if clean_name.casefold() in existing:
            raise ValueError(f"Map {clean_name!r} already exists.")
        return clean_name

    @staticmethod
    def _clean_mobs(mobs: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                clean
                for mob in mobs
                if (clean := str(mob).strip())
            )
        )

    def _write_legacy_import_marker(self, name: str) -> None:
        directory = self.map_directory(name)
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / self.SKIP_LEGACY_IMPORT_MARKER
        marker.write_text(
            "Persistent map was created or reset explicitly; do not import legacy runs.\n",
            encoding="utf-8",
        )

    @staticmethod
    def _safe_remove_tree(directory: Path, root: Path) -> None:
        if not directory.exists():
            return
        resolved_root = root.resolve()
        resolved_directory = directory.resolve()
        if resolved_directory == resolved_root or not resolved_directory.is_relative_to(
            resolved_root
        ):
            raise ValueError(f"Refusing to remove unsafe map path: {directory}")
        shutil.rmtree(resolved_directory)

    def _persist(self) -> None:
        payload = {
            "version": 1,
            "default_map": self._default_name,
            "maps": [
                {
                    "name": profile.name,
                    "slug": profile.slug,
                    "mobs": list(profile.mobs),
                }
                for profile in self._profiles.values()
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def _load(self) -> tuple[dict[str, MapProfile], str]:
        if not self.path.is_file():
            raise FileNotFoundError(f"Map catalog is missing: {self.path}")
        payload: Any = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Map catalog root must be an object")
        raw_maps = payload.get("maps")
        if not isinstance(raw_maps, list) or not raw_maps:
            raise ValueError("Map catalog must contain at least one map")

        profiles: dict[str, MapProfile] = {}
        used_slugs: set[str] = set()
        used_names: set[str] = set()
        for raw in raw_maps:
            if not isinstance(raw, dict):
                raise ValueError("Each map catalog entry must be an object")
            name = str(raw.get("name", "")).strip()
            if not name:
                raise ValueError("Map names cannot be empty")
            slug = str(raw.get("slug") or self.slugify(name)).strip()
            folded_name = name.casefold()
            folded_slug = slug.casefold()
            if not slug or folded_slug in used_slugs:
                raise ValueError(f"Duplicate or invalid map slug: {slug!r}")
            raw_mobs = raw.get("mobs", [])
            if not isinstance(raw_mobs, list):
                raise ValueError(f"Map {name!r} mobs must be a list")
            mobs = self._clean_mobs(raw_mobs)
            if folded_name in used_names:
                raise ValueError(f"Duplicate map name: {name!r}")
            profiles[name] = MapProfile(name=name, slug=slug, mobs=mobs)
            used_names.add(folded_name)
            used_slugs.add(folded_slug)

        default_name = str(payload.get("default_map") or next(iter(profiles))).strip()
        if default_name not in profiles:
            raise ValueError(f"Default map {default_name!r} is not in the catalog")
        return profiles, default_name
