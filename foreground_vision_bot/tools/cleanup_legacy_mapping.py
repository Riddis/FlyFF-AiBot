from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


OBSOLETE_FILES = (
    "install_obstacle_vision_v2_1.py",
    "uninstall_obstacle_vision_v2_1.py",
    "summarize_obstacle_dataset.py",
    "review_obstacle_samples.py",
    "prompt_obstacle_label.py",
    "requirements_obstacle_vision.txt",
    "OBSTACLE_VISION_V2_1_DROPIN.md",
    "NATIVE_POSITION_V0_1.md",
    "NATIVE_POSITION_V0_2.md",
    "tools/summarize_native_shadow.py",
    "mapper/AdaptiveMapper.py.before_obstacle_vision_v2_1",
    "tests/test_floor_not_floor_dropin.py",
    "tests/test_native_position_shadow.py",
    "tests/test_summarize_native_shadow.py",
    "train_mapper_policy.py",
    "evaluate_mapper_policy.py",
    "requirements_mapper_rl.txt",
)

OBSOLETE_DIRECTORIES = (
    "mapper/obstacle_vision",
    "datasets/obstacle_vision",
)

VISUAL_MAP_FILES = (
    "map.json",
    "occupancy.npy",
    "visits.npy",
    "map_preview.png",
)

HOOK_BEGIN = "# BEGIN OBSTACLE_VISION_V2_1_DROPIN"
HOOK_END = "# END OBSTACLE_VISION_V2_1_DROPIN"


def _remove_hook(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    start = text.find(HOOK_BEGIN)
    end = text.find(HOOK_END)
    if start < 0 or end < 0 or end < start:
        return False
    end += len(HOOK_END)
    cleaned = text[:start].rstrip() + "\n" + text[end:].lstrip("\r\n")
    path.write_text(cleaned, encoding="utf-8")
    return True


def _archive_visual_maps(root: Path) -> list[Path]:
    maps_root = root / "mapper" / "maps"
    if not maps_root.is_dir():
        return []
    archived: list[Path] = []
    for map_dir in maps_root.iterdir():
        if not map_dir.is_dir() or (map_dir / "coordinate_frame.json").is_file():
            continue
        existing = [map_dir / name for name in VISUAL_MAP_FILES if (map_dir / name).exists()]
        if not existing:
            continue
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        destination = map_dir / f"legacy_visual_map_{stamp}"
        destination.mkdir(parents=True, exist_ok=False)
        for source in existing:
            shutil.move(str(source), str(destination / source.name))
        archived.append(destination)
    return archived


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    removed: list[str] = []

    if _remove_hook(root / "mapper" / "AdaptiveMapper.py"):
        removed.append("legacy obstacle-vision import hook")

    for relative in OBSOLETE_FILES:
        path = root / relative
        if path.is_file():
            path.unlink()
            removed.append(relative)

    for relative in OBSOLETE_DIRECTORIES:
        path = root / relative
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(relative + "/")

    mapping_models = root / "models" / "mapping"
    if mapping_models.is_dir():
        shutil.rmtree(mapping_models)
        mapping_models.mkdir(parents=True, exist_ok=True)
        (mapping_models / ".gitkeep").touch()
        removed.append("obsolete visual-mapper RL model archive")

    archives = _archive_visual_maps(root)

    print("Coordinate mapper cleanup complete.")
    if removed:
        print("Removed:")
        for item in removed:
            print(f"  - {item}")
    if archives:
        print("Archived incompatible visual maps:")
        for archive in archives:
            print(f"  - {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
