from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Two directories deep under the repository root (devtools/native/);
# direct invocation sets sys.path[0] to this file's own directory, not
# the repository root, so the packages below need it added explicitly
# (same bootstrap pattern devtools/native/probe_native_position.py
# already uses).
APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from assets.Assets import MobInfo  # noqa: E402
from position import (  # noqa: E402
    create_native_monster_provider,
    create_native_monster_provider_from_process_id,
)
from utils.helpers import get_window_handlers  # noqa: E402


def _parse_int(value: str) -> int:
    return int(value, 0)


def _resolve_window(args: argparse.Namespace) -> tuple[str, int]:
    windows = get_window_handlers()
    if args.hwnd is not None:
        for title, handle in windows.items():
            if int(handle) == args.hwnd:
                return title, int(handle)
        return f"HWND 0x{args.hwnd:X}", args.hwnd

    needle = (args.window_title or "Flyff").casefold()
    matches = [
        (title, int(handle))
        for title, handle in windows.items()
        if needle in title.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(
            f"No visible window title contains {args.window_title or 'Flyff'!r}. "
            "Use --window-title or --hwnd."
        )
    lines = "\n".join(
        f"  0x{handle:X}  {title}" for title, handle in sorted(matches)
    )
    raise SystemExit(
        "More than one window matched. Choose one with --hwnd:\n" + lines
    )


def _registered_species() -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for name, entry in MobInfo.get_all_mobs().items():
        if not isinstance(entry, dict):
            continue
        raw = entry.get("species_id")
        if isinstance(raw, bool):
            continue
        try:
            species_id = int(raw)
        except (TypeError, ValueError):
            continue
        result.setdefault(species_id, []).append(str(name))
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect FlyFF native actor slots without writing process memory."
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--window-title", help="case-insensitive title substring")
    target.add_argument("--hwnd", type=_parse_int, help="window handle, e.g. 0x123456")
    target.add_argument("--pid", type=int, help="Neuz.exe process ID")
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="native-unit radius around the player (default from config)",
    )
    parser.add_argument(
        "--all-species",
        action="store_true",
        help="include every active actor species instead of registered mobs only",
    )
    parser.add_argument(
        "--force-rediscovery",
        action="store_true",
        help="ignore the provider's actor-slot cache",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON",
    )
    parser.add_argument(
        "--capture-target",
        action="store_true",
        help="print only the currently targeted actor and its species ID",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="maximum actor rows to print in text mode; use 0 for all",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.pid is not None:
        title = f"Neuz.exe PID {args.pid}"
        hwnd = None
        provider = create_native_monster_provider_from_process_id(args.pid)
    else:
        title, hwnd = _resolve_window(args)
        provider = create_native_monster_provider(hwnd)
    if provider is None:
        raise SystemExit("Native monster reader is disabled in position/native_monsters.json")

    try:
        if args.capture_target:
            actor = provider.capture_selected_actor()
            payload = {
                "window": title,
                "hwnd": hwnd,
                "pid": args.pid,
                "base_address": f"0x{actor.base_address:08X}",
                "species_id": actor.species_id,
                "hp": actor.hp,
                "x": actor.x,
                "y": actor.y,
                "z": actor.z,
                "distance_native": actor.distance_native,
            }
            print(json.dumps(payload, indent=2))
            return 0

        registered = _registered_species()
        allowed = None if args.all_species else set(registered)
        if allowed == set():
            raise SystemExit(
                "No mobs have a captured species_id yet. Use the GUI Add mob capture "
                "button, or add --all-species for diagnostics."
            )
        actors = provider.read_active_actors(
            allowed_species_ids=allowed,
            vision_radius_native=args.radius,
            force_rediscovery=args.force_rediscovery,
        )
        diagnostics = provider.last_diagnostics

        if args.json:
            output = {
                "window": title,
                "hwnd": hwnd,
                "pid": args.pid,
                "radius_native": (
                    provider.config.vision_radius_native
                    if args.radius is None
                    else args.radius
                ),
                "actors": [
                    {
                        "base_address": f"0x{actor.base_address:08X}",
                        "species_id": actor.species_id,
                        "names": registered.get(actor.species_id, []),
                        "hp": actor.hp,
                        "x": actor.x,
                        "y": actor.y,
                        "z": actor.z,
                        "distance_native": actor.distance_native,
                    }
                    for actor in actors
                ],
                "diagnostics": (
                    None
                    if diagnostics is None
                    else {
                        name: getattr(diagnostics, name)
                        for name in diagnostics.__dataclass_fields__
                    }
                ),
            }
            print(json.dumps(output, indent=2))
            return 0

        radius = (
            provider.config.vision_radius_native
            if args.radius is None
            else args.radius
        )
        if hwnd is None:
            print(f"Process: {title}")
        else:
            print(f"Window: {title} (0x{hwnd:X})")
        print(f"Vision radius: {radius:.1f} native units")
        print(f"Map scale equivalent: {radius / 1.6:.1f} cells at 1.6 units/cell")
        print(f"Matching active actors: {len(actors)}")
        print("")
        species_counts = Counter(actor.species_id for actor in actors)
        if species_counts:
            print("Species counts:")
            for species_id, count in sorted(species_counts.items()):
                names = ", ".join(registered.get(species_id, [])) or "unregistered"
                print(f"  id={species_id:<5d} count={count:<4d} {names}")
            print("")

        row_limit = max(0, int(args.limit))
        displayed = actors if row_limit == 0 else actors[:row_limit]
        for actor in displayed:
            names = ", ".join(registered.get(actor.species_id, [])) or "unregistered"
            print(
                f"0x{actor.base_address:08X}  id={actor.species_id:<5d} "
                f"hp={actor.hp:<10d} distance={actor.distance_native:7.2f} "
                f"x={actor.x:9.2f} y={actor.y:8.2f} z={actor.z:9.2f} "
                f"{names}"
            )
        if row_limit > 0 and len(actors) > row_limit:
            print(f"... {len(actors) - row_limit} additional actors omitted")
        if diagnostics is not None:
            print("")
            print(
                "Discovery: "
                f"regions={diagnostics.discovery_regions_read}/"
                f"{diagnostics.discovery_regions_considered}, "
                f"read={diagnostics.discovery_bytes_read / (1024 * 1024):.1f} MiB, "
                f"time={diagnostics.discovery_elapsed_seconds:.3f}s, "
                f"world_matches={diagnostics.world_pointer_matches}, "
                f"invalid_matches={diagnostics.rejected_invalid_actor}, "
                f"read_failures={diagnostics.discovery_read_failures}"
            )
            print(
                "Actors: "
                f"slots={diagnostics.discovered_slots}, "
                f"wrong_world={diagnostics.rejected_wrong_world}, "
                f"inactive={diagnostics.rejected_not_present}, "
                f"dead={diagnostics.rejected_dead}, "
                f"species={diagnostics.rejected_species}, "
                f"outside_radius={diagnostics.rejected_distance}, "
                f"unreadable={diagnostics.unreadable_cached_slots}"
            )
        return 0
    finally:
        provider.close()


if __name__ == "__main__":
    sys.exit(main())
