from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from time import monotonic, sleep
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from position.IndependentMonsterRediscovery import (  # noqa: E402
    MonsterRediscoveryResult,
    rediscover_known_layout_monsters,
)
from position.IndependentNativeReader import (  # noqa: E402
    IndependentActorSlotRead,
    IndependentNativeReader,
    IndependentNativeSnapshot,
)
from position.MonsterConfig import (  # noqa: E402
    DEFAULT_MONSTER_CONFIG_PATH,
    load_native_monster_config,
)
from position.NativeTraceTargets import discover_trace_targets  # noqa: E402
from position.PointerScanWorkflow import ReadableRegionIndex  # noqa: E402
from position.policy import LIVE_ATTACH_POLICY  # noqa: E402
from position.Win32ProcessMemory import Win32ProcessMemory  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value, 0)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value, 0)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def _monster_hp(value: str) -> tuple[int, int]:
    try:
        species_text, hp_text = value.split("=", 1)
        species = int(species_text, 0)
        hp = int(hp_text, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected SPECIES=HP") from error
    if species <= 0 or hp <= 0:
        raise argparse.ArgumentTypeError("species and HP must be positive")
    return species, hp


def _find_window_handle(title_fragment: str) -> int:
    import win32gui  # type: ignore[import-untyped]

    wanted = title_fragment.strip().casefold()
    matches: list[tuple[int, str]] = []

    def visit(hwnd: int, _extra: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = str(win32gui.GetWindowText(hwnd))
        if wanted in title.casefold():
            matches.append((int(hwnd), title))

    win32gui.EnumWindows(visit, None)
    if not matches:
        raise SystemExit(f"No visible window title contained {title_fragment!r}")
    if len(matches) > 1:
        raise SystemExit("Multiple windows matched; use --window-handle")
    return matches[0][0]


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _state_map(
    snapshot: IndependentNativeSnapshot,
) -> dict[int, IndependentActorSlotRead]:
    return {state.base: state for state in snapshot.actor_states}


def _derive_slot_lifecycle_events(
    previous: dict[int, IndependentActorSlotRead],
    current: dict[int, IndependentActorSlotRead],
    *,
    kill_event_radius: float,
) -> dict[str, list[dict[str, object]]]:
    """Derive lifecycle events from persistent reusable slot state.

    A death is only the confirmed live-HP transition ``> 0 -> 0`` for the same
    target species in the same slot. Generic disappearance, unreadability, an
    active-field mismatch, or a species change is never promoted to a kill.
    """

    deaths: list[dict[str, object]] = []
    spawns: list[dict[str, object]] = []
    reuses: list[dict[str, object]] = []
    for base in sorted(set(previous) | set(current)):
        before = previous.get(base)
        after = current.get(base)
        if before is None or after is None:
            continue
        if before.species > 0 and after.species > 0 and before.species != after.species:
            reuses.append(
                {
                    "base": base,
                    "from_species": before.species,
                    "to_species": after.species,
                    "from_hp": before.hp,
                    "to_hp": after.hp,
                }
            )
        if (
            before.target_species
            and after.target_species
            and before.species == after.species
            and before.hp > 0
            and after.hp == 0
        ):
            deaths.append(
                {
                    "base": base,
                    "species": before.species,
                    "previous_hp": before.hp,
                    "hp": after.hp,
                    "distance_native": before.distance_native,
                    "probable_kill": before.distance_native <= kill_event_radius,
                }
            )
        if (
            after.target_species
            and after.hp > 0
            and (
                before.species <= 0
                or before.hp == 0
                or before.species != after.species
            )
        ):
            spawns.append(
                {
                    "base": base,
                    "species": after.species,
                    "hp": after.hp,
                    "previous_species": before.species,
                    "previous_hp": before.hp,
                }
            )
    return {"deaths": deaths, "spawns": spawns, "reuses": reuses}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover the local player and exact monster cohort once, then "
            "continuously read both without a world pointer or debugger."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid", type=_positive_int)
    target.add_argument("--window-handle", type=_positive_int)
    target.add_argument("--window-title")
    parser.add_argument("--config", type=Path, default=DEFAULT_MONSTER_CONFIG_PATH)
    parser.add_argument("--spawn-x", type=float, default=253.0)
    parser.add_argument("--spawn-z", type=float, default=86.0)
    parser.add_argument("--player-hp", type=_positive_int, required=True)
    parser.add_argument(
        "--current-hp-offset",
        type=_nonnegative_int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--monster-hp",
        type=_monster_hp,
        action="append",
        required=True,
        metavar="SPECIES=HP",
    )
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--sample-seconds", type=float, default=60.0)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help=(
            "Internal monster-state polling interval. Console/report samples remain "
            "controlled by --sample-interval."
        ),
    )
    parser.add_argument(
        "--kill-event-radius",
        type=float,
        default=None,
        help=(
            "Maximum last-known player distance for counting a living-to-nonliving "
            "transition as a probable kill. Defaults to the native reader radius."
        ),
    )
    parser.add_argument("--maximum-scan-mib", type=_positive_int, default=1536)
    parser.add_argument("--object-span", type=_positive_int, default=0x4000)
    parser.add_argument(
        "--slots-each-direction",
        type=_nonnegative_int,
        default=31,
        help=(
            "Expand each exact full-health anchor across its reusable 32-slot "
            "slab so dead, damaged, and differently populated slots are cached."
        ),
    )
    parser.add_argument(
        "--rediscovery-interval",
        type=float,
        default=5.0,
        help=(
            "Seconds between background exact-anchor scans that merge newly "
            "loaded actor slabs. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--rediscovery-timeout",
        type=float,
        default=45.0,
        help="Maximum duration of one background rediscovery scan.",
    )
    parser.add_argument(
        "--rediscovery-chunk-mib",
        type=_positive_int,
        default=4,
        help="Read chunk size for background species scans.",
    )
    parser.add_argument("--vision-radius", type=float, default=250.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APP_ROOT / "training_logs" / "independent_native_reader",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hwnd = (
        _find_window_handle(args.window_title)
        if args.window_title is not None
        else args.window_handle
    )
    if args.pid is not None:
        memory = Win32ProcessMemory(args.pid)
    else:
        memory = Win32ProcessMemory.from_window_handle(int(hwnd))
    report_path = args.output_dir / f"independent_native_reader_{_timestamp()}.json"
    report: dict[str, Any] = {
        "schema_version": 6,
        "status": "starting",
        "samples": [],
        "rediscovery_history": [],
        "command": sys.argv,
    }
    started = monotonic()
    deadline = started + args.timeout
    try:
        config = load_native_monster_config(args.config)
        module = memory.module_info(config.module_name)
        regions = memory.readable_regions(
            maximum_address=config.maximum_scan_address,
            private_only=False,
        )
        readable = ReadableRegionIndex.build(regions)
        print(
            f"Attached read-only scanner to pid={memory.pid}; module={module.name}; "
            f"base=0x{module.base_address:X}; size=0x{module.size:X}"
        )
        exact_hp = dict(args.monster_hp)
        print(
            "Exact monster anchors are mandatory: "
            + ", ".join(f"species {s} HP {hp}" for s, hp in exact_hp.items())
        )

        def check() -> None:
            if monotonic() >= deadline:
                raise TimeoutError("independent reader discovery timed out")

        def progress(bytes_scanned: int, regions_scanned: int, counts: dict[str, int]) -> None:
            print(
                f"Discovery scan: {bytes_scanned / (1 << 20):.0f} MiB; "
                f"regions={regions_scanned}; species_hits={counts.get('species_hits', 0)}, "
                f"spawn_x={counts.get('spawn_x', 0)}"
            )

        discovery = discover_trace_targets(
            memory,
            regions=regions,
            readable=readable,
            module=module,
            species_hp=exact_hp,
            spawn_x=args.spawn_x,
            spawn_z=args.spawn_z,
            player_hp=args.player_hp,
            species_offset=config.species_offset,
            active_species_offset=config.active_species_offset,
            hp_offset=config.hp_offset,
            x_offset=config.x_offset,
            y_offset=config.y_offset,
            z_offset=config.z_offset,
            self_pointer_offset=config.self_pointer_offset,
            coordinate_limit=config.maximum_absolute_coordinate,
            object_span=args.object_span,
            maximum_scan_bytes=args.maximum_scan_mib << 20,
            check=check,
            progress=progress,
            attach_policy=LIVE_ATTACH_POLICY,
        )
        report["discovery"] = discovery.to_dict()
        if discovery.outcome != "success" or discovery.player is None:
            report["status"] = f"discovery_{discovery.outcome}"
            _atomic_json(report_path, report)
            print(f"Discovery failed: {discovery.outcome}: {discovery.message}")
            print(f"Saved report: {report_path}")
            return 2

        reader = IndependentNativeReader(
            memory,
            module,
            discovery,
            configured_player_offset=config.player_pointer_offset,
            monster_current_hp_offset=config.hp_offset,
            monster_active_species_offset=config.active_species_offset,
            current_hp_offset=args.current_hp_offset,
            expected_full_hp_by_species=exact_hp,
            object_span=args.object_span,
            slots_each_direction=args.slots_each_direction,
        )
        report["reader"] = {
            "selected_player_slot": reader.selected_player_slot,
            "player_slot_offsets": [
                slot - module.base_address for slot in reader.player_slots
            ],
            "actor_stride": reader.actor_stride,
            "actor_slots": list(reader.actor_slots),
            "player_hp_offset": reader.player_hp_offset,
            "monster_live_hp_offset": reader.monster_hp_offset,
            "configured_monster_hp_candidate": reader.configured_monster_hp_offset,
            "configured_monster_hp_anchor_matches": reader.configured_hp_matches,
            "configured_monster_hp_anchor_samples": reader.configured_hp_samples,
            "monster_active_species_candidate": reader.monster_active_species_offset,
            "active_species_matches": reader.active_species_matches,
            "active_species_samples": reader.active_species_samples,
            "active_species_reliable": reader.active_species_reliable,
            "slots_each_direction": args.slots_each_direction,
            "deprecated_current_hp_offset_argument": reader.deprecated_current_hp_offset,
            "discovery_active_species_is_cross_slot_alias": (
                reader.active_species_is_cross_slot_alias
            ),
        }
        print(
            f"Independent player pointer: {module.name}+0x"
            f"{reader.selected_player_slot - module.base_address:X}; "
            f"fallback_aliases={len(reader.player_slots) - 1}"
        )
        print(
            f"Independent monster cache: slots={len(reader.actor_slots)}; "
            f"stride={None if reader.actor_stride is None else f'0x{reader.actor_stride:X}'}"
        )
        print(
            f"Player live HP: +0x{reader.player_hp_offset:X}; "
            f"monster live HP / exact full-health discovery field: "
            f"+0x{reader.monster_hp_offset:X}"
        )
        print(
            f"Configured HP candidate +0x{reader.configured_monster_hp_offset:X}: "
            f"exact-anchor support={reader.configured_hp_matches}/"
            f"{reader.configured_hp_samples}; configured active candidate "
            f"+0x{reader.monster_active_species_offset:X}: species support="
            f"{reader.active_species_matches}/{reader.active_species_samples}; "
            f"active gate={'enabled' if reader.active_species_reliable else 'disabled'}"
        )
        if args.current_hp_offset is not None:
            print(
                f"Deprecated --current-hp-offset 0x{args.current_hp_offset:X} "
                "was supplied. The configured monster layout takes precedence."
            )
        poll_interval = max(0.01, float(args.poll_interval))
        output_interval = max(poll_interval, float(args.sample_interval))
        kill_event_radius = (
            float(config.vision_radius_native)
            if args.kill_event_radius is None
            else max(0.0, float(args.kill_event_radius))
        )
        rediscovery_interval = max(0.0, float(args.rediscovery_interval))
        rediscovery_timeout = max(1.0, float(args.rediscovery_timeout))
        report["sampling_config"] = {
            "poll_interval_seconds": poll_interval,
            "output_interval_seconds": output_interval,
            "kill_event_radius_native": kill_event_radius,
            "rediscovery_interval_seconds": rediscovery_interval,
            "rediscovery_timeout_seconds": rediscovery_timeout,
            "rediscovery_chunk_mib": args.rediscovery_chunk_mib,
        }
        print(
            "Continuous read started. Monster state is polled every "
            f"{poll_interval:.3f}s; console/report output remains every "
            f"{output_interval:.3f}s."
        )
        if rediscovery_interval > 0:
            print(
                "Background slab rediscovery enabled every "
                f"{rediscovery_interval:.1f}s; polling continues during scans."
            )
        report["status"] = "sampling"
        _atomic_json(report_path, report)
        sample_stop = monotonic() + max(0.1, args.sample_seconds)
        next_output = monotonic()
        sample_index = 0
        allowed = set(exact_hp)
        previous_states: dict[int, IndependentActorSlotRead] | None = None
        pending_deaths: dict[tuple[int, int], dict[str, object]] = {}
        pending_spawns: dict[tuple[int, int], dict[str, object]] = {}
        pending_reuses: dict[tuple[int, int, int], dict[str, object]] = {}
        cumulative_probable_kills = 0
        rediscovery_queue: Queue[tuple[str, object]] = Queue()
        rediscovery_thread: Thread | None = None
        rediscovery_index = 0
        next_rediscovery = monotonic() + rediscovery_interval

        def run_rediscovery() -> None:
            scan_memory = Win32ProcessMemory(memory.pid)
            try:
                result = rediscover_known_layout_monsters(
                    scan_memory,
                    template=reader.monster_targets[0],
                    species_hp=exact_hp,
                    maximum_address=config.maximum_scan_address,
                    coordinate_limit=config.maximum_absolute_coordinate,
                    chunk_size=args.rediscovery_chunk_mib << 20,
                    deadline=monotonic() + rediscovery_timeout,
                )
                rediscovery_queue.put(("success", result))
            except Exception as error:
                rediscovery_queue.put(
                    ("error", f"{type(error).__name__}: {error}")
                )
            finally:
                scan_memory.close()

        while monotonic() < sample_stop:
            now = monotonic()
            try:
                while True:
                    status, payload = rediscovery_queue.get_nowait()
                    rediscovery_index += 1
                    if status == "success":
                        result = payload
                        assert isinstance(result, MonsterRediscoveryResult)
                        merge = reader.merge_monster_targets(
                            result.targets,
                            slots_each_direction=args.slots_each_direction,
                        )
                        record = {
                            "index": rediscovery_index,
                            "status": "success",
                            "elapsed_seconds": monotonic() - started,
                            "scan": result.evidence.to_dict(),
                            "merge": merge.to_dict(),
                        }
                        report["rediscovery_history"].append(record)
                        report["reader"]["actor_slots"] = list(reader.actor_slots)
                        report["reader"]["rediscovered_anchor_count"] = len(
                            reader.monster_targets
                        )
                        print(
                            f"Rediscovery {rediscovery_index}: exact_anchors="
                            f"{result.evidence.exact_full_hp_anchors}; "
                            f"new_anchors={merge.new_anchors}; "
                            f"new_slots={merge.new_slots}; cached={merge.total_slots}; "
                            f"scan={result.evidence.elapsed_seconds:.2f}s"
                        )
                    else:
                        report["rediscovery_history"].append(
                            {
                                "index": rediscovery_index,
                                "status": "error",
                                "elapsed_seconds": monotonic() - started,
                                "error": str(payload),
                            }
                        )
                        print(f"Rediscovery {rediscovery_index} failed: {payload}")
                    rediscovery_thread = None
                    next_rediscovery = monotonic() + rediscovery_interval
            except Empty:
                pass

            if (
                rediscovery_interval > 0
                and (rediscovery_thread is None or not rediscovery_thread.is_alive())
                and now >= next_rediscovery
            ):
                rediscovery_thread = Thread(
                    target=run_rediscovery,
                    name="native-monster-rediscovery",
                    daemon=True,
                )
                rediscovery_thread.start()
                next_rediscovery = float("inf")

            snapshot = reader.snapshot(
                allowed_species=allowed,
                vision_radius_native=args.vision_radius,
            )
            current_states = _state_map(snapshot)
            if previous_states is not None:
                events = _derive_slot_lifecycle_events(
                    previous_states,
                    current_states,
                    kill_event_radius=kill_event_radius,
                )
                for event in events["deaths"]:
                    pending_deaths[(int(event["base"]), int(event["species"]))] = event
                for event in events["spawns"]:
                    pending_spawns[(int(event["base"]), int(event["species"]))] = event
                for event in events["reuses"]:
                    pending_reuses[(
                        int(event["base"]),
                        int(event["from_species"]),
                        int(event["to_species"]),
                    )] = event
            previous_states = current_states

            now = monotonic()
            if now >= next_output or now + poll_interval >= sample_stop:
                sample = snapshot.to_dict()
                sample["elapsed_seconds"] = now - started
                probable_kills = [
                    event
                    for event in pending_deaths.values()
                    if bool(event["probable_kill"])
                ]
                sample["transitions"] = {
                    "deaths": list(pending_deaths.values()),
                    "spawns": list(pending_spawns.values()),
                    "slot_reuses": list(pending_reuses.values()),
                    "probable_kills": probable_kills,
                    # Compatibility summaries now derive only from explicit HP
                    # lifecycle events, never generic disappearance.
                    "became_nonliving": sorted(
                        int(event["base"]) for event in pending_deaths.values()
                    ),
                    "became_living": sorted(
                        int(event["base"]) for event in pending_spawns.values()
                    ),
                }
                cumulative_probable_kills += len(probable_kills)
                sample["cumulative_probable_kills"] = cumulative_probable_kills
                report["samples"].append(sample)
                sample_index += 1
                print(
                    f"Sample {sample_index}: player=({snapshot.player.x:.3f}, "
                    f"{snapshot.player.y:.3f}, {snapshot.player.z:.3f}) "
                    f"hp={snapshot.player.hp}; "
                    f"living={snapshot.living_monsters}; "
                    f"visible={snapshot.visible_living_monsters}; "
                    f"damaged={snapshot.damaged_monsters}; "
                    f"zero_hp={snapshot.zero_hp_monsters}; "
                    f"other_species={snapshot.other_species_slots}; "
                    f"empty={snapshot.empty_slots}; "
                    f"active_mismatch={snapshot.active_field_mismatches}; "
                    f"cached={snapshot.cached_actor_slots}; "
                    f"deaths=+{len(pending_deaths)}; "
                    f"spawns=+{len(pending_spawns)}; "
                    f"reuses=+{len(pending_reuses)}; "
                    f"probable_kills=+{len(probable_kills)} "
                    f"(total={cumulative_probable_kills})"
                )
                _atomic_json(report_path, report)
                pending_deaths.clear()
                pending_spawns.clear()
                pending_reuses.clear()
                next_output = now + output_interval
            sleep(poll_interval)

        report["status"] = "success"
        _atomic_json(report_path, report)
        print(f"Independent read completed; saved report: {report_path}")
        return 0
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        _atomic_json(report_path, report)
        print(f"Independent read interrupted; saved report: {report_path}")
        return 130
    except Exception as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        _atomic_json(report_path, report)
        print(f"Independent reader failed: {type(error).__name__}: {error}")
        print(f"Saved report: {report_path}")
        return 1
    finally:
        memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
