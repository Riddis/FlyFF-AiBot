from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
import sys
from pathlib import Path
from time import monotonic

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from position.AggregateMonsterRootScan import (  # noqa: E402
    save_aggregate_report,
    scan_aggregate_monster_roots,
)
from position.AutonomousPointerSelection import (  # noqa: E402
    historical_direct_offset_counts,
    load_matching_snapshot_history,
    prove_player_and_rank_direct_slots,
)
from position.AnchoredPointerDiscovery import (  # noqa: E402
    AnchoredPlayerObservation,
    PointerRecoveryHints,
    discover_anchored_pointer_candidate,
)
from position.MonsterConfig import (  # noqa: E402
    DEFAULT_MONSTER_CONFIG_PATH,
    load_native_monster_config,
)
from position.PointerScanWorkflow import (  # noqa: E402
    ReadableRegionIndex,
    load_compatible_snapshots,
    make_snapshot,
    rank_stable_paths,
    resolve_pointer_path,
    save_snapshot,
    scan_module_pointer_index,
    scan_module_rooted_paths,
    wait_for_player_movement,
)
from position.Win32ProcessMemory import Win32ProcessMemory  # noqa: E402


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a decimal integer or 0x-prefixed hexadecimal value"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a decimal integer or 0x-prefixed hexadecimal value"
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def _monster_hp(value: str) -> tuple[int, int]:
    try:
        species_text, hp_text = value.split("=", 1)
        species = int(species_text, 0)
        hp = int(hp_text, 0)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "expected SPECIES=HP, for example 944=400236"
        ) from error
    if species <= 0 or hp <= 0:
        raise argparse.ArgumentTypeError("species and HP must be positive")
    return species, hp


def _find_window_handle(title_fragment: str) -> int:
    try:
        import win32gui  # type: ignore[import-untyped]
    except ImportError as error:
        raise SystemExit(
            "--window-title requires pywin32; use --pid or --window-handle instead"
        ) from error

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
        details = "\n".join(
            f"  0x{hwnd:X}: {title}" for hwnd, title in matches[:20]
        )
        raise SystemExit(
            "Multiple windows matched; pass --window-handle for one of:\n"
            f"{details}"
        )
    return matches[0][0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automate the read-only Cheat Engine workflow used to recover the "
            "FlyFF local-player pointer: anchor the live object, prove it with "
            "passive structural samples, enumerate module roots/chains, and "
            "intersect normalized paths across relogs or client restarts. "
            "Movement is an optional diagnostic, not an acceptance requirement."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid", type=_positive_int, help="FlyFF process ID")
    target.add_argument(
        "--window-handle",
        type=_positive_int,
        help="FlyFF HWND in decimal or 0x-prefixed hexadecimal",
    )
    target.add_argument(
        "--window-title",
        help="unique case-insensitive fragment of the visible FlyFF title",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MONSTER_CONFIG_PATH,
        help="native_monsters.json used for search bounds and layout deltas",
    )
    parser.add_argument(
        "--species",
        type=_positive_int,
        action="append",
        default=[],
        help="known active monster species; repeat as needed (default: 944)",
    )
    parser.add_argument("--spawn-x", type=float, default=253.0)
    parser.add_argument("--spawn-z", type=float, default=86.0)
    parser.add_argument(
        "--player-hp",
        type=_positive_int,
        required=True,
        help="exact current player HP shown in the status panel",
    )
    parser.add_argument(
        "--player-max-hp",
        type=_positive_int,
        help="exact maximum player HP; defaults to --player-hp",
    )
    parser.add_argument(
        "--monster-hp",
        type=_monster_hp,
        action="append",
        required=True,
        metavar="SPECIES=HP",
        help=(
            "required exact full-health monster anchor; repeat once for every "
            "requested species, for example --monster-hp 944=400236"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="overall discovery deadline in seconds",
    )
    parser.add_argument(
        "--maximum-scan-mib",
        type=_positive_int,
        default=1536,
        help="maximum private memory read by the anchor scan",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=APP_ROOT / "training_logs" / "pointer_scans",
        help="directory retaining normalized snapshots across runs",
    )
    parser.add_argument(
        "--passive-samples",
        type=_positive_int,
        default=24,
        help="passive player/slot samples used for autonomous acceptance",
    )
    parser.add_argument(
        "--passive-interval",
        type=float,
        default=0.10,
        help="seconds between passive samples",
    )
    parser.add_argument(
        "--require-movement",
        action="store_true",
        help="add controlled movement confirmation after passive acceptance",
    )
    parser.add_argument(
        "--skip-movement",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--movement-timeout",
        type=float,
        default=180.0,
        help="seconds to wait for movement and a stable stop",
    )
    parser.add_argument(
        "--movement-minimum",
        type=float,
        default=0.5,
        help="minimum native X/Z displacement required",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="skip aggregate monster-cohort world/manager discovery",
    )
    parser.add_argument(
        "--aggregate-seconds",
        type=float,
        default=45.0,
        help=(
            "seconds to sample the whole monster cohort; ordinary farming during "
            "this window strengthens the differential signal"
        ),
    )
    parser.add_argument(
        "--aggregate-interval",
        type=float,
        default=1.0,
        help="seconds between aggregate cohort samples",
    )
    parser.add_argument(
        "--aggregate-object-span",
        type=_positive_int,
        default=0x4000,
        help="bytes inspected in each validated monster actor",
    )
    parser.add_argument(
        "--aggregate-holder-span",
        type=_positive_int,
        default=0x4000,
        help="bytes inspected in each module-rooted holder candidate",
    )
    parser.add_argument(
        "--aggregate-min-support",
        type=_positive_int,
        default=4,
        help="minimum distinct cohort members supporting one candidate",
    )
    parser.add_argument(
        "--aggregate-maximum-scan-mib",
        type=_positive_int,
        default=1536,
        help="maximum readable memory scanned once for cohort references",
    )
    parser.add_argument(
        "--aggregate-path-candidates",
        type=_non_negative_int,
        default=12,
        help="top aggregate candidates receiving bounded pointer-path scans",
    )
    parser.add_argument(
        "--max-depth",
        type=_non_negative_int,
        default=2,
        help="maximum module-rooted pointer-chain depth",
    )
    parser.add_argument(
        "--field-span",
        type=_positive_int,
        default=0x1000,
        help="bytes inspected per manager/object node",
    )
    parser.add_argument(
        "--maximum-roots",
        type=_positive_int,
        default=12000,
        help="maximum module pointer roots explored",
    )
    parser.add_argument(
        "--maximum-nodes",
        type=_positive_int,
        default=100000,
        help="maximum object nodes explored by the bounded path scan",
    )
    return parser


def _save_anchor_report(
    *,
    result: object,
    directory: Path,
    memory: object,
    module: object,
    species: tuple[int, ...],
    monster_hp: tuple[tuple[int, int], ...],
    player_hp: int,
    player_max_hp: int,
    spawn_x: float,
    spawn_z: float,
) -> Path:
    """Persist the completed anchor result before any acceptance gate exits."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now(UTC)
    stamp = captured_at.strftime("%Y%m%dT%H%M%S_%fZ")
    path = destination / f"pointer_anchor_scan_{stamp}.json"
    payload = {
        "schema_version": 1,
        "captured_at_utc": captured_at.isoformat(),
        "pid": int(getattr(memory, "pid")),
        "module": {
            "name": str(getattr(module, "name")),
            "path": str(getattr(module, "path")),
            "base": int(getattr(module, "base_address")),
            "size": int(getattr(module, "size")),
        },
        "request": {
            "species": list(species),
            "spawn_x": float(spawn_x),
            "spawn_z": float(spawn_z),
            "player_hp": int(player_hp),
            "player_max_hp": int(player_max_hp),
            "monster_hp": [
                {"species": int(item_species), "hp": int(hp)}
                for item_species, hp in monster_hp
            ],
            "monster_hp_filter_mode": "exact",
        },
        "result": asdict(result),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _select_observation(
    outcome: str,
    observations: tuple[AnchoredPlayerObservation, ...],
) -> AnchoredPlayerObservation:
    unique = {item.player_base: item for item in observations}
    if len(unique) == 1:
        return next(iter(unique.values()))
    if not unique:
        raise SystemExit(
            f"Anchor outcome {outcome!r} did not expose a stable player object."
        )
    details = "\n".join(
        f"  player=0x{item.player_base:X}, direct_slots={len(item.direct_module_slots)}"
        for item in unique.values()
    )
    raise SystemExit(
        "Multiple stable spawn/HP player objects remained; no automatic choice "
        f"was made:\n{details}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be positive")
    if args.movement_timeout <= 0.0:
        raise SystemExit("--movement-timeout must be positive")
    if args.movement_minimum <= 0.0:
        raise SystemExit("--movement-minimum must be positive")
    if args.passive_interval < 0.0:
        raise SystemExit("--passive-interval cannot be negative")
    if args.field_span % 4:
        raise SystemExit("--field-span must be divisible by four")
    if args.aggregate_seconds < 0.0:
        raise SystemExit("--aggregate-seconds cannot be negative")
    if args.aggregate_interval <= 0.0:
        raise SystemExit("--aggregate-interval must be positive")
    if args.aggregate_object_span % 4:
        raise SystemExit("--aggregate-object-span must be divisible by four")
    if args.aggregate_holder_span % 4:
        raise SystemExit("--aggregate-holder-span must be divisible by four")
    if args.aggregate_min_support < 2:
        raise SystemExit("--aggregate-min-support must be at least two")

    config = load_native_monster_config(args.config)
    if args.pid is not None:
        memory = Win32ProcessMemory(args.pid)
    elif args.window_handle is not None:
        memory = Win32ProcessMemory.from_window_handle(args.window_handle)
    else:
        memory = Win32ProcessMemory.from_window_handle(
            _find_window_handle(args.window_title)
        )

    deadline = monotonic() + float(args.timeout)

    def check() -> None:
        if monotonic() >= deadline:
            raise TimeoutError("pointer workflow exceeded --timeout")

    try:
        module = memory.module_info(config.module_name)
        regions = memory.readable_regions(
            maximum_address=config.maximum_scan_address,
            private_only=False,
        )
        readable = ReadableRegionIndex.build(regions)
        print(
            f"Attached pid={memory.pid}; module={module.name}; "
            f"base=0x{module.base_address:X}; size=0x{module.size:X}",
            flush=True,
        )
        print(
            f"Indexed {len(readable.starts)} merged readable intervals.",
            flush=True,
        )

        module_refs = scan_module_pointer_index(
            memory,
            module,
            readable,
            chunk_size=config.discovery_chunk_bytes,
            check=check,
        )
        module_slots = sum(len(slots) for slots in module_refs.values())
        print(
            f"Indexed {module_slots} module pointer slots across "
            f"{len(module_refs)} readable targets.",
            flush=True,
        )

        species = tuple(dict.fromkeys(args.species or [944]))
        maximum_hp = args.player_max_hp or args.player_hp
        monster_hp = tuple(dict(args.monster_hp).items())
        monster_hp_map = dict(monster_hp)
        missing_monster_hp = tuple(
            item_species
            for item_species in species
            if item_species not in monster_hp_map
        )
        if missing_monster_hp:
            missing = ", ".join(str(value) for value in missing_monster_hp)
            raise SystemExit(
                "Every requested --species requires an exact --monster-hp "
                f"anchor; missing species: {missing}"
            )
        print(
            "Monster HP hard anchors (exact): "
            + ", ".join(
                f"species {item_species}={hp}"
                for item_species, hp in monster_hp
            ),
            flush=True,
        )
        hints = PointerRecoveryHints(
            known_species_ids=species,
            player_spawn_x=args.spawn_x,
            player_spawn_z=args.spawn_z,
            player_current_hp=args.player_hp,
            player_max_hp=maximum_hp,
            monster_hp_by_species=monster_hp,
            require_exact_monster_hp=True,
            movement_minimum_native=args.movement_minimum,
        )

        def progress(evidence: object) -> None:
            scanned = int(getattr(evidence, "anchor_bytes_scanned"))
            species_hits = int(getattr(evidence, "species_value_matches"))
            spawn_hits = int(getattr(evidence, "spawn_x_matches"))
            print(
                f"Anchor scan: {scanned / (1 << 20):.0f} MiB; "
                f"species_hits={species_hits}; spawn_hits={spawn_hits}",
                flush=True,
            )

        result = discover_anchored_pointer_candidate(
            memory,
            regions=regions,
            slot_refs=module_refs,
            hints=hints,
            module_base=module.base_address,
            module_stop=module.base_address + module.size,
            configured_player_slot=(
                module.base_address + config.player_pointer_offset
            ),
            configured_world_slot=(
                module.base_address + config.world_pointer_offset
            ),
            species_offset=config.species_offset,
            active_species_offset=config.active_species_offset,
            hp_offset=config.hp_offset,
            x_offset=config.x_offset,
            y_offset=config.y_offset,
            z_offset=config.z_offset,
            configured_world_field_offset=config.world_offset,
            configured_self_offset=config.self_pointer_offset,
            coordinate_limit=config.maximum_absolute_coordinate,
            maximum_address=config.maximum_scan_address,
            chunk_size=config.discovery_chunk_bytes,
            cancellation=None,
            deadline=deadline,
            readable_contains=readable.contains,
            check=check,
            maximum_scan_bytes=args.maximum_scan_mib << 20,
            progress_callback=progress,
        )
        print(f"Anchor outcome: {result.outcome}: {result.message}", flush=True)
        anchor_report_path = _save_anchor_report(
            result=result,
            directory=args.history_dir,
            memory=memory,
            module=module,
            species=species,
            monster_hp=monster_hp,
            player_hp=args.player_hp,
            player_max_hp=maximum_hp,
            spawn_x=args.spawn_x,
            spawn_z=args.spawn_z,
        )
        print(f"Saved anchor report: {anchor_report_path}", flush=True)
        observation = _select_observation(result.outcome, result.unlinked_players)
        direct_offsets = tuple(
            slot - module.base_address
            for slot in observation.direct_module_slots
            if module.base_address <= slot < module.base_address + module.size
        )
        print(
            f"Stable player object: 0x{observation.player_base:X}; "
            f"position=({observation.x:.3f}, {observation.y:.3f}, "
            f"{observation.z:.3f}); HP={observation.current_hp}/"
            f"{observation.maximum_hp}",
            flush=True,
        )
        if direct_offsets:
            print(
                "Direct module aliases: "
                + ", ".join(
                    f"{module.name}+0x{offset:X}" for offset in direct_offsets
                ),
                flush=True,
            )
        history = load_matching_snapshot_history(args.history_dir, module)
        history_counts = historical_direct_offset_counts(history)
        passive = prove_player_and_rank_direct_slots(
            memory,
            observation,
            module,
            regions,
            configured_player_offset=config.player_pointer_offset,
            historical_sessions_by_offset=history_counts,
            samples=args.passive_samples,
            interval_seconds=args.passive_interval,
            maximum_absolute_coordinate=config.maximum_absolute_coordinate,
            check=check,
        )
        print(f"Passive proof: {passive.message}", flush=True)
        if passive.selected_slot is not None:
            selected = passive.selected_slot
            print(
                "Autonomous player pointer: "
                f"{module.name}+0x{selected.module_offset:X}; "
                f"writable={selected.writable}; "
                f"history={selected.historical_sessions}; "
                f"fallbacks={len(passive.fallback_slots)}",
                flush=True,
            )
        if not passive.accepted:
            raise SystemExit(
                "Passive player proof did not reach the autonomous acceptance gate."
            )

        from position.PointerScanWorkflow import MovementEvidence

        if args.require_movement and not args.skip_movement:
            print(
                "Optional diagnostic: move the character at least "
                f"{args.movement_minimum:.3f} native units, then stop.",
                flush=True,
            )
            movement = wait_for_player_movement(
                memory,
                observation,
                minimum_distance_native=args.movement_minimum,
                timeout_seconds=args.movement_timeout,
                check=check,
            )
        else:
            movement = MovementEvidence(
                confirmed=False,
                distance_native=passive.coordinate_change_native,
                final_x=observation.x,
                final_y=observation.y,
                final_z=observation.z,
                samples=passive.total_samples,
                message=(
                    "Controlled movement was not required; the player and direct "
                    "module slot passed passive structural validation."
                ),
            )
        print(f"Movement diagnostic: {movement.message}", flush=True)

        print(
            f"Scanning bounded module-rooted paths (depth={args.max_depth}, "
            f"field_span=0x{args.field_span:X})...",
            flush=True,
        )
        paths = scan_module_rooted_paths(
            memory,
            module,
            readable,
            module_refs,
            observation.player_base,
            configured_root_offsets=(
                config.player_pointer_offset,
                config.world_pointer_offset,
            ),
            maximum_depth=args.max_depth,
            field_span=args.field_span,
            maximum_roots=args.maximum_roots,
            maximum_nodes=args.maximum_nodes,
            check=check,
        )
        verified_paths = tuple(
            path
            for path in paths
            if resolve_pointer_path(memory, module.base_address, path)
            == observation.player_base
        )
        print(
            f"Found {len(verified_paths)} currently resolving module-rooted path(s).",
            flush=True,
        )
        for path in verified_paths[:40]:
            chain = "".join(f" -> +0x{offset:X}" for offset in path.field_offsets)
            print(
                f"  {module.name}+0x{path.root_module_offset:X}{chain}",
                flush=True,
            )
        if len(verified_paths) > 40:
            print(f"  ... {len(verified_paths) - 40} more saved to JSON", flush=True)

        snapshot = make_snapshot(
            memory=memory,
            module=module,
            observation=observation,
            movement=movement,
            paths=verified_paths,
            anchor_outcome=result.outcome,
            monster_hp_anchors=monster_hp,
        )
        destination = save_snapshot(snapshot, args.history_dir)
        print(f"Saved snapshot: {destination}", flush=True)

        compatible = load_compatible_snapshots(args.history_dir, snapshot)
        ranked = rank_stable_paths(compatible)
        if len(compatible) < 2:
            print(
                "One capture is not enough to call an address restart-stable. "
                "Relog or restart the client, return to the same spawn, and run "
                "this command again; history intersection is automatic.",
                flush=True,
            )
        else:
            print(
                f"Compared {len(compatible)} compatible capture(s). Top stable paths:",
                flush=True,
            )
            for item in ranked[:20]:
                chain = "".join(
                    f" -> +0x{offset:X}" for offset in item.path.field_offsets
                )
                marker = "STABLE" if item.stable_across_all_sessions else "partial"
                print(
                    f"  [{marker} {item.sessions_present}/{item.total_sessions}; "
                    f"movement {item.movement_confirmed_sessions}/"
                    f"{item.total_sessions}] {module.name}+0x"
                    f"{item.path.root_module_offset:X}{chain}",
                    flush=True,
                )

        if not args.skip_aggregate:
            if len(result.monster_cohort) < 2:
                print(
                    "Aggregate scan skipped: the anchor stage did not expose a "
                    "validated monster cohort.",
                    flush=True,
                )
            else:
                print(
                    "Preparing aggregate monster-cohort scan. Keep the nearby "
                    "cohort mostly untouched during the one-time reference scan; "
                    "the tool will print when the combat sampling window starts.",
                    flush=True,
                )

                def aggregate_progress(bytes_read: int, matches: int) -> None:
                    print(
                        f"Aggregate reference scan: {bytes_read / (1 << 20):.0f} "
                        f"MiB; cohort_refs={matches}",
                        flush=True,
                    )

                def aggregate_sampling_started(sample_count: int) -> None:
                    print(
                        "Aggregate combat sampling started. Farm normally and kill "
                        "any number of nearby monsters now; no individual monster "
                        f"is tracked ({sample_count} samples planned).",
                        flush=True,
                    )

                aggregate = scan_aggregate_monster_roots(
                    memory,
                    result.monster_cohort,
                    module,
                    regions,
                    readable,
                    module_refs,
                    duration_seconds=args.aggregate_seconds,
                    interval_seconds=args.aggregate_interval,
                    object_span=args.aggregate_object_span,
                    holder_span=args.aggregate_holder_span,
                    minimum_support=args.aggregate_min_support,
                    maximum_scan_bytes=args.aggregate_maximum_scan_mib << 20,
                    maximum_depth=args.max_depth,
                    path_field_span=args.field_span,
                    maximum_roots=args.maximum_roots,
                    maximum_nodes=args.maximum_nodes,
                    path_candidate_limit=args.aggregate_path_candidates,
                    coordinate_limit=config.maximum_absolute_coordinate,
                    check=check,
                    progress=aggregate_progress,
                    on_sampling_started=aggregate_sampling_started,
                )
                aggregate_path = save_aggregate_report(
                    aggregate,
                    args.history_dir,
                )
                print(
                    "Aggregate result: "
                    f"cohort={aggregate.cohort_size}; "
                    f"samples={aggregate.sample_count}; "
                    f"changed_actors={aggregate.changed_actor_count}; "
                    f"transition_events={aggregate.transition_events}; "
                    f"references={aggregate.reference_matches}; "
                    f"candidates={len(aggregate.candidates)}",
                    flush=True,
                )
                print(f"Saved aggregate report: {aggregate_path}", flush=True)
                if aggregate.changed_actor_count == 0:
                    print(
                        "No combat differential was observed. Structural rankings "
                        "were saved, but rerunning while farming will provide "
                        "stronger evidence.",
                        flush=True,
                    )
                print("Top aggregate world/manager candidates:", flush=True)
                for index, candidate in enumerate(aggregate.candidates[:20], 1):
                    field = (
                        f" field=+0x{candidate.actor_field_offset:X}"
                        if candidate.actor_field_offset is not None
                        else ""
                    )
                    path_text = "unrooted"
                    if candidate.pointer_paths:
                        path = candidate.pointer_paths[0]
                        chain = "".join(
                            f" -> +0x{offset:X}" for offset in path.field_offsets
                        )
                        path_text = (
                            f"{module.name}+0x{path.root_module_offset:X}{chain}"
                        )
                    marker = "RECOMMENDED" if candidate.recommended else "candidate"
                    print(
                        f"  {index:02d}. [{marker}] {candidate.kind} "
                        f"target=0x{candidate.target_base:X}{field}; "
                        f"coverage(avg/min)="
                        f"{candidate.average_active_coverage:.2f}/"
                        f"{candidate.minimum_active_coverage:.2f}; "
                        f"changed_support={candidate.changed_actor_support}; "
                        f"baseline={candidate.baseline_support}; root={path_text}",
                        flush=True,
                    )

        return 0 if passive.accepted else 2
    except TimeoutError as error:
        print(f"Timed out: {error}", file=sys.stderr)
        return 3
    finally:
        memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
