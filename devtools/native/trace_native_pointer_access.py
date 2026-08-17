from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from position.MonsterConfig import (  # noqa: E402
    DEFAULT_MONSTER_CONFIG_PATH,
    load_native_monster_config,
)
from position.NativeAccessTracer import (  # noqa: E402
    AccessTracePhaseResult,
    AccessWatchpoint,
    NativeAccessTraceError,
    NativeHardwareAccessTracer,
    chunk_watchpoints,
    instruction_hit_ranking,
)
from position.NativeTraceTargets import (  # noqa: E402
    TraceTargetDiscovery,
    discover_trace_targets,
)
from position.PointerScanWorkflow import ReadableRegionIndex  # noqa: E402
from position.policy import LIVE_ATTACH_POLICY  # noqa: E402
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


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _module_relative(address: int, module_base: int, module_size: int) -> str:
    if module_base <= address < module_base + module_size:
        return f"Neuz.exe+0x{address - module_base:X}"
    return f"0x{address:X}"


def _phase_summary(
    phase: AccessTracePhaseResult,
    *,
    module_base: int,
    module_size: int,
) -> dict[str, Any]:
    grouped: dict[int, dict[str, Any]] = {}
    for hit in phase.hits:
        instruction = hit.instruction
        if instruction is None:
            continue
        address = int(instruction.address)
        item = grouped.setdefault(
            address,
            {
                "address": address,
                "module_relative": _module_relative(
                    address,
                    module_base,
                    module_size,
                ),
                "instruction": instruction.text,
                "bytes": instruction.bytes_hex,
                "hits": 0,
                "watch_labels": Counter(),
                "actor_registers": Counter(),
                "threads": Counter(),
                "effective_addresses": Counter(),
                "frame_returns": Counter(),
            },
        )
        item["hits"] += 1
        item["watch_labels"][hit.watch_label] += 1
        item["threads"][str(hit.thread_id)] += 1
        for register, actor in hit.actor_registers.items():
            item["actor_registers"][f"{register}={actor}"] += 1
        for effective in instruction.effective_addresses:
            item["effective_addresses"][f"0x{effective:X}"] += 1
        for return_address in hit.frame_returns:
            if module_base <= return_address < module_base + module_size:
                item["frame_returns"][
                    _module_relative(return_address, module_base, module_size)
                ] += 1
    normalized: list[dict[str, Any]] = []
    for item in grouped.values():
        normalized.append(
            {
                **item,
                "watch_labels": dict(item["watch_labels"].most_common()),
                "actor_registers": dict(item["actor_registers"].most_common()),
                "threads": dict(item["threads"].most_common()),
                "effective_addresses": dict(
                    item["effective_addresses"].most_common(32)
                ),
                "frame_returns": dict(item["frame_returns"].most_common(32)),
            }
        )
    normalized.sort(
        key=lambda item: (
            len(item["watch_labels"]),
            item["hits"],
            -item["address"],
        ),
        reverse=True,
    )
    return {
        "phase": phase.to_dict(),
        "instruction_summary": normalized,
    }


def _build_data_watchpoints(
    discovery: TraceTargetDiscovery,
    *,
    maximum_direct_slots: int,
    monster_count: int,
) -> tuple[AccessWatchpoint, ...]:
    assert discovery.player is not None
    targets: list[AccessWatchpoint] = [
        AccessWatchpoint(
            "player.x",
            discovery.player.base + discovery.player.x_offset,
            "access",
            4,
        ),
        AccessWatchpoint(
            "player.hp",
            discovery.player.base + discovery.player.hp_offset,
            "access",
            4,
        ),
    ]
    for index, monster in enumerate(discovery.monsters[:monster_count]):
        targets.extend(
            (
                AccessWatchpoint(
                    f"monster[{index}].species",
                    monster.base + monster.species_offset,
                    "access",
                    4,
                ),
                AccessWatchpoint(
                    f"monster[{index}].hp",
                    monster.base + monster.hp_offset,
                    "access",
                    4,
                ),
            )
        )
    for index, slot in enumerate(
        discovery.player.direct_module_slots[:maximum_direct_slots]
    ):
        targets.append(
            AccessWatchpoint(
                f"player.slot[{index}]",
                slot,
                "access",
                4,
            )
        )
    # Preserve order but remove accidental duplicate addresses.
    unique: dict[tuple[int, str], AccessWatchpoint] = {}
    for target in targets:
        unique[(target.address, target.access)] = target
    return tuple(unique.values())


def _build_execute_watchpoints(addresses: tuple[int, ...]) -> tuple[AccessWatchpoint, ...]:
    return tuple(
        AccessWatchpoint(f"instruction[{index}]", address, "execute", 1)
        for index, address in enumerate(addresses)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automate Cheat Engine's 'find out what accesses this address' "
            "workflow for the 32-bit FlyFF client. The tool first finds exact "
            "player and full-health monster objects, then uses rotating x86 "
            "hardware watchpoints to discover the instructions accessing their "
            "fields and direct player slots. A second execute-breakpoint phase "
            "captures those instructions before execution with registers and "
            "call-stack evidence. No process memory is written."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid", type=_positive_int)
    target.add_argument("--window-handle", type=_positive_int)
    target.add_argument("--window-title")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MONSTER_CONFIG_PATH,
    )
    parser.add_argument("--spawn-x", type=float, default=253.0)
    parser.add_argument("--spawn-z", type=float, default=86.0)
    parser.add_argument("--player-hp", type=_positive_int, required=True)
    parser.add_argument(
        "--monster-hp",
        type=_monster_hp,
        action="append",
        required=True,
        metavar="SPECIES=HP",
        help=(
            "mandatory exact full-health monster anchor; repeat for additional "
            "species, for example 944=400236"
        ),
    )
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--maximum-scan-mib", type=_positive_int, default=1536)
    parser.add_argument("--object-span", type=_positive_int, default=0x4000)
    parser.add_argument("--discovery-seconds", type=float, default=20.0)
    parser.add_argument("--replay-seconds", type=float, default=20.0)
    parser.add_argument("--hits-per-target", type=_positive_int, default=40)
    parser.add_argument("--maximum-round-hits", type=_positive_int, default=160)
    parser.add_argument("--maximum-replay-hits", type=_positive_int, default=400)
    parser.add_argument("--trace-monsters", type=_positive_int, default=1)
    parser.add_argument("--maximum-direct-slots", type=_non_negative_int, default=16)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APP_ROOT / "training_logs" / "pointer_traces",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be positive")
    if args.discovery_seconds <= 0.0:
        raise SystemExit("--discovery-seconds must be positive")
    if args.replay_seconds <= 0.0:
        raise SystemExit("--replay-seconds must be positive")
    if args.object_span % 4:
        raise SystemExit("--object-span must be divisible by four")

    config = load_native_monster_config(args.config)
    if args.pid is not None:
        memory = Win32ProcessMemory(args.pid)
    elif args.window_handle is not None:
        memory = Win32ProcessMemory.from_window_handle(args.window_handle)
    else:
        memory = Win32ProcessMemory.from_window_handle(
            _find_window_handle(args.window_title)
        )

    started_at = datetime.now(UTC).isoformat()
    report_path = args.output_dir / f"native_access_trace_{_timestamp()}.json"
    report: dict[str, Any] = {
        "schema_version": 2,
        "started_at": started_at,
        "status": "starting",
        "command": sys.argv,
        "target_discovery": None,
        "data_phases": [],
        "execute_phase": None,
        "selected_execute_instructions": [],
        "error": None,
    }
    _atomic_json(report_path, report)
    deadline = monotonic() + float(args.timeout)

    def check() -> None:
        if monotonic() >= deadline:
            raise TimeoutError("native access trace exceeded --timeout")

    try:
        module = memory.module_info(config.module_name)
        regions = memory.readable_regions(
            maximum_address=config.maximum_scan_address,
            private_only=False,
        )
        readable = ReadableRegionIndex.build(regions)
        report["process"] = {
            "pid": memory.pid,
            "module_name": module.name,
            "module_path": module.path,
            "module_base": module.base_address,
            "module_size": module.size,
        }
        report["anchors"] = {
            "spawn_x": args.spawn_x,
            "spawn_z": args.spawn_z,
            "player_hp": args.player_hp,
            "exact_monster_hp": dict(args.monster_hp),
            "configured_offset_hints": {
                "species": config.species_offset,
                "hp": config.hp_offset,
                "x": config.x_offset,
                "y": config.y_offset,
                "z": config.z_offset,
            },
        }
        report["status"] = "scanning_exact_targets"
        _atomic_json(report_path, report)
        print(
            f"Attached read-only scanner to pid={memory.pid}; module={module.name}; "
            f"base=0x{module.base_address:X}; size=0x{module.size:X}",
            flush=True,
        )
        print(
            "Exact monster anchors are mandatory: "
            + ", ".join(
                f"species {species} HP {hp}" for species, hp in args.monster_hp
            ),
            flush=True,
        )

        def scan_progress(
            bytes_scanned: int,
            regions_scanned: int,
            matches: dict[str, int],
        ) -> None:
            details = ", ".join(f"{name}={count}" for name, count in matches.items())
            print(
                f"Trace anchor scan: {bytes_scanned / (1 << 20):.0f} MiB; "
                f"regions={regions_scanned}; {details}",
                flush=True,
            )

        discovery = discover_trace_targets(
            memory,
            regions=regions,
            readable=readable,
            module=module,
            species_hp=dict(args.monster_hp),
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
            chunk_size=config.discovery_chunk_bytes,
            maximum_scan_bytes=args.maximum_scan_mib << 20,
            check=check,
            progress=scan_progress,
            attach_policy=LIVE_ATTACH_POLICY,
        )
        report["target_discovery"] = discovery.to_dict()
        report["status"] = f"target_discovery_{discovery.outcome}"
        _atomic_json(report_path, report)
        print(f"Target discovery: {discovery.outcome}: {discovery.message}", flush=True)
        print(f"Saved partial report: {report_path}", flush=True)
        if discovery.player is None or not discovery.monsters:
            evidence = discovery.evidence
            print(
                "Dynamic anchor evidence: "
                f"monster_bases={evidence.monster_base_hypotheses}, "
                f"monsters={evidence.monster_candidates}, "
                f"layout_ties={evidence.monster_layout_ties}, "
                f"monster_hp_reject={evidence.monster_hp_rejections}, "
                f"player_hp_reject={evidence.player_hp_rejections}, "
                f"player_self_reject={evidence.player_self_rejections}",
                flush=True,
            )
            return 2

        player = discovery.player
        print(
            f"Player object 0x{player.base:X}; position=({player.x:.3f}, "
            f"{player.y:.3f}, {player.z:.3f}); HP={player.hp} "
            f"at +0x{player.hp_offset:X}; coordinates=+0x{player.x_offset:X}/"
            f"+0x{player.y_offset:X}/+0x{player.z_offset:X}; "
            f"self_aliases={len(player.self_pointer_offsets)}; "
            f"direct_module_slots={len(player.direct_module_slots)}",
            flush=True,
        )
        if player.direct_module_slots:
            print(
                "Direct player slots: "
                + ", ".join(
                    f"{module.name}+0x{slot - module.base_address:X}"
                    for slot in player.direct_module_slots
                ),
                flush=True,
            )
        first_monster = discovery.monsters[0]
        print(
            f"Exact full-health monster actors: {len(discovery.monsters)}; "
            f"species=+0x{first_monster.species_offset:X}; "
            f"active=+0x{first_monster.active_species_offset:X}; "
            f"HP=+0x{first_monster.hp_offset:X}",
            flush=True,
        )

        data_targets = _build_data_watchpoints(
            discovery,
            maximum_direct_slots=args.maximum_direct_slots,
            monster_count=args.trace_monsters,
        )
        rounds = chunk_watchpoints(data_targets)
        tracer = NativeHardwareAccessTracer(
            memory,
            pid=memory.pid,
            module_base=module.base_address,
            module_size=module.size,
            player_base=player.base,
            monster_bases=(item.base for item in discovery.monsters),
            status=lambda message: print(message, flush=True),
        )
        data_phases: list[AccessTracePhaseResult] = []
        report["status"] = "data_watchpoint_tracing"
        _atomic_json(report_path, report)
        for index, watchpoints in enumerate(rounds, 1):
            check()
            print(
                f"Data watchpoint round {index}/{len(rounds)} for "
                f"{args.discovery_seconds:.1f}s. Keep the game running normally.",
                flush=True,
            )
            phase = tracer.trace_phase(
                watchpoints,
                phase=f"data_round_{index}",
                duration_seconds=min(
                    args.discovery_seconds,
                    max(0.1, deadline - monotonic()),
                ),
                maximum_hits=args.maximum_round_hits,
                maximum_hits_per_label=args.hits_per_target,
                module_only=True,
            )
            data_phases.append(phase)
            report["data_phases"].append(
                _phase_summary(
                    phase,
                    module_base=module.base_address,
                    module_size=module.size,
                )
            )
            _atomic_json(report_path, report)
            print(
                f"Round {index}: hits={len(phase.hits)}; "
                f"outside_module_ignored={phase.ignored_outside_module}; "
                f"saved={report_path}",
                flush=True,
            )
            if phase.process_exited:
                report["status"] = "process_exited_during_data_trace"
                _atomic_json(report_path, report)
                return 3

        selected = instruction_hit_ranking(
            data_phases,
            module_base=module.base_address,
            module_size=module.size,
            limit=4,
        )
        report["selected_execute_instructions"] = [
            {
                "address": address,
                "module_relative": _module_relative(
                    address,
                    module.base_address,
                    module.size,
                ),
            }
            for address in selected
        ]
        _atomic_json(report_path, report)
        if not selected:
            report["status"] = "no_module_access_instruction_found"
            _atomic_json(report_path, report)
            print(
                "No Neuz.exe instruction was captured. The partial data-watchpoint "
                f"report remains at {report_path}",
                flush=True,
            )
            return 4

        print("Selected access instructions:", flush=True)
        for address in selected:
            print(
                f"  {_module_relative(address, module.base_address, module.size)}",
                flush=True,
            )

        if not args.skip_replay:
            check()
            print(
                "Execute replay started. These breakpoints trigger before each "
                "selected access instruction so the report gets pre-instruction "
                "registers, effective addresses, and call-stack evidence.",
                flush=True,
            )
            replay = tracer.trace_phase(
                _build_execute_watchpoints(selected),
                phase="execute_replay",
                duration_seconds=min(
                    args.replay_seconds,
                    max(0.1, deadline - monotonic()),
                ),
                maximum_hits=args.maximum_replay_hits,
                maximum_hits_per_label=args.maximum_replay_hits,
                module_only=True,
            )
            report["execute_phase"] = _phase_summary(
                replay,
                module_base=module.base_address,
                module_size=module.size,
            )
            if replay.process_exited:
                report["status"] = "process_exited_during_execute_replay"
            else:
                report["status"] = "completed"
            _atomic_json(report_path, report)
            print(
                f"Execute replay: hits={len(replay.hits)}; saved={report_path}",
                flush=True,
            )
        else:
            report["status"] = "completed_without_execute_replay"
            _atomic_json(report_path, report)

        print("Top discovered instructions:", flush=True)
        combined = []
        for item in report["data_phases"]:
            combined.extend(item["instruction_summary"])
        combined.sort(
            key=lambda item: (
                len(item["watch_labels"]),
                item["hits"],
            ),
            reverse=True,
        )
        for item in combined[:20]:
            print(
                f"  {item['module_relative']}: {item['instruction']} "
                f"hits={item['hits']} targets={list(item['watch_labels'])} "
                f"actor_regs={item['actor_registers']}",
                flush=True,
            )
        print(f"Final trace report: {report_path}", flush=True)
        return 0
    except (TimeoutError, NativeAccessTraceError, OSError) as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        _atomic_json(report_path, report)
        print(f"Access trace failed: {error}", file=sys.stderr, flush=True)
        print(f"Partial report saved: {report_path}", file=sys.stderr, flush=True)
        return 5
    except KeyboardInterrupt:
        report["status"] = "cancelled"
        report["error"] = "KeyboardInterrupt"
        _atomic_json(report_path, report)
        print(f"Cancelled; partial report saved: {report_path}", flush=True)
        return 130
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        _atomic_json(report_path, report)
        raise
    finally:
        memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
