from __future__ import annotations

# pyright: reportImplicitRelativeImport=false
"""Run an observation-only farming telemetry session against a live process.

Deliberately independent of ``Bot``/``Gui``/``RuntimeController``: it attaches
providers directly through ``position.create_native_provider_attachment``
(the same read-only factory ``Bot.__init__`` uses under the hood) and never
imports anything from ``farming.control``. There is no code path in this
script capable of pressing a key.

Example (stationary session, native-only, 10 minutes, whole Tower):
    python apps/telemetry_cli.py --window-title Flyff ^
        --map "Tower AoE" --duration-seconds 600

Stop early with Ctrl+C -- the writer flushes and closes cleanly either way.
"""

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

# One directory deep under the repository root (apps/); direct invocation
# (python apps/telemetry_cli.py) sets sys.path[0] to this file's own
# directory, not the repository root, so the root-level packages below
# need it added explicitly.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from assets.Assets import MobInfo  # noqa: E402
from devtools.telemetry.observation_telemetry import (  # noqa: E402
    TelemetryObserver,
    TelemetrySessionRole,
    TelemetryRunSummary,
    build_session_provenance,
    open_session,
)
from farming.map_context import FarmingMapContext  # noqa: E402
from position import create_native_provider_attachment  # noqa: E402
from position.native_process_service import NativeProcessService  # noqa: E402
from utils.helpers import get_window_handlers  # noqa: E402
from worker_manager import CancellationToken  # noqa: E402


def _parse_int(value: str) -> int:
    return int(value, 0)


def _resolve_window(args: argparse.Namespace) -> tuple[str, int]:
    if args.hwnd is not None:
        return f"HWND 0x{args.hwnd:X}", args.hwnd
    windows = get_window_handlers()
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
    lines = "\n".join(f"  0x{handle:X}  {title}" for title, handle in sorted(matches))
    raise SystemExit("More than one window matched. Choose one with --hwnd:\n" + lines)


def _species_ids(names_or_all: str | None) -> set[int] | None:
    if names_or_all is None:
        return None  # observe every active species -- appropriate for a diagnostic session
    result: set[int] = set()
    for token in names_or_all.split(","):
        token = token.strip()
        if not token:
            continue
        result.add(int(token))
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Observation-only farming telemetry: watch and log the native "
            "reader stack without ever constructing anything capable of "
            "sending keyboard input."
        )
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--window-title", help="case-insensitive title substring")
    target.add_argument("--hwnd", type=_parse_int, help="window handle, e.g. 0x123456")
    parser.add_argument(
        "--map",
        default=None,
        help="completed coordinate-map name; adds derived Tower-cell coordinates "
        "and pins map_content_hash into session provenance (optional)",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="comma-separated species IDs to restrict actor reads to; default "
        "observes every active species",
    )
    parser.add_argument(
        "--vision-radius-native",
        type=float,
        default=1400.0,
        help="native-unit actor scan radius (default matches the shipped 50-cell "
        "farming default at 1.6 native units/cell -- change together with --map)",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=0.05,
        help="delay between samples; the recorder's dedicated calibration "
        "captures ran near this rate (default 0.05)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="stop automatically after this many seconds; omit to run until "
        "Ctrl+C",
    )
    parser.add_argument(
        "--output-dir",
        default="telemetry_sessions",
        help="directory to write <session_id>.session.json and "
        "<session_id>.samples.jsonl into",
    )
    parser.add_argument(
        "--session-role",
        choices=[role.value for role in TelemetrySessionRole],
        default=TelemetrySessionRole.CALIBRATION_DEVELOPMENT.value,
        help="frozen into session provenance before the run starts; the first "
        "real session should stay calibration_development, not "
        "untouched_validation",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="free-text note frozen into session provenance (e.g. "
        "'stationary, Tower center, human at keyboard doing nothing')",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    title, hwnd = _resolve_window(args)

    attachment = create_native_provider_attachment(hwnd)
    if attachment.service is None or attachment.position_provider is None:
        raise SystemExit(
            "Native position/monster reading is disabled in the position/*.json "
            "configs -- nothing to observe."
        )
    service: NativeProcessService = attachment.service
    position_provider = attachment.position_provider
    monster_provider = attachment.monster_provider
    if monster_provider is None:
        raise SystemExit("Native monster reading is disabled; telemetry needs it.")

    map_context: FarmingMapContext | None = None
    if args.map:
        map_context = FarmingMapContext.load(args.map, require_forbidden=True)

    allowed_species = _species_ids(args.species)
    if allowed_species is None:
        registered = {
            int(entry["species_id"])
            for entry in MobInfo.get_all_mobs().values()
            if isinstance(entry, dict) and not isinstance(entry.get("species_id"), bool)
            and isinstance(entry.get("species_id"), int)
        }
        print(
            f"No --species given: observing every active species "
            f"({len(registered)} registered mobs also known by name)."
        )

    cancellation = CancellationToken()

    # Read-only preflight, mirroring farming.trainer.build_live_farming_runtime
    # up to (never past) world_reader.read_frame() -- no focus request, no
    # DirectFarmingControl, ever.
    snapshot = service.read_pointer_snapshot()
    refresh = monster_provider.refresh_slot_cache(
        snapshot, cancellation=cancellation, deadline=time.monotonic() + 5.0
    )
    if not refresh.ready:
        attachment.close()
        raise SystemExit(f"Actor-cache preflight failed: {refresh.outcome.value}: {refresh.message}")

    session = build_session_provenance(
        session_role=TelemetrySessionRole(args.session_role),
        map_context=map_context,
        selected_species_ids=sorted(allowed_species) if allowed_species else (),
        vision_radius_native=args.vision_radius_native,
        notes=args.notes,
    )
    writer = open_session(args.output_dir, session)
    print(f"Session {session.session_id} ({session.session_role}) -> {writer.path}")
    print(
        f"bot_git_commit={session.bot_git_commit} dirty={session.bot_git_dirty} "
        f"map={session.map_name} map_hash={session.map_content_hash}"
    )
    print(
        f"monotonic clock: {session.monotonic_clock.implementation} "
        f"(resolution={session.monotonic_clock.resolution:.2e}s) | "
        f"perf_counter clock: {session.perf_counter_clock.implementation} "
        f"(resolution={session.perf_counter_clock.resolution:.2e}s)"
    )

    observer = TelemetryObserver(
        service,
        position_provider,
        monster_provider,
        writer,
        session,
        cancellation,
        allowed_species_ids=allowed_species,
        vision_radius_native=args.vision_radius_native,
        sample_interval_seconds=args.sample_interval_seconds,
        map_context=map_context,
        target_hwnd=hwnd,
        status_callback=print,
    )

    # Ctrl+C requests a graceful stop through the same cancellation path
    # observer.run() already handles on its own (checked once per sample
    # interval); it must never propagate as a raw KeyboardInterrupt out of
    # run(), since that would skip its normal writer.stop() bookkeeping path.
    def _handle_sigint(_signum: int, _frame: object) -> None:
        print("Stopping (Ctrl+C received, flushing telemetry)...")
        cancellation.cancel()

    signal.signal(signal.SIGINT, _handle_sigint)

    timer: threading.Timer | None = None
    if args.duration_seconds is not None:
        timer = threading.Timer(args.duration_seconds, cancellation.cancel)
        timer.daemon = True
        timer.start()

    summary: TelemetryRunSummary
    try:
        summary = observer.run()
    finally:
        if timer is not None:
            timer.cancel()
        attachment.close()

    print(
        f"Done: attempted={summary.samples_attempted} "
        f"read_failures={summary.read_failures} "
        f"written={summary.samples_written} dropped={summary.samples_dropped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
