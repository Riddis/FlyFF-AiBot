"""Headless (non-GUI) recorder entrypoint.

A specialist entrypoint, launched only as an independent OS subprocess
by the canonical dev app (never imported into it -- ``recorder`` is
explicitly excluded from ``apps/dev_app.py``'s import closure, see
``docs/architecture/SYSTEM_OVERVIEW.md`` and
``tests/test_dev_app_import_closure.py``). Reuses
``recorder.session.RecorderController`` untouched -- this is not a
second recorder implementation, only a non-interactive driver for it,
used for both automatic OPERATIONAL_FEEDBACK recording during live
farming/training and explicit CONTROLLED_EXPERIMENT sessions
(docs/PROJECT_GOALS.md section 6).

Stopped by writing anything to the file named by --stop-signal-file, or
by SIGINT/SIGTERM. Prints one JSON object per line to stdout for the
launching process to parse (the same explicit-argv, no-PYTHONPATH-
injection subprocess pattern this project already uses for every other
specialist entrypoint).
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from recorder.provenance import ExperimentProvenance  # noqa: E402
from recorder.session import RecorderController  # noqa: E402


def _emit(**payload: object) -> None:
    print(json.dumps(payload), flush=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hwnd", type=int, required=True)
    parser.add_argument("--window-title", required=True)
    parser.add_argument("--player-full-hp", type=int, required=True)
    parser.add_argument("--keyboard-layout", default="qwerty")
    parser.add_argument("--eva-hotkey", default="F1")
    parser.add_argument(
        "--purpose",
        choices=("OPERATIONAL_FEEDBACK", "CONTROLLED_EXPERIMENT"),
        required=True,
    )
    parser.add_argument(
        "--controller-type",
        choices=("HUMAN_CONTROLLED", "BOT_POLICY_CONTROLLED", "SCRIPTED_CONTROLLED"),
        default="BOT_POLICY_CONTROLLED",
    )
    parser.add_argument("--protocol-id", default=None)
    parser.add_argument("--hypothesis", default=None)
    parser.add_argument(
        "--data-use-role",
        choices=("FITTING_ELIGIBLE", "VALIDATION_HOLDOUT", "DIAGNOSTIC_ONLY"),
        default="FITTING_ELIGIBLE",
    )
    parser.add_argument(
        "--stop-signal-file",
        required=True,
        help="Polled; recording stops once this file exists.",
    )
    parser.add_argument("--attach-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--stop-poll-seconds", type=float, default=0.5)
    return parser.parse_args(argv)


def run(argv: list[str]) -> int:
    args = _parse_args(argv)
    stop_signal_file = Path(args.stop_signal_file)

    provenance = ExperimentProvenance(
        purpose=args.purpose,
        controller_type=args.controller_type,
        protocol_id=args.protocol_id,
        hypothesis=args.hypothesis,
        data_use_role=args.data_use_role,
    )
    controller = RecorderController(experiment_provenance=provenance)

    controller.attach_async(
        hwnd=args.hwnd,
        title=args.window_title,
        player_full_hp=args.player_full_hp,
        keyboard_layout=args.keyboard_layout,
        eva_hotkey=args.eva_hotkey,
    )

    deadline = time.monotonic() + args.attach_timeout_seconds
    attached = False
    while time.monotonic() < deadline:
        try:
            event = controller.events.get(timeout=0.5)
        except queue.Empty:
            continue
        _emit(**event)
        if event["type"] == "attached":
            attached = True
            break
        if event["type"] in ("error", "attach_cancelled"):
            return 1
    if not attached:
        _emit(type="error", context="attach", message="Attach timed out.")
        controller.close()
        return 1

    controller.start_logging()
    _emit(type="recording_started")

    while not stop_signal_file.exists():
        try:
            event = controller.events.get(timeout=args.stop_poll_seconds)
        except queue.Empty:
            continue
        _emit(**event)
        if event["type"] in ("finished", "error"):
            controller.close()
            return 0 if event["type"] == "finished" else 1

    controller.end_logging()
    while True:
        event = controller.events.get()
        _emit(**event)
        if event["type"] in ("finished", "error"):
            controller.close()
            return 0 if event["type"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
