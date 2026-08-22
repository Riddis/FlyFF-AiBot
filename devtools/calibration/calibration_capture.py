"""2026-08-12: minimal, READ-ONLY high-rate calibration capture tool.

Built to answer one question before asking Ridd to perform any real
recording: can this recorder's already-attached native reader sustain a
much higher (~20-40Hz) position/key-state sampling rate than the full
farming pipeline's historically-achieved ~0.355s median frame interval?

This deliberately reuses only the CHEAP primitives already proven in the
existing recorder -- attach_native_client() for one-time pointer
discovery/restore, IndependentNativeReader.read_player() for position
(a handful of direct memory reads, NOT a scan), KeyboardSampler for W/A/D
state (GetAsyncKeyState, cheap regardless of rate), and foreground_window()
for focus. It deliberately SKIPS the full actor/monster-slot scan
(reader.snapshot()) that the normal recording session couples to every
loop iteration -- that scan, not position reading, is what the codebase's
own comments identify as the expensive part.

Heading is NOT read live here. The existing recorder derives heading from
consecutive position deltas with a threshold gate tuned for its slower
frame rate; at high sample rates that threshold likely needs retuning,
which is exactly the kind of thing better done in offline analysis (with
smoothing/windowing flexibility) than baked into a live capture loop. So
this tool logs dense (t, x, z, key-state, focus) and heading is derived
afterward from the trajectory, not during capture.

Usage:
    python devtools/calibration/calibration_capture.py --benchmark [--seconds 15]
        Attaches, then runs an UNPACED (as-fast-as-possible) loop for the
        given duration, reporting achieved position-read rate, inter-
        sample dt distribution, and read failure rate. Writes nothing.
        Run this FIRST -- it's the answer to "is 20-40Hz feasible here".

    python devtools/calibration/calibration_capture.py --capture --seconds 60 --out FILE.csv [--rate-hz 40]
        Attaches, then runs a PACED loop (sleep-based, targeting
        --rate-hz) for the given duration, writing one CSV row per
        successful sample: elapsed_s,player_x,player_z,forward,left,right,
        jump,focused. This is the actual data-collection mode, only meant
        to be used once --benchmark has confirmed a usable rate.

Nothing in this tool sends input or writes to game memory -- it only
calls the recorder's existing read-only primitives.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from threading import Event

# Two directories deep under the repository root (devtools/calibration/);
# previously relied on this script's own directory being the repo root
# (true only when it lived directly at repo root, pre-Phase-10).
APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from devtools.recorder.config import RecorderConfig  # noqa: E402
from devtools.recorder.keyboard import KeyboardSampler  # noqa: E402
from devtools.recorder.native_capture import AttachedNativeClient, attach_native_client  # noqa: E402
from devtools.recorder.windows import find_client_windows, foreground_window  # noqa: E402
from position.IndependentNativeReader import IndependentNativeReadError  # noqa: E402


def _select_window(config: RecorderConfig):
    windows = find_client_windows(config.window_title_prefix)
    if not windows:
        raise SystemExit(
            f"No client window found with title prefix {config.window_title_prefix!r}. "
            "Is FlyFF running and logged in?"
        )
    if len(windows) == 1:
        print(f"Using window: {windows[0].title}")
        return windows[0]
    print("Multiple matching windows found:")
    for i, w in enumerate(windows):
        print(f"  [{i}] {w.title}")
    choice = int(input("Select window index: ").strip())
    return windows[choice]


def _attach(config: RecorderConfig) -> AttachedNativeClient:
    window = _select_window(config)
    hp_text = input("Current full HP: ").strip()
    player_full_hp = int(hp_text)
    cancellation = Event()

    def status(message: str) -> None:
        print(f"[attach] {message}")

    print("Attaching (this may take a while on first run; instant if a saved profile exists)...")
    attached = attach_native_client(
        hwnd=window.hwnd,
        title=window.title,
        player_full_hp=player_full_hp,
        config=config,
        cancellation=cancellation,
        status=status,
    )
    print(f"Attached to PID {attached.pid} ({attached.title}).")
    return attached


def _sample_once(attached: AttachedNativeClient, keyboard: KeyboardSampler):
    """One read attempt: (ok, player_x, player_z, key_snapshot, focused)."""
    focused = foreground_window() == attached.hwnd
    key = keyboard.sample()
    try:
        player = attached.reader.read_player()
        return True, player.x, player.z, key, focused
    except IndependentNativeReadError:
        return False, None, None, key, focused


def run_benchmark(attached: AttachedNativeClient, keyboard: KeyboardSampler, seconds: float) -> None:
    print(f"\nRunning UNPACED benchmark for {seconds:.0f}s -- move around normally in-game if you like.")
    timestamps: list[float] = []
    ok_count = 0
    fail_count = 0
    deadline = time.monotonic() + seconds
    start = time.monotonic()
    while time.monotonic() < deadline:
        now = time.monotonic()
        ok, _x, _z, _key, _focused = _sample_once(attached, keyboard)
        timestamps.append(now)
        if ok:
            ok_count += 1
        else:
            fail_count += 1
    elapsed = time.monotonic() - start
    total = ok_count + fail_count

    dts = [b - a for a, b in zip(timestamps, timestamps[1:])]
    print(f"\n=== Benchmark result ===")
    print(f"elapsed: {elapsed:.2f}s  total loop iterations: {total}  "
          f"achieved loop rate: {total/elapsed:.1f} Hz")
    print(f"successful position reads: {ok_count} ({100*ok_count/max(1,total):.1f}%)  "
          f"failed reads: {fail_count}")
    if dts:
        print(f"inter-iteration dt (seconds): mean={statistics.mean(dts):.5f} "
              f"median={statistics.median(dts):.5f} "
              f"p95={sorted(dts)[int(0.95*len(dts))]:.5f} max={max(dts):.5f}")
    print("\nIf achieved loop rate is comfortably above ~20-40 Hz, a real calibration\n"
          "capture session at that rate is feasible with this tool as-is.")


def run_capture(attached: AttachedNativeClient, keyboard: KeyboardSampler, seconds: float,
                 rate_hz: float, out_path: Path) -> None:
    period = 1.0 / rate_hz
    print(f"\nRunning PACED capture at target {rate_hz:.0f}Hz for {seconds:.0f}s -> {out_path}")
    print("Perform your controlled trials now (see the protocol doc). Ctrl+C to stop early.")
    rows: list[tuple] = []
    start = time.monotonic()
    deadline = start + seconds
    next_tick = start
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            ok, x, z, key, focused = _sample_once(attached, keyboard)
            if ok:
                rows.append((now - start, x, z,
                             bool(key.mask & 1), bool(key.mask & 2), bool(key.mask & 4),
                             bool(key.mask & 8), focused))
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\nStopped early by user.")

    elapsed = time.monotonic() - start
    with open(out_path, "w") as f:
        f.write("elapsed_s,player_x,player_z,forward,left,right,jump,focused\n")
        for row in rows:
            f.write(f"{row[0]:.5f},{row[1]:.5f},{row[2]:.5f},{int(row[3])},{int(row[4])},"
                    f"{int(row[5])},{int(row[6])},{int(row[7])}\n")
    print(f"\nWrote {len(rows)} samples over {elapsed:.1f}s ({len(rows)/max(elapsed,1e-9):.1f} Hz achieved) to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", action="store_true", help="run the unpaced feasibility benchmark")
    parser.add_argument("--capture", action="store_true", help="run a real paced capture session")
    parser.add_argument("--seconds", type=float, default=15.0, help="duration in seconds")
    parser.add_argument("--rate-hz", type=float, default=40.0, help="target sample rate for --capture")
    parser.add_argument("--out", type=Path, default=Path("calibration_output.csv"), help="output CSV for --capture")
    parser.add_argument("--layout", choices=("qwerty", "azerty"), default="qwerty")
    parser.add_argument("--eva-hotkey", default="F9", help="pick an F-key/number-key you will NOT press during trials")
    args = parser.parse_args()

    if not args.benchmark and not args.capture:
        parser.error("pass --benchmark (run this first) or --capture")

    config = RecorderConfig.load()
    attached = _attach(config)
    keyboard = KeyboardSampler(layout=args.layout, eva_hotkey=args.eva_hotkey)
    try:
        if args.benchmark:
            run_benchmark(attached, keyboard, args.seconds)
        if args.capture:
            run_capture(attached, keyboard, args.seconds, args.rate_hz, args.out)
    finally:
        attached.close()


if __name__ == "__main__":
    main()
