"""2026-08-12: full trial-level analysis of a movement_calibration.csv
produced by calibration_capture.py --capture. Segments the dense
(t, x, z, key-state) stream into individual trials (maximal runs of a
single primitive's key-state), then measures each trial's net distance
and net heading change robustly (whole-trial displacement / windowed
bearing comparison, not tick-by-tick differencing, which is noisy at the
game's own position-update quantization scale -- confirmed separately:
median nonzero per-tick step is ~0.341 native units regardless of
primitive, with occasional 0x/2x ticks from sampling-vs-server-tick phase
drift).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

NATIVE_UNITS_PER_CELL = 1.6
BEARING_WINDOW_TICKS = 5  # ~0.1s at 50Hz -- smooths single-tick quantization noise
MIN_WINDOW_DISPLACEMENT_NATIVE = 0.15  # skip near-zero windows (bearing undefined/noisy)
STATIONARY_GAP_SECONDS = 0.15  # 2026-08-12b: NOT the instructed 2.0s -- measured directly
# from this session's actual NONE-run durations (338 runs, ALL >=0.18s, median 0.66s,
# only 8 ever reached 2.0s -- Ridd paced faster than instructed but consistently left a
# real gap every time). 0.15s sits safely below every observed gap while still being far
# above any single-tick noise floor, so it captures all 338 as valid separators without
# risk of treating a genuine pause as noise.


def classify(row) -> str:
    f, l, r, j = row.forward, row.left, row.right, row.jump
    if f and j:
        return "JUMP"
    if f and l and not r:
        return "LEFT"
    if f and r and not l:
        return "RIGHT"
    if f and not l and not r:
        return "STRAIGHT"
    return "NONE"


def _raw_runs(df: pd.DataFrame) -> list[dict]:
    """Maximal contiguous runs of a single action (including NONE), with
    each run's tick span and wall-clock duration."""
    runs = []
    start = 0
    n = len(df)
    actions = df.action.values
    elapsed = df.elapsed_s.values
    for i in range(1, n + 1):
        if i == n or actions[i] != actions[start]:
            runs.append({
                "action": actions[start], "start_idx": start, "end_idx": i - 1,
                "duration_s": elapsed[i - 1] - elapsed[start],
            })
            start = i
    return runs


def segment_trials(df: pd.DataFrame) -> list[dict]:
    """2026-08-12b fix (user-caught bug): the original version segmented
    on every RAW action change, which splits a single human trial attempt
    into spurious extra pieces whenever the two steering keys don't land
    on the exact same tick -- e.g. a LEFT attempt (Z-down, then Q-down a
    tick or two later, then Q-up before Z-up) produces
    STRAIGHT(brief)->LEFT(core)->STRAIGHT(brief), and the old code counted
    the two brief STRAIGHT fragments as independent STRAIGHT trials. This
    silently inflated the STRAIGHT trial count (198 seen vs ~110 intended
    by the protocol) while LEFT/RIGHT (112/113 vs ~110 intended) were
    already close to correct, since each attempt normally has only one
    core run of its OWN primitive.

    Fixed by segmenting in two phases: first find trial BOUNDARIES from
    genuine >=2s stationary (all-NONE) gaps -- exactly what the protocol's
    "stand still between trials" instruction was for -- then, within each
    gap-bounded block, classify the block by its DOMINANT action (most
    ticks) and use that action's single longest contiguous run inside the
    block as the trial's analysis window. A block with no non-NONE ticks
    at all (pure idle) is dropped. A block that accidentally contains two
    real attempts (gap under 2s) yields only its dominant one -- an
    under-count, not the original over-count, and a safer failure mode."""
    df = df.reset_index(drop=True)
    df["action"] = df.apply(classify, axis=1)
    runs = _raw_runs(df)

    blocks: list[list[dict]] = []
    current: list[dict] = []
    for run in runs:
        if run["action"] == "NONE" and run["duration_s"] >= STATIONARY_GAP_SECONDS:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(run)
    if current:
        blocks.append(current)

    trials = []
    for block in blocks:
        tick_counts: dict[str, int] = {}
        for run in block:
            if run["action"] == "NONE":
                continue
            tick_counts[run["action"]] = tick_counts.get(run["action"], 0) + (run["end_idx"] - run["start_idx"] + 1)
        if not tick_counts:
            continue
        dominant = max(tick_counts, key=tick_counts.get)
        core_runs = [r for r in block if r["action"] == dominant]
        longest = max(core_runs, key=lambda r: r["end_idx"] - r["start_idx"])
        if (longest["end_idx"] - longest["start_idx"] + 1) < 2:
            continue
        trials.append({"action": dominant, "start_idx": longest["start_idx"], "end_idx": longest["end_idx"]})
    return trials


def path_length_cells(sub: pd.DataFrame) -> float:
    """Total PATH LENGTH (sum of consecutive-tick step distances), not net
    endpoint-to-endpoint displacement. These diverge badly once a trial
    turns enough to curve its path into a loop -- net displacement can
    shrink or even reverse while the character keeps moving the whole
    time. Path length is also what the simulator's per-tick model
    actually represents (each control interval is one straight hop), and
    is robust to the ~0.341-native-unit-per-tick quantization/phase-drift
    pattern confirmed separately, since summing ticks cancels 0x/2x noise."""
    dx = sub.player_x.diff().to_numpy()[1:]
    dz = sub.player_z.diff().to_numpy()[1:]
    return float(np.hypot(dx, dz).sum()) / NATIVE_UNITS_PER_CELL


def cumulative_heading_change(sub: pd.DataFrame) -> float | None:
    """Total SIGNED rotation over the whole trial, correctly unwrapped so
    turns exceeding +-180deg (confirmed to happen: observed turn rates of
    ~65-190deg/s times multi-second holds) don't alias to the wrong
    magnitude or even the wrong sign, which a naive
    normalize(bearing_end - bearing_start) does."""
    n = len(sub)
    bearings = []
    for i in range(0, n - BEARING_WINDOW_TICKS):
        j = i + BEARING_WINDOW_TICKS
        dx = sub.player_x[j] - sub.player_x[i]
        dz = sub.player_z[j] - sub.player_z[i]
        d = np.hypot(dx, dz)
        if d >= MIN_WINDOW_DISPLACEMENT_NATIVE:
            bearings.append(np.arctan2(dz, dx))
    if len(bearings) < 2:
        return None
    unwrapped = np.unwrap(np.asarray(bearings))
    return float(unwrapped[-1] - unwrapped[0])


def analyze(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    trials = segment_trials(df)
    rows = []
    for t in trials:
        start, end, action = t["start_idx"], t["end_idx"], t["action"]
        sub = df.iloc[start:end + 1].reset_index(drop=True)
        n_local = len(sub)
        duration_s = sub.elapsed_s.iloc[-1] - sub.elapsed_s.iloc[0]
        distance_cells = path_length_cells(sub)
        heading_change = cumulative_heading_change(sub)

        rows.append({
            "action": action, "duration_s": duration_s, "n_ticks": n_local,
            "distance_cells": distance_cells, "heading_change_radians": heading_change,
            "focused_all": bool(sub.focused.all()),
        })
    return pd.DataFrame(rows)


def summarize(x: np.ndarray, label: str) -> str:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return f"{label}: n=0"
    return (f"{label}: n={x.size:4d} mean={np.mean(x):7.4f} median={np.median(x):7.4f} "
            f"std={np.std(x):7.4f} p05={np.percentile(x,5):7.4f} p95={np.percentile(x,95):7.4f} "
            f"min={np.min(x):7.4f} max={np.max(x):7.4f}")


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("movement_calibration.csv")
    trials = analyze(csv_path)
    trials = trials[trials.focused_all]
    trials.to_csv("calibration_trials.csv", index=False)
    print(f"Segmented {len(trials)} usable trials (focused throughout) from {csv_path}", flush=True)
    print(trials.action.value_counts(), flush=True)

    for action in ("STRAIGHT", "LEFT", "RIGHT"):
        sub = trials[trials.action == action]
        print(f"\n{'='*80}\n{action} (n={len(sub)} trials)\n{'='*80}")
        print(summarize(sub.duration_s.values, "duration_s"))
        print(summarize(sub.distance_cells.values, "distance_cells (net, whole trial)"))
        if action in ("LEFT", "RIGHT"):
            hc = sub.heading_change_radians.dropna()
            print(summarize(hc.values, "heading_change_radians (net, whole trial)"))

        # rate (per second) and per-0.2s-equivalent, to compare against the
        # simulator's control_interval_seconds=0.20 convention
        sub = sub.copy()
        sub["distance_per_sec"] = sub.distance_cells / sub.duration_s
        print(summarize(sub["distance_per_sec"].values, "distance_cells/sec"))
        print(f"  => distance per 0.20s control interval: mean={sub['distance_per_sec'].mean()*0.20:.4f} "
              f"median={sub['distance_per_sec'].median()*0.20:.4f} cells")
        if action in ("LEFT", "RIGHT"):
            sub["turn_per_sec"] = sub.heading_change_radians / sub.duration_s
            hc_rate = sub["turn_per_sec"].dropna()
            print(summarize(hc_rate.values, "turn_radians/sec"))
            print(f"  => turn per 0.20s control interval: mean={hc_rate.mean()*0.20:.4f} "
                  f"median={hc_rate.median()*0.20:.4f} rad")

        # duration-bucketed scaling check
        print(f"\n-- distance_cells by duration bucket (tests linear scaling) --")
        buckets = [0.0, 0.35, 0.75, 2.0, 4.0]
        sub["dur_bucket"] = pd.cut(sub.duration_s, buckets)
        for bucket, g in sub.groupby("dur_bucket", observed=True):
            if len(g) == 0:
                continue
            print(f"  dur in {bucket}: n={len(g):3d} mean_duration={g.duration_s.mean():.3f}s "
                  f"mean_distance={g.distance_cells.mean():.3f}cells "
                  f"distance/duration={g.distance_cells.mean()/g.duration_s.mean():.3f}cells/s", flush=True)
            if action in ("LEFT", "RIGHT"):
                ghc = g.heading_change_radians.dropna()
                if len(ghc):
                    print(f"      mean_heading_change={np.degrees(ghc.mean()):.2f}deg "
                          f"turn/duration={np.degrees(ghc.mean())/g.duration_s.mean():.2f}deg/s")

    # distance-turn correlation, TRIAL level (whole-trial, not tick-level)
    print(f"\n{'='*80}\ndistance <-> |heading_change| correlation (trial-level)\n{'='*80}")
    for action in ("LEFT", "RIGHT"):
        sub = trials[trials.action == action].dropna(subset=["heading_change_radians"])
        if len(sub) > 2:
            corr = np.corrcoef(sub.distance_cells, sub.heading_change_radians.abs())[0, 1]
            print(f"{action}: r={corr:.4f} (n={len(sub)})")

    print(f"\n{'='*80}\nCurrent fitted model for comparison\n{'='*80}")
    print("STRAIGHT: distance=2.517+-0.947 cells/0.2s, turn=0.0")
    print("LEFT:     distance=0.964+-0.962 cells/0.2s, turn=+0.184+-0.211 rad/0.2s")
    print("RIGHT:    distance=0.966+-1.005 cells/0.2s, turn=-0.148+-0.195 rad/0.2s")


if __name__ == "__main__":
    main()
