"""2026-08-12: analysis of movement_calibration_steering.csv -- the
deployment-matching protocol (forward held continuously, LEFT/RIGHT
pulsed on top of already-moving STRAIGHT), as opposed to the earlier
stationary-start protocol. Segmentation here is simple and unambiguous:
forward stays down almost the whole session, so a "pulse" is just a
contiguous LEFT or RIGHT run (forward+turn together) -- no gap threshold
needed, unlike the stationary-start protocol.

For each pulse, also captures the immediately-preceding and following
STRAIGHT runs (when forward continuity holds across the boundary) to
measure pre-pulse baseline speed and post-pulse return-to-straight
behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

NATIVE_UNITS_PER_CELL = 1.6
BEARING_WINDOW_TICKS = 5
MIN_WINDOW_DISPLACEMENT_NATIVE = 0.15


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


def raw_runs(df: pd.DataFrame) -> list[dict]:
    runs = []
    start = 0
    n = len(df)
    actions = df.action.values
    elapsed = df.elapsed_s.values
    for i in range(1, n + 1):
        if i == n or actions[i] != actions[start]:
            runs.append({"action": actions[start], "start_idx": start, "end_idx": i - 1,
                         "duration_s": elapsed[i - 1] - elapsed[start]})
            start = i
    return runs


def path_length_cells(df: pd.DataFrame, s: int, e: int) -> float:
    dx = df.player_x.values[s + 1:e + 1] - df.player_x.values[s:e]
    dz = df.player_z.values[s + 1:e + 1] - df.player_z.values[s:e]
    return float(np.hypot(dx, dz).sum()) / NATIVE_UNITS_PER_CELL


def cumulative_heading_change(df: pd.DataFrame, s: int, e: int) -> float | None:
    px, pz = df.player_x.values, df.player_z.values
    n = e - s + 1
    bearings = []
    for i in range(s, s + n - BEARING_WINDOW_TICKS):
        j = i + BEARING_WINDOW_TICKS
        dx, dz = px[j] - px[i], pz[j] - pz[i]
        d = np.hypot(dx, dz)
        if d >= MIN_WINDOW_DISPLACEMENT_NATIVE:
            bearings.append(np.arctan2(dz, dx))
    if len(bearings) < 2:
        return None
    unwrapped = np.unwrap(np.asarray(bearings))
    return float(unwrapped[-1] - unwrapped[0])


def mean_speed_cells_per_sec(df: pd.DataFrame, s: int, e: int) -> float | None:
    if e <= s:
        return None
    duration = df.elapsed_s.values[e] - df.elapsed_s.values[s]
    if duration <= 0:
        return None
    return path_length_cells(df, s, e) / duration


def analyze(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["action"] = df.apply(classify, axis=1)
    runs = raw_runs(df)

    rows = []
    for idx, run in enumerate(runs):
        if run["action"] not in ("LEFT", "RIGHT"):
            continue
        s, e = run["start_idx"], run["end_idx"]
        if (e - s + 1) < 2:
            continue
        duration_s = df.elapsed_s.values[e] - df.elapsed_s.values[s]
        distance_cells = path_length_cells(df, s, e)
        heading_change = cumulative_heading_change(df, s, e)
        focused_all = bool(df.focused.values[s:e + 1].all())

        pre_run = runs[idx - 1] if idx > 0 else None
        pre_speed = None
        if pre_run is not None and pre_run["action"] == "STRAIGHT" and pre_run["duration_s"] >= 0.3:
            ps, pe = pre_run["start_idx"], pre_run["end_idx"]
            # last ~0.3s of the pre-pulse straight run, as a clean immediate baseline
            baseline_start = pe
            while baseline_start > ps and df.elapsed_s.values[pe] - df.elapsed_s.values[baseline_start] < 0.3:
                baseline_start -= 1
            pre_speed = mean_speed_cells_per_sec(df, baseline_start, pe)

        post_run = runs[idx + 1] if idx + 1 < len(runs) else None
        post_speed_early = None  # first ~0.3s after release
        post_heading_settle = None
        if post_run is not None and post_run["action"] == "STRAIGHT" and post_run["duration_s"] >= 0.2:
            ns, ne = post_run["start_idx"], post_run["end_idx"]
            early_end = ns
            while early_end < ne and df.elapsed_s.values[early_end] - df.elapsed_s.values[ns] < 0.3:
                early_end += 1
            post_speed_early = mean_speed_cells_per_sec(df, ns, early_end)
            post_heading_settle = cumulative_heading_change(df, ns, ne)

        rows.append({
            "action": run["action"], "duration_s": duration_s, "distance_cells": distance_cells,
            "heading_change_radians": heading_change, "focused_all": focused_all,
            "pre_speed_cells_per_sec": pre_speed, "post_speed_early_cells_per_sec": post_speed_early,
            "post_heading_settle_radians": post_heading_settle,
        })
    return pd.DataFrame(rows)


def summarize(x: np.ndarray, label: str) -> str:
    x = x[~np.isnan(x)] if x.dtype.kind == "f" else x
    if x.size == 0:
        return f"{label}: n=0"
    return (f"{label}: n={x.size:4d} mean={np.mean(x):7.4f} median={np.median(x):7.4f} "
            f"std={np.std(x):7.4f} p05={np.percentile(x,5):7.4f} p95={np.percentile(x,95):7.4f}")


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("movement_calibration_steering.csv")
    pulses = analyze(csv_path)
    pulses = pulses[pulses.focused_all]
    pulses.to_csv("calibration_steering_pulses.csv", index=False)
    print(f"Segmented {len(pulses)} usable pulses from {csv_path}")
    print(pulses.action.value_counts())

    for action in ("LEFT", "RIGHT"):
        sub = pulses[pulses.action == action].copy()
        print(f"\n{'='*80}\n{action} (n={len(sub)} pulses)\n{'='*80}")
        print(summarize(sub.duration_s.values, "duration_s"))
        sub["distance_per_sec"] = sub.distance_cells / sub.duration_s
        hc = sub.heading_change_radians.dropna()
        sub["turn_per_sec"] = sub.heading_change_radians / sub.duration_s

        print(f"\n-- by duration bucket (first-0.2s response vs longer holds) --")
        buckets = [0.0, 0.10, 0.30, 0.75, 2.0]
        sub["dur_bucket"] = pd.cut(sub.duration_s, buckets, include_lowest=True)
        for bucket, g in sub.groupby("dur_bucket", observed=True):
            if len(g) == 0:
                continue
            ghc = g.heading_change_radians.dropna()
            gpre = g.pre_speed_cells_per_sec.dropna()
            print(f"  dur in {bucket}: n={len(g):3d} mean_dur={g.duration_s.mean():.3f}s "
                  f"distance/dur={g.distance_cells.mean()/g.duration_s.mean():.3f}cells/s "
                  f"(=> per 0.2s: {g.distance_cells.mean()/g.duration_s.mean()*0.2:.3f} cells) "
                  f"pre_pulse_baseline_speed={gpre.mean() if len(gpre) else float('nan'):.3f}cells/s")
            if len(ghc):
                rate = ghc.mean() / g.duration_s.mean()
                print(f"      mean_heading_change={np.degrees(ghc.mean()):.2f}deg "
                      f"turn/dur={np.degrees(rate):.2f}deg/s (=> per 0.2s: {np.degrees(rate*0.2):.2f}deg, {rate*0.2:.4f}rad)")

        print(f"\n-- return-to-straight (immediately after release) --")
        post_speed = sub.post_speed_early_cells_per_sec.dropna()
        print("  " + summarize(post_speed.values, "post-release first-0.3s speed (cells/s)"))
        post_settle = sub.post_heading_settle_radians.dropna()
        print("  " + summarize(np.degrees(post_settle.values), "post-release straight-run heading drift (deg, over its own duration)"))

    print(f"\n{'='*80}\nLEFT vs RIGHT symmetry\n{'='*80}")
    for action in ("LEFT", "RIGHT"):
        sub = pulses[(pulses.action == action) & (pulses.duration_s >= 0.75)]
        hc = sub.heading_change_radians.dropna()
        rate = (hc / sub.loc[hc.index, "duration_s"]).mean() if len(hc) else float("nan")
        print(f"{action} (duration>=0.75s, n={len(sub)}): turn_rate={np.degrees(rate):.2f}deg/s "
              f"=> per 0.2s: {np.degrees(rate*0.2):.2f}deg ({rate*0.2:.4f}rad)")

    print(f"\n{'='*80}\nFor comparison\n{'='*80}")
    print("Current fitted model:      LEFT_dist=0.964 LEFT_turn=+0.184  RIGHT_dist=0.966 RIGHT_turn=-0.148 (per 0.2s)")
    print("Stationary-start live calibration steady-state: LEFT_dist=2.642 LEFT_turn=+0.871  RIGHT_dist=2.641 RIGHT_turn=-0.874 (per 0.2s)")


if __name__ == "__main__":
    main()
