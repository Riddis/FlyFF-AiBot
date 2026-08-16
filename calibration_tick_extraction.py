"""2026-08-12: extracts exact 0.20s control-tick-indexed dynamics from
movement_calibration_steering.csv (the deployment-matched, forward-
latched recording). This is what RecordedFarmingEnv.step() actually
needs: per-consecutive-steering-tick distance/turn, not whole-pulse
averages over broad duration buckets.

Each LEFT/RIGHT pulse is aligned to its turn-key-down tick (index 0 =
the first tick where the pulse's action is active). At 50Hz, one 0.2s
control tick = 10 samples. Windows are FIXED tick-count slices (not
time-interpolated) since the achieved rate is a very precise, consistent
50.00Hz -- confirmed separately (dt mean/median ~0.02003s, std tiny).

tick 1: samples [0,10)     (0.0-0.2s into the pulse)
tick 2: samples [10,20)    (0.2-0.4s)
tick 3: samples [20,30)    (0.4-0.6s)
tick 4: samples [30,40)    (0.6-0.8s)
tick 5: samples [40,50)    (0.8-1.0s)
steady: every complete 10-sample window at offset >=50 (t>=1.0s),
        pooled across all qualifying pulses (not just one window each)

Only pulses long enough to fully cover a given window contribute to it.
Also extracts the last 0.2s of each pulse's immediately-preceding
STRAIGHT run as an equivalent STRAIGHT tick sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

NATIVE_UNITS_PER_CELL = 1.6
BEARING_SUBWINDOW_TICKS = 5  # half of one 10-tick (0.2s) control window
TICKS_PER_CONTROL_INTERVAL = 10  # 0.2s at 50Hz
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


def window_distance_cells(df: pd.DataFrame, s: int, e: int) -> float:
    """s,e are absolute df row indices; window is [s, e] inclusive, e-s+1 samples."""
    dx = df.player_x.values[s + 1:e + 1] - df.player_x.values[s:e]
    dz = df.player_z.values[s + 1:e + 1] - df.player_z.values[s:e]
    return float(np.hypot(dx, dz).sum()) / NATIVE_UNITS_PER_CELL


START_ANCHOR_TICKS = 3  # short lookahead just for a t=0 anchor point
START_ANCHOR_MIN_DISPLACEMENT_NATIVE = 0.08  # proportionally lower than the 5-tick threshold


def pulse_bearing_curve_impl(df: pd.DataFrame, s0: int, e0: int) -> tuple[np.ndarray, np.ndarray] | None:
    """2026-08-12b fix (bug found via cross-check against the already-
    validated whole-pulse duration-bucket analysis): the original
    approach compared two 5-tick sub-window bearings and took their
    difference as "the window's heading change" -- but each sub-window's
    bearing estimates the direction at roughly ITS OWN MIDPOINT, so the
    two midpoints are only ~0.1s apart, not the full 0.2s window span.
    That silently compresses/distorts any real ramp (confirmed: it
    produced an artificially FLAT ~22deg at every tick, including tick 1,
    contradicting the validated duration-bucket finding of ~14deg at
    tick 1 rising to ~47deg steady).

    Fixed by building ONE continuous unwrapped bearing-vs-time curve for
    the WHOLE pulse (5-tick lookahead bearing estimate at every tick,
    still smoothing single-tick position-quantization noise, but now
    unwrapped and interpolated), so each 0.2s window's heading change can
    be read as the TRUE difference between the curve's value at the
    window's exact start and end boundaries, not a compressed proxy."""
    px, pz = df.player_x.values, df.player_z.values
    times, bearings = [], []
    for i in range(s0, e0 + 1 - BEARING_SUBWINDOW_TICKS):
        j = i + BEARING_SUBWINDOW_TICKS
        dx, dz = px[j] - px[i], pz[j] - pz[i]
        d = np.hypot(dx, dz)
        if d < MIN_WINDOW_DISPLACEMENT_NATIVE:
            continue
        # associate with the midpoint of the lookahead span, in ticks
        # relative to the pulse start (s0) -- used only for interpolation.
        times.append((i - s0) + BEARING_SUBWINDOW_TICKS / 2.0)
        bearings.append(np.arctan2(dz, dx))
    if len(times) < 2:
        return None
    times = np.asarray(times, dtype=np.float64)
    unwrapped = np.unwrap(np.asarray(bearings, dtype=np.float64))
    return times, unwrapped


def pulse_bearing_curve(df: pd.DataFrame, s0: int, e0: int) -> tuple[np.ndarray, np.ndarray] | None:
    """2026-08-12c fix: the plain 5-tick sliding window's first sample is
    centered at t=2.5 ticks, leaving t=0 (the exact key-down boundary --
    needed for tick 1, the single most decision-relevant number) outside
    the interpolation range entirely (tick 1 came back n=0). Prepends a
    dedicated SHORT (3-tick) anchor estimate at exactly t=0, accepting a
    small forward bias (it averages heading over ticks [0,3], not a true
    instant) in exchange for having any measurement there at all, then
    continues with the normal, better-smoothed 5-tick curve from t=2.5
    onward -- interpolation between t=0 and t=2.5 is a short, well-
    conditioned gap, not an extrapolation."""
    px, pz = df.player_x.values, df.player_z.values
    n_ticks = e0 - s0 + 1
    anchor = None
    if n_ticks > START_ANCHOR_TICKS:
        dx = px[s0 + START_ANCHOR_TICKS] - px[s0]
        dz = pz[s0 + START_ANCHOR_TICKS] - pz[s0]
        if np.hypot(dx, dz) >= START_ANCHOR_MIN_DISPLACEMENT_NATIVE:
            anchor = float(np.arctan2(dz, dx))

    end_anchor = None
    if n_ticks > START_ANCHOR_TICKS:
        dx = px[e0] - px[e0 - START_ANCHOR_TICKS]
        dz = pz[e0] - pz[e0 - START_ANCHOR_TICKS]
        if np.hypot(dx, dz) >= START_ANCHOR_MIN_DISPLACEMENT_NATIVE:
            end_anchor = float(np.arctan2(dz, dx))

    rest = pulse_bearing_curve_impl(df, s0, e0)
    if rest is None:
        # no interior samples -- fall back to just the two edge anchors, if both exist
        if anchor is not None and end_anchor is not None:
            end_unwrapped = end_anchor + 2.0 * np.pi * np.round((anchor - end_anchor) / (2.0 * np.pi))
            return np.asarray([0.0, float(n_ticks)]), np.asarray([anchor, end_unwrapped])
        return None

    rest_times, rest_unwrapped = rest
    times_list = [rest_times]
    unwrapped_list = [rest_unwrapped]
    prefix_times, prefix_unwrapped = [], []
    suffix_times, suffix_unwrapped = [], []
    if anchor is not None:
        # align anchor onto the same branch as the rest-of-curve's first value
        # (add whichever multiple of 2pi brings it closest), NOT np.unwrap
        # (which would leave anchor untouched and adjust the wrong element).
        anchor_unwrapped = anchor + 2.0 * np.pi * np.round((rest_unwrapped[0] - anchor) / (2.0 * np.pi))
        prefix_times, prefix_unwrapped = [0.0], [anchor_unwrapped]
    if end_anchor is not None:
        end_unwrapped = end_anchor + 2.0 * np.pi * np.round((rest_unwrapped[-1] - end_anchor) / (2.0 * np.pi))
        suffix_times, suffix_unwrapped = [float(n_ticks)], [end_unwrapped]

    times = np.concatenate([np.asarray(prefix_times), rest_times, np.asarray(suffix_times)])
    unwrapped = np.concatenate([np.asarray(prefix_unwrapped), rest_unwrapped, np.asarray(suffix_unwrapped)])
    return times, unwrapped


def boundary_headings(curve: tuple[np.ndarray, np.ndarray] | None, boundary_ticks: list[float]) -> list[float | None]:
    """Interpolates the unwrapped bearing curve at each requested
    pulse-relative tick offset; returns None for boundaries outside the
    curve's covered range (can't extrapolate reliably)."""
    if curve is None:
        return [None] * len(boundary_ticks)
    times, unwrapped = curve
    out = []
    for t in boundary_ticks:
        if t < times[0] or t > times[-1]:
            out.append(None)
        else:
            out.append(float(np.interp(t, times, unwrapped)))
    return out


def extract(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["action"] = df.apply(classify, axis=1)
    runs = raw_runs(df)

    rows = []
    for idx, run in enumerate(runs):
        if run["action"] not in ("LEFT", "RIGHT"):
            continue
        s0, e0 = run["start_idx"], run["end_idx"]
        n_ticks = e0 - s0 + 1
        if not bool(df.focused.values[s0:e0 + 1].all()):
            continue

        # how many complete 10-tick windows fit (ticks 1..5, plus steady beyond)
        max_k = (n_ticks) // TICKS_PER_CONTROL_INTERVAL
        curve = pulse_bearing_curve(df, s0, e0)
        boundary_ticks = [k * TICKS_PER_CONTROL_INTERVAL for k in range(max_k + 1)]
        heads = boundary_headings(curve, boundary_ticks)

        for k in range(max_k):
            lo = s0 + k * TICKS_PER_CONTROL_INTERVAL
            hi = lo + TICKS_PER_CONTROL_INTERVAL
            tick_label = (k + 1) if k < 5 else "steady"
            h0, h1 = heads[k], heads[k + 1]
            heading_change = (h1 - h0) if (h0 is not None and h1 is not None) else None
            rows.append({
                "action": run["action"], "tick": tick_label,
                "distance_cells": window_distance_cells(df, lo, hi - 1),
                "heading_change_radians": heading_change,
            })

        # equivalent STRAIGHT window: last 0.2s of the immediately-preceding STRAIGHT run
        pre_run = runs[idx - 1] if idx > 0 else None
        if pre_run is not None and pre_run["action"] == "STRAIGHT":
            ps, pe = pre_run["start_idx"], pre_run["end_idx"]
            if (pe - ps + 1) >= TICKS_PER_CONTROL_INTERVAL and bool(df.focused.values[ps:pe + 1].all()):
                lo = pe - TICKS_PER_CONTROL_INTERVAL + 1
                # build the curve over the FULL preceding run (not just the
                # last 10 ticks) so the 5-tick sliding window has enough
                # margin to reach the run's actual end boundary -- same
                # edge issue as the tick-1 anchor fix, now at the far end.
                pre_curve = pulse_bearing_curve(df, ps, pe)
                boundary_a = float(lo - ps)
                boundary_b = float(pe - ps + 1)
                pre_heads = boundary_headings(pre_curve, [boundary_a, boundary_b])
                pre_hc = (pre_heads[1] - pre_heads[0]) if (pre_heads[0] is not None and pre_heads[1] is not None) else None
                rows.append({
                    "action": "STRAIGHT", "tick": "pre-pulse",
                    "distance_cells": window_distance_cells(df, lo, pe),
                    "heading_change_radians": pre_hc,
                })
    return pd.DataFrame(rows)


def summarize(x: np.ndarray, label: str) -> str:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return f"{label}: n=0"
    return (f"{label}: n={x.size:4d} mean={np.mean(x):7.4f} median={np.median(x):7.4f} std={np.std(x):7.4f} "
            f"p05={np.percentile(x,5):7.4f} p25={np.percentile(x,25):7.4f} "
            f"p75={np.percentile(x,75):7.4f} p95={np.percentile(x,95):7.4f}")


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("movement_calibration_steering.csv")
    ticks = extract(csv_path)
    ticks.to_csv("calibration_tick_extraction.csv", index=False)

    tick_order = [1, 2, 3, 4, 5, "steady"]
    print(f"{'tick':>10s}  {'action':>7s}  distance_cells                                                  heading_change_radians")
    results = {}
    for action in ("LEFT", "RIGHT"):
        for t in tick_order:
            g = ticks[(ticks.action == action) & (ticks.tick == t)]
            d = g.distance_cells.values
            h = g.heading_change_radians.dropna().values
            results[(action, t)] = (d, h)
            print(f"\n{str(t):>10s}  {action:>7s}  " + summarize(d, "distance/0.2s"))
            print(f"{'':>10s}  {'':>7s}  " + summarize(h, "turn/0.2s (rad)") +
                  f"  [{np.degrees(np.median(h)) if h.size else float('nan'):.2f}deg median]")

    pre = ticks[ticks.tick == "pre-pulse"]
    print(f"\n{'pre-pulse':>10s}  {'STRAIGHT':>7s}  " + summarize(pre.distance_cells.values, "distance/0.2s"))
    print(f"{'':>10s}  {'':>7s}  " + summarize(pre.heading_change_radians.dropna().values, "turn/0.2s (rad)"))

    print(f"\n{'='*100}\nLEFT vs RIGHT symmetry per tick (|magnitude| comparison)\n{'='*100}")
    for t in tick_order:
        dl, hl = results[("LEFT", t)]
        dr, hr = results[("RIGHT", t)]
        dl_med, dr_med = (np.median(dl) if dl.size else float("nan")), (np.median(dr) if dr.size else float("nan"))
        hl_med, hr_med = (np.median(hl) if hl.size else float("nan")), (np.median(hr) if hr.size else float("nan"))
        print(f"tick={str(t):>8s}  distance: LEFT={dl_med:.4f} RIGHT={dr_med:.4f} (ratio={dl_med/dr_med if dr_med else float('nan'):.3f})  "
              f"|turn|: LEFT={hl_med:.4f} RIGHT={abs(hr_med):.4f} (ratio={hl_med/abs(hr_med) if hr_med else float('nan'):.3f})")

    print(f"\n{'='*100}\nFor comparison\n{'='*100}")
    print("Current fitted model:      LEFT_dist=0.964 LEFT_turn=+0.184  RIGHT_dist=0.966 RIGHT_turn=-0.148 (per 0.2s)")


if __name__ == "__main__":
    main()
