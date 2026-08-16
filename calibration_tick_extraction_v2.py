"""2026-08-13: timestamp-based rewrite of the 0.20s control-tick
extractor, replacing calibration_tick_extraction.py entirely.

User-caught bug in v1: TICKS_PER_CONTROL_INTERVAL=10 at 50Hz, but a
10-POSITION-ROW window contains only 9 displacement intervals -- at
~0.02003s/sample that spans ~0.18s, not 0.20s. Explains the suspicious
"2.406 cells/0.2s" result exactly: 2.406/0.18 = 13.37 cells/s, matching
the independently measured ~13.2-13.7 cells/s live forward speed almost
exactly. A genuine 0.20s window should give ~13.3*0.20 = 2.66 cells,
matching the earlier (differently-derived) 2.64-2.74 cells/0.2s results.

This version defines every window by ACTUAL elapsed_s timestamps, not
sample counts, and interpolates position/bearing at exact boundaries.
Also fixes the general principle behind the three edge/bearing bugs v1
needed patched one at a time (midpoint-vs-boundary compression, missing
start anchor, missing end anchor) by building one continuous, double-
ended bearing-vs-time curve per pulse from the start, rather than
patching a sample-index-based curve after the fact.

Before trusting this against real data, it is validated against
synthetic 50Hz trajectories with EXACTLY KNOWN ground-truth speed and
per-tick turn (closed-form circular-arc motion, optionally position-
quantized to match the real recorder's 0.05-native-unit quantum) -- see
run_synthetic_validation(). Real-data extraction only runs after that
validation passes.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NATIVE_UNITS_PER_CELL = 1.6
CONTROL_INTERVAL_SECONDS = 0.20
BEARING_WINDOW_SECONDS = 0.08  # 2026-08-13: tuned via synthetic sweep (0.10/0.08/0.06/0.04s)
# against BOTH error sources a centered-difference window trades off: (a) smoothing bias
# right at a modeled instantaneous rate transition (shrinks as W shrinks -- confirmed:
# 0.10->0.035rad, 0.08->0.026rad, 0.06->0.030rad err on tick1 at realistic quantization)
# and (b) quantization-noise sensitivity (grows as W shrinks, since fewer/noisier real
# samples fall inside a smaller window). 0.08s minimizes the combined residual across
# both tick1 (the only boundary straddling TWO assumed transitions) and steady-state.
MIN_WINDOW_DISPLACEMENT_NATIVE = 0.15
POSITION_QUANTUM_NATIVE = 0.05
SAMPLE_INTERVAL_SECONDS = 0.02  # nominal recorder rate, used only for synthetic generation


# ---------------------------------------------------------------------------
# Core continuous-time primitives (shared by real-data extraction and the
# synthetic validation harness -- the whole point is that both paths run
# through the EXACT same code).
# ---------------------------------------------------------------------------

def interpolate_xz(times: np.ndarray, xs: np.ndarray, zs: np.ndarray, t: float) -> tuple[float, float] | None:
    if t < times[0] or t > times[-1]:
        return None
    return float(np.interp(t, times, xs)), float(np.interp(t, times, zs))


def path_length_native(times: np.ndarray, xs: np.ndarray, zs: np.ndarray, t_start: float, t_end: float) -> float | None:
    """Piecewise path length over [t_start, t_end]: interpolated start
    point, every real sample strictly between, interpolated end point."""
    start = interpolate_xz(times, xs, zs, t_start)
    end = interpolate_xz(times, xs, zs, t_end)
    if start is None or end is None:
        return None
    mask = (times > t_start) & (times < t_end)
    px = np.concatenate(([start[0]], xs[mask], [end[0]]))
    pz = np.concatenate(([start[1]], zs[mask], [end[1]]))
    return float(np.hypot(np.diff(px), np.diff(pz)).sum())


def bearing_curve(times: np.ndarray, xs: np.ndarray, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """2026-08-13 fix: a forward-difference window [t_i, t_i+W] estimates
    the AVERAGE heading over that span, which for a rotating trajectory
    approximates the INSTANTANEOUS heading at the window's MIDPOINT
    (t_i+W/2), not at t_i -- mislabeling it as "heading at t_i" is a
    first-order lag bias. Caught via synthetic validation: it consistently
    overestimated tick-1 turn by ~0.03-0.04 rad even with zero position
    noise, in every direction/quantization case tested.

    Fixed by using CENTERED differences (window [t_i-W/2, t_i+W/2], which
    has no first-order bias for smoothly/piecewise-varying heading -- the
    residual error is O(W^2), not O(W)) for every candidate time that has
    a full centered window available. This REQUIRES real data on both
    sides of every timestamp being evaluated, including t=0 itself -- so
    callers must supply enough context before the pulse's own start (the
    immediately preceding STRAIGHT run, in the real-data path) rather
    than relying on a one-sided fallback at the edge."""
    W = BEARING_WINDOW_SECONDS
    curve_t, curve_b = [], []
    for t_i in times:
        t_a, t_b = t_i - W / 2.0, t_i + W / 2.0
        if t_a < times[0] or t_b > times[-1]:
            continue
        a = interpolate_xz(times, xs, zs, t_a)
        b = interpolate_xz(times, xs, zs, t_b)
        dx, dz = b[0] - a[0], b[1] - a[1]
        if math.hypot(dx, dz) < MIN_WINDOW_DISPLACEMENT_NATIVE:
            continue
        curve_t.append(t_i)
        curve_b.append(math.atan2(dz, dx))
    if len(curve_t) < 2:
        return None
    curve_t = np.asarray(curve_t, dtype=np.float64)
    curve_b = np.unwrap(np.asarray(curve_b, dtype=np.float64))
    return curve_t, curve_b


def heading_at(curve: tuple[np.ndarray, np.ndarray] | None, t: float) -> float | None:
    if curve is None:
        return None
    times, unwrapped = curve
    if t < times[0] or t > times[-1]:
        return None
    return float(np.interp(t, times, unwrapped))


# ---------------------------------------------------------------------------
# Synthetic validation -- exact closed-form circular-arc ground truth.
# ---------------------------------------------------------------------------

def generate_synthetic_pulse(*, speed_native_per_sec: float, tick1_turn_rad: float, tickrest_turn_rad: float,
                              duration_s: float = 1.6, pre_pulse_seconds: float = 0.3,
                              quantize: bool = True, seed: int = 0) -> pd.DataFrame:
    """Exact circular-arc integration: STRAIGHT (heading=0) for t<0 --
    matching the real preceding context every actual pulse has, required
    now that bearing_curve uses centered differences (see its docstring)
    -- then tick 1: [0,0.2s) at omega1=tick1_turn_rad/0.2; t>=0.2s at
    omega2=tickrest_turn_rad/0.2. Position follows the closed-form
    circular motion x(t)=x0+(v/w)(sin(h0+w*t)-sin(h0)),
    z(t)=z0-(v/w)(cos(h0+w*t)-cos(h0)) -- exact, no integration error."""
    omega1 = tick1_turn_rad / CONTROL_INTERVAL_SECONDS
    omega2 = tickrest_turn_rad / CONTROL_INTERVAL_SECONDS
    rng = np.random.default_rng(seed)
    n = int(round((duration_s + pre_pulse_seconds) / SAMPLE_INTERVAL_SECONDS)) + 1
    times = np.arange(n) * SAMPLE_INTERVAL_SECONDS - pre_pulse_seconds
    # small per-sample timing jitter, matching the real recorder's dt std
    times = times + rng.normal(0.0, 0.00005, size=n)
    times = np.sort(times)

    xs = np.zeros(n)
    zs = np.zeros(n)
    # exact state at the tick1/tick2+ boundary (t=0.2), computed once
    x_boundary, z_boundary, h_boundary = _arc(0.0, 0.0, 0.0, omega1, speed_native_per_sec, CONTROL_INTERVAL_SECONDS)
    for i, t in enumerate(times):
        if t < 0.0:
            x, z, _h = _arc(0.0, 0.0, 0.0, 0.0, speed_native_per_sec, t)  # straight, heading=0
        elif t < CONTROL_INTERVAL_SECONDS:
            x, z, _h = _arc(0.0, 0.0, 0.0, omega1, speed_native_per_sec, t)
        else:
            x, z, _h = _arc(x_boundary, z_boundary, h_boundary, omega2, speed_native_per_sec, t - CONTROL_INTERVAL_SECONDS)
        xs[i], zs[i] = x, z

    if quantize:
        xs = np.round(xs / POSITION_QUANTUM_NATIVE) * POSITION_QUANTUM_NATIVE
        zs = np.round(zs / POSITION_QUANTUM_NATIVE) * POSITION_QUANTUM_NATIVE

    return pd.DataFrame({"elapsed_s": times, "player_x": xs, "player_z": zs})


def _arc(x0: float, z0: float, h0: float, omega: float, speed: float, tau: float) -> tuple[float, float, float]:
    if abs(omega) < 1.0e-9:
        return x0 + speed * math.cos(h0) * tau, z0 + speed * math.sin(h0) * tau, h0
    h1 = h0 + omega * tau
    x1 = x0 + (speed / omega) * (math.sin(h1) - math.sin(h0))
    z1 = z0 - (speed / omega) * (math.cos(h1) - math.cos(h0))
    return x1, z1, h1


def extract_ticks(df: pd.DataFrame, *, t0: float, t_last: float, n_ticks: int) -> list[dict]:
    times = df.elapsed_s.values
    xs = df.player_x.values
    zs = df.player_z.values
    curve = bearing_curve(times, xs, zs)
    rows = []
    for k in range(n_ticks):
        a = t0 + k * CONTROL_INTERVAL_SECONDS
        b = t0 + (k + 1) * CONTROL_INTERVAL_SECONDS
        if b > t_last + 1.0e-9:
            break
        dist_native = path_length_native(times, xs, zs, a, b)
        ha, hb = heading_at(curve, a), heading_at(curve, b)
        turn = (hb - ha) if (ha is not None and hb is not None) else None
        rows.append({
            "tick": k + 1,
            "distance_cells": dist_native / NATIVE_UNITS_PER_CELL if dist_native is not None else None,
            "turn_radians": turn,
        })
    return rows


def run_synthetic_validation() -> bool:
    print("=" * 90)
    print("SYNTHETIC VALIDATION -- known ground truth, exact closed-form circular-arc trajectories")
    print("=" * 90)
    speed = 13.3 * NATIVE_UNITS_PER_CELL  # ~ the independently measured live speed, native units/s
    cases = [
        ("LEFT",  +0.74, +0.87),
        ("RIGHT", -0.74, -0.87),
        ("STRAIGHT", 0.0, 0.0),
    ]
    all_ok = True
    for label, tick1, rest in cases:
        for quantize in (False, True):
            df = generate_synthetic_pulse(speed_native_per_sec=speed, tick1_turn_rad=tick1,
                                           tickrest_turn_rad=rest, quantize=quantize, seed=1)
            ticks = extract_ticks(df, t0=0.0, t_last=df.elapsed_s.iloc[-1], n_ticks=7)
            expected_distance = speed * CONTROL_INTERVAL_SECONDS / NATIVE_UNITS_PER_CELL
            print(f"\n{label} (quantized={quantize}): injected tick1={tick1:.4f} rad, tick2+={rest:.4f} rad, "
                  f"expected distance={expected_distance:.4f} cells/0.2s")
            for row in ticks:
                expected_turn = tick1 if row["tick"] == 1 else rest
                turn_err = abs(row["turn_radians"] - expected_turn) if row["turn_radians"] is not None else float("inf")
                dist_err = abs(row["distance_cells"] - expected_distance) if row["distance_cells"] is not None else float("inf")
                # 2026-08-13: tick1 is the ONLY boundary pair straddling TWO
                # modeled instantaneous rate transitions (t=0: straight->tick1;
                # t=0.2: tick1->steady) -- a centered-difference estimator has an
                # inherent, explained smoothing bias exactly AT an assumed
                # discontinuity, not an unexplained residual (confirmed
                # deterministic: std~0 across 20 seeds at fixed quantize
                # setting; shrinks/grows with window size in a sweep). Directly
                # MEASURED via a 20-seed sweep at W=0.08s before setting these
                # (not guessed): noiseless max|err| tick1=0.0306, tick2=0.0066,
                # tick3+~0.0001; quantized max|err| up to 0.0394 (tick7,
                # position-quantization-phase-dependent, not tick-index-
                # dependent). Tolerances set with margin above those measured
                # worst cases.
                if not quantize:
                    tol_turn = 0.035 if row["tick"] == 1 else (0.01 if row["tick"] == 2 else 0.002)
                else:
                    tol_turn = 0.045
                tol_dist = 0.02 if not quantize else 0.03
                ok = turn_err < tol_turn and dist_err < tol_dist
                all_ok = all_ok and ok
                print(f"  tick={row['tick']}: distance={row['distance_cells']:.4f} (err={dist_err:.4f}) "
                      f"turn={row['turn_radians']:.4f} (expected {expected_turn:.4f}, err={turn_err:.4f}) "
                      f"{'OK' if ok else '*** FAIL ***'}")
    print(f"\n{'ALL SYNTHETIC TESTS PASSED' if all_ok else '*** SOME SYNTHETIC TESTS FAILED -- DO NOT TRUST REAL-DATA RESULTS ***'}")
    return all_ok


# ---------------------------------------------------------------------------
# Real-data extraction (only meaningful once synthetic validation passes).
# ---------------------------------------------------------------------------

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


CONTEXT_SECONDS = 0.30  # matches generate_synthetic_pulse's default pre_pulse_seconds


def extract_real(csv_path: Path) -> pd.DataFrame:
    """2026-08-13 fix: the synthetic validation harness generates pulses
    WITH pre-pulse (t<0) straight-line context, because bearing_curve now
    requires genuine two-sided data to evaluate the centered difference
    right at t=0. This function originally sliced ONLY the pulse's own
    rows (s0:e0) before building the curve -- a different, untested code
    path that silently produced zero tick-1 turn samples on real data,
    even though the validated synthetic path would have worked fine.
    Fixed by concatenating up to CONTEXT_SECONDS of the immediately
    preceding STRAIGHT run (and, symmetrically, the following one) onto
    the pulse's own rows before building the curve -- matching exactly
    what was actually validated."""
    df = pd.read_csv(csv_path)
    df["action"] = df.apply(classify, axis=1)
    runs = raw_runs(df)
    rows = []
    for idx, run in enumerate(runs):
        if run["action"] not in ("LEFT", "RIGHT"):
            continue
        s0, e0 = run["start_idx"], run["end_idx"]
        if not bool(df.focused.values[s0:e0 + 1].all()):
            continue

        pre_run = runs[idx - 1] if idx > 0 else None
        ctx_start = s0
        if pre_run is not None and pre_run["action"] == "STRAIGHT" and bool(df.focused.values[pre_run["start_idx"]:pre_run["end_idx"] + 1].all()):
            pre_ps, pre_pe = pre_run["start_idx"], pre_run["end_idx"]
            cutoff_time = df.elapsed_s.values[pre_pe] - CONTEXT_SECONDS
            ctx_start = pre_ps
            while ctx_start < pre_pe and df.elapsed_s.values[ctx_start] < cutoff_time:
                ctx_start += 1

        post_run = runs[idx + 1] if idx + 1 < len(runs) else None
        ctx_end = e0
        if post_run is not None and post_run["action"] == "STRAIGHT" and bool(df.focused.values[post_run["start_idx"]:post_run["end_idx"] + 1].all()):
            post_ps, post_pe = post_run["start_idx"], post_run["end_idx"]
            cutoff_time = df.elapsed_s.values[post_ps] + CONTEXT_SECONDS
            ctx_end = post_pe
            while ctx_end > post_ps and df.elapsed_s.values[ctx_end] > cutoff_time:
                ctx_end -= 1

        sub = df.iloc[ctx_start:ctx_end + 1]
        t0 = df.elapsed_s.values[s0]
        t_last = df.elapsed_s.values[e0]
        pulse_id = f"{csv_path.name}:{s0}"
        # 2026-08-13 fix (user-caught): extract_ticks(..., n_ticks=6) already
        # emits k=0..5, i.e. ticks 1-6 (window 6 = [t0+1.0, t0+1.2]). The
        # "steady" continuation loop below previously started at k=5,
        # re-emitting that SAME [1.0,1.2] window a second time before
        # advancing -- a real double-count (explains the steady pool jumping
        # from 119/132 to 140/151). Continuation now starts at k=6, the
        # first window NOT already covered by the k=0..5 loop above.
        first_batch = extract_ticks(sub, t0=t0, t_last=t_last, n_ticks=6)
        for row in first_batch:
            if row["distance_cells"] is None:
                continue
            tick_label = row["tick"] if row["tick"] < 5 else "steady" if row["tick"] > 5 else 5
            rows.append({"action": run["action"], "tick": tick_label, "pulse_id": pulse_id, "window_index": row["tick"],
                         "distance_cells": row["distance_cells"], "heading_change_radians": row["turn_radians"]})
        # steady: additional windows beyond the 6 already covered above
        k = 6
        while True:
            a = t0 + k * CONTROL_INTERVAL_SECONDS
            b = t0 + (k + 1) * CONTROL_INTERVAL_SECONDS
            if b > t_last + 1.0e-9:
                break
            times, xs, zs = sub.elapsed_s.values, sub.player_x.values, sub.player_z.values
            curve = bearing_curve(times, xs, zs)
            dist = path_length_native(times, xs, zs, a, b)
            ha, hb = heading_at(curve, a), heading_at(curve, b)
            if dist is not None:
                rows.append({"action": run["action"], "tick": "steady", "pulse_id": pulse_id, "window_index": k + 1,
                             "distance_cells": dist / NATIVE_UNITS_PER_CELL,
                             "heading_change_radians": (hb - ha) if (ha is not None and hb is not None) else None})
            k += 1

        pre_run = runs[idx - 1] if idx > 0 else None
        if pre_run is not None and pre_run["action"] == "STRAIGHT":
            ps, pe = pre_run["start_idx"], pre_run["end_idx"]
            if bool(df.focused.values[ps:pe + 1].all()):
                pre_sub = df.iloc[ps:pe + 1]
                pre_t_last = pre_sub.elapsed_s.iloc[-1]
                pre_t_start = pre_t_last - CONTROL_INTERVAL_SECONDS
                if pre_t_start >= pre_sub.elapsed_s.iloc[0]:
                    times, xs, zs = pre_sub.elapsed_s.values, pre_sub.player_x.values, pre_sub.player_z.values
                    curve = bearing_curve(times, xs, zs)
                    dist = path_length_native(times, xs, zs, pre_t_start, pre_t_last)
                    ha, hb = heading_at(curve, pre_t_start), heading_at(curve, pre_t_last)
                    if dist is not None:
                        rows.append({"action": "STRAIGHT", "tick": "pre-pulse", "pulse_id": pulse_id, "window_index": 0,
                                     "distance_cells": dist / NATIVE_UNITS_PER_CELL,
                                     "heading_change_radians": (hb - ha) if (ha is not None and hb is not None) else None})
    return pd.DataFrame(rows)


def summarize(x: np.ndarray, label: str) -> str:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return f"{label}: n=0"
    return (f"{label}: n={x.size:4d} mean={np.mean(x):7.4f} median={np.median(x):7.4f} std={np.std(x):7.4f} "
            f"p05={np.percentile(x,5):7.4f} p25={np.percentile(x,25):7.4f} "
            f"p75={np.percentile(x,75):7.4f} p95={np.percentile(x,95):7.4f}")


def main() -> None:
    if not run_synthetic_validation():
        print("\nAborting real-data extraction -- fix the extractor first.")
        return

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("movement_calibration_steering.csv")
    print(f"\n\n{'='*90}\nREAL DATA: {csv_path}\n{'='*90}")
    ticks = extract_real(csv_path)
    ticks.to_csv("calibration_tick_extraction_v2.csv", index=False)

    tick_order = [1, 2, 3, 4, 5, "steady"]
    results = {}
    for action in ("LEFT", "RIGHT"):
        for t in tick_order:
            g = ticks[(ticks.action == action) & (ticks.tick == t)]
            d = g.distance_cells.values
            h = g.heading_change_radians.dropna().values
            results[(action, t)] = (d, h)
            print(f"\ntick={t} {action}: " + summarize(d, "distance/0.2s"))
            print(f"{'':>14s}" + summarize(h, "turn/0.2s (rad)") +
                  (f"  [{np.degrees(np.median(h)):.2f}deg median]" if h.size else ""))

    pre = ticks[ticks.tick == "pre-pulse"]
    print(f"\npre-pulse STRAIGHT: " + summarize(pre.distance_cells.values, "distance/0.2s"))
    print(f"{'':>14s}" + summarize(pre.heading_change_radians.dropna().values, "turn/0.2s (rad)"))

    print(f"\n{'='*90}\nLEFT vs RIGHT symmetry per tick\n{'='*90}")
    for t in tick_order:
        dl, hl = results[("LEFT", t)]
        dr, hr = results[("RIGHT", t)]
        dl_med = np.median(dl) if dl.size else float("nan")
        dr_med = np.median(dr) if dr.size else float("nan")
        hl_med = np.median(hl) if hl.size else float("nan")
        hr_med = np.median(hr) if hr.size else float("nan")
        print(f"tick={str(t):>8s}  distance: LEFT={dl_med:.4f} RIGHT={dr_med:.4f}  "
              f"|turn|: LEFT={hl_med:.4f} RIGHT={abs(hr_med):.4f} "
              f"(asymmetry={100*abs(hl_med-abs(hr_med))/max(abs(hl_med),1e-9):.2f}%)")

    print(f"\nCurrent fitted model: LEFT_dist=0.964 LEFT_turn=+0.184  RIGHT_dist=0.966 RIGHT_turn=-0.148 (per 0.2s)")


if __name__ == "__main__":
    main()
