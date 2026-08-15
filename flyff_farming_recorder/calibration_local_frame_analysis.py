"""2026-08-13: intra-tick trajectory geometry analysis, per explicit user
instruction to check what actually happens WITHIN a 0.20s control tick
(not just its net distance/turn totals) before implementing anything.

The current simulator's _move_player() does: heading += full_turn (all at
once), THEN move full_distance in a straight line along the new heading.
That approximation was mild for the legacy ~10deg turns; at the newly-
calibrated ~50deg/0.2s it is a large geometric error (confirmed against
the user's own worked example below).

Uses ONLY the already-recorded deployment-matched calibration
(movement_calibration_steering.csv) -- no new recording. For every
qualifying 0.20s LEFT/RIGHT window, transforms the dense 50Hz trajectory
into the LOCAL FRAME of the window's start heading, and compares four
candidate intra-tick kinematics models against the actual measured path:
  (a) turn-then-translate (current simulator);
  (b) translate-then-turn;
  (c) straight chord along the midpoint heading;
  (d) constant-curvature arc (uses the window's own measured path length
      and heading delta as the arc's length/angle).
Reports endpoint RMSE and intermediate-path-shape RMSE for each, on the
REAL data, without tuning toward any expected winner.

The local-frame extractor itself is validated against synthetic ground
truth (both an arc-shaped and a turn-then-translate-shaped trajectory)
BEFORE being trusted on real data, same discipline as the tick-level
timestamp extractor.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from calibration_tick_extraction_v2 import (
    CONTROL_INTERVAL_SECONDS, NATIVE_UNITS_PER_CELL, POSITION_QUANTUM_NATIVE, SAMPLE_INTERVAL_SECONDS,
    _arc, bearing_curve, classify, heading_at, interpolate_xz, path_length_native, raw_runs,
)

S_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # intermediate comparison points
CONTEXT_SECONDS = 0.30


# ---------------------------------------------------------------------------
# Local-frame trajectory extractor
# ---------------------------------------------------------------------------

def local_frame_trajectory(times: np.ndarray, xs: np.ndarray, zs: np.ndarray, curve, t_start: float, t_end: float):
    """Returns dict with distance_cells, turn_radians, endpoint (forward,
    lateral) in cells, and intermediate (s, forward, lateral) samples at
    S_FRACTIONS -- all in the LOCAL FRAME of the heading at t_start
    (forward = start-heading direction, lateral = +90deg-left of it,
    matching the established sin(relative_angle)>0=LEFT convention:
    local_forward = dx*cos(h0)+dz*sin(h0), local_lateral = -dx*sin(h0)+dz*cos(h0)).
    Returns None if any required boundary/heading is unavailable."""
    h0 = heading_at(curve, t_start)
    if h0 is None:
        return None
    x0 = interpolate_xz(times, xs, zs, t_start)
    x1 = interpolate_xz(times, xs, zs, t_end)
    if x0 is None or x1 is None:
        return None
    h1 = heading_at(curve, t_end)
    dist_native = path_length_native(times, xs, zs, t_start, t_end)
    if dist_native is None:
        return None
    cos_h, sin_h = math.cos(h0), math.sin(h0)

    def to_local(x, z):
        dx, dz = x - x0[0], z - x0[1]
        return dx * cos_h + dz * sin_h, -dx * sin_h + dz * cos_h

    ef, el = to_local(*x1)
    intermediate = []
    for s in S_FRACTIONS:
        t = t_start + s * (t_end - t_start)
        p = interpolate_xz(times, xs, zs, t)
        if p is None:
            continue
        f, l = to_local(*p)
        intermediate.append((s, f / NATIVE_UNITS_PER_CELL, l / NATIVE_UNITS_PER_CELL))

    return {
        "distance_cells": dist_native / NATIVE_UNITS_PER_CELL,
        "turn_radians": (h1 - h0) if h1 is not None else None,
        "endpoint_forward_cells": ef / NATIVE_UNITS_PER_CELL,
        "endpoint_lateral_cells": el / NATIVE_UNITS_PER_CELL,
        "intermediate": intermediate,
    }


# ---------------------------------------------------------------------------
# Candidate kinematics models -- each predicts local-frame (forward,
# lateral) as a function of normalized time/arc-length fraction s, given
# the window's own measured distance (path length) and turn (heading delta).
# ---------------------------------------------------------------------------

def predict_turn_then_translate(distance: float, turn: float, s: float) -> tuple[float, float]:
    return s * distance * math.cos(turn), s * distance * math.sin(turn)


def predict_translate_then_turn(distance: float, turn: float, s: float) -> tuple[float, float]:
    return s * distance, 0.0


def predict_midpoint_chord(distance: float, turn: float, s: float) -> tuple[float, float]:
    half = turn / 2.0
    return s * distance * math.cos(half), s * distance * math.sin(half)


def predict_constant_curvature(distance: float, turn: float, s: float) -> tuple[float, float]:
    if abs(turn) < 1.0e-9:
        return s * distance, 0.0
    r = distance / turn
    return r * math.sin(turn * s), r * (1.0 - math.cos(turn * s))


CANDIDATES = {
    "turn_then_translate (current simulator)": predict_turn_then_translate,
    "translate_then_turn": predict_translate_then_turn,
    "midpoint_chord": predict_midpoint_chord,
    "constant_curvature_arc": predict_constant_curvature,
}


def predict_short_ramp(distance: float, turn: float, s: float, ramp_fraction: float = 0.25) -> tuple[float, float]:
    """A CONTINUOUS (not discontinuous) approximation of "sharp turn then
    straight": angular rate ramps linearly from 0 up to a peak over the
    first ramp_fraction of the tick, then holds constant -- unlike
    turn_then_translate this has no literal heading jump, so it's a
    fair (well-posed) synthetic ground truth for validating a
    continuous-trajectory extractor, while still being a much sharper/
    earlier turn than the smooth constant-curvature arc."""
    if s <= 0.0:
        return 0.0, 0.0
    omega_peak = turn / (1.0 - ramp_fraction / 2.0)
    n_steps = max(4, int(round(400 * s)))
    ds = s / n_steps
    heading = 0.0
    x, z = 0.0, 0.0
    s_cursor = 0.0
    for _ in range(n_steps):
        s_mid = s_cursor + ds / 2.0
        omega = omega_peak * (s_mid / ramp_fraction) if s_mid < ramp_fraction else omega_peak
        heading += omega * ds
        x += distance * ds * math.cos(heading)
        z += distance * ds * math.sin(heading)
        s_cursor += ds
    return x, z


# ---------------------------------------------------------------------------
# Synthetic validation of the extractor itself, BEFORE trusting real data.
# ---------------------------------------------------------------------------

def generate_shaped_trajectory(shape: str, *, distance: float, turn: float, quantize: bool, seed: int,
                                pre_context_seconds: float = 0.3, post_context_seconds: float = 0.3) -> pd.DataFrame:
    """Ground truth generated directly from a KNOWN candidate shape
    (not necessarily the arc) via fine numerical sub-stepping, so the
    extractor's recovery can be checked against each shape independently
    -- proves the extractor isn't secretly biased toward detecting arcs.

    Includes pre_context_seconds of t<0 straight-line (heading=0) motion,
    matching the real preceding-STRAIGHT-run context every real pulse
    has -- required for heading_at(t=0) to be well-defined. ALSO includes
    post_context_seconds of t>0.2 motion continuing the shape's own final
    heading in a straight line, required for heading_at(t=0.2) to have
    a full centered window on its far side too (same "far boundary needs
    context" issue found and fixed in the tick-level extractor)."""
    rng = np.random.default_rng(seed)
    total = pre_context_seconds + CONTROL_INTERVAL_SECONDS + post_context_seconds
    n = int(round(total / SAMPLE_INTERVAL_SECONDS)) + 1
    times = np.arange(n) * SAMPLE_INTERVAL_SECONDS - pre_context_seconds + rng.normal(0.0, 0.00005, size=n)
    times = np.sort(times)
    fn = CANDIDATES.get(shape) or {"constant_curvature_arc": predict_constant_curvature,
                                    "short_ramp": predict_short_ramp}[shape]
    final_f, final_l = fn(distance, turn, 1.0)
    # final heading, from a tiny step before s=1, for the post-context straight continuation
    eps = 1.0e-4
    f0, l0 = fn(distance, turn, 1.0 - eps)
    final_heading = math.atan2(final_l - l0, final_f - f0)
    xs = np.zeros(n)
    zs = np.zeros(n)
    for i, t in enumerate(times):
        if t < 0.0:
            xs[i], zs[i] = t * (distance / CONTROL_INTERVAL_SECONDS) * NATIVE_UNITS_PER_CELL, 0.0
        elif t <= CONTROL_INTERVAL_SECONDS:
            s = max(0.0, min(1.0, t / CONTROL_INTERVAL_SECONDS))
            f, l = fn(distance, turn, s)
            xs[i], zs[i] = f * NATIVE_UNITS_PER_CELL, l * NATIVE_UNITS_PER_CELL
        else:
            over = t - CONTROL_INTERVAL_SECONDS
            speed = distance / CONTROL_INTERVAL_SECONDS
            f = final_f + speed * over * math.cos(final_heading)
            l = final_l + speed * over * math.sin(final_heading)
            xs[i], zs[i] = f * NATIVE_UNITS_PER_CELL, l * NATIVE_UNITS_PER_CELL
    if quantize:
        xs = np.round(xs / POSITION_QUANTUM_NATIVE) * POSITION_QUANTUM_NATIVE
        zs = np.round(zs / POSITION_QUANTUM_NATIVE) * POSITION_QUANTUM_NATIVE
    return pd.DataFrame({"elapsed_s": times, "player_x": xs, "player_z": zs})


def run_synthetic_validation() -> bool:
    """2026-08-13b: dropped turn_then_translate as a ground-truth SHAPE
    here (kept as a candidate PREDICTION formula below, that's unrelated)
    -- it requires a literal instantaneous heading jump, which no
    physically-continuous trajectory can have, so no continuous-
    trajectory extractor can be expected to recover it (confirmed
    directly: error stayed ~0.7-0.94 cells regardless of how much pre/
    post context was supplied, while the genuinely continuous arc case's
    error, ~0.05-0.14 cells, matched the ALREADY-characterized heading-
    boundary bias from the tick-level extractor almost exactly). Testing
    against an ill-posed case doesn't validate anything; it just measures
    how ill-posed it is. Validated instead against TWO continuous shapes:
    the arc, and a short-ramp (angular rate ramps over the first quarter
    of the tick, then holds) that's still a fair, physically-realizable
    trajectory but structurally different from a pure arc."""
    print("=" * 90)
    print("LOCAL-FRAME EXTRACTOR SYNTHETIC VALIDATION (continuous ground truth only)")
    print("=" * 90)
    all_ok = True
    for shape_name, shape_fn in [("constant_curvature_arc", predict_constant_curvature),
                                   ("short_ramp", predict_short_ramp)]:
        for distance, turn in [(2.74, 0.874), (2.74, 0.80), (2.5, 0.5)]:
            for quantize in (False, True):
                df = generate_shaped_trajectory(shape_name, distance=distance, turn=turn, quantize=quantize, seed=3)
                times, xs, zs = df.elapsed_s.values, df.player_x.values, df.player_z.values
                curve = bearing_curve(times, xs, zs)
                result = local_frame_trajectory(times, xs, zs, curve, 0.0, CONTROL_INTERVAL_SECONDS)
                if result is None:
                    print(f"{shape_name} dist={distance} turn={turn} quantize={quantize}: NO RESULT -- FAIL")
                    all_ok = False
                    continue
                expected_ef, expected_el = shape_fn(distance, turn, 1.0)
                ef_err = abs(result["endpoint_forward_cells"] - expected_ef)
                el_err = abs(result["endpoint_lateral_cells"] - expected_el)
                # 2026-08-13b: measured directly via an 8-15 seed sweep per
                # shape before setting this (not guessed): arc max|err| up
                # to 0.141 cells (its t=0 rate discontinuity, matching the
                # already-characterized heading-boundary bias); short_ramp
                # (continuous even in rate, only accel is discontinuous)
                # max|err| only up to 0.034 cells. One tolerance covering
                # both with margin.
                tol = 0.18
                ok = ef_err < tol and el_err < tol
                all_ok = all_ok and ok
                print(f"{shape_name} dist={distance} turn={turn} quantize={quantize}: "
                      f"endpoint=({result['endpoint_forward_cells']:.4f},{result['endpoint_lateral_cells']:.4f}) "
                      f"expected=({expected_ef:.4f},{expected_el:.4f}) err=({ef_err:.4f},{el_err:.4f}) "
                      f"{'OK' if ok else '*** FAIL ***'}")
    print(f"\n{'ALL LOCAL-FRAME SYNTHETIC TESTS PASSED' if all_ok else '*** SOME FAILED -- DO NOT TRUST REAL-DATA RESULTS ***'}")
    return all_ok


# ---------------------------------------------------------------------------
# Real-data analysis.
# ---------------------------------------------------------------------------

def collect_real_windows(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    df["action"] = df.apply(classify, axis=1)
    runs = raw_runs(df)
    windows = []
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
            cutoff = df.elapsed_s.values[pre_pe] - CONTEXT_SECONDS
            ctx_start = pre_ps
            while ctx_start < pre_pe and df.elapsed_s.values[ctx_start] < cutoff:
                ctx_start += 1
        post_run = runs[idx + 1] if idx + 1 < len(runs) else None
        ctx_end = e0
        if post_run is not None and post_run["action"] == "STRAIGHT" and bool(df.focused.values[post_run["start_idx"]:post_run["end_idx"] + 1].all()):
            post_ps, post_pe = post_run["start_idx"], post_run["end_idx"]
            cutoff = df.elapsed_s.values[post_ps] + CONTEXT_SECONDS
            ctx_end = post_pe
            while ctx_end > post_ps and df.elapsed_s.values[ctx_end] > cutoff:
                ctx_end -= 1

        sub = df.iloc[ctx_start:ctx_end + 1]
        times, xs, zs = sub.elapsed_s.values, sub.player_x.values, sub.player_z.values
        curve = bearing_curve(times, xs, zs)
        t0 = df.elapsed_s.values[s0]
        t_last = df.elapsed_s.values[e0]
        k = 0
        while True:
            a, b = t0 + k * CONTROL_INTERVAL_SECONDS, t0 + (k + 1) * CONTROL_INTERVAL_SECONDS
            if b > t_last + 1.0e-9:
                break
            result = local_frame_trajectory(times, xs, zs, curve, a, b)
            if result is not None and result["turn_radians"] is not None:
                result["action"] = run["action"]
                result["tick"] = "1" if k == 0 else "steady"
                windows.append(result)
            k += 1
    return windows


def rmse_report(windows: list[dict], label: str) -> None:
    if not windows:
        print(f"{label}: n=0")
        return
    print(f"\n{label} (n={len(windows)}):")
    mean_distance = float(np.mean([w["distance_cells"] for w in windows]))
    mean_turn = float(np.mean([w["turn_radians"] for w in windows]))
    print(f"  mean distance={mean_distance:.4f} cells, mean turn={mean_turn:.4f} rad ({math.degrees(mean_turn):.2f}deg)")
    print(f"  actual mean endpoint=({np.mean([w['endpoint_forward_cells'] for w in windows]):.4f}, "
          f"{np.mean([w['endpoint_lateral_cells'] for w in windows]):.4f})")

    for name, fn in CANDIDATES.items():
        endpoint_sq_errs = []
        path_sq_errs = []
        for w in windows:
            d, t = w["distance_cells"], w["turn_radians"]
            pf, pl = fn(d, t, 1.0)
            ef, el = w["endpoint_forward_cells"], w["endpoint_lateral_cells"]
            endpoint_sq_errs.append((pf - ef) ** 2 + (pl - el) ** 2)
            for s, af, al in w["intermediate"]:
                mf, ml = fn(d, t, s)
                path_sq_errs.append((mf - af) ** 2 + (ml - al) ** 2)
        endpoint_rmse = math.sqrt(np.mean(endpoint_sq_errs))
        path_rmse = math.sqrt(np.mean(path_sq_errs))
        print(f"    {name:45s} endpoint_RMSE={endpoint_rmse:.4f} cells   path_RMSE={path_rmse:.4f} cells")


def main() -> None:
    if not run_synthetic_validation():
        print("\nAborting real-data analysis -- fix the extractor first.")
        return

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("movement_calibration_steering.csv")
    print(f"\n\n{'='*90}\nREAL DATA: {csv_path}\n{'='*90}")
    windows = collect_real_windows(csv_path)

    # mirror RIGHT (negate turn and lateral) so LEFT/RIGHT can be pooled
    mirrored = []
    for w in windows:
        w2 = dict(w)
        if w["action"] == "RIGHT":
            w2["turn_radians"] = -w["turn_radians"]
            w2["endpoint_lateral_cells"] = -w["endpoint_lateral_cells"]
            w2["intermediate"] = [(s, f, -l) for s, f, l in w["intermediate"]]
        mirrored.append(w2)

    for action in ("LEFT", "RIGHT"):
        for tick in ("1", "steady"):
            rmse_report([w for w in windows if w["action"] == action and w["tick"] == tick], f"{action} tick={tick}")

    print(f"\n{'='*90}\nPOOLED (RIGHT mirrored to LEFT sign)\n{'='*90}")
    for tick in ("1", "steady"):
        rmse_report([w for w in mirrored if w["tick"] == tick], f"POOLED tick={tick}")


if __name__ == "__main__":
    main()
