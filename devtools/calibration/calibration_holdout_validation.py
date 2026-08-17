"""2026-08-13: HOLDOUT synthetic validation, run end-to-end through
extract_real() itself (the actual function real data goes through,
including pre/post-pulse context concatenation) rather than the lower-
level extract_ticks() helper the first validation grid used exclusively.

Per explicit user instruction: W=0.08 and the tolerances are FROZEN --
this script does not tune anything against these results, it only
reports the errors on parameter combinations NOT used to pick W or the
tolerances. Includes one smooth-ramp case (gradual angular acceleration
via numerical integration) instead of only instantaneous rate
transitions, to check whether the estimator's known transition-boundary
bias is specific to a literal step-function assumption.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from calibration_tick_extraction_v2 import (
    CONTROL_INTERVAL_SECONDS, NATIVE_UNITS_PER_CELL, POSITION_QUANTUM_NATIVE, SAMPLE_INTERVAL_SECONDS,
    _arc, extract_real,
)

PRE_STRAIGHT_SECONDS = 0.5
POST_STRAIGHT_SECONDS = 0.5
PULSE_SECONDS = 1.6


def _write_session_csv(path: Path, *, speed: float, direction: str, angle_fn, quantize: bool, seed: int,
                        phase_offset_s: float = 0.0) -> None:
    """angle_fn(tau) -> heading at time tau since pulse start (tau>=0);
    for tau<0 (pre-pulse), heading is always 0 (straight)."""
    rng = np.random.default_rng(seed)
    total = PRE_STRAIGHT_SECONDS + PULSE_SECONDS + POST_STRAIGHT_SECONDS
    n = int(round(total / SAMPLE_INTERVAL_SECONDS)) + 1
    times = np.arange(n) * SAMPLE_INTERVAL_SECONDS - PRE_STRAIGHT_SECONDS + phase_offset_s
    times = times + rng.normal(0.0, 0.00005, size=n)
    times = np.sort(times)

    xs = np.zeros(n)
    zs = np.zeros(n)
    # numerically integrate heading(t) via angle_fn, then position via
    # fine sub-stepping -- generic, works for both instantaneous-step and
    # smooth-ramp angle_fn definitions without needing a closed form.
    fine_dt = 0.0005
    t_cursor = times[0]
    x_cursor, z_cursor = 0.0, 0.0
    # walk forward from times[0] (negative, pre-pulse) accumulating position
    cursor_idx = 0
    t = t_cursor
    while cursor_idx < n:
        target_t = times[cursor_idx]
        while t < target_t - 1e-12:
            step = min(fine_dt, target_t - t)
            h = angle_fn(t) if t >= 0 else 0.0
            x_cursor += speed * math.cos(h) * step
            z_cursor += speed * math.sin(h) * step
            t += step
        xs[cursor_idx] = x_cursor
        zs[cursor_idx] = z_cursor
        cursor_idx += 1

    if quantize:
        xs = np.round(xs / POSITION_QUANTUM_NATIVE) * POSITION_QUANTUM_NATIVE
        zs = np.round(zs / POSITION_QUANTUM_NATIVE) * POSITION_QUANTUM_NATIVE

    forward = np.ones(n, dtype=int)
    left = np.zeros(n, dtype=int)
    right = np.zeros(n, dtype=int)
    in_pulse = (times >= 0.0) & (times < PULSE_SECONDS)
    if direction == "LEFT":
        left[in_pulse] = 1
    else:
        right[in_pulse] = 1
    jump = np.zeros(n, dtype=int)
    focused = np.ones(n, dtype=int)

    df = pd.DataFrame({"elapsed_s": times, "player_x": xs, "player_z": zs,
                        "forward": forward, "left": left, "right": right, "jump": jump, "focused": focused})
    df.to_csv(path, index=False)


def _expected_turn(tick_or_steady, tick1: float, steady: float) -> float:
    return tick1 if tick_or_steady == 1 else steady


def run_step_grid() -> list[dict]:
    results = []
    speeds = [20.0, 22.4]  # native units/s -- both UNSEEN (tuning used 13.3*1.6=21.28)
    turn_pairs = [(0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]  # (tick1, steady) rad -- all unseen (tuning used 0.74/0.87)
    for speed in speeds:
        for tick1, steady in turn_pairs:
            for quantize in (False, True):
                for direction in ("LEFT", "RIGHT"):
                    sign = 1.0 if direction == "LEFT" else -1.0
                    omega1 = sign * tick1 / CONTROL_INTERVAL_SECONDS
                    omega2 = sign * steady / CONTROL_INTERVAL_SECONDS
                    x_b, z_b, h_b = _arc(0.0, 0.0, 0.0, omega1, speed, CONTROL_INTERVAL_SECONDS)

                    def angle_fn(tau, omega1=omega1, omega2=omega2, h_b=h_b):
                        if tau < CONTROL_INTERVAL_SECONDS:
                            return omega1 * tau
                        return h_b + omega2 * (tau - CONTROL_INTERVAL_SECONDS)

                    path = Path(f"_holdout_tmp.csv")
                    _write_session_csv(path, speed=speed, direction=direction, angle_fn=angle_fn,
                                        quantize=quantize, seed=hash((speed, tick1, steady, quantize, direction)) % 10000)
                    ticks = extract_real(path)
                    ticks = ticks[ticks.action == direction]
                    expected_distance = speed * CONTROL_INTERVAL_SECONDS / NATIVE_UNITS_PER_CELL
                    for _, row in ticks.iterrows():
                        if row["tick"] == "pre-pulse":
                            continue
                        tick_num = row["window_index"]
                        expected_turn = sign * (tick1 if tick_num == 1 else steady)
                        turn_err = abs(row["heading_change_radians"] - expected_turn) if pd.notna(row["heading_change_radians"]) else None
                        dist_err = abs(row["distance_cells"] - expected_distance)
                        results.append({
                            "case": "step", "speed": speed, "tick1": tick1, "steady": steady, "quantize": quantize,
                            "direction": direction, "tick": tick_num, "dist_err": dist_err, "turn_err": turn_err,
                        })
                    path.unlink(missing_ok=True)
    return results


def run_smooth_ramp_grid() -> list[dict]:
    """Angular rate ramps LINEARLY from 0 to omega_steady over
    RAMP_SECONDS, instead of jumping instantaneously at t=0.2 -- tests
    whether the estimator's known transition-boundary bias is an
    artifact of assuming a literal step function."""
    results = []
    RAMP_SECONDS = 0.3
    speed = 21.0  # unseen
    for steady_turn in (0.8, 1.0):
        for quantize in (False, True):
            for direction in ("LEFT", "RIGHT"):
                sign = 1.0 if direction == "LEFT" else -1.0
                omega_steady = sign * steady_turn / CONTROL_INTERVAL_SECONDS

                def angle_fn(tau, omega_steady=omega_steady):
                    if tau < RAMP_SECONDS:
                        # heading = integral of linearly-ramping omega(t) = omega_steady * t/RAMP
                        return 0.5 * omega_steady * (tau ** 2) / RAMP_SECONDS
                    h_at_ramp_end = 0.5 * omega_steady * RAMP_SECONDS
                    return h_at_ramp_end + omega_steady * (tau - RAMP_SECONDS)

                path = Path("_holdout_ramp_tmp.csv")
                _write_session_csv(path, speed=speed, direction=direction, angle_fn=angle_fn,
                                    quantize=quantize, seed=hash((steady_turn, quantize, direction, "ramp")) % 10000)
                ticks = extract_real(path)
                ticks = ticks[ticks.action == direction]
                expected_distance = speed * CONTROL_INTERVAL_SECONDS / NATIVE_UNITS_PER_CELL
                for _, row in ticks.iterrows():
                    if row["tick"] == "pre-pulse":
                        continue
                    tick_num = row["window_index"]
                    t_a = (tick_num - 1) * CONTROL_INTERVAL_SECONDS
                    t_b = tick_num * CONTROL_INTERVAL_SECONDS
                    expected_turn = angle_fn(t_b) - angle_fn(t_a)
                    turn_err = abs(row["heading_change_radians"] - expected_turn) if pd.notna(row["heading_change_radians"]) else None
                    dist_err = abs(row["distance_cells"] - expected_distance)
                    results.append({
                        "case": "smooth_ramp", "speed": speed, "steady": steady_turn, "quantize": quantize,
                        "direction": direction, "tick": tick_num, "expected_turn": expected_turn,
                        "dist_err": dist_err, "turn_err": turn_err,
                    })
                path.unlink(missing_ok=True)
    return results


def main() -> None:
    print("=" * 90)
    print("HOLDOUT VALIDATION -- end-to-end through extract_real(), W=0.08 and tolerances FROZEN")
    print("(reporting errors only, not tuning against these results)")
    print("=" * 90)

    step_results = run_step_grid()
    df = pd.DataFrame(step_results)
    print(f"\n--- STEP-TRANSITION GRID: {len(df)} tick observations across "
          f"{df[['speed','tick1','steady','quantize','direction']].drop_duplicates().shape[0]} unseen parameter combos ---")
    df_dist = df.dropna(subset=["dist_err"])
    print(f"distance error: mean={df_dist.dist_err.mean():.4f} median={df_dist.dist_err.median():.4f} "
          f"max={df_dist.dist_err.max():.4f} (cells)")
    df_turn = df.dropna(subset=["turn_err"])
    for tick in sorted(df_turn.tick.unique(), key=lambda t: (t != 1, t)):
        g = df_turn[df_turn.tick == tick]
        print(f"  tick={tick}: n={len(g)} turn_err mean={g.turn_err.mean():.4f} median={g.turn_err.median():.4f} "
              f"max={g.turn_err.max():.4f} rad")
    missing_turn = df.turn_err.isna().sum()
    print(f"missing turn values (curve context insufficient): {missing_turn}/{len(df)}")

    ramp_results = run_smooth_ramp_grid()
    rdf = pd.DataFrame(ramp_results)
    print(f"\n--- SMOOTH-RAMP GRID: {len(rdf)} tick observations ---")
    rdf_dist = rdf.dropna(subset=["dist_err"])
    print(f"distance error: mean={rdf_dist.dist_err.mean():.4f} max={rdf_dist.dist_err.max():.4f} (cells)")
    rdf_turn = rdf.dropna(subset=["turn_err"])
    for tick in sorted(rdf_turn.tick.unique()):
        g = rdf_turn[rdf_turn.tick == tick]
        print(f"  tick={tick}: n={len(g)} turn_err mean={g.turn_err.mean():.4f} median={g.turn_err.median():.4f} "
              f"max={g.turn_err.max():.4f} rad  (expected_turn examples: {sorted(g.expected_turn.round(3).unique())[:4]})")
    missing_turn_ramp = rdf.turn_err.isna().sum()
    print(f"missing turn values: {missing_turn_ramp}/{len(rdf)}")

    df.to_csv("calibration_holdout_step_results.csv", index=False)
    rdf.to_csv("calibration_holdout_ramp_results.csv", index=False)


if __name__ == "__main__":
    main()
