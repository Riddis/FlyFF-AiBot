"""Classify an episode's movement trace as steering-persistent, physically
stagnant, or a productive sustained turn.

A long run of identical steering choices is not itself evidence of collapse:
following a curved boundary or making a wide, deliberate turn produces the
same signature. Physical stagnation instead requires evidence the player
genuinely stopped making progress -- low unique-cell growth and low
displacement over a trailing window -- independent of what the steering
head happened to output during that window. A window can only be classified
as stagnant if it exhibits that evidence; a persistent steering run without
stagnant windows is a productive sustained turn, not a collapse.
"""

from __future__ import annotations

from typing import Any


def classify_episode_movement(
    *,
    steering_choices: list[int],
    unique_cells_trace: list[int],
    total_distance_trace: list[float],
    window: int = 100,
    min_unique_cell_growth: int = 3,
    min_distance_growth_cells: float = 5.0,
    steering_persistence_fraction: float = 0.30,
) -> dict[str, Any]:
    """``unique_cells_trace`` and ``total_distance_trace`` are the running
    totals reported after each step (monotonically non-decreasing), aligned
    one-to-one with ``steering_choices``.
    """

    n = len(steering_choices)
    if n == 0:
        return {
            "max_consecutive_steering_run": 0,
            "steering_persistent": False,
            "physical_stagnation": False,
            "stagnant_window_count": 0,
            "total_windows": 0,
            "productive_sustained_turn": False,
        }
    if len(unique_cells_trace) != n or len(total_distance_trace) != n:
        raise ValueError("traces must align one-to-one with steering_choices")

    max_run = 1
    run = 1
    run_start = 0
    longest_run_start = 0
    longest_run_end = 0
    for i in range(1, n):
        if steering_choices[i] == steering_choices[i - 1]:
            run += 1
        else:
            run = 1
            run_start = i
        if run > max_run:
            max_run = run
            longest_run_start = run_start
            longest_run_end = i
    steering_persistent = max_run >= steering_persistence_fraction * n

    stagnant_windows = 0
    total_windows = 0
    stagnant_during_longest_run = False
    start = 0
    while start < n - 1:
        end = min(n - 1, start + window)
        if end <= start:
            break
        total_windows += 1
        unique_growth = unique_cells_trace[end] - unique_cells_trace[start]
        distance_growth = total_distance_trace[end] - total_distance_trace[start]
        is_stagnant = unique_growth <= min_unique_cell_growth and distance_growth <= min_distance_growth_cells
        if is_stagnant:
            stagnant_windows += 1
            if start < longest_run_end and end > longest_run_start:
                stagnant_during_longest_run = True
        start += window

    physical_stagnation = stagnant_windows > 0
    productive_sustained_turn = steering_persistent and not stagnant_during_longest_run

    return {
        "max_consecutive_steering_run": max_run,
        "steering_persistent": bool(steering_persistent),
        "physical_stagnation": bool(physical_stagnation),
        "stagnant_window_count": int(stagnant_windows),
        "total_windows": int(total_windows),
        "productive_sustained_turn": bool(productive_sustained_turn),
    }
