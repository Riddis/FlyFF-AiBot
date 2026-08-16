from __future__ import annotations

from simulator.movement_classification import classify_episode_movement


def test_long_run_with_growing_coverage_is_a_productive_sustained_turn() -> None:
    n = 200
    steering_choices = [1] * n  # constant LEFT the whole episode
    unique_cells_trace = [min(i + 1, n) for i in range(n)]  # steadily gains new cells
    total_distance_trace = [float(i) for i in range(n)]  # steadily moves

    result = classify_episode_movement(
        steering_choices=steering_choices,
        unique_cells_trace=unique_cells_trace,
        total_distance_trace=total_distance_trace,
    )

    assert result["max_consecutive_steering_run"] == n
    assert result["steering_persistent"] is True
    assert result["physical_stagnation"] is False
    assert result["productive_sustained_turn"] is True


def test_long_run_with_flat_coverage_is_physical_stagnation() -> None:
    n = 200
    steering_choices = [2] * n  # constant RIGHT the whole episode
    unique_cells_trace = [5] * n  # never explores past the first few cells
    total_distance_trace = [1.0] * n  # never actually moves

    result = classify_episode_movement(
        steering_choices=steering_choices,
        unique_cells_trace=unique_cells_trace,
        total_distance_trace=total_distance_trace,
    )

    assert result["max_consecutive_steering_run"] == n
    assert result["steering_persistent"] is True
    assert result["physical_stagnation"] is True
    assert result["productive_sustained_turn"] is False


def test_short_alternating_steering_is_neither_persistent_nor_stagnant() -> None:
    n = 200
    steering_choices = [0, 1, 2] * (n // 3) + [0] * (n % 3)
    unique_cells_trace = [min(i + 1, n) for i in range(n)]
    total_distance_trace = [float(i) for i in range(n)]

    result = classify_episode_movement(
        steering_choices=steering_choices,
        unique_cells_trace=unique_cells_trace,
        total_distance_trace=total_distance_trace,
    )

    assert result["steering_persistent"] is False
    assert result["physical_stagnation"] is False
    assert result["productive_sustained_turn"] is False


def test_stagnant_window_outside_the_longest_run_does_not_taint_it() -> None:
    # A brief stagnant patch happens *before* a long, genuinely productive run;
    # the long run itself should still read as productive, not stagnant.
    stagnant_prefix = 100
    productive_run = 150
    n = stagnant_prefix + productive_run
    steering_choices = [0] * stagnant_prefix + [1] * productive_run
    unique_cells_trace = [1] * stagnant_prefix + [
        1 + i for i in range(1, productive_run + 1)
    ]
    total_distance_trace = [0.0] * stagnant_prefix + [
        float(i) for i in range(1, productive_run + 1)
    ]

    result = classify_episode_movement(
        steering_choices=steering_choices,
        unique_cells_trace=unique_cells_trace,
        total_distance_trace=total_distance_trace,
    )

    assert result["max_consecutive_steering_run"] == productive_run
    assert result["productive_sustained_turn"] is True


def test_empty_episode_returns_safe_defaults() -> None:
    result = classify_episode_movement(
        steering_choices=[], unique_cells_trace=[], total_distance_trace=[]
    )
    assert result["max_consecutive_steering_run"] == 0
    assert result["steering_persistent"] is False
    assert result["physical_stagnation"] is False
    assert result["productive_sustained_turn"] is False
