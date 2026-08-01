from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import pytest
from mapper.RotationModel import (
    DirectionIdleResponseCurve,
    DirectionRotationProfile,
    IdleResponseCurves,
    NeutralTimeoutSample,
    RotationSample,
    RotationTiming,
    StateAwareRotationModel,
    TurnDirection,
    TurnMemoryMode,
    TurnMemoryPolicy,
    TurnTransition,
    TurnTransitionTracker,
    _isotonic,
    fit_neutral_timeout,
    fit_rotation_model,
    validate_neutral_timeout,
)


def _curve(mode: TurnMemoryMode = TurnMemoryMode.DECAYS_TO_NEUTRAL):
    progress = (
        (0.0, 0.25, 0.75, 1.0)
        if mode is TurnMemoryMode.DECAYS_TO_NEUTRAL
        else (0.0, 0.2, 0.55, 0.72)
    )
    return DirectionIdleResponseCurve(
        mode=mode,
        idle_seconds=(0.0, 0.5, 1.0, 2.0),
        response_progress=progress,
        source_sample_count=8,
        observed_horizon_seconds=2.0,
        stateful_response_degrees=20.0,
        reference_response_degrees=30.0,
        maximum_monotonic_adjustment_degrees=0.2,
    )


def _model(mode: TurnMemoryMode = TurnMemoryMode.DECAYS_TO_NEUTRAL):
    profile = DirectionRotationProfile(
        neutral=RotationTiming(300.0, 0.0, 4, 0.2),
        same_direction=RotationTiming(250.0, 0.0, 4, 0.2),
        reversal=RotationTiming(200.0, 0.0, 4, 0.2),
    )
    policy = (
        TurnMemoryPolicy(mode, 2.0, 2.0)
        if mode is TurnMemoryMode.DECAYS_TO_NEUTRAL
        else TurnMemoryPolicy(mode, 2.0)
    )
    curves = IdleResponseCurves(_curve(mode), _curve(mode))
    return StateAwareRotationModel(profile, profile, policy, curves)


def _sample(direction, transition, held, degrees):
    return RotationSample(direction, transition, held, held, held, degrees, 0.9)


def _probe(direction, idle, degrees, transition, uncertainty=0.4):
    return NeutralTimeoutSample(
        direction=direction,
        requested_idle_seconds=idle,
        observed_idle_seconds=idle,
        measured_degrees=degrees,
        uncertainty_degrees=uncertainty,
        confidence=0.9,
        conditioning_transition=transition,
    )


def test_robust_rotation_fit_keeps_reversal_dead_time() -> None:
    samples = [
        _sample(
            TurnDirection.LEFT,
            TurnTransition.REVERSAL,
            duration,
            300.0 * (duration - 0.04),
        )
        for duration in (0.10, 0.14, 0.18, 0.22, 0.26)
    ]
    samples.append(_sample(TurnDirection.LEFT, TurnTransition.REVERSAL, 0.30, 110.0))
    model = fit_rotation_model(
        samples, fallback_left_seconds_90=0.34, fallback_right_seconds_90=0.35
    )
    assert model.left.reversal.rate_degrees_per_second == pytest.approx(300.0)
    assert model.left.reversal.dead_time_seconds == pytest.approx(0.04)
    assert model.left.reversal.sample_count == 5


def test_rotation_fit_uses_suppressed_probe_to_estimate_dead_time() -> None:
    samples = [
        _sample(
            TurnDirection.LEFT,
            TurnTransition.REVERSAL,
            duration,
            degrees,
        )
        for duration, degrees in (
            (0.06, 0.0),
            (0.10, 12.0),
            (0.14, 24.0),
        )
    ]

    model = fit_rotation_model(
        samples,
        fallback_left_seconds_90=0.34,
        fallback_right_seconds_90=0.35,
    )

    assert model.left.reversal.rate_degrees_per_second == pytest.approx(300.0)
    assert model.left.reversal.dead_time_seconds == pytest.approx(0.06)
    assert model.left.reversal.sample_count == 3


def test_rotation_fit_keeps_conservative_dead_time_when_all_probes_suppress() -> None:
    samples = [
        _sample(
            TurnDirection.LEFT,
            TurnTransition.REVERSAL,
            duration,
            0.0,
        )
        for duration in (0.04, 0.06, 0.08)
    ]

    model = fit_rotation_model(
        samples,
        fallback_left_seconds_90=0.34,
        fallback_right_seconds_90=0.35,
    )

    assert model.left.reversal.is_fallback
    assert model.left.reversal.dead_time_seconds == pytest.approx(0.08)
    assert model.left.reversal.sample_count == 3


def test_persistent_tracker_never_discards_direction_from_elapsed_time() -> None:
    tracker = TurnTransitionTracker(
        TurnMemoryPolicy(TurnMemoryMode.PERSISTENT_OBSERVED, 6.0)
    )
    tracker.record(TurnDirection.LEFT, completed_at=0.0)
    transition, idle = tracker.classify(TurnDirection.RIGHT, now=100.0)
    assert transition is TurnTransition.REVERSAL
    assert idle == pytest.approx(100.0)


def test_neutral_tracker_clears_state_only_after_validated_threshold() -> None:
    tracker = TurnTransitionTracker(
        TurnMemoryPolicy(TurnMemoryMode.DECAYS_TO_NEUTRAL, 2.0, 1.5)
    )
    tracker.record(TurnDirection.LEFT, completed_at=0.0)
    assert tracker.classify(TurnDirection.RIGHT, now=1.4)[0] is TurnTransition.REVERSAL
    assert tracker.classify(TurnDirection.RIGHT, now=1.5)[0] is TurnTransition.NEUTRAL


def test_turn_tracker_rejects_boolean_timeout() -> None:
    with pytest.raises(TypeError, match="cannot be boolean"):
        TurnTransitionTracker(True)


def test_persistent_curve_clamps_at_last_observed_knot() -> None:
    curve = _curve(TurnMemoryMode.PERSISTENT_OBSERVED)
    assert curve.progress_at(200.0) == pytest.approx(0.72)


def test_persistent_model_never_blends_state_timing_toward_neutral() -> None:
    model = _model(TurnMemoryMode.PERSISTENT_OBSERVED)

    assert model.seconds_for(
        TurnDirection.LEFT,
        TurnTransition.SAME_DIRECTION,
        60.0,
        idle_seconds=200.0,
    ) == pytest.approx(model.left.same_direction.seconds_for(60.0))
    assert model.seconds_for(
        TurnDirection.LEFT,
        TurnTransition.REVERSAL,
        60.0,
        idle_seconds=200.0,
    ) == pytest.approx(model.left.reversal.seconds_for(60.0))


def test_abrupt_transition_is_valid() -> None:
    curve = DirectionIdleResponseCurve(
        TurnMemoryMode.DECAYS_TO_NEUTRAL,
        (0.0, 0.55, 0.75),
        (0.0, 0.0, 1.0),
        6,
        0.75,
        20.0,
        30.0,
        0.0,
    )
    assert 0.0 < curve.progress_at(0.65) < 1.0


def test_persistent_curve_refuses_fake_neutral_endpoint() -> None:
    with pytest.raises(ValueError, match="must not claim neutral"):
        DirectionIdleResponseCurve(
            TurnMemoryMode.PERSISTENT_OBSERVED,
            (0.0, 1.0),
            (0.0, 1.0),
            2,
            1.0,
            20.0,
            30.0,
            0.0,
        )


def test_model_round_trip_uses_schema_v3() -> None:
    original = _model()
    payload = original.to_dict()
    assert payload["version"] == 3
    assert (
        IdleResponseCurves.from_dict(payload["idle_response_curves"]).to_dict()[
            "version"
        ]
        == 2
    )
    assert StateAwareRotationModel.from_dict(payload) == original


def test_v2_rotation_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 3"):
        StateAwareRotationModel.from_dict({"version": 2})


def test_unknown_previous_direction_uses_conservative_timing() -> None:
    model = _model()
    assert model.seconds_for_degrees("left", 60.0, None) == pytest.approx(0.3)


def _neutral_scan(converged: bool) -> list[NeutralTimeoutSample]:
    samples: list[NeutralTimeoutSample] = []
    delays = (0.40, 0.60, 0.85, 1.20, 1.70, 2.20)
    for direction in TurnDirection:
        for delay, response in zip(
            delays, (20.0, 22.0, 25.0, 28.0, 29.5, 30.0), strict=True
        ):
            samples.append(_probe(direction, delay, response, TurnTransition.REVERSAL))
        same_tail = (30.1, 29.9) if converged else (34.0, 34.1)
        for delay, response in zip((1.70, 2.20), same_tail, strict=True):
            samples.append(
                _probe(direction, delay, response, TurnTransition.SAME_DIRECTION)
            )
    return samples


def test_isotonic_fit_terminates_on_logged_nonmonotonic_shape() -> None:
    values = [
        0.0,
        0.3883,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.016,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    fitted = _isotonic(values)

    assert fitted == pytest.approx(
        [0.0, 0.19415, 0.19415] + [0.3016] * 10,
    )
    assert all(previous <= current for previous, current in pairwise(fitted))


def test_fit_detects_neutral_convergence_from_same_and_reversal_tails() -> None:
    fit = fit_neutral_timeout(_neutral_scan(True), safety_margin_seconds=0.25)
    assert fit.turn_memory_policy.mode is TurnMemoryMode.DECAYS_TO_NEUTRAL
    assert fit.turn_memory_policy.neutral_after_seconds is not None
    assert fit.idle_response_curves.left.response_progress[-1] == 1.0


def test_decay_fit_records_small_uncertainty_bounded_isotonic_adjustment() -> None:
    samples = [
        replace(sample, measured_degrees=21.5)
        if (
            sample.conditioning_transition is TurnTransition.REVERSAL
            and sample.observed_idle_seconds == 0.85
        )
        else sample
        for sample in _neutral_scan(True)
    ]

    fit = fit_neutral_timeout(samples)

    assert (
        fit.idle_response_curves.left.maximum_monotonic_adjustment_degrees
        == pytest.approx(0.25)
    )
    assert (
        fit.idle_response_curves.right.maximum_monotonic_adjustment_degrees
        == pytest.approx(0.25)
    )


def test_decay_fit_rejects_isotonic_correction_beyond_measurement_error() -> None:
    samples = [
        replace(
            sample,
            measured_degrees=(28.0 if sample.observed_idle_seconds == 0.60 else 10.0),
        )
        if (
            sample.conditioning_transition is TurnTransition.REVERSAL
            and sample.observed_idle_seconds in {0.60, 0.85}
        )
        else sample
        for sample in _neutral_scan(True)
    ]

    with pytest.raises(ValueError, match="too non-monotonic"):
        fit_neutral_timeout(samples)


def test_fit_can_use_combined_uncertainty_from_two_strict_headings() -> None:
    samples = [
        replace(sample, uncertainty_degrees=4.0) for sample in _neutral_scan(True)
    ]

    fit = fit_neutral_timeout(
        samples,
        safety_margin_seconds=0.25,
        maximum_sample_uncertainty_degrees=6.0,
    )

    assert fit.turn_memory_policy.mode is TurnMemoryMode.DECAYS_TO_NEUTRAL
    assert fit.left.neutral_uncertainty_degrees == pytest.approx(4.0)
    assert fit.right.neutral_uncertainty_degrees == pytest.approx(4.0)


def test_fit_retains_zero_degree_suppressed_reversal_samples() -> None:
    samples = [
        replace(sample, measured_degrees=0.0)
        if (
            sample.conditioning_transition is TurnTransition.REVERSAL
            and sample.observed_idle_seconds <= 0.60
        )
        else sample
        for sample in _neutral_scan(True)
    ]

    fit = fit_neutral_timeout(samples, safety_margin_seconds=0.25)

    assert fit.idle_response_curves.left.stateful_response_degrees == 0.0
    assert fit.idle_response_curves.right.stateful_response_degrees == 0.0
    assert fit.idle_response_curves.left.response_progress[0] == 0.0


def test_fit_preserves_persistent_direction_state() -> None:
    fit = fit_neutral_timeout(_neutral_scan(False), safety_margin_seconds=0.25)
    assert fit.turn_memory_policy.mode is TurnMemoryMode.PERSISTENT_OBSERVED
    assert fit.turn_memory_policy.neutral_after_seconds is None
    assert fit.neutral_after_seconds is None
    assert set(fit.idle_response_curves.left.response_progress) == {0.0}
    assert set(fit.idle_response_curves.right.response_progress) == {0.0}


def test_c114_nonrepeatable_tail_cannot_become_actionable_curve() -> None:
    left_reversal = (
        (0.431, 14.5),
        (0.474, 7.2),
        (0.633, 22.7),
        (0.951, 22.8),
        (1.163, 22.9),
        (1.361, 6.7),
        (1.703, 0.0),
        (2.213, 10.5),
        (2.805, 11.0),
        (3.512, 7.7),
        (4.316, 8.0),
        (5.111, 7.1),
        (6.005, 2.2),
    )
    left_same = ((4.302, 19.7), (5.106, 15.4), (6.001, 25.1))
    samples = [
        _probe(TurnDirection.LEFT, idle, response, transition, uncertainty=2.0)
        for transition, observations in (
            (TurnTransition.REVERSAL, left_reversal),
            (TurnTransition.SAME_DIRECTION, left_same),
        )
        for idle, response in observations
    ]
    samples.extend(
        sample
        for sample in _neutral_scan(False)
        if sample.direction is TurnDirection.RIGHT
    )

    with pytest.raises(ValueError, match="tail was not repeatable"):
        fit_neutral_timeout(samples)


def test_validation_checks_same_and_reversal_convergence() -> None:
    fit = fit_neutral_timeout(_neutral_scan(True))
    validation = [
        _probe(direction, 2.5, response, transition)
        for direction in TurnDirection
        for transition, response in (
            (TurnTransition.SAME_DIRECTION, 30.1),
            (TurnTransition.REVERSAL, 29.9),
        )
    ]
    validate_neutral_timeout(fit, validation)
    bad = [
        _probe(direction, 2.5, response, transition)
        for direction in TurnDirection
        for transition, response in (
            (TurnTransition.SAME_DIRECTION, 35.0),
            (TurnTransition.REVERSAL, 29.0),
        )
    ]
    with pytest.raises(ValueError, match="did not match"):
        validate_neutral_timeout(fit, bad)

    equally_wrong = [
        _probe(direction, 2.5, 10.0, transition)
        for direction in TurnDirection
        for transition in (
            TurnTransition.SAME_DIRECTION,
            TurnTransition.REVERSAL,
        )
    ]
    with pytest.raises(ValueError, match="absolute neutral response"):
        validate_neutral_timeout(fit, equally_wrong)


def test_validation_requires_persistent_gap_to_repeat() -> None:
    fit = fit_neutral_timeout(_neutral_scan(False))
    persistent = [
        _probe(direction, 2.5, response, transition)
        for direction in TurnDirection
        for transition, response in (
            (TurnTransition.SAME_DIRECTION, 34.0),
            (TurnTransition.REVERSAL, 30.0),
        )
    ]
    validate_neutral_timeout(fit, persistent)

    unexpectedly_neutral = [
        _probe(direction, 2.5, response, transition)
        for direction in TurnDirection
        for transition, response in (
            (TurnTransition.SAME_DIRECTION, 30.1),
            (TurnTransition.REVERSAL, 29.9),
        )
    ]
    with pytest.raises(ValueError, match="persistent direction response"):
        validate_neutral_timeout(fit, unexpectedly_neutral)

    shifted_but_same_gap = [
        _probe(direction, 2.5, response, transition)
        for direction in TurnDirection
        for transition, response in (
            (TurnTransition.SAME_DIRECTION, 44.0),
            (TurnTransition.REVERSAL, 40.0),
        )
    ]
    with pytest.raises(ValueError, match="persistent direction response"):
        validate_neutral_timeout(fit, shifted_but_same_gap)


def test_validation_rejects_unrepeatable_pairs_even_when_medians_match() -> None:
    fit = fit_neutral_timeout(_neutral_scan(False))
    validation = [
        _probe(direction, 2.5, response, transition)
        for direction in TurnDirection
        for transition, responses in (
            (TurnTransition.SAME_DIRECTION, (24.0, 44.0)),
            (TurnTransition.REVERSAL, (20.0, 40.0)),
        )
        for response in responses
    ]

    with pytest.raises(ValueError, match="not repeatable"):
        validate_neutral_timeout(fit, validation)
