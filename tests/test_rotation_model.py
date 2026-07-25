from __future__ import annotations

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
    TurnTransition,
    fit_neutral_timeout,
    fit_rotation_model,
    validate_neutral_timeout,
)


def _idle_curves(
    *,
    neutral_after_seconds: float = 2.0,
) -> IdleResponseCurves:
    curve = DirectionIdleResponseCurve(
        idle_seconds=(0.0, 0.5, 1.0, neutral_after_seconds),
        response_progress=(0.0, 0.2, 0.7, 1.0),
        source_sample_count=8,
        stateful_response_degrees=20.0,
        neutral_response_degrees=30.0,
        maximum_monotonic_adjustment_degrees=0.2,
    )
    return IdleResponseCurves(left=curve, right=curve)


def _sample(
    direction: TurnDirection,
    transition: TurnTransition,
    held_seconds: float,
    measured_degrees: float,
) -> RotationSample:
    return RotationSample(
        direction=direction,
        transition=transition,
        requested_seconds=held_seconds,
        clamped_seconds=held_seconds,
        held_seconds=held_seconds,
        measured_degrees=measured_degrees,
        confidence=0.9,
    )


def test_robust_fit_recovers_reversal_dead_time_and_rejects_outlier() -> None:
    samples = [
        _sample(
            TurnDirection.LEFT,
            TurnTransition.REVERSAL,
            duration,
            300.0 * (duration - 0.04),
        )
        for duration in (0.10, 0.14, 0.18, 0.22, 0.26)
    ]
    samples.append(
        _sample(
            TurnDirection.LEFT,
            TurnTransition.REVERSAL,
            0.30,
            110.0,
        )
    )

    model = fit_rotation_model(
        samples,
        fallback_left_seconds_90=0.34,
        fallback_right_seconds_90=0.35,
    )
    timing = model.left.reversal

    assert timing.rate_degrees_per_second == pytest.approx(300.0)
    assert timing.dead_time_seconds == pytest.approx(0.04)
    assert timing.sample_count == 5
    assert timing.median_error_degrees == pytest.approx(0.0)
    assert not timing.is_fallback
    assert model.right.reversal.is_fallback


def test_direction_only_lookup_distinguishes_same_and_reversal() -> None:
    profile = DirectionRotationProfile(
        neutral=RotationTiming(250.0, 0.02, 3, 0.5),
        same_direction=RotationTiming(300.0, 0.01, 3, 0.5),
        reversal=RotationTiming(250.0, 0.05, 3, 0.5),
    )
    model = StateAwareRotationModel(
        left=profile,
        right=profile,
        neutral_after_seconds=2.0,
        idle_response_curves=_idle_curves(),
    )

    same = model.seconds_for_degrees(
        "left",
        90.0,
        previous_direction="left",
    )
    reversal = model.seconds_for_degrees(
        "left",
        90.0,
        previous_direction="right",
    )
    unknown = model.seconds_for_degrees(
        "left",
        90.0,
        previous_direction=None,
    )

    assert same == pytest.approx(0.31)
    assert reversal == pytest.approx(0.41)
    assert unknown == pytest.approx(reversal)
    assert model.seconds_for_degrees("left", 0.0, None) == 0.0


def test_unmeasured_neutral_transition_uses_conservative_fitted_response() -> None:
    samples = []
    for duration in (0.10, 0.14, 0.18):
        samples.append(
            _sample(
                TurnDirection.LEFT,
                TurnTransition.SAME_DIRECTION,
                duration,
                300.0 * duration,
            )
        )
        samples.append(
            _sample(
                TurnDirection.LEFT,
                TurnTransition.REVERSAL,
                duration,
                250.0 * (duration - 0.04),
            )
        )

    model = fit_rotation_model(
        samples,
        fallback_left_seconds_90=0.34,
        fallback_right_seconds_90=0.35,
    )

    assert model.left.neutral.is_fallback
    assert model.left.neutral.dead_time_seconds >= model.left.reversal.dead_time_seconds
    assert model.left.neutral.seconds_for(30.0) >= model.left.reversal.seconds_for(30.0)


def test_fixed_duration_neutral_probes_do_not_fit_scheduler_jitter_as_dead_time() -> (
    None
):
    samples = [
        RotationSample(
            direction=TurnDirection.LEFT,
            transition=TurnTransition.NEUTRAL,
            requested_seconds=0.10,
            clamped_seconds=0.10,
            held_seconds=held,
            measured_degrees=degrees,
            confidence=0.9,
        )
        for held, degrees in (
            (0.0998, 29.8),
            (0.1001, 30.2),
            (0.1004, 30.0),
            (0.0999, 30.1),
        )
    ]

    model = fit_rotation_model(
        samples,
        fallback_left_seconds_90=0.34,
        fallback_right_seconds_90=0.35,
    )

    assert not model.left.neutral.is_fallback
    assert model.left.neutral.dead_time_seconds == 0.0
    assert model.left.neutral.seconds_90 == pytest.approx(0.3, abs=0.004)


def test_rotation_model_json_round_trip() -> None:
    timing = RotationTiming(
        rate_degrees_per_second=280.0,
        dead_time_seconds=0.025,
        sample_count=4,
        median_error_degrees=1.2,
    )
    profile = DirectionRotationProfile(timing, timing, timing)
    original = StateAwareRotationModel(
        left=profile,
        right=profile,
        neutral_after_seconds=2.5,
        idle_response_curves=_idle_curves(neutral_after_seconds=2.5),
    )

    restored = StateAwareRotationModel.from_dict(original.to_dict())

    assert restored == original


def test_idle_response_curve_interpolates_transition_toward_neutral() -> None:
    profile = DirectionRotationProfile(
        neutral=RotationTiming(300.0, 0.0, 4, 0.2),
        same_direction=RotationTiming(250.0, 0.0, 4, 0.2),
        reversal=RotationTiming(200.0, 0.0, 4, 0.2),
    )
    model = StateAwareRotationModel(
        left=profile,
        right=profile,
        neutral_after_seconds=2.0,
        idle_response_curves=_idle_curves(),
    )

    at_start = model.seconds_for(
        TurnDirection.LEFT,
        TurnTransition.REVERSAL,
        60.0,
        idle_seconds=0.0,
    )
    halfway_between_knots = model.seconds_for(
        TurnDirection.LEFT,
        TurnTransition.REVERSAL,
        60.0,
        idle_seconds=0.75,
    )
    at_safety_endpoint = model.seconds_for(
        TurnDirection.LEFT,
        TurnTransition.REVERSAL,
        60.0,
        idle_seconds=2.0,
    )

    assert at_start == pytest.approx(0.3)
    # 0.75s interpolates curve progress halfway between 0.2 and 0.7.
    assert halfway_between_knots == pytest.approx(0.255)
    assert at_safety_endpoint == pytest.approx(0.2)


def test_rotation_model_deserialization_requires_idle_response_curve() -> None:
    timing = RotationTiming(280.0, 0.02, 3, 0.5)
    profile = DirectionRotationProfile(timing, timing, timing)
    payload = StateAwareRotationModel(
        left=profile,
        right=profile,
        neutral_after_seconds=2.0,
    ).to_dict()

    with pytest.raises((KeyError, ValueError), match="idle_response|version"):
        StateAwareRotationModel.from_dict(payload)


def _neutral_sample(
    direction: TurnDirection,
    idle_seconds: float,
    measured_degrees: float,
    *,
    uncertainty_degrees: float = 0.6,
) -> NeutralTimeoutSample:
    return NeutralTimeoutSample(
        direction=direction,
        requested_idle_seconds=idle_seconds,
        observed_idle_seconds=idle_seconds + 0.01,
        measured_degrees=measured_degrees,
        uncertainty_degrees=uncertainty_degrees,
        confidence=0.9,
    )


def _valid_neutral_scan() -> list[NeutralTimeoutSample]:
    delays = (0.45, 0.65, 0.90, 1.25, 1.70, 2.25, 2.90, 3.65)
    samples: list[NeutralTimeoutSample] = []
    for direction, responses in (
        (
            TurnDirection.LEFT,
            (20.0, 22.0, 24.0, 29.7, 30.1, 30.0, 29.9, 30.2),
        ),
        (
            TurnDirection.RIGHT,
            (21.0, 23.0, 25.0, 27.0, 31.8, 32.0, 32.2, 31.9),
        ),
    ):
        samples.extend(
            _neutral_sample(direction, delay, response)
            for delay, response in zip(delays, responses, strict=True)
        )
    return samples


def test_neutral_timeout_fit_uses_slower_direction_plus_safety_margin() -> None:
    fit = fit_neutral_timeout(
        _valid_neutral_scan(),
        safety_margin_seconds=0.25,
        maximum_idle_seconds=4.0,
    )

    assert fit.left.last_stateful_seconds == pytest.approx(0.91)
    assert fit.left.first_neutral_seconds == pytest.approx(1.26)
    assert fit.right.last_stateful_seconds == pytest.approx(1.26)
    assert fit.right.first_neutral_seconds == pytest.approx(1.71)
    assert fit.neutral_after_seconds == pytest.approx(1.96)
    assert fit.to_dict()["neutral_after_seconds"] == pytest.approx(1.96)
    assert fit.idle_response_curves.left.progress_at(0.0) == 0.0
    assert 0.0 < fit.idle_response_curves.left.progress_at(1.10) < 1.0
    assert fit.idle_response_curves.left.progress_at(1.96) == 1.0
    assert (
        fit.to_dict()["idle_response_curves"]["version"]  # type: ignore[index]
        == 1
    )

    validation = [
        _neutral_sample(
            direction,
            fit.neutral_after_seconds - 0.01,
            response,
        )
        for direction, response in (
            (TurnDirection.LEFT, 30.2),
            (TurnDirection.LEFT, 29.8),
            (TurnDirection.RIGHT, 32.1),
            (TurnDirection.RIGHT, 31.8),
        )
    ]
    validate_neutral_timeout(fit, validation)


def test_neutral_timeout_fit_rejects_indistinguishable_state_response() -> None:
    samples = [
        _neutral_sample(direction, delay, 30.0)
        for direction in TurnDirection
        for delay in (0.45, 0.65, 0.90, 1.25, 1.70, 2.25)
    ]

    with pytest.raises(ValueError, match="state-dependent response"):
        fit_neutral_timeout(samples, maximum_idle_seconds=3.0)


def test_neutral_timeout_fit_rejects_non_monotonic_decay() -> None:
    samples = _valid_neutral_scan()
    samples = [
        (
            _neutral_sample(sample.direction, 1.70, 24.0)
            if sample.direction is TurnDirection.LEFT
            and abs(sample.requested_idle_seconds - 1.70) < 0.001
            else sample
        )
        for sample in samples
    ]

    with pytest.raises(ValueError, match="non-monotonic"):
        fit_neutral_timeout(samples, maximum_idle_seconds=4.0)


def test_idle_response_curve_rejects_internal_stateful_decrease() -> None:
    samples = _valid_neutral_scan()
    samples = [
        (
            _neutral_sample(sample.direction, 0.90, 20.5)
            if sample.direction is TurnDirection.LEFT
            and abs(sample.requested_idle_seconds - 0.90) < 0.001
            else sample
        )
        for sample in samples
    ]

    with pytest.raises(ValueError, match="curve is non-monotonic"):
        fit_neutral_timeout(samples, maximum_idle_seconds=4.0)


def test_neutral_timeout_fit_rejects_divergent_tail() -> None:
    samples = _valid_neutral_scan()
    samples = [
        (
            _neutral_sample(sample.direction, 3.65, 24.0)
            if sample.direction is TurnDirection.LEFT
            and abs(sample.requested_idle_seconds - 3.65) < 0.001
            else sample
        )
        for sample in samples
    ]

    with pytest.raises(ValueError, match="stable plateau"):
        fit_neutral_timeout(samples, maximum_idle_seconds=4.0)


def test_neutral_timeout_validation_rejects_gradual_underturn() -> None:
    fit = fit_neutral_timeout(
        _valid_neutral_scan(),
        safety_margin_seconds=0.25,
        maximum_idle_seconds=4.0,
    )
    validation = [
        _neutral_sample(
            direction,
            fit.neutral_after_seconds - 0.01,
            response,
        )
        for direction, response in (
            (TurnDirection.LEFT, 24.0),
            (TurnDirection.LEFT, 24.5),
            (TurnDirection.RIGHT, 26.0),
            (TurnDirection.RIGHT, 26.5),
        )
    ]

    with pytest.raises(ValueError, match="did not match"):
        validate_neutral_timeout(fit, validation)
