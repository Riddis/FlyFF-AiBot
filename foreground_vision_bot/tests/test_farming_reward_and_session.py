from __future__ import annotations

import pytest
from farming.map_features import MapCellRisk
from farming.reward import RewardCalculator, RewardEvidence
from farming.session import (
    SessionClassification,
    SessionEndReason,
    SessionEvidence,
    SessionOutcome,
    classify_session_outcome,
)


def test_sampled_forbidden_entry_is_policy_termination_with_strong_penalty() -> None:
    outcome = classify_session_outcome(
        SessionEvidence(
            sampled_forbidden_traversal=True,
            displacement_cells=40.0,
        )
    )
    assert outcome.reason is SessionEndReason.FORBIDDEN_ZONE_ENTERED
    assert outcome.classification is SessionClassification.POLICY_TERMINATION
    assert outcome.gym_terminated
    assert not outcome.gym_truncated
    assert outcome.policy_caused

    reward = RewardCalculator().calculate(
        RewardEvidence(
            native_kill_delta=1,
            forbidden_distance_cells=0.0,
            session_outcome=outcome,
        )
    )
    assert reward.components.kill == pytest.approx(1.0)
    assert reward.components.teleport_trigger == pytest.approx(-50.0)
    assert reward.total <= -64.0


def test_confirmed_large_jump_is_external_without_policy_penalty() -> None:
    outcome = classify_session_outcome(
        SessionEvidence(
            started_inside_warning_radius=True,
            displacement_cells=30.0,
            external_teleport_confirmed=True,
        )
    )
    assert outcome.reason is SessionEndReason.EXTERNAL_TELEPORT
    assert outcome.classification is SessionClassification.EXTERNAL_TRUNCATION
    assert outcome.gym_truncated
    assert not outcome.policy_caused
    assert not outcome.allow_auto_reset

    reward = RewardCalculator().calculate(
        RewardEvidence(
            native_kill_delta=2,
            density_delta=100,
            elapsed_seconds=5.0,
            eva_attempted=True,
            eva_available=False,
            contact=True,
            forbidden_distance_cells=1.0,
            session_outcome=outcome,
        )
    )
    assert reward.total == pytest.approx(2.0)
    assert reward.components.kill == pytest.approx(2.0)
    assert all(
        value == 0.0
        for name, value in reward.components.as_dict().items()
        if name != "kill"
    )


def test_unconfirmed_coordinate_discontinuity_does_not_end_session() -> None:
    outcome = classify_session_outcome(
        SessionEvidence(
            displacement_cells=100.0,
            external_teleport_confirmed=False,
        )
    )
    assert outcome.reason is SessionEndReason.NONE
    assert outcome.classification is SessionClassification.CONTINUING


def test_jump_flair_reward_is_tiny_and_only_applies_when_performed() -> None:
    calculator = RewardCalculator()
    ordinary = calculator.calculate(RewardEvidence(elapsed_seconds=0.2))
    jumped = calculator.calculate(
        RewardEvidence(elapsed_seconds=0.2, jump_performed=True)
    )

    assert ordinary.components.jump_flair == 0.0
    assert jumped.components.jump_flair == pytest.approx(0.001)
    assert jumped.total - ordinary.total == pytest.approx(0.001)


def test_native_kills_are_the_only_kill_reward_input() -> None:
    reward = RewardCalculator().calculate(
        RewardEvidence(
            native_kill_delta=3,
            density_delta=100,
            elapsed_seconds=0.2,
            contact=True,
        )
    )
    assert reward.components.kill == pytest.approx(3.0)
    assert reward.components.density == pytest.approx(0.2)
    assert reward.components.time == pytest.approx(-0.002)
    assert reward.components.contact == pytest.approx(-0.035)
    assert reward.total == pytest.approx(sum(reward.components.as_dict().values()))


def test_map_risk_penalties_distinguish_buffer_obstacle_and_red_trigger() -> None:
    calculator = RewardCalculator()
    safe = calculator.calculate(RewardEvidence())
    buffer = calculator.calculate(
        RewardEvidence(map_cell_risk=MapCellRisk.OBSTACLE_BUFFER)
    )
    obstacle = calculator.calculate(
        RewardEvidence(map_cell_risk=MapCellRisk.OBSTACLE)
    )
    trigger_outcome = SessionOutcome.forbidden_zone_entered()
    trigger = calculator.calculate(
        RewardEvidence(
            map_cell_risk=MapCellRisk.TELEPORT_TRIGGER,
            forbidden_distance_cells=0.0,
            session_outcome=trigger_outcome,
        )
    )

    assert safe.components.obstacle_buffer == 0.0
    assert safe.components.obstacle_cell == 0.0
    assert buffer.components.obstacle_buffer == pytest.approx(-0.025)
    assert buffer.components.obstacle_cell == 0.0
    assert obstacle.components.obstacle_buffer == 0.0
    assert obstacle.components.obstacle_cell == pytest.approx(-0.75)
    assert abs(trigger.components.teleport_trigger) > abs(
        obstacle.components.obstacle_cell
    )


def test_invalid_eva_and_valid_miss_are_distinct_components() -> None:
    calculator = RewardCalculator()
    invalid = calculator.calculate(
        RewardEvidence(
            density_delta=10,
            eva_attempted=True,
            eva_available=False,
            contact=True,
        )
    )
    miss = calculator.calculate(RewardEvidence(eva_attempted=True, eva_available=True))

    assert invalid.components.invalid_eva == pytest.approx(-0.1)
    assert invalid.components.eva_miss == 0.0
    assert invalid.components.density == 0.0
    assert invalid.components.contact == 0.0
    assert miss.components.invalid_eva == 0.0
    assert miss.components.eva_miss == pytest.approx(-0.05)


def test_user_cancellation_and_fatal_failure_do_not_allow_auto_reset() -> None:
    cancelled = classify_session_outcome(SessionEvidence(user_cancelled=True))
    assert cancelled.reason is SessionEndReason.USER_CANCELLED
    assert cancelled.classification is SessionClassification.USER_CANCELLATION
    assert cancelled.gym_truncated
    assert not cancelled.allow_auto_reset

    fatal = classify_session_outcome(SessionEvidence(fatal_error=RuntimeError("boom")))
    assert fatal.reason is SessionEndReason.FATAL_RUNTIME_ERROR
    assert fatal.should_raise
    assert not fatal.gym_terminated
    assert not fatal.gym_truncated

    calculator = RewardCalculator()
    for outcome in (cancelled, fatal):
        reward = calculator.calculate(
            RewardEvidence(
                native_kill_delta=3,
                density_delta=10,
                elapsed_seconds=1.0,
                eva_attempted=True,
                eva_available=False,
                contact=True,
                forbidden_distance_cells=0.0,
                session_outcome=outcome,
            )
        )
        assert reward.total == 0.0
        assert all(value == 0.0 for value in reward.components.as_dict().values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("native_kill_delta", True),
        ("native_kill_delta", 1.5),
        ("density_delta", False),
        ("density_delta", 2.5),
    ],
)
def test_reward_deltas_require_strict_integer_evidence(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="integer"):
        RewardEvidence(**{field: value})  # type: ignore[arg-type]


def test_every_session_reason_has_one_exhaustive_classification() -> None:
    expected = {
        SessionEndReason.NONE: SessionClassification.CONTINUING,
        SessionEndReason.FORBIDDEN_ZONE_ENTERED: (
            SessionClassification.POLICY_TERMINATION
        ),
        SessionEndReason.EXTERNAL_TELEPORT: (SessionClassification.EXTERNAL_TRUNCATION),
        SessionEndReason.MAP_TRANSITION: SessionClassification.EXTERNAL_TRUNCATION,
        SessionEndReason.CLIENT_EXITED: SessionClassification.EXTERNAL_TRUNCATION,
        SessionEndReason.POINTER_GRACE_EXHAUSTED: (
            SessionClassification.EXTERNAL_TRUNCATION
        ),
        SessionEndReason.USER_CANCELLED: SessionClassification.USER_CANCELLATION,
        SessionEndReason.FOCUS_LOST: SessionClassification.EXTERNAL_TRUNCATION,
        SessionEndReason.FATAL_RUNTIME_ERROR: SessionClassification.FATAL_ERROR,
    }
    assert set(expected) == set(SessionEndReason)

    for reason in SessionEndReason:
        for classification in SessionClassification:
            policy_caused = reason is SessionEndReason.FORBIDDEN_ZONE_ENTERED
            if classification is expected[reason]:
                outcome = SessionOutcome(
                    reason=reason,
                    classification=classification,
                    policy_caused=policy_caused,
                )
                assert outcome.classification is classification
            else:
                with pytest.raises(ValueError, match="requires classification"):
                    SessionOutcome(
                        reason=reason,
                        classification=classification,
                        policy_caused=policy_caused,
                    )

    with pytest.raises(ValueError, match="policy_caused"):
        SessionOutcome(
            reason=SessionEndReason.FORBIDDEN_ZONE_ENTERED,
            classification=SessionClassification.POLICY_TERMINATION,
            policy_caused=False,
        )
