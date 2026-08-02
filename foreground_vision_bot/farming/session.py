from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class SessionClassification(str, Enum):
    CONTINUING = "continuing"
    POLICY_TERMINATION = "policy_termination"
    EXTERNAL_TRUNCATION = "external_truncation"
    USER_CANCELLATION = "user_cancellation"
    FATAL_ERROR = "fatal_error"


class SessionEndReason(str, Enum):
    NONE = "none"
    FORBIDDEN_ZONE_ENTERED = "forbidden_zone_entered"
    EXTERNAL_TELEPORT = "external_teleport"
    MAP_TRANSITION = "map_transition"
    SESSION_TIME_EXPIRED = "session_time_expired"
    CLIENT_EXITED = "client_exited"
    POINTER_GRACE_EXHAUSTED = "pointer_grace_exhausted"
    USER_CANCELLED = "user_cancelled"
    FOCUS_LOST = "focus_lost"
    FATAL_RUNTIME_ERROR = "fatal_runtime_error"


_CLASSIFICATION_BY_REASON: dict[SessionEndReason, SessionClassification] = {
    SessionEndReason.NONE: SessionClassification.CONTINUING,
    SessionEndReason.FORBIDDEN_ZONE_ENTERED: (SessionClassification.POLICY_TERMINATION),
    SessionEndReason.EXTERNAL_TELEPORT: SessionClassification.EXTERNAL_TRUNCATION,
    SessionEndReason.MAP_TRANSITION: SessionClassification.EXTERNAL_TRUNCATION,
    SessionEndReason.SESSION_TIME_EXPIRED: (SessionClassification.EXTERNAL_TRUNCATION),
    SessionEndReason.CLIENT_EXITED: SessionClassification.EXTERNAL_TRUNCATION,
    SessionEndReason.POINTER_GRACE_EXHAUSTED: (
        SessionClassification.EXTERNAL_TRUNCATION
    ),
    SessionEndReason.USER_CANCELLED: SessionClassification.USER_CANCELLATION,
    SessionEndReason.FOCUS_LOST: SessionClassification.EXTERNAL_TRUNCATION,
    SessionEndReason.FATAL_RUNTIME_ERROR: SessionClassification.FATAL_ERROR,
}
if frozenset(_CLASSIFICATION_BY_REASON) != frozenset(SessionEndReason):
    raise RuntimeError("SessionEndReason classification table is not exhaustive")

_EXTERNAL_END_REASONS = frozenset(
    reason
    for reason, classification in _CLASSIFICATION_BY_REASON.items()
    if classification is SessionClassification.EXTERNAL_TRUNCATION
)


@dataclass(frozen=True, slots=True)
class SessionOutcome:
    """Typed policy/session result used before adapting to Gym semantics."""

    reason: SessionEndReason
    classification: SessionClassification
    policy_caused: bool = False
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.reason, SessionEndReason):
            raise ValueError("reason must be a SessionEndReason")
        if not isinstance(self.classification, SessionClassification):
            raise ValueError("classification must be a SessionClassification")
        if not isinstance(self.policy_caused, bool):
            raise ValueError("policy_caused must be boolean")
        if not isinstance(self.detail, str):
            raise ValueError("detail must be a string")

        expected_classification = _CLASSIFICATION_BY_REASON[self.reason]
        if self.classification is not expected_classification:
            raise ValueError(
                f"{self.reason.value} requires classification "
                f"{expected_classification.value}, not {self.classification.value}"
            )
        expected_policy_cause = self.reason is SessionEndReason.FORBIDDEN_ZONE_ENTERED
        if self.policy_caused is not expected_policy_cause:
            raise ValueError(
                f"{self.reason.value} requires policy_caused={expected_policy_cause}"
            )

    @property
    def should_stop_session(self) -> bool:
        return self.classification is not SessionClassification.CONTINUING

    @property
    def gym_terminated(self) -> bool:
        return self.classification is SessionClassification.POLICY_TERMINATION

    @property
    def gym_truncated(self) -> bool:
        return self.classification in {
            SessionClassification.EXTERNAL_TRUNCATION,
            SessionClassification.USER_CANCELLATION,
        }

    @property
    def allow_auto_reset(self) -> bool:
        """Whether an RL wrapper may silently start another live episode."""

        return self.classification is SessionClassification.CONTINUING

    @property
    def should_raise(self) -> bool:
        return self.classification is SessionClassification.FATAL_ERROR

    @classmethod
    def continuing(cls) -> "SessionOutcome":
        return cls(
            reason=SessionEndReason.NONE,
            classification=SessionClassification.CONTINUING,
        )

    @classmethod
    def forbidden_zone_entered(
        cls,
        detail: str = "sampled trigger occupancy or traversal was proven",
    ) -> "SessionOutcome":
        return cls(
            reason=SessionEndReason.FORBIDDEN_ZONE_ENTERED,
            classification=SessionClassification.POLICY_TERMINATION,
            policy_caused=True,
            detail=detail,
        )

    @classmethod
    def external(
        cls,
        reason: SessionEndReason,
        detail: str = "",
    ) -> "SessionOutcome":
        if not isinstance(reason, SessionEndReason):
            raise ValueError("reason must be a SessionEndReason")
        if reason not in _EXTERNAL_END_REASONS:
            raise ValueError(f"{reason.value} is not an external truncation reason")
        return cls(
            reason=reason,
            classification=SessionClassification.EXTERNAL_TRUNCATION,
            detail=detail,
        )

    @classmethod
    def cancelled(cls, detail: str = "") -> "SessionOutcome":
        return cls(
            reason=SessionEndReason.USER_CANCELLED,
            classification=SessionClassification.USER_CANCELLATION,
            detail=detail,
        )

    @classmethod
    def fatal(cls, detail: str) -> "SessionOutcome":
        return cls(
            reason=SessionEndReason.FATAL_RUNTIME_ERROR,
            classification=SessionClassification.FATAL_ERROR,
            detail=detail,
        )


@dataclass(frozen=True, slots=True)
class SessionEvidence:
    """Evidence collected from coherent before/after samples.

    ``sampled_forbidden_traversal`` must only be set when bounded intermediate
    samples actually observed the trigger.  A line projected across a large
    coordinate jump is deliberately not proof.
    """

    user_cancelled: bool = False
    client_exited: bool = False
    session_time_expired: bool = False
    map_transition: bool = False
    focus_lost: bool = False
    pointer_grace_exhausted: bool = False
    external_teleport_confirmed: bool = False
    sampled_forbidden_occupancy: bool = False
    sampled_forbidden_traversal: bool = False
    started_inside_warning_radius: bool = False
    displacement_cells: float | None = None
    teleport_jump_threshold_cells: float = 25.0
    fatal_error: BaseException | None = None

    def __post_init__(self) -> None:
        threshold = float(self.teleport_jump_threshold_cells)
        if not isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "teleport_jump_threshold_cells must be finite and positive"
            )
        if self.displacement_cells is not None:
            displacement = float(self.displacement_cells)
            if not isfinite(displacement) or displacement < 0.0:
                raise ValueError("displacement_cells must be finite and non-negative")

    @property
    def large_discontinuity(self) -> bool:
        return bool(
            self.displacement_cells is not None
            and self.displacement_cells >= self.teleport_jump_threshold_cells
        )

    @property
    def forbidden_entry_proven(self) -> bool:
        return bool(
            self.sampled_forbidden_occupancy or self.sampled_forbidden_traversal
        )


def classify_session_outcome(evidence: SessionEvidence) -> SessionOutcome:
    """Classify session evidence without string matching or proximity guesses."""

    if evidence.fatal_error is not None:
        return SessionOutcome.fatal(
            f"{type(evidence.fatal_error).__name__}: {evidence.fatal_error}"
        )
    if evidence.user_cancelled:
        return SessionOutcome.cancelled("worker cancellation was requested")
    if evidence.client_exited:
        return SessionOutcome.external(
            SessionEndReason.CLIENT_EXITED,
            "the attached client process exited",
        )
    if evidence.session_time_expired:
        return SessionOutcome.external(
            SessionEndReason.SESSION_TIME_EXPIRED,
            "the farm-map session time expired",
        )
    if evidence.forbidden_entry_proven:
        return SessionOutcome.forbidden_zone_entered()
    if evidence.map_transition:
        return SessionOutcome.external(
            SessionEndReason.MAP_TRANSITION,
            "the selected farm map changed",
        )
    if evidence.external_teleport_confirmed:
        return SessionOutcome.external(
            SessionEndReason.EXTERNAL_TELEPORT,
            "a stable repeated coordinate discontinuity confirmed an external teleport",
        )
    if evidence.pointer_grace_exhausted:
        return SessionOutcome.external(
            SessionEndReason.POINTER_GRACE_EXHAUSTED,
            "the bounded player-pointer grace period expired",
        )
    if evidence.focus_lost:
        return SessionOutcome.external(
            SessionEndReason.FOCUS_LOST,
            "the client lost input focus",
        )
    # Starting near the warning radius is intentionally not policy proof.
    return SessionOutcome.continuing()
