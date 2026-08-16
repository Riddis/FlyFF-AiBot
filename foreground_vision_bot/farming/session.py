"""B1 compatibility re-exports for the canonical session contract."""

# BRIDGE B1 — removed in Phase 7
from flyff_farming_simulator.farming.session import (
    SessionClassification,
    SessionEndReason,
    SessionEvidence,
    SessionOutcome,
    classify_session_outcome,
)

__all__ = [
    "SessionClassification",
    "SessionEndReason",
    "SessionEvidence",
    "SessionOutcome",
    "classify_session_outcome",
]
