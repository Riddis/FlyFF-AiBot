"""Retained repository-qualified facade for the canonical farming API."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_CANONICAL = (_REPOSITORY_ROOT / "farming" / "__init__.py").resolve()
_canonical_spec = importlib.util.find_spec("farming")
_canonical_origin = (
    Path(_canonical_spec.origin).resolve()
    if _canonical_spec is not None and _canonical_spec.origin is not None
    else None
)
if _canonical_origin != _EXPECTED_CANONICAL:
    raise ImportError(
        "The retained foreground_vision_bot.farming facade requires the "
        f"repository-root farming package at {_EXPECTED_CANONICAL}; got {_canonical_origin}."
    )

from farming import (
    ACTION_COUNT,
    ACTION_NAMES,
    ACTIVE_METADATALESS_MODEL_CONTRACT_HASH,
    ACTIVE_METADATALESS_MODEL_SHA256,
    CURRENT_MODEL_CONTRACT,
    EVENT_NAMES,
    MODEL_CONTRACT_HASH,
    OBSERVATION_FIELDS,
    OBSERVATION_SCHEMA_HASH,
    OBSERVATION_SCHEMA_ID,
    OBSERVATION_SIZE,
    POLICY_ACTION_HEAD_NAMES,
    POLICY_ACTION_NVECS,
    STEERING_NAMES,
    ActorObservation,
    DirectPathState,
    FarmingAction,
    FarmingCommand,
    FarmingEvent,
    FarmingMapFeatures,
    ModelContractError,
    ModelContractMetadata,
    ModelContractValidation,
    ModelSpaceSignature,
    ObservationBuilder,
    ObservationFrame,
    ObservationScales,
    PlayerObservation,
    RewardCalculator,
    RewardComponents,
    RewardConfig,
    RewardEvidence,
    RewardResult,
    SessionClassification,
    SessionEndReason,
    SessionEvidence,
    SessionOutcome,
    SteeringAction,
    classify_session_outcome,
    coerce_farming_action,
    coerce_farming_command,
    validate_model_contract,
)

__all__ = [
    "ACTION_COUNT",
    "ACTION_NAMES",
    "ACTIVE_METADATALESS_MODEL_CONTRACT_HASH",
    "ACTIVE_METADATALESS_MODEL_SHA256",
    "CURRENT_MODEL_CONTRACT",
    "MODEL_CONTRACT_HASH",
    "OBSERVATION_FIELDS",
    "OBSERVATION_SCHEMA_HASH",
    "OBSERVATION_SCHEMA_ID",
    "OBSERVATION_SIZE",
    "ActorObservation",
    "DirectPathState",
    "FarmingAction",
    "EVENT_NAMES",
    "POLICY_ACTION_HEAD_NAMES",
    "POLICY_ACTION_NVECS",
    "STEERING_NAMES",
    "FarmingCommand",
    "FarmingEvent",
    "SteeringAction",
    "FarmingMapFeatures",
    "ModelContractError",
    "ModelContractMetadata",
    "ModelContractValidation",
    "ModelSpaceSignature",
    "ObservationBuilder",
    "ObservationFrame",
    "ObservationScales",
    "PlayerObservation",
    "RewardCalculator",
    "RewardComponents",
    "RewardConfig",
    "RewardEvidence",
    "RewardResult",
    "SessionClassification",
    "SessionEndReason",
    "SessionEvidence",
    "SessionOutcome",
    "classify_session_outcome",
    "coerce_farming_action",
    "coerce_farming_command",
    "validate_model_contract",
]
