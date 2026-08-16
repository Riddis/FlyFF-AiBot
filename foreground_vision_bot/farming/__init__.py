"""B1 fallback facade for the canonical farming public API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# BRIDGE B1 — removed in Phase 7
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_repository_key = os.path.normcase(str(_REPOSITORY_ROOT.resolve()))
if all(
    os.path.normcase(str(Path(entry or ".").resolve())) != _repository_key
    for entry in sys.path
):
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from flyff_farming_simulator.farming import (
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
