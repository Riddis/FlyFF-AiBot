"""Canonical farming domain primitives and explicit live runtime."""

from .actions import ACTION_COUNT, ACTION_NAMES, FarmingAction, coerce_farming_action
from .map_features import DirectPathState, FarmingMapFeatures
from .model_contract import (
    ACTIVE_METADATALESS_MODEL_CONTRACT_HASH,
    ACTIVE_METADATALESS_MODEL_SHA256,
    CURRENT_MODEL_CONTRACT,
    MODEL_CONTRACT_HASH,
    ModelContractError,
    ModelContractMetadata,
    ModelContractValidation,
    ModelSpaceSignature,
    validate_model_contract,
)
from .observation import (
    OBSERVATION_FIELDS,
    OBSERVATION_SCHEMA_HASH,
    OBSERVATION_SCHEMA_ID,
    OBSERVATION_SIZE,
    ActorObservation,
    ObservationBuilder,
    ObservationFrame,
    ObservationScales,
    PlayerObservation,
)
from .reward import (
    RewardCalculator,
    RewardComponents,
    RewardConfig,
    RewardEvidence,
    RewardResult,
)
from .session import (
    SessionClassification,
    SessionEndReason,
    SessionEvidence,
    SessionOutcome,
    classify_session_outcome,
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
    "validate_model_contract",
]
