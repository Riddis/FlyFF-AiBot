"""Canonical farming domain primitives with dependency-lazy public exports."""

from importlib import import_module
from pkgutil import extend_path


# BRIDGE B1 — removed in Phase 7
__path__ = extend_path(__path__, __name__)

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

_EXPORT_MODULES = {
    **{name: ".actions" for name in (
        "ACTION_COUNT", "ACTION_NAMES", "EVENT_NAMES", "POLICY_ACTION_HEAD_NAMES",
        "POLICY_ACTION_NVECS", "STEERING_NAMES", "FarmingAction", "FarmingCommand",
        "FarmingEvent", "SteeringAction", "coerce_farming_action", "coerce_farming_command",
    )},
    **{name: ".map_features" for name in ("DirectPathState", "FarmingMapFeatures")},
    **{name: ".model_contract" for name in (
        "ACTIVE_METADATALESS_MODEL_CONTRACT_HASH", "ACTIVE_METADATALESS_MODEL_SHA256",
        "CURRENT_MODEL_CONTRACT", "MODEL_CONTRACT_HASH", "ModelContractError",
        "ModelContractMetadata", "ModelContractValidation", "ModelSpaceSignature",
        "validate_model_contract",
    )},
    **{name: ".observation" for name in (
        "OBSERVATION_FIELDS", "OBSERVATION_SCHEMA_HASH", "OBSERVATION_SCHEMA_ID",
        "OBSERVATION_SIZE", "ActorObservation", "ObservationBuilder", "ObservationFrame",
        "ObservationScales", "PlayerObservation",
    )},
    **{name: ".reward" for name in (
        "RewardCalculator", "RewardComponents", "RewardConfig", "RewardEvidence", "RewardResult",
    )},
    **{name: ".session" for name in (
        "SessionClassification", "SessionEndReason", "SessionEvidence", "SessionOutcome",
        "classify_session_outcome",
    )},
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
