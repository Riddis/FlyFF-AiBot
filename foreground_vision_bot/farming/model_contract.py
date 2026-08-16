"""B1 compatibility re-exports for the canonical model contract."""

# BRIDGE B1 — removed in Phase 7
from flyff_farming_simulator.farming.model_contract import (
    ACTIVE_METADATALESS_MODEL_CONTRACT_HASH,
    ACTIVE_METADATALESS_MODEL_SHA256,
    CURRENT_MODEL_CONTRACT,
    MODEL_CONTRACT_HASH,
    MODEL_CONTRACT_METADATA_VERSION,
    ModelContract,
    ModelContractError,
    ModelContractMetadata,
    ModelContractSource,
    ModelContractValidation,
    ModelSpaceSignature,
    sha256_file,
    validate_model_contract,
)

__all__ = [
    "ACTIVE_METADATALESS_MODEL_CONTRACT_HASH",
    "ACTIVE_METADATALESS_MODEL_SHA256",
    "CURRENT_MODEL_CONTRACT",
    "MODEL_CONTRACT_HASH",
    "MODEL_CONTRACT_METADATA_VERSION",
    "ModelContract",
    "ModelContractError",
    "ModelContractMetadata",
    "ModelContractSource",
    "ModelContractValidation",
    "ModelSpaceSignature",
    "sha256_file",
    "validate_model_contract",
]
