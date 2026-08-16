from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .model_contract import (
    ModelContractMetadata,
    ModelContractValidation,
    ModelSpaceSignature,
    sha256_file,
    validate_model_contract,
)


@dataclass(frozen=True, slots=True)
class ValidatedModel:
    model: object
    artifact_path: Path
    validation: ModelContractValidation


def resolve_model_artifact(path: str | Path) -> Path:
    resolved = Path(path)
    return (
        resolved if resolved.suffix.lower() == ".zip" else resolved.with_suffix(".zip")
    )


def load_and_validate_model(
    path: str | Path,
    loader: Callable[[str], object],
) -> ValidatedModel:
    """Load without a live env and reject incompatibility before input."""

    artifact = resolve_model_artifact(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"No farming policy exists at {artifact}")
    digest = sha256_file(artifact)
    model = loader(str(artifact))
    spaces = ModelSpaceSignature.from_spaces(
        getattr(model, "observation_space", None),
        getattr(model, "action_space", None),
    )
    raw_metadata = getattr(model, "farming_contract_metadata", None)
    metadata: ModelContractMetadata | Mapping[str, object] | None
    if raw_metadata is None or isinstance(raw_metadata, ModelContractMetadata):
        metadata = raw_metadata
    elif isinstance(raw_metadata, Mapping):
        metadata = raw_metadata
    else:
        raise ValueError("Embedded farming model metadata must be a mapping")
    validation = validate_model_contract(
        spaces,
        metadata=metadata,
        artifact_sha256=digest,
    )
    return ValidatedModel(model, artifact, validation)


def validate_new_model(model: object) -> ModelContractValidation:
    spaces = ModelSpaceSignature.from_spaces(
        getattr(model, "observation_space", None),
        getattr(model, "action_space", None),
    )
    return validate_model_contract(
        spaces,
        metadata=ModelContractMetadata.current(),
    )
