from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from json import dumps
from numbers import Integral
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np

from .actions import ACTION_NAMES, FarmingAction
from .observation import (
    OBSERVATION_SCHEMA_HASH,
    OBSERVATION_SCHEMA_ID,
    OBSERVATION_SIZE,
    observation_schema_hash,
)

ACTIVE_METADATALESS_MODEL_SHA256: Final = (
    "3ACB0437EA1B7F7BF42DFCDF4DA3B4C097540A702EC856F5AA59BA2D76FADFF2"
)
MODEL_CONTRACT_METADATA_VERSION: Final = 1


class ModelContractError(ValueError):
    """A model cannot safely run with the canonical farming semantics."""


def _validate_sha256(value: str, *, field_name: str) -> str:
    normalized = str(value).strip().upper()
    if len(normalized) != 64 or any(
        character not in "0123456789ABCDEF" for character in normalized
    ):
        raise ModelContractError(
            f"{field_name} must be a 64-character hexadecimal SHA-256"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class ModelContract:
    observation_schema_id: str
    observation_schema_hash: str
    observation_size: int
    action_names: tuple[str, ...]
    action_values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observation_schema_id, str) or not (
            self.observation_schema_id
        ):
            raise ValueError("observation_schema_id cannot be empty")
        object.__setattr__(
            self,
            "observation_schema_hash",
            _validate_sha256(
                self.observation_schema_hash,
                field_name="observation_schema_hash",
            ),
        )
        if isinstance(self.observation_size, bool) or not isinstance(
            self.observation_size,
            Integral,
        ):
            raise ValueError("observation_size must be an integer")
        if self.observation_size < 1:
            raise ValueError("observation_size must be positive")
        object.__setattr__(self, "observation_size", int(self.observation_size))
        if not isinstance(self.action_names, tuple) or any(
            not isinstance(name, str) or not name for name in self.action_names
        ):
            raise ValueError("action names must be a tuple of non-empty strings")
        if not isinstance(self.action_values, tuple) or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in self.action_values
        ):
            raise ValueError("action values must be a tuple of integers")
        object.__setattr__(
            self,
            "action_values",
            tuple(int(value) for value in self.action_values),
        )
        if not self.action_names or len(self.action_names) != len(self.action_values):
            raise ValueError("action names and values must be non-empty and aligned")
        if len(set(self.action_names)) != len(self.action_names):
            raise ValueError("action names must be unique")
        if len(set(self.action_values)) != len(self.action_values):
            raise ValueError("action values must be unique")

    @property
    def action_count(self) -> int:
        return len(self.action_names)

    def descriptor(self) -> dict[str, object]:
        action_pairs = tuple(zip(self.action_names, self.action_values, strict=True))
        return {
            "observation": {
                "schema_id": self.observation_schema_id,
                "schema_hash": self.observation_schema_hash,
                "size": self.observation_size,
                "dtype": "float32",
            },
            "action_space": {
                "kind": "Discrete",
                "count": self.action_count,
                "start": 0,
                "coercion": "integral index only; booleans and foreign enums rejected",
            },
            "actions": [{"name": name, "value": value} for name, value in action_pairs],
            "action_execution": {
                "persistent_movement_values": tuple(
                    value for name, value in action_pairs if name.startswith("RUN_")
                ),
                "cast_eva_value": next(
                    (value for name, value in action_pairs if name == "CAST_EVA"),
                    None,
                ),
                "run_forward_jump_value": next(
                    (
                        value
                        for name, value in action_pairs
                        if name == "RUN_FORWARD_JUMP"
                    ),
                    None,
                ),
                "movement_persists_until_replaced": True,
                "cast_eva_releases_movement": False,
                "jump_always_executes_forward_and_space": True,
                "jump_action_is_never_masked_or_degraded": True,
                "jump_flair_reward_has_cooldown": True,
            },
        }

    @property
    def semantic_hash(self) -> str:
        payload = dumps(
            self.descriptor(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest().upper()


CURRENT_MODEL_CONTRACT = ModelContract(
    observation_schema_id=OBSERVATION_SCHEMA_ID,
    observation_schema_hash=OBSERVATION_SCHEMA_HASH,
    observation_size=OBSERVATION_SIZE,
    action_names=ACTION_NAMES,
    action_values=tuple(int(action) for action in FarmingAction),
)
MODEL_CONTRACT_HASH: Final = (
    "A166C3A6D1349FA4A1AB734834B3171639D1050BF908F1D4585D94F68E108AAC"
)
# The approved metadata-less artifact predates the jump action. It remains
# recorded for explicit rejection/migration diagnostics, but cannot run under
# the five-action contract because its Discrete(4) space fails closed first.
ACTIVE_METADATALESS_MODEL_CONTRACT_HASH: Final = (
    "03E1DA9C110611659DA10DF3CE27117C78E15F9E316ED080E4B75911768A8B18"
)


@dataclass(frozen=True, slots=True)
class ModelSpaceSignature:
    observation_shape: tuple[int, ...]
    observation_dtype: str
    action_count: int
    action_start: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation_shape, tuple) or not self.observation_shape:
            raise ValueError("observation_shape must contain positive dimensions")
        dimensions: list[int] = []
        for dimension in self.observation_shape:
            if isinstance(dimension, bool) or not isinstance(dimension, Integral):
                raise ValueError("observation_shape dimensions must be integers")
            if dimension < 1:
                raise ValueError("observation_shape must contain positive dimensions")
            dimensions.append(int(dimension))
        object.__setattr__(self, "observation_shape", tuple(dimensions))

        try:
            dtype = np.dtype(self.observation_dtype)
        except (TypeError, ValueError) as error:
            raise ValueError("observation_dtype must be a valid NumPy dtype") from error
        object.__setattr__(self, "observation_dtype", dtype.name)

        if isinstance(self.action_count, bool) or not isinstance(
            self.action_count,
            Integral,
        ):
            raise ValueError("action_count must be an integer")
        if self.action_count < 1:
            raise ValueError("action_count must be positive")
        if isinstance(self.action_start, bool) or not isinstance(
            self.action_start,
            Integral,
        ):
            raise ValueError("action_start must be an integer")
        object.__setattr__(self, "action_count", int(self.action_count))
        object.__setattr__(self, "action_start", int(self.action_start))

    @classmethod
    def from_spaces(
        cls,
        observation_space: object,
        action_space: object,
    ) -> "ModelSpaceSignature":
        raw_shape = getattr(observation_space, "shape", None)
        if raw_shape is None:
            raise ModelContractError("Model observation space does not expose a shape")
        if not isinstance(raw_shape, (tuple, list)):
            raise ModelContractError(
                f"Model observation shape is invalid: {raw_shape!r}"
            )
        try:
            shape = tuple(
                _strict_space_integer(dimension, "observation shape dimension")
                for dimension in raw_shape
            )
        except ModelContractError:
            raise
        raw_dtype = getattr(observation_space, "dtype", None)
        if raw_dtype is None:
            raise ModelContractError("Model observation space does not expose a dtype")
        try:
            observation_dtype = np.dtype(raw_dtype).name
        except (TypeError, ValueError) as error:
            raise ModelContractError(
                f"Model observation dtype is invalid: {raw_dtype!r}"
            ) from error
        raw_action_count = getattr(action_space, "n", None)
        if raw_action_count is None:
            raise ModelContractError(
                "Model action space is not a scalar discrete space"
            )
        try:
            action_count = _strict_space_integer(raw_action_count, "action count")
        except ModelContractError as error:
            raise ModelContractError(
                "Model action space is not a scalar discrete space"
            ) from error
        raw_action_start = getattr(action_space, "start", None)
        if raw_action_start is None:
            raise ModelContractError("Model action space does not expose a start index")
        action_start = _strict_space_integer(raw_action_start, "action start")
        return cls(
            observation_shape=shape,
            observation_dtype=observation_dtype,
            action_count=action_count,
            action_start=action_start,
        )


@dataclass(frozen=True, slots=True)
class ModelContractMetadata:
    metadata_version: int
    contract_hash: str
    observation_schema_id: str
    observation_schema_hash: str
    observation_size: int
    action_names: tuple[str, ...]
    action_values: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.metadata_version, bool) or not isinstance(
            self.metadata_version,
            Integral,
        ):
            raise ModelContractError("metadata_version must be an integer")
        if self.metadata_version != MODEL_CONTRACT_METADATA_VERSION:
            raise ModelContractError(
                "Unsupported farming model metadata version "
                f"{self.metadata_version}; expected "
                f"{MODEL_CONTRACT_METADATA_VERSION}"
            )
        object.__setattr__(self, "metadata_version", int(self.metadata_version))
        object.__setattr__(
            self,
            "contract_hash",
            _validate_sha256(self.contract_hash, field_name="contract_hash"),
        )
        object.__setattr__(
            self,
            "observation_schema_hash",
            _validate_sha256(
                self.observation_schema_hash,
                field_name="observation_schema_hash",
            ),
        )
        if not isinstance(self.observation_schema_id, str) or not (
            self.observation_schema_id
        ):
            raise ModelContractError("observation_schema_id cannot be empty")
        if isinstance(self.observation_size, bool) or not isinstance(
            self.observation_size,
            Integral,
        ):
            raise ModelContractError("observation_size must be an integer")
        if self.observation_size < 1:
            raise ModelContractError("observation_size must be positive")
        object.__setattr__(self, "observation_size", int(self.observation_size))
        if not isinstance(self.action_names, tuple) or any(
            not isinstance(name, str) or not name for name in self.action_names
        ):
            raise ModelContractError(
                "action_names must be a tuple of non-empty strings"
            )
        if not isinstance(self.action_values, tuple) or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in self.action_values
        ):
            raise ModelContractError("action_values must be a tuple of integers")
        object.__setattr__(
            self,
            "action_values",
            tuple(int(value) for value in self.action_values),
        )
        if len(self.action_names) != len(self.action_values):
            raise ModelContractError(
                "action_names and action_values must have equal length"
            )

    @classmethod
    def current(cls) -> "ModelContractMetadata":
        contract = CURRENT_MODEL_CONTRACT
        return cls(
            metadata_version=MODEL_CONTRACT_METADATA_VERSION,
            contract_hash=contract.semantic_hash,
            observation_schema_id=contract.observation_schema_id,
            observation_schema_hash=contract.observation_schema_hash,
            observation_size=contract.observation_size,
            action_names=contract.action_names,
            action_values=contract.action_values,
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> "ModelContractMetadata":
        try:
            action_names_value = payload["action_names"]
            action_values_value = payload["action_values"]
            if not isinstance(action_names_value, (list, tuple)):
                raise TypeError("action_names must be an array")
            if not isinstance(action_values_value, (list, tuple)):
                raise TypeError("action_values must be an array")
            return cls(
                metadata_version=_strict_int(
                    payload["metadata_version"],
                    "metadata_version",
                ),
                contract_hash=str(payload["contract_hash"]),
                observation_schema_id=str(payload["observation_schema_id"]),
                observation_schema_hash=str(payload["observation_schema_hash"]),
                observation_size=_strict_int(
                    payload["observation_size"],
                    "observation_size",
                ),
                action_names=tuple(str(value) for value in action_names_value),
                action_values=tuple(
                    _strict_int(value, "action_values item")
                    for value in action_values_value
                ),
            )
        except KeyError as error:
            raise ModelContractError(
                f"Model contract metadata is missing {error.args[0]!r}"
            ) from error
        except (TypeError, ValueError) as error:
            if isinstance(error, ModelContractError):
                raise
            raise ModelContractError(
                f"Invalid model contract metadata: {error}"
            ) from error

    def as_dict(self) -> dict[str, object]:
        return {
            "metadata_version": self.metadata_version,
            "contract_hash": self.contract_hash,
            "observation_schema_id": self.observation_schema_id,
            "observation_schema_hash": self.observation_schema_hash,
            "observation_size": self.observation_size,
            "action_names": list(self.action_names),
            "action_values": list(self.action_values),
        }


class ModelContractSource(str, Enum):
    EMBEDDED_METADATA = "embedded_metadata"
    ACTIVE_LEGACY_HASH = "active_legacy_hash"


@dataclass(frozen=True, slots=True)
class ModelContractValidation:
    source: ModelContractSource
    contract_hash: str
    artifact_sha256: str | None


def validate_model_contract(
    spaces: ModelSpaceSignature,
    *,
    metadata: ModelContractMetadata | Mapping[str, object] | None = None,
    artifact_sha256: str | None = None,
    expected: ModelContract = CURRENT_MODEL_CONTRACT,
) -> ModelContractValidation:
    """Fail closed before input when model dimensions or semantics differ."""

    expected_shape = (expected.observation_size,)
    if spaces.observation_shape != expected_shape:
        raise ModelContractError(
            "Farming model observation-space mismatch: "
            f"model has {spaces.observation_shape}, expected {expected_shape} "
            f"for {expected.observation_schema_id}"
        )
    if spaces.observation_dtype != "float32":
        raise ModelContractError(
            "Farming model observation-dtype mismatch: "
            f"model has {spaces.observation_dtype}, expected float32"
        )
    if spaces.action_count != expected.action_count:
        raise ModelContractError(
            "Farming model action-space mismatch: "
            f"model has Discrete({spaces.action_count}), expected "
            f"Discrete({expected.action_count}) with "
            f"{', '.join(expected.action_names)}"
        )
    if spaces.action_start != 0:
        raise ModelContractError(
            "Farming model action-space start mismatch: "
            f"model starts at {spaces.action_start}, expected 0"
        )

    _reject_runtime_semantic_drift()

    if metadata is not None:
        parsed = (
            metadata
            if isinstance(metadata, ModelContractMetadata)
            else ModelContractMetadata.from_mapping(metadata)
        )
        mismatches = _metadata_mismatches(parsed, expected)
        if mismatches:
            raise ModelContractError(
                "Farming model semantic contract mismatch: " + "; ".join(mismatches)
            )
        return ModelContractValidation(
            source=ModelContractSource.EMBEDDED_METADATA,
            contract_hash=expected.semantic_hash,
            artifact_sha256=_optional_sha256(artifact_sha256),
        )

    digest = _optional_sha256(artifact_sha256)
    if digest is None:
        raise ModelContractError(
            "Metadata-less farming model requires an artifact SHA-256; "
            "resume was rejected before input"
        )
    if digest != ACTIVE_METADATALESS_MODEL_SHA256:
        raise ModelContractError(
            "Unknown metadata-less farming model: dimensions match, but "
            f"SHA-256 {digest} is not the approved active artifact"
        )
    if (
        expected != CURRENT_MODEL_CONTRACT
        or expected.semantic_hash != ACTIVE_METADATALESS_MODEL_CONTRACT_HASH
    ):
        raise ModelContractError(
            "Approved metadata-less farming artifact is bound to contract "
            f"{ACTIVE_METADATALESS_MODEL_CONTRACT_HASH}; runtime semantic drift "
            "was rejected before input"
        )
    return ModelContractValidation(
        source=ModelContractSource.ACTIVE_LEGACY_HASH,
        contract_hash=ACTIVE_METADATALESS_MODEL_CONTRACT_HASH,
        artifact_sha256=digest,
    )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _metadata_mismatches(
    metadata: ModelContractMetadata,
    expected: ModelContract,
) -> list[str]:
    comparisons: tuple[tuple[str, Any, Any], ...] = (
        ("contract_hash", metadata.contract_hash, expected.semantic_hash),
        (
            "observation_schema_id",
            metadata.observation_schema_id,
            expected.observation_schema_id,
        ),
        (
            "observation_schema_hash",
            metadata.observation_schema_hash,
            expected.observation_schema_hash,
        ),
        ("observation_size", metadata.observation_size, expected.observation_size),
        ("action_names", metadata.action_names, expected.action_names),
        ("action_values", metadata.action_values, expected.action_values),
    )
    return [
        f"{name}={actual!r}, expected {wanted!r}"
        for name, actual, wanted in comparisons
        if actual != wanted
    ]


def _strict_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelContractError(f"{field_name} must be an integer")
    return value


def _strict_space_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ModelContractError(f"Model {field_name} must be an integer")
    return int(value)


def _reject_runtime_semantic_drift() -> None:
    runtime_schema_hash = observation_schema_hash()
    if runtime_schema_hash != OBSERVATION_SCHEMA_HASH:
        raise ModelContractError(
            "Runtime observation semantic drift detected: descriptor hash "
            f"{runtime_schema_hash} does not match frozen hash "
            f"{OBSERVATION_SCHEMA_HASH}"
        )
    runtime_contract = ModelContract(
        observation_schema_id=OBSERVATION_SCHEMA_ID,
        observation_schema_hash=runtime_schema_hash,
        observation_size=OBSERVATION_SIZE,
        action_names=ACTION_NAMES,
        action_values=tuple(int(action) for action in FarmingAction),
    )
    if runtime_contract.semantic_hash != MODEL_CONTRACT_HASH:
        raise ModelContractError(
            "Runtime farming model semantic drift detected: contract hash "
            f"{runtime_contract.semantic_hash} does not match frozen hash "
            f"{MODEL_CONTRACT_HASH}"
        )


def _optional_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    return _validate_sha256(value, field_name="artifact_sha256")
