from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from json import dumps
from numbers import Integral
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np

from .actions import (
    EVENT_NAMES,
    POLICY_ACTION_HEAD_NAMES,
    POLICY_ACTION_NVECS,
    STEERING_NAMES,
)
from .observation import OBSERVATION_SIZE, observation_schema_hash
from .observation_contract import OBSERVATION_SCHEMA_HASH, OBSERVATION_SCHEMA_ID

# No metadata-less scalar-action model is compatible with the factorized
# contract. These names remain exported so older reporting/import code fails
# closed with a useful message instead of breaking at import time.
ACTIVE_METADATALESS_MODEL_SHA256: Final = ""
ACTIVE_METADATALESS_MODEL_CONTRACT_HASH: Final = ""
MODEL_CONTRACT_METADATA_VERSION: Final = 2


class ModelContractError(ValueError):
    """A model cannot safely run with the canonical farming semantics."""


def _validate_sha256(value: str, *, field_name: str) -> str:
    normalized = str(value).strip().upper()
    if len(normalized) != 64 or any(c not in "0123456789ABCDEF" for c in normalized):
        raise ModelContractError(f"{field_name} must be a 64-character hexadecimal SHA-256")
    return normalized


_FACTOR_ACTION_NAMES: tuple[str, ...] = tuple(
    [f"STEERING_{name}" for name in STEERING_NAMES]
    + [f"EVENT_{name}" for name in EVENT_NAMES]
)
_FACTOR_ACTION_VALUES: tuple[int, ...] = tuple(range(len(_FACTOR_ACTION_NAMES)))


@dataclass(frozen=True, slots=True)
class ModelContract:
    observation_schema_id: str
    observation_schema_hash: str
    observation_size: int
    action_names: tuple[str, ...]
    action_values: tuple[int, ...]
    action_head_names: tuple[str, ...]
    action_nvec: tuple[int, ...]

    @property
    def action_count(self) -> int:
        return int(np.prod(self.action_nvec, dtype=np.int64))

    def descriptor(self) -> dict[str, object]:
        return {
            "observation": {
                "schema_id": self.observation_schema_id,
                "schema_hash": self.observation_schema_hash,
                "size": self.observation_size,
                "dtype": "float32",
            },
            "action_space": {
                "kind": "MultiDiscrete",
                "head_names": list(self.action_head_names),
                "nvec": list(self.action_nvec),
                "start": [0 for _ in self.action_nvec],
            },
            "factor_labels": [
                {"name": name, "metadata_value": value}
                for name, value in zip(self.action_names, self.action_values, strict=True)
            ],
            "action_execution": {
                "forward_key_latched_while_active": True,
                "steering_head": {"0": "STRAIGHT", "1": "LEFT", "2": "RIGHT"},
                "event_head": {"0": "NONE", "1": "CAST_EVA", "2": "JUMP"},
                "event_does_not_replace_steering": True,
                "release_forward_on_focus_loss_pause_end_or_error": True,
                "jump_action_is_never_masked_or_degraded": True,
                "jump_flair_reward_has_cooldown": True,
            },
        }

    @property
    def semantic_hash(self) -> str:
        return sha256(
            dumps(self.descriptor(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest().upper()


CURRENT_MODEL_CONTRACT = ModelContract(
    observation_schema_id=OBSERVATION_SCHEMA_ID,
    observation_schema_hash=OBSERVATION_SCHEMA_HASH,
    observation_size=OBSERVATION_SIZE,
    action_names=_FACTOR_ACTION_NAMES,
    action_values=_FACTOR_ACTION_VALUES,
    action_head_names=POLICY_ACTION_HEAD_NAMES,
    action_nvec=POLICY_ACTION_NVECS,
)
MODEL_CONTRACT_HASH: Final = CURRENT_MODEL_CONTRACT.semantic_hash


@dataclass(frozen=True, slots=True)
class ModelSpaceSignature:
    observation_shape: tuple[int, ...]
    observation_dtype: str
    action_nvec: tuple[int, ...]
    action_start: tuple[int, ...]

    @property
    def action_count(self) -> int:
        return int(np.prod(self.action_nvec, dtype=np.int64))

    @classmethod
    def from_spaces(cls, observation_space: object, action_space: object) -> "ModelSpaceSignature":
        raw_shape = getattr(observation_space, "shape", None)
        if not isinstance(raw_shape, (tuple, list)):
            raise ModelContractError("Model observation space does not expose a valid shape")
        shape = tuple(_strict_space_integer(v, "observation shape dimension") for v in raw_shape)
        raw_dtype = getattr(observation_space, "dtype", None)
        if raw_dtype is None:
            raise ModelContractError("Model observation space does not expose a dtype")
        try:
            dtype = np.dtype(raw_dtype).name
        except (TypeError, ValueError) as error:
            raise ModelContractError(f"Model observation dtype is invalid: {raw_dtype!r}") from error

        raw_nvec = getattr(action_space, "nvec", None)
        if raw_nvec is None:
            raw_n = getattr(action_space, "n", None)
            if raw_n is not None:
                raise ModelContractError(
                    f"Model action space is scalar Discrete({raw_n}); expected MultiDiscrete{POLICY_ACTION_NVECS}"
                )
            raise ModelContractError("Model action space is not MultiDiscrete")
        nvec_array = np.asarray(raw_nvec)
        if nvec_array.ndim != 1 or nvec_array.size < 1:
            raise ModelContractError("Model MultiDiscrete nvec must be one-dimensional")
        nvec = tuple(_strict_space_integer(v, "action nvec item") for v in nvec_array.tolist())
        raw_start = getattr(action_space, "start", None)
        if raw_start is None:
            start = tuple(0 for _ in nvec)
        else:
            start_array = np.asarray(raw_start)
            if start_array.shape != nvec_array.shape:
                raise ModelContractError("Model MultiDiscrete start shape does not match nvec")
            start = tuple(_strict_space_integer(v, "action start item") for v in start_array.tolist())
        return cls(shape, dtype, nvec, start)


@dataclass(frozen=True, slots=True)
class ModelContractMetadata:
    metadata_version: int
    contract_hash: str
    observation_schema_id: str
    observation_schema_hash: str
    observation_size: int
    action_names: tuple[str, ...]
    action_values: tuple[int, ...]
    action_head_names: tuple[str, ...]
    action_nvec: tuple[int, ...]

    @classmethod
    def current(cls) -> "ModelContractMetadata":
        contract = CURRENT_MODEL_CONTRACT
        return cls(
            MODEL_CONTRACT_METADATA_VERSION,
            contract.semantic_hash,
            contract.observation_schema_id,
            contract.observation_schema_hash,
            contract.observation_size,
            contract.action_names,
            contract.action_values,
            contract.action_head_names,
            contract.action_nvec,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ModelContractMetadata":
        try:
            version = _strict_int(payload["metadata_version"], "metadata_version")
            if version != MODEL_CONTRACT_METADATA_VERSION:
                raise ModelContractError(
                    f"Unsupported farming model metadata version {version}; expected {MODEL_CONTRACT_METADATA_VERSION}"
                )
            return cls(
                version,
                _validate_sha256(str(payload["contract_hash"]), field_name="contract_hash"),
                str(payload["observation_schema_id"]),
                _validate_sha256(str(payload["observation_schema_hash"]), field_name="observation_schema_hash"),
                _strict_int(payload["observation_size"], "observation_size"),
                tuple(str(v) for v in _array(payload, "action_names")),
                tuple(_strict_int(v, "action_values item") for v in _array(payload, "action_values")),
                tuple(str(v) for v in _array(payload, "action_head_names")),
                tuple(_strict_int(v, "action_nvec item") for v in _array(payload, "action_nvec")),
            )
        except KeyError as error:
            raise ModelContractError(f"Model contract metadata is missing {error.args[0]!r}") from error

    def as_dict(self) -> dict[str, object]:
        return {
            "metadata_version": self.metadata_version,
            "contract_hash": self.contract_hash,
            "observation_schema_id": self.observation_schema_id,
            "observation_schema_hash": self.observation_schema_hash,
            "observation_size": self.observation_size,
            "action_names": list(self.action_names),
            "action_values": list(self.action_values),
            "action_head_names": list(self.action_head_names),
            "action_nvec": list(self.action_nvec),
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
    if spaces.observation_shape != (expected.observation_size,):
        raise ModelContractError(
            f"Farming model observation-space mismatch: model has {spaces.observation_shape}, expected {(expected.observation_size,)}"
        )
    if spaces.observation_dtype != "float32":
        raise ModelContractError(
            f"Farming model observation-dtype mismatch: model has {spaces.observation_dtype}, expected float32"
        )
    if spaces.action_nvec != expected.action_nvec:
        raise ModelContractError(
            f"Farming model action-space mismatch: model has MultiDiscrete{spaces.action_nvec}, expected MultiDiscrete{expected.action_nvec}"
        )
    if spaces.action_start != tuple(0 for _ in expected.action_nvec):
        raise ModelContractError(
            f"Farming model action-space start mismatch: model starts at {spaces.action_start}, expected zeros"
        )
    _reject_runtime_semantic_drift()
    if metadata is None:
        raise ModelContractError(
            "Metadata-less farming checkpoints are incompatible with the factorized steering/event contract"
        )
    parsed = metadata if isinstance(metadata, ModelContractMetadata) else ModelContractMetadata.from_mapping(metadata)
    mismatches = _metadata_mismatches(parsed, expected)
    if mismatches:
        raise ModelContractError("Farming model semantic contract mismatch: " + "; ".join(mismatches))
    return ModelContractValidation(
        ModelContractSource.EMBEDDED_METADATA,
        expected.semantic_hash,
        _optional_sha256(artifact_sha256),
    )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _metadata_mismatches(metadata: ModelContractMetadata, expected: ModelContract) -> list[str]:
    pairs: tuple[tuple[str, Any, Any], ...] = (
        ("contract_hash", metadata.contract_hash, expected.semantic_hash),
        ("observation_schema_id", metadata.observation_schema_id, expected.observation_schema_id),
        ("observation_schema_hash", metadata.observation_schema_hash, expected.observation_schema_hash),
        ("observation_size", metadata.observation_size, expected.observation_size),
        ("action_names", metadata.action_names, expected.action_names),
        ("action_values", metadata.action_values, expected.action_values),
        ("action_head_names", metadata.action_head_names, expected.action_head_names),
        ("action_nvec", metadata.action_nvec, expected.action_nvec),
    )
    return [f"{name}={actual!r}, expected {wanted!r}" for name, actual, wanted in pairs if actual != wanted]


def _reject_runtime_semantic_drift() -> None:
    if observation_schema_hash() != OBSERVATION_SCHEMA_HASH:
        raise ModelContractError("Runtime observation semantic drift detected")
    if CURRENT_MODEL_CONTRACT.semantic_hash != MODEL_CONTRACT_HASH:
        raise ModelContractError("Runtime farming model semantic drift detected")


def _strict_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ModelContractError(f"{field_name} must be an integer")
    return int(value)


def _strict_space_integer(value: object, field_name: str) -> int:
    return _strict_int(value, f"Model {field_name}")


def _array(payload: Mapping[str, object], name: str) -> list[object] | tuple[object, ...]:
    value = payload[name]
    if not isinstance(value, (list, tuple)):
        raise ModelContractError(f"{name} must be an array")
    return value


def _optional_sha256(value: str | None) -> str | None:
    return None if value is None else _validate_sha256(value, field_name="artifact_sha256")
