from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from farming.model_contract import (
    CURRENT_MODEL_CONTRACT,
    MODEL_CONTRACT_HASH,
    ModelContractError,
    ModelContractMetadata,
    ModelContractSource,
    ModelSpaceSignature,
    validate_model_contract,
)


def _current_spaces() -> ModelSpaceSignature:
    return ModelSpaceSignature(
        observation_shape=(923,),
        observation_dtype="float32",
        action_nvec=(3, 3),
        action_start=(0, 0),
    )


def test_factorized_metadata_binds_latched_forward_contract() -> None:
    metadata = ModelContractMetadata.current()
    validation = validate_model_contract(_current_spaces(), metadata=metadata)

    assert metadata.contract_hash == MODEL_CONTRACT_HASH
    assert metadata.action_head_names == ("steering", "event")
    assert metadata.action_nvec == (3, 3)
    assert validation.source is ModelContractSource.EMBEDDED_METADATA
    descriptor = CURRENT_MODEL_CONTRACT.descriptor()
    assert descriptor["action_space"] == {
        "kind": "MultiDiscrete",
        "head_names": ["steering", "event"],
        "nvec": [3, 3],
        "start": [0, 0],
    }
    assert descriptor["action_execution"]["forward_key_latched_while_active"] is True
    assert descriptor["action_execution"]["event_does_not_replace_steering"] is True


def test_scalar_discrete_model_is_rejected() -> None:
    with pytest.raises(ModelContractError, match="scalar Discrete"):
        ModelSpaceSignature.from_spaces(
            SimpleNamespace(shape=(923,), dtype=np.dtype("float32")),
            SimpleNamespace(n=5, start=0),
        )


def test_wrong_multidiscrete_shape_is_rejected() -> None:
    with pytest.raises(ModelContractError, match="action-space mismatch"):
        validate_model_contract(
            ModelSpaceSignature(
                observation_shape=(923,),
                observation_dtype="float32",
                action_nvec=(3, 2),
                action_start=(0, 0),
            ),
            metadata=ModelContractMetadata.current(),
        )


def test_metadata_less_checkpoint_is_rejected() -> None:
    with pytest.raises(ModelContractError, match="Metadata-less"):
        validate_model_contract(_current_spaces(), artifact_sha256="A" * 64)


def test_same_space_changed_action_semantics_is_rejected() -> None:
    payload = ModelContractMetadata.current().as_dict()
    payload["action_head_names"] = ["event", "steering"]
    with pytest.raises(ModelContractError, match="action_head_names"):
        validate_model_contract(_current_spaces(), metadata=payload)


def test_space_signature_reads_multidiscrete_without_gym_import() -> None:
    signature = ModelSpaceSignature.from_spaces(
        SimpleNamespace(shape=(923,), dtype=np.dtype("float32")),
        SimpleNamespace(nvec=np.asarray([3, 3]), start=np.asarray([0, 0])),
    )
    assert signature == _current_spaces()
    assert CURRENT_MODEL_CONTRACT.semantic_hash == MODEL_CONTRACT_HASH


def test_dtype_and_start_fail_closed() -> None:
    with pytest.raises(ModelContractError, match="observation-dtype mismatch"):
        validate_model_contract(
            ModelSpaceSignature((923,), "float64", (3, 3), (0, 0)),
            metadata=ModelContractMetadata.current(),
        )
    with pytest.raises(ModelContractError, match="start mismatch"):
        validate_model_contract(
            ModelSpaceSignature((923,), "float32", (3, 3), (0, 1)),
            metadata=ModelContractMetadata.current(),
        )
