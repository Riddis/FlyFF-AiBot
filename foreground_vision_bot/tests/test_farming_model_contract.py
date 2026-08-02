from __future__ import annotations

from types import SimpleNamespace

import farming.model_contract as model_contract_module
import numpy as np
import pytest
from farming.model_contract import (
    ACTIVE_METADATALESS_MODEL_CONTRACT_HASH,
    ACTIVE_METADATALESS_MODEL_SHA256,
    CURRENT_MODEL_CONTRACT,
    MODEL_CONTRACT_HASH,
    ModelContract,
    ModelContractError,
    ModelContractMetadata,
    ModelContractSource,
    ModelSpaceSignature,
    validate_model_contract,
)
from farming.observation import OBSERVATION_SCHEMA_HASH


def _current_spaces() -> ModelSpaceSignature:
    return ModelSpaceSignature(
        observation_shape=(482,),
        observation_dtype="float32",
        action_count=5,
        action_start=0,
    )


def test_new_model_metadata_binds_semantic_observation_and_action_contract() -> None:
    metadata = ModelContractMetadata.current()
    validation = validate_model_contract(_current_spaces(), metadata=metadata)

    assert metadata.contract_hash == MODEL_CONTRACT_HASH
    assert metadata.observation_schema_id == "native-unified-482-v3"
    assert metadata.action_names == (
        "RUN_FORWARD",
        "RUN_FORWARD_LEFT",
        "RUN_FORWARD_RIGHT",
        "CAST_EVA",
        "RUN_FORWARD_JUMP",
    )
    assert metadata.action_values == (0, 1, 2, 3, 4)
    assert validation.source is ModelContractSource.EMBEDDED_METADATA
    descriptor = CURRENT_MODEL_CONTRACT.descriptor()
    assert descriptor["action_space"] == {
        "kind": "Discrete",
        "count": 5,
        "start": 0,
        "coercion": "integral index only; booleans and foreign enums rejected",
    }
    assert descriptor["action_execution"] == {
        "persistent_movement_values": (0, 1, 2, 4),
        "cast_eva_value": 3,
        "run_forward_jump_value": 4,
        "movement_persists_until_replaced": True,
        "cast_eva_releases_movement": False,
        "jump_always_executes_forward_and_space": True,
        "jump_action_is_never_masked_or_degraded": True,
        "jump_flair_reward_has_cooldown": True,
    }


def test_old_metadata_less_four_action_model_is_rejected_after_jump_upgrade() -> None:
    with pytest.raises(ModelContractError, match="action-space mismatch"):
        validate_model_contract(
            ModelSpaceSignature(
                observation_shape=(482,),
                observation_dtype="float32",
                action_count=4,
                action_start=0,
            ),
            artifact_sha256=ACTIVE_METADATALESS_MODEL_SHA256,
        )

    with pytest.raises(ModelContractError, match="bound to contract"):
        validate_model_contract(
            _current_spaces(),
            artifact_sha256=ACTIVE_METADATALESS_MODEL_SHA256,
        )


def test_approved_artifact_hash_cannot_bypass_same_shape_semantic_drift() -> None:
    drifted = ModelContract(
        observation_schema_id=CURRENT_MODEL_CONTRACT.observation_schema_id,
        observation_schema_hash=CURRENT_MODEL_CONTRACT.observation_schema_hash,
        observation_size=CURRENT_MODEL_CONTRACT.observation_size,
        action_names=(
            "RUN_FORWARD",
            "RUN_FORWARD_RIGHT",
            "RUN_FORWARD_LEFT",
            "CAST_EVA",
            "RUN_FORWARD_JUMP",
        ),
        action_values=CURRENT_MODEL_CONTRACT.action_values,
    )

    with pytest.raises(ModelContractError, match="bound to contract"):
        validate_model_contract(
            _current_spaces(),
            artifact_sha256=ACTIVE_METADATALESS_MODEL_SHA256,
            expected=drifted,
        )


def test_model_space_mismatch_fails_before_semantic_or_hash_fallback() -> None:
    with pytest.raises(ModelContractError, match="observation-space mismatch"):
        validate_model_contract(
            ModelSpaceSignature(
                observation_shape=(481,),
                observation_dtype="float32",
                action_count=5,
                action_start=0,
            ),
            artifact_sha256=ACTIVE_METADATALESS_MODEL_SHA256,
        )
    with pytest.raises(ModelContractError, match="action-space mismatch"):
        validate_model_contract(
            ModelSpaceSignature(
                observation_shape=(482,),
                observation_dtype="float32",
                action_count=4,
                action_start=0,
            ),
            artifact_sha256=ACTIVE_METADATALESS_MODEL_SHA256,
        )


def test_same_shape_model_with_changed_action_semantics_is_rejected() -> None:
    payload = ModelContractMetadata.current().as_dict()
    payload["action_names"] = [
        "RUN_FORWARD",
        "RUN_FORWARD_RIGHT",
        "RUN_FORWARD_LEFT",
        "CAST_EVA",
        "RUN_FORWARD_JUMP",
    ]

    with pytest.raises(ModelContractError, match="action_names"):
        validate_model_contract(_current_spaces(), metadata=payload)


def test_same_shape_v2_map_model_is_rejected_after_map_risk_upgrade() -> None:
    payload = ModelContractMetadata.current().as_dict()
    payload["observation_schema_id"] = "native-unified-482-v2"
    payload["observation_schema_hash"] = (
        "48304E57C6A71ADFC3CE1B687B5849FFCA7CBF4B41B203346C96F467E7D79323"
    )

    with pytest.raises(ModelContractError, match="observation_schema"):
        validate_model_contract(_current_spaces(), metadata=payload)


def test_space_signature_reads_box_shape_and_discrete_count_without_gym_import() -> (
    None
):
    signature = ModelSpaceSignature.from_spaces(
        SimpleNamespace(shape=(482,), dtype=np.dtype("float32")),
        SimpleNamespace(n=5, start=0),
    )
    assert signature == _current_spaces()
    assert CURRENT_MODEL_CONTRACT.semantic_hash == MODEL_CONTRACT_HASH


def test_space_signature_rejects_coercible_or_missing_space_contract_fields() -> None:
    with pytest.raises(ModelContractError, match="observation shape dimension"):
        ModelSpaceSignature.from_spaces(
            SimpleNamespace(shape=(482.0,), dtype=np.dtype("float32")),
            SimpleNamespace(n=5, start=0),
        )
    with pytest.raises(ModelContractError, match="scalar discrete"):
        ModelSpaceSignature.from_spaces(
            SimpleNamespace(shape=(482,), dtype=np.dtype("float32")),
            SimpleNamespace(n=5.0, start=0),
        )
    with pytest.raises(ModelContractError, match="start index"):
        ModelSpaceSignature.from_spaces(
            SimpleNamespace(shape=(482,), dtype=np.dtype("float32")),
            SimpleNamespace(n=5),
        )
    with pytest.raises(ModelContractError, match="dtype"):
        ModelSpaceSignature.from_spaces(
            SimpleNamespace(shape=(482,)),
            SimpleNamespace(n=5, start=0),
        )


def test_model_space_dtype_and_discrete_start_fail_closed() -> None:
    with pytest.raises(ModelContractError, match="observation-dtype mismatch"):
        validate_model_contract(
            ModelSpaceSignature(
                observation_shape=(482,),
                observation_dtype="float64",
                action_count=5,
                action_start=0,
            ),
            artifact_sha256=ACTIVE_METADATALESS_MODEL_SHA256,
        )
    with pytest.raises(ModelContractError, match="start mismatch"):
        validate_model_contract(
            ModelSpaceSignature(
                observation_shape=(482,),
                observation_dtype="float32",
                action_count=5,
                action_start=1,
            ),
            artifact_sha256=ACTIVE_METADATALESS_MODEL_SHA256,
        )


def test_literal_contract_hashes_are_pinned_and_runtime_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert OBSERVATION_SCHEMA_HASH == (
        "0132FB13764E83FD3754AC895E872479537C45EDB2AAB83B73F87448397F69EC"
    )
    assert MODEL_CONTRACT_HASH == (
        "AB596A58173B724A57EA8FE5C71A66F6C102ACB5DD3E52C375F8FA80B2C89ABA"
    )
    assert ACTIVE_METADATALESS_MODEL_CONTRACT_HASH == (
        "03E1DA9C110611659DA10DF3CE27117C78E15F9E316ED080E4B75911768A8B18"
    )

    monkeypatch.setattr(
        model_contract_module,
        "observation_schema_hash",
        lambda: "F" * 64,
    )
    with pytest.raises(ModelContractError, match="semantic drift"):
        validate_model_contract(
            _current_spaces(),
            metadata=ModelContractMetadata.current(),
        )
