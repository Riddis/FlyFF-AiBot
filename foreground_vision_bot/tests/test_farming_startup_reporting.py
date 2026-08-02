from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from farming.model_contract import ModelContractError, ModelContractMetadata
from farming.reporting import (
    atomic_save_model,
    atomic_write_json,
    save_session_artifacts,
)
from farming.startup import load_and_validate_model


class FakeModel:
    def __init__(self) -> None:
        self.farming_contract_metadata: dict[str, object] | None = None

    def save(self, path: str) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("policy.txt", "validated")


def test_model_preflight_loads_without_live_env_and_validates_metadata(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "policy.zip"
    artifact.write_bytes(b"model")
    captured: list[str] = []

    def loader(path: str):
        captured.append(path)
        return SimpleNamespace(
            observation_space=gym.spaces.Box(
                -1.0,
                1.0,
                shape=(482,),
                dtype=np.float32,
            ),
            action_space=gym.spaces.Discrete(5, start=0),
            farming_contract_metadata=ModelContractMetadata.current().as_dict(),
        )

    validated = load_and_validate_model(artifact, loader)

    assert captured == [str(artifact)]
    assert validated.validation.contract_hash == (
        ModelContractMetadata.current().contract_hash
    )


def test_model_preflight_rejects_space_mismatch_before_caller_can_start_input(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "policy.zip"
    artifact.write_bytes(b"model")

    def loader(_path: str):
        return SimpleNamespace(
            observation_space=gym.spaces.Box(
                -1.0,
                1.0,
                shape=(482,),
                dtype=np.float32,
            ),
            action_space=gym.spaces.Discrete(4, start=0),
            farming_contract_metadata=ModelContractMetadata.current().as_dict(),
        )

    with pytest.raises(ModelContractError, match="Discrete"):
        load_and_validate_model(artifact, loader)


def test_atomic_json_replace_failure_preserves_previous_file_and_cleans_temp(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "report.json"
    destination.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(_source, _destination) -> None:
        raise PermissionError("locked")

    with pytest.raises(PermissionError, match="locked"):
        atomic_write_json(destination, {"new": True}, replace=fail_replace)

    assert json.loads(destination.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob(".*.tmp.json")) == []


def test_atomic_model_save_embeds_contract_and_validates_zip(tmp_path: Path) -> None:
    model = FakeModel()

    record = atomic_save_model(model, tmp_path / "policy")

    assert Path(record.path).is_file()
    assert len(record.sha256) == 64
    assert model.farming_contract_metadata == ModelContractMetadata.current().as_dict()


def test_session_manifest_points_to_validated_model_and_report(tmp_path: Path) -> None:
    artifacts = save_session_artifacts(
        FakeModel(),
        model_path=tmp_path / "policy.zip",
        report_path=tmp_path / "session.json",
        manifest_path=tmp_path / "latest.json",
        report={"status": "external_end", "reward": 0.0},
    )

    manifest = json.loads(Path(artifacts.manifest.path).read_text(encoding="utf-8"))
    assert manifest["model"]["sha256"] == artifacts.model.sha256
    assert manifest["report"]["sha256"] == artifacts.report.sha256
