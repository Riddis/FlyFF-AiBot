from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .ActionMask import valid_action_names
from .PolicyTypes import MapperAction


@dataclass(frozen=True)
class PolicyRecommendation:
    action: MapperAction
    model_path: str
    valid_actions: tuple[str, ...] = ()


class MapperRLPolicy:
    """Optional policy loader supporting v1.8 MaskablePPO and old PPO models."""

    def __init__(
        self,
        model: Any,
        model_path: Path,
        *,
        supports_action_masks: bool,
    ) -> None:
        self.model = model
        self.model_path = model_path
        self.supports_action_masks = supports_action_masks

    @classmethod
    def load(cls, model_path: Path) -> "MapperRLPolicy":
        if not model_path.is_file() and not model_path.with_suffix(".zip").is_file():
            raise FileNotFoundError(f"Mapper RL policy is missing: {model_path}")

        algorithm = _metadata_algorithm(model_path)
        if algorithm == "MaskablePPO":
            try:
                from sb3_contrib import MaskablePPO
            except ImportError as error:
                raise RuntimeError(
                    "Mapper RL v1.8 requires sb3-contrib. Install with: "
                    "pip install -r requirements_mapper_rl.txt"
                ) from error
            return cls(
                MaskablePPO.load(str(model_path)),
                model_path,
                supports_action_masks=True,
            )

        try:
            from stable_baselines3 import PPO
        except ImportError as error:
            raise RuntimeError(
                "Mapper RL shadow mode requires Stable-Baselines3. Install with: "
                "pip install -r requirements_mapper_rl.txt"
            ) from error
        return cls(
            PPO.load(str(model_path)),
            model_path,
            supports_action_masks=False,
        )

    def recommend(
        self,
        observation: dict[str, object],
        *,
        action_masks: NDArray[np.bool_] | None = None,
    ) -> PolicyRecommendation:
        if self.supports_action_masks:
            if action_masks is None:
                raise ValueError("MaskablePPO recommendation requires an action mask")
            action, _state = self.model.predict(
                observation,
                deterministic=True,
                action_masks=np.asarray(action_masks, dtype=np.bool_),
            )
        else:
            action, _state = self.model.predict(observation, deterministic=True)
        value = int(action.item() if hasattr(action, "item") else action)
        return PolicyRecommendation(
            action=MapperAction(value),
            model_path=str(self.model_path),
            valid_actions=(
                valid_action_names(action_masks)
                if action_masks is not None
                else tuple(action.name for action in MapperAction)
            ),
        )


def write_policy_metadata(path: Path, payload: dict[str, object]) -> None:
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metadata_algorithm(model_path: Path) -> str:
    candidates = (
        model_path.with_suffix(".metadata.json"),
        model_path.with_suffix("").with_suffix(".metadata.json"),
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        algorithm = str(payload.get("algorithm", "")).strip()
        if algorithm:
            return algorithm
    return "PPO"
