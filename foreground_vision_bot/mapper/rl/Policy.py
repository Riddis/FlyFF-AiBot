from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .PolicyTypes import MapperAction


@dataclass(frozen=True)
class PolicyRecommendation:
    action: MapperAction
    model_path: str


class MapperRLPolicy:
    """Thin optional Stable-Baselines3 policy loader used by shadow mode."""

    def __init__(self, model: Any, model_path: Path) -> None:
        self.model = model
        self.model_path = model_path

    @classmethod
    def load(cls, model_path: Path) -> "MapperRLPolicy":
        try:
            from stable_baselines3 import PPO
        except ImportError as error:
            raise RuntimeError(
                "Mapper RL shadow mode requires stable-baselines3. Install with: "
                "pip install stable-baselines3 gymnasium tensorboard"
            ) from error
        if not model_path.is_file() and not model_path.with_suffix(".zip").is_file():
            raise FileNotFoundError(f"Mapper RL policy is missing: {model_path}")
        model = PPO.load(str(model_path))
        return cls(model, model_path)

    def recommend(self, observation: dict[str, object]) -> PolicyRecommendation:
        action, _state = self.model.predict(observation, deterministic=True)
        value = int(action.item() if hasattr(action, "item") else action)
        return PolicyRecommendation(
            action=MapperAction(value),
            model_path=str(self.model_path),
        )


def write_policy_metadata(path: Path, payload: dict[str, object]) -> None:
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
