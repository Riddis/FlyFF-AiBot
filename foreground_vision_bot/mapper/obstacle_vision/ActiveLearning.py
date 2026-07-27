from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .FloorAppearance import FloorPrediction


class SuggestedLabel(str, Enum):
    CLEAR = "clear"
    BLOCKED = "blocked"
    IGNORE = "ignore"


@dataclass(frozen=True)
class ActiveLearningDecision:
    prompt: bool
    suggested_label: SuggestedLabel
    reason: str


def decide_active_learning(
    *,
    motion_label: str,
    motion_confidence: float,
    camera_obscured: bool,
    teleport_likely: bool,
    floor_prediction: FloorPrediction,
    prompt_on_disagreement: bool = True,
) -> ActiveLearningDecision:
    label = str(motion_label).lower()
    confidence = float(motion_confidence)
    if camera_obscured or teleport_likely:
        return ActiveLearningDecision(
            True,
            SuggestedLabel.IGNORE,
            "The camera or scene is visually unreliable.",
        )
    if label not in {"clear", "blocked"} or confidence < 0.60:
        suggested = SuggestedLabel.IGNORE
        if floor_prediction.available and floor_prediction.obstacle_risk is not None:
            suggested = (
                SuggestedLabel.BLOCKED
                if floor_prediction.obstacle_risk >= 0.65
                else SuggestedLabel.CLEAR
            )
        return ActiveLearningDecision(
            True,
            suggested,
            "Motion validation could not confidently label the corridor.",
        )
    if (
        prompt_on_disagreement
        and floor_prediction.available
        and floor_prediction.obstacle_risk is not None
    ):
        risk = floor_prediction.obstacle_risk
        if label == "clear" and risk >= 0.78:
            return ActiveLearningDecision(
                True,
                SuggestedLabel.CLEAR,
                "Motion says clear, but the mature floor model sees a strong blocker-like mismatch.",
            )
        if label == "blocked" and risk <= 0.22:
            return ActiveLearningDecision(
                True,
                SuggestedLabel.BLOCKED,
                "Motion says blocked, but the mature floor model sees ordinary floor.",
            )
    return ActiveLearningDecision(
        False,
        SuggestedLabel.CLEAR if label == "clear" else SuggestedLabel.BLOCKED,
        "High-confidence automatic motion label.",
    )
