from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mapper.OccupancyGrid import OccupancyGrid

from .LiveObservation import LivePolicyMemory, build_live_policy_input
from .Policy import MapperRLPolicy


@dataclass(frozen=True)
class ShadowDecision:
    action: str
    enabled: bool
    status: str
    valid_actions: tuple[str, ...] = ()


class MapperShadowPlanner:
    """Ask the RL policy for advice while the deterministic planner stays in control."""

    def __init__(
        self,
        *,
        enabled: bool,
        model_path: Path,
        output_path: Path,
    ) -> None:
        self.enabled = bool(enabled)
        self.model_path = model_path
        self.output_path = output_path
        self.memory = LivePolicyMemory()
        self.policy: MapperRLPolicy | None = None
        self.warning: str | None = None
        self._handle = None
        if self.enabled:
            try:
                self.policy = MapperRLPolicy.load(model_path)
            except Exception as error:  # noqa: BLE001 - shadow mode must not stop mapping.
                self.warning = str(error)
                self.enabled = False

    def recommend(self, grid: OccupancyGrid) -> ShadowDecision:
        if not self.enabled or self.policy is None:
            return ShadowDecision("", False, self.warning or "disabled")
        try:
            policy_input = build_live_policy_input(grid, self.memory)
            recommendation = self.policy.recommend(
                policy_input.observation,
                action_masks=policy_input.action_mask,
            )
            return ShadowDecision(
                recommendation.action.name,
                True,
                "ok",
                recommendation.valid_actions,
            )
        except Exception as error:  # noqa: BLE001 - fail open to deterministic planner.
            self.warning = f"Mapper RL shadow recommendation failed: {error}"
            self.enabled = False
            return ShadowDecision("", False, self.warning)

    def record(
        self,
        *,
        step: int,
        actual_action: str,
        actual_reason: str,
        recommendation: ShadowDecision,
        outcome: str,
        pose_known: bool,
    ) -> None:
        if not recommendation.enabled:
            return
        if self._handle is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.output_path.open("a", encoding="utf-8")
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "step": int(step),
            "actual_action": str(actual_action),
            "actual_reason": str(actual_reason),
            "shadow_action": recommendation.action,
            "valid_actions": list(recommendation.valid_actions),
            "agrees": recommendation.action == actual_action,
            "outcome": str(outcome),
            "pose_known": bool(pose_known),
        }
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
