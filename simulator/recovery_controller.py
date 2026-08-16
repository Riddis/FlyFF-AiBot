"""Phase 1: deterministic bounded recovery controller.

A thin, stateful wrapper around the policy's own action selection -- it never
touches the neural network or training. It watches live-reproducible signals
(per-tick displacement, contacts, commanded event) accumulated over a
trailing window, and when they corroborate genuine no-progress (not just one
low-displacement tick, which an EVA cast legitimately produces), takes over
steering for a bounded escape sequence before handing control back.

This is deliberately NOT a fix for the underlying capability gap (the
steering branch still cannot see obstacles) -- it is a safety net that works
identically in the simulator and, once ported, live, without retraining or
touching the observation contract. simulator/local_clearance.py supplies the
one map query it needs, built from the same primitive that already populates
the observation's local_map block, not from the simulator-specific
movement_path_clear motion model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from farming.actions import FarmingEvent, SteeringAction
from .local_clearance import sample_heading_relative_clearance


class RecoveryState(str, Enum):
    NORMAL = "normal"
    RECOVERING = "recovering"
    GIVEN_UP = "given_up"


@dataclass
class RecoveryConfig:
    history_window: int = 20
    no_progress_displacement_threshold: float = 0.15
    no_progress_ticks_required: int = 15
    min_contacts_in_window: int = 3
    debounce_ticks: int = 15
    escape_turn_ticks: int = 8
    escape_progress_check_ticks: int = 6
    escape_progress_threshold_cells: float = 1.0
    max_escape_attempts: int = 4


@dataclass
class InterventionRecord:
    trigger_tick: int
    trigger_evidence: dict[str, Any]
    end_tick: int | None = None
    attempts: int = 0
    directions_tried: list[str] = field(default_factory=list)
    displacement_before: float = 0.0
    displacement_after: float | None = None
    outcome: str = "in_progress"  # "recovered" | "gave_up" | "in_progress"


class RecoveryController:
    """Call ``step`` once per environment tick, before stepping the
    environment, with the policy's intended action. Returns the action to
    actually issue -- either the policy's own action unchanged, or a bounded
    recovery override.
    """

    def __init__(self, config: RecoveryConfig | None = None) -> None:
        self.config = config or RecoveryConfig()
        self.state = RecoveryState.NORMAL
        self._displacement_history: deque[float] = deque(maxlen=self.config.history_window)
        self._contact_history: deque[int] = deque(maxlen=self.config.history_window)
        self._eva_history: deque[bool] = deque(maxlen=self.config.history_window)
        self._debounce_remaining = 0
        self._phase_tick = 0
        self._attempts = 0
        self._direction_order = ["left", "right"]
        self._current_direction: str | None = None
        self._pre_escape_reference_displacement = 0.0
        self._current_record: InterventionRecord | None = None
        self.interventions: list[InterventionRecord] = []

    def _detect_stagnation(self) -> dict[str, Any] | None:
        cfg = self.config
        if len(self._displacement_history) < cfg.history_window:
            return None
        no_progress_ticks = sum(
            1
            for displacement, was_eva in zip(self._displacement_history, self._eva_history)
            if displacement < cfg.no_progress_displacement_threshold and not was_eva
        )
        contacts_in_window = sum(self._contact_history)
        if no_progress_ticks < cfg.no_progress_ticks_required:
            return None
        if contacts_in_window < cfg.min_contacts_in_window:
            return None
        return {
            "no_progress_ticks": no_progress_ticks,
            "contacts_in_window": contacts_in_window,
            "window": cfg.history_window,
        }

    def step(
        self,
        *,
        tick: int,
        player_x: float,
        player_z: float,
        heading: float,
        displacement_this_tick: float,
        contact_this_tick: bool,
        map_model: Any,
        policy_steering: int,
        policy_event: int,
    ) -> tuple[int, int]:
        cfg = self.config
        was_eva = policy_event == int(FarmingEvent.CAST_EVA)
        self._displacement_history.append(displacement_this_tick)
        self._contact_history.append(1 if contact_this_tick else 0)
        self._eva_history.append(was_eva)
        if self._debounce_remaining > 0:
            self._debounce_remaining -= 1

        if self.state == RecoveryState.NORMAL:
            if self._debounce_remaining == 0:
                evidence = self._detect_stagnation()
                if evidence is not None:
                    self._enter_recovery(tick, evidence)
            if self.state == RecoveryState.NORMAL:
                return int(policy_steering), int(policy_event)

        if self.state == RecoveryState.RECOVERING:
            return self._run_recovery_tick(
                tick, player_x=player_x, player_z=player_z, heading=heading, map_model=map_model
            )

        # GIVEN_UP: stop intervening for the rest of the episode, report the
        # policy's own action so downstream metrics reflect its real choice.
        return int(policy_steering), int(policy_event)

    def _enter_recovery(self, tick: int, evidence: dict[str, Any]) -> None:
        self.state = RecoveryState.RECOVERING
        self._attempts = 0
        self._phase_tick = 0
        self._pre_escape_reference_displacement = sum(self._displacement_history)
        self._current_record = InterventionRecord(
            trigger_tick=tick, trigger_evidence=evidence, displacement_before=sum(self._displacement_history)
        )

    def _run_recovery_tick(
        self, tick: int, *, player_x: float, player_z: float, heading: float, map_model: Any
    ) -> tuple[int, int]:
        cfg = self.config
        record = self._current_record
        assert record is not None

        if self._current_direction is None:
            self._current_direction = self._choose_direction(player_x, player_z, heading, map_model)
            record.directions_tried.append(self._current_direction)
            record.attempts += 1
            self._phase_tick = 0
            self._pre_escape_reference_displacement = sum(self._displacement_history)

        self._phase_tick += 1
        steering = int(SteeringAction.LEFT if self._current_direction == "left" else SteeringAction.RIGHT)

        if self._phase_tick >= cfg.escape_turn_ticks + cfg.escape_progress_check_ticks:
            gained = sum(list(self._displacement_history)[-cfg.escape_progress_check_ticks :])
            if gained >= cfg.escape_progress_threshold_cells:
                self._exit_recovery(tick, outcome="recovered")
            else:
                self._attempts += 1
                self._current_direction = None
                if self._attempts >= cfg.max_escape_attempts:
                    self._give_up(tick)

        return steering, int(FarmingEvent.NONE)

    def _choose_direction(self, player_x: float, player_z: float, heading: float, map_model: Any) -> str:
        try:
            scores = sample_heading_relative_clearance(map_model, player_x, player_z, heading)
            if scores["left"] != scores["right"]:
                return "left" if scores["left"] > scores["right"] else "right"
        except Exception:
            pass
        return self._direction_order[self._attempts % len(self._direction_order)]

    def _exit_recovery(self, tick: int, *, outcome: str) -> None:
        record = self._current_record
        assert record is not None
        record.end_tick = tick
        record.outcome = outcome
        record.displacement_after = sum(self._displacement_history)
        self.interventions.append(record)
        self._current_record = None
        self._current_direction = None
        self.state = RecoveryState.NORMAL
        self._debounce_remaining = self.config.debounce_ticks
        # Clear no-progress history so the same stale evidence can't
        # immediately re-trigger before fresh ticks accumulate.
        self._displacement_history.clear()
        self._contact_history.clear()
        self._eva_history.clear()

    def _give_up(self, tick: int) -> None:
        record = self._current_record
        assert record is not None
        record.end_tick = tick
        record.outcome = "gave_up"
        record.displacement_after = sum(self._displacement_history)
        self.interventions.append(record)
        self._current_record = None
        self.state = RecoveryState.GIVEN_UP
