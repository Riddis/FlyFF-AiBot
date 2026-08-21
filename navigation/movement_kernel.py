"""2026-08-13: the ONE authoritative steering-tick kinematics kernel,
replacing the "turn fully, then translate fully" model that was
previously reimplemented independently in RecordedFarmingEnv._move_player,
simulator/steering_oracle.py (three sites), and
simulator/kinodynamic_route_planner.py's mean-transition math.

Built from the deployment-matched live calibration (see
run_logs/REPLACEMENT_MOVEMENT_MODEL_SPEC_2026-08-13.md and
run_logs/movement_calibration_local_frame_analysis_output.txt), which
measured -- with no tuning toward an expected winner, validated against
synthetic ground truth first -- that constant-curvature-arc kinematics
fits the real per-tick trajectory roughly 10x better (endpoint RMSE) and
7x better (path-shape RMSE) than turn-then-translate. At the legacy
~10deg/tick that approximation was mild; at the corrected ~45-50deg/tick
it is a large, separate error on top of the scalar corrections.

Two other findings drive this module's shape:
  1. Distance (path length) is constant regardless of steering direction
     or steering age -- confirmed across three independent measurement
     protocols. So there is exactly one distance constant, not one per
     action.
  2. Turn magnitude depends on whether the CURRENT steering direction
     matches the PREVIOUS tick's steering direction (a freshly-started
     LEFT/RIGHT turn is measurably weaker than a continuing one, reaching
     steady state by the second consecutive tick) -- so the transition
     is stateful: pose + previous_steering + current_steering -> next
     pose + next_previous_steering. This makes previous_steering part of
     the Markov state for both the environment and the kinodynamic
     planner, and -- since the live controller always knows what it
     itself commanded last tick, this is not privileged simulator
     information -- part of the PPO observation too.

Deliberately NOT injecting noise here (see the spec's "Noise" section):
the measured stds are confirmed upper bounds contaminated by a recorder-
clock aliasing artifact (near-perfect LOW/HIGH alternation, not IID
noise), not validated physical randomness. This kernel is intentionally
deterministic; any future stochastic term is a separate, explicit
decision layered on top, not inferred from that contaminated spread.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from . import movement_kinematics

# 2026-08-13: explicit physics-model version tag, so serialized world/config
# data and run provenance manifests can record which movement model a
# checkpoint/dataset/evaluation actually used, rather than requiring readers
# to infer it from a date or filename. RecordedFarmingEnv._move_player (and
# the planner/oracle successor generation it shares) unconditionally use this
# kernel now -- there is no runtime switch back to the legacy per-action
# Gaussian model -- so this constant records a FACT about the current
# codebase, not a live configuration option. Old serialized worlds/
# checkpoints predating this constant are treated as "legacy_recorded_iid"
# by ABSENCE of this tag (see RecordedWorldModel.movement_physics_model's
# default and run_provenance.build_run_manifest), never silently
# reinterpreted as calibrated-arc.
MOVEMENT_PHYSICS_MODEL_ID = "live_calibrated_arc"
LEGACY_MOVEMENT_PHYSICS_MODEL_ID = "legacy_recorded_iid"

# --- Calibrated constants -----------------------------------------------
# Derived directly from flyff_farming_recorder/calibration_tick_extraction_v2.csv
# (the validated, holdout-tested extraction -- see
# run_logs/movement_calibration_local_frame_analysis_output.txt and the
# tick-extraction output referenced in run_logs/OVERNIGHT_20260809_PIPELINE.md
# 2026-08-13/2026-08-13b entries). Recomputed 2026-08-13 via:
#   pooled mean distance_cells across STRAIGHT + LEFT + RIGHT, all ticks: 2.738491 (n=789)
#   pooled mean |heading_change_radians|, tick==1 (LEFT + mirrored RIGHT): 0.792978 (n=113)
#   pooled mean |heading_change_radians|, tick=='steady' (LEFT + mirrored RIGHT): 0.873649 (n=241)
PATH_LENGTH_CELLS_PER_TICK = 2.738491
ONSET_TURN_RADIANS = 0.792978
STEADY_TURN_RADIANS = 0.873649

DEFAULT_SUBSTEPS = 10  # FROZEN 2026-08-13 after scratchpad_substep_convergence_study.py:
# tested 10/20/40 substeps on 7 geometries (open LEFT/RIGHT, straight wall
# approach, corner pass, two deliberately-targeted near-tangent-wall
# placements along the arc's own path, narrow passage) -- contact/no-contact
# classification was identical across all substep counts in every case, and
# |position(10 substeps) - position(40 substeps)| stayed under 0.12 cells
# throughout (smaller than one map cell). No disagreement found despite
# specifically probing for it, not merely assumed from the ~50Hz recorder
# rate. A perfectly tangent adversarial geometry that WOULD flip
# classification cannot be ruled out (inherent to any discrete substep
# approximation) but wasn't found across this representative sweep.


class SteeringDirection(IntEnum):
    """NONE = no steering key held (STRAIGHT, or any non-turning action
    such as pure forward/EVA/jump). Deliberately just 3 states, not an
    unbounded steering-age counter -- the calibration showed turn
    magnitude is flat from the second consecutive steering tick onward,
    so only "is this a fresh turn or a continuing one" matters."""
    NONE = 0
    LEFT = 1
    RIGHT = 2


@dataclass(frozen=True)
class AdvanceResult:
    x: float
    z: float
    heading: float
    contact: bool
    next_previous_steering: SteeringDirection


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def resolve_signed_turn_radians(current: SteeringDirection, previous: SteeringDirection) -> float:
    """The stateful transition rule, per the calibration:
      current STRAIGHT (NONE)          -> 0.0
      current LEFT,  previous == LEFT  -> +STEADY_TURN_RADIANS
      current LEFT,  otherwise         -> +ONSET_TURN_RADIANS   (covers fresh start AND direct RIGHT->LEFT reversal)
      current RIGHT, previous == RIGHT -> -STEADY_TURN_RADIANS
      current RIGHT, otherwise         -> -ONSET_TURN_RADIANS
    Direct LEFT<->RIGHT reversal was never directly measured by the
    calibration (the protocol always returned to STRAIGHT between
    pulses); treating it as a fresh onset primitive is the documented,
    explicit default, not a measured fact -- see the spec's "What this
    data does NOT resolve" section."""
    if current == SteeringDirection.NONE:
        return 0.0
    magnitude = STEADY_TURN_RADIANS if current == previous else ONSET_TURN_RADIANS
    return magnitude if current == SteeringDirection.LEFT else -magnitude


def arc_endpoint_local(distance_cells: float, turn_radians: float) -> tuple[float, float]:
    """Closed-form single-shot constant-curvature-arc endpoint, in the
    LOCAL FRAME of the starting heading (forward = start-heading
    direction, lateral = +90deg-left of it). Degenerates to straight-
    line motion as turn_radians -> 0."""
    if abs(turn_radians) < 1.0e-9:
        return distance_cells, 0.0
    radius = distance_cells / turn_radians
    forward = radius * math.sin(turn_radians)
    lateral = radius * (1.0 - math.cos(turn_radians))
    return forward, lateral


def arc_endpoint_world(x: float, z: float, heading: float, distance_cells: float, turn_radians: float,
                        cell_size: float) -> tuple[float, float, float]:
    """Closed-form single-shot arc endpoint in WORLD/native coordinates --
    no collision checking. Useful for cheap heuristics (e.g. the
    kinodynamic planner's admissible-heuristic distance estimate); NOT
    used for the actual executed/collision-checked motion, which goes
    through advance_player_tick's substep integration instead."""
    forward, lateral = arc_endpoint_local(distance_cells, turn_radians)
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    dx_native = (forward * cos_h - lateral * sin_h) * cell_size
    dz_native = (forward * sin_h + lateral * cos_h) * cell_size
    new_heading = _normalize_angle(heading + turn_radians)
    return x + dx_native, z + dz_native, new_heading


def advance_player_tick(
    map_model: Any,
    x: float,
    z: float,
    heading: float,
    previous_steering: SteeringDirection,
    current_steering: SteeringDirection,
    *,
    distance_scale: float = 1.0,
    substeps: int = DEFAULT_SUBSTEPS,
) -> AdvanceResult:
    """THE authoritative steering-tick kinematics function. Pure/side-
    effect-free (takes previous_steering as an explicit input, returns
    next_previous_steering as an explicit output) so it is safe for the
    kinodynamic planner's search to call directly, not just the live
    environment -- callers own persisting the returned
    next_previous_steering for their own next call.

    Approximates the constant-curvature arc via `substeps` small
    turn-then-move increments, each passed through the EXISTING
    collision/slide primitive (movement_kinematics.advance_with_slide),
    so the collision-checked swept path follows the curve rather than a
    single long chord -- this is what turning ~50deg/tick (vs. the
    legacy ~10deg/tick this substep approach was never needed for)
    actually requires for a physically faithful collision sweep."""
    turn_total = resolve_signed_turn_radians(current_steering, previous_steering) * max(0.0, float(distance_scale))
    cell_size = map_model.native_units_per_cell
    distance_total_native = PATH_LENGTH_CELLS_PER_TICK * max(0.0, float(distance_scale)) * cell_size

    n = max(1, int(substeps))
    turn_per_substep = turn_total / n
    distance_per_substep = distance_total_native / n

    cur_x, cur_z, cur_heading = x, z, heading
    contact_any = False
    for _ in range(n):
        cur_heading = _normalize_angle(cur_heading + turn_per_substep)
        dx = math.cos(cur_heading) * distance_per_substep
        dz = math.sin(cur_heading) * distance_per_substep
        cur_x, cur_z, contact = movement_kinematics.advance_with_slide(map_model, cur_x, cur_z, dx, dz)
        contact_any = contact_any or contact

    return AdvanceResult(cur_x, cur_z, cur_heading, contact_any, current_steering)
