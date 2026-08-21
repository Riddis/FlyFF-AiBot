# Replacement movement-model spec (revised 2026-08-13c)

**Status: core kernel implemented and tested.** `simulator/movement_kernel.py`
(the one authoritative kinematics function described below) is built and
covered by `tests/test_movement_kernel.py` (23 tests) plus a dedicated
substep-convergence study; `RecordedFarmingEnv._move_player`,
`simulator/kinodynamic_route_planner.py`'s successor generation, and
`simulator/steering_oracle.py`'s three physics sites now all call through
it, eliminating the four independent reimplementations that previously
existed. **Not yet done:** the PPO observation/policy expansion (§
"Previous steering is Markov state" below), physics-version tagging on
serialized world/config data, curriculum regeneration, and PPO retraining
-- see `run_logs/OVERNIGHT_20260809_PIPELINE.md` for the up-to-date
sequencing status.

**2026-08-13b revision**: the original draft specified scalar distance/
turn constants but was silent on intra-tick GEOMETRY -- i.e. it implicitly
assumed the current simulator's turn-then-translate execution
(`RecordedFarmingEnv._move_player()`: apply the full turn to heading,
THEN move the full distance in a straight line along the new heading).
That approximation was mild at the legacy ~10deg/tick; at the newly-
measured ~50deg/tick it is a large error in its own right, independent of
the scalar corrections. Measured directly (see new section below):
constant-curvature-arc kinematics fits the real trajectory roughly 10x
better (endpoint RMSE) and 7x better (path-shape RMSE) than turn-then-
translate, with no tuning toward that outcome. This revision folds that
finding in and distinguishes three previously-conflated quantities: path
length/speed, heading delta, and the resulting displacement trajectory.

## Evidence this is based on

Three independent lines of measurement, each cross-checked against the
others and against synthetic ground truth:

1. Historical-recording reparse (`scratchpad_movement_provenance_*`) --
   showed the existing model's LEFT/RIGHT statistics are built almost
   entirely from brief taps (no run longer than 3 accepted pairs exists
   in either source recording), and that even the samples surviving the
   fitter's own turn-rate filter are dominated (56.5%/63.0%) by the
   exact-onset interval, which is measurably weaker-turning than later
   samples.
2. Stationary-start live calibration (`calibration_analysis.py`) --
   423 controlled trials, established the general shape (steady speed
   is duration-independent once moving; turn ramps with hold duration)
   before the deployment-matched protocol made it precise.
3. Deployment-matched live calibration (forward latched, steering
   pulsed on top), extracted at exact 0.20s control-tick resolution
   with timestamp-based windowing, validated against synthetic
   trajectories with known closed-form ground truth (both a tuning grid
   and a separate frozen-parameter holdout grid, including one smooth-
   ramp case), run through the actual production extraction path
   (`extract_real()`), not just a lower-level helper. This is the
   primary source for the constants below.

Two measurement artifacts were identified and are explicitly NOT carried
into this spec: (a) a sampling-clock aliasing pattern in the raw per-
window distance measurement (near-perfect LOW/HIGH alternation, not IID
noise -- confirmed via transition-matrix analysis: P(HIGH|prev=LOW)=
0.937, mean run-length=1.07) means the pooled MEAN is used, not a
bimodal sampler; (b) turn measurements show similar (messier) clustering
for the same reason. Both are recorder/timing artifacts, not measured
game physics, and are excluded from the noise model below.

## Proposed structure

Replace the current IID-per-action model:

```
action -> distance ~ Gaussian(mean[action], std[action])
       -> turn     ~ Gaussian(mean[action], std[action])
```

with a state-dependent model:

```
(current_steering_direction, previous_steering_direction) -> distance, turn
```

Concretely (**revised 2026-08-13c**: the original draft framed the state
as an unbounded `consecutive_steering_ticks` counter; as-shipped it is
just the PREVIOUS tick's steering direction, a 3-way value, not a
counter -- see below for why):

- **`steering_direction`** (current or previous): one of `NONE` (no
  steering key held -- STRAIGHT), `LEFT`, `RIGHT`.
- Since tick 2 through steady are statistically indistinguishable from
  each other (see the Turn table below), the only distinction that
  matters is "is this tick's steering a fresh start (onset) or a
  continuation of the immediately preceding tick's steering (steady)" --
  a 3-way `SteeringDirection` (`NONE`/`LEFT`/`RIGHT`) capturing just the
  PREVIOUS tick is sufficient; there is no information in a longer
  running count that an unbounded `consecutive_steering_ticks` would add.
  This is `movement_kernel.SteeringDirection` and
  `movement_kernel.resolve_signed_turn_radians(current, previous)` as
  shipped.
- **Reset rule**: switching to STRAIGHT resets `previous_steering` to
  `NONE` for the tick after. Switching LEFT<->RIGHT directly (no
  STRAIGHT tick in between) treats the new direction as fresh (onset) --
  this specific case was not directly measured (the calibration protocol
  always returned to STRAIGHT between pulses) and is called out below as
  the one remaining open question, not something this data confirms.

This directly matches what `RecordedFarmingEnv.step()` needs: the
environment already tracks the previous action each tick, so deriving
`previous_steering` is a small, local addition, not a new subsystem
(as shipped: `RecordedFarmingEnv.previous_steering`, updated from
`AdvanceResult.next_previous_steering` after every `_move_player` call).

## Previous steering is Markov state (2026-08-13c addition)

The corrected physics makes steering state part of the environment
state, not just an environment-internal implementation detail, because
turn magnitude now depends on `previous_steering` -- two otherwise-
identical `(x, z, heading)` states with different `previous_steering`
have genuinely different successors. This has two required consequences,
both load-bearing, not optional refinements:

1. **Kinodynamic planner**: `previous_steering` must be part of
   `KinoState` AND part of the closed-set/search state key -- as shipped,
   `kinodynamic_route_planner.KinoState.previous_steering` and
   `_state_key(x, z, heading_bin, previous_steering)`. Without this, two
   search nodes with identical position/heading but different steering
   history would incorrectly merge, silently propagating the wrong
   (onset vs. steady) turn magnitude into the search.
2. **PPO observation**: since the live controller always knows what
   steering it itself commanded on the previous tick, this is NOT
   privileged simulator information the student shouldn't see (unlike,
   say, the oracle's ground-truth map access) -- it is exactly the kind
   of information a real controller already has for free. The steering
   branch's sidecar features must therefore expand to include a 3-way
   one-hot `prev_straight` / `prev_left` / `prev_right`, alongside the
   existing 6 target-geometry + 3 physical-clearance +
   recent_progress + recent_contact features (11 -> 14 total steering-
   branch input; total policy observation grows from 925 to 928 --
   `NavigationHistoryWrapper`'s `SIDECAR_SIZE` from 2 to 5). **Not yet
   implemented** as of this revision -- see the pending-work list at the
   top of this document.

## Intra-tick geometry (new -- this is what changed)

The scalar figures below are three DIFFERENT quantities that the current
simulator conflates into one straight-line hop:

- **path length** (`distance`, cells) -- how far along the actual curved
  path the character travels during the tick. This is what the live
  calibration measures (dense 50Hz path integration), and it is NOT the
  same as straight-line endpoint displacement once the tick turns
  meaningfully.
- **heading delta** (`turn`, radians) -- how much the character's facing
  direction changes over the tick.
- **displacement trajectory** -- where the character's POSITION actually
  ends up (and passes through) during the tick, which depends on how
  path length and heading delta are combined moment-to-moment, not just
  their two totals.

Measured directly which of four candidate ways of combining them best
matches the real trajectory (local-frame extraction from the same
deployment-matched recording, no new recording, validated against
synthetic continuous ground truth first, no tuning toward an expected
winner):

| candidate | LEFT tick1 endpoint RMSE | LEFT steady endpoint RMSE | pooled steady path RMSE |
|---|---|---|---|
| turn-then-translate (current simulator) | 1.191 | 1.185 | 0.865 |
| translate-then-turn | 0.985 | 1.161 | 0.500 |
| straight chord at midpoint heading | 0.155 | 0.122 | 0.258 |
| **constant-curvature arc** | **0.128** | **0.085** | **0.115** |

Constant-curvature arc wins decisively in every pool (LEFT/RIGHT x
tick1/steady), by roughly 10x on endpoint error and 7x on path-shape
error versus the current simulator. Full results:
`run_logs/movement_calibration_local_frame_analysis_output.txt`.

**Proposed kinematics**: within a tick, the character moves along a
circular arc of length = `distance` (path length) and total turn angle =
`turn` (heading delta), i.e. treat `distance` and `turn` as defining a
constant-angular-rate arc for the duration of the tick, not "turn fully,
then move straight." In closed form (radius `r = distance/turn`, local
frame at tick-start heading):

```
forward(s) = r * sin(turn * s)
lateral(s) = r * (1 - cos(turn * s))       # s in [0,1], fraction of tick
```

(For `turn ~= 0`, i.e. STRAIGHT, this degenerates to plain straight-line
motion, `forward(s) = distance * s`, `lateral(s) = 0` -- no special case
needed beyond a small-angle guard against division by ~zero.)

**Collision execution**: rather than one long straight `advance_with_
slide()` call per tick (the current design), integrate the arc through
several small deterministic substeps, each a short straight segment
through the EXISTING `advance_with_slide()` collision code -- so the
collision-checked swept path follows the curve, not a chord. Proposed
starting point: 10 substeps per 0.20s tick (matching the calibration's
own ~50Hz resolution), but this needs a convergence check (compare
endpoint and contact outcomes at 10 vs 20 vs 40 substeps on representative
geometry) before freezing the count -- not assumed from the calibration
rate alone.

## Distance (forward displacement / path length)

**Single value, independent of steering direction or steering age**,
since the corrected pooled means agree to within ~0.6% across every
condition measured:

| condition | mean distance (cells/0.2s) | n |
|---|---|---|
| STRAIGHT | 2.750 | 158 |
| LEFT (all ticks pooled) | 2.734 | 339 |
| RIGHT (all ticks pooled) | 2.735 | 342 |

**Proposed constant: `distance ~= 2.74 cells` per control tick, applied
identically regardless of steering state.** (Using the LEFT/RIGHT pooled
figures rather than STRAIGHT's slightly-higher 2.750, since LEFT/RIGHT
together have almost 4x STRAIGHT's sample count here and the three
values already agree within measurement noise -- this is a minor
judgment call, not a strong claim that STRAIGHT is truly 0.6% faster.)

**As shipped**: `movement_kernel.PATH_LENGTH_CELLS_PER_TICK = 2.738491`
-- the full-precision pooled mean across STRAIGHT+LEFT+RIGHT, all ticks
(n=789), computed programmatically from
`calibration_tick_extraction_v2.csv` rather than retyped from this
table's rounded 2.74.

This is a ~2.7x increase from the current model's LEFT/RIGHT distance
(0.964/0.966) and a much smaller ~9% increase from STRAIGHT's (2.517).

## Turn (heading change)

| steering state | mean turn (rad/0.2s) | mean turn (deg) | n | std (rad) |
|---|---|---|---|---|
| STRAIGHT | ~0.0000 | ~0 | 158 | 0.005 (noise floor) |
| LEFT, tick 1 | +0.802 | +46.0 | 57 | 0.037 |
| LEFT, tick 2+ (steady) | +0.8728 | +50.0 | 337 (pooled ticks 2-5+steady) | 0.027 |
| RIGHT, tick 1 | -0.790 | -45.2 | 56 | 0.039 |
| RIGHT, tick 2+ (steady) | -0.8744 | -50.1 | 340 (pooled) | 0.028 |

Tick 2 through steady are statistically indistinguishable from each
other (differences of 0.0004-0.009 rad across ticks 2/3/4/5/steady,
well inside the measurement noise floor) -- **no separate tick3/tick4/
tick5 parameters are warranted**, confirming the user's expectation.
LEFT/RIGHT magnitude asymmetry is small at every tick (0.07-0.19% for
ticks 2+; 1.55% at tick 1, which also has the highest std of any tick,
suggesting this may be genuine residual onset variability rather than a
measurement artifact, though this is not fully resolved).

**Proposed constants** (averaging LEFT/RIGHT magnitude at each age,
since the asymmetry is small and not clearly distinguishable from
residual onset noise -- do NOT preserve the historical model's much
larger ~14-24% asymmetry, which the live data does not support):

```
STRAIGHT:            turn = 0.0
LEFT/RIGHT, tick 1:   |turn| = 0.796 rad (45.6deg), signed by direction
LEFT/RIGHT, tick 2+:  |turn| = 0.8736 rad (50.05deg), signed by direction
```

This is roughly a 4.3x increase (tick 1) to 4.9x increase (steady) over
the current model's LEFT (0.184) and a 5.4x to 5.9x increase over RIGHT
(0.148) -- confirming this is a materially different control regime, not
a calibration tweak.

**As shipped**: `movement_kernel.ONSET_TURN_RADIANS = 0.792978` and
`STEADY_TURN_RADIANS = 0.873649` -- full-precision pooled LEFT+mirrored-
RIGHT magnitudes (onset n=113, steady n=241, the latter corrected after
a duplicate-window bug fix moved the steady pool from 140/151 to
115/126 per side), computed programmatically rather than retyped from
this table's rounded 0.796/0.8736.

## Noise / stochasticity

**Recommendation: start near-deterministic (a small residual std, not
the current model's std-approximately-equal-to-mean spread), because
the data does not currently support a larger figure.** Specifically:

- The raw per-window measurement stds shown above (0.005-0.039 rad,
  ~0.11 cells for distance before pooling-away the aliasing) are UPPER
  BOUNDS on true physical randomness, not confirmed measurements of it
  -- both distance and turn measurements are shown to carry a real
  recorder/sampling-clock artifact component, and no method in this
  investigation separates that from genuine game-physics noise.
- Rather than guess a discount factor, the recommendation is to ship
  the replacement model with **zero or minimal injected noise** (e.g.
  a small fixed epsilon well under the measured stds, only if the
  training code requires a nonzero std to avoid degenerate sampling)
  and treat any RL exploration noise as an explicit, separately-tuned
  training-time concern, not something inferred from this measurement.
- If training later reveals a genuine need for stochasticity in the
  physics itself (as opposed to policy-level exploration noise), that
  should be a deliberate, separate decision with its own justification
  -- not a default carried over from measurement artifacts.

## What this data does NOT resolve (flagged, not guessed)

- **Direct LEFT<->RIGHT reversal** (no intervening STRAIGHT tick): never
  directly measured. The reset-to-tick-1 rule above is a reasonable
  default given release-to-straight showed no meaningful angular carry-
  over, but is unvalidated for the reversal case specifically. Per the
  user's earlier note, this can be calibrated later only if it turns
  out to matter.
- **Tick-1's small residual asymmetry** (1.55% vs <0.2% for later
  ticks): not confidently attributed to real physics vs. onset
  measurement noise.
- **True physical noise floor**: not isolated from measurement/
  recorder-clock artifacts by this investigation; treated as unknown
  rather than assumed small or large.

## One authoritative kinematics function

Several places in this codebase used to reproduce the legacy turn-then-
translate formula independently (`RecordedFarmingEnv._move_player`, the
kinodynamic planner's `_apply_primitive`/`_apply_outcome`, and three
sites inside the oracle: `_mean_turn_and_distance`, `_one_real_tick`,
`_robust_envelope_safe`). That's exactly how a router/environment/oracle
mismatch could happen again -- e.g. the kinodynamic planner reasoning
about one arc shape while the environment executes a different one.

**As shipped** (`simulator/movement_kernel.py`):

```python
def resolve_signed_turn_radians(current: SteeringDirection, previous: SteeringDirection) -> float: ...
def arc_endpoint_local(distance_cells: float, turn_radians: float) -> tuple[float, float]: ...
def arc_endpoint_world(x, z, heading, distance_cells, turn_radians, cell_size) -> tuple[float, float, float]: ...

def advance_player_tick(
    map_model, x, z, heading, previous_steering: SteeringDirection, current_steering: SteeringDirection, *,
    distance_scale: float = 1.0, substeps: int = DEFAULT_SUBSTEPS,
) -> AdvanceResult:  # (x, z, heading, contact, next_previous_steering)
    ...
```

implementing the arc-plus-substep kinematics above (deterministic core --
no noise is injected, per the Noise section above; any future noise term
is a separate, explicit concern layered on top, not baked into this
function). Pure/side-effect-free (`previous_steering` in, `next_
previous_steering` out) so it is safe for the planner's search to call
directly. **Every physics consumer now uses this one function, not its
own formula**: `RecordedFarmingEnv._move_player`, `kinodynamic_route_
planner._successor_state`/`_arc_edge_check` (successor generation and
curved-sweep collision checking both), and `steering_oracle._one_real_
tick` (itself now the single physics entry point the oracle's escape
search, beam search, and terminal-viability probe all funnel through,
replacing the three original reimplementation sites). `arc_endpoint_
world` is also used directly for the planner's cheap A* heuristic and
successor sketch, where the full substep-integrated collision check
would be unnecessarily expensive for a coarse route search.

## Before touching the simulator (per the user's explicit sequencing)

1. **DONE.** Mechanical replay tests: `tests/test_movement_kernel.py`
   (23 tests) covers `resolve_signed_turn_radians`'s full onset/steady/
   reversal transition table, `arc_endpoint_local`'s closed-form
   correctness (including the exact worked example distance=2.74/
   turn=0.874 -> (2.404, 1.123) used during design review), and
   `advance_player_tick`'s open-map behavior converging to the closed-
   form arc as substeps increase.
2. **DONE.** Convergence/invariance check:
   `scratchpad_substep_convergence_study.py` compared 10/20/40 substeps
   on 7 geometries (open LEFT/RIGHT, wall approach, corner pass, two
   near-tangent-wall placements, narrow passage) -- contact/no-contact
   classification was identical across all substep counts in every case,
   and `|pos(10) - pos(40)|` stayed under 0.12 cells throughout.
   `DEFAULT_SUBSTEPS = 10` frozen in `movement_kernel.py` with this
   measurement documented in a comment (not assumed from the
   calibration's ~50Hz rate alone). A perfectly tangent adversarial
   geometry that would flip classification cannot be ruled out in
   principle but was not found despite deliberately probing for it.
3. **NOT YET DONE.** Regenerate synthetic curriculum physics from the
   new model, and refit/replace `models/recorded_world.json.gz`'s
   movement fields (or its successor representation, since the current
   `MovementModel` dataclass -- one Gaussian per discrete action, single
   straight-line hop -- cannot represent steering-state or arc
   kinematics and needs a structural change, not just new constants).
   This must also introduce an explicit physics/model-version tag (e.g.
   `legacy_recorded_iid` vs `live_calibrated_arc`) so old serialized
   worlds/checkpoints are never silently reinterpreted under the new
   semantics.
4. **NOT YET DONE.** Expand the PPO observation/policy for the
   previous-steering Markov state (see "Previous steering is Markov
   state" above), then retrain the generalized-waypoint PPO navigator
   from scratch under the corrected physics, multiple seeds, before
   returning to the single-obstacle/router experiments.
5. Preserve the existing router implementation, its regression tests,
   and all prior obstacle-evaluation results -- relabel them explicitly
   as legacy-movement-model results, not evidence about how a
   navigator trained under the corrected physics will behave. The
   router's own bugs and fixes (envelope machinery, waypoint
   compression, etc.) remain valid engineering knowledge independent
   of which physics model it routes for. (The envelope machinery itself
   -- both the planner's `PRIMITIVE_ENVELOPES` and the oracle's sigma-
   probed robust-safety tier -- was deleted outright, not kept dormant,
   once the calibrated model turned out to be deterministic: there is no
   variance left for it to guard against. Its *lessons* -- e.g. the
   terminal continuation-viability gate, the provenance-tracked escape
   search -- remain in the deterministic replacements.)
