# Navigation & Movement

**Confidence: VERIFIED_CONTRACT** for package ownership and the module
docstrings cited directly; **HISTORICAL_EVIDENCE** for the calibration
measurement claims (ported from the module's own docstring, which
itself cites `run_logs/` evidence not independently re-run this phase).

## 1. Canonical navigation package

`navigation/` (Phase-9 canonical, `SHARED_RUNTIME_CORE`):

| Module | Owns |
|---|---|
| `navigation/kinodynamic_route_planner.py` | The heading-aware kinodynamic route planner: `plan_route`, `select_persistent_waypoint`, `TargetPersistenceController` |
| `navigation/movement_kernel.py` | The one authoritative steering-tick kinematics kernel: `SteeringDirection`, `AdvanceResult`, `advance_player_tick`, `resolve_signed_turn_radians` |
| `navigation/movement_kinematics.py` | Supporting kinematics helpers |
| `navigation/navigation_evidence.py` | `NavigationStepEvidence`, `sidecar_values_from_history`, `previous_steering_one_hot` — the sidecar-feature construction the checkpoint ABI's 5-element sidecar (see `DATA_AND_MODEL_CONTRACTS.md`) is built from |
| `navigation/map_protocol.py` | Shared map protocol types |

Confirmed clean of `gymnasium`/`stable_baselines3`/`torch`/`recorder`/
training-only `simulator.*` (`tests/test_navigation_dependency_
boundary.py`, still passing).

## 2. Why the kinodynamic route planner replaced the old 2D path design

`navigation/kinodynamic_route_planner.py`'s own docstring records the
design history: it replaced `simulator/route_waypoint_generator.py`'s
"2D path → filter candidate waypoint → hope PPO can execute it" design
after diagnostics showed geometric reachability alone is insufficient.
FlyFF movement is **not holonomic** — forward is latched, there is no
stop/reverse, and LEFT/RIGHT change both position *and* heading
simultaneously — so a cell can be geometrically reachable while being a
poor immediate target from the current heading. Two concrete failure
modes motivated the replacement: (1) insufficient approach clearance
causing collision before reaching a waypoint, and (2) a waypoint reached
safely but the *next* target requiring a bearing change large enough to
push the frozen PPO into a constant-turn/circling attractor. A
53°/5-cell "geometrically valid" waypoint that was not kinodynamically
executable was the same failure in miniature — tightening a single
`MAX_HEADING_CHANGE` threshold on the old design just traded collisions
for "no valid waypoint found" (the old design's four independent hard
AND-gates — clearance, corridor, heading, progress — proved too brittle
a formulation).

**`simulator/route_waypoint_generator.py` status:** superseded,
simulator-specific historical predecessor. Not deleted (no `removal_
gate` registered against it, not audited as a Phase-12 deletion
candidate), but not the canonical route-planning implementation. New
work should use `navigation/kinodynamic_route_planner.py`.

**`simulator/router_waypoint_env.py` status:** `SIMULATOR_ONLY` /
`TRAINING_ONLY` — a simulator-specific Gym environment adapter around
the canonical planner, explicitly listed in `forbidden_first_party_
prefixes` for the future runtime candidate (it is training/environment
implementation, not shared runtime).

## 3. The movement kernel — one authoritative kinematics model

`navigation/movement_kernel.py`'s docstring records: this module
replaces a "turn fully, then translate fully" model that was previously
reimplemented independently in three places
(`RecordedFarmingEnv._move_player`, `simulator/steering_oracle.py` in
three sites, and `simulator/kinodynamic_route_planner.py`'s own mean-
transition math) — a duplication-of-truth problem this module
eliminates.

**Physics model ID:** `MOVEMENT_PHYSICS_MODEL_ID = "live_calibrated_
arc"` — a provenance tag recording which kinematics model produced a
given trajectory/dataset, checked by `tests/test_physics_version_tag_
provenance_only.py` and `tests/test_run_provenance.py`.

**Calibration source (HISTORICAL_EVIDENCE, per the module's own
docstring citations):** built from deployment-matched live calibration
(`simulator/run_logs/REPLACEMENT_MOVEMENT_MODEL_SPEC_2026-08-13.md`; the module's
own docstring additionally cites a
`movement_calibration_local_frame_analysis_output.txt` under `run_logs/`
that is not currently present in the tracked tree — a pre-existing gap
in the source citation itself, not introduced by this documentation),
which
measured — validated against synthetic ground truth first, with no
tuning toward an expected winner — that constant-curvature-arc
kinematics fits real per-tick trajectories roughly 10x better (endpoint
RMSE) and 7x better (path-shape RMSE) than turn-then-translate. Two
further findings shape the kernel: (1) per-tick distance is constant
regardless of steering direction or steering age (confirmed across
three independent measurement protocols — exactly one distance
constant, not one per action); (2) turn magnitude depends on whether the
current steering direction matches the *previous* tick's steering
direction — a freshly-started turn is measurably weaker than a
continuing one, reaching steady state by the second consecutive tick.
This is why the model is **stateful with respect to previous
steering** — see section 4.

## 4. Stateful previous steering

The movement kernel and the checkpoint ABI's sidecar features (`prev_
straight`/`prev_left`/`prev_right`, `navigation/navigation_evidence.py`)
both depend on the *previous tick's* steering direction, not just the
current action. Any evaluation harness that re-augments an observation
without passing the actual previous steering value will silently
default to `NONE` — this was a real, discovered bug (see
`MISTAKES.md` for the corrected-observation
router evaluation incident) that corrupted every routing evaluation
result produced before the fix. When writing or reviewing any evaluation
harness around router/navigation behavior, always verify `previous_
steering` is threaded through correctly — this is exactly the kind of
mistake `MISTAKES.md`'s "observation/reward/action wiring" category
exists to prevent repeating.

## 5. History/evidence split

`navigation/navigation_evidence.py`'s `NavigationStepEvidence` and
`sidecar_values_from_history` construct the 5-element sidecar
(`recent_progress`, `recent_contact`, and the 3 previous-steering
one-hot values) from a rolling tick history, separate from the 923-
element raw observation. This split exists so history-derived features
(needing multiple ticks of context) are computed once, centrally, rather
than each environment/evaluation harness re-deriving them independently
— the exact class of bug section 4 describes.

## 6. Simulator adapters vs. the shared implementation

`simulator/` contains environment/training adapters
(`environment.py`, `router_waypoint_env.py`, `static_waypoint_env.py`,
`single_obstacle_env.py`, `synthetic.py`, `steering_oracle.py`,
`navigation_history.py`) that **consume** the canonical
`navigation/*` implementation via re-exports (e.g.
`SteeringDirection`/`advance_player_tick` re-exported from
`navigation.movement_kernel`) — they are training/environment glue, not
a second implementation of the kinematics or route-planning algorithms.
`simulator/kinodynamic_route_planner.py` and `simulator/movement_
kernel.py` are the two Phase-9 **runtime ABI compatibility shims**
(pickle module-identity only, zero behavior) — see
`DATA_AND_MODEL_CONTRACTS.md` section 1c. Do not confuse these
behavior-free shims with the simulator's own real adapter code in the
other files listed above.

## 7. What the generalized-waypoint checkpoint is (and is not)

The current frozen checkpoint (0051200, `DATA_AND_MODEL_CONTRACTS.md`)
is a **generic navigation/waypoint-following policy** — trained to
navigate toward a target waypoint while avoiding obstacles, using the
split-branch steering/event-value architecture. Its event branch is
effectively `EVENT_NONE` in this lineage: **it is not an attack/event
controller.** Do not interpret its event-branch output as combat intent,
and do not assume it encodes any monster-approach or engagement
behavior beyond pure waypoint navigation, without separate evidence.

## Evidence / Sources

- `navigation/kinodynamic_route_planner.py`, `navigation/
  movement_kernel.py`, `navigation/navigation_evidence.py` (direct
  source reads, module docstrings)
- `docs/migration/PHASE9_NAVIGATION_OWNER_ANALYSIS.md`,
  `docs/migration/codex_handoff/PHASE9_REPORT.md`
- `MISTAKES.md` (previous-steering observation bug)
- `CANONICAL_OWNERS.toml` (`qualified_persistent_waypoint_selector`,
  `shared_movement_kernel`, `shared_navigation_evidence` concepts)
- `tests/test_navigation_dependency_boundary.py`,
  `tests/test_physics_version_tag_provenance_only.py`
