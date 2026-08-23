# Curriculum Training Pipeline (canonical Basic -> Advanced)

**Confidence: VERIFIED_CONTRACT** for module roles, call paths, and the
current frozen-navigation/router-integration state (direct source reads,
cited below); **HISTORICAL_EVIDENCE** for the graduation-history claims
ported from docstrings/run logs. **Current truth (read this before section
2's historical narrative): the production router IS exercised by every
stage of this curriculum** — the frozen 0051200 checkpoint, driven by
`navigation.kinodynamic_route_planner`, owns steering end to end via
`simulator.navigation_subpolicy.FrozenNavigationSteering`; the curriculum's
own trainable policy owns only learned target selection + the event/EVA
action. See section 4/5 for the full current architecture and section 2 for
why an earlier version of this same document said the opposite.

## 1. What this is

The canonical **Basic -> Beginner -> Intermediate -> Advanced** curriculum
(`simulator/curriculum_stages.py`'s vocabulary) is the **training
pipeline for the future generic full-farming baseline** — see
[`PROJECT_GOALS.md`](../PROJECT_GOALS.md) and
[`docs/agent/PROJECT_RULES.md`](../agent/PROJECT_RULES.md) section 1. It
is not an evaluation-only harness for an externally-supplied checkpoint.
Each stage **trains and grows one continuous policy lineage**, saved as
`models/canonical_<stage>_graduated.zip`, each stage's training starting
from the previous stage's graduated checkpoint:

| Stage | Entrypoint | Method | Internal stage id |
|---|---|---|---|
| Basic | `simulator/tools/RUN_CANONICAL_BASIC.py` | Fresh init -> human-recording BC bootstrap -> recovery-assisted DAgger rounds (supervised only, **no PPO** — see `simulator/basic_environment.py`'s module docstring for why) | `early` |
| Beginner | `simulator/tools/RUN_CANONICAL_BEGINNER.py` | Recovery-off PPO continuation from `canonical_basic_graduated.zip` | `early` |
| Intermediate | `simulator/tools/RUN_CANONICAL_INTERMEDIATE.py` | Recovery-off PPO continuation from `canonical_beginner_graduated.zip` | `intermediate` |
| Advanced | `simulator/tools/RUN_CANONICAL_ADVANCED.py` | Recovery-off PPO continuation from `canonical_intermediate_graduated.zip` | `advanced` |

Graduation at each stage is an absolute bar (stable competence, no
collapse, healthy coverage/kill-rate floors — see `curriculum_stages.py`'s
module docstring), not "better than this stage's own zero-shot baseline."

Supporting modules:

- `simulator/curriculum_stages.py` — translates the canonical stage names
  above onto the internal `early`/`intermediate`/`advanced` identifiers
  used throughout `synthetic.py`, curriculum manifests, and generated
  map/world assets. **The one place this translation happens** — new
  code should look stages up through `CANONICAL_STAGES` here rather than
  inventing another spelling.
- `simulator/curriculum_manifests.py` — immutable per-stage evaluation
  manifests: `HeldoutManifest` (representative unseen same-stage
  layouts), `ChallengeManifest` (fixed regression scenarios + fresh
  challenge-family siblings), `GeneratorValidationManifest` (suspected
  generator-constraint violations, excluded from scoring until a human
  confirms).
- `simulator/milestone_evaluator.py` — the **one evaluator both training
  and grading use**: `run_episode` rolls a loaded policy through one
  episode and reports kills/hour, contact/collision statistics, steering/
  event-vs-teacher agreement, and movement classification;
  `evaluate_heldout`/`evaluate_challenge` (plus `_parallel` variants)
  score a checkpoint across a manifest's layouts. Read-only with respect
  to training (never calls `policy.learn()`).
- `simulator/basic_environment.py`, `simulator/beginner_transition.py` —
  per-stage round orchestration (DAgger mining / PPO chunk + rehearsal +
  graduation check).

## 2. Historical note: this curriculum originally did not exercise the production router (superseded — see section 4/5 for current state)

**CURRENT STATE FIRST, since this section's own heading described the
opposite of today's architecture: every stage now exercises the production
router via the frozen 0051200 checkpoint (`FrozenNavigationSteering`) —
see section 4/5. What follows is the original 2026-08-22 investigation that
established the (now-corrected) gap below; kept because the reasoning and
evidence trail remain useful context for why the recovery happened, not
because the conclusion still holds.**

`milestone_evaluator.py::run_episode` (the evaluator every
`RUN_CANONICAL_*.py` stage and every `evaluate_heldout`/
`evaluate_challenge` call uses) selects its steering target every tick
from the environment's own native target-approach geometry:

```python
angle = env.best_group_relative_angle()
...
candidates = env._visible_candidates()
if candidates:
    ...
    _potential, best_actor_id = env._group_approach_potential(candidates, geodesic_field)
```

(`simulator/milestone_evaluator.py`, `run_episode`, direct-bearing block).
It never imports or calls `navigation.kinodynamic_route_planner.
plan_route`, `select_persistent_waypoint`, or
`TargetPersistenceController` — confirmed by a direct read of the whole
function body, not an import-grep (the project's own precedent for why
that distinction matters: `MISTAKES.md`'s "`run_episode_general_router`
silently fed `previous_steering=NONE`" entry). The curriculum's own
policies are trained end-to-end against this same direct-bearing
representation (`simulator/basic_environment.py`'s `scripted_command
("obstacle_aware", env)` teacher), not against router-selected
waypoints.

This is a deliberate, documented separation, not an oversight:
`navigation/kinodynamic_route_planner.py`'s own docstring (see
[`NAVIGATION_AND_MOVEMENT.md`](NAVIGATION_AND_MOVEMENT.md) section 2)
records that the router replaced a "2D path -> filter candidate waypoint
-> hope the policy can execute it" design precisely because FlyFF
movement is non-holonomic — the frozen navigation checkpoint
`models/generalized_waypoint_both_seed2_0051200.zip`
(`DATA_AND_MODEL_CONTRACTS.md` section 1) was trained specifically to
navigate to router-selected waypoints, not to steer directly at a raw
target bearing.

**Consequence (as of the original 2026-08-22 investigation, now SUPERSEDED
for every stage — see section 4/5):** this curriculum's `RUN_CANONICAL_*.py`/
`milestone_evaluator.py` machinery could not, as originally written, be used
to evaluate 0051200 (or any router-driven navigation checkpoint) — there was
no router in its loop to exercise, and its own graduated checkpoints were a
different policy lineage entirely (split-branch policies trained end-to-end
on direct target bearing, not on router waypoints). Building a router-driven
variant of this evaluation loop was new integration work, not a fix to
existing wiring — see the 2026-08-22 `MISTAKES.md` entry ("task premise
assumed the canonical Basic->Advanced curriculum was an evaluation-only
harness for an externally-supplied frozen checkpoint") for the concrete
incident this was discovered from. That integration work is now complete for
every stage (Basic through Advanced, training and evaluation both) — section
4 records the recovered target architecture and section 5 records the final
per-stage/per-module status. The statement above ("this curriculum does not
exercise the production router") therefore now describes historical/
pre-recovery behavior for Beginner onward, not the current state; Basic's
own training loop still does not call the router directly either, but it now
delegates steering entirely to `FrozenNavigationSteering`, which does.

## 3. Where router/navigation-checkpoint qualification actually lives

The frozen navigation checkpoint's own qualification uses a **separate,
deliberately decoupled** evaluation stack instead:

- `tests/helpers/router_qualification_harness.py` — `run_episode_general_
  router`/`summarize_general_router`, which DOES call `plan_route`/
  `select_persistent_waypoint`/`TargetPersistenceController` directly,
  driving a loaded PPO checkpoint (e.g. 0051200) to a router-selected
  waypoint each tick.
- `simulator/run_logs/OVERNIGHT_20260815_MONSTER_APPROACH_BASELINE.md` —
  the 850M complete-bot monster-approach evaluation (120 episodes, 6
  strata), whose own header states it used "a fresh, dedicated pool
  ..., decoupled from the canonical curriculum_manifests system." That
  phase concluded closed: 199/200 required kills, one collision
  (`obstacle_approach[3]`, diagnosed as an extreme-heading x nearby-wall
  interaction, not a general router/steering defect), recommendation "do
  not retrain steering," next work flagged as Tower digital-twin
  reconstruction — not further synthetic navigation evaluation.

Do not conflate the two systems: a "does the curriculum still pass"
question is about the Basic->Advanced policy lineage above; a "does the
router/frozen navigation checkpoint still hold up" question belongs to
the `router_qualification_harness.py`/850M-pool lineage, not this
pipeline.

## 4. Recovered architecture: frozen navigation sub-policy (2026-08-22)

**RECOVERED INTENDED DESIGN: the full-farming policy this curriculum
trains does NOT relearn steering.** Steering execution belongs entirely to
the frozen navigation checkpoint (0051200) driven by the production router
(`plan_route`/`select_persistent_waypoint`/`TargetPersistenceController`);
the curriculum's own trainable policy owns only the farming/event (EVA)
action. Recovered from repository evidence, not invented — **Confidence:
HIGH**, from four independent, converging sources:

1. `simulator/basic_training.py::build_fresh_basic_policy`'s own docstring:
   *"no zero-init transplant, no loaded weights of any kind — the
   historical 15k checkpoint and all of its descendants are benchmarks
   only, never a parent for this lineage."* (0051200, 51,200 timesteps, is
   a later refinement in that same navigation-checkpoint lineage —
   `simulator/evaluations/phase2_vs_15k_on_training_maps.json`.)
2. [`PROJECT_GOALS.md`](../PROJECT_GOALS.md) §1: *"Do not retrain it
   [0051200] merely to make it fit this strategy."*
3. `simulator/run_logs/OVERNIGHT_20260815_MONSTER_APPROACH_BASELINE.md`:
   *"isolating whether STEERING specifically still needed training. It
   does not. Recommendation: do not retrain steering."*
4. `tests/helpers/router_qualification_harness.py` already implements
   exactly this "call a frozen `PPO.load()`ed model as a black box, driven
   by externally-computed router waypoints" pattern.

A direct, necessary consequence for the PPO stages (Beginner/Intermediate/
Advanced) that no historical precedent resolves on its own: if the
trainable policy still *samples and logs* a steering action every tick
that then never executes (0051200's steering executes instead), that
reproduces — for PPO specifically — the exact on-policy log-prob/credit-
assignment inconsistency `simulator/basic_environment.py`'s own module
docstring explains at length for why recovery-override and PPO cannot
mix. There is no way to avoid this within stock SB3 PPO other than
genuinely shrinking the trainable action space so it never includes
steering. This was confirmed with the user (2026-08-22) rather than
decided unilaterally.

**Superseded by the completing piece (2026-08-22, same day): learned
farming-target selection.** The trainable policy initially landed with a
`Discrete(len(FarmingEvent))` action space (event only) — but WHICH
actor/group to pursue was still being decided by the environment's own
deterministic `_nearest_reachable_actor_id`/`_best_group_actor_id`
hysteresis, not learned. `docs/PROJECT_GOALS.md` §1 lists "target
selection" as part of the full-farming behavior 0051200 explicitly does
NOT provide, so an event-only trainable policy left target selection with
no real owner at all. `simulator/farming_target_policy.py` closes this:
the trainable policy's action space is now
**`MultiDiscrete([TARGET_ACTION_SIZE, len(FarmingEvent)])` =
`MultiDiscrete([13, len(FarmingEvent)])`** — `0` = `KEEP_CURRENT_TARGET_
ACTION` (persistence, not a fresh pick), `1..12` = select the actor
occupying that `farming.observation.DIRECT_ACTOR_SLOTS` slot in the
*same* observation the policy already reads (no new indexing scheme). The
deterministic heuristic remains a candidate/reachability source and, for
Basic only, a BC teacher (`deterministic_target_teacher_action`) — it is
never again the runtime decision-maker for what to farm once a policy
exists to make that decision. This is a genuinely narrower, one-time
extension of the same on-policy-consistency rule above: a *target*
action that got silently overridden before execution would reproduce the
identical PPO log-prob inconsistency the steering exclusion was built to
avoid, so the same "the trainable action space must be exactly what
executes" discipline now covers target selection too, not just steering.

Basic (pure BC/DAgger, no rollout buffer, no log-prob) never had a
different constraint here in the first place — but building Basic
directly against the same target+event action space (rather than keeping
its historical `SplitSteeringNavigationPolicy` dual-head shape and
bridging it later) turned out to eliminate the Basic→Beginner checkpoint
bridge entirely: see section 5.

## 5. Frozen-navigation + learned-target-selection reconnection: current integration status (COMPLETE, 2026-08-22)

Canonical shared components (`simulator/navigation_subpolicy.py`,
`tests/test_navigation_subpolicy.py`, 12 tests; `simulator/
farming_target_policy.py`, `tests/test_farming_target_policy.py`, 13
tests): `FrozenNavigationSteering` (the per-tick steering oracle:
production router + frozen 0051200, target-id-driven replanning),
`PersistentFarmingTarget` (the learned counterpart of the environment's
own target hysteresis — folds persistence and death/invalidation into one
tick's resolved target id), `FarmingPolicyWrapper` (the `gym.Wrapper`
every stage now trains/evaluates through — exposes
`MultiDiscrete([TARGET_ACTION_SIZE, len(FarmingEvent)])` over
`Box(RAW_OBSERVATION_SIZE,)`, resolves the sampled target action, drives
steering automatically, composes `[steering, event]` into the underlying
env's native step), and `run_composed_episode` (the one composed-episode
rollout function every evaluator dispatches through — see section 12).
`FrozenNavigationSteering`'s own mechanics are a faithful extraction of
the already-validated 2026-08-15 850M mechanism
(`simulator/scratchpad/scratchpad_monster_approach_baseline_eval.py`:
`_synthetic_candidate`, `_observation_without_side_effects`,
`previous_steering` threading) — ported, not reimplemented; target
selection (`farming_target_policy.py`) is a from-scratch design (no prior
implementation existed anywhere in this repository's history — confirmed
by `git log -S`/`-G` archaeology before designing it), reusing the
observation's own existing `DIRECT_ACTOR_SLOTS` ordering rather than
inventing a second actor-indexing scheme.

`FrozenNavigationWrapper` (event-only, `Discrete(len(FarmingEvent))`) and
the Basic→Beginner checkpoint bridge (`simulator/factorized_v193_training.
py::transfer_event_head_to_event_only_policy`, `simulator.
beginner_transition.build_event_only_ppo_from_basic_checkpoint`) both
still exist in the repository (with their own passing tests,
`tests/test_navigation_subpolicy.py`/`tests/test_event_head_transplant.py`)
but are **superseded and no longer used by the training/evaluation
pipeline** — nothing under `simulator/tools/RUN_CANONICAL_*.py` or the
current `simulator/beginner_transition.py`/`simulator/navigation_ppo.py`
calls either one. Building Basic directly in the target+event action
shape (rather than keeping its historical `SplitSteeringNavigationPolicy`
dual-head shape and bridging it to event-only afterward) made the bridge
itself unnecessary: there is no cross-architecture boundary left to
bridge at Basic → Beginner, since all four stages now share one policy
architecture (`simulator.split_branch_policy.SplitFarmingTargetEventPolicy`)
end to end. Retained only as tested legacy, not deleted, since deleting
working, still-passing code that documents the immediately-prior design is
not required by this task and the two are cheap to keep side by side.

| Stage | Steering source | Target-selection source | Status |
|---|---|---|---|
| Basic (`simulator/basic_environment.py::_roll_basic_episode`, `simulator/tools/RUN_CANONICAL_BASIC.py`) | `FrozenNavigationSteering` | `SplitFarmingTargetEventPolicy`'s own target head, BC/DAgger-trained against `deterministic_target_teacher_action` (the environment's `_best_group_actor_id` heuristic, used ONLY as a label source) | **DONE, tested** (`tests/test_basic_stage_frozen_navigation_integration.py`, 4 tests — proves the frozen navigator is actually called, proves the net has no steering-named parameters at all, proves DAgger mining and the event-only supervised update never touch the target head). Stage 3a (scripted-teacher steering BC) and a later steering-BC residue were removed earlier (`MISTAKES.md`, "Basic round loop still trained the vestigial steering head..."); Basic's target-teacher bootstrap stage was added in this pass. |
| Beginner / Intermediate / Advanced PPO training (`simulator/navigation_ppo.py::balanced_training_vec_env_farming_policy`/`resume_ppo_chunk_farming_policy`, `simulator/beginner_transition.py::continue_farming_policy_ppo_chunk`/`rehearse_farming_policy_on_basic_data`, `simulator/tools/RUN_CANONICAL_BEGINNER.py`/`RUN_CANONICAL_INTERMEDIATE.py`/`RUN_CANONICAL_ADVANCED.py`) | `FarmingPolicyWrapper` (drives `FrozenNavigationSteering` toward the resolved target) | `FarmingPolicyWrapper` resolves the policy's own sampled target action via `PersistentFarmingTarget`; the deterministic heuristic plays no runtime role from Beginner onward | **DONE, tested.** One shared `continue_farming_policy_ppo_chunk` function used identically by all three stages (`tests/test_beginner_transition.py`, 5 tests; `tests/test_intermediate_advanced_event_only_continuation.py`, parametrized over `intermediate`/`advanced`). Each script guards every checkpoint load with its own `_require_farming_policy_action_space` (`tests/test_canonical_script_action_space_guards.py`, 6 cases across all three scripts), so a stale event-only or dual-head checkpoint can never be silently resumed. |
| Evaluation — `simulator/milestone_evaluator.py::evaluate_heldout`/`evaluate_challenge` (+ `_parallel`) (heldout/challenge, Beginner/Intermediate/Advanced) | `navigation_steering=`/`use_frozen_navigation=True` dispatches through `navigation_subpolicy.run_composed_episode` instead of the direct-bearing `run_episode` | Resolved inside `run_composed_episode` from the loaded `farming_policy`'s own target head | **DONE, tested.** All three `RUN_CANONICAL_*.py` scripts pass `use_frozen_navigation=True`. |
| Evaluation — `simulator/basic_milestone_evaluator.py` (Basic's own per-round assisted-mode metrics) | `FrozenNavigationSteering` via `_roll_basic_episode` (shared with training) | Basic's own trained target head, via the same `_roll_basic_episode` | **DONE, tested** (`tests/test_basic_milestone_evaluator.py`). Reports `target_disagreement_rate`/`event_disagreement_rate` as two separate numbers, not one combined figure — see this module's own docstring for why (the project's own event-head-collapse history hid behind a combined rate previously). |
| Evaluation — `simulator/beginner_transition.py::zero_shot_raw_diagnostic`/`_parallel` | `FrozenNavigationSteering` via `run_composed_episode` | The just-graduated Basic checkpoint's own target head (same composition every other stage's evaluator uses — no `event_forward` override needed now that Basic shares the one policy architecture) | **DONE, tested.** Purely a Beginner-starting-point diagnostic, never a Basic graduation gate — see `simulator/basic_milestone_evaluator.py`'s module docstring. |

**Reward ownership**: `FarmingPolicyWrapper.step()` excludes the
"approach" reward component (`simulator.reward_model` — movement progress
toward the resolved target, purely a function of the executed steering
action) from the reward it returns, i.e. from what actually enters the
PPO rollout buffer, for the same reason the retired `FrozenNavigation
Wrapper` did — plus subtracts `INVALID_TARGET_SELECTION_PENALTY` on a
tick where the sampled target action selected an empty observation slot
(a real, well-defined outcome to discourage, analogous to the reward
model's existing small invalid-EVA-attempt penalties, not catastrophic).
**Revisited, not changed, for target selection specifically**: excluding
"approach" is still correct even though target selection now legitimately
affects travel cost (a bad target choice costs real time/distance) —
that cost is not lost, it is realized entirely through every OTHER reward
term (fewer kills per unit time, more exposure to contact/obstacle
penalties, less kill-rate reward) rather than through a component whose
formula is purely a function of the executed steering action the policy
does not own. Rewarding target choice through the same "approach"
component that would also reward the frozen navigator's own steering
quality would re-couple the two responsibilities this whole architecture
exists to keep separate.

**Checkpoint/provenance contracts**: `SplitFarmingTargetEventPolicy`
checkpoints (`MultiDiscrete([TARGET_ACTION_SIZE, len(FarmingEvent)])`,
`Box(RAW_OBSERVATION_SIZE,)`) are a distinct, explicit contract from
0051200's own frozen navigation ABI — see `DATA_AND_MODEL_CONTRACTS.md`
section 1f. They never go through `farming.model_contract.
validate_model_contract` (that gate is the LIVE/production runtime's own,
and this whole curriculum has always been simulator-only and out of its
scope — same as every checkpoint lineage before it; that gate still
correctly rejects a stale scalar `Discrete(5)` checkpoint, confirmed
untouched). Every checkpoint's own provenance manifest
(`simulator/run_provenance.py`) records `navigation_subpolicy.
farming_policy_architecture_contract()` — the paired frozen navigation
checkpoint's path and SHA-256 — because a full-farming checkpoint is not
reproducible, or even executable, by itself without knowing which
navigator it composes with. Basic's own two save call sites
(`simulator/basic_training.py::save_checkpoint_with_provenance`, called
from `RUN_CANONICAL_BASIC.py`'s bootstrap and milestone saves) did not pass
this `architecture_contract` through until 2026-08-23 — every Basic
checkpoint's manifest silently inherited `build_run_manifest`'s own
`SplitSteeringNavigationPolicy`/928-value default instead, wrong for the
`SplitFarmingTargetEventPolicy`/923-value checkpoints Basic actually saves;
corrected 2026-08-23, see `tests/test_basic_checkpoint_provenance.py` and
`MISTAKES.md`'s same-day "partial architecture_contract override" entry.

**Resume-identity gate** (pre-merge blocker remediation, 2026-08-23):
`simulator/curriculum_resume_identity.py` defines the minimum identity
(`generation_id`, `policy_action_nvec`, `raw_observation_size`,
`navigation_checkpoint_sha256`, and each stage's own `declared_parent_
checkpoint`) a stored `canonical_<stage>_run_summary.json` round or cached
evaluation (zero-shot diagnostic, pre-/post-rehearsal) must carry to be
trusted for resume/reuse. Beginner/Intermediate/Advanced previously resumed
`consecutive_passes`/`current_checkpoint`/cached evaluations from any
same-named file purely because it existed on disk — including the tracked
pre-2026-08-21-root-collapse summaries that still embed obsolete
`flyff_farming_simulator/` checkpoint paths and a stale `consecutive_
passes=2`. A stored artifact whose identity is missing or mismatched is now
archived byte-identical (never mutated in place) and ignored for resume;
the current run starts fresh from the stage's own declared parent
checkpoint. See `tests/test_curriculum_resume_identity.py`.

**Target invalidation on planner failure** (pre-merge blocker remediation,
2026-08-23): a selected target that is still alive/present but that the
production router cannot currently produce a route to (`SteeringTickResult.
planner_failure=True`) is now treated the same as death/disappearance —
`PersistentFarmingTarget.invalidate()` clears the resolved target to `NONE`
and `FrozenNavigationSteering.reset()` clears its own stale route/
persistence-controller/snapshot state, both in `FarmingPolicyWrapper.step`
and `run_composed_episode`. Previously the target stayed latched
indefinitely (steering held at `NONE` every tick) once a route could not be
found, silently substituting "no movement" for what should have been a new
target decision. Invalidation is immediate (one tick, on the router's own
authoritative failure signal) and never a heuristic substitution of a
different actor — the next tick's `KEEP` resolves to no-target until the
policy explicitly selects again. See `info["target_invalidated_by_planner_
failure"]` for the per-tick telemetry flag and `tests/
test_farming_target_policy.py`'s planner-failure/disappearance/respawn
lifecycle tests.

**Zero-collision hard graduation gate** (`docs/PROJECT_GOALS.md` §2a):
Beginner, Intermediate, and Advanced all gate graduation on
`total_collision_events` (`distinct_contact_events` — genuine collision
EVENTS from `milestone_evaluator._contact_event_stats`, not the older
`contacts_per_100_distance` tick-rate proxy) being exactly zero across
EVERY raw evaluation role — heldout, unseen_templates, and (Beginner
only; Intermediate/Advanced have no challenge manifest yet, so heldout is
their only bar) challenge. Zero collisions is a binary admission
requirement (`docs/PROJECT_GOALS.md` §2a), not a metric traded off
against a role's difficulty — challenge's own deliberately-stressful
framing governs its other, genuinely looser thresholds (contacts-per-
distance, stagnation) only, never collisions; an earlier revision of
Beginner's own script briefly allowed exactly 1 collision event on
challenge specifically, which was a contract violation, corrected
2026-08-23. `total_collision_events` is computed from the evaluator's own
exact per-episode sum (`milestone_evaluator._summarize_episodes`'s
`total_distinct_contact_events` field, `sum(episode["distinct_contact_
events"] for episode in results)`) — each `RUN_CANONICAL_*.py`'s
`_aggregate()` previously reconstructed a total as `round(median *
n_episodes)`, which is mathematically invalid (e.g. per-episode counts
`[0, 0, 1]` have median 0, silently rounding a real collision down to 0)
and could pass a round with genuine, uncounted collisions whenever they
were concentrated in a minority of episodes below the median; corrected
2026-08-23, see `tests/test_collision_aggregation_exact_total.py`.
Advanced's `AUTO_GRADUATION_ENABLED=False` bypass flag
(disabled 2026-08-08 pending this exact fix) has been removed entirely,
restoring real, unattended auto-graduation for Advanced on the same
standard as the other stages.

## Evidence / Sources

- `simulator/tools/RUN_CANONICAL_BASIC.py`, `RUN_CANONICAL_BEGINNER.py`,
  `RUN_CANONICAL_INTERMEDIATE.py`, `RUN_CANONICAL_ADVANCED.py` (direct
  source reads, module docstrings)
- `simulator/curriculum_stages.py`, `simulator/curriculum_manifests.py`
  (direct source reads, module docstrings)
- `simulator/milestone_evaluator.py::run_episode` (direct full-body
  source read — confirms the direct-bearing steering target and the
  absence of any router import/call)
- `tests/helpers/router_qualification_harness.py` (direct source read —
  confirms `plan_route`/`select_persistent_waypoint`/
  `TargetPersistenceController` usage)
- `simulator/run_logs/OVERNIGHT_20260815_MONSTER_APPROACH_BASELINE.md`
  (850M evaluation closure, "decoupled from the canonical
  curriculum_manifests system")
- [`NAVIGATION_AND_MOVEMENT.md`](NAVIGATION_AND_MOVEMENT.md) section 2
  (why the router exists — non-holonomic movement), section 7 (what the
  frozen 0051200 checkpoint is and is not)
- [`DATA_AND_MODEL_CONTRACTS.md`](DATA_AND_MODEL_CONTRACTS.md) section 1
  (0051200's frozen ABI)
- `MISTAKES.md`, 2026-08-22 entries (the incidents this document, and its
  subsequent target-selection extension, were written from)
- `simulator/navigation_subpolicy.py`, `tests/test_navigation_subpolicy.py`
  (frozen-steering oracle + retired event-only wrapper, still tested as
  legacy), `simulator/factorized_v193_training.py::
  transfer_event_head_to_event_only_policy`, `tests/
  test_event_head_transplant.py` (retired Basic→Beginner checkpoint
  bridge, still tested as legacy but unused by the current pipeline — see
  section 5)
- `simulator/farming_target_policy.py`, `tests/test_farming_target_policy.py`
  (learned target selection: `PersistentFarmingTarget`, `FarmingPolicy
  Wrapper`, `deterministic_target_teacher_action` — direct source reads
  and 13 passing tests, 2026-08-22)
- `simulator/split_branch_policy.py::SplitFarmingTargetEventPolicy`,
  `tests/test_split_branch_policy.py` (the one trainable policy
  architecture shared by all four stages)
- `simulator/basic_environment.py`, `simulator/basic_training.py`,
  `simulator/basic_milestone_evaluator.py`, `tests/
  test_basic_stage_frozen_navigation_integration.py`, `tests/
  test_basic_training_pipeline.py`, `tests/test_basic_environment.py`,
  `tests/test_basic_milestone_evaluator.py` (Basic rebuilt directly
  against the target+event action space — direct source reads and passing
  tests, 2026-08-22)
- `simulator/navigation_ppo.py::balanced_training_vec_env_farming_policy`/
  `resume_ppo_chunk_farming_policy`, `simulator/beginner_transition.py::
  continue_farming_policy_ppo_chunk`/`rehearse_farming_policy_on_basic_data`,
  `simulator/milestone_evaluator.py::evaluate_heldout`/`evaluate_challenge`
  (`navigation_steering=`/`use_frozen_navigation=`), `RUN_CANONICAL_
  BEGINNER.py`/`RUN_CANONICAL_INTERMEDIATE.py`/`RUN_CANONICAL_ADVANCED.py`
  (Beginner/Intermediate/Advanced training + evaluation, all sharing one
  policy lineage — direct source reads, and `tests/
  test_beginner_transition.py`/`tests/
  test_intermediate_advanced_event_only_continuation.py`/`tests/
  test_canonical_script_action_space_guards.py`, 2026-08-22)
- `simulator/reward_model.py` (reward-component classification, including
  the target-selection re-examination of the approach-exclusion decision),
  `farming/model_contract.py` (confirmed untouched, confirmed the
  curriculum pipeline has never gone through it)
