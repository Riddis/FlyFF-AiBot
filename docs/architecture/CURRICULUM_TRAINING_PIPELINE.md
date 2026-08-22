# Curriculum Training Pipeline (canonical Basic -> Advanced)

**Confidence: VERIFIED_CONTRACT** for module roles, call paths, and the
router-independence claim (direct source reads, cited below);
**HISTORICAL_EVIDENCE** for the graduation-history claims ported from
docstrings/run logs.

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

## 2. This curriculum does not exercise the production router

**This is the fact most likely to be assumed incorrectly by anyone
approaching this pipeline from the navigation/router side of the
codebase — verify it again from source if it matters to your task,
rather than trusting this document alone.**

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
decided unilaterally: **the PPO stages' trainable policy has a
`Discrete(len(FarmingEvent))` action space, not `MultiDiscrete([3,3])`.**
Basic (pure BC/DAgger, no rollout buffer, no log-prob) has no such
constraint — its checkpoint keeps `SplitSteeringNavigationPolicy`'s
existing dual-head architecture unchanged (for BC/DAgger tooling
compatibility), with the steering head simply never trained and never
executed (vestigial by design, not by neglect).

## 5. Frozen-navigation reconnection: current integration status (COMPLETE, 2026-08-22)

Canonical shared component (`simulator/navigation_subpolicy.py`,
`tests/test_navigation_subpolicy.py`, 12 tests): `FrozenNavigationSteering`
(the per-tick oracle: production router + frozen 0051200, target-id-driven
replanning), `FrozenNavigationWrapper` (a `gym.Wrapper` exposing
`Discrete(len(FarmingEvent))` and `Box(RAW_OBSERVATION_SIZE,)` to a
trainable policy, driving steering automatically), and `run_composed_episode`
(the one composed-episode rollout function every evaluator dispatches
through — see section 12). All are a faithful extraction of the
already-validated 2026-08-15 850M mechanism
(`simulator/scratchpad/scratchpad_monster_approach_baseline_eval.py`:
`_synthetic_candidate`, `_observation_without_side_effects`,
`previous_steering` threading) — ported, not reimplemented.

Basic -> Beginner checkpoint bridge (`simulator/factorized_v193_training.
py::transfer_event_head_to_event_only_policy`, wrapped by `simulator.
beginner_transition.build_event_only_ppo_from_basic_checkpoint`,
`tests/test_event_head_transplant.py`, 2 tests): copies a
`SplitSteeringNavigationPolicy`'s `event_net`/`vf_net`/`event_out`/
`value_net` weights into a fresh plain `ActorCriticPolicy`
(`net_arch=dict(pi=event_net_arch, vf=vf_net_arch)`, `Discrete` action
space) — proven via `torch.testing.assert_close` that the transferred
policy's event-action distribution and value estimate are **bit-for-bit
identical** to the source's own event branch on real observations, not
merely "runs without error." The steering branch is discarded entirely,
per section 4 (nothing in it was ever meaningfully trained).

| Stage | Steering source | Status |
|---|---|---|
| Basic (`simulator/basic_environment.py::_roll_basic_episode`, `simulator/tools/RUN_CANONICAL_BASIC.py`) | `FrozenNavigationSteering` (target-id from the environment's own native hysteresis) | **DONE, tested** (`tests/test_basic_stage_frozen_navigation_integration.py`, 5 tests — sentinel-injection proof the net's own steering head can never reach the executed action, plus two tests proving neither DAgger mining nor the round's supervised loss can be moved by that head). Stage 3a (scripted-teacher steering BC) removed from `RUN_CANONICAL_BASIC.py`; a later residue (a per-round steering-only BC update that survived that removal) found and removed too — see `MISTAKES.md`, "Basic round loop still trained the vestigial steering head after the frozen-navigation recovery." |
| Beginner PPO training (`simulator/navigation_ppo.py::balanced_training_vec_env_event_only`/`resume_ppo_chunk_event_only`, `simulator/beginner_transition.py::build_event_only_ppo_from_basic_checkpoint`/`continue_event_only_ppo_chunk`/`rehearse_event_only_on_basic_data`, `simulator/tools/RUN_CANONICAL_BEGINNER.py`) | `FrozenNavigationWrapper` + the transferred event-only policy | **DONE, tested** (`tests/test_beginner_event_only_ppo_integration.py`, 7 tests: env action-space contract, event-distribution-preserving transfer, frozen-checkpoint-bytes-untouched, on-policy steering/event composition, rehearsal + composed evaluation, a dual-head-checkpoint rejection, navigator-SHA provenance). Verified end to end with a real (tiny-curriculum) PPO chunk + evaluation run, not tests alone. |
| Intermediate / Advanced PPO training (`RUN_CANONICAL_INTERMEDIATE.py`/`RUN_CANONICAL_ADVANCED.py`) | Same `continue_event_only_ppo_chunk`/`rehearse_event_only_on_basic_data` — no checkpoint bridge needed, the Beginner-graduated checkpoint is already event-only | **DONE, tested** (`tests/test_intermediate_advanced_event_only_continuation.py`, parametrized over both internal stage ids). Both scripts guard every checkpoint load with `_require_event_only_action_space`, so a pre-recovery dual-head checkpoint (Advanced's own historical round checkpoints, none of which remain on disk) can never be silently resumed. |
| Evaluation — `simulator/milestone_evaluator.py::evaluate_heldout`/`evaluate_challenge` (+ `_parallel`) (heldout/challenge, Beginner/Intermediate/Advanced) | `navigation_steering=`/`use_frozen_navigation=True` dispatches through `navigation_subpolicy.run_composed_episode` instead of the direct-bearing `run_episode` | **DONE, tested.** `RUN_CANONICAL_BEGINNER.py`, `RUN_CANONICAL_INTERMEDIATE.py`, and `RUN_CANONICAL_ADVANCED.py` all pass `use_frozen_navigation=True`. |
| Evaluation — `simulator/basic_milestone_evaluator.py` (Basic's own per-round assisted-mode metrics) | `FrozenNavigationSteering` via `_roll_basic_episode` (shared with training) | **CONFIRMED already correct** — audited directly, no fix needed (see section 11 of the audit this table summarizes). |
| Evaluation — `simulator/beginner_transition.py::zero_shot_raw_diagnostic`/`_parallel` | `FrozenNavigationSteering` via `run_composed_episode(..., event_forward=_dual_head_event_forward)` (reads only Basic's dual-head net's event branch) | **DONE, fixed.** Previously direct-bearing (`_raw_policy_forward` read the net's own steering head); now reuses the same composed-episode function the event-only stages use, with a small `event_forward` override so it works against Basic's still-dual-head-shaped checkpoint. |

**Reward ownership** (section 15 of the recovery task): `FrozenNavigationWrapper.step()`
excludes the "approach" reward component (`simulator.reward_model` —
movement progress toward the target, purely a function of the executed
steering action) from the reward it returns, i.e. from what actually enters
the PPO rollout buffer — that component's whole purpose was to shape a
steering skill this policy no longer owns. Every other reward term (kill,
EVA penalties/bonuses, contact/obstacle/teleport penalties, jump flair)
is left untouched: they are either event-controllable, a shared outcome, or
a navigation-caused-but-not-steering-shaping outcome signal uncorrelated
with (not biasing) this policy's own action.

**Checkpoint/provenance contracts** (sections 17–19): event-only checkpoints
(`Discrete(len(FarmingEvent))`, plain `ActorCriticPolicy`, `Box(RAW_OBSERVATION_SIZE,)`)
are a distinct, explicit contract from 0051200's own frozen navigation ABI —
see `DATA_AND_MODEL_CONTRACTS.md` section 1f. They never go through
`farming.model_contract.validate_model_contract` (that gate is the LIVE/
production runtime's own, and this whole curriculum has always been
simulator-only and out of its scope — same as the pre-recovery `SplitSteering
NavigationPolicy`/Phase-2 checkpoints before it; that gate still correctly
rejects a stale scalar `Discrete(5)` checkpoint, untouched). Every event-only
checkpoint's own provenance manifest (`simulator/run_provenance.py`) records
`navigation_subpolicy.event_only_architecture_contract()` — the paired
frozen navigation checkpoint's path and SHA-256 — because an event-only
checkpoint is not reproducible, or even executable, by itself without
knowing which navigator it composes with.

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
- `MISTAKES.md`, 2026-08-22 entry (the incident this document was
  written from)
- `simulator/navigation_subpolicy.py`, `tests/test_navigation_subpolicy.py`,
  `simulator/factorized_v193_training.py::transfer_event_head_to_event_only_policy`,
  `tests/test_event_head_transplant.py`,
  `tests/test_basic_stage_frozen_navigation_integration.py` (section 4/5
  reconnection work — direct source reads and passing tests, 2026-08-22)
- `simulator/navigation_ppo.py::balanced_training_vec_env_event_only`/
  `resume_ppo_chunk_event_only`, `simulator/beginner_transition.py::
  build_event_only_ppo_from_basic_checkpoint`/`continue_event_only_ppo_chunk`/
  `rehearse_event_only_on_basic_data`/`save_event_only_checkpoint_with_
  provenance`, `simulator/milestone_evaluator.py::evaluate_heldout`/
  `evaluate_challenge` (`navigation_steering=`/`use_frozen_navigation=`),
  `RUN_CANONICAL_BEGINNER.py`/`RUN_CANONICAL_INTERMEDIATE.py`/
  `RUN_CANONICAL_ADVANCED.py` (Beginner/Intermediate/Advanced training +
  evaluation reconnection — direct source reads, a real tiny-curriculum PPO
  chunk + evaluation smoke run, and `tests/test_beginner_event_only_ppo_
  integration.py`/`tests/test_intermediate_advanced_event_only_
  continuation.py`, 13 tests total, 2026-08-22)
- `simulator/reward_model.py` (reward-component classification — section 15),
  `farming/model_contract.py` (confirmed untouched, confirmed the curriculum
  pipeline has never gone through it — section 18)
