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

**Consequence:** this curriculum's `RUN_CANONICAL_*.py`/
`milestone_evaluator.py` machinery cannot be used, as-is, to evaluate
0051200 (or any router-driven navigation checkpoint) — there is no
router in its loop to exercise, and its own graduated checkpoints are a
different policy lineage entirely (925-input-history split-branch
policies trained end-to-end on direct target bearing, not on router
waypoints). Building a router-driven variant of this evaluation loop
would be new integration work, not a fix to existing wiring — see the
2026-08-22 `MISTAKES.md` entry ("task premise assumed the canonical
Basic->Advanced curriculum was an evaluation-only harness for an
externally-supplied frozen checkpoint") for the concrete incident this
was discovered from.

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
