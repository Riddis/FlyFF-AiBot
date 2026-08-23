# Data & Model Contracts

## 1. The frozen checkpoint ABI

**Confidence: VERIFIED_CONTRACT.** Evidence: `docs/migration/
CHECKPOINT_INVENTORY.tsv` (313 rows), R10 ruler check (`migration_
integrity.py check` — `r10_checkpoint_count: 313`, `r10_module_
reference_rows: 317`, `r10_failures: []`), `docs/migration/codex_
handoff/PHASE9_REPORT.md`, `docs/migration/codex_handoff/
PHASE10_REPORT.md` section on the 0051200 reload, and
`simulator/split_branch_policy.py` (direct source read).

The canonical, currently-frozen checkpoint every ABI/pickle-compat test
in this repository exercises:

```
Path:   models/generalized_waypoint_both_seed2_0051200.zip
SHA256: 87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50
Policy: simulator.split_branch_policy.SplitSteeringNavigationPolicy
```

**Observation:** `Box(928, float32)` — 923 raw features + 5 sidecar
features. Sidecar = 2 temporal (`recent_progress`, `recent_contact`) + 3
previous-steering one-hot (`prev_straight`, `prev_left`, `prev_right`).

**Action:** `MultiDiscrete([3, 3])` — a steering branch and an
event/value branch, not a single flat `Discrete` space. This
**supersedes** any older documentation describing a `Discrete(5)`
action space (`RUN_FORWARD`/`RUN_FORWARD_LEFT`/`RUN_FORWARD_RIGHT`/
`CAST_EVA`/`RUN_FORWARD_JUMP`) — that description matches a pre-
navigation-refactor policy generation, not the current frozen
checkpoint. See section 4 below.

**Timesteps:** `num_timesteps=51200` (hence the checkpoint's `_0051200`
suffix).

### 1a. Split-branch network structure

`simulator/split_branch_policy.py` defines the exact classes the
checkpoint's pickle stream references by `__module__.__qualname__`:
`SplitSteeringNavigationPolicy`, `SplitSteeringEventPolicy`,
`GeometryAugmentedFeaturesExtractor`, `SplitBranchExtractor`,
`SplitSteeringEventHead`, `NavigationAugmentedFeaturesExtractor`.

- The steering branch and the event/value branch consume distinct
  network inputs — the steering network's `in_features` is 14 (a
  reduced/geometry-focused slice), while the event/value branch draws
  on the raw observation more broadly.
- The event head is effectively `EVENT_NONE` in the current generalized-
  waypoint checkpoint lineage — **this is not an attack/event
  controller.** It is a pure navigation/waypoint-following policy; any
  event-branch output should not be interpreted as combat/cast
  intent without separate verification.

### 1b. Why `simulator/split_branch_policy.py` remains a runtime ABI surface

It is not a "simulator algorithm" awaiting relocation to `navigation/*`
— its role is checkpoint deserializability. Moving these class
definitions would change `__module__.__qualname__` and break
`pickle.loads()` on every existing checkpoint. It is classified
`RUNTIME_ABI_COMPATIBILITY` in `CANONICAL_OWNERS.toml`
(`resolution_phase = "NEVER_WITHOUT_CHECKPOINT_GATE"`), distinct from
the canonical navigation algorithm implementation under `navigation/*`.

### 1c. Pickle / dataclass module-identity compatibility (retired 2026-08-21)

`KinoState`/`RouteEdgeInfo` (`navigation/kinodynamic_route_planner.py`)
and `AdvanceResult` (`navigation/movement_kernel.py`) previously carried
`__module__` pinned to `simulator.kinodynamic_route_planner`/
`simulator.movement_kernel`, backed by zero-behavior re-export shim
files at those paths, so that `pickle.loads()` of a live instance would
find a real importable module at the pinned path. The post-migration
compatibility purge re-examined what that pinning actually protected: a
static pickle disassembly (`pickletools.dis`, no execution) of every
internal file inside `models/generalized_waypoint_both_seed2_
0051200.zip` found zero references to either module or to any of these
three classes anywhere in the checkpoint. The pins existed solely for
`tests/fixtures/migration/router_kernel.json` (a Phase-3 G8c
migration-continuity fixture; its own capture manifest already labeled
its role "migration continuity, not renewed scientific qualification"),
which no current test or product code reads or validates. Both shim
files were deleted and the `__module__` pins removed; all three classes
now carry their natural `navigation.*` identity, proven via a
fresh-subprocess pickle round-trip
(`tests/test_pickle_module_identity_compat.py`). See
[ADR 0002](../decisions/0002-preserve-abi-compatibility-shims.md)'s
Retirement section for the full evidence trail. `simulator/
split_branch_policy.py` (1b above) is unaffected — the checkpoint
genuinely does reference it.

### 1d. R10 and checkpoint immutability

R10 is the ruler's checkpoint-corpus integrity rule: 313 tracked
checkpoint files, 317 serialized module-reference rows, currently zero
failures. **Never rewrite or regenerate a checkpoint file.** No
checkpoint has been retrained, refit, or overwritten by this migration
— every phase's report documents `git diff` against the ABI-relevant
modules as empty when no ABI change was made, and a fresh `PPO.load()`
+ observation/action-space/timestep check whenever one was.

### 1e. Final shipped checkpoint: unresolved

Which checkpoint (0051200 or a future one) actually ships in a future
runtime derivative is **not decided**. 0051200 is simply the frozen
corpus member every current ABI test exercises — not a shipping
decision. Recorded as an `unresolved_future_choices` entry in
`tools/future_runtime_profile/dependency_profiles.toml`.

### 1f. Full-farming curriculum checkpoints: `SplitFarmingTargetEventPolicy` (2026-08-22, COMPLETE)

Under the recovered frozen-navigation-sub-policy architecture
(`docs/architecture/CURRICULUM_TRAINING_PIPELINE.md` section 4), the
canonical Basic->Advanced curriculum's own trainable checkpoint uses one
**target+event** contract, shared unchanged across all four stages
(Basic through Advanced — no per-stage variation, no cross-architecture
bridge anywhere in the lineage): observation `Box(RAW_OBSERVATION_SIZE,)`
(923, not the 928-value navigation-sidecar-augmented observation — that
sidecar is `FrozenNavigationSteering`'s own internal concern, never the
trainable policy's input), action `MultiDiscrete([TARGET_ACTION_SIZE,
len(FarmingEvent)])` = `MultiDiscrete([13, 3])` (target selection: `0`=
keep current target, `1..12`=select the actor in that observation's own
`DIRECT_ACTOR_SLOTS` slot; event: `FarmingEvent`), policy class
`simulator.split_branch_policy.SplitFarmingTargetEventPolicy` — a
two-head split-branch architecture (target head + event head, distinct
from 0051200's own steering+event split), no steering action or
steering-shaped branch at all. This is **distinct from and not a
replacement for** 0051200's own frozen `MultiDiscrete([3,3])`/928-value
ABI (section 1), which is untouched (never loaded through
`farming.model_contract.validate_model_contract` either — that gate is
scoped to the LIVE/production runtime, and this whole curriculum has
always been out of its scope, so it still correctly rejects a stale
scalar `Discrete(5)` checkpoint without any change here).

There is no Basic -> Beginner checkpoint bridge: Basic is trained
directly in this same `MultiDiscrete([13,3])` shape (target head BC/
DAgger-trained against the environment's own deterministic best-group
heuristic used purely as a teacher, never as the runtime decision-maker;
event head trained the same way it always was), so Beginner continues
Basic's own graduated checkpoint via a plain `PPO.load`, exactly like
Intermediate continues Beginner's and Advanced continues Intermediate's
(`simulator/beginner_transition.py::continue_farming_policy_ppo_chunk`,
used identically by all three PPO stages). Every checkpoint's provenance
manifest records the paired frozen navigation checkpoint's path and
SHA-256 (`simulator/navigation_subpolicy.py::
farming_policy_architecture_contract`) — see
`CURRICULUM_TRAINING_PIPELINE.md` section 5 for the full integration
status (Basic through Advanced, training and evaluation both: DONE,
tested).

An earlier, superseded revision of this same recovery effort (same day)
used a narrower **event-only** contract (`Discrete(len(FarmingEvent))`,
plain `ActorCriticPolicy`, target selection left to the environment's own
deterministic hysteresis) with a Basic -> Beginner checkpoint-bridge
function (`simulator/factorized_v193_training.py::
transfer_event_head_to_event_only_policy`, wrapped by `simulator.
beginner_transition.build_event_only_ppo_from_basic_checkpoint`). Both
still exist in the repository with their own passing tests but are
**unused by the current training/evaluation pipeline** — retained as
tested legacy documenting the immediately-prior design, not deleted.
`simulator/navigation_subpolicy.py::event_only_architecture_contract` was
renamed to `farming_policy_architecture_contract` as part of the same
change; nothing currently calls the old name.

## 2. Observation/action/reward contract (farming package)

**Confidence: VERIFIED_CONTRACT for the canonical owners; BEST_CURRENT_
ESTIMATE for some configuration-key-level detail not re-verified this
phase — see the note at the end of this section.**

Canonical owners (per `CANONICAL_OWNERS.toml`):

- `farming/observation_contract.py` — `OBSERVATION_SCHEMA_ID`
- `farming/observation.py` — `OBSERVATION_SIZE`, plus the permanent
  `OBSERVATION_SCHEMA_HASH` re-export
- `farming/actions.py` — `FarmingAction`, `SteeringAction`,
  `FarmingEvent`, `POLICY_ACTION_NVECS`
- `farming/model_contract.py` — `CURRENT_MODEL_CONTRACT`,
  `ModelContractMetadata`, `validate_model_contract`
- `farming/reward.py` — `RewardCalculator`, `RewardComponents`
- `farming/session.py` — `SessionOutcome`, `SessionEndReason`

A saved model embeds its own semantic contract hash; a model saved under
an older/incompatible contract fails preflight rather than being
silently resumed with changed semantics. New model checkpoints (like
0051200) use the current `MultiDiscrete([3,3])` split-branch contract
described in section 1, not the older `Discrete(5)` five-action
contract some pre-migration documentation described.

**Configuration schema note:** `native_farming.json`'s exact key set
(`checkpoint_frequency`, `control_interval_seconds`, teleport/obstacle
penalty tunables, native-memory grace/poll intervals, etc.) is
documented in detail in a predecessor architecture document that
predates this migration's action-space changes. That key-level detail
was **not individually re-verified against current `farming/config.py`
during Phase 13** (out of practical scope for this pass) — treat it as
`BEST_CURRENT_ESTIMATE`, not `VERIFIED_CONTRACT`, until someone
cross-checks it against the live schema. The *existence and canonical
ownership* of `native_farming.json` itself, however, is verified —
see `docs/migration/PHASE11_RUNTIME_RESOURCE_MANIFEST.tsv` row 2
(sha `45e97da66a958a29455ab1eda0c465dfc14b9b8be8ef30fd73521513b8507d0d`).

## Evidence / Sources

- `docs/migration/CHECKPOINT_INVENTORY.tsv`
- `docs/migration/codex_handoff/PHASE9_REPORT.md`,
  `PHASE10_REPORT.md`, `PHASE11_REPORT.md`
- `simulator/split_branch_policy.py`,
  `navigation/kinodynamic_route_planner.py`,
  `navigation/movement_kernel.py`
- `CANONICAL_OWNERS.toml` (`serialized_split_policy_api`,
  `serialized_farming_training_api` concepts)
- [ADR 0002](../decisions/0002-preserve-abi-compatibility-shims.md)
  (Retirement section: the pickle-identity shims' retirement evidence)
- `tests/test_pickle_module_identity_compat.py`
- `docs/migration/PHASE11_RUNTIME_RESOURCE_MANIFEST.tsv`
