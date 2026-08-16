# PHASE 2 — FREEZE FINGERPRINTS

> **CORRECTION — G10b WITHDRAWN (review finding, applied forward).**
> This report originally claimed G10b GREEN and "PHASE 3 SAFE TO CONSIDER: YES".
> **Both claims are withdrawn.** Review found that the executor hit the section-8.1
> STOP condition (categories 3-6 ambiguous, no source-backed identification, no
> Phase-0 load baseline) and continued behind a self-invented "lexicographically
> first unselected" rule instead of stopping. Determinism does not cure that:
> because no prior baseline existed, this selection would have defined the
> **permanent first baseline**, so selection provenance was required.
>
> Current status: **G10b = BLOCKED_PENDING_AUTHORIZED_SELECTION**, exit condition
> **E = FAIL/PENDING**, **Phase 2 is NOT complete**, and
> **PHASE 3 SAFE TO CONSIDER: NO**.
>
> A read-only audit found **0 of 4** ambiguous categories are uniquely determined
> by pre-existing evidence. See `docs/migration/PHASE2_G10B_SELECTION_AMENDMENT.md`
> and `docs/migration/PHASE2_G10B_SELECTION_AUDIT.tsv` (328 candidate rows).
> No further `PPO.load` has been run.
>
> Section 8 below is retained unedited as the historical record of the
> provisional run; its results are **diagnostic only** and do not satisfy G10b.
> Everything else in this report — the portability repair, G4, G10a, G11, the
> ruler results, and the Phase-1 verification — remains accepted and unchanged.

Executor: **Claude**. Branch: `refactor/consolidation-phase1` (**not pushed**).
Strategy unchanged: `C -> A, defer B`.

---

## 1. Claude's independent Phase-1 verification

Verified against the actual repository, not against `PHASE1_REPORT.md`.

### Commits checked

| SHA | Subject | Contents |
|---|---|---|
| `80090cbad1dd0ef2ce09d87e01dc162f5a0306d6` | Phase 1 blocked: authoritative consolidation plan missing | 5 handoff journals only |
| `61e1abefdba029cf826ac8bf1c2191d41c7b2ceb` | Phase 1: add migration integrity ruler and frozen baseline | 7 files, registries/tooling/tests only |
| `31cf236e0808a82cc50873832963c41f25dd9184` | Phase 1: finalize ruler report and handoff | 5 handoff journals only |
| `ad61b991e4af436eef8705b49978990464cc28f5` | Phase 1 hardening: close ruler integrity blind spots | 5 files, ruler + baseline only |
| `9cf13873cd4a5ca43df02f28c86463bfd8a823c9` | Phase 1 hardening: finalize amendment report | 5 handoff journals only |

### Scope integrity

- `dc734bb..9cf1387` touches exactly 12 paths, all migration tooling/docs/tests/registries.
- **Zero** changes under `foreground_vision_bot/`, `flyff_farming_recorder/`, `flyff_farming_simulator/`.
- `CHECKPOINT_INVENTORY.tsv`, `CHECKPOINT_MODULE_REFERENCES.tsv`, `ARTIFACT_MANIFEST.tsv`, `EFFECTIVE_CONFIG_BASELINE.json` — untouched by Phase 1.
- Index empty, worktree clean, all three protected tags at their exact expected SHAs.

### Files read completely

`CANONICAL_OWNERS.toml`, `BRIDGES.md`, `BASELINE_VIOLATIONS.json`, `BASELINE_VIOLATIONS.md`,
`DUPLICATE_CONTENT_REPORT.tsv`, `tools/migration_integrity.py`, `tests/test_migration_integrity.py`,
`CHECKPOINT_INVENTORY.tsv`, `CHECKPOINT_MODULE_REFERENCES.tsv`, `ARTIFACT_MANIFEST.tsv`,
`EFFECTIVE_CONFIG_BASELINE.json`, and all five `codex_handoff/` journals.

### R7c hardening — implementation inspected, then independently probed

The implementation was read directly (`registered_reexports`), then exercised with a
**9-case probe written independently of Codex's own tests**:

| case | expected | result |
|---|---|---|
| `from X import ControlledSymbol` (no `__all__`) | flagged | flagged |
| `from X import ControlledSymbol as PublicName` | flagged | flagged |
| `from X import ControlledSymbol as _ControlledSymbol` | **not** flagged | not flagged |
| private alias listed in `__all__` | flagged | flagged |
| relative `from .owner import ControlledSymbol` | flagged | flagged |
| `from X import *` | not flagged | not flagged |
| `import X.owner` | not flagged | not flagged |
| unrelated symbol | not flagged | not flagged |
| function-scope import | not flagged | not flagged |

Registered shim suppression confirmed, and an unregistered rogue re-export in the same
run was still caught. **Result: PASS.**

Noted, non-blocking: `from X import *` is not treated as a re-export. It never names a
controlled symbol, so it is outside R7c's stated contract — recorded as an observation,
not a defect.

### Counts, R9, R10, bridges

Formal check exit **0**:

```
R6 = 7   R7a = 35   R7b = 0   R7c = 200
r9_violations = 0
r10_checkpoint_count = 313   r10_module_reference_rows = 317   r10_failures = []
torch_modules_added = []     bridge_errors = []   ownership_errors = []
```

R10 classification and resolution verified explicitly:

| module | classification | resolves to |
|---|---|---|
| `farming.sb3_training` | repository-local | `…/Flyff RL - Phase1/foreground_vision_bot/farming/sb3_training.py` |
| `simulator.split_branch_policy` | repository-local | `…/Flyff RL - Phase1/flyff_farming_simulator/simulator/split_branch_policy.py` |
| `stable_baselines3.common.policies` | external | `…/.venv/Lib/site-packages/stable_baselines3/common/policies.py` |

Both repo-local ABI modules resolve **inside this worktree**, not to an identically named
installed package and not to the sibling reference tree.

- **B3**: real source confirmed in `flyff_farming_simulator/tools/inventory_recordings.py` —
  `_RECORDER_ROOT` (line 24), `sys.path.insert(0, str(_RECORDER_ROOT))` (lines 25-26),
  `from recorder.movement_classification import MovementControlClassifier` (line 32).
- **B4**: `historical-reproduction-baseline-20260815` → `a90de59232b81753c1b2ea35b8990325c26674e5`, exact. Not retargeted.
- **B1/B2**: `status = "future"`, `locations = []` — future and uninstalled.
- `CANONICAL_OWNERS.toml current_phase` is the sole active-phase source; `PHASE_7` bridges valid before Phase 7 and expired at Phase 7.

### Tests and determinism

- Focused migration-integrity suite: **17 passed**.
- D1 reproduced **byte-identically**: 119 exact pairs, 31 AST-similar, SHA-256 `187a8de50d27379ad983c0885fbcb31c8ac46aaa2dc39bd296164d5330e0a5d5`.

### Decision

**PHASE 1 VERIFIED BY CLAUDE — ACCEPTED.**

---

## 2. Phase-0 artifact portability defect discovered by G11

Phase 2's first real finding. This section exists because G11 initially **could not pass**,
and the cause was neither artifact drift nor a Codex error.

### Why Phase 0 passed in the original worktree but failed in a fresh one

Phase 0 froze raw SHA-256 bytes for 27 tracked artifacts in `ARTIFACT_MANIFEST.tsv`, and
those hashes were computed in the **original reference worktree**, whose text files were
checked out with CRLF. Phase 0 then added `.gitattributes` line 1 `* text=auto eol=lf`
(commit `b8206bb`, "set forward-looking LF text policy") and a narrow `-text`
byte-preservation list (commit `a90de59`) covering eight historical-guard files.

Any worktree created **after** that policy materializes text files with LF. So the frozen
CRLF hashes stopped reproducing on a fresh checkout — while remaining perfectly valid in
the original tree and in the external snapshot. This is the **same defect class already
recorded as D7/D8** in `DECISION_LOG.md`, recurring on a wider file set that never got the
same exemption treatment.

Measured, not assumed:

| location | manifest entries reproduced |
|---|---|
| original reference tree | **348 / 348** |
| external Phase-0 snapshot | **348 / 348** |
| fresh consolidation worktree (before repair) | 15 of 27 tracked; **12 failed** |

### The exact 12 affected files

Derived **mechanically** from the manifest against all three locations and required to
equal the expected set before any repair was allowed:

1. `flyff_farming_simulator/map_assets/map.json`
2. `foreground_vision_bot/mapper/maps/tower_aoe/map.json`
3. `flyff_farming_simulator/recordings/INDEX.json`
4. `flyff_farming_simulator/recordings/INDEX.md`
5. `flyff_farming_recorder/calibration_holdout_ramp_results.csv`
6. `flyff_farming_recorder/calibration_holdout_step_results.csv`
7. `flyff_farming_recorder/calibration_steering_pulses.csv`
8. `flyff_farming_recorder/calibration_tick_extraction.csv`
9. `flyff_farming_recorder/calibration_tick_extraction_v2.csv`
10. `flyff_farming_recorder/calibration_trials.csv`
11. `flyff_farming_recorder/movement_calibration.csv`
12. `flyff_farming_recorder/movement_calibration_steering.csv`

**Correct framing:** only **2 of the 6** Tower map artifacts were affected (the two
`map.json` copies). `occupancy.npy` and both `coordinate_frame.json` copies already
reproduced their frozen hashes and were deliberately left untouched. The other ten are the
two recording index files and the eight calibration CSVs — not Tower map files. The repair
was driven by the demonstrated mismatch set, never broadened by directory.

### Why this stores existing frozen bytes rather than changing content

Every one of the 12 was mechanically proven to differ from the frozen Phase-0 bytes by
**line-ending representation only**: CR-stripped byte comparison equal, and structural
comparison equal (JSON re-serialized with sorted keys; CSV parsed to row tuples). Had any
file shown a non-EOL difference the repair would have stopped without overwriting it.

The exact bytes were copied raw from the verified external Phase-0 snapshot. Nothing was
regenerated, reparsed, resaved, or programmatically line-ending-converted. Each file was
required to equal its manifest SHA-256 **and** size on disk before staging, and each staged
blob's content SHA-256 was then required to equal the manifest value — confirming git
stored raw bytes rather than a transformed copy.

Evidence: `docs/migration/PHASE0_ARTIFACT_PORTABILITY_REPAIR.tsv`, every row
`difference_class = line_endings_only`.

**This is not a product-behavior change and must never be described as modifying map or
calibration content.**

### Repair commit and fresh-worktree proof

Repair commit: **`39a32a58c1cab800e5024604e7eeaba952d0a46f`** — "Preserve exact Phase-0
artifact bytes across fresh worktrees". `current_phase` deliberately stayed `1` across this
commit and advanced to `2` only after the proof below passed.

A genuinely disposable worktree was created from the repair commit with **no manual file
copying** (`git worktree add --detach`). In it:

| check | result |
|---|---|
| all 27 tracked manifest entries reproduce | **27 / 27** |
| the 12 repaired paths reproduce | 12 / 12 |
| both `map.json` = `faaf8633…fe815` | PASS |
| both `occupancy.npy` = `62fa3c9e…d789b` | PASS |
| both `coordinate_frame.json` = `40339f6c…a0414` | PASS |
| paired Tower files byte-identical | 3 / 3 |
| `.skip_legacy_import` present | PASS |
| no semantic parsed value changed vs pre-repair | 12 / 12 |

### Confirmation

No protected tag was retargeted. `pre-consolidation-head`, `historical-reproduction-baseline-20260815`,
and `pre-consolidation-complete` all remain at their exact original SHAs. No historical result
was rewritten. The `-text` additions are additive; the global `* text=auto eol=lf` policy
remains in force and `git add --renormalize` was **not** run.

---

## 3. Phase-2 base SHA

`9cf13873cd4a5ca43df02f28c86463bfd8a823c9`

## 4. Final Phase-2 HEAD

`the commit containing this file; resolve exactly with git rev-parse HEAD` (documentation commit; the three preceding Phase-2 SHAs below are exact)

## 5. Every Phase-2 commit

### `39a32a58c1cab800e5024604e7eeaba952d0a46f` — Preserve exact Phase-0 artifact bytes across fresh worktrees

```
.gitattributes
docs/migration/PHASE0_ARTIFACT_PORTABILITY_REPAIR.tsv
docs/migration/codex_handoff/COMMAND_LOG.tsv
docs/migration/codex_handoff/HANDOFF.md
docs/migration/codex_handoff/STATE.json
docs/migration/codex_handoff/TEST_LOG.md
flyff_farming_recorder/calibration_holdout_ramp_results.csv
flyff_farming_recorder/calibration_holdout_step_results.csv
flyff_farming_recorder/calibration_steering_pulses.csv
flyff_farming_recorder/calibration_tick_extraction.csv
flyff_farming_recorder/calibration_tick_extraction_v2.csv
flyff_farming_recorder/calibration_trials.csv
flyff_farming_recorder/movement_calibration.csv
flyff_farming_recorder/movement_calibration_steering.csv
flyff_farming_simulator/map_assets/map.json
flyff_farming_simulator/recordings/INDEX.json
flyff_farming_simulator/recordings/INDEX.md
foreground_vision_bot/mapper/maps/tower_aoe/map.json
```

### `b7355a9359143739ad725486f99b72d5391cd945` — Phase 2: freeze contract, physics, and map fingerprints as hard gates

```
CANONICAL_OWNERS.toml
docs/migration/PHASE2_FINGERPRINTS.toml
docs/migration/tests/test_phase2_fingerprints.py
docs/migration/tools/phase2_fingerprints.py
docs/migration/tools/phase2_representative_load.py
```

### `98c4aead0824aacb4a86eca574ef277f12570379` — Phase 2: freeze generated checkpoint and representative-load evidence

```
docs/migration/DUPLICATE_CONTENT_REPORT.tsv
docs/migration/PHASE2_CHECKPOINT_SUPPLEMENT.tsv
docs/migration/PHASE2_REPRESENTATIVE_LOAD_BASELINE.tsv
docs/migration/PHASE2_REPRESENTATIVE_SELECTION.tsv
```

### Documentation commit (this file) — Phase 2: report and handoff journals

```
docs/migration/codex_handoff/COMMAND_LOG.tsv
docs/migration/codex_handoff/HANDOFF.md
docs/migration/codex_handoff/PHASE2_REPORT.md
docs/migration/codex_handoff/STATE.json
docs/migration/codex_handoff/TEST_LOG.md
```

---

## 6. G4 — contract fingerprints

All literals live in one migration-owned registry, `docs/migration/PHASE2_FINGERPRINTS.toml`.
They are **not** duplicated across test files and are **not** placed in production runtime
modules. Every value was read from current source and then pinned; **no product constant was
edited to match**.

| pinned literal | value | source location(s) |
|---|---|---|
| `OBSERVATION_SCHEMA_ID` | `native-unified-923-v4` | `flyff_farming_simulator/farming/observation.py:30`, `foreground_vision_bot/farming/observation.py:30`, `flyff_farming_recorder/recorder/session.py:52` |
| `OBSERVATION_SCHEMA_HASH` | `F2D568C1C4A4B5F577C9C2E36A37B1C5533C2CE28D415846C3B68EC293C84609` | `…/farming/observation.py:461` (both roots), `recorder/session.py:53` |
| `OBSERVATION_SIZE` | `923` | `…/farming/observation.py:56` (both roots) |
| `POLICY_ACTION_NVECS` | `(3, 3)` | `…/farming/actions.py:95` (both roots) |
| `SIDECAR_SIZE` | `5` | `flyff_farming_simulator/simulator/navigation_history.py:65` |
| `POLICY_INPUT_SIZE` | `928` | `flyff_farming_simulator/simulator/navigation_history.py:66` |
| `RAW_OBSERVATION_SIZE` | `923` | `flyff_farming_simulator/simulator/navigation_history.py:62` |
| `MODEL_CONTRACT_METADATA_VERSION` | `2` | `…/farming/model_contract.py:31` (both roots) |

### PHYSICS_VERSION — exact value and name resolution

The accepted plan calls this `PHYSICS_VERSION`, but **no symbol of that name exists in
source**. The authoritative physics-model version tag in the current
`flyff_farming_simulator/simulator/movement_kernel.py` is:

```
MOVEMENT_PHYSICS_MODEL_ID = "live_calibrated_arc"          # line 61
LEGACY_MOVEMENT_PHYSICS_MODEL_ID = "legacy_recorded_iid"   # line 62
```

**PHYSICS_VERSION frozen value: `live_calibrated_arc`** — read from source, never taken
from the task prompt. The registry records this name resolution explicitly rather than
inventing a symbol, and a test asserts that no symbol literally named `PHYSICS_VERSION`
exists, so the resolution cannot silently rot.

The calibrated constants the tag names are pinned alongside it, so a constant change cannot
hide behind an unchanged tag string:

| constant | value | line |
|---|---|---|
| `PATH_LENGTH_CELLS_PER_TICK` | `2.738491` | 73 |
| `ONSET_TURN_RADIANS` | `0.792978` | 74 |
| `STEADY_TURN_RADIANS` | `0.873649` | 75 |
| `DEFAULT_SUBSTEPS` | `10` | 77 |

### Recomputation result (5.2)

Not a hardcoded-vs-hardcoded comparison. Each implementation was required to **recompute**
its schema hash in an isolated subprocess (`farming.*` exists in two roots and cannot be
imported twice in one interpreter):

| owner | `OBSERVATION_SCHEMA_HASH` constant | `observation_schema_hash()` recomputed | agree |
|---|---|---|---|
| `flyff_farming_simulator` | `F2D568C1…C84609` | `F2D568C1…C84609` | yes |
| `foreground_vision_bot` | `F2D568C1…C84609` | `F2D568C1…C84609` | yes |
| `flyff_farming_recorder` | `F2D568C1…C84609` (metadata-only, no builder) | n/a | n/a |

Also verified: `len(OBSERVATION_FIELDS) == OBSERVATION_SIZE == 923` in both roots;
`POLICY_ACTION_NVECS == (len(SteeringAction), len(FarmingEvent)) == (3,3)`;
`SIDECAR_SIZE == 2 + 3`; `POLICY_INPUT_SIZE == 923 + 5`.

**G4: GREEN — 0 failures.**

---

## 7. G10a — independent checkpoint metadata invariance

Preserved artifact source used (read-only, nothing copied into the worktree):

```
C:\Users\Ridd\FlyffRL_Backups\pre_consolidation_20260815\Flyff RL\
```

Its checkpoint files were confirmed to agree with the Phase-0 `ARTIFACT_MANIFEST` records
before use (**348/348**). The consolidation worktree contains only **1** of the 313
checkpoints (the tracked `0051200` ZIP), so running G10a there would have been the exact
trap the plan warned about; all 313 were read from the snapshot.

| metric | result |
|---|---|
| checkpoints represented | **313 / 313** |
| Phase-0 fields compared per row | 12 |
| field mismatches | **0** |
| serialized module references reproduced | **317 / 317**, exact set equality |
| artifact byte changes | none |

Fields compared exactly: repo-relative path, `size_bytes`, `sha256`, `policy_class_module`,
`policy_class_qualname`, `sb3_version`, `farming_contract_metadata_version`,
`obs_space_type`, `obs_space_shape`, `obs_space_dtype`, `action_space_type`,
`action_space_spec`, `loadable_under_current_source`.

### Did the Phase-0 TSV contain every Revision-2-required field?

**No.** The actual Phase-0 header has 13 columns and contains **no `policy_kwargs` and no
`net_arch`**. Those were **not** backdated into the Phase-0 file, which remains byte-identical
to its preservation-checkpoint state.

Instead `docs/migration/PHASE2_CHECKPOINT_SUPPLEMENT.tsv` (313 rows) freezes the missing
architecture metadata, derived from the same unchanged preserved bytes, with every row keyed
by **repo-relative path + checkpoint SHA-256** and labelled:

> `first frozen during Phase 2 because Phase-0 inventory did not contain this field`

Supplemental fields first frozen in Phase 2: `policy_kwargs_json`, `net_arch_json`,
`obs_space_low_min`, `obs_space_high_max`, `action_space_start`, `contract_hash`,
`contract_observation_schema_id`, `contract_observation_schema_hash`,
`contract_observation_size`, `serialized_repo_local_references`.

Later migration gates consume **both** the immutable Phase-0 baseline and this supplement.

### Full serialized-reference scan

Checkpoint ABI is more than `policy_class`, and that Phase-0 discovery is preserved. All
five hard repo-local paths are present in the reproduced corpus:

```
simulator.split_branch_policy.SplitSteeringNavigationPolicy   (275 rows)
simulator.split_branch_policy.SplitSteeringEventPolicy        (5 rows)
farming.sb3_training.TerminalPrefixRolloutBuffer              (2 rows)
farming.sb3_training.TrainingBoundary                         (2 rows)
farming.sb3_training.TrainingBoundaryKind                     (2 rows)
NONE/NONE                                                     (31 rows)
```

Worth recording: the first regeneration produced **315** rows, not 317. The two missing rows
were `farming.sb3_training.TrainingBoundaryKind` on the bot models, whose `STACK_GLOBAL`
module operand arrives via `BINGET` from the pickle **memo** rather than as a literal string.
The opcode scanner was corrected to track the memo, after which the corpus reproduced exactly.
The frozen Phase-0 evidence was never altered to accommodate the scanner.

**G10a: GREEN — 0 failures.**

---

## 8. G10b — representative real `PPO.load` gate
> **WITHDRAWN — PROVISIONAL/DIAGNOSTIC ONLY. Does not satisfy G10b.**
> Retained unedited below as the historical record of what was actually run.
> The file selection was not source-backed; see the amendment.

**No Phase-0 real-load baseline existed.** No comparison to a nonexistent baseline is
claimed. Because product source has not changed since the preservation checkpoint, this is
transparently established now:

> **first real-load outcome frozen in Phase 2; no earlier baseline existed**

### Selection frozen BEFORE execution

`docs/migration/PHASE2_REPRESENTATIVE_SELECTION.tsv` was written and inspected before any
`PPO.load` ran, so no file could be chosen after seeing which ones load. 17 checkpoints
across all seven declared categories, all distinct.

**Deviation, recorded honestly:** the accepted plan names seven categories but identifies
exact files for only three (the seed2 lineage, all five `split_branch_pilot*`, both bot
models — 13 files). Categories 3-6 ("one declared 925-era", "one declared 928-era", "one
`canonical_advanced_ppo_*`", "one from `_quarantine/`") have many candidates, and no
Phase-0 or Phase-1 evidence declares which. Rather than cherry-pick, an explicit
deterministic rule was fixed in advance and encoded in the tool: *within a category take the
lexicographically first repo-relative path in the frozen Phase-0 inventory that is not
already selected, resolving narrow name-based categories before broad shape-based ones.*
The rule is independent of any load outcome and reproduces byte-identically. **If you want
different representatives for categories 3-6, say so and the selection can be re-declared.**

### Outcomes

14 loaded, 3 failed, **0 gate failures**.

| category | checkpoint | sha256 | outcome | policy class / exception | obs space | action space | meta |
|---|---|---|---|---|---|---|---|
| seed2 lineage | `generalized_waypoint_both_seed2_0010240.zip` | `b1f23cd221126595…` | loaded | `simulator.split_branch_policy.SplitSteeringNavigationPolicy` | `Box(-1.0, 1.0, (928,), float32)` | `MultiDiscrete([3 3])` | ABSENT |
| seed2 lineage | `generalized_waypoint_both_seed2_0020480.zip` | `40c77ab44b0b3019…` | loaded | `…SplitSteeringNavigationPolicy` | `(928,)` | `MultiDiscrete([3 3])` | ABSENT |
| seed2 lineage | `generalized_waypoint_both_seed2_0030720.zip` | `3cf06930e9131318…` | loaded | `…SplitSteeringNavigationPolicy` | `(928,)` | `MultiDiscrete([3 3])` | ABSENT |
| seed2 lineage | `generalized_waypoint_both_seed2_0040960.zip` | `ef88f491d7a2b9bc…` | loaded | `…SplitSteeringNavigationPolicy` | `(928,)` | `MultiDiscrete([3 3])` | ABSENT |
| seed2 lineage | `generalized_waypoint_both_seed2_0051200.zip` | `87bd8d3e0be88b7f…` | loaded | `…SplitSteeringNavigationPolicy` | `(928,)` | `MultiDiscrete([3 3])` | ABSENT |
| seed2 lineage | `generalized_waypoint_both_seed2_0061440.zip` | `71007cd8f6d48d3e…` | loaded | `…SplitSteeringNavigationPolicy` | `(928,)` | `MultiDiscrete([3 3])` | ABSENT |
| split_branch_pilot | `split_branch_pilot.zip` | `b39a20a8bd06befc…` | loaded | `simulator.split_branch_policy.SplitSteeringEventPolicy` | `(923,)` | `MultiDiscrete([3 3])` | 2 |
| split_branch_pilot | `split_branch_pilot_5000.zip` | `4821a288afdb7eeb…` | loaded | `…SplitSteeringEventPolicy` | `(923,)` | `MultiDiscrete([3 3])` | 2 |
| split_branch_pilot | `split_branch_pilot_10000.zip` | `bbca9e6b89897d00…` | loaded | `…SplitSteeringEventPolicy` | `(923,)` | `MultiDiscrete([3 3])` | 2 |
| split_branch_pilot | `split_branch_pilot_15000.zip` | `9cee9563874853c7…` | loaded | `…SplitSteeringEventPolicy` | `(923,)` | `MultiDiscrete([3 3])` | 2 |
| split_branch_pilot | `split_branch_pilot_teacher.zip` | `b4742049725be4f9…` | loaded | `…SplitSteeringEventPolicy` | `(923,)` | `MultiDiscrete([3 3])` | 2 |
| era_928 | `generalized_waypoint_both_seed0_0010240.zip` | `7f652416c4d4f03f…` | loaded | `…SplitSteeringNavigationPolicy` | `(928,)` | `MultiDiscrete([3 3])` | ABSENT |
| bot model | `native_strategy_map_context_ppo.zip` | `06435efe0f309286…` | loaded | `stable_baselines3.common.policies.ActorCriticPolicy` | `(923,)` | `Discrete(5)` | 1 |
| bot model | `native_strategy_map_risk_ppo.zip` | `4e606ac59dd0dc6f…` | loaded | `stable_baselines3.common.policies.ActorCriticPolicy` | `(482,)` | `Discrete(5)` | 1 |
| canonical_advanced_ppo | `canonical_advanced_ppo_010k.zip` | `8b6521cab20b642f…` | **failed** | `ValueError` | — | — | ABSENT |
| quarantine | `canonical_basic_bootstrap_BROKEN_event_head_never_learned_eva_20260808.zip` | `d090da4f506655d8…` | **failed** | `ValueError` | — | — | ABSENT |
| era_925 | `canonical_basic_bootstrap_BROKEN_steering_and_event_collapse_20260807.zip` | `a584ff711283d3ec…` | **failed** | `ValueError` | — | — | ABSENT |

Exact exception, identical for all three failures:

```
ValueError: NavigationAugmentedFeaturesExtractor requires a 928-value observation
(923 raw + 5 navigation-history sidecar)
```

These are 925-era checkpoints predating the 5-value sidecar. **An expected failure is valid
evidence**; the exact type and message are frozen rather than repaired. The gate is *same
declared outcome + same class/space contract*, never "everything must load".

Every successful load resolved to exactly the `type(model.policy).__module__` /
`__qualname__` recorded in the checkpoint inventory (`matches_inventory = True` for all 14).
All loaded 928 navigation models carry the recorded `(928,)` observation and factorized
`[3,3]` action contract.

The two bot models were expected to *possibly* fail current contract validation; they in
fact loaded cleanly. `PPO.load` does not run farming contract validation, so this is
consistent, and their outcome is now frozen either way.

No model was modified, no training or live prediction occurred, and each load ran in its own
subprocess with exactly one repository root on `PYTHONPATH`. The repository `.venv`
interpreter was used.

**G10b: WITHDRAWN — BLOCKED_PENDING_AUTHORIZED_SELECTION.** The run completed with 0 internal gate failures, but the selection was not source-backed, so this does not constitute a valid frozen first baseline.

---

## 9. G11 — authoritative Tower map byte fingerprints

| path | pinned & actual SHA-256 |
|---|---|
| `flyff_farming_simulator/map_assets/occupancy.npy` | `62fa3c9ec3aed0b3b134b82577292c0a8a67b0acc4111fde3a36e3d2684d789b` |
| `foreground_vision_bot/mapper/maps/tower_aoe/occupancy.npy` | `62fa3c9ec3aed0b3b134b82577292c0a8a67b0acc4111fde3a36e3d2684d789b` |
| `flyff_farming_simulator/map_assets/map.json` | `faaf8633457bc1bcdb61c781c8ca62c6f2e008174ed5b284c3d6c08df92fe815` |
| `foreground_vision_bot/mapper/maps/tower_aoe/map.json` | `faaf8633457bc1bcdb61c781c8ca62c6f2e008174ed5b284c3d6c08df92fe815` |
| `flyff_farming_simulator/map_assets/coordinate_frame.json` | `40339f6c397d38fe01d5b3a5300e5b9b6d499f06292f436b1f91ea34523a0414` |
| `foreground_vision_bot/mapper/maps/tower_aoe/coordinate_frame.json` | `40339f6c397d38fe01d5b3a5300e5b9b6d499f06292f436b1f91ea34523a0414` |

Pairwise equality: `occupancy.npy` identical (1002129 B), `map.json` identical (4868 B),
`coordinate_frame.json` identical (106 B).

Marker `foreground_vision_bot/mapper/maps/tower_aoe/.skip_legacy_import`: **present**.

The two copies intentionally remain duplicated — no deduplication. No JSON was
loaded/resaved/reformatted; `occupancy.npy` was not rewritten. Git blob IDs are recorded in
the gate output as supplementary evidence only and never replace the raw-byte contract.

G12 was **not** performed. The intentional live `obstacle_radius_cells = 2` vs simulator
`obstacle_radius_cells = 0` difference remains frozen and was not reconciled.

**G11: GREEN — 0 failures.**

---

## 10. Phase-1 ruler after Phase 2

Formal check at `current_phase = 2`, exit **0**:

| rule | Phase-1 baseline | after Phase 2 |
|---|---|---|
| R6 | 7 | **7** |
| R7a | 35 | **35** |
| R7b | 0 | **0** |
| R7c | 200 | **200** |
| R9 | 0 | **0** |
| R10 failures | 0 | **0** (313 checkpoints / 317 references) |
| torch imported | no | **no** |
| bridge errors | 0 | **0** |
| ownership errors | 0 | **0** |

No new R6/R7 debt was created by Phase 2, and no debt baseline was regenerated to hide
anything — `BASELINE_VIOLATIONS.json` and `BASELINE_VIOLATIONS.md` are untouched since Phase 1.

- **B1/B2**: still `future`, `locations = []` — uninstalled, and still valid at Phase 2 (gate `PHASE_7`).
- **B3**: source evidence still matches.
- **B4**: tag still exactly `a90de59232b81753c1b2ea35b8990325c26674e5`.

D1 changed by exactly one row — the `map.json` duplicate pair's digest now records the
restored frozen bytes (`af65e773…` → `faaf8633…`). Counts unchanged: 119 exact / 31
AST-similar / 150 rows. D1 is diagnostic-only and never gates.

Tests: **31 passed** (17 Phase-1 migration-integrity + 14 Phase-2 fingerprint).

---

## 11. Proof no product source changed

`git diff --name-status dc734bb..HEAD -- '*.py'` returns only **new migration tooling**
under `docs/migration/`:

```
A  docs/migration/tests/test_migration_integrity.py
A  docs/migration/tests/test_phase2_fingerprints.py
A  docs/migration/tools/migration_integrity.py
A  docs/migration/tools/phase2_fingerprints.py
A  docs/migration/tools/phase2_representative_load.py
```

**Zero** Python source changed under `foreground_vision_bot/`, `flyff_farming_recorder/`, or
`flyff_farming_simulator/`. None of `farming/actions.py`, `farming/observation.py`,
`farming/model_contract.py`, `simulator/navigation_history.py`, `simulator/movement_kernel.py`,
router, telemetry, position readers, or recorder schemas was touched.

The only product-path entries in the whole Phase-2 range are the 12 portability-repair
artifacts, whose content is byte-identical to the frozen Phase-0 manifest values and
semantically identical to their pre-repair parsed content (section 2).

Checkpoint and model bytes: **unchanged** — the preserved corpus was read strictly read-only
and nothing was copied into the worktree. Map bytes: **restored to their frozen Phase-0
values**, never altered.

Phase-0 evidence files untouched since `dc734bb`: `CHECKPOINT_INVENTORY.tsv`,
`CHECKPOINT_MODULE_REFERENCES.tsv`, `ARTIFACT_MANIFEST.tsv`, `EFFECTIVE_CONFIG_BASELINE.json`.

Protected tags unchanged:

```
pre-consolidation-head                     -> 51dc25b2be0aafb091e22a17505767c1bec79552
historical-reproduction-baseline-20260815  -> a90de59232b81753c1b2ea35b8990325c26674e5
pre-consolidation-complete                 -> dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34
```

---

## 12. Worktree/index status

Clean worktree, empty index after the final documentation commit. Branch
`refactor/consolidation-phase1` remains **unpushed** (`git ls-remote --heads origin` returns
nothing for it). `git diff --check` clean outside the intentionally CRLF-restored artifacts,
where it reports one "trailing whitespace" per restored CR byte — that is the repair itself,
and excluding those 12 paths `diff --check` is completely clean.

---

## 13. Deviations and missing Phase-0 evidence

1. **Phase-0 artifact portability defect** (section 2) — 12 of 27 tracked manifest artifacts
   did not reproduce on a fresh checkout. Repaired by storing the frozen bytes with narrow
   `-text` rules. Not artifact drift.
2. **Phase-0 inventory lacked `policy_kwargs` / `net_arch`** — not backdated; frozen instead
   in a clearly labelled Phase-2 supplement (section 7).
3. **No Phase-0 real-load baseline existed** — not claimed; established transparently as the
   first Phase-2 load baseline (section 8).
4. **G10b categories 3-6 were ambiguous in the accepted plan** — resolved by an explicit
   deterministic pre-declared rule rather than cherry-picking, and flagged here for override
   (section 8).
5. **`PHYSICS_VERSION` has no such symbol in source** — resolved to
   `MOVEMENT_PHYSICS_MODEL_ID` and recorded explicitly rather than invented (section 6).
6. **R7c does not flag `from X import *`** — observation only; star imports never name a
   controlled symbol. Not a defect against R7c's stated contract.

---

## 14. Conclusion

**PHASE 3 SAFE TO CONSIDER: NO**

Phase 2 is **not** complete. Exit condition E (G10b) is FAIL/PENDING, blocked on an explicit
representative-selection decision that only the coordinator can authorize. Phase 3 was not
begun and is not authorized. The expensive Phase-3 behavior fixtures — 10k 923-vector parity, neighbour-count
boundary fuzz, bounded-geodesic equivalence, both derived-map loader dumps, router/kernel
golden sweep, effective-config golden baseline, archive decode baseline — are out of scope
here.


---

## 15. Exit-condition status after correction

| | condition | status |
|---|---|---|
| A | Claude independently accepted Phase 1 | PASS |
| B | `current_phase = 2` | PASS |
| C | G4 green | PASS |
| D | G10a green | PASS |
| E | **G10b green** | **FAIL / PENDING — blocked on authorized selection** |
| F | G11 green | PASS |
| G | Phase-1 ruler green (R6=7 R7a=35 R7b=0 R7c=200 R9=0 R10=0) | PASS |
| H | B1/B2 uninstalled | PASS |
| I | B4 tag unchanged | PASS |
| J | No product source changed | PASS |
| K | No checkpoint/model/map bytes changed | PASS |
| L | Worktree/index clean | PASS |
| M | Branch unpushed | PASS |

**Phase 2 is NOT complete.** Twelve of thirteen exit conditions hold; E blocks completion.
