# Canonical Basic: eval-pipeline hotfix + complete 6-round result review (2026-08-24)

Follow-up to the 2026-08-23 canonical Basic run
(`training_logs/canonical_basic_20260823_234310.log`/`.stderr.log`, 6
milestone checkpoints, `MISTAKES.md` 2026-08-24 "canonical Basic relaunch:
every round's async raw diagnostic silently crashed"). That run's training
result (checkpoints, assisted-mode milestone metrics, event/target BC) was
already valid; only the evaluation pipeline was broken. This task fixed the
pipeline and re-ran evaluation only -- no retraining, no new checkpoints,
no DAgger/BC updates.

## Bug 1 -- raw-diagnostic path resolution

**Root cause**: every curriculum-manifest `curriculum_path` field (and
`challenge_family_curriculum_path`/`FixedRegressionScenario.curriculum_path`)
is stored relative to `simulator/` itself, e.g.
`"curricula/synthetic_curriculum_heldout/curriculum.json"` resolves to
`simulator/curricula/synthetic_curriculum_heldout/curriculum.json`. Neither
`beginner_transition.zero_shot_raw_diagnostic[_parallel]` nor any of
`milestone_evaluator.py`'s heldout/challenge evaluators resolved this --
they passed the stored string straight into `SyntheticCurriculum.load`/
`run_composed_episode`/`run_episode`, which interpreted it relative to the
process's actual cwd. `RUN_CANONICAL_BASIC.py` launches its eval worker
with `cwd=str(ROOT)` (repo root), one directory above `simulator/`, so the
lookup landed on a path that does not exist. The curriculum itself was
never missing -- a prior `MISTAKES.md` entry's "directory does not exist
anywhere in the repo" claim was incorrect (see the 2026-08-24 correction
entry); the file has existed, unchanged, since the `runtime/`-extraction
consolidation commit `1ce4138`.

**Canonical path-resolution rule (fix)**:
`simulator.curriculum_manifests.resolve_manifest_curriculum_path()` --
resolves relative to `SIMULATOR_ROOT = Path(__file__).resolve().parent`
(the `simulator/` package directory), passes an already-absolute path
through unchanged. Applied at every call site in `beginner_transition.py`
and `milestone_evaluator.py` that turns a manifest curriculum_path into a
filesystem path (the latter is not yet exercised by any real run --
Beginner/Intermediate/Advanced have not started -- but carried the
identical latent bug).

Audited every manifest under `simulator/evaluations/manifests/`
(Basic diagnostics' `early_heldout.json` plus every Beginner/Intermediate/
Advanced heldout/challenge manifest): all share the same relative-to-
`simulator/` convention, and one shared resolver now covers all of them
(`tests/test_curriculum_manifests.py::
test_every_current_manifest_curriculum_path_resolves_to_an_existing_file`).

## Bug 2 -- alarm-ordering

**Prior behavior**: `_basic_round_eval_worker.py`'s `main()` ran the
assisted milestone evaluator, then the (informational-only) raw
diagnostic, then computed/logged alarms from the milestone report. A raw
diagnostic crash (Bug 1) meant execution never reached the alarm block --
silently, for all 6 rounds.

**Corrected behavior**: alarm calculation/logging now runs immediately
after the milestone evaluator, before the raw diagnostic. A raw
diagnostic failure can still happen but can no longer suppress the alarm
check.

## Bug 3 -- worker exit-status detection

**Prior behavior**: `RUN_CANONICAL_BASIC.py`'s Stage 5 waited on every
dispatched eval worker (`proc.wait()`) and then unconditionally printed
"All N dispatched evaluation worker(s) finished." regardless of exit code
-- exactly how 6 crashed workers were reported as a clean finish.

**Corrected behavior**: extracted to `collect_eval_worker_results()`,
which waits on every worker (never kills one early) and then raises
`RuntimeError` naming every round whose worker exited non-zero, before
any "finished successfully" message is printed.

## Tests

`tests/test_curriculum_manifests.py` (path resolution against the real
manifests) and the new `tests/test_basic_round_eval_worker.py` (alarm
survives a deliberate raw-diagnostic failure; `collect_eval_worker_results`
raises on a real subprocess exit code, not a mock). Focused suite run:
`test_curriculum_manifests.py`, `test_basic_round_eval_worker.py`,
`test_beginner_transition.py`, `test_basic_milestone_evaluator.py`,
`test_milestone_evaluator.py`, `test_milestone_evaluator_recovery.py`,
`test_scratchpad_tool_import_safety.py`, `test_basic_checkpoint_
provenance.py`, `test_basic_stage_frozen_navigation_integration.py`,
`test_basic_environment.py`, `test_basic_training_pipeline.py` -- 66
passed, 0 failed. `git diff --check`: PASS.

Hotfix branch `fix/basic-raw-diagnostic-eval-worker`, commit `33082dd`;
merged to `main` at `4b1953d`; pushed.

## Checkpoint integrity

SHA256 of all six `models/canonical_basic_milestone_00{1..6}.zip`
recorded before any change and re-verified identical after the hotfix,
merge, and the full re-evaluation run below -- no checkpoint was
regenerated or overwritten.

| round | SHA256 |
|---|---|
| 1 | `c997e6e23e8fc606669a7b92919db7e9ec8119051951e35c8d129da0ebcadc94` |
| 2 | `53a78ebeac7e4c1a94154219e91a0aa116f76cdc5b897ac39079862cf23c1de8` |
| 3 | `9250d16ebd56d032d8cdbddb113201005fde70ef419be2179de1a2863c2abd1c` |
| 4 | `4809028d8f585668838c345c0f4681ba3b2bf8a84546cdd0cd99fa643a3a4a73` |
| 5 | `6824cd90134f08ed9fbcf4ac2c4a4bb2d7db5454d75c433261199f4e60dedbc7` |
| 6 | `8cca022a269cf84492f2b8955e9a58729fac11321975656fabe4d10c066bdb5d` |

## Re-evaluation of the 6 existing checkpoints

Reran the real, post-hotfix `_basic_round_eval_worker.py` sequentially
against each existing checkpoint (same script, same argv contract, same
cwd as `RUN_CANONICAL_BASIC.py`'s own dispatch -- not a reimplementation).
All 6 rounds exited 0; all 6 wrote both
`canonical_basic_milestone_XXX_report.json` (assisted) and
`canonical_basic_milestone_XXX_raw_diagnostic.json` (raw). No DAgger data
collected, no BC update, no new checkpoints written -- these evaluation
JSONs were the only files this step produced (alongside the already-
tracked `canonical_basic_dagger_round*.npz`/target-teacher-dataset/
bootstrap-dataset files left over in the working tree from the original
2026-08-23 run, untouched by this task and out of scope for this review).

### Assisted-mode milestone metrics (recovery-assisted; canonical alarm thresholds: `intervention_ticks_fraction.median >= 0.85`, `dominant_layout_intervention_share >= 0.85`, `gave_up_episode_fraction >= 0.5`)

| round | intervention_count (median/max) | contacts_per_step (median/max) | dominant_layout_share | gave_up_fraction | canonical alarm |
|---|---|---|---|---|---|
| 1 | 0 / 2 | 0.0 / 0.1925 | 0.0 | 0.0 | none |
| 2 | 0 / 0 | 0.00125 / 0.86 | 0.0 | 0.0 | none |
| 3 | 0 / 1 | 0.00375 / 0.8425 | 0.0 | 0.0 | none |
| 4 | 0 / 4 | 0.00375 / 0.8425 | **1.0** | 0.0 | **"one layout accounts for 100.0% of interventions"** |
| 5 | 0 / 2 | 0.0 / 0.1975 | **1.0** | 0.0 | **"one layout accounts for 100.0% of interventions"** |
| 6 | 0 / 3 | 0.005 / 0.8675 | 0.6667 | 0.0 | none (below 0.85 threshold) |

Rounds 4 and 5's `dominant_layout_intervention_share = 1.0` alarms are
confirmed by the canonical worker's own logic (not manual eyeballing).
The absolute intervention counts behind that share are tiny (round 4:
max 4 across all seeds/layouts, median 0; round 5: max 2, median 0) --
the share-based threshold cannot distinguish "1 of 1 interventions in one
layout" from "50 of 50," so a single-digit intervention count concentrated
in one layout by chance is sufficient to trip it. This is a real property
of the alarm's design (share, not count, is thresholded), not evidence by
itself of a systemic collision/navigation regression -- `intervention_
ticks_fraction.median = 0.0` and `gave_up_episode_fraction = 0.0` in both
rounds show recovery essentially never fired at all.

### Raw (recovery-off, zero-shot) diagnostic -- informational only, never a Basic graduation gate

Fields actually emitted by `zero_shot_raw_diagnostic_parallel` per layout:
`n_episodes`, `physical_stagnation_episodes`, `mean_contacts_per_100_
distance`. No completion/KPH/navigation-progress field is emitted by this
diagnostic (Basic's zero-shot raw rollout measures physical stagnation and
collision-contact density only) -- none is reported below that the tool
does not actually produce. 8 layouts x 2 seeds = 16 episodes per round.

| round | stagnation episodes (of 16) | mean contacts/100 distance (avg across 8 layouts) | worst layout (contacts/100) |
|---|---|---|---|
| 1 | 6 (38%) | 94.7 | `02_early_wide_neck_high_typical` (321.5) |
| 2 | 7 (44%) | 38.4 | `06_early_wide_neck_high_typical` (107.6) |
| 3 | 4 (25%) | 27.6 | `06_early_wide_neck_high_typical` (107.7) |
| 4 | 6 (38%) | 39.0 | `01_early_open_field_typical_fast` (153.8) |
| 5 | 3 (19%) | 13.3 | `04_early_wide_neck_typical_bursty` (61.3) |
| 6 | 5 (31%) | 76.9 | `01_early_open_field_typical_fast` (266.4) |

Full per-round, per-layout breakdown is in each `canonical_basic_
milestone_XXX_raw_diagnostic.json`.

## Basic training trend (assisted-mode target/event accuracy from the
2026-08-23 run, unchanged/not rerun by this task)

- target accuracy: 0.625 -> 0.667 -> 0.632 -> 0.536 -> 0.538 -> 0.483
  (declining after round 2)
- event accuracy: 0.645 -> 0.707 -> 0.727 -> 0.678 -> 0.755 -> 0.750
  (event gate PASS all 6 rounds; generally rising/stable)
- recovery-assisted: median interventions = 0 and gave_up_fraction = 0
  every round (confirmed again in this re-evaluation)
- raw (zero-shot) collision/stagnation: noisy across rounds, no monotonic
  trend either direction -- round 5 is the cleanest (19% stagnation, 13.3
  mean contacts/100), round 2 the worst on stagnation (44%), round 1 the
  worst on contact density (94.7 mean, 321.5 worst-layout)
- alarms: rounds 4 and 5 only, both `dominant_layout_intervention_share`,
  both tiny-absolute-count concentration per above

## Checkpoint selection

**Recommended: checkpoint 006 (the final round, per Basic's own
contract)**. Reasoning:

- Basic has no automated zero-shot/zero-collision graduation gate (unlike
  Beginner) -- the raw diagnostic here is explicitly diagnostic evidence
  for the Basic-to-Beginner handoff decision, not a selection criterion
  with a defined threshold to select an "earlier-round winner" against.
- The event gate (the one hard, defined Basic contract requirement) PASSes
  at every round including 006, and event accuracy is at its second-
  highest at round 6 (0.750, only below round 5's 0.755).
- Round 6 carries no canonical alarm (0.6667 dominant-layout share, below
  the 0.85 threshold) -- rounds 4 and 5, the only alarmed rounds, are NOT
  candidates on alarm grounds alone even if a "pick an earlier round"
  policy existed.
- Round 6's raw diagnostic is not the cleanest (76.9 mean contacts/100,
  above rounds 2/3/4/5) but is not an outlier low either, and the raw
  diagnostic trend across all 6 rounds is noisy/non-monotonic rather than
  showing a rounds-4-6 regression -- there is no round that both (a)
  strictly dominates 006 on every metric class (target, event, recovery,
  raw) and (b) has a defined contractual reason to prefer it over the
  final round under Basic's own contract.
- The target-head accuracy decline (0.667 at round 2 down to 0.483 at
  round 6) is real and worth carrying into the Beginner-preparation
  conversation, but is not, by itself, a Basic-contract failure -- Basic's
  defined pass/fail gate is the event gate, which PASSES throughout, and
  recovery-assisted intervention/give-up rates (the other defined signal)
  are at their best possible value (0) at every round including 006.

## Answers to the review questions

- **Is the target-head decline concerning enough to prevent moving on?**
  No, not by itself against Basic's own defined contract (event gate +
  recovery-assisted intervention/give-up rate), but it is real and should
  be tracked explicitly once Beginner's own PPO training begins (Beginner
  continues checkpoint 006 directly, so it inherits whatever the decline
  reflects).
- **Do the raw diagnostics expose a navigation/collision failure hidden by
  assisted mode?** They expose that raw (recovery-off) rollouts have
  meaningfully more physical stagnation and collision-contact density than
  assisted mode ever shows (expected -- assisted mode's recovery
  controller is specifically what raw mode removes), but no round shows a
  raw-diagnostic collapse (e.g. near-100% stagnation, or an order-of-
  magnitude contact-density outlier) that assisted mode's near-zero
  intervention/give-up rates would have concealed. The gap between
  assisted (near-perfect) and raw (noisy, non-trivial stagnation/contacts)
  is itself exactly what a "Beginner starting-point diagnostic" is
  supposed to surface -- it is informational evidence of how much of
  Beginner's job (raw, unassisted competence) is not yet done, not a
  hidden Basic failure.
- **Are rounds 4/5 dominant-layout alarms substantive or just tiny-count
  concentration?** Confirmed tiny-count concentration (max 4 and max 2
  interventions respectively, both medians 0, both `intervention_ticks_
  fraction.median = 0.0` and `gave_up_episode_fraction = 0.0`) -- not
  evidence of a systemic collision/navigation regression.
- **Is checkpoint 006 still the correct Basic output to carry into
  Beginner?** Yes, per the Checkpoint selection reasoning above.
