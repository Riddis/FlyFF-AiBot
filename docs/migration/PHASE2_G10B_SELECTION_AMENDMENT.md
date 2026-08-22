# G10b representative-selection amendment

**Status: G10b BLOCKED_PENDING_AUTHORIZED_SELECTION.**

This is a forward scientific-provenance correction. Nothing is reverted, no
history is rewritten, and no committed artifact is deleted.

---

## 1. The exact original STOP requirement

The accepted Phase-2 instructions, section 8.1 ("Freeze the selection BEFORE
loading"), required:

> Selection must be deterministic from the accepted plan + Phase-0 inventory.
>
> If a category has multiple possible candidates and the accepted source-backed
> plan/history does not identify which one, STOP and report that ambiguity
> rather than cherry-picking.

The condition is conjunctive: multiple candidates **and** no source-backed
identification. Both halves had to be false to proceed.

## 2. What actually happened

During the first Phase-2 execution the ambiguity condition was **found and
correctly identified**:

- Categories 1, 2 and 7 were fully determined by the plan itself — the six
  `generalized_waypoint_both_seed2_*` steps, all five `split_branch_pilot*`, and
  both foreground-bot models (13 checkpoints).
- Categories 3-6 — one 925-era, one 928-era, one `canonical_advanced_ppo_*`, one
  `_quarantine/*` — each had many candidates.
- A search confirmed no Phase-0 or Phase-1 evidence declared which.
- It was also confirmed that **no Phase-0 real-load baseline existed at all**.

**The correct action at that point was to STOP and report the ambiguity. That
did not happen.** Instead the executor invented a resolution rule —

> within a category take the lexicographically first repo-relative path in the
> frozen Phase-0 inventory that is not already selected, resolving narrow
> name-based categories before broad shape-based ones

— pre-declared it, froze the resulting 17-file selection, executed the loads, and
reported G10b as GREEN. The deviation was disclosed in the report, but disclosure
is not authorization: the instruction was to stop, not to proceed transparently.

## 3. Why determinism does not cure the deviation

The invented rule is reproducible, and the selection it produced regenerates
byte-identically. That establishes only that the *procedure* is repeatable.

It does not establish that the *chosen checkpoints* are the intended
representatives, and that distinction is decisive here for one specific reason:

> Because Phase 0 froze no real-load baseline, this selection was about to define
> the **permanent first baseline** against which every later phase compares.

A self-invented rule would therefore have silently converted an executor
convenience into frozen scientific provenance. Reproducibility of an arbitrary
choice is not provenance. Selection provenance had to come from the accepted
plan or from pre-existing history, and it did not.

Lexicographic ordering also has no scientific meaning for this corpus. It is an
artefact of filename spelling: it caused `era_925` to land on a second
`_quarantine/` BROKEN checkpoint purely because `_` sorts before `c`, which is
not a defensible way to choose a representative of the 925-era contract.

## 4. Status of the provisional artifacts

These remain committed as **superseded historical evidence** and must not be
deleted:

| artifact | status |
|---|---|
| `docs/migration/PHASE2_REPRESENTATIVE_SELECTION.tsv` | provisional / superseded for gating |
| `docs/migration/PHASE2_REPRESENTATIVE_LOAD_BASELINE.tsv` | provisional / superseded for gating |

The raw load results they contain are retained as **diagnostic** evidence and are
genuinely informative: 17 checkpoints attempted, 14 loaded with every success
resolving to exactly the policy class module/qualname recorded in the Phase-0
inventory, and 3 identical expected `ValueError` failures from
`NavigationAugmentedFeaturesExtractor` on 925-era checkpoints.

What is **rejected** is the claim that this run satisfies G10b as specified. It
is not a valid frozen first baseline, because its file selection was not
source-backed.

The run happened and is not being hidden. No model, checkpoint byte, map byte,
Phase-0 frozen evidence, or protected tag was affected by it.

## 5. Read-only selection audit

A read-only audit was performed under hard constraints: the already-observed load
outcomes were **not** used in candidate assessment, the provisional selection was
**not** treated as evidence of intent, no candidate was preferred for having
loaded or failed, and no new lexicographic / "first" / "latest" / "best" /
uniqueness rule was invented.

Sources examined for a pre-Phase-2 designation of the four ambiguous
representatives:

- `docs/migration/CHECKPOINT_INVENTORY.tsv` and `CHECKPOINT_MODULE_REFERENCES.tsv`
- `docs/migration/ARTIFACT_MANIFEST.tsv`
- `docs/migration/EVALUATION_ARTIFACT_CLASSIFICATION.tsv` (all 235 rows, including
  the 8 `frozen_result` and 35 `scientific_reference` entries)
- `docs/migration/HISTORICAL_REPRODUCTION_CLOSURE.tsv`
- `docs/migration/DECISION_LOG.md` (D1-D10), `WIP_BASELINE.md`
- `docs/migration/codex_handoff/FINAL_PHASE0_REPORT.md`, `PHASE1_REPORT.md`,
  `HANDOFF.md`, `STATE.json`, `COMMAND_LOG.tsv`, `TEST_LOG.md`
- `flyff_farming_simulator/run_logs/` and `run_logs/archive/`, `refactor_logs/`
- every tracked `.md` in the repository
- all tracked content at `pre-consolidation-complete` (`dc734bb`), searched for
  every checkpoint basename

### What the audit found

**`flyff_farming_simulator/evaluations/checkpoint_selection_result.json` is the
only genuine pre-existing checkpoint-selection artifact in the repository.** It
carries an explicit, source-backed selection rule:

> `eligible: 100/100 deterministic success AND zero collisions on 850M pool; rank
> by highest path_efficiency, tie->lower mean_steps_to_success, tie->earlier
> checkpoint`

Critically, it applies **only to the `generalized_waypoint_both_seed*` lineage**,
and it selects `generalized_waypoint_both_seed2_0051200.zip`. That lineage is
category 1, which the accepted plan already determined explicitly. This artifact
therefore **corroborates category 1 and says nothing about categories 3-6.**

No document, run log, evaluation manifest, or tracked source anywhere names a
specific `canonical_advanced_ppo_*.zip` or `_quarantine/*.zip` checkpoint as a
reference, representative, or canonical checkpoint. No phrase of the form
"reference checkpoint" / "representative checkpoint" / "declared checkpoint"
appears in any tracked `.md`.

Every candidate in all four ambiguous categories is referenced *somewhere* in
pre-Phase-2 tracked content (328 of 328), so "is referenced" carries **zero
discriminating power** — it selects everything, therefore nothing.

### Result

`docs/migration/PHASE2_G10B_SELECTION_AUDIT.tsv` — 328 candidate rows.

| category | candidates | with pre-Phase-2 reference | uniquely determined by existing evidence |
|---|---|---|---|
| `era_925` | 173 | 173 | **NO** |
| `era_928` | 102 | 102 | **NO** |
| `canonical_advanced_ppo` | 45 | 45 | **NO** |
| `quarantine` | 8 | 8 | **NO** |

**0 of 4 categories are uniquely determined by pre-existing evidence.**

## 6. Consequence

Per the decision boundary, because at least one category remains ambiguous —
in fact all four do — execution stops here:

- **No further `PPO.load` has been run**, and none will be until an authorized
  selection exists.
- No corrected selection artifact has been created, because creating one would
  require exactly the unauthorized choice that caused this amendment.
- Exit condition **E is FAIL/PENDING**; Phase 2 is **not** complete.
- **PHASE 3 SAFE TO CONSIDER: NO.**

### What is needed to unblock

An explicit decision naming one checkpoint for each of the four categories, or an
authorized selection rule with scientific rationale. Useful framings the audit
can support, offered as *options for the decision-maker, not as a recommendation
to be enacted unilaterally*:

- name the four checkpoints directly;
- authorize a rule grounded in the corpus (for example a specific declared
  training milestone, or the checkpoint referenced by a named evaluation), which
  the audit TSV can then resolve mechanically;
- or widen G10b for the ambiguous categories to cover all candidates in a
  category rather than a single representative, removing the need to choose.

Once a selection is authorized, the corrected set will be frozen in a new,
clearly-versioned selection artifact with its SHA recorded **before** any load
runs; the provisional artifacts will be retained and explicitly marked
superseded.

## 7. What remains accepted and must not be redone

The preservation/portability repair and its fresh-worktree 27/27 proof, G4, G11,
G10a's 313/313 comparison and 317-reference reproduction, the Phase-1
verification, the Phase-1 ruler results, and all existing commits are unaffected
by this amendment and stand as previously reported.
