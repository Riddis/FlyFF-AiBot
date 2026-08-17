# Phase 13 Report — Living Project Knowledge + Agent Operating Rules + Context Hygiene + Final Consolidation Cleanup

## 0. Authorization and scope

Authorized: "PHASE 13 — LIVING PROJECT KNOWLEDGE + AGENT OPERATING
RULES + CONTEXT HYGIENE + FINAL CONSOLIDATION CLEANUP," plus two
addenda received mid-execution: (1) the shutdown/overnight-autonomy
skill addendum, raising the initial skill cap from 3 to 5 and adding
`finish-current-task-and-shutdown`/`overnight-autonomous-work`; (2) the
clean-repository-snapshot addendum, raising the cap from 5 to 6 and
adding `prepare-clean-repo-snapshot`. Phase 14, live validation, and
deployment work are explicitly NOT authorized. Both addenda were
incorporated in full — nothing from the original authorization was
skipped because of the interruption.

## 1. Entry state (verified before any mutation)

- Branch: `refactor/consolidation-phase1` — exact.
- Entry HEAD: `da92d436bfad84a992214a5a368ddfc0c72230df`, subject "Phase
  12 P12-CORRECTION: make retained-shim retirement condition explicit"
  — exact match.
- Worktree clean, index empty, no upstream, branch absent from origin.
- `CANONICAL_OWNERS.toml`: `current_phase = 12`. Phase 12 accepted
  complete. G5/G5-P2 pending. Future deployment derivation profile:
  PASS. The 16 Phase-12-transitioned shims carry `removal_gate =
  "NEVER"` + `retirement_condition = "TEST_CONTRACT_RETIREMENT"`; the 3
  genuinely permanent shims remain untagged by that condition. R1b
  exception unchanged. No live/deployment derivative exists.
- Ruler: `ok: true`, `R6=0 R7a=0 R7b=0 R7c=204 R9=0 R10=0`.
- Protected tags exact (re-verified at entry and exit): `pre-
  consolidation-head` = `51dc25b2be0aafb091e22a17505767c1bec79552`,
  `historical-reproduction-baseline-20260815` =
  `a90de59232b81753c1b2ea35b8990325c26674e5`, `pre-consolidation-
  complete` = `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.

All entry conditions held; no STOP triggered at entry.

## 2. Final HEAD

Resolve with `git rev-parse HEAD` after the P13-DOC commit lands.

## 3. Phase-13 commits

| Commit | Subject |
|---|---|
| `0d1b1ad` | P13-A: current-state documentation system |
| `ef647ad` | P13-B: agent entrypoints, shared project rules, six project skills |
| `79efad9` | P13-C: project-knowledge check + clean-repo-snapshot tooling |
| `7aeb568` | P13-D: correct the stale Ruff per-file-ignore path (Phase-12 cleanup item A) |
| *(pending)* | P13-DOC: report, journals, `current_phase=13` |

No commit was reset, amended, rebased, or force-pushed. No `git add -A`/
`.`/`-u` was used — every commit staged explicit paths.

## 4. Documentation structure created

```
docs/
  README.md                              (cold-start index)
  KNOWN_DEBT.md
  GLOSSARY.md
  architecture/
    SYSTEM_OVERVIEW.md
    COMPONENT_OWNERSHIP.md
    DATA_AND_MODEL_CONTRACTS.md
    POSITION_AND_POINTER_RECOVERY.md
    RECORDING_TELEMETRY_AND_ARCHIVES.md
    MAPS_AND_COORDINATE_FRAMES.md
    NAVIGATION_AND_MOVEMENT.md
  validation/
    README.md
    G5_REAL_CLIENT_VALIDATION.md
    VALIDATION_TEMPLATE.md
  operations/
    DEVELOPMENT_WORKFLOWS.md
    TESTING_AND_GATES.md
  decisions/
    README.md
    0001..0006 (six ADRs)
  agent/
    PROJECT_RULES.md
```

Deviated from the suggested skeleton in two deliberate ways, both
"conceptual separation preserved, exact filenames not mechanical" per
the authorization's own allowance: `RUNTIME_AND_PROCESS_FLOWS.md` and
`DEPENDENCY_BOUNDARIES.md` were folded into `SYSTEM_OVERVIEW.md` +
`COMPONENT_OWNERSHIP.md` rather than kept as separate files (the
content is tightly coupled and splitting it further would have meant
cross-referencing the same few facts across more files, contrary to
"depth without duplication"); `ARTIFACT_AND_PROVENANCE_RULES.md` was
folded into `TESTING_AND_GATES.md`/`docs/agent/PROJECT_RULES.md`
section 6 rather than kept separate (the content is a short, stable
list better placed alongside the rules that enforce it). No file count
inflation — 22 documentation files total, each substantial.

## 5. Knowledge model (Section 3 of the authorization)

Maintained as six distinct roles, never merged into one document:

- **A. Current canonical documentation** — `docs/architecture/`.
- **B. `MISTAKES.md`** — `flyff_farming_simulator/MISTAKES.md`, untouched
  in structure/model, only its cross-referenced path corrected in
  `CLAUDE.md` (section 8 below).
- **C. Validation/experiment records** — `docs/validation/`.
- **D. Architectural decisions** — `docs/decisions/`.
- **E. Migration/historical evidence** — `docs/migration/`, explicitly
  NOT the primary current-architecture reference after this phase (see
  `docs/README.md`'s own statement to that effect).
- **F. Machine contracts** — `CANONICAL_OWNERS.toml`, tests, manifests,
  hashes, frozen artifacts — referenced from the docs above, not
  duplicated into prose.

## 6. Conversation knowledge made durable

Every example the authorization listed was checked against current
repository evidence and, where supported, written into a canonical
document rather than left in conversation:

| Fact | Where it now lives | Confidence |
|---|---|---|
| Full dev app is the canonical product | `SYSTEM_OVERVIEW.md` §1, ADR 0003 | VERIFIED_CONTRACT |
| Future deployment derived from canonical source, never forked | `SYSTEM_OVERVIEW.md` §1/§5, ADR 0001 | VERIFIED_CONTRACT |
| Standalone/live derivative intentionally deferred | ADR 0003, `KNOWN_DEBT.md` | VERIFIED_CONTRACT (every phase's own authorization text) |
| G5 intentionally pending | `validation/README.md`, `validation/G5_REAL_CLIENT_VALIDATION.md` | Status fact, not fabricated |
| G5-P2 conditional on a discrimination-policy change | `POSITION_AND_POINTER_RECOVERY.md` §4, `G5_REAL_CLIENT_VALIDATION.md` §2 | VERIFIED_CONTRACT (`position/policy.py` has exactly 2 current `PlayerDiscrimination` values; no change made) |
| Agents never execute live FlyFF tests | `PROJECT_RULES.md` §2, ADR 0004, both `CLAUDE.md`/`AGENTS.md` | Rule, restated absolutely |
| Live tests prepared by agents, executed by the user | `preparing-controlled-validation` skill | Rule |
| Phase numbers are not evidence of obsolescence | ADR 0005, `COMPONENT_OWNERSHIP.md` §3b | VERIFIED_CONTRACT (the P12-A2→P12-CORRECTION incident itself) |
| Compatibility retirement tied to actual semantic/evidence gates | `retirement_condition = "TEST_CONTRACT_RETIREMENT"` mechanism, ADR 0005 | VERIFIED_CONTRACT |
| Historical results reproduced from frozen commits/tags | `validation/README.md`, `PROJECT_RULES.md` §10 | VERIFIED_CONTRACT |
| Observation/measurement/inference/assumption/unresolved distinct | `validation/README.md` confidence scale, used throughout `architecture/` | Methodology, applied |
| Riddims vs. partial Poot/WFC recorder population | `RECORDING_TELEMETRY_AND_ARCHIVES.md` §6 | HISTORICAL_EVIDENCE, labeled |
| `respawn_candidate` is statistical, not identity | `RECORDING_TELEMETRY_AND_ARCHIVES.md` §7 | Methodological rule, stated |
| Actor frames are sequential, not simultaneous reads | `RECORDING_TELEMETRY_AND_ARCHIVES.md` §5 | HISTORICAL_EVIDENCE/ASSUMPTION, labeled |
| Dev-bot correctness before deployment slimming | ADR 0003, `SYSTEM_OVERVIEW.md` §7 | VERIFIED_CONTRACT |

Nothing in this table was silently promoted to `VERIFIED_CONTRACT`
without a source read this phase (`position/policy.py`, `position/
RecoveredNativeProfile.py`, `simulator/split_branch_policy.py`,
`navigation/*` docstrings, `CANONICAL_OWNERS.toml`, and the Phase 9–12
reports were all read directly — see each doc's own "Evidence /
Sources" section).

## 7. Depth without duplication

Detailed technical content (checkpoint ABI, pointer-recovery mechanics,
navigation/movement calibration history, map hashes) exists in exactly
one canonical location per topic; other docs cross-link rather than
re-explain. Example: `SYSTEM_OVERVIEW.md` links to `DATA_AND_MODEL_
CONTRACTS.md` for the checkpoint ABI rather than restating it;
`PROJECT_RULES.md` links to the same file rather than duplicating the
ABI-shim list.

## 8. `CLAUDE.md` / `MISTAKES.md` path correction

`CLAUDE.md` (repo root, tracked) said "MISTAKES.md (same directory as
this file)." The actual tracked `MISTAKES.md` lives at
`flyff_farming_simulator/MISTAKES.md` — a different directory. Both
files were touched by the same Phase-7 root-collapse commit
(`bfc5c6d`), which is where this became stale (CLAUDE.md stayed at
root; MISTAKES.md did not move with it, or vice versa — the collapse
commit's own history doesn't distinguish which). Corrected: `CLAUDE.md`
now states the real path explicitly and points to `docs/agent/
PROJECT_RULES.md` §8 for the full rule. The existing Cardinal Rule
model (skim before non-trivial work, log immediately, terse LLM-
retrieval-oriented entries, no rewriting old entries) was preserved,
not replaced.

## 9. Absolute live-execution rule

Stated in `docs/agent/PROJECT_RULES.md` §2, both `CLAUDE.md` and
`AGENTS.md`, ADR 0004, and restated inside `preparing-controlled-
validation` and `overnight-autonomous-work`. Covers attach, read-only
observation, native reader tests, pointer recovery, telemetry, recorder
collection, calibration, control/input, G5, G5-P2, live farming, live
training. **No agent live execution occurred during Phase 13** — every
check this phase was offline (source reads, static analysis, test
runs, ruler checks, a single throwaway local ZIP-packaging test-run,
deleted afterward).

## 10. Future G5 documentation

`docs/validation/G5_REAL_CLIENT_VALIDATION.md` created as a procedure/
status document. Status: **PENDING / NOT RUN**, and remains so — no
result was fabricated. Contains the accepted 5-point G5 contract, the
G5-P2 conditional trigger, and full template placeholder sections
(test date, commit, client build, PID/restart status, config/policy,
hypothesis, frozen criteria, commands, evidence locations, observations,
criterion-by-criterion result, discoveries, limitations, confidence,
assumptions confirmed/falsified, canonical-doc consequences, `MISTAKES.
md` consequence, follow-up) via `docs/validation/VALIDATION_TEMPLATE.md`.

## 11. Documentation-maintenance rule

Stated in `PROJECT_RULES.md` §9: the trigger is "did the project's
knowledge change," not "was this a large code change." Examples from
the authorization (native-timing discoveries, pointer-field semantics,
G5 outcomes, unused config, checkpoint constraints, historical-evidence
confidence revisions, a dead-looking file turning out load-bearing, a
phase gate encoding a wrong retirement assumption) are all included
verbatim or paraphrased as worked examples.

## 12. Context hygiene

`PROJECT_RULES.md` §11 states the core principle (repository docs are
durable memory, conversation is temporary working memory), the
compaction/clearing checkpoints, the cold-start progressive-loading
order, and the "if this conversation vanished, could a fresh agent
recover state from the repository" test. `docs/decisions/0006-repo-
docs-are-durable-memory.md` records the rationale independently.

## 13. Shared Claude/Codex project rules

`docs/agent/PROJECT_RULES.md` is the one canonical rules document (14
sections: product direction, live-execution prohibition, canonical
ownership, test/gate discipline, scientific integrity, immutable
artifacts, git discipline, `MISTAKES.md`, documentation-maintenance,
forward-correction, context hygiene, STOP conditions, the two
operating-mode skills' boundaries, the snapshot skill). It does not
duplicate the full architecture manual — it points to `docs/README.md`.

## 14. `CLAUDE.md` / `AGENTS.md`

Both reviewed/created per the authorization's own instructions: no
guessed import syntax was used (Claude Code's repository-local skill
discovery convention, `.claude/skills/<name>/SKILL.md`, is used with
confidence since it is a real, current, documented Claude Code
convention — not invented for this repository; no existing skill
directory existed to conflict with it). `AGENTS.md` explicitly states
this repository has no separate first-class Codex skill-discovery
mechanism and points Codex at the same `.claude/skills/` files as
ordinary procedural documentation, rather than inventing an unsupported
Codex-specific location. Neither file duplicates the shared rules —
both point to `docs/agent/PROJECT_RULES.md`.

## 15. Project skills — final set of six

Per both addenda, the cap was raised twice during this phase (3→5→6)
and both raises were incorporated in full:

1. `maintaining-project-knowledge`
2. `preparing-controlled-validation`
3. `making-safe-repository-changes`
4. `finish-current-task-and-shutdown`
5. `overnight-autonomous-work`
6. `prepare-clean-repo-snapshot`

No further skill was added. Skill discovery: `.claude/skills/<name>/
SKILL.md`, verified against no pre-existing skill directory in this
repository.

### 15a. `finish-current-task-and-shutdown`

Location: `.claude/skills/finish-current-task-and-shutdown/SKILL.md`.
User-invoked only (explicit natural-language phrasing sufficient, no
exact-name requirement). An explicit shutdown request is itself
sufficient authorization — no second confirmation asked. Shutdown
ordering: repository durable first, shutdown is the final machine-
affecting action; `SHUTDOWN: FAILED — <reason>` is the required output
if the OS shutdown mechanism itself is blocked, never a false claim of
success. Not activated during Phase 13.

### 15b. `overnight-autonomous-work`

Location: `.claude/skills/overnight-autonomous-work/SKILL.md`.
User-invoked only. States the standing-autonomy scope (section A5 of
the addendum, reproduced in full) and the exact hard blockers it never
overrides (live execution, artifact immutability, git safety —
no push/force-push/history rewrite, no gate weakening). Overnight log
location/format: `docs/agent/overnight/<local-date-time>.md`
(directory not yet created — no overnight run has occurred; the skill
itself specifies the convention). Task-selection policy: the 7-item
priority order from the addendum, verbatim. Context-compaction
behavior overnight: compact at safe checkpoints if the client supports
it; never stop merely to wake the user for `/compact` if it doesn't.
Morning-report behavior: read the log + git state before answering,
required content list, and the confidence-reporting requirement
(`HIGH`/`MEDIUM`/`LOW` + reason, never describing an offline result as
live-validated). Project-complete stop condition: the offline
dev-bot-readiness definition from addendum section A10, reproduced
verbatim — reaching it does not authorize live training. **Not
activated during Phase 13** — no overnight log directory exists, and
none should, since the mode was never invoked.

### 15c. `prepare-clean-repo-snapshot`

Location: `.claude/skills/prepare-clean-repo-snapshot/SKILL.md`. Tool:
`tools/create_clean_repo_snapshot.py`. `REVIEW_CLEAN` is the (only)
default profile. Represents the **current worktree** via `git ls-files
--cached --others --exclude-standard`, not `git archive HEAD` — a dirty
worktree is snapshotted with its dirty state recorded, not refused.
Default exclusions: `.git/`, virtualenvs, all Python/tool caches
(including nested `__pycache__`/etc. at any depth — a real bug found
and fixed by the accompanying test, see P13-C's commit message),
databases, build output, and bulky ML/recording artifacts under
`models/`/`recordings/`/`evaluations/` by name-pattern, while leaving
small manifests/hashes/JSON summaries untouched. Secret/personal-data
protection: a fixed denylist of sensitive filename patterns; if a
tracked candidate matches, snapshot creation **refuses** rather than
silently omitting or redacting the source file. Metadata format:
`_REPO_SNAPSHOT_INFO.json` + `_REPO_SNAPSHOT_FILES.txt` (+ a
worktree-diff-stat when dirty). Validation: archive opens, non-empty,
metadata present, no `.git`/venv/cache/db content, no path escaping, no
sensitive file, reported via a checklist before success is declared —
packaging validation only, never the product test suite. Output
location: `exports/repo_snapshots/`, now gitignored (confirmed: `exports/`
added to `.gitignore` this phase, P13-C). Generated ZIPs are never
committed. Confirmed via one manual test-run (1702 files, 61.5 MB
compressed, verified `.git`/`.venv`/`__pycache__`/checkpoint-free),
then the test artifact was deleted — no large ZIP was left in the
repository or its exports as proof.

**Final total number of initial project skills: 6.**

## 16. Cleanup items from Phase 12 (Section 26)

**A. `pyproject.toml`'s stale Ruff ignore.** Resolved (commit `7aeb568`).
`foreground_vision_bot/mapper/[A-Z]*.py` never matched anything
(`foreground_vision_bot/mapper/` is entirely untracked — `git ls-files`
confirms zero files). Corrected to `mapper/[A-Z]*.py`, the canonical
current mapper package, which uses the identical PascalCase
module-naming convention this ignore exists to suppress (`ruff check
mapper/ --select N999` confirmed real, currently-suppressed findings
there). Single-directory-level glob semantics preserved exactly as the
original entry had — not expanded to `mapper/rl/` or other nested
PascalCase files, which the original entry never covered either. Zero
test/behavioral impact confirmed: `ruff` is not wired into any test or
CI gate in this repository.

**B. `flyff_farming_recorder/requirements.txt`** and **C.
`foreground_vision_bot/foreground_vision_farm.json`.** Re-audited, not
deleted, not silently resolved. Both still carry the frozen Phase-7
`PHASE7_MOVE_MANIFEST.tsv` `resolution_phase = PHASE_11` deferred-
collision label that Phase 11 never actually addressed (it explicitly
declined to build any standalone/live bot or decide build ownership).
Resolving either requires an actual product/deployment decision — which
this phase's own authorization instructs not to invent. Explicitly
re-deferred, now recorded in `docs/KNOWN_DEBT.md` (current-state
documentation, not only the Phase-12 forensic audit). No file content
changed for either.

## 17. Knowledge-integrity check

`tools/check_project_knowledge.py` (`python -m tools.check_project_
knowledge`). One consolidated gate, 9 checks: documentation index
reachability, referenced-current-path existence, canonical-owner
agreement with `CANONICAL_OWNERS.toml`, runtime-ABI path existence,
G5/G5-P2 status coherence, agent-rule linkage, skill metadata/discovery,
internal current-doc link resolution, no unqualified stale old-root
paths. Result on the final repository state (after this report exists):
**PASS** (verified in section 20 below). Does not reproduce product
behavior tests — pickle/router/archive/dependency/checkpoint contracts
remain owned by their own existing tests, unchanged.

`tests/test_check_project_knowledge.py` (4 focused tests) and `tests/
test_create_clean_repo_snapshot.py` (5 focused tests) — not one test
per documentation file. Both caught real bugs during development (see
P13-C's commit message for the two specific fixes), confirming the
checks actually detect what they claim to detect rather than being
tautologically green.

## 18. No live execution / no training

No FlyFF launch, no attach, no live telemetry, no live recording, no
live native diagnostics, no G5, no G5-P2, no client control, no live
training. No `PPO.learn()` call, no 820M run, no new scientific
evaluation result produced for documentation purposes.

## 19. Scientific artifact immutability

No checkpoint, recording, evaluation JSON, calibration CSV, Tower map
byte, Phase-3/6 fixture, historical reproduction snapshot, frozen
baseline, protected tag, or B4 evidence was modified or regenerated.
`git status --short` at every checkpoint this phase showed only the
intended new/modified documentation, rules, tooling, and the one
`pyproject.toml` config line.

## 20. Verification

```
migration_integrity.py check -> ok: true, R6=0 R7a=0 R7b=0 R7c=204
  (unchanged) R9=0 R10=0 (313 checkpoints, 317 module references, zero
  failures)
python -m future_runtime_profile.derive_runtime_manifest -> PASS
  (89 candidate modules, 0 forbidden edges, 0 missing files, 0
  duplicate-ownership issues, 1 exception applied -- unaffected by any
  Phase-13 change)
pytest docs/migration/tests/ -> 77 passed, 0 failed (unchanged)
pytest tests/test_check_project_knowledge.py
  tests/test_create_clean_repo_snapshot.py -> 9 passed
python -m tools.check_project_knowledge -> PROJECT KNOWLEDGE: PASS
  (all 9 checks green once this report exists to satisfy its own
  forward reference from docs/KNOWN_DEBT.md)
git diff --check -> clean
```

**The full ~9-minute `tests/` product suite was intentionally not
re-run this phase** — per the authorization's own instruction, it is
warranted only when a change touches product/runtime behavior or import
structure. Phase 13's only non-documentation/non-tooling change is the
single-line `pyproject.toml` Ruff config correction, confirmed to have
zero test/CI wiring impact. The accepted Phase-12 baseline (1190
passed, 4 established pre-existing/unrelated failures, 2 skipped, 1
xfailed) stands unchanged and unverified-by-rerun this phase, by
design.

## 21. Protected refs / worktree / branch status

`pre-consolidation-head`, `historical-reproduction-baseline-20260815`,
`pre-consolidation-complete` all confirmed unchanged (section 1 values,
re-verified at report time). Worktree clean, index empty. Branch
`refactor/consolidation-phase1`, no upstream, absent from `origin` — no
push performed.

## 22. G5/G5-P2 status

**G5: NOT RUN / PENDING. G5-P2: NOT RUN / PENDING (conditional — no
discrimination-policy change has been made).**

## 23. `AGENT LIVE EXECUTION: NONE`

Confirmed for the whole of Phase 13.

## 24. Documentation impact of this phase

This entire phase *is* a documentation-impact phase by design —
sections 4–17 above are the full account. `docs/migration/` itself was
not rewritten (only this new report was added, per the established
forward-only convention).

## 25. Final conclusion

**PHASE 13 COMPLETE: YES.**
**PHASE 14 SAFE TO CONSIDER: YES** (readiness only — the durable
documentation/rules/tooling foundation this phase built is a
prerequisite for lower-context-cost future work, not a decision about
what Phase 14 should contain).
**PHASE 14 AUTHORIZED: NO.**
