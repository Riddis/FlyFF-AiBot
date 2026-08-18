# Phase 14 — Final Product Analysis

Companion narrative to `docs/migration/PHASE14_CAPABILITY_AUDIT.tsv`
(73 rows, capability-by-capability evidence). This document answers the
Phase-14 primary question in prose; the TSV is the underlying evidence
table.

## 1. Primary question

*"After Phases 0–13, is this repository now one coherent canonical
development product whose required pre-consolidation capabilities,
scientific contracts, artifacts, workflows, and compatibility surfaces
are all accounted for and whose offline verification is strong enough
to declare the MIGRATION complete?"*

**Answer, with evidence: YES**, subject to the exact scope stated
throughout (migration/consolidation completeness — not overall project
completeness, not live validation, not deployment readiness). See
`docs/migration/codex_handoff/PHASE14_REPORT.md` section 34 for the
formal conclusion and sections 15–27 for the offline verification
evidence this conclusion rests on.

## 2. Capability-conservation result

All 73 audited capabilities (categories A–J, per the authorization's
own list) trace to a current, positively-identified disposition — none
show "no current owner / no intentional-retirement decision / no
compatibility replacement / no documented future-only disposition" (the
Section-9 stop-and-fix trigger). Summary by disposition:

| Disposition | Count | Meaning |
|---|---|---|
| `CURRENT_CANONICAL` (incl. dual/mixed rows) | ~55 | Actively used by the current dev app, recorder, or simulator CLI |
| `TRAINING_ONLY` / `SIMULATOR_ONLY` | ~6 | Real, present, correctly excluded from the dev-app/future-runtime closure |
| `DEV_TOOL` | ~4 | Phase-10 devtools surface, correctly dev-only |
| `ABI_COMPATIBILITY` | ~3 | Checkpoint/pickle identity shims, permanent by construction |
| `HISTORICAL_COMPATIBILITY` | ~7 | Frozen evidence (B4, checkpoint corpus, archive provenance) |
| `USER_RUN_LIVE_VALIDATION_PENDING` | 3 (overlapping with `CURRENT_CANONICAL`) | G5-gated, correctly still pending |

No capability was found silently dropped. Every `TRAINING_ONLY`/
`HISTORICAL_COMPATIBILITY`/`ABI_COMPATIBILITY` classification traces to
a specific prior-phase decision with cited evidence (Phase 7 collapse,
Phase 9 navigation refactor, Phase 10 devtools creation, Phase 11
dependency-boundary audit, Phase 12 shim-retirement gating).

## 3. Legacy-root residue — final disposition

At Phase-14 entry, 39 tracked files remained under the three original
sibling roots. All now have a final, non-ambiguous disposition — zero
`AMBIGUOUS_BLOCKER` remains:

| Root | Entry count | Final count | Disposition |
|---|---|---|---|
| `foreground_vision_bot/` | 10 | 9 | 9 `COMPATIBILITY_REQUIRED` B1 shims (`farming/*.py`) — `TEST_CONTRACT_RETIREMENT`-conditioned, `test_phase4_contracts.py`-gated. 1 file (`foreground_vision_farm.json`) proven `SAFE_TO_DELETE` and removed (section 4 below). |
| `flyff_farming_recorder/` | 26 | 25 | 25 `COMPATIBILITY_REQUIRED` B2 shims + resources (23 `position/*.py` + 2 `.json`) — `test_phase5_contracts.py`-gated. 1 file (`requirements.txt`) proven `MOVE_TO_CANONICAL_OWNER` (its content merged into root `requirements.txt`) and removed (section 4 below). |
| `flyff_farming_simulator/` | 3 | 1 | `MISTAKES.md` classified `PROJECT_WIDE_FILE_IN_OLD_LOCATION`, moved to repo root (section 5). `requirements.txt` proven `SAFE_TO_DELETE` (fully redundant, section 4). `README.md` classified `ARCHIVE_AS_HISTORY` — retained in place with a `SUPERSEDED` banner (section 6), real historical narrative value preserved. |

Total: 39 → 35 tracked legacy-root files, all with cited evidence.
Nothing was deleted without positive proof; nothing ambiguous was
silently resolved.

## 4. The two deferred collision files — final resolution

Both files Phase 12/13 repeatedly, correctly deferred (because prior
audits could not truthfully resolve them) now have final dispositions,
found by tracing an actual mechanism rather than re-guessing:

### 4a. `flyff_farming_recorder/requirements.txt` → `MOVE_TO_CANONICAL_OWNER`

Content: `msgpack>=1.0`, `pywin32>=306`. Tracing revealed this was a
**genuine migration gap**, not a stylistic duplicate: `msgpack` (a
confirmed `DUAL_ROLE` dependency of `simulator/schema.py`/`recorder/*`,
per `PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md`) was **never merged**
into the canonical root `requirements.txt` during the Phase-7 collapse
— a fresh environment following `pip install -r requirements.txt` would
be missing it. `pywin32` was already present at root (a bare version
comment, no floor; the installed version, 312, already exceeds the
dropped `>=306` floor — no behavioral loss). **Fix**: added
`msgpack # 1.2.1` to root `requirements.txt`; removed the now-fully-
redundant file (`git rm`, history preserved via the commit, not a
rename since the destination already existed with unrelated content).

### 4b. `foreground_vision_bot/foreground_vision_farm.json` → `DELETE_PROVEN_OBSOLETE`

Tracing `Gui.py`'s `sg.user_settings_filename(path=".")` call into
PySimpleGUI's own source (`UserSettings._compute_filename`) proved the
settings filename is derived from `sys.modules["__main__"].__file__`'s
basename — under the current entrypoint (`apps/dev_app.py`) that
produces `dev_app.json`, never `foreground_vision_farm.json`. Both
existing copies (this one and the root-level orphan Phase 11 already
flagged) are PySimpleGUI auto-generated settings snapshots keyed to the
**pre-Phase-7 entrypoint's file name** — proven orphaned by mechanism,
not merely by "zero current references." **Fix**: removed this copy
(`git rm`). The root-level copy was left untouched (outside this
specific instruction's named scope) but its `unresolved_future_choices`
entry was removed since the disposition question is now answered; see
`docs/architecture/SYSTEM_OVERVIEW.md` section 3a for the full
mechanism writeup (a genuinely new, previously-undocumented fact this
audit surfaced) and `docs/KNOWN_DEBT.md`'s "Known runtime limitations"
for the resulting CWD-dependent-settings limitation this leaves behind
(intentionally not fixed — a real fix requires a product decision about
where settings should canonically live, which Section 29 of the
authorization forbids inventing here).

Neither resolution required a user product decision — both were
positively provable offline.

## 5. `MISTAKES.md` relocation

Traced every reference: zero programmatic path dependencies anywhere
(`git grep` for `open`/`Path`/`read_text` against `MISTAKES.md` returns
nothing — it is exclusively a human/agent-read reference file). All
current-doc prose references (7 files: `CLAUDE.md`, `AGENTS.md`,
`docs/README.md`, `docs/agent/PROJECT_RULES.md`, two architecture docs,
`tools/check_project_knowledge.py`) were mechanically updated. Moved via
`git mv flyff_farming_simulator/MISTAKES.md MISTAKES.md` (history
preserved as a rename, confirmed via `git status`). Content byte-
identical — not rewritten. See section 9 of `docs/agent/PROJECT_RULES.md`
(pre-existing) plus `docs/decisions/` for why this is the semantically
correct final location: the file is project-wide (categories span
navigation, position, PPO internals, environment/tooling, process/
meta — never scoped to just the simulator), and `CLAUDE.md`/`AGENTS.md`
(the actual project-wide entrypoints) already lived at repo root, not
under `flyff_farming_simulator/`.

## 6. Stale-current-doc audit

`docs/{ARCHITECTURE,CONFIGURATION,RUNBOOK,POINTER_RECOVERY_REFERENCE}.md`
and `flyff_farming_simulator/README.md` were confirmed, on direct
inspection, to lack any self-identifying "this is superseded" marker —
`docs/README.md`'s Phase-13 note pointing *at* them was not sufficient,
since a reader opening one of these files directly (not via the index)
would see no warning. Added a short, non-rewriting banner to the top of
each five files, stating what is stale, what superseded it, and (where
applicable) what remains substantively accurate and why. No content
below the banner was altered — this preserves the historical evidence
exactly as Section 14 requires, while closing the "could mislead a cold
reader" risk.

## 7. New knowledge discovered and made durable this phase

Per the authorization's Section 28 documentation-completeness test:

- **GUI settings persistence is CWD-dependent by construction**
  (section 4b above) — a genuinely new, previously-undocumented fact
  about current `Gui.py` behavior, now in
  `docs/architecture/SYSTEM_OVERVIEW.md` section 3a and
  `docs/KNOWN_DEBT.md`.
- **`msgpack` was missing from the canonical dependency declaration** —
  a real migration gap, now fixed and documented (section 4a above).
- Both facts would otherwise have existed only in this audit's
  reasoning — now persisted to the repository per the durable-memory
  principle (`docs/decisions/0006-repo-docs-are-durable-memory.md`).

No other important architectural decision, limitation, scientific
assumption, workflow, ownership fact, or validation state was found to
exist only in conversation history during this audit — the current
documentation system (built in Phase 13) was usable as the primary
source for this entire audit, with `docs/migration/` consulted only to
prove specific historical/provenance claims (exactly the intended usage
pattern per Section 3 of the authorization).

## 8. Conclusion

Migration/consolidation is complete: one coherent canonical development
product, zero unexplained capability loss, zero unresolved legacy-root
residue, two long-deferred collision files finally and correctly
resolved (one revealing and fixing a real dependency-declaration gap),
`MISTAKES.md` relocated to its semantically correct project-wide
location, and every previously-ambiguous old document now honestly
labeled. See `docs/validation/FINAL_OFFLINE_MIGRATION_ACCEPTANCE.md`
for the full offline verification evidence and
`docs/migration/codex_handoff/PHASE14_REPORT.md` for the complete
phase account.
