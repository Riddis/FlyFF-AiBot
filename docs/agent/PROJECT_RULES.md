# Project Rules (Claude & Codex)

The permanent, shared repository rules. Concise by design — for the
knowledge map, see [`docs/README.md`](../README.md); this file states
rules, not architecture.

## 1. Product direction

- The canonical product is the **full development application**
  (`apps/dev_app.py` and everything it drives). See
  [ADR 0003](../decisions/0003-dev-bot-first.md).
- A future deployment/live derivative is derived from this **same**
  canonical source tree — never a copied fork, never built speculatively
  ahead of dev-bot readiness. See
  [ADR 0001](../decisions/0001-canonical-source-single-tree.md).

## 2. Absolute live-execution prohibition

**No agent may ever execute a live FlyFF test — in whole or in part.**
This covers: attaching to a running client, observation-only
attachment, native reader tests, pointer recovery, telemetry, recorder
collection, calibration, control/input tests, G5, G5-P2, live farming,
live training. Read-only does not make a test agent-runnable.

When live evidence is needed: inspect source/evidence, define the
question, freeze acceptance criteria *before* the test, hand the user
the exact procedure, **stop**, wait for the user to run it, then analyze
only the returned evidence. Use the `preparing-controlled-validation`
skill. See [ADR 0004](../decisions/0004-live-validation-by-user-only.md).

This rule is never overridden by any operating mode, including
`overnight-autonomous-work`.

## 3. Canonical ownership & compatibility caution

- Check `CANONICAL_OWNERS.toml` before assuming any file is dead —
  directory name or age is not evidence. See
  [`docs/architecture/COMPONENT_OWNERSHIP.md`](../architecture/COMPONENT_OWNERSHIP.md).
- Never widen the one registered R1b exception
  (`runtime_controller.py` → `farming.trainer`, 4 exact symbols).
- Never touch the checkpoint-ABI/pickle module-identity compatibility
  shims (`simulator/split_branch_policy.py`,
  `simulator/kinodynamic_route_planner.py`,
  `simulator/movement_kernel.py`) — permanent by construction. See
  [ADR 0002](../decisions/0002-preserve-abi-compatibility-shims.md).
- A migration phase number is **never**, by itself, evidence that a
  retained compatibility surface is safe to remove — check the actual
  retirement condition. See
  [ADR 0005](../decisions/0005-phase-is-not-evidence-of-retirement.md).

## 4. Test / gate discipline

Run the right gate for the change, not reflexively the biggest one. See
[`docs/operations/TESTING_AND_GATES.md`](../operations/TESTING_AND_GATES.md)
for the standard commands and accepted baselines. A pytest exit code of
1 can be the *correct* result when exactly the four accepted baseline
failures remain — never normalize away a fifth. Never weaken a frozen
acceptance gate after seeing a result.

## 5. Scientific integrity

Immutable evidence, deterministic seeds/paired sets where applicable,
frozen acceptance criteria declared before a run, no post-hoc gate
weakening, historical reproduction from frozen commits/tags (never
rewritten code). Never present inference as direct observation; never
convert correlation/statistical evidence into identity or causality
without support. Use the evidence-confidence labels in
[`docs/validation/README.md`](../validation/README.md) where evidentiary
strength matters.

## 6. Immutable artifacts

Never modify or regenerate: checkpoints, recordings, evaluation JSONs,
calibration CSVs, the authoritative Tower map bytes, Phase-3/6 fixtures,
historical reproduction snapshots, frozen baselines
(`docs/migration/BASELINE_VIOLATIONS.json` and `docs/migration/BASELINE_VIOLATIONS.md`), protected tags
(`pre-consolidation-head`, `historical-reproduction-baseline-20260815`,
`pre-consolidation-complete`), or B4 evidence.

## 7. Git discipline

Explicit-path staging only — never `git add -A`/`git add .`. No
`reset --hard`, `rebase`, `amend`, or force-push without explicit
per-instance user authorization. Never push without explicit
authorization. Before any command that could discard uncommitted work,
run `git status` first. Prefer coherent, narrow commits over one giant
change. Never rewrite an already-landed commit to fix a mistake — add a
forward-correcting commit instead (see section 10).

## 8. `MISTAKES.md`

`MISTAKES.md` (repository root) is the project's fast-retrieval
mistake log — not an architecture manual (that's `docs/architecture/`).
Before non-trivial work — especially coordinate systems/geometry,
observation/reward/action wiring, statistics/counting, fallback/
edge-case completeness, pointer/native-reader assumptions,
serialization/checkpoint compatibility, file/archive/history handling,
or a mathematical/scientific claim — skim relevant entries first.
Whenever a mistake, wrong assumption, incomplete audit, reasoning error,
or agent process failure is found (self-caught or user-caught), add an
entry **immediately**, using the template at the top of that file — not
batched for later. This applies to agent reasoning mistakes, not only
code bugs. Keep entries terse and LLM-retrieval-oriented. Never rewrite
an old entry to look correct in hindsight — append a clarification. If
the mistake changes current project understanding, also update the
relevant canonical doc; if discovered through validation, also preserve
that evidence.

## 9. Documentation-maintenance rule

Any task that changes the system **or** changes the project's
understanding of the system must leave canonical documentation accurate
before that task is considered complete. The trigger is "did the
project's knowledge change," not "was this a large code change" — e.g.
discovering a config option is unused, or that an apparently dead
compatibility file is actually load-bearing, requires a doc update even
with zero product code touched. For routine changes that genuinely do
not alter or clarify project knowledge, a task may report
`Documentation impact: none — <brief concrete reason>` — do not require
ceremonial edits for every helper-function change. Use the
`maintaining-project-knowledge` skill.

## 10. Forward correction, never historical rewriting

`docs/migration/` is the forensic/historical record — it is never
rewritten to match current understanding. If an old report contains an
obsolete belief: current docs state current truth, the historical report
remains historical truth, and a forward correction (a new section, a new
journal entry) may point between them where useful. This applies equally
to conversation-level corrections: acknowledge a mistake plainly and fix
it forward, never silently smooth it over.

## 11. Context hygiene — repository docs are durable memory

Repository documentation is durable project memory; conversation context
is temporary working memory. Persist important discoveries, decisions,
assumptions, unresolved questions, and validation results in the
repository before relying on compaction or clearing. Conceptual test:
*if this conversation disappeared right now, could a fresh agent recover
the important project state from the repository?* If not, persist it
first. See [ADR 0006](../decisions/0006-repo-docs-are-durable-memory.md).

Compact at natural checkpoints (a subtask completes, an investigation
finishes before implementation begins, substantial now-irrelevant tool
output has accumulated) — not merely because a fixed number of turns
elapsed. Before compacting: ensure discoveries are documented,
`MISTAKES.md` is updated where applicable, decisions are recorded,
commits are made when appropriate. If the client supports invoking
`/compact` (or equivalent) directly, use it; otherwise tell the user the
context is safe to compact and request it. Never pretend a client
command executed when it did not.

At the end of a fully completed, independent task, recommend a fresh
context for the next independent task (`/clear` or equivalent) — but
never before the final task report/handoff is complete, and never in a
way that suppresses that report.

**Cold-start loading order:** `CLAUDE.md`/`AGENTS.md` →
`docs/agent/PROJECT_RULES.md` (this file) → `docs/README.md` →
task-relevant canonical docs → relevant `MISTAKES.md` entries →
task-specific source/tests. Do not automatically ingest all of
`docs/migration/`, all historical reports, or every architecture file
unless the task actually needs them.

## 12. STOP conditions / no silent scope expansion

Stop and surface to the user rather than deciding unilaterally when: the
entry state for an authorized task doesn't match what was specified; an
instruction's literal condition fails (e.g. "only if X already exists"
and X does not); two materially different product directions are
plausible and evidence doesn't establish which is wanted; a change would
touch a frozen/immutable artifact; or a decision is genuinely ambiguous
and consequential. Do not expand a task's scope merely because time or
context remains — see the `overnight-autonomous-work` skill for the one
explicit, user-invoked exception to this, which still respects every
other rule in this document.

## 13. Operating-mode skills (user-invoked only)

Two skills change the agent's default operating posture. **Neither
activates automatically** — both require an explicit user invocation
(not necessarily the exact skill name; natural phrasing like "finish
this and shut down" or "work overnight" is sufficient).

- **`finish-current-task-and-shutdown`**: complete the current task to
  its normal definition of done (source, tests, docs, `MISTAKES.md`,
  commits, clean git state), make everything durable, then shut down the
  machine as the final action. An explicit shutdown request is
  sufficient authorization — do not ask for a second confirmation. If
  shutdown itself is blocked, report `SHUTDOWN: FAILED — <reason>` and
  leave the durable state intact; never claim a shutdown that didn't
  happen.
- **`overnight-autonomous-work`**: standing authorization to continue
  useful **offline** project work without repeatedly asking for
  approval on ordinary engineering decisions. It never overrides section
  2 (live execution), section 6 (immutable artifacts), section 7 (git
  discipline — no push/force-push/history rewrite), or section 5
  (scientific integrity — no gate-weakening). Reaching a point where the
  next required step is live evidence is a hard overnight stop
  condition. Maintains one dated durable log under
  `docs/agent/overnight/`.

See each skill's own `SKILL.md` for full workflow detail — this section
states only that they exist and their non-negotiable boundaries.

## 14. `prepare-clean-repo-snapshot` skill

A packaging-only skill for producing a compact, review-ready ZIP of the
current worktree for external inspection. Never modifies product/source
state; never commits; excludes caches/virtualenvs/databases/bulky
generated artifacts by default; protects obvious secrets; represents the
current worktree (not merely `HEAD`). See that skill's own `SKILL.md`.
This is a review snapshot, not a backup — it does not replace Git
history, protected tags, or artifact archives.

## Project skills (six, initial set)

1. `maintaining-project-knowledge`
2. `preparing-controlled-validation`
3. `making-safe-repository-changes`
4. `finish-current-task-and-shutdown`
5. `overnight-autonomous-work`
6. `prepare-clean-repo-snapshot`

No further skill should be added without explicit authorization.

## Knowledge map

This file states rules. For architecture, ownership, contracts,
validation status, and known debt, start at
[`docs/README.md`](../README.md).
