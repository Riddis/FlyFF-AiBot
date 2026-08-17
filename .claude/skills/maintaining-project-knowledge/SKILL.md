---
name: maintaining-project-knowledge
description: Decide where new project knowledge belongs (canonical docs, MISTAKES.md, validation evidence, an ADR, or known debt), preserve evidence, and run the lightweight knowledge-integrity check before considering the context safe to compact or clear. Use whenever a task discovers a fact, corrects an assumption, or changes understanding of the system — regardless of whether product code changed.
---

# Maintaining Project Knowledge

## When to use this

Any time a task **discovers or changes project knowledge** — not only
when it changes code. Examples that trigger this skill with zero code
changes: discovering native read timing differs from what was believed,
learning a pointer field has different semantics, a live validation
confirming/falsifying an assumption, discovering a config option is
unused, discovering a checkpoint compatibility constraint, revising
confidence in historical population evidence, discovering an
apparently-dead compatibility file is actually load-bearing, or learning
a prior phase gate encoded an incorrect retirement assumption.

## Procedure

1. **Discover/change knowledge.** State plainly what changed and why it
   matters.
2. **Identify the canonical document.** Use this classification (see
   `docs/README.md`'s knowledge model for the full picture):
   - Current architectural/contract fact → the relevant
     `docs/architecture/*.md`.
   - A mistake, wrong assumption, or reasoning error → `flyff_farming_
     simulator/MISTAKES.md`, using its entry template. **Do this
     immediately if applicable, in the same pass** — do not defer it.
   - Something a specific experiment/validation observed →
     `docs/validation/` (use `docs/validation/VALIDATION_TEMPLATE.md`).
   - A decision whose *rationale* is worth preserving independently of
     the code → a new entry in `docs/decisions/` (only if not already
     adequately covered — most ownership/dependency facts belong in
     `CANONICAL_OWNERS.toml` + `COMPONENT_OWNERSHIP.md`, not a new ADR).
   - Something intentionally retained/pending/unresolved →
     `docs/KNOWN_DEBT.md`.
   - How the repository reached its current state (forensic) →
     `docs/migration/` — but this is written once per phase, not a
     general destination for new knowledge going forward.
3. **Preserve evidence.** Cite the source: a test name, a file+line, a
   hash, a command output, a validation record. Confidence language
   (`VERIFIED_CONTRACT` / `USER_RUN_LIVE_VALIDATION` /
   `HISTORICAL_EVIDENCE` / `BEST_CURRENT_ESTIMATE` / `INFERENCE` /
   `ASSUMPTION` / `UNRESOLVED`, defined in `docs/validation/README.md`)
   where evidentiary strength matters — never present inference as
   direct observation.
4. **Update current truth.** Edit the canonical document identified in
   step 2. Keep `docs/migration/` untouched — forward-correct instead of
   rewriting historical reports (see `docs/agent/PROJECT_RULES.md`
   section 10).
5. **Update `MISTAKES.md`** if a wrong assumption was involved, even if
   step 2 already routed the main content elsewhere.
6. **Run the lightweight knowledge-integrity check:**
   ```powershell
   .\.venv\Scripts\python.exe tools\check_project_knowledge.py
   ```
   Fix anything it flags before proceeding.
7. **Context is now safe to compact/clear if appropriate** — the
   important state is durable in the repository, not only in
   conversation. See `docs/agent/PROJECT_RULES.md` section 11 for the
   compaction/clearing checkpoints themselves.

## What this skill is not

Not a mandate to create documentation ceremony for routine
implementation changes that don't alter or clarify project knowledge —
for those, state `Documentation impact: none — <brief concrete reason>`
and move on (`docs/agent/PROJECT_RULES.md` section 9).
