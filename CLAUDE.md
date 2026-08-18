# Project instructions: FlyFF AiBot

This is the agent entrypoint for this repository. Read in this order:

1. **`docs/agent/PROJECT_RULES.md`** — the permanent shared Claude/Codex
   rules (product direction, live-execution prohibition, canonical
   ownership, test/gate discipline, scientific integrity, immutable
   artifacts, git discipline, context hygiene, STOP conditions, the
   `MISTAKES.md` rule in full, and the six project skills).
2. **`docs/README.md`** — the current-project-knowledge index. Use it to
   find the architecture doc relevant to your task rather than reading
   everything.
3. **`MISTAKES.md`** (repository root, same directory as this file —
   moved here from `flyff_farming_simulator/MISTAKES.md` in Phase 14
   once its project-wide scope and the absence of any hardcoded path
   dependency were confirmed; see `docs/migration/codex_handoff/
   PHASE14_REPORT.md`) — skim relevant entries before non-trivial work,
   per `PROJECT_RULES.md` section 8.

## Cardinal rule: `MISTAKES.md`

This project keeps a running log of mistakes and wrong assumptions at
`MISTAKES.md`. Upholding it is a standing rule, not a one-time task —
see `docs/agent/PROJECT_RULES.md` section 8 for the full rule (this
summary is intentionally short so it stays skimmable):

- **Before** starting non-trivial work — especially anything touching
  coordinate systems/geometry, observation/reward wiring, statistics or
  counting, fallback/edge-case completeness in an algorithm, pointer/
  native-reader assumptions, serialization/checkpoint compatibility,
  file archival, or a mathematical/scientific claim — skim `MISTAKES.md`
  for related past entries first.
- **Whenever** a mistake, bug, or wrong assumption is found — whether
  self-caught or pointed out by the user — add an entry to
  `MISTAKES.md` immediately. Don't batch this for later.
- This applies to mistakes in reasoning/plans/claims, not only bugs
  found in the simulator code itself.

## Non-negotiable

- **Never execute live FlyFF work** — no attach, no read, no control, no
  recording, no calibration, no training against a live client. See
  `docs/agent/PROJECT_RULES.md` section 2. Live evidence is always
  prepared by an agent and run by the user.
- Important knowledge belongs in repository docs, not only in chat — see
  `docs/agent/PROJECT_RULES.md` section 11.
- Compact context at safe natural checkpoints when useful; recommend a
  fresh context after a fully completed independent task.

## Project skills

Claude Code's repository-local skill discovery location is
`.claude/skills/<name>/SKILL.md` — the same six skills are also
installed for Codex as thin wrappers under `.agents/skills/<name>/
SKILL.md` (Codex's own native discovery location, a genuinely separate
mechanism, not the same directory read two ways). See `AGENTS.md` if
you need the Codex-side detail; the canonical, authoritative body for
every skill lives under `.claude/skills/` regardless of which client
reads it.
