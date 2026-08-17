# Codex instructions: FlyFF AiBot

This is Codex's repository instruction entrypoint, kept in sync with
`CLAUDE.md` (Claude Code's equivalent entrypoint) rather than
maintained as a divergent copy. Read in this order:

1. **`docs/agent/PROJECT_RULES.md`** — the permanent shared Claude/Codex
   rules: product direction, the absolute live-execution prohibition,
   canonical ownership, test/gate discipline, scientific integrity,
   immutable artifacts, git discipline, context hygiene, STOP
   conditions, the `MISTAKES.md` rule, and the six project skills.
2. **`docs/README.md`** — the current-project-knowledge index. Find the
   architecture doc relevant to your task rather than reading
   everything.
3. **`flyff_farming_simulator/MISTAKES.md`** — skim relevant entries
   before non-trivial work, per `PROJECT_RULES.md` section 8.

## Non-negotiable

- **Never execute live FlyFF work.** No attach, read, control, recording,
  calibration, or training against a live client — not even read-only
  observation. Live evidence is always prepared by an agent and run by
  the user. See `docs/agent/PROJECT_RULES.md` section 2.
- Important knowledge belongs in repository docs, not only in chat/
  session history — see `docs/agent/PROJECT_RULES.md` section 11.
- No `reset --hard`/`rebase`/`amend`/force-push without explicit
  per-instance authorization; explicit-path staging only, never a broad
  add-everything.

## Project skills

Codex's native repository-local skill discovery location is
`.agents/skills/<name>/SKILL.md` — Codex scans `.agents/skills` in
every directory from the current working directory up to the
repository root, and supports both explicit invocation (`$skill-name`)
and implicit selection when a task matches a skill's `description`
(confirmed against Codex's own skills documentation,
`developers.openai.com/codex/skills`). This is a genuinely separate
mechanism from Claude Code's `.claude/skills/<name>/SKILL.md` — an
earlier version of this file incorrectly claimed no such mechanism
existed; see `flyff_farming_simulator/MISTAKES.md` for that correction.

The same six skills are installed for both clients, as thin,
non-divergent surfaces over one canonical implementation each:

1. `maintaining-project-knowledge`
2. `preparing-controlled-validation`
3. `making-safe-repository-changes`
4. `finish-current-task-and-shutdown` (user-invoked only — see its
   `SKILL.md`)
5. `overnight-autonomous-work` (user-invoked only — see its `SKILL.md`)
6. `prepare-clean-repo-snapshot`

**Canonical bodies** (the full workflow/procedure/rules for each skill)
live at `.claude/skills/<name>/SKILL.md`. **`.agents/skills/<name>/
SKILL.md`** holds a thin wrapper for each — same name, same
description (for Codex's implicit-matching to work), and a pointer
telling Codex to read and follow the canonical body. If a wrapper's
wording ever differs from its canonical body, the canonical body
controls. This is six logical skills exposed through two client-native
discovery surfaces, not twelve skills — no further skill should be
added to either surface without explicit authorization, and neither
surface should become a second, independently-maintained
implementation.

## Context hygiene (tool-agnostic)

Use whatever your host environment calls "compact supported context" and
"start a fresh session" at the natural checkpoints described in
`docs/agent/PROJECT_RULES.md` section 11 — this repository does not
assume Claude and Codex expose identical slash commands, so that section
is written in generic terms. Never claim a context-management action
happened if your environment did not actually perform it.
