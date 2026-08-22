# ADR 0006: Repository documentation is durable project memory; conversation is temporary

## Status

Accepted, Phase 13.

## Context

Phases 0–12 of this migration produced unusually deep, hard-won
knowledge — architecture, ownership, dependency direction, checkpoint
ABI, position/pointer-recovery mechanics, scientific validation
principles, discovered mistakes and their corrections — but much of it
existed primarily in agent-conversation history and scattered migration
reports rather than in a durable, current-state-focused documentation
system. A conversation can be lost, compacted, or simply not available
to the next session; a fact that exists only there is one incident away
from having to be rediscovered from scratch.

## Decision

Repository documentation (`docs/architecture/`, `docs/validation/`,
`docs/operations/`, `docs/decisions/`, `docs/KNOWN_DEBT.md`,
`MISTAKES.md`) is the durable project memory. Conversation context is
temporary working memory. Any task that changes the system *or* changes
the project's understanding of the system must leave the canonical
documentation accurate before that task is considered complete — the
trigger is "did the project's knowledge change," not "was this a large
code change." A fresh agent session should be able to reconstruct
important project state from the repository alone.

Conceptual test: *if this conversation disappeared right now, could
another competent agent recover the important project state from the
repository?* If no, persist it before considering the task complete or
the context disposable.

## Consequences

- `docs/agent/PROJECT_RULES.md` carries this as a standing rule,
  referenced from both `CLAUDE.md` and `AGENTS.md`.
- The `maintaining-project-knowledge` skill operationalizes the
  discover → classify → preserve evidence → update current truth →
  update `MISTAKES.md` if applicable → run the knowledge-integrity
  check → context-safe-to-compact/clear workflow.
- Cold-start sessions use progressive context loading
  (`CLAUDE.md`/`AGENTS.md` → `docs/agent/PROJECT_RULES.md` →
  `docs/README.md` → task-relevant docs → relevant `MISTAKES.md`
  entries → task-specific source) rather than ingesting the entire
  documentation corpus or migration history by default.
- `docs/migration/` remains historical/forensic evidence — it is not
  rewritten to match current understanding; forward corrections point
  between old and current where useful (see ADR 0005 for a worked
  example of this convention).

## Evidence

`docs/agent/PROJECT_RULES.md`, `docs/README.md`,
`.claude/skills/maintaining-project-knowledge/` (or the repository's
actual supported skill location — see that skill's own doc for the
verified path).
