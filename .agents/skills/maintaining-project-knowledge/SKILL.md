---
name: maintaining-project-knowledge
description: Decide where new project knowledge belongs (canonical docs, MISTAKES.md, validation evidence, an ADR, or known debt), preserve evidence, and run the lightweight knowledge-integrity check before considering the context safe to compact or clear. Use whenever a task discovers a fact, corrects an assumption, or changes understanding of the system — regardless of whether product code changed.
---

# Maintaining Project Knowledge (Codex discovery wrapper)

This is a thin Codex-native discovery wrapper, not a second
implementation. Read and follow the canonical, authoritative skill body
in full:

```
.claude/skills/maintaining-project-knowledge/SKILL.md
```

The canonical body controls if this wrapper's wording ever differs from
it. This wrapper exists only so Codex's repository-local skill scan
(`.agents/skills/`) can discover the same skill Claude Code discovers
under `.claude/skills/` — the workflow, procedure, and rules are defined
once, at the canonical path above.
