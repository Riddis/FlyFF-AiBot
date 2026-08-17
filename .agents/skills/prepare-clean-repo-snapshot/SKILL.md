---
name: prepare-clean-repo-snapshot
description: Produce a compact, review-ready ZIP of the current worktree (not just HEAD) for sending to another agent for inspection. Excludes .git, virtualenvs, caches, databases, and bulky generated ML/recording artifacts by default; protects obvious secrets; never modifies product/source state or commits anything. Use when the user asks to "zip the repo", "package the repo for review", or similar.
---

# Prepare Clean Repo Snapshot (Codex discovery wrapper)

This is a thin Codex-native discovery wrapper, not a second
implementation. Read and follow the canonical, authoritative skill body
in full:

```
.claude/skills/prepare-clean-repo-snapshot/SKILL.md
```

The canonical body controls if this wrapper's wording ever differs from
it. The tool it calls (`tools/create_clean_repo_snapshot.py`) is shared
by both surfaces — invoke it the same way regardless of which client is
running.
