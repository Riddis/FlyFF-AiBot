---
name: making-safe-repository-changes
description: The reusable methodology for making any repository change safely — inspect git state, respect canonical ownership and immutable artifacts, make narrow coherent changes, test appropriately, keep project knowledge current, and leave a clean state. Use this as the default approach for implementation/refactor/fix work in this repository.
---

# Making Safe Repository Changes

The methodology developed across this repository's consolidation
migration (Phases 0–13), generalized for ongoing use.

## Procedure

1. **Inspect exact git state.** Branch, HEAD, `git status`, whether the
   worktree is clean. Never assume.
2. **Read relevant current docs + `MISTAKES.md`.** Start at
   `docs/README.md`'s task-oriented table to find the right architecture
   doc; skim relevant `MISTAKES.md` categories before touching an
   error-prone area (coordinate systems, observation/reward/action
   wiring, statistics/counting, pointer/native-reader assumptions,
   serialization/checkpoint compatibility, file/archive handling).
3. **Understand canonical ownership.** Check `CANONICAL_OWNERS.toml`
   and `docs/architecture/COMPONENT_OWNERSHIP.md` before assuming a file
   is dead, redundant, or safe to move — directory name or age is not
   evidence.
4. **Identify immutable artifacts** the change must not touch:
   checkpoints, recordings, evaluation JSONs, calibration CSVs, the
   authoritative Tower map bytes, frozen fixtures, protected git tags,
   B4 evidence. See `docs/agent/PROJECT_RULES.md` section 6.
5. **Characterize before changing.** For anything non-trivial, confirm
   your understanding of current behavior against source/tests before
   editing — don't reason from memory or a prior document alone.
6. **Make a narrow, coherent change.** Resist unrelated cleanup riding
   along with the actual task.
7. **Explicit staging.** Never `git add -A`/`git add .`. Stage the exact
   paths that belong to this change.
8. **Focused tests first**, broader tests when risk warrants — see
   `docs/operations/TESTING_AND_GATES.md` for which gate fits which kind
   of change. A metadata-only change does not need the full ~9-minute
   product suite; a change touching product/runtime behavior or import
   structure does.
9. **Update project knowledge.** Use the `maintaining-project-knowledge`
   skill — update docs/`MISTAKES.md` if the change altered or clarified
   project understanding, then run
   `.\.venv\Scripts\python.exe tools\check_project_knowledge.py`.
10. **Leave a clean state.** `git status` clean or fully explained,
    `git diff --check` clean, no stray debug output.
11. **Forward-correct mistakes rather than rewrite history.** If you
    find you got something wrong mid-task (or after a commit already
    landed), fix it forward with a new commit/entry — never amend a
    landed commit or silently rewrite an earlier claim. See
    `docs/agent/PROJECT_RULES.md` section 10.
12. **Make task state durable before suggesting context compaction or a
    fresh session.** See `docs/agent/PROJECT_RULES.md` section 11.

## Never

- Destructive git operations (`reset --hard`, `rebase`, `amend`,
  force-push) without explicit per-instance user authorization.
- Push without explicit authorization.
- Assume a phase number or old-looking path means something is safe to
  delete — see [ADR 0005](../../../docs/decisions/0005-phase-is-not-evidence-of-retirement.md).
- Widen the R1b exception or touch the checkpoint-ABI/pickle-identity
  shims (`simulator/split_branch_policy.py`,
  `simulator/kinodynamic_route_planner.py`,
  `simulator/movement_kernel.py`).
