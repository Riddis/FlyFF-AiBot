---
name: finish-current-task-and-shutdown
description: User-invoked only. Finish the CURRENT task to its normal definition of done (source, tests, docs, MISTAKES.md, commits, clean git state), make everything durable, produce the normal final report, then shut down the computer as the final action. Triggered by explicit requests like "finish this and shut down" or "finish the current task and turn off the computer" — not by any inference, and not automatically.
---

# Finish Current Task and Shutdown

## Invocation

**User-invoked only.** Activates on an explicit request — natural
phrasing is enough (e.g. "finish this and shut down", "finish the
current task and turn off the computer"); the user does not need the
exact skill name. Never infer this mode from unrelated wording, and
never activate it merely because it is late or because shutting down
seems convenient.

This skill is **not** "shut down immediately." It is "complete the
current task to its normal definition of done, then shut down."

## Shutdown authorization

An explicit request to finish the current task and shut down the PC is
**sufficient authorization for the final OS shutdown**. Do not ask for a
second confirmation before executing it.

## Workflow

1. Continue the current task.
2. Resolve ordinary implementation/test/documentation issues normally.
3. **Do not expand into unrelated next tasks** merely because time
   remains — this skill finishes the current task, it does not start
   the next one.
4. Bring the current task to its normal completion state, including
   where applicable: source changes, focused tests, broader tests when
   warranted, documentation updates, `MISTAKES.md` updates,
   validation/evidence records, the project-knowledge check
   (`tools/check_project_knowledge.py`), commits, and a clean final git
   state.
5. If an ordinary recoverable problem is discovered: resolve it (use
   `making-safe-repository-changes`).
6. If an **absolute blocker** prevents completion: stop trying to force
   it, preserve the blocker durably (in the repository, not only in
   chat), produce the normal handoff, then proceed to shutdown unless
   shutdown itself would make the situation unsafe (e.g. mid-write on a
   file, an unresolved merge state).
7. Ensure anything important that currently exists only in conversation
   has been written to the repository before shutdown — apply the
   `maintaining-project-knowledge` durability test.
8. Produce the normal final task report/handoff.
9. Record at minimum: final HEAD, task completion status, tests/gates
   run and their results, documentation impact, unresolved issues, the
   blocker if any, worktree/index state, and the next recommended
   action.
10. **Shutdown is the final machine-affecting action.** Do not modify
    code after initiating shutdown, do not leave an important edit only
    in an unsaved buffer, do not leave the only description of a blocker
    in chat, do not start another task before shutting down.

## Shutdown mechanics

Use the appropriate operating-system shutdown mechanism for the actual
environment — inspect the OS rather than guessing (this repository's
primary development environment is Windows; use the Windows shutdown
mechanism there, e.g. `shutdown /s /t <delay>`, not a Unix command). A
short delay is acceptable to let terminal/chat output flush, but never
use that delay window to begin new work.

If the client permits a final user-visible response before the shutdown
command executes, provide it. If the execution model makes that
unreliable, write the complete durable handoff to the repository first,
then initiate shutdown.

**If shutdown execution is itself blocked** (permissions, tooling, a
sandboxed environment with no shutdown capability): do not claim the
machine was shut down. Report exactly:

```
SHUTDOWN: FAILED — <reason>
```

and leave the exact durable final state described in the handoff.

## Context handling

No need to `/clear` after invoking shutdown — the session is ending
with the machine. The same durability test still applies before
shutdown: if this conversation vanished right now, could a fresh agent
recover the important task state from the repository? If not, make it
durable first.
