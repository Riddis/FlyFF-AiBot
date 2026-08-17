---
name: overnight-autonomous-work
description: User-invoked only. Standing authorization to continue useful OFFLINE project work for an extended period without repeatedly requesting approval on ordinary engineering decisions, so the user can go to sleep. Never overrides the live-execution prohibition, artifact immutability, git safety, or scientific-integrity rules. Triggered by explicit requests like "work overnight" or "keep working while I sleep" — not by any inference, and not automatically.
---

# Overnight Autonomous Work

## Invocation

**User-invoked only.** Activates on an explicit request — natural
phrasing is enough (e.g. "work overnight", "keep working overnight",
"continue working while I sleep"); the user does not need the exact
skill name. Never activate this mode merely because it is late at
night or because it would be convenient.

This is intentionally a **higher-agency mode**: once invoked, the point
is that the user should be able to go to sleep. Do not repeatedly stop
for ordinary decisions the agent can safely and reasonably resolve
itself.

## What overnight invocation authorizes

Standing authorization, without repeated re-confirmation, to:
continue the current task; fix issues discovered while completing it;
perform necessary supporting refactors; run appropriate offline tests;
update documentation and `MISTAKES.md`; investigate failures;
make evidence-backed engineering decisions; commit coherent completed
work; move to the next logical **offline** task when the current one is
complete; continue through subsequent offline tasks/phases when their
predeclared acceptance gates are satisfied; run offline simulator/
training/evaluation work when it is within the project's then-current
authorized path; continue making progress until a defined overnight
stop condition occurs (section "End states" below).

Do not stop merely because "normally I would ask whether to continue."

## What overnight mode never overrides

This mode **never** permits: launching or attaching to FlyFF; executing
G5 or G5-P2; live telemetry, live recording, live pointer recovery, live
calibration, live client control, live farming, or live training;
pushing to remote (unless separately authorized); force-push; rewriting
git history; weakening a frozen acceptance gate after seeing results;
rewriting scientific evidence; fabricating validation; silently
discarding failing evidence; bypassing an explicitly frozen safety/
quality gate merely to keep moving.

**All live FlyFF execution remains user-run, absolutely.** If overnight
work reaches the point where live evidence is the next required step,
that is itself an overnight stop condition — prepare the user-run
procedure (`preparing-controlled-validation`) and preserve it for the
morning; do not attempt it.

## Decision-making standard

Prefer resolving issues yourself over asking questions. For an ordinary
engineering ambiguity: inspect source/docs/tests/evidence, identify the
viable options, choose the best-supported reversible option, implement
it, test it, and record the reasoning in the overnight log when
materially important. Do not create artificial blockers merely because
several acceptable implementations exist — use project priorities,
architecture rules, and evidence to decide. When genuinely uncertain,
prefer narrow, reversible, evidence-preserving changes. Do not lower
quality merely because the user is absent.

## Genuine blockers (stop for these; nothing else)

Stop only when there is no safe productive path forward without user
input, or the project-complete condition is reached (below). Genuine
blockers: live FlyFF execution is required; credentials/secrets/user
account interaction is required; an irreversible/destructive decision
needs user choice; two materially different product directions exist
and repository evidence does not establish which is wanted; required
source/data/artifact is unavailable and cannot be reconstructed safely;
a frozen contract conflicts with the requested next step and resolving
it requires changing user-defined acceptance criteria; hardware/OS/
environment failure; context/tool limits make safe continuation
impossible; the next action would violate a hard repository rule.

**Not** blockers: a failing test, a difficult bug, or a design choice
with one clearly better evidence-backed answer. If blocked on one
subtask but other independent authorized work remains safe to do,
continue that work — stop only when no meaningful authorized progress
remains.

## Task selection (after the current task completes)

1. Directly required follow-up from the completed task.
2. Blockers preventing the canonical development bot from becoming
   complete.
3. Highest-priority documented current work.
4. Unresolved offline correctness/architecture issues.
5. Simulator/digital-twin correctness work.
6. Authorized offline model/curriculum work.
7. Documented technical debt that materially blocks readiness.

Use `docs/README.md`, `docs/KNOWN_DEBT.md`, current validation state,
and the current handoff/state journals to select useful next work. Do
not polish low-value cosmetics while a known functional blocker exists;
do not begin speculative unrelated features.

## Project-complete stop condition

The long-term offline project-complete condition:

```
the canonical DEVELOPMENT bot is fully feature-ready
AND
the model has successfully completed the entire defined OFFLINE curriculum
AND
that curriculum includes successful operation on the simulated Tower map
AND
all required offline readiness gates are satisfied
AND
the remaining next step is user-run live validation/training
```

Do not declare this from intuition — it must be supported by the
project's documented completion/readiness criteria and evidence. When
reached, **stop** and record:

```
OFFLINE PROJECT GOAL REACHED
READY FOR USER-RUN LIVE TRAINING/VALIDATION
```

then wait for the user. Reaching this state does **not** authorize live
training.

## Mandatory overnight log

Maintain one durable, dated log file per run under
`docs/agent/overnight/`, named with the local start date/time, e.g.
`docs/agent/overnight/2026-08-18_0105.md`. Detailed but event-oriented —
do not dump raw shell output or thousands of lines of test logs into it.
Update it **during** the run, at every material checkpoint, not
reconstructed from memory afterward. If context compaction occurs,
update the log *before* compacting.

Suggested structure:

```markdown
# Overnight work log

Started:
Starting HEAD:
Starting task:
User authorization:
Overall status:

## Timeline

[01:12] Task ...
- objective / action / result / evidence / decision / confidence / next step

[02:03] Issue discovered ...
- symptom / diagnosis / options considered / chosen resolution / why / tests+result

## Commits
## Tests / gates
## Issues encountered
## Decisions made autonomously
## Documentation / MISTAKES updates
## Final status
```

At minimum record: timestamps, tasks attempted/completed, important
discoveries, failures and how they were handled, autonomous decisions
and assumptions, commits, tests/gates, documentation updates,
model/training/evaluation progress where applicable, blockers,
confidence, current task, next intended step, and why the run stopped.

## Git discipline overnight

Same rules as always (`making-safe-repository-changes`): explicit
staging, coherent commits at stable checkpoints rather than one giant
uncommitted tree, no push, no history rewrite, no meaningless commits
just because time passed. Record every commit hash in the overnight
log.

## Test / scientific discipline overnight

Never "make the tests green somehow." When a gate fails: diagnose it,
determine whether the implementation or the assumption is wrong,
preserve the failing evidence, update `MISTAKES.md` when appropriate,
fix root causes where supported, rerun, and never weaken the acceptance
criterion after seeing the result. For offline model work: preserve
seeds/configuration/provenance, preserve holdouts, never contaminate
evaluation, never choose a gate after seeing the outcome, preserve
scientifically meaningful failed runs, and keep exploratory work
distinct from accepted evidence.

## Context hygiene overnight

Use `docs/agent/PROJECT_RULES.md` section 11's checkpoints. Before a
safe compaction point: commit appropriate completed work, update
canonical docs and `MISTAKES.md`, append the overnight log, record
current HEAD/task/next step. If the client allows the agent to invoke
compaction itself, do so at natural durable checkpoints. If it does not,
**do not stop overnight work merely to ask the sleeping user to
compact** — continue with the best available context management. If
eventual context exhaustion makes safe continuation impossible, preserve
all state, mark it an absolute blocker, and stop. Do not use `/clear` (or
equivalent) during an active overnight run unless the client has a
verified mechanism that preserves/reloads the durable task state safely
— prefer compaction over clearing while work is in progress.

## Do not wait for approval overnight

Do not send questions like "Would you like me to continue?", "Should I
fix this?", "Do you want me to run the tests?", or "Should I move on to
the next task?" when the answer is determinable from the overnight
authorization, project rules, current documentation, evidence, frozen
gates, or documented priorities. Decide and continue. Only genuine
user-dependent blockers (above) justify stopping for input.

## End states

Every overnight run ends in exactly one of:

```
CONTINUING_UNTIL_SESSION_LIMIT
BLOCKED_REQUIRES_USER
OFFLINE_PROJECT_GOAL_REACHED
ENVIRONMENT_FAILURE
SAFE_STOP_AFTER_MAXIMUM_USEFUL_PROGRESS
```

Prefer reaching `BLOCKED_REQUIRES_USER` or `OFFLINE_PROJECT_GOAL_
REACHED` where possible. Do not choose
`SAFE_STOP_AFTER_MAXIMUM_USEFUL_PROGRESS` merely because the task became
difficult or several hours passed.

## If the user returns mid-run

The user's new instruction takes priority immediately. Reach a safe
checkpoint where possible, append current state to the overnight log,
and report what is in progress before switching. Do not continue
blindly against a new direct instruction.

## Morning handoff

When the user next says something like "good morning", "morning", or
"how did it go?", treat it as a request for the overnight handoff if a
recent overnight log exists. Before answering: read the most recent
overnight log, inspect current git HEAD/status, inspect the final
recorded tests/gates, and verify whether work stopped blocked,
completed, or is still mid-task. Then give a real report — never just
"Good morning."

The morning report should summarize (concise prose; detailed chronology
stays in the log): overnight start/end status and elapsed period;
starting and current HEAD; commits made; tasks completed; current task;
major progress; important discoveries; issues encountered and how each
was resolved; unresolved issues; blockers requiring user input;
tests/gates and outcomes; offline training/curriculum progress where
applicable; documentation/`MISTAKES.md` updates; any autonomous
architectural decisions; a confidence assessment (below); whether
anything should be independently reviewed; the exact reason the run
stopped; and the recommended next action.

### Confidence reporting

Always include a reasoned confidence assessment, e.g.:

```
Confidence: HIGH
Reason: full test suite green, 3 focused regression tests added, offline-only changes
```

Base it on test coverage, evidence quality, unresolved assumptions,
whether live evidence is still missing, whether high-risk architecture/
model changes were involved, and whether results reproduced
consistently. Never describe a purely offline result as live-validated.

## Interaction with `finish-current-task-and-shutdown`

Distinct skills. "Work overnight" does not imply "shut down when done."
"Finish this and shut down" does not imply "continue through the entire
project overnight." If the user explicitly combines them (e.g. "work
overnight and shut down when you hit a blocker or finish"): follow this
skill until a defined end state, make state fully durable, write the
final overnight log/handoff, then execute shutdown as the final machine
action per `finish-current-task-and-shutdown`.

## Phase-13 note

Creating and documenting this skill is not authorization to activate
it. Overnight mode was not actually invoked during Phase 13.
