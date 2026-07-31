# HUMAN TEST REQUIRED â€” Consolidated live-client acceptance

Task: `DOC-003`. Automated refactoring is complete before this protocol. Run
once with a disposable/low-risk character state and stop immediately on any
unexpected input.

## Startup conditions

- FlyFF is visible, logged in, and standing in a known safe Tower AoE cell away from red teleport cells.
- The normal fixed camera/layout is in use.
- Tower AoE and at least one captured native mob species are selected.
- Preserve `models/farming/native_strategy_ppo.zip`; do not delete or rename it.

## Protocol (maximum 20 minutes, excluding the optional session-expiry edge)

1. Launch using `RUNBOOK.md`, attach the correct window, and wait up to 10 seconds for Bot Vision/FPS.
2. Click **Native Health**. Record the status/log; expect `healthy`, correct map, nonzero pointer bases/generation, focus state, and cached actor facts without a recovery scan.
3. Start **Native Dry Run** for 15â€“30 seconds. Verify movement begins, forward persists through steering, and Stop returns within 2 seconds with no held key.
4. Start Dry Run again, alt-tab away, and do not refocus for more than two seconds. Expect a focus terminal and released movement. Repeat once, refocusing during the grace period; expect the session to continue.
5. Start Training for 30â€“60 seconds, including at least one EVA if the policy chooses it. Verify F1 does not visibly cancel the held forward/steering state after animation lock. Press Stop; expect a clean completion/report and responsive GUI.
6. Start **Run Trained Agent** for 15â€“30 seconds, then Stop. Verify no legacy target-index/navigation behavior or backward action appears.
7. With control stopped, click **Recover Pointers**, then immediately Stop. Expect cancellation within 2 seconds, no GUI freeze, and no automatic JSON offset write.
8. Start Dry Run and close the GUI. Expect orderly shutdown within 8 seconds, no remaining `flyff-*` non-daemon process/thread, and no held movement key.

Optional session-expiry edge: near the server's daily farm timeout, start a
short training session and allow the server teleport. Expect an external
session classification, no teleport policy penalty, model/report/manifest
publication, released keys, and worker completion.

## Return evidence / failure indicators

Return the GUI log, newest `training_logs/farming/native_sessions/*.json` and
manifest, and a screenshot only if useful. Report any attach/preview delay over
10 seconds, Stop over 2 seconds, shutdown over 8 seconds, wrong map/pointer
facts, repeated recovery scans, held keys, backward movement, circling/target
navigation, EVA releasing movement, model overwrite after fatal failure, or a
missing/incorrect external-session report.

Safe cleanup: press Stop, wait for completion, foreground FlyFF and verify all
movement keys are up, then close the GUI. If Stop fails, background FlyFF before
ending the Python process.

