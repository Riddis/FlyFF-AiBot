# PTR-LIVE-001 - Anchored current-client pointer acceptance

Run this protocol once against the same current FlyFF client. Do not restart,
reattach, run dry-run, or invoke recovery a third time between the two recovery
samples. The first sample discovers a candidate; the second confirms movement
of that exact candidate.

## Evidence to retain

- The complete GUI log from attach through close.
- The exact current and maximum player HP entered for each recovery sample.
- `position/native_position.json` and `position/native_monsters.json` after the
  run, plus adjacent `.pre_pointer_recovery.bak` files if created.
- Every recovery progress/outcome line, especially the old-self near-match,
  monster consensus, inferred field, spawn, direct/chain, and movement counts.

## Exact protocol

1. Start FlyFF and the bot normally. Select **Tower AoE** and select both
   **Captain Asterius** (species 944) and **Captain Dantalian** (species 948).
   Attach the correct FlyFF window and verify Bot Vision/FPS remain responsive.
2. In FlyFF, place the character at the Tower AoE native spawn corresponding to
   map `(0.0, 0.0)`: native X `253.0`, Z `86.0`. Keep the character stationary.
   Ensure multiple selected monsters are active nearby. Do not stand in a
   teleport trigger.
3. Read the character sheet and enter the exact integer **Player HP** and
   **Max HP** in the GUI recovery fields. Record both numbers in the returned
   log. Stop any control task, then click **Recover Pointers** exactly once.
4. While the worker runs, move the GUI window once to confirm repainting, but do
   not move the character. Wait for completion. Expected result is
   `recovery_movement_required`; it is not a failure. The log must show at least
   either two validated actors spanning species 944/948 or at least three
   validated actors from one of them, nonzero monster-base hypotheses, zero
   distinct layout ties, the `self_aliases` count, inferred
   species/active/HP/XYZ/world/self field offsets and support, one stable
   spawn/player candidate, and a direct or one-hop player and world reference.
   `layout_ties` must be zero; `self_aliases` may be greater than one because
   repeated, same-cohort self fields are validated aliases, not different actor
   layouts. It must also print the exact entered spawn/HP anchors.
   Neither JSON config may change and no recovery backup may be created yet.
5. If the first result is anything other than `recovery_movement_required`,
   stop the protocol without retrying and return the complete log and unchanged
   configs. Do not perform the movement or a second recovery. Outcomes such as
   monster consensus not found, actor layout
   inconclusive, spawn player not found, ambiguity, deadline, or cancellation
   are valid diagnostic evidence.
6. Without restarting or reattaching, focus FlyFF and manually move the
   character 3-5 native units away from the spawn (roughly 2-3 map cells). Stop
   all movement and remain stationary. Do not return to X `253.0`, Z `86.0`.
7. Re-read the exact current and maximum HP. Update both GUI fields if needed,
   record them, and click **Recover Pointers** exactly once more. Do not click
   any other diagnostic between the two samples.
8. Expected second result is recovery success using
   `anchored_movement_confirmation`/`anchored_movement`, with movement observed,
   healthy after-state, nonzero coherent player/world bases, and generation
   advanced. This call must not repeat the full module/private-memory scan.
   Both configs must update together: matching player relative slot and player
   chain, the monster world slot/chain, and inferred `layout.world_offset` and
   `layout.self_pointer_offset`. Adjacent pre-recovery backups must preserve the
   prior files.
9. Click **Native Health** once. Expect `healthy`, the same module identity,
   the recovered chain/field summary, and a Tower local coordinate consistent
   with the short movement.
10. Put a harmless application in the foreground and start **Native Dry Run
    (No Learning)**. Native preflight must succeed before autofocus or any
    input. Let it run 10-15 seconds, then press **Stop** once. Verify prompt
    completion and released movement keys.
11. Close the application. Expect bounded worker joins, final input release,
    and no lingering `flyff-*` project worker.

## Stop conditions

At any non-success outcome, do not loop recovery, edit offsets manually, or
weaken validation. Return the complete log, both HP pairs, both current config
files, and any recovery backups. If HP changed after the first sample, the
second sample must contain the updated exact current HP; an incorrect value is
expected to invalidate the pending candidate.

## Acceptance

Pass only when the first call is movement-gated with no write, the second call
confirms the same candidate without rescanning and performs one transactional
write, Native Health becomes healthy, dry-run activates no input before native
validation, and Stop/close remain prompt. Until this live evidence is returned,
`PTR-LIVE-001` remains open.
