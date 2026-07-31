# PTR-LIVE-001 - Current-client pointer acceptance

Run this protocol once against the same current FlyFF client that reported
absolute player slot `0x6352B8` as null. Do not run the older consolidated live
protocol first. Keep the character logged in, alive, stationary, and outside a
Tower teleport cell.

## Evidence to retain

- The complete GUI log from attach through close.
- `position/native_position.json` and `position/native_monsters.json` after the
  run, plus any adjacent `.pre_pointer_recovery.bak` files.
- The exact outcome and rejection-summary line from **Recover Pointers**.
- If dry-run starts, the first native preflight line and the final Stop line.

## Procedure

1. Launch the application normally, select **Tower AoE** and at least one known
   monster species, then attach the correct FlyFF window. Confirm Bot Vision and
   FPS remain responsive.
2. Click **Native Health**. Record the summary containing PID, module name/path,
   module base/size, pointer width, configured relative offsets, absolute slots,
   health, map cell, actor-cache count, OCR state, and focus state.
3. If health is already `healthy`, continue to step 6. Otherwise click **Recover
   Pointers** exactly once. While it runs, move the GUI window and confirm it
   repaints. Do not click recovery again.
4. Expected successful discovery evidence:
   - strategy is `module_image`;
   - the result is `recovery_succeeded` and after-health is `healthy`;
   - player and world bases are nonzero and generation advances;
   - there is no ambiguity/instability outcome;
   - both JSON files contain the same recovered player relative offset, and the
     monster JSON contains the independently recovered world relative offset;
   - both writes completed together and the pre-recovery backups remain.
5. If recovery reports `not_found`, `deadline`, or `ambiguous`, stop this
   protocol without retrying. Return the final summary with `strategy`, scanned
   bytes/slots, validated count, and the self/world/coordinate/HP/non-player/
   missing-world/unstable/ambiguous rejection counts. This is a valid diagnostic
   result and must not be worked around by weakening validation.
6. Click **Native Health** once more. Expect `healthy` with the same module
   identity, coherent player/world bases, and the new current absolute slots.
7. Put another harmless application in the foreground, then start **Native Dry
   Run (No Learning)**. Expected order is: cheap pointer preflight (or one
   bounded non-persisting startup recovery), verified native state, focus
   acquisition/manual grace, actor/map environment preflight, then control.
   There must be no camera-discovery sweep and no input before native state is
   verified.
8. If startup recovery cannot resolve state, expect one concise
   `No input was activated` status and a normal control-worker completion. A
   `NativePointerSnapshotError` traceback is a failure.
9. If dry-run starts, let it run for 10-15 seconds, confirm native position and
   actors update, then press **Stop** once. Verify movement keys are released and
   completion is prompt.
10. Start **Recover Pointers** once more and immediately press **Stop**. Expect a
    cancelled or cache-hit completion promptly, no GUI freeze, no second scan,
    and no partial config write.
11. Close the application. Expect bounded worker joins, one final input release,
    and no lingering `flyff-*` project worker.

## Acceptance

Pass when attach/health remain responsive; module-image discovery either
produces a stable, unambiguous, transactionally persisted player/world pair or
produces actionable rejection evidence; a healthy pair permits dry-run only
after native validation and focus ordering; expected startup failure is clean
and input-safe; cancellation, Stop, and close are prompt.
