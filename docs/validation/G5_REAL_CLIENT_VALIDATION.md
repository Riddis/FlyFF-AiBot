# G5 — Real-Client Position/Pointer-Recovery Validation

**STATUS: PENDING / NOT RUN.**

This is a procedure and status document. It contains no results because
none exist yet. G5 is intentionally pending — not accidentally
forgotten — until the development bot's position/pointer-recovery stack
is judged ready for real-client evidence gathering. See
`docs/architecture/POSITION_AND_POINTER_RECOVERY.md` section 3 and
`docs/decisions/0004-live-validation-by-user-only.md`.

**No agent may execute any part of this procedure.** Agents prepare it;
the user runs it. See `docs/agent/PROJECT_RULES.md`.

## 1. Accepted G5 contract

G5 passes only when **all five** of the following are demonstrated:

1. Correct player recovery and monster-anchor exclusion in **at least
   two independent real sessions** using the merged `position/` stack
   with `LIVE_ATTACH_POLICY`.
2. **At least one session after a fresh client restart** — new PID,
   fresh actor allocations (not merely a reattach to an already-running
   client).
3. `RecoveredNativeProfile` **save → restore → successful fast-start**
   (the stable cross-process profile path — see
   `POSITION_AND_POINTER_RECOVERY.md` section 1 — actually skips full
   recovery on a subsequent launch).
4. Presence activation reaches the **same validated
   `presence_validation_source`** state observed before this migration
   began (i.e. the migration did not silently degrade presence
   validation).
5. A recorder session using `RECORDING_ATTACH_POLICY` creates an archive
   that **passes G7 end-to-end**
   (`docs/architecture/RECORDING_TELEMETRY_AND_ARCHIVES.md` section 3).

## 2. G5-P2 (conditional, separate from G5)

G5-P2 is **only required** if `LIVE_ATTACH_POLICY`'s
`player_discrimination` is intentionally changed away from
`LEGACY_SPECIES_ACTIVE` (e.g. to `EXACT_MONSTER_ANCHORS` or a new
strategy such as a candidate `anchor_base_exclusion`). No such change
has been proposed or made — see `POSITION_AND_POINTER_RECOVERY.md`
section 4. Until it is, G5-P2 has no procedure to prepare.

## 3. Preparation workflow (what an agent does)

Use the `preparing-controlled-validation` skill. In summary:

1. Confirm the exact commit/config the test will run against.
2. Confirm the 5 criteria above are still the frozen, unmodified
   acceptance bar (do not weaken them to make a future result look
   better).
3. Construct the exact command(s)/procedure for the user to run.
4. Specify exactly what evidence to preserve (logs, `Native Health`
   output, the recording archive, the `RecoveredNativeProfile` JSON
   before/after).
5. **STOP.** Hand the procedure to the user. Do not execute any part of
   it.
6. After the user returns evidence: analyze it, fill in
   `docs/validation/VALIDATION_TEMPLATE.md`, determine PASS/FAIL/
   INCONCLUSIVE per criterion, update `POSITION_AND_POINTER_RECOVERY.md`
   and `MISTAKES.md` as appropriate.

## 4. Record (fill in only after a real user-run session)

```
Status: PENDING
```

When a real session occurs, append a filled-in copy of
`docs/validation/VALIDATION_TEMPLATE.md` below this line, then update
the status line above and `docs/validation/README.md`'s status table.

---

*No entries yet.*
