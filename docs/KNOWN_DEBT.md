# Known Debt & Open Questions

Separated by kind — not everything unresolved is a bug, and not
everything retained is "debt" in the pejorative sense. Each item states
**why** it exists, not just that it exists.

## Intentionally retained compatibility (not debt to "clean up")

| Item | Why it exists | Retirement condition |
|---|---|---|
| 16 `TEST_CONTRACT_RETIREMENT` shims (`foreground_vision_bot/farming/*`, `flyff_farming_recorder/position/*`) | Load-bearing for `docs/migration/tests/test_phase{4,5}_contracts.py`'s frozen historical-reproduction checks | The specific test contract requiring each must be deliberately retired/replaced first — not a phase number. See [ADR 0005](decisions/0005-phase-is-not-evidence-of-retirement.md) |
| 3 checkpoint-ABI/pickle-identity shims (`simulator/split_branch_policy.py`, `simulator/kinodynamic_route_planner.py`, `simulator/movement_kernel.py`) | Required for `pickle.loads()` of live checkpoint/fixture instances at their pinned `__module__` path | `removal_gate = "NEVER"` — genuinely permanent, not conditional. See [ADR 0002](decisions/0002-preserve-abi-compatibility-shims.md) |
| R1b coupling (`runtime_controller.py` → `farming.trainer`) | All four functions require the live, already-attached `Bot` instance as their first parameter; cannot cross a subprocess boundary without a real attachment redesign or an explicitly-forbidden IPC bridge | A deliberate farming-runtime/attachment redesign — not assigned to any phase |
| B4 historical reproduction (git tag `historical-reproduction-baseline-20260815`) | Permanent commit/worktree address for the proven 2026-08-15 820M historical reproduction | `NEVER` — permanent by design |

## Pending empirical validation (not debt)

| Item | Status | Detail |
|---|---|---|
| G5 (real-client position/pointer-recovery validation) | PENDING — not run | [`docs/validation/G5_REAL_CLIENT_VALIDATION.md`](validation/G5_REAL_CLIENT_VALIDATION.md) |
| G5-P2 (discrimination-policy change validation) | PENDING — conditional, only required if `LIVE_ATTACH_POLICY` discrimination changes | Same document, section 2 |

## Unresolved future product choices

- Final shipped checkpoint (0051200 is the frozen ABI-test corpus
  member, not a shipping decision).
- Whether vision-based OCR/UI-detection/minimap-heading template
  matching is retained or replaced by pure-native reading in a future
  runtime derivative.
- Whether `runtime_controller.py`'s `farming.trainer` coupling (R1b) is
  omitted or redesigned for a future derivative.
- Final entrypoint name/location for any future derivative (explicitly
  not `apps/live_bot.py` per every phase's own prohibition).
- Whether `requirements.txt`/`requirements-training.txt`'s split changes
  to reflect `torch`/`gymnasium`/`stable_baselines3`'s `DUAL_ROLE`
  classification.

Source of record: `future_runtime_profile/dependency_profiles.toml`'s
`unresolved_future_choices`.

## Cleanup deferred from Phase 12, resolved in Phase 13/14

- `pyproject.toml`'s Ruff per-file-ignore
  `"foreground_vision_bot/mapper/[A-Z]*.py"` — corrected in Phase 13.
  See `docs/migration/codex_handoff/PHASE13_REPORT.md`.
- `flyff_farming_recorder/requirements.txt` — **resolved in Phase 14**:
  its content (`msgpack>=1.0`, `pywin32>=306`) was a genuine migration
  gap — `msgpack` (needed by `simulator/schema.py`/`recorder/*`, DUAL_
  ROLE per `PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md`) was never merged
  into the canonical root `requirements.txt` during the Phase-7
  collapse. Merged (`pywin32` was already present; the installed
  version 312 already satisfies the dropped `>=306` floor); the
  now-fully-redundant file was removed (`git rm`). See
  `docs/migration/codex_handoff/PHASE14_REPORT.md`.
- `foreground_vision_bot/foreground_vision_farm.json` and the root-level
  `foreground_vision_farm.json` — **fully resolved**: both are
  PySimpleGUI-auto-generated settings snapshots keyed to the pre-
  Phase-7 entrypoint's file name (`foreground_vision_farm.py`), proven
  orphaned by reading PySimpleGUI's own filename-derivation source (see
  `docs/architecture/SYSTEM_OVERVIEW.md` section 3a) — not merely
  "zero current references." The `foreground_vision_bot/` copy was
  removed in Phase 14 (`git rm`); the root-level copy was left
  untouched at the time (outside that cleanup's authorized scope) and
  was removed in the 2026-08-21 repository cleanup, which explicitly
  authorized it.

## Known runtime limitations

- **GUI settings persistence is CWD-dependent.** `Gui.py`'s `sg.
  user_settings_filename(path=".")` call lets PySimpleGUI derive both
  the settings filename (from the entrypoint's own file name) and its
  storage location (the process's current working directory) rather
  than using a fixed, repo-relative path. Launching the dev app from
  different working directories produces different, ungoverned
  `dev_app.json` files. Not a migration regression (the same call
  pattern predates consolidation) and not fixed in Phase 14 (a real
  fix — pinning an explicit filename/path — is a product decision about
  where user settings should canonically live, not evidence-driven
  cleanup). See `docs/architecture/SYSTEM_OVERVIEW.md` section 3a.

## Real scientific uncertainty (reader timing / population)

See `docs/architecture/RECORDING_TELEMETRY_AND_ARCHIVES.md` sections
5–8 for full detail:

- **Sequential, not simultaneous, actor-frame reads** — no guaranteed
  sub-frame per-actor timestamp precision.
- **Poot/WFC historical recorder layout is only partially populated**
  — weaker evidence base than the fuller Riddims population; don't
  treat conclusions from it as equally strong.
- **`respawn_candidate` is statistical evidence, not identity** — never
  promote it to a monster-identity claim without direct evidence.
- **Physical spawn loci remain unproven** beyond one established Tower
  spawn anchor.
- **Monster speed is not cleanly identifiable** from current archives
  (a consequence of the sequential-read limitation).

## Documentation debt (Phase 13 scope, explicit)

- `docs/CONFIGURATION.md`'s `native_farming.json` key-level table was
  **not** individually re-verified against current `farming/config.py`
  during Phase 13 (practical scope limit) — treat it as
  `BEST_CURRENT_ESTIMATE`, not `VERIFIED_CONTRACT`, until re-checked.
- `docs/POINTER_RECOVERY_REFERENCE.md`'s deep recovery-mechanics detail
  was ported into `docs/architecture/POSITION_AND_POINTER_RECOVERY.md`
  as `HISTORICAL_EVIDENCE`/`BEST_CURRENT_ESTIMATE`, not re-derived line
  by line from source this phase (the `position/` package itself is
  confirmed unchanged by every migration phase, which is the basis for
  treating it as still substantively accurate).
- The 16 unregistered `flyff_farming_recorder/position/*.py` files
  (siblings of the 7 formally registered `TEST_CONTRACT_RETIREMENT`
  shims) are not individually listed in `CANONICAL_OWNERS.toml`'s
  `[[shim]]` table, even though they are equally protected by the same
  B2 manifest exact-match test contract — a registry-completeness gap,
  documentation-only, not a functional problem.
