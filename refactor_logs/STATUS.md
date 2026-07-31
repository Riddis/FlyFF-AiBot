# Refactor Status

- Phase: automated anchored current-client discovery complete; awaiting the single `PTR-LIVE-001` Win32 movement-correlation protocol.
- Branch: `feature/adaptive-mapper`.
- Validated anchored implementation checkpoint: `84559c6ce6ff63a86604a4c71aff8ae2308cdb98`; this journal reconciliation follows it.
- Protected pre-refactor commit: `174208614c7c8a916bd7c0dce5cbbb5f2a4e5239` through `protected/pre-codex-refactor` and `backup/pre-codex-refactor`.
- Production route: `runtime_controller.py -> farming.trainer -> UnifiedFarmingEnv / SessionAwarePPO`.
- Active model: `models/farming/native_strategy_ppo.zip`; SHA-256 `3ACB0437EA1B7F7BF42DFCDF4DA3B4C097540A702EC856F5AA59BA2D76FADFF2` unchanged.
- Latest automated acceptance: 545 passed, 1 skipped in 9.27 seconds.
- Quality: changed production/test scope passes Ruff F/I; native production BasedPyright reports zero errors (existing warning-level typing debt remains); diff hygiene passes.
- Live evidence retained: `Neuz.exe` base `0xB0000`, size `0x943000`, 32-bit; all 4,096 legacy candidates rejected at the historical self field. GUI responsiveness, cancellation, no-input startup failure, Stop, and shutdown passed.
- Implemented next strategy: selected species consensus, inferred current world/self actor fields, Tower spawn `(253.0, 86.0)` plus exact current/max HP ranking, stable direct/one-hop module references, and mandatory second-sample movement confirmation.
- Safety boundary: ordinary reads and Native Health are scan-free; recovery is managed, bounded, cancellable, single-flight, and off Tk. The first anchored sample cannot apply or persist anything. Only exact HP plus coherent movement and stable repeated reads authorize transactional persistence.
- Diagnostic extension: legacy self mismatches receive up to 1,024 read-only world/reference/coordinate/HP/player near-match probes; those counters cannot promote a rejected candidate.
- User-owned deletions remain excluded. `foreground_vision_bot.zip` and `refactor_logs.zip` remain ignored and untouched.
- Open task: `PTR-LIVE-001` only. Run `refactor_logs/manual_tests/PTR-LIVE-001_current_client_pointer_acceptance.md` and return the requested log/config evidence.
