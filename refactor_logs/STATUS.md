# Refactor Status

- Phase: automated refactor complete; consolidated live-client acceptance pending.
- Branch: `feature/adaptive-mapper`.
- Validated code checkpoint: `9cbc1f938d2b7f528b5962ec466f06459e6063e4`; the final journal-only checkpoint follows it.
- Protected pre-refactor commit: `174208614c7c8a916bd7c0dce5cbbb5f2a4e5239` through immutable tag `protected/pre-codex-refactor` and branch `backup/pre-codex-refactor`.
- Production route: `runtime_controller.py -> farming.trainer -> UnifiedFarmingEnv / SessionAwarePPO`.
- Legacy farming monkeypatch, movement PPO, target/orbit/navigation, patch backup, installer, and generated-log paths: removed.
- Active model: `models/farming/native_strategy_ppo.zip`; SHA-256 `3ACB0437EA1B7F7BF42DFCDF4DA3B4C097540A702EC856F5AA59BA2D76FADFF2` unchanged.
- Final automated test: 532 passed, 1 skipped in 8.60 seconds.
- Quality: canonical farming/native/runtime changed scope compiles and passes Ruff F/I plus error-level BasedPyright. Repository-wide legacy mapper/test import-order debt and PySimpleGUI typing debt are classified, not introduced by this refactor.
- Static gates: no versioned farming patch or removed navigator/executor production reference; no movement model file; both protected refs peel to the protected SHA.
- Tree runnable: yes under automated/fake integration coverage.
- User-owned changes excluded: deleted root `AGENTS.md`, `README.md`, and `foreground_vision_farm.json`.
- User backups ignored and untouched: `foreground_vision_bot.zip`, `refactor_logs.zip`.
- Blocker: only real FlyFF/Tk/Win32 behavior and the external session edge remain unverified.
- Next action: run `refactor_logs/manual_tests/DOC-003_live_client_acceptance.md` and return its requested logs/reports.
