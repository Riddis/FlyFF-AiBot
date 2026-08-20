# Phase 08 — Final Validation

Status: automated acceptance complete; live-client protocol pending.
# Documentation and live-validation boundary

`foreground_vision_bot/ARCHITECTURE.md`, `RUNBOOK.md`, and `CONFIGURATION.md`
document the canonical production path, ownership, operations, recovery, and
migrations. `manual_tests/DOC-003_live_client_acceptance.md` consolidates the
remaining real FlyFF/Tk/Win32 checks. No live result is claimed by automated
evidence.

Final result: 532 passed, 1 skipped in 8.60 seconds. Canonical changed scope
compiles and passes Ruff F/I and error-level BasedPyright. Production searches
find no versioned farming patches or removed navigator/executor dependencies;
the movement-model directory is absent, both protected refs peel to the
protected SHA, and the active model hash is unchanged. Phase 08 is complete
apart from the explicitly human-only live protocol.
