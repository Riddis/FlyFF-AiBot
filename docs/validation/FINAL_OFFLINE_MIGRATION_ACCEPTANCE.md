# Final Offline Migration Acceptance — Phase 14

- Commit under test (working tree at time of evidence gathering): `3c9e12f0dab3022e882b1813adc4037edc8576ce`, then all Phase-14 changes described below (final HEAD recorded in `PHASE14_REPORT.md` once committed).
- Branch: `refactor/consolidation-phase1`. Unpushed, no upstream, not on origin.
- Scope: **migration/consolidation completeness only.** This record does NOT assert G5, G5-P2, live-client validation, live training, deployment readiness, or overall project completion. Those remain out of scope by design (see `docs/migration/codex_handoff/PHASE14_REPORT.md` section 33).
- Exclusions: no FlyFF launch, no client attach, no live training, no G5/G5-P2 execution. `apps/recorder_app.py`'s GUI entrypoint was deliberately NOT run to completion after it was observed to block in a GUI event loop rather than exit on `--help` (see section 9 below) — stopped via `TaskStop` before any window could plausibly persist, verified safe by source inspection afterward instead.

## 1. Capability conservation

`docs/migration/PHASE14_CAPABILITY_AUDIT.tsv` (73 rows, categories A–J) and `docs/migration/PHASE14_FINAL_PRODUCT_ANALYSIS.md` (narrative) — every pre-consolidation capability traces to a positively-identified current disposition (`CURRENT_CANONICAL`, `DEV_TOOL`, `TRAINING_ONLY`, `HISTORICAL_COMPATIBILITY`, `ABI_COMPATIBILITY`, `USER_RUN_LIVE_VALIDATION_PENDING`). Zero rows show "no current owner." Result: **PASS**.

## 2. Legacy-root residue

39 → 35 tracked files remain under `foreground_vision_bot/` (9), `flyff_farming_recorder/` (25), `flyff_farming_simulator/` (1) — all with a final, non-ambiguous disposition (`COMPATIBILITY_REQUIRED` shims gated by `test_phase4_contracts`/`test_phase5_contracts`-equivalent tests below, or `ARCHIVE_AS_HISTORY`/`PROJECT_WIDE_FILE_IN_OLD_LOCATION` for the two moved/superseded files). Zero `AMBIGUOUS_BLOCKER`. Result: **PASS**.

## 3. Two deferred collision files — final resolution

- `flyff_farming_recorder/requirements.txt` → content (`msgpack>=1.0`, `pywin32>=306`) merged into root `requirements.txt` (`msgpack` was a genuine migration gap: a `DUAL_ROLE` dependency of `simulator/schema.py`/`recorder/*` never declared at canonical root). File removed (`git rm`).
- `foreground_vision_bot/foreground_vision_farm.json` → proven orphaned by mechanism: `PySimpleGUI`'s `user_settings_filename(path=".")` derives the settings filename from `sys.modules["__main__"].__file__`'s basename; under the current entrypoint (`apps/dev_app.py`) this always produces `dev_app.json`, never `foreground_vision_farm.json`. Confirmed via direct `inspect.getsource` of `UserSettings._compute_filename`. File removed (`git rm`).

Neither required a live/product decision — both resolved with positive offline proof. Result: **PASS**.

## 4. MISTAKES.md relocation

`git mv flyff_farming_simulator/MISTAKES.md MISTAKES.md` — history preserved as a rename, content byte-identical. Zero programmatic path dependencies (confirmed via `git grep` for `open`/`Path`/`read_text` against the filename — none). Seven prose-reference files updated mechanically. Result: **PASS**.

## 5. Stale-document audit

`docs/{ARCHITECTURE,CONFIGURATION,RUNBOOK,POINTER_RECOVERY_REFERENCE}.md` and `flyff_farming_simulator/README.md` each gained a self-identifying banner (SUPERSEDED or, for `POINTER_RECOVERY_REFERENCE.md`, a lighter "prior-generation detail, largely still accurate" note) linking to the current canonical doc. No content below any banner was altered. Result: **PASS**.

## 6. Full offline product test suite (Section 16)

Command: `pytest tests/` (exit code captured directly via `$?`, never through a `tail`/`head` pipe).

First run (before this phase's two test-harness fixes): `PYTEST_EXIT_CODE=1`, `5 failed, 1201 passed, 2 skipped, 1 xfailed`.

Investigation found:
- Two of the five failing node IDs (`test_farming_training_session.py::test_normal_training_status_is_concise_and_uses_total_model_steps`, `test_farming_training_session.py::test_training_callback_publishes_structured_session_statistics`) were a **test-harness gap, not a product defect**: both call `_TrainingCallback._on_step()` directly, bypassing `stable_baselines3.common.callbacks.BaseCallback.on_step()` — the real SB3 call path, which syncs `self.num_timesteps = self.model.num_timesteps` before invoking `self._on_step()` (confirmed via `inspect.getsource(BaseCallback.on_step)`). The fakes set `callback.model.num_timesteps` but never `callback.num_timesteps`, so the production code (`farming/trainer.py`, correctly reading `self.num_timesteps`) always saw `0`. `farming/trainer.py` has zero diff since the Phase-7 collapse commit (`bfc5c6d`) — this predates every consolidation phase. **Fixed**: both tests now set `callback.num_timesteps = callback.model.num_timesteps` before calling `_on_step()`, replicating what SB3's real `on_step()` does. Both now pass.
- A fifth (`test_check_project_knowledge.py::test_current_repository_passes_every_check`) was self-inflicted by this phase's own not-yet-written `PHASE14_REPORT.md` forward-reference in `docs/KNOWN_DEBT.md`/`docs/architecture/SYSTEM_OVERVIEW.md`. Self-resolves once that report exists (see section 9 below).

Intermediate run (after the two test fixes, before `PHASE14_REPORT.md` existed): **3 failed, 1203 passed, 2 skipped, 1 xfailed** — the self-inflicted forward-reference plus 2 of the original 4 established failures. Official final run (after `PHASE14_REPORT.md` and `docs/validation/README.md`'s link were in place, resolving the self-inflicted failure): **2 failed, 1204 passed, 2 skipped, 1 xfailed** — exactly the 2 remaining established failures, both fully diagnosed below, zero other failures. See section 7 for the individual audit of the 4 originally-established failures.

## 7. Four-failure audit (Section 17)

| Node ID | Classification | Evidence |
|---|---|---|
| `test_navigation_dataset.py::test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts` | `PRE_EXISTING_ENVIRONMENTAL/ARTIFACT` | `FileNotFoundError: models\split_branch_pilot_15000.zip.zip` — the test's own path construction double-appends `.zip`. Pre-existing across every prior phase's baseline; not touched by migration. Not fixed this phase (Section 29: not a migration defect, not narrowly in scope without touching the test's path-construction logic more broadly than a one-line change would responsibly cover under this audit's time budget). |
| `test_farming_training_session.py::test_normal_training_status_is_concise_and_uses_total_model_steps` | `RESOLVED_OFFLINE_NOW` | Root-caused (see section 6): test bypassed SB3's `on_step()` sync wrapper. Fixed this phase; now passing. |
| `test_farming_training_session.py::test_training_callback_publishes_structured_session_statistics` | `RESOLVED_OFFLINE_NOW` | Same root cause as above. Fixed this phase; now passing. |
| `test_farming_environment_lifecycle.py::test_focus_loss_during_eva_discards_kill_and_transition` | `REAL_PRODUCT_DEFECT` (masked by a separate, now-fixed test-fake bug) | Two layers found. (a) The test's own fake `FocusDroppingKillTracker.begin_cast` returned a bare `object()` instead of a `CastWindow`, causing `farming/environment.py:1192`'s `cast_window.candidates` access to raise `AttributeError` before the test's real assertion was ever reached — fixed by returning `CastWindow(0.0, ())`, a minimally-shaped real instance. (b) With that masking bug removed, the test's actual assertion (`step.info["native_kill_delta"] == 0`, i.e. a kill confirmed while focus was lost during the EVA cast should be discarded) now fails for real: `1 == 0`. Source inspection of `farming/environment.py`'s `step()` (lines ~800–846) shows the EVA/cast branch calls `self.kill_tracker.confirm_cast(...)` and then unconditionally `break`s (line 837) — it never re-checks `self.control.is_target_foreground()` after `confirm_cast()` returns, unlike the movement-only branch (line 840), which does. `farming/environment.py` and this test have zero diff since the Phase-7 collapse commit (`bfc5c6d`) — this is a genuine, pre-existing gap between the test's stated intent and the implementation, not a migration regression. **Not fixed this phase**: implementing "discard a kill confirmed during a focus-loss window" requires a product decision among multiple valid designs (re-check focus after `confirm_cast`; track focus state across the whole cast window; a different session-transition outcome) and touches live-control-adjacent code — squarely a Section 29/30 stop-and-defer case, not a narrow offline-safe fix. |

Net effect: of the 4 originally-established failures, 2 are now resolved (test-harness fixes, zero product-code change), 1 remains a documented pre-existing environmental artifact, and 1 remains a documented pre-existing real product defect — now understood far more precisely than "known," with the masking bug in its own test fixed as a genuine, narrow improvement. No new (5th, non-self-inflicted) failure was introduced. No established failure's node ID was replaced with a different one.

## 8. Offline functional gates (Section 18)

| Gate | Result |
|---|---|
| Knowledge checker (`python tools/check_project_knowledge.py`) | 8/9 PASS; only failure is the expected `PHASE14_REPORT.md` forward-reference, self-resolving once written |
| `docs/migration/tests/` | 77 passed, 0 failed |
| Ruler (`migration_integrity.py check`) | `ok=true`, `R6=0 R7a=0 R7b=0 R7c=204 R9=0`, `r10_failures=[]` — exact match to entry-state baseline |
| Future deployment derivation profile | PASS, 89 candidate modules, 0 forbidden edges, 0 missing files, 0 duplicate-ownership issues, 1 exception (`runtime_controller.py -> farming.trainer`) |
| Checkpoint fresh-process load | `PPO.load("models/generalized_waypoint_both_seed2_0051200.zip")` succeeds in a fresh interpreter; SHA-256 `87bd8d3e0be88b7f243ad6c9b35ff6d3f8bde1f37b35334febf936ec115cda50` exact match |
| Checkpoint ABI | policy `simulator.split_branch_policy.SplitSteeringNavigationPolicy`, observation `Box(-1.0, 1.0, (928,), float32)`, action `MultiDiscrete([3 3])` — exact match |
| Archive parity (`test_archive_schema_legacy_compat.py`) | passed |
| Map hashes | occupancy `62fa3c9ec3aed0b3b134b82577292c0a8a67b0acc4111fde3a36e3d2684d789b`, map.json `faaf8633457bc1bcdb61c781c8ca62c6f2e008174ed5b284c3d6c08df92fe815`, coordinate_frame `40339f6c397d38fe01d5b3a5300e5b9b6d499f06292f436b1f91ea34523a0414` — exact match to all three expected values |
| Live/sim map profile distinction | `LIVE_TOWER_PROFILE(obstacle_radius_cells=2, teleport_radius_cells=2.0)` vs `SIM_TOWER_PROFILE(obstacle_radius_cells=0, teleport_radius_cells=2)` confirmed distinct in `farming/map_profile.py`, consumed separately by `farming/map_context.py` and `simulator/map_model.py` |
| Navigation/movement contract tests (`test_movement_kernel.py`, `test_movement_classification.py`, `test_navigation_history.py`, `test_navigation_dependency_boundary.py`, `test_pure_navigation_env.py`) | 85 passed (combined with map tests below) |
| Map contract tests (`test_farming_map_features.py`, `test_farming_map_masks.py`, `test_map_persistence.py`, `test_map_catalog_management.py`, `test_local_navigation_features.py`) | included in the 85 above |
| Dev-app/devtools tests (`test_devtools_gui_tools.py`, `test_gui_devtools_wiring.py`, `test_devtools_process_orchestrator.py`) | passed |
| Recorder/position/native tests (`test_recorder_core.py`, `test_native_position_provider.py`, `test_native_monster_provider.py`, `test_position_config.py`, `test_position_factory.py`, `test_recovered_native_profile.py`) | passed |
| Pickle module-identity ABI (`test_pickle_module_identity_compat.py`) | passed |
| Canonical entrypoint/CWD tests (`test_canonical_module_invocation.py`, `test_phase11_cwd_independence.py`) | passed |
| Clean-repo-snapshot tests (`test_create_clean_repo_snapshot.py`) | passed |
| Farming model/observation contract (`test_farming_model_contract.py`, `test_farming_observation_contract.py`) | passed |
| `git diff --check` | clean, exit 0 |

## 9. Dev application final assembly (Section 19) and user-facing entrypoints (Section 20)

`apps/dev_app.py` constructs `gui = Gui("DarkAmber")` and `bot = Bot()` at **module level** (lines 21–22), not inside `main()` — confirmed via source inspection (`if __name__ == "__main__":` guards only `main()` at line 55, not the module-level construction). This means `apps/dev_app.py` is not safely importable in an automated test/audit process without constructing live GUI/Bot objects; this was not redesigned for test aesthetics (Section 19's explicit instruction), only documented accurately, matching Phase 9–11's prior findings.

Entrypoint audit, offline-safe only:
- `python -m apps.simulator_cli --help` → prints usage (subcommands: `validate-recording`, `build-model`, `inspect`, `smoke-test`, `benchmark`, `export-demos`, `compare-policies`, `train`, `generate-synthetic`, `inspect-synthetic`, `smoke-test-synthetic`, `train-synthetic`, `evaluate-synthetic`). Confirmed safe and functional.
- `python -m apps.telemetry_cli --help` → prints usage (`--window-title`/`--hwnd`, `--map`, `--species`, `--vision-radius-native`, `--sample-interval-seconds`, ...). Confirmed safe and functional.
- `python -m apps.recorder_app --help` → **did not exit**; the command was killed via `TaskStop` after it produced no output within its timeout. Source inspection (`apps/recorder_app.py`) explains why: the module has no argparse/`--help` handling at all — `if __name__ == "__main__":` calls `recorder.gui.run_gui()` unconditionally, which blocks in a GUI event loop. This is a genuine, newly-documented fact about this entrypoint (it does not attach to FlyFF — the recorder GUI is passive — but it is not `--help`-probeable, and any offline audit of it must rely on source inspection, never direct invocation without a bounded timeout and pre-armed kill path, exactly as was done here).

## 10. Position offline gates (Section 25)

`test_recovered_native_profile.py`, `test_native_position_provider.py`, `test_native_monster_provider.py`, `test_position_config.py`, `test_position_factory.py` all passed — offline-only (no native process attach). G5 and G5-P2 remain **PENDING**; nothing here is elevated to "live validated."

## 11. Artifact immutability (Section 26)

No Phase-14 change touched: checkpoints (`models/*.zip` — only read for hashing/loading), recordings, `mapper/maps/tower_aoe/*` (only read for hashing), calibration CSVs, historical evaluation results, Phase-3 goldens, the frozen ruler baseline, the historical reproduction snapshot, or any protected tag. Protected tags re-verified unchanged: `pre-consolidation-head=51dc25b2be0aafb091e22a17505767c1bec79552`, `historical-reproduction-baseline-20260815=a90de59232b81753c1b2ea35b8990325c26674e5`, `pre-consolidation-complete=dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.

## 12. Remaining live-validation dependencies

G5 (live-client validation) and G5-P2 (conditional live follow-on) remain **PENDING** — untouched, unattempted, no agent live execution occurred at any point in this phase.

## 13. Remaining intentional compatibility

9 `foreground_vision_bot/farming/*.py` shims + 25 `flyff_farming_recorder/position/*.py` shims and resources remain `COMPATIBILITY_REQUIRED`, gated by their respective contract test suites (unchanged this phase). 3 permanent ABI-compatibility modules (`simulator.split_branch_policy`, `simulator.kinodynamic_route_planner`, `simulator.movement_kernel`) remain distinct from the 16 conditional shims — unchanged this phase.

## 14. Known debt (unchanged or newly documented this phase)

- GUI settings persistence is CWD-dependent by construction (`docs/architecture/SYSTEM_OVERVIEW.md` section 3a, `docs/KNOWN_DEBT.md`) — newly documented, intentionally not fixed (needs a product decision).
- `test_focus_loss_during_eva_discards_kill_and_transition`'s real assertion failure (section 7 above) — newly precisely diagnosed, intentionally not fixed (needs a product decision).
- `test_mine_navigation_dataset...`'s double-`.zip` path bug — pre-existing, unchanged.
- `apps/recorder_app.py` has no `--help`/argparse handling — newly documented.

## 15. Confidence

**HIGH.** Every claim in this document is backed by a directly-executed command or direct source inspection captured in this session; no claim is carried forward from memory without re-verification.

## 16. Final conclusion

Migration/consolidation is **COMPLETE** for the scope defined in `docs/migration/codex_handoff/PHASE14_REPORT.md`. Overall project completion is explicitly **NOT** claimed — G5, G5-P2, and all future model/deployment work remain outstanding. See `docs/migration/PHASE14_FINAL_PRODUCT_ANALYSIS.md` for the full narrative and `docs/migration/codex_handoff/PHASE14_REPORT.md` for the complete phase account.
