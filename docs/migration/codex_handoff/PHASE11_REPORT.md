# Phase 11 Report — Dependency / Package Boundary + Future Deployment-Derivation Readiness

## 0. Authorization and scope

Authorized: "PHASE 11 — DEPENDENCY / PACKAGE BOUNDARY + FUTURE DEPLOYMENT-
DERIVATION READINESS." Explicitly NOT authorized: Phase 12 and later.
Binding product direction: finish/validate the canonical development bot
first; later derive a stripped deployment/live bot from the SAME
canonical source (never a copied fork). Future deployment readiness in
Phase 11 is STATIC/ARCHITECTURAL only — no `apps/live_bot.py`, no second
Bot implementation, no copied runtime source tree, no standalone build
(PyInstaller/Nuitka/cx_Freeze/installer/dist output), no final-model
selection, no wiring 0051200 into game control, no FlyFF launch, no
G5/G5-P2, no training/refit, no 820M reruns.

## 1. Entry state (verified before any mutation)

- Branch: `refactor/consolidation-phase1` — exact.
- Entry HEAD: `77dc6e518cf415bd75c7b0a2d8c46cc6b17f90b6`, subject "Phase 10
  P10-CORRECTION: report/journal correction after the GUI completion" —
  exact match to the authorization's required entry HEAD.
- Worktree clean, index empty.
- `CANONICAL_OWNERS.toml`: `current_phase = 10`. Phase 10 (with the
  GUI-completion correction) complete.
- Protected tags exact and unchanged throughout this phase:
  `pre-consolidation-head` = `51dc25b2be0aafb091e22a17505767c1bec79552`,
  `historical-reproduction-baseline-20260815` =
  `a90de59232b81753c1b2ea35b8990325c26674e5`, `pre-consolidation-complete`
  = `dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34`.

All entry conditions held; no STOP triggered at entry.

## 2. Final HEAD

`0af2a44ba8aad52ce6dfdd767635b1d434f02799` prior to this report's own
documentation commit; resolve the true final HEAD with `git rev-parse
HEAD` after the P11-DOC commit lands.

## 3. Phase-11 commits

| Commit | Subject | Files |
|---|---|---|
| `07dcca5` | P11-A: dependency/resource/entrypoint audit | `docs/migration/PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md`, `docs/migration/PHASE11_RUNTIME_RESOURCE_MANIFEST.tsv` (new) |
| `c6f1d4f` | P11-B: dependency profile + dry-run resolver | `future_runtime_profile/{__init__.py,dependency_profiles.toml,derive_runtime_manifest.py}` (new) |
| `0af2a44` | P11-C: bootstrap registry + canonical-invocation/derivability/CWD-independence tests + two profile-precision fixes | `docs/migration/tools/phase11_path_bootstrap_registry.py` (new); `tests/test_{canonical_module_invocation,future_derivation_profile,path_bootstrap_registry,phase11_cwd_independence}.py` (new); `future_runtime_profile/{dependency_profiles.toml,derive_runtime_manifest.py}` (modified); `docs/migration/PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md` (modified, forward-reference correction) |
| *(pending)* | P11-DOC: report, journals, `current_phase=11` | this file; `STATE.json`/`HANDOFF.md`/`COMMAND_LOG.tsv`/`TEST_LOG.md`; `CANONICAL_OWNERS.toml` |

No commit was reset, amended, rebased, or force-pushed. No `git add -A`/
`.`/`-u` was used at any point — every commit staged explicit paths.

## 4. First-party module classification

Full table: `PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md` section 1 (~25
rows). Summary:

- **CANONICAL_DEV_APP**: `apps/*.py`, `Bot.py`, `Gui.py`,
  `runtime_controller.py`, `runtime_bus.py`, `worker_manager.py`,
  `capture_service.py`, `preview_service.py`, `project_paths.py`.
- **SHARED_RUNTIME_CORE**: ~22 of ~25 `farming/*.py` files, `position/*`,
  `navigation/*`, most of `mapper/*` (excluding the 3 training files
  below), `libs/*`, `utils/*`, `assets/*`, plus `simulator/schema.py` and
  `legacy/manifest_compat.py` (the canonical archive/recording reader and
  its compat logic — corrected here from an earlier, wrong
  "forbidden"/devtool treatment; see section 10 below).
- **TRAINING_ONLY inside an otherwise-shared package**:
  `farming/{sb3_adapter,sb3_training,trainer}.py`,
  `mapper/rl/{FeatureExtractor,GymEnv,OfflineTraining}.py` — the only
  files under a shared_runtime_packages directory that directly import
  `torch`/`gymnasium`/`stable_baselines3` (confirmed via `git grep`).
- **RUNTIME_ABI_COMPATIBILITY**: `simulator.split_branch_policy`,
  `simulator.kinodynamic_route_planner`, `simulator.movement_kernel` —
  checkpoint-ABI/pickle-compatibility surfaces, not simulator algorithms.
- **DEV_ONLY**: `devtools/*`.
- **RECORDER_ONLY**: `recorder/*`.
- **SIMULATOR_ONLY / TRAINING_ONLY**: the rest of `simulator/*`
  (`environment.py`, `router_waypoint_env.py`, `static_waypoint_env.py`,
  `single_obstacle_env.py`, `synthetic.py`, `basic_training.py`,
  `navigation_dataset.py`, `navigation_history.py`, `cli.py`,
  `fair_time_cli.py`, curriculum/training modules).
- **TRAINING_ONLY (root)**: `RUN_CANONICAL_{ADVANCED,BASIC,BEGINNER,
  INTERMEDIATE}.py`, `run_fair_time_simulator.py`,
  `run_reward_audited_simulator.py`.
- **HISTORICAL_ONLY**: `refactor_logs/`.
- **TEST_ONLY**: `tests/`, `docs/migration/tests/`.
- **PACKAGING_ONLY**: `FlyffFarmingRecorder.spec`.

## 5. Third-party dependency role classification

Full table: analysis doc section 2. `torch`/`gymnasium`/
`stable_baselines3` are classified **DUAL_ROLE**: needed for training AND
for a future runtime's inference path (via `simulator.split_branch_policy`,
an ABI-compatibility module a future derivative may load a checkpoint
through) — never excluded from the future-runtime-candidate profile
merely because training also imports them, per the authorization's
explicit warning. Positive evidence, not assumption: traced
`mapper/rl/__init__.py`'s own top-level imports line-by-line and confirmed
the three heavy training files are reached only via
`mapper/rl/OfflineTraining.py`'s own imports and `train_mapper_offline.py`
directly — never via `mapper/rl/__init__.py`'s PEP 562 lazy dispatch —
so `farming/map_context.py` (canonical shared) stays clean of the
training stack. `numpy`/`opencv-python`/`PySimpleGUI`/`pywin32`/
`pytesseract` etc./`pyfiglet`/`pytest`+`pyinstaller`/`tensorboard`+`rich`/
`msgpack` classified per their actual real, single-purpose role.

## 6. Canonical module invocation

`python -m apps.{dev_app,recorder_app,simulator_cli,telemetry_cli}` all
resolve; established as the non-bootstrap-dependent canonical invocation
form (vs. direct-script invocation, which depends on each file's own
`sys.path` self-bootstrap). Formalized in
`tests/test_canonical_module_invocation.py` (4 tests, all passed):
`apps.dev_app` checked via `importlib.util.find_spec` only — it
constructs a live `Bot()`/`Gui(...)` unconditionally at true module level
(confirmed: `gui = Gui("DarkAmber")` / `bot = Bot()` at `apps/dev_app.py`
lines 21–22, not inside `main()`, not guarded), so it is never actually
imported or executed by any Phase-11 test. `apps.recorder_app` checked
via a plain `import` (its `run_gui()` call is guarded by
`if __name__ == "__main__"`, which is false for a plain import — never
fires). `apps.simulator_cli`/`apps.telemetry_cli` exercised via their
real `-m ... --help` form (both use argparse; confirmed safe and
side-effect-free). No test in this phase opens a real GUI window,
attaches to a live game process, or launches FlyFF.

## 7. sys.path bootstrap inventory

`docs/migration/tools/phase11_path_bootstrap_registry.py`'s
`REGISTERED_BOOTSTRAPS` enumerates 38 files across five categories:
`apps/*.py` (4, Phase-10-added), `devtools/native/*.py` +
`devtools/archives/*.py` + `devtools/calibration/calibration_capture.py`
(6, Phase-10-added dev-tool self-bootstraps),
`RUN_CANONICAL_*.py`/`_basic_round_eval_worker.py`/18 root
`scratchpad_*.py`/`tools/friend_pointer_recovery_test.py` (21,
pre-existing, unrelated to Phase 10/11, unchanged), `tests/conftest.py` +
`tests/test_simulator_core.py` (2, pre-existing top-level test
bootstraps). `docs/migration/**` and `refactor_logs/profiles/*.py`
(historical/dead, zero current references) are explicitly carved out of
registry scope — each already has its own established governance (the
migration-integrity ruler framework; historical-immutability treatment,
respectively). No occurrence resolves to a sibling old worktree, a
`.pth` file, `sitecustomize.py`, or an environment-only `PYTHONPATH`
requirement — every match resolves `Path(__file__)`-relative to a
location inside this same repository. `tests/test_path_bootstrap_
registry.py` (3 tests, all passed) AST-scans every tracked `.py` file
(excluding the two carved-out prefixes) and fails on any new
unregistered `sys.path.insert`/`append` call or any stale registry
entry no longer backed by a real call.

## 8. Machine-readable dependency profile

`future_runtime_profile/dependency_profiles.toml` (schema_version=1):
`[profiles.development]`, `[profiles.recorder]`, `[profiles.simulator]`,
`[profiles.training]`, `[profiles.testing]`, and the primary
`[profiles.future_runtime_candidate]` table (`source_strategy =
"canonical_source"`, `copied_fork = false`, `shared_runtime_packages`,
`excluded_from_shared_entry_walk`, `additional_shared_entry_files`,
`runtime_abi_compatibility_modules`, `candidate_third_party`,
`candidate_runtime_resources`, `forbidden_first_party_prefixes`,
`unresolved_future_choices`, and one
`[[known_exact_exceptions]]` entry for the R1b coupling).

**Not placed under `packaging/`** (the authorization's own suggested
location): a real, well-known PyPI library of that name is a transitive
`pip`/`setuptools`/`pytest` dependency, confirmed installed in this venv
(`site-packages/packaging/__init__.py`, v26.2). Empirically confirmed
Python's import system resolves that regular package (has `__init__.py`)
ahead of a same-named local directory lacking one, regardless of
`sys.path` order — `import packaging` with `sys.path.insert(0, '')`
still resolved to `site-packages/packaging/__init__.py`. `python -m
packaging.derive_runtime_manifest` would therefore have silently run
against the wrong "packaging" entirely. Renamed to
`future_runtime_profile/` (confirmed collision-free: `import
future_runtime_profile` raises `ImportError` before this phase created
it) — invoking the authorization's own escape hatch: "If `packaging/`
does not exist or another location is clearly more coherent, document
and use the source-backed equivalent." The sys.path bootstrap registry
(section 7) needed the same treatment for a different reason: it covers
the whole live app/devtools/scratchpad surface, not specifically the
future-runtime-candidate profile, so it lives in
`docs/migration/tools/` instead, alongside this migration's other
phase-scoped tooling.

## 9. Dry-run derivation resolver

`future_runtime_profile/derive_runtime_manifest.py`: self-contained
static-AST closure walker (deliberately not shared with
`tests/test_dev_app_import_closure.py`'s equivalent — different entry-
point sets, independently verifiable). Builds nothing: no PyInstaller, no
file copy, no dist output — read-only report only.
`python -m future_runtime_profile.derive_runtime_manifest`:

```
FUTURE DEPLOYMENT DERIVATION PROFILE: PASS
  candidate first-party modules: 89
  ABI compatibility modules: ['simulator.split_branch_policy', 'simulator.kinodynamic_route_planner', 'simulator.movement_kernel']
  candidate resources: ['native_farming.json', 'position/native_monsters.json', 'position/native_position.json']
  exceptions applied: ['runtime_controller.py -> farming.trainer']
  forbidden dependency edges: []
  missing tracked files: []
  duplicate ownership issues: []
  unresolved future choices (6)
```

This is a genuine, traced result, not a fabricated one — see section 10
for two real precision bugs found and fixed while building the gate test
around this resolver, both of which changed the candidate-module count.

## 10. Precision fixes found during verification

1. **Training-only files inflating the candidate closure.**
   `farming/{sb3_adapter,sb3_training,trainer}.py` and
   `mapper/rl/{FeatureExtractor,GymEnv,OfflineTraining}.py` were walked
   as unconditional entry points merely because `shared_runtime_packages`
   globs whole directories (`farming/`, `mapper/`) and these files happen
   to live inside them — even though section 4's own classification
   marks them TRAINING_ONLY. This meant `farming.trainer`'s own
   `torch`/`gymnasium`/`stable_baselines3` imports entered the "candidate"
   surface unconditionally, rather than only through the registered R1b
   exact exception from `runtime_controller.py`. Fixed via
   `excluded_from_shared_entry_walk`; candidate-module count 96 → 88;
   re-verified `farming.trainer`/`sb3_training`/`sb3_adapter` do not
   appear in `candidate_first_party_modules`.
2. **`simulator.schema`/`legacy` wrongly forbidden.** The original
   `forbidden_first_party_prefixes` list blocked `simulator.schema` and
   the whole `legacy` package wholesale — but this document's own section
   1 (and section 4 above) classifies both as SHARED_RUNTIME_CORE (the
   canonical archive/recording reader and its compat logic, corrected
   from an earlier devtool/`archives/`-based misreading per Phase-10
   section 1). Neither was actually reached by the pre-fix entry walk, so
   this was a latent misclassification, not an active false failure —
   but it would have wrongly flagged either module as forbidden the
   moment something legitimately imported them. Fixed: removed both from
   `forbidden_first_party_prefixes`, added `additional_shared_entry_files
   = ["simulator/schema.py", "legacy/manifest_compat.py"]` so the
   resolver actively walks and vouches for their own closure (both import
   only stdlib + `msgpack`, confirmed via direct source read) rather than
   merely not-forbidding them by omission. Candidate-module count 88 → 89.
3. **Stale forward-reference.** The analysis doc's section 3 pointed to
   `packaging/path_bootstrap_registry.py` — the authorization's own
   suggested location, written before the packaging/-collision was
   discovered (section 8) and before the registry itself was built.
   Corrected to the registry's actual location (section 7).

## 11. Runtime ABI compatibility modules

`simulator.split_branch_policy`, `simulator.kinodynamic_route_planner`,
`simulator.movement_kernel` — explicitly distinguished from ordinary
shared-runtime algorithm modules. Tracked, present, and (per the
resolver's duplicate-ownership check, re-verified this phase) re-export
only relative to their canonical `navigation/*` owners: `compat_defs`
(class/def names) for both `simulator/kinodynamic_route_planner.py` and
`simulator/movement_kernel.py` is empty — zero routing/movement
implementation lives there. `simulator.split_branch_policy` is the
checkpoint-ABI owner (real architecture/feature-extraction code) and is
not moved — classified RUNTIME_ABI_COMPATIBILITY, not dragged into the
future candidate as a "simulator algorithm."

## 12. R1b exception (unwidened)

Exactly one registered exception, carried forward unchanged from Phase
10: `runtime_controller.py` → `farming.trainer`, for exactly 4 named
symbols (`dry_run_native_farming`, `run_native_farming_agent`,
`train_native_farming`, `validate_native_farming_data`). `git diff HEAD
-- runtime_controller.py` remains empty (confirmed again this phase).
Recorded in the profile with `status =
"KNOWN_DEV_RUNTIME_COUPLING_NOT_PART_OF_FUTURE_DERIVATION_CONTRACT"` —
explicitly not assigned to any phase for redesign. `tests/test_future_
derivation_profile.py::test_point_08` asserts exactly one exception is
applied, never silently expanded.

## 13. Runtime resource manifest

`docs/migration/PHASE11_RUNTIME_RESOURCE_MANIFEST.tsv` (21 rows,
committed P11-A, unchanged this phase): `native_farming.json`,
`position/native_{monsters,position}.json` (SHARED_RUNTIME_CORE inputs),
`recorder_config.json` (RECORDER_ONLY), `foreground_vision_farm.json`
(confirmed orphaned — zero `.py` references — flagged as an
`unresolved_future_choice`, not deleted), `recording_provenance.json`
(Phase-8 legacy-attestation registry), the Tower `map_assets/` /
`mapper/maps/tower_aoe/` byte-identical pairs (Phase-2/6 frozen source,
confirmed via SHA-256), OCR/UI-detection/minimap template images
(retained conditionally on future vision-vs-native-reading choice), the
0051200 checkpoint (flagged `unresolved_future_choice` for final-model
selection — not decided this phase), `project_paths.py`/
`devtools/session_context.py` directory-root rows, and
`FlyffFarmingRecorder.spec`'s `datas` entries (PACKAGING_ONLY,
unaffected).

## 14. Working-directory / path independence

`tests/test_phase11_cwd_independence.py` (5 tests, all passed):
`project_paths.APP_ROOT`/`resolve_app_path`, the new
`derive_runtime_manifest.REPO` and `derive()` call, and (re-confirmed
within this phase's own test plan) `devtools.session_context.
resolve_session_context` all resolve identically when invoked from a
subprocess whose CWD is an unrelated temp directory — none assume
`os.getcwd() == repo root`.

## 15. Dependency-direction result

Shared/core → devtools/recorder-impl/simulator-training/historical-
compat/test-code: forbidden, enforced by
`tests/test_future_derivation_profile.py::test_point_02` (zero forbidden
edges) and `test_point_03` (zero `simulator.*` modules outside the ABI
set and the now-explicitly-included `simulator.schema`), covering the
full `shared_runtime_packages` + `additional_shared_entry_files` closure
— a repository-wide generalization of the same one-way guard the
pre-existing `tests/test_navigation_dependency_boundary.py` (navigation-
scoped) and `tests/test_devtools_dependency_direction.py` (devtools-
scoped) already establish. devtools/recorder/simulator/training →
shared/core: allowed (one-way), unaffected.

## 16. Future-derivability gate

`tests/test_future_derivation_profile.py`: 12 individually named proof-
point tests (single ownership of farming/position/navigation; zero
forbidden dev/recorder/simulator-training/scratchpad edges; simulator
training/environment absent from the shared closure; ABI-compatibility
modules distinguished and re-export-only; no copied fork declared;
profile resolves only tracked source; resolution self-contained to this
worktree — no old preservation worktree needed; exactly one registered
exception; DUAL_ROLE third-party not excluded; candidate resources
tracked/present; bundled training-only files excluded from the
candidate; overall `report.ok is True`), all passed. This is a
static/architectural result: it does not mean a runtime derivative
exists, is ready, or has been built.

## 17. No copied fork / no standalone build

Confirmed by direct action, not just profile declaration: no
`apps/live_bot.py` was created, no second `Bot` implementation exists, no
runtime source tree was copied anywhere, no PyInstaller/Nuitka/
cx_Freeze/installer was run for a future live bot, no dist/ output was
produced. `source_strategy = "canonical_source"` / `copied_fork = false`
in the profile reflect what was actually done, not merely what was
declared.

## 18. R10 / checkpoint ABI

Not reloaded this phase — no product/import-path code affecting ABI was
changed (all Phase-11 changes are under `future_runtime_profile/`,
`docs/migration/`, and new `tests/` files; `git diff` against every
ABI-relevant module — `runtime_controller.py`, `simulator/split_branch_
policy.py`, `simulator/kinodynamic_route_planner.py`,
`simulator/movement_kernel.py` — is empty). Per the authorization's own
Section 18 condition, the ruler's R10 result is sufficient:
`r10_failures: []`, `r10_checkpoint_count: 313`,
`r10_module_reference_rows: 317` (unchanged from Phase 10's final
verification).

## 19. Ruler result

`migration_integrity.py check`: `ok: true`, `R6=0 R7a=0 R7b=0 R9=0
R10=0`, `R7c` baseline count `204` (unchanged from Phase 10's final
verification — no unexplained growth; every new R7c-relevant entry this
phase, if any, is accounted for by the same pre-existing re-export
pattern already ratcheted).

## 20. Full test-suite result

`pytest tests/`: **1190 passed** (1166 baseline + 24 new Phase-11
tests), 2 skipped, 1 xfailed, **4 failed** — the identical 4 pre-
existing/unrelated failures as every prior phase
(`test_farming_environment_lifecycle.py::
test_focus_loss_during_eva_discards_kill_and_transition`,
`test_farming_training_session.py::
test_normal_training_status_is_concise_and_uses_total_model_steps`,
`test_farming_training_session.py::
test_training_callback_publishes_structured_session_statistics`,
`test_navigation_dataset.py::
test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts`
— the last a pre-existing gitignored-artifact gap,
`models/split_branch_pilot_15000.zip` vs. the test's own
`.zip.zip`-suffixed lookup) — zero new failures. `pytest docs/migration/
tests/`: **76 passed**. `git diff --check`: clean.

## 21. Scientific/historical immutability

No frozen evidence file was touched: `BASELINE_VIOLATIONS.json`/`.md`
untouched (forward supplements only, per the established
`DEFAULT_SUPPLEMENTS` pattern — no new supplement was needed this
phase); `docs/migration/CHECKPOINT_INVENTORY.tsv` (313 rows) untouched;
`refactor_logs/` untouched (documented HISTORICAL_ONLY, not reorganized);
no root `scratchpad_*.py` file was moved, renamed, or bulk-reorganized
(Section 16 of the authorization).

## 22. Protected refs / worktree / branch status

`pre-consolidation-head`, `historical-reproduction-baseline-20260815`,
`pre-consolidation-complete` all confirmed unchanged (section 1 values,
re-verified at report time). Worktree clean, index empty. Branch
`refactor/consolidation-phase1`, no upstream, absent from `origin` — no
push performed.

## 23. G5/G5-P2 / training / live launch

None run, none attempted. No training, no refit, no 820M rerun, no FlyFF
launch, no live game-window attach.

## 24. Explicit no-standalone-bot-built statement

No standalone or live bot was built, packaged, or made runnable this
phase. `future_runtime_profile/` produces a **read-only report**, never
an artifact. Nothing under `models/`, `dist/`, or any new packaging
output was created or modified for a future runtime.

## 25. `PHASE 12` implication

**PHASE 12 SAFE TO CONSIDER: YES** — the dependency/package boundary and
future-derivation profile are now static-verified and gate-tested.
**PHASE 12 AUTHORIZED: NO** — not requested, not begun, no code or docs
for it exist.

## 26. Final conclusion

**PHASE 11 COMPLETE: YES.**
