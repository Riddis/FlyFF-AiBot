# Phase 11 — Dependency / Package Boundary + Future Deployment-Derivation Readiness: Analysis

Audit performed against the actual collapsed repository at HEAD
`77dc6e518cf415bd75c7b0a2d8c46cc6b17f90b6`, tracing real import edges
(`git grep` + direct source reads), not directory-name assumptions.

## 1. First-party module/package classification

| Path | Classification | Evidence |
|---|---|---|
| `apps/dev_app.py` | CANONICAL_DEV_APP | Entrypoint constructing `Gui`+`Bot`; Phase-10 canonical dev-app entrypoint |
| `apps/recorder_app.py` | CANONICAL_DEV_APP (recorder launcher) | Thin `recorder.gui.run_gui()` launcher |
| `apps/simulator_cli.py` | CANONICAL_DEV_APP (simulator launcher) | Thin `simulator.cli.main()` launcher |
| `apps/telemetry_cli.py` | CANONICAL_DEV_APP (telemetry launcher) | Thin CLI over `devtools.telemetry.observation_telemetry` |
| `Bot.py`, `Gui.py`, `runtime_controller.py`, `runtime_bus.py`, `worker_manager.py`, `capture_service.py`, `preview_service.py`, `project_paths.py` | CANONICAL_DEV_APP | Confirmed clean of `simulator.*`/`recorder.*`/`legacy.*`/`torch`/`gymnasium`/`stable_baselines3` (Phase-10 `test_dev_app_import_closure.py`), except the one registered `runtime_controller.py`→`farming.trainer` exception (section 5) |
| `farming/` (all files except `sb3_adapter.py`, `sb3_training.py`, `trainer.py`) | SHARED_RUNTIME_CORE | Canonical shared observation/action/reward/session contract, imported by `simulator`, `navigation` (indirectly via `farming.actions`/`farming.map_features`), `devtools.telemetry`, and the dev app |
| `farming/sb3_adapter.py`, `farming/sb3_training.py`, `farming/trainer.py` | TRAINING_ONLY (dual-role dependency host, see section 2) | Only these 3 of ~25 `farming/*.py` files import `torch`/`gymnasium`/`stable_baselines3` directly (confirmed via `git grep`); `trainer.py` is reached from `runtime_controller.py` only through the one registered exact exception |
| `position/` | SHARED_RUNTIME_CORE | Confirmed clean of `torch`/`gymnasium`/`stable_baselines3`; native attach/read mechanism used by `Bot.py`, `devtools.telemetry`, `devtools.native.*` |
| `position/profiling/` | DEV_ONLY | Phase-5-established dev/recording-only profiling layer, absent from the live import closure by design |
| `navigation/` (`kinodynamic_route_planner.py`, `movement_kernel.py`, `movement_kinematics.py`, `navigation_evidence.py`, `map_protocol.py`) | SHARED_RUNTIME_CORE | Phase-9's canonical navigation package; confirmed clean of `gymnasium`/`stable_baselines3`/`torch`/`recorder`/training-only `simulator.*` (Phase-9 `test_navigation_dependency_boundary.py`, still passing) |
| `mapper/` (all files except `mapper/rl/{FeatureExtractor,GymEnv,OfflineTraining}.py`) | SHARED_RUNTIME_CORE | `mapper.rl.LayoutSources`/`OpenArena`/`PolicyTypes`/`SimulatorCore`/`ActionMask`/`Observation`/`ProceduralDungeon` (reached via `mapper/rl/__init__.py`, itself reached by `farming/map_context.py`'s `from mapper.rl.LayoutSources import load_real_map`) traced line-by-line: confirmed clean of `torch`/`gymnasium`/`stable_baselines3` |
| `mapper/rl/FeatureExtractor.py`, `mapper/rl/GymEnv.py`, `mapper/rl/OfflineTraining.py` | TRAINING_ONLY | Import `torch`/`gymnasium` directly; reached only by `mapper/rl/OfflineTraining.py`'s own imports and `train_mapper_offline.py` — confirmed **not** imported by `mapper/rl/__init__.py`, so not reachable through `farming.map_context`'s import path |
| `recorder/` | RECORDER_ONLY | Canonical writer; confirmed clean of `simulator.*`/`torch`/`gymnasium`/`stable_baselines3`/`devtools.*` |
| `simulator/schema.py` | SHARED_RUNTIME_CORE (archive reader) | Corrected per Phase-10 §1: the actual canonical archive/recording reader (not `archives/`, which does not exist) |
| `legacy/manifest_compat.py` | SHARED_RUNTIME_CORE (compat logic behind the reader) | Genuinely historical, non-dataclass compatibility logic; canonical, not a devtool |
| `simulator/split_branch_policy.py` | RUNTIME_ABI_COMPATIBILITY | See section 5 — checkpoint ABI, not simulator algorithm |
| `simulator/kinodynamic_route_planner.py`, `simulator/movement_kernel.py` | RUNTIME_ABI_COMPATIBILITY | Phase-9 permanent, behavior-free pickle re-export shims — see section 5 |
| `simulator/` (everything else: `environment.py`, `router_waypoint_env.py`, `static_waypoint_env.py`, `single_obstacle_env.py`, `synthetic.py`, `basic_training.py`, `navigation_dataset.py`, `navigation_history.py`, `cli.py`, `fair_time_cli.py`, curriculum/training modules) | SIMULATOR_ONLY / TRAINING_ONLY | Training/environment implementation; never imported by the dev-app closure (Phase-10 `test_dev_app_import_closure.py` `DISALLOWED_MODULE_PREFIXES`) |
| `devtools/` (all files) | DEV_ONLY | Phase-10's own package; confirmed zero canonical package imports it (`tests/test_devtools_dependency_direction.py`) |
| `libs/`, `utils/`, `assets/` | SHARED_RUNTIME_CORE (dev-app-facing utility) | `win32*`/`cv2` wrappers and static mob data consumed by `Bot.py`/`Gui.py`/`runtime_controller.py`/`mapper.*` |
| `scratchpad_*.py` (~110 root files) | HISTORICAL_ONLY / TEST_ONLY (mixed, per-file) | Not bulk-reclassified this phase — Phase-9's finding (some are frozen historical evidence via `scratchpad_historical_reproduction_guard.REQUIRED_FILES`, some are test-owned-harness sources, most are ordinary completed-investigation debris) still holds; see section 6 |
| `RUN_CANONICAL_{ADVANCED,BASIC,BEGINNER,INTERMEDIATE}.py`, `run_fair_time_simulator.py`, `run_reward_audited_simulator.py` | TRAINING_ONLY | Root training-orchestration entrypoints; Phase-10 `DEFER_PHASE13`, unchanged this phase |
| `refactor_logs/` | HISTORICAL_ONLY | Pre-Phase-0 refactor archive (own `STATE.json`/`HANDOFF.md`/`CHANGES.jsonl`); confirmed zero references from any current active source (`git grep` outside itself) |
| `tests/`, `docs/migration/tests/` | TEST_ONLY | Test suites |
| `docs/migration/tools/` | TEST_ONLY (migration tooling) | Standalone migration-integrity tooling, deliberately Torch-free by its own design |
| `FlyffFarmingRecorder.spec`, `build_recorder_exe.ps1`, `build_recorder_installer.ps1`, `FlyffFarmingRecorderInstaller.iss`, `uninstall_recorder.ps1` | PACKAGING_ONLY | Recorder packaging, mechanically updated in Phase 10, not touched this phase (section 15 of the authorization) |
| `tools/friend_pointer_recovery_test.py` + its packaging quintet | DEV_ONLY / PACKAGING_ONLY | Left in `tools/` by Phase-10's own evidenced decision (own hardcoded packaging path) |

No `AMBIGUOUS_STOP` classification was required — every module traced to a
determinate role via actual import-edge evidence.

## 2. Third-party dependency role classification

| Package | Role(s) | Evidence |
|---|---|---|
| `torch`, `gymnasium`, `stable_baselines3` | **DUAL_ROLE** (`TRAINING` + `RUNTIME_INFERENCE`) | Used by `farming/{trainer,sb3_training,sb3_adapter}.py` and `simulator/*` for training, **and** required by `simulator.split_branch_policy` (a `RUNTIME_ABI_COMPATIBILITY` module) to deserialize/run the frozen 0051200 checkpoint via `PPO.load()` — a future runtime derivative that loads and runs this checkpoint needs these packages too. Per the authorization's explicit instruction, **not classified as training-only and not slated for exclusion** from any future runtime candidate. `requirements-training.txt` currently isolates them from the base `requirements.txt`; this phase does not change that split (a packaging decision, not made here — see section 8's `unresolved_future_choices`) |
| `numpy` | DUAL_ROLE (`RUNTIME_INFERENCE`, `SIMULATOR`, `TRAINING`, `GUI_DEV`) | Used throughout `farming`/`position`/`navigation`/`mapper`/`simulator`/`Bot.py`/`Gui.py` |
| `opencv-python` (`cv2`) | GUI_DEV / DUAL_ROLE | Used by `Bot.py`/`Gui.py`/`mapper/*`/`libs/*`/`assets/*` for vision/rendering; a future headless runtime derivative may or may not need it depending on whether it retains vision-based fallbacks — recorded as an `unresolved_future_choices` item, not decided here |
| `PySimpleGUI` | GUI_DEV | `Gui.py` only |
| `pywin32` (`win32api`/`win32gui`/`win32con`/`pywintypes`) | RUNTIME_NATIVE | `libs/*`, `utils/*` — window/keyboard/focus interaction; `position/*` uses `ctypes` directly rather than `pywin32` for native memory reads |
| `pytesseract`, `pyautogui`, `keyboard`, `pynput`, `pyttsx3`, `pytweening` | GUI_DEV / RUNTIME_NATIVE | Input/OCR/TTS support consumed by `libs/*`/`Bot.py` |
| `pyfiglet` | GUI_DEV | `utils.helpers.print_logo`, used by `apps/dev_app.py`'s console banner only |
| `pytest`, `pyinstaller` | TEST / BUILD | `requirements-dev.txt` only |
| `tensorboard`, `rich` | TRAINING | `requirements-training.txt` only; training-loop diagnostics, not needed for inference |
| `msgpack` | DUAL_ROLE (`RECORDER`, `RUNTIME_INFERENCE`-adjacent) | Archive frame/event serialization (`simulator/schema.py`, `recorder/*`); also listed in `FlyffFarmingRecorder.spec`'s `hiddenimports` |

No `AMBIGUOUS_STOP` — `torch`/`gymnasium`/`stable_baselines3` were the one
package family requiring the "trace actual use, don't assume by name"
treatment the authorization specifically warns about, and that tracing
produced a determinate `DUAL_ROLE` classification, not an unresolved one.

## 3. `sys.path.insert`/`sys.path.append` inventory (Section 7)

Full inventory and classification: `packaging/path_bootstrap_registry.py`'s
`REGISTERED_BOOTSTRAPS` (also the source of truth for
`tests/test_path_bootstrap_registry.py`, section 9). Summary by category:

- **`apps/*.py`** (4 files, Phase-10-added): `Path(__file__).resolve().parents[1]`
  — developer-compatibility direct-script bootstraps, explicitly sanctioned
  by this phase's Section 7 as acceptable and NOT the future deployment
  architecture (section 4 below establishes canonical `python -m` invocation
  as the non-bootstrap-dependent form).
- **`devtools/native/*.py`, `devtools/archives/list_world_model_eligible.py`,
  `devtools/calibration/calibration_capture.py`** (6 files, Phase-10-added):
  same category, dev-tool self-bootstraps.
- **`devtools/archives/sort_new_recordings.py`**: same-directory sibling
  import bootstrap (imports `inventory_recordings` from its own directory),
  not a repository-root bootstrap.
- **`RUN_CANONICAL_{ADVANCED,BASIC,BEGINNER,INTERMEDIATE}.py`,
  `_basic_round_eval_worker.py`, ~20 root `scratchpad_*.py` files,
  `tools/friend_pointer_recovery_test.py`**: pre-existing, unrelated to
  Phase 10/11, self-bootstraps for direct root-level script invocation.
  Unchanged this phase (Section 16: no bulk scratchpad reorganization).
- **`refactor_logs/profiles/{runtime_lifecycle_mock,runtime_native_pointer_harness}.py`**:
  historical, dead code (confirmed zero current references) — excluded
  from the active registry scope, documented as HISTORICAL_ONLY rather
  than registered as a live bootstrap.

No occurrence resolves to a sibling old worktree, a `.pth` file, a
`sitecustomize.py`, or an environment-only `PYTHONPATH` requirement —
confirmed by direct inspection of every match above (all resolve
`Path(__file__)`-relative to a location inside this same repository).

## 4. Canonical module invocation (Section 7)

Tested via `python -m apps.<name> --help`/safe-argv equivalents (never by
opening a real GUI window):

| Entrypoint | `python -m apps.X` result | Notes |
|---|---|---|
| `apps.dev_app` | Resolves; module-level side effects (`gui = Gui(...)`, `bot = Bot()`) execute on import, same as direct-script invocation always did — not tested by actually running `main()` | Characterized via `runpy.run_path`/AST, never by opening a window |
| `apps.recorder_app` | Resolves cleanly | `run_gui()` guarded by `if __name__ == "__main__"` |
| `apps.simulator_cli` | Resolves; `--help` exits 0 with usage text | Safe to invoke directly |
| `apps.telemetry_cli` | Resolves; `--help` exits 0 with usage text | Safe to invoke directly |

All four resolve their imports identically whether invoked via
`python -m apps.X` (which puts the repository root on `sys.path[0]`
automatically, needing no bootstrap) or via direct script invocation
(`python apps/X.py`, which needs the Phase-10 bootstrap since
`sys.path[0]` becomes `apps/` otherwise). Both forms are proven equivalent
by `tests/test_canonical_module_invocation.py` (section 9) — canonical
module invocation requires no new bootstrap of any kind; direct-script
invocation keeps working through the existing, already-classified
Phase-10 bootstraps.

## 5. Runtime ABI compatibility modules (Section 5)

| Module | Classification | Why not simulator algorithm |
|---|---|---|
| `simulator/split_branch_policy.py` | RUNTIME_ABI_COMPATIBILITY | Defines the exact policy/feature-extractor classes (`SplitSteeringNavigationPolicy`, `SplitSteeringEventPolicy`, `GeometryAugmentedFeaturesExtractor`, `SplitBranchExtractor`, `SplitSteeringEventHead`, `NavigationAugmentedFeaturesExtractor`) the frozen 0051200 checkpoint's pickle stream references by exact `__module__.__qualname__`. Contains architecture/feature-extraction code, but its role is checkpoint deserializability, not navigation/movement algorithm ownership — those live in `navigation/*` |
| `simulator/kinodynamic_route_planner.py` | RUNTIME_ABI_COMPATIBILITY | Phase-9 permanent, behavior-free re-export shim (`KinoState`, `RouteEdgeInfo`) — zero routing implementation, proven by AST scan (`tests/test_pickle_module_identity_compat.py::test_compat_shims_contain_no_duplicate_behavioral_definitions`, still passing) |
| `simulator/movement_kernel.py` | RUNTIME_ABI_COMPATIBILITY | Same, for `AdvanceResult` — zero movement-kernel implementation |

Canonical algorithm ownership, confirmed unchanged since Phase 9:
`navigation/kinodynamic_route_planner.py` (routing/persistence),
`navigation/movement_kernel.py` (movement kernel),
`navigation/movement_kinematics.py`, `navigation/navigation_evidence.py`.
`git diff 77dc6e5 -- navigation/ simulator/split_branch_policy.py
simulator/kinodynamic_route_planner.py simulator/movement_kernel.py` is
empty as of this audit (no Phase-11 code has run yet).

## 6. R1b exception (Section 6)

Unchanged from Phase 10, unwidened, not touched this phase:

```
importer:              runtime_controller.py
dependency:             farming.trainer
permitted_symbols:      dry_run_native_farming, run_native_farming_agent,
                         train_native_farming, validate_native_farming_data
status:                 PRE_EXISTING_SOURCE_BACKED_EXCEPTION
introduced_by_phase10:  false
```

Recorded for the future-derivation profile (section 8/packaging profile)
as:

```
KNOWN_DEV_RUNTIME_COUPLING_NOT_PART_OF_FUTURE_DERIVATION_CONTRACT
```

This means a future deployment derivative built from canonical source
would need to omit or separately redesign this one GUI-triggered training
control path — that redesign is explicitly NOT assigned to Phase 11 (or
any specific future phase); it is recorded as an open item.

## 7. Historical/research scope (Section 16)

No bulk scratchpad move performed. No `research/` directory created. The
~110 root `scratchpad_*.py` files, the 3 explicitly frozen historical
files (`scratchpad_general_router_episode.py`,
`scratchpad_beginner_navigation_mix_pools.py`,
`scratchpad_legacy_qualified_selector.py`, governed by
`scratchpad_historical_reproduction_guard.REQUIRED_FILES` and B4), and the
test-owned harness copies (`tests/helpers/router_qualification_harness.py`,
`tests/helpers/beginner_navigation_mix_harness.py`) all remain exactly
where they were at Phase-10 exit.
