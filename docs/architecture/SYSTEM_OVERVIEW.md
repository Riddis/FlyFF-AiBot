# System Overview

**Confidence: VERIFIED_CONTRACT** unless otherwise noted. Evidence:
`docs/migration/PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md`, `tests/
test_dev_app_import_closure.py`, `tests/test_canonical_module_
invocation.py`, `docs/migration/codex_handoff/PHASE10_REPORT.md`.

## 1. The canonical product is the full development application

This repository has one canonical product under active development: the
**full development bot/application** — `apps/dev_app.py` plus its GUI,
specialist subprocess tooling, and the shared `farming`/`position`/
`navigation`/`mapper` runtime it drives. There is **no separate
stripped-down "live bot" product today**, and building one is explicitly
out of scope until the development bot itself is judged sufficiently
complete (see [`docs/decisions/0003-dev-bot-first.md`](../decisions/0003-dev-bot-first.md)).

A future deployment/live derivative, when it is built, will be **derived
from this same canonical source tree** — never a copied fork. See
[`docs/decisions/0001-canonical-source-single-tree.md`](../decisions/0001-canonical-source-single-tree.md)
and `future_runtime_profile/` (a static, non-building dry-run resolver —
see section 5 below) for why and how that derivation is kept provable
without ever actually being performed yet.

## 2. Canonical entrypoints

| Entrypoint | Role | Invocation |
|---|---|---|
| `apps/dev_app.py` | The canonical development GUI: constructs `Gui`+`Bot` at true module level, drives live attach/farming/training/specialist orchestration | `python -m apps.dev_app` (or direct script; both resolve identically — see `tests/test_canonical_module_invocation.py`) |
| `apps/recorder_app.py` | Recorder specialist launcher (`recorder.gui.run_gui()`), also launchable as an independent subprocess from the dev app | `python -m apps.recorder_app` |
| `apps/simulator_cli.py` | Simulator/training CLI (`simulator.cli.main()`) | `python -m apps.simulator_cli --help` |
| `apps/telemetry_cli.py` | Observation-only native telemetry CLI, deliberately independent of `Bot`/`Gui`/`RuntimeController` — cannot press a key | `python -m apps.telemetry_cli --help` |

`python -m apps.X` is the canonical, non-bootstrap-dependent invocation
form (it puts the repository root on `sys.path[0]` automatically).
Direct-script invocation (`python apps/X.py`) still works via each
file's own Phase-10 `sys.path` self-bootstrap, registered in
`docs/migration/tools/phase11_path_bootstrap_registry.py`.

`apps/dev_app.py` constructs a live `Bot()`/`Gui(...)` **unconditionally
at true module level** (not inside `main()`, not guarded by `if
__name__ == "__main__"`) — importing this module at all has real
side effects. Never `import apps.dev_app` from a script or test; use
`importlib.util.find_spec` if you only need to prove it resolves.

## 3. Process / runtime architecture

```text
apps/dev_app.py
  -> Gui (main thread; owns all PySimpleGUI/Tk widgets)
     -> RuntimeController
        -> WorkerManager (CAPTURE / PREVIEW / CONTROL / DIAGNOSTIC workers)
           -> CaptureService -> WindowCapture
           -> PreviewService -> Bot.build_preview
           -> farming.trainer  [the ONE registered R1b exception edge]
              -> FarmingMapContext, position.*, DirectFarmingControl,
                 UnifiedFarmingEnv / PPO
        -> DevToolsGuiController (Phase-10 addition)
           -> SpecialistProcessManager
              -> subprocess.Popen(explicit argv/cwd/env, no PYTHONPATH
                 injection) launching recorder / telemetry / simulator /
                 native-diagnostic / archive / calibration specialists
happening in each specialist subprocess's own process, not in-process
     -> devtools.artifact_inventory (read-only reader over recordings/
        INDEX.json, recording_provenance.json,
        docs/migration/CHECKPOINT_INVENTORY.tsv)
     -> devtools.session_context.resolve_session_context()
        (repo_root/models_dir/map_assets_dir/recordings_dir/
        evaluations_dir/telemetry_sessions_dir — all CWD-independent,
        `Path(__file__).resolve().parents[1]`-based)
```

**`RuntimeBus`** (`runtime_bus.py`, stdlib-only): one shared instance
constructed by `Gui.py`, passed into `RuntimeController` and
`DevToolsGuiController` alike. Bounded-log (`log`/`drain_logs`) plus
reliable lifecycle pub/sub (`complete`/`fail`/`drain_completions`/
`drain_failures`, `alert`/`drain_alerts`), polled every ~50ms by
`Gui.__refresh_runtime`. This is the one shared logging/lifecycle
mechanism between the in-process worker pipeline and the
subprocess-orchestration layer — neither duplicates it.

**Specialist subprocess model** (Phase 10): the dev app launches
recorder/telemetry/simulator/native-diagnostic/archive/calibration tools
as independent OS processes (`SpecialistCommand` registry, 16 tracked
entrypoints, `Path.is_file()` resolution only, never `importlib`), not
as in-process imports. This is *why* the R1b boundary matters (section 4)
— everything reachable from the dev app's own import closure is meant
to stay free of `recorder`/`simulator` training code/`torch`/
`gymnasium`/`stable_baselines3`, with subprocess launch as the sanctioned
way to reach those specialists instead of importing them.

## 3a. GUI settings persistence is CWD-dependent by construction

**Confidence: VERIFIED_CONTRACT**, discovered during the Phase-14
migration-closure audit and previously undocumented anywhere. `Gui.py`
calls `sg.user_settings_filename(path=".")` (PySimpleGUI) with **no
explicit filename** — PySimpleGUI derives one itself from
`sys.modules["__main__"].__file__`'s basename (confirmed by reading
PySimpleGUI's own `UserSettings._compute_filename` source directly).
Two consequences:

1. **The settings filename tracks the entrypoint's actual invoked file
   name, not a fixed name.** Under the current canonical entrypoint
   (`apps/dev_app.py`, however invoked), the generated filename is
   `dev_app.json` — not `foreground_vision_farm.json`, the name it
   produced under the pre-Phase-7 entrypoint (`foreground_vision_
   farm.py`). This explains two now-removed tracked artifacts,
   `foreground_vision_farm.json` (root) and
   `foreground_vision_bot/foreground_vision_farm.json`: both were
   PySimpleGUI-generated snapshots from the **old** entrypoint name,
   orphaned the moment the entrypoint was renamed — confirmed via this
   exact mechanism, not merely "zero current references." See
   `docs/migration/codex_handoff/PHASE14_REPORT.md` for the full
   resolution.
2. **The settings file's location is CWD-relative** (`path="."`) — it
   is written to whatever directory is current when the GUI launches,
   not a fixed repo-relative path. Launching `apps/dev_app.py` from
   different working directories produces different, ungoverned
   `dev_app.json` files in different places. This is a real, current
   limitation (not a bug introduced by consolidation — the same
   `path="."` call pattern predates it) — see `docs/KNOWN_DEBT.md`.

## 4. The R1b exception — one real coupling, not yet resolved

`runtime_controller.py` imports exactly four symbols
(`dry_run_native_farming`, `run_native_farming_agent`,
`train_native_farming`, `validate_native_farming_data`) from
`farming.trainer`, lazily, inside a function body. This is the **one**
registered exception to "the dev app's import closure excludes
recorder/simulator-training/legacy/torch/gymnasium/stable_baselines3."

**Why it exists:** all four functions take `bot: FarmingBot` (the live,
already-attached `Bot` instance with an open native window handle and
active capture threads, constructed once at `apps/dev_app.py` startup)
as their first parameter. This cannot cross a subprocess boundary
without either a real farming-runtime/attachment redesign or an
explicitly-forbidden IPC bridge (confirmed pre-existing across every
phase's `git diff` — this coupling was never introduced by the
migration, only discovered and characterized by it).

**Status:** current dev-app functionality, not redesigned, not removed,
not assigned to any phase. `farming.trainer` is excluded from the future
runtime candidate's own closure (see `future_runtime_profile/`) but
remains fully present and load-bearing in the current dev app. Never
widen this exception past its exact four symbols — `tests/
test_dev_app_import_closure.py::TestExceptionMechanismIsExact` enforces
this mechanically.

## 5. Future deployment derivation is a static proof, not a build

`future_runtime_profile/dependency_profiles.toml` +
`derive_runtime_manifest.py` (`python -m future_runtime_profile.
derive_runtime_manifest`) is a **read-only, non-building dry-run
resolver**. It statically walks the shared-runtime import closure
(`farming`, `position`, `navigation`, `mapper`, `libs`, `utils`,
`assets`, plus `simulator/schema.py` and `legacy/manifest_compat.py`)
and reports whether that closure is clean of forbidden dev/recorder/
simulator-training surfaces, and whether the checkpoint-ABI compatibility
modules and required runtime resources are present and tracked. It
builds nothing: no PyInstaller, no file copy, no `dist/` output. Current
result: **PASS** — see `docs/architecture/COMPONENT_OWNERSHIP.md` for
what that closure actually contains and
`docs/migration/codex_handoff/PHASE11_REPORT.md` for the full audit.

The profile deliberately does **not** decide: the final shipped
checkpoint, whether vision-based OCR/UI-detection is retained or
replaced by pure-native reading, whether `runtime_controller.py`'s
`farming.trainer` coupling is redesigned, the final entrypoint name, or
whether `torch`/`gymnasium`/`stable_baselines3`'s DUAL_ROLE
classification changes the `requirements*.txt` split. These are recorded
as `unresolved_future_choices`, not silently resolved.

## 6. What each shared package owns

| Package | Owns | Consumed by |
|---|---|---|
| `farming/` | Observation/action/reward/session contract, farming-loop training entrypoints (`trainer.py`, `sb3_adapter.py`, `sb3_training.py` are the training-only exception, see [COMPONENT_OWNERSHIP.md](COMPONENT_OWNERSHIP.md)) | `simulator`, `navigation` (indirectly), `devtools.telemetry`, the dev app |
| `position/` | Native process attachment, pointer recovery, actor/monster discrimination, `AttachPolicy`/`RecoveredNativeProfile` | `Bot.py`, `devtools.telemetry`, `devtools.native.*`, `recorder` (via its own compatibility facades — see [COMPONENT_OWNERSHIP.md](COMPONENT_OWNERSHIP.md)) |
| `navigation/` | The one authoritative kinodynamic route planner + movement kernel (Phase-9 canonical) | `simulator/*` env/training code, the two `simulator/*` ABI re-export shims |
| `mapper/` | Map catalog, coordinate mapping, editor GUI, offline RL-map tooling (`mapper/rl/{FeatureExtractor,GymEnv,OfflineTraining}.py` are the training-only exception) | `farming/map_context.py`, `Gui.py` |
| `recorder/` | Canonical demonstration/session recording writer | `apps/recorder_app.py`, standalone PyInstaller build |
| `simulator/schema.py` + `legacy/manifest_compat.py` | Canonical archive/recording **reader** (`RecordingArchive`/`RecordedFrame`/`RecordedActor`/`RecordedEvent`) — corrected classification, not `archives/` (which does not exist) | `tools.inventory_recordings`, `devtools.archives.*`, tests |
| `devtools/` | Phase-10 dev-only orchestration (process manager, artifact inventory, session context, native/calibration/archive tool collection) | `apps/dev_app.py` only |
| `simulator/` (rest) | Training/environment implementation (router/static/single-obstacle waypoint envs, curriculum, CLI) | Training scripts, `docs/migration/tests/`, never the dev app |

## 7. Why the stripped deployment product has not been built

Building a standalone/live bot now would mean maintaining two divergent
implementations of the same behavior — exactly the kind of drift this
migration spent 12 phases eliminating (see `docs/migration/
DECISION_LOG.md` and the Phase 0–12 reports for the original
multi-root/copied-facade problem). The binding product direction,
carried through every phase's authorization since Phase 10, is: finish
the development bot, validate it against a real client when useful, and
only then derive a slimmed deployment build from the same canonical
source — never before, and never as a parallel fork. See
[`docs/decisions/0003-dev-bot-first.md`](../decisions/0003-dev-bot-first.md).

## Evidence / Sources

- `docs/migration/PHASE11_DEPENDENCY_BOUNDARY_ANALYSIS.md` (full
  first-party/third-party classification with import-edge evidence)
- `docs/migration/codex_handoff/PHASE10_REPORT.md` (GUI/specialist
  process orchestration design and tests)
- `docs/migration/codex_handoff/PHASE11_REPORT.md`,
  `docs/migration/codex_handoff/PHASE12_REPORT.md`
- `tests/test_dev_app_import_closure.py`,
  `tests/test_canonical_module_invocation.py`,
  `tests/test_future_derivation_profile.py`
- `future_runtime_profile/dependency_profiles.toml`
