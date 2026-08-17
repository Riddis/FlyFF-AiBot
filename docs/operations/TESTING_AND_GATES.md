# Testing & Gates

Standard commands a future agent/human actually needs. This is not a
log of every one-off migration command ever run — those live in
`docs/migration/codex_handoff/COMMAND_LOG.tsv`.

All commands below assume the repository root as the working directory
and the project's `.venv` Python. Substitute your own venv path where
shown.

## Core product test suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

**Accepted baseline:** 1190 passed, 4 established pre-existing/unrelated
failures, 2 skipped, 1 xfailed. The four accepted failures are:

- `tests/test_farming_environment_lifecycle.py::
  test_focus_loss_during_eva_discards_kill_and_transition`
- `tests/test_farming_training_session.py::
  test_normal_training_status_is_concise_and_uses_total_model_steps`
- `tests/test_farming_training_session.py::
  test_training_callback_publishes_structured_session_statistics`
- `tests/test_navigation_dataset.py::
  test_mine_navigation_dataset_produces_all_four_categories_on_real_layouts`
  (a pre-existing gitignored-artifact gap —
  `models/split_branch_pilot_15000.zip` vs. the test's own
  `.zip.zip`-suffixed lookup)

**A pytest exit code of 1 is expected and correct** when exactly these
four failures remain — it is not a build break. **A fifth failure is
never accepted** without investigation; do not normalize or hide any
new failure. If a shell pipeline (e.g. `| tail -N`) is used, remember it
masks pytest's own exit code with the pipeline's — check pytest's exit
code directly when it matters (see `docs/migration/codex_handoff/
STATE.json`'s `phase10_pytest_exit_code_note` for the specific mistake
this note exists to prevent repeating).

## Migration integrity ruler

```powershell
.\.venv\Scripts\python.exe docs\migration\tools\migration_integrity.py check
```

**Accepted baseline:** `ok: true`, `R6=0 R7a=0 R7b=0 R7c=204 R9=0
R10=0`. `R10` additionally reports `r10_checkpoint_count: 313`,
`r10_module_reference_rows: 317`, `r10_failures: []`. `R7c` may
*decrease* if a compatibility surface is legitimately retired; it must
never grow without an explicit forward supplement
(`docs/migration/POST_PHASE*_R7C_SUPPLEMENT.tsv`) explaining exactly
which new entries are accounted for.

## Migration test suite

```powershell
.\.venv\Scripts\python.exe -m pytest docs\migration\tests\ -q
```

**Accepted baseline:** 77 passed, 0 failed (76 pre-Phase-13 + 1 new
Phase-12 gate-semantics test).

## Future deployment derivation profile

```powershell
.\.venv\Scripts\python.exe -m future_runtime_profile.derive_runtime_manifest
```

**Accepted result:** `FUTURE DEPLOYMENT DERIVATION PROFILE: PASS` — 89
candidate first-party modules, 3 ABI-compatibility modules, 0 forbidden
dependency edges, 0 missing tracked files, 0 duplicate-ownership issues,
1 exception applied (the R1b coupling). This does not mean a runtime
derivative exists or is ready to build — see
`docs/architecture/SYSTEM_OVERVIEW.md` section 5.

## Project knowledge check

```powershell
.\.venv\Scripts\python.exe tools\check_project_knowledge.py
```

Lightweight documentation-integrity gate (see
`docs/agent/PROJECT_RULES.md`). Verifies cheap, machine-checkable facts
about the documentation set itself — it does not reproduce product
behavior tests.

## Relevant focused test groups

| Group | Command | Covers |
|---|---|---|
| R1b / dev-app boundary | `pytest tests/test_dev_app_import_closure.py tests/test_devtools_dependency_direction.py -q` | The one registered R1b exception stays exact; devtools never imported by canonical packages |
| Phase-9 pickle compatibility | `pytest tests/test_pickle_module_identity_compat.py -q` | `KinoState`/`RouteEdgeInfo`/`AdvanceResult` module-identity shims |
| Navigation dependency boundary | `pytest tests/test_navigation_dependency_boundary.py -q` | `navigation/` stays clean of training-only imports |
| Canonical entrypoint invocation | `pytest tests/test_canonical_module_invocation.py -q` | `python -m apps.X` resolves for all four entrypoints, without opening a GUI or attaching to a game process |
| sys.path bootstrap registry | `pytest tests/test_path_bootstrap_registry.py -q` | No new unregistered `sys.path` bootstrap |
| Future-derivability gate | `pytest tests/test_future_derivation_profile.py -q` | The 12 individually named derivability proof points |
| CWD independence | `pytest tests/test_phase11_cwd_independence.py -q` | Path resolution never assumes `os.getcwd() == repo root` |

## `git diff --check`

Run after any change touching tracked text files, before considering a
commit final:

```powershell
git diff --check
```

Catches trailing whitespace / mixed line-ending problems before they
land.

## When to run the broad suite vs. focused tests

Documentation-only changes do **not** require re-running the ~9-minute
`tests/` suite. Run it when a change touches product/runtime behavior
or import structure in a way that plausibly affects it. Metadata-only
changes (e.g. `CANONICAL_OWNERS.toml` field edits) warrant the focused
groups above plus the migration suite, not the full broad suite — see
`docs/migration/codex_handoff/PHASE12_REPORT.md` section 7b for a worked
example of this judgment call.
