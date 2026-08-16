# Phase-0 Test and Gate Log

## Inherited authoritative test baseline

These results were completed before Codex took over and are preserved from
`docs/migration/WIP_BASELINE.md`. Codex has not rerun the broad suites.

| root | exact invocation detail | result | classification |
|---|---|---|---|
| `foreground_vision_bot` | Full suite; command recorded by the prior executor in its session evidence | 709 collected; 705 passed, 3 failed, 1 skipped | Three genuine pre-existing failures |
| `flyff_farming_simulator` | Full suite with writable explicit `--basetemp` | 357 collected; 355 passed, 0 failed/errors, 2 skipped | Clean product-code baseline |
| `flyff_farming_recorder` | Full suite with writable explicit `--basetemp` | 24 passed | Clean product-code baseline |

The default pytest temp location caused 66 simulator and 3 recorder
`PermissionError: [WinError 5]` setup errors. Those 69 errors disappeared with a
writable explicit `--basetemp` and are classified as an environment ACL defect.

Exact pre-existing bot failures:

1. `tests/test_farming_environment_lifecycle.py::test_focus_loss_during_eva_discards_kill_and_transition`
2. `tests/test_farming_training_session.py::test_normal_training_status_is_concise_and_uses_total_model_steps`
3. `tests/test_farming_training_session.py::test_training_callback_publishes_structured_session_statistics`

## Codex continuation gates

Broad suites and the expensive 820M reproduction are deliberately not being
repeated.

### Artifact-manifest verifier — harness failure

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: verify all rows before accepting the uncommitted manifest.
- Start/end: 2026-08-16 01:59 local; under one second.
- Producer exit code: **1**.
- Exact command: PowerShell inline loop over `ARTIFACT_MANIFEST.tsv` using
  `git ls-files --error-unmatch -- $rel` followed by `git check-ignore -q`.
- Result: PowerShell promoted the expected pathspec stderr for the first ignored
  checkpoint into a terminating error. No artifact discrepancy was reported and
  no file changed.
- Classification: verifier-harness error, corrected on the next run.

### Artifact-manifest byte/status verifier

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: mechanically verify every manifest row and representative status.
- Start/end: 2026-08-16 02:00:00–02:00:27 local.
- Producer exit code: **0**.
- Exact command: PowerShell imported `docs/migration/ARTIFACT_MANIFEST.tsv`,
  rejected duplicate/backslash/missing paths, compared `Get-Item.Length` and
  `Get-FileHash -Algorithm SHA256`, classified tracked paths from
  `git ls-files -- $rel`, classified remaining paths with
  `git check-ignore -q -- $rel`, and exited 1 if any recorded value differed.
- Result: 348/348 rows passed; 708,385,568 bytes; 27 tracked; 321 ignored;
  0 untracked; 0 errors. Representative cases: tracked 0051200 checkpoint,
  ignored alternate model, ignored recording ZIP, ignored 107 MB scratch ZIP,
  tracked calibration CSV.
- Classification: passed preservation gate.

### Independent evidence-set consistency verifier

- CWD: `C:\Users\Ridd\Documents\Repos\Flyff RL`
- Reason: ensure the manifest is complete rather than merely self-consistent.
- Start/end: 2026-08-16 02:02 local; 1.4 seconds.
- Producer exit code: **0**.
- Exact command: PowerShell independently enumerated `*.zip` below the two model
  roots and recordings root, `*.npz` below datasets, calibration CSVs, the six
  explicit map assets, three recording metadata files, and the scratch contract
  archive; then set-compared those paths/categories with the manifest and
  cross-checked checkpoint inventory, module references, and closure rows.
- Result: exact 348-row set; categories checkpoint 313, recording archive 8,
  recording metadata 3, dataset 9, map asset 6, calibration corpus 8, scratch
  contract archive 1. Inventory 313; reference rows 317; closure 37; expected
  compatibility counts 275/5/2/2/2; 0 errors.
- Classification: passed preservation gate.
