# WIP Baseline — dirty state and test results BEFORE any execution-dependency staging

Captured at Phase-0 §1, while HEAD = `ee5b898`. This file describes the working tree
**before** any currently-modified execution-dependency file was staged into any commit.

**This is a preservation record, NOT a validation record.** Nothing here is claimed to be
known-good. It exists so a later reader can tell what was already broken/in-flight before
the migration touched anything.

## 1. Modified tracked files (34)

`in_820M_closure` marks files that the historical-reproduction closure requires; those are
committed separately (see `HISTORICAL_REPRODUCTION_CLOSURE.tsv`) and their inclusion means
only that they are needed to reproduce that one evaluation — not that they are validated.

| # | path | sha256 (working tree) | diffstat | classification | in_820M_closure | notes |
|---|------|----------------------|----------|----------------|-----------------|-------|
| 1 | `flyff_farming_simulator/MISTAKES.md` | `aae97cacd0a44d17…` | 1 file changed, 137 insertions(+) | other known work | no | appends mistake-log entries; no code semantics |
| 2 | `flyff_farming_simulator/RUN_CANONICAL_ADVANCED.py` | `0bc7baef81daff6c…` | 1 file changed, 58 insertions(+), 4 deletions(-) | other known work | no | AUTO_GRADUATION_ENABLED=False gating pending collision-metric review (2026-08-08) |
| 3 | `flyff_farming_simulator/RUN_CANONICAL_BEGINNER.py` | `94b3474fb374a547…` | 1 file changed, 2 insertions(+), 1 deletion(-) | other known work | no | logs requested vs actual timesteps |
| 4 | `flyff_farming_simulator/RUN_CANONICAL_INTERMEDIATE.py` | `a80b120300f8536c…` | 1 file changed, 4 insertions(+), 2 deletions(-) | other known work | no | passes canonical_stage for provenance; timestep logging |
| 5 | `flyff_farming_simulator/scratchpad_ppo_pure_navigation_v2.py` | `3ef1f0c9c922f88b…` | 1 file changed, 10 insertions(+) | other known work | no | docstring-only SUPERSEDED notice |
| 6 | `flyff_farming_simulator/simulator/basic_environment.py` | `4a53dbf933386852…` | 1 file changed, 2 insertions(+), 2 deletions(-) | calibrated-arc migration work | no | replaces hardcoded 925 with POLICY_INPUT_SIZE (sidecar expansion) |
| 7 | `flyff_farming_simulator/simulator/basic_training.py` | `0c31eda9355a548a…` | 1 file changed, 24 insertions(+), 5 deletions(-) | calibrated-arc migration work | no | SIDECAR_SIZE / movement_kernel / previous_steering wiring |
| 8 | `flyff_farming_simulator/simulator/beginner_transition.py` | `f97d8f3819d1f3cb…` | 1 file changed, 18 insertions(+), 5 deletions(-) | other known work | no | canonical_stage provenance + _contact_event_stats |
| 9 | `flyff_farming_simulator/simulator/environment.py` | `4d4870508c50e0d8…` | 1 file changed, 46 insertions(+), 26 deletions(-) | calibrated-arc migration work | YES | adds previous_steering state; kinodynamic/movement_kernel integration |
| 10 | `flyff_farming_simulator/simulator/milestone_evaluator.py` | `cbf8f38ffb2f68c9…` | 1 file changed, 117 insertions(+), 1 deletion(-) | other known work | no | persistent-contact event statistics helper |
| 11 | `flyff_farming_simulator/simulator/navigation_dataset.py` | `4151515616b2dc28…` | 1 file changed, 2 insertions(+), 2 deletions(-) | calibrated-arc migration work | no | POLICY_INPUT_SIZE instead of hardcoded 925 |
| 12 | `flyff_farming_simulator/simulator/navigation_history.py` | `d654a26c0f3aa23f…` | 1 file changed, 77 insertions(+), 29 deletions(-) | calibrated-arc migration work | YES | TEMPORAL_SIDECAR_SIZE=2 + PREVIOUS_STEERING_SIDECAR_SIZE=3 -> SIDECAR_SIZE=5 |
| 13 | `flyff_farming_simulator/simulator/navigation_ppo.py` | `367c8ff7d8397a73…` | 1 file changed, 12 insertions(+) | other known work | no | reports actual_timesteps vs requested (rollout-math audit) |
| 14 | `flyff_farming_simulator/simulator/progress_reporting.py` | `9b539b34598b4f47…` | 1 file changed, 19 insertions(+), 7 deletions(-) | other known work | no | chunk-relative progress/ETA fix for resumed lineages |
| 15 | `flyff_farming_simulator/simulator/pure_navigation_env.py` | `10288f574ae147cb…` | 1 file changed, 167 insertions(+), 104 deletions(-) | other known work | no | PPO pure-navigation ablation target-selection rebuild |
| 16 | `flyff_farming_simulator/simulator/run_provenance.py` | `89c7496bbe5f7e97…` | 1 file changed, 7 insertions(+) | calibrated-arc migration work | no | records movement_kernel physics version in provenance |
| 17 | `flyff_farming_simulator/simulator/split_branch_policy.py` | `85244a82df645e30…` | 1 file changed, 11 insertions(+), 5 deletions(-) | calibrated-arc migration work | YES | features extractor accepts SIDECAR_SIZE-derived policy input |
| 18 | `flyff_farming_simulator/simulator/steering_oracle.py` | `abff9b6ec9a3eb5b…` | 1 file changed, 299 insertions(+), 366 deletions(-) | calibrated-arc migration work | no | kinodynamic/movement_kernel + previous_steering oracle rework |
| 19 | `flyff_farming_simulator/simulator/synthetic.py` | `75910027d43cc0b2…` | 1 file changed, 36 insertions(+), 42 deletions(-) | calibrated-arc migration work | no | movement_kernel / previous_steering in synthetic generation |
| 20 | `flyff_farming_simulator/simulator/world_model.py` | `8fd8626766b375ad…` | 1 file changed, 24 insertions(+) | calibrated-arc migration work | YES | movement_kernel-based world model additions |
| 21 | `flyff_farming_simulator/synthetic_curriculum_advanced_training_v1/curriculum.json` | `7aea457e3fe6e0da…` | 1 file changed, 49 insertions(+), 1 deletion(-) | other known work | no | adds advanced curriculum variants |
| 22 | `flyff_farming_simulator/tests/test_basic_environment.py` | `f5db874c0615f38c…` | 1 file changed, 2 insertions(+), 2 deletions(-) | calibrated-arc migration work | no | asserts POLICY_INPUT_SIZE rather than 925 |
| 23 | `flyff_farming_simulator/tests/test_basic_training_pipeline.py` | `f0cebec0100e8101…` | 1 file changed, 11 insertions(+), 9 deletions(-) | calibrated-arc migration work | no | movement_kernel-aware pipeline test |
| 24 | `flyff_farming_simulator/tests/test_deep_review.py` | `45b702ac1b56344c…` | 1 file changed, 10 insertions(+), 7 deletions(-) | calibrated-arc migration work | no | movement_kernel references |
| 25 | `flyff_farming_simulator/tests/test_fine_tune_steering_branch.py` | `e3cd60bbbb45e3bc…` | 1 file changed, 9 insertions(+), 2 deletions(-) | calibrated-arc migration work | no | SIDECAR_SIZE / previous_steering |
| 26 | `flyff_farming_simulator/tests/test_navigation_dataset.py` | `41076c8ca8496775…` | 1 file changed, 3 insertions(+), 2 deletions(-) | calibrated-arc migration work | no | POLICY_INPUT_SIZE |
| 27 | `flyff_farming_simulator/tests/test_navigation_history.py` | `d195581b5ed479f6…` | 1 file changed, 76 insertions(+), 19 deletions(-) | calibrated-arc migration work | no | temporal + previous-steering sidecar tests |
| 28 | `flyff_farming_simulator/tests/test_pure_navigation_env.py` | `a99a878d74b9d99c…` | 1 file changed, 157 insertions(+), 42 deletions(-) | other known work | no | ablation wrapper tests (2026-08-09/10) |
| 29 | `flyff_farming_simulator/tests/test_run_provenance.py` | `cff23ce86f9c129c…` | 1 file changed, 14 insertions(+), 2 deletions(-) | calibrated-arc migration work | no | calibrated_arc / movement_kernel provenance assertions |
| 30 | `flyff_farming_simulator/tests/test_steering_expansion_transplant.py` | `218a8ab3fb0b48c2…` | 1 file changed, 6 insertions(+), 2 deletions(-) | calibrated-arc migration work | no | SIDECAR_SIZE transplant test |
| 31 | `flyff_farming_simulator/tests/test_steering_oracle_escape_robust.py` | `f8440bbd9c65fc5e…` | 1 file changed, 157 insertions(+), 200 deletions(-) | calibrated-arc migration work | no | steering oracle / previous_steering |
| 32 | `flyff_farming_simulator/tests/test_steering_oracle_v3_terminal_gate.py` | `2a3dedb8d31e9906…` | 1 file changed, 124 insertions(+), 110 deletions(-) | calibrated-arc migration work | no | steering oracle v3 terminal gate |
| 33 | `flyff_farming_simulator/tests/test_synthetic_layout_validation.py` | `174ee7399cf0e479…` | 1 file changed, 24 insertions(+), 35 deletions(-) | calibrated-arc migration work | no | movement_kernel-aware layout validation |
| 34 | `flyff_farming_simulator/tests/test_temporal_sidecar_parity.py` | `f9427786185b809c…` | 1 file changed, 67 insertions(+), 25 deletions(-) | calibrated-arc migration work | no | previous_steering sidecar parity |

Classification counts: calibrated-arc migration work = 22, other known work = 12, unknown = 0.

Full sha256 values (untruncated):

```
aae97cacd0a44d1724e251ad4e225d05111f1c7aaf35eb1dbb0c7749477e4730  flyff_farming_simulator/MISTAKES.md
0bc7baef81daff6c87e5e189a8b5f88ff8c73265c432544601913c99752fdc55  flyff_farming_simulator/RUN_CANONICAL_ADVANCED.py
94b3474fb374a5475e4f5dcc00f2d8dd1a9a9470e3b348d59aa7cb6a37024e1d  flyff_farming_simulator/RUN_CANONICAL_BEGINNER.py
a80b120300f8536c596790db3c0d667b99615ca7eb5439757a4f54f03ba73e19  flyff_farming_simulator/RUN_CANONICAL_INTERMEDIATE.py
3ef1f0c9c922f88b15023fdf2389a2b71844123d736e6ddf73d5fe67189db0e9  flyff_farming_simulator/scratchpad_ppo_pure_navigation_v2.py
4a53dbf9333868526af0e60cd440fe705b82c6dabc68ebc74fe05d9d7c67cc8e  flyff_farming_simulator/simulator/basic_environment.py
0c31eda9355a548a61c1e9c37c8ccf1da402f32971d8523f4642103e61e7ad10  flyff_farming_simulator/simulator/basic_training.py
f97d8f3819d1f3cb07195ec9db01f81067a8b6e61798e8d0bd28e7088b9a3c78  flyff_farming_simulator/simulator/beginner_transition.py
4d4870508c50e0d8f82961cdd9c493f27c04e207cb015f5634cf1c7a405cd014  flyff_farming_simulator/simulator/environment.py
cbf8f38ffb2f68c9a9eb92052a4beab84c0d94492b4a45a50f1680006d6c1259  flyff_farming_simulator/simulator/milestone_evaluator.py
4151515616b2dc2876bb68a473aeb3a1fd56e18d0e6ea689286581afb0ab2dcb  flyff_farming_simulator/simulator/navigation_dataset.py
d654a26c0f3aa23ff1168b927b572f7f30882a7612424e16929dda4eab8c59c0  flyff_farming_simulator/simulator/navigation_history.py
367c8ff7d8397a73febe1978983091c2397bbf8732fdfab584d771b24e0fffe1  flyff_farming_simulator/simulator/navigation_ppo.py
9b539b34598b4f47c728b4413f4b981d9173a52a1150df418b0f5ac989787006  flyff_farming_simulator/simulator/progress_reporting.py
10288f574ae147cbff9315c64b55fcab65e415a523cc4f9b03a0d80e02e77692  flyff_farming_simulator/simulator/pure_navigation_env.py
89c7496bbe5f7e978b4b9bacce150ea8be56c24b5a53c2a44e254ccf44ed624f  flyff_farming_simulator/simulator/run_provenance.py
85244a82df645e303cc4edde4e89fd02b6b5b071017467a6a39e63cb93275e26  flyff_farming_simulator/simulator/split_branch_policy.py
abff9b6ec9a3eb5b5370fbadf317cb2b2eb02c691387caa6a20ab06d0d6d5436  flyff_farming_simulator/simulator/steering_oracle.py
75910027d43cc0b272ffee31e2c0d6f88bded48768b72b6e49655058358eef47  flyff_farming_simulator/simulator/synthetic.py
8fd8626766b375ada3a33ec820b988a531eacb09b686d3d2139f3657b2517c35  flyff_farming_simulator/simulator/world_model.py
7aea457e3fe6e0da86b5384eb750c6a1a0d700de2a34ba8e175ad3a513d893a2  flyff_farming_simulator/synthetic_curriculum_advanced_training_v1/curriculum.json
f5db874c0615f38cf60c92779e081988538788d1abb367f90873575683ccac1c  flyff_farming_simulator/tests/test_basic_environment.py
f0cebec0100e8101a78df390ae8e9d0fb1c1604d0d5b133d4cc511db5dfb3d4f  flyff_farming_simulator/tests/test_basic_training_pipeline.py
45b702ac1b56344c59ce2f444c3a8db2a82725fa801ead59d1353149fc6f2a62  flyff_farming_simulator/tests/test_deep_review.py
e3cd60bbbb45e3bcca83965945c86c37e9f57e3ddc2b73a9def638180c86bb45  flyff_farming_simulator/tests/test_fine_tune_steering_branch.py
41076c8ca8496775afface450242ef7af9b1fafc583abc9d5872c3b945d2fb5a  flyff_farming_simulator/tests/test_navigation_dataset.py
d195581b5ed479f6e05dcfcaff20bef253a5060e84168a089a68a99eea1dbed5  flyff_farming_simulator/tests/test_navigation_history.py
a99a878d74b9d99c46f54ebffb2a863cf44857cfcc8a7af700700b7f0af8e0bb  flyff_farming_simulator/tests/test_pure_navigation_env.py
cff23ce86f9c129c4976c1602ba59c576daaa2dfd3350fad80ff254370c76e8c  flyff_farming_simulator/tests/test_run_provenance.py
218a8ab3fb0b48c2cce29655f85edaeab18ce900eeee1d5c2cb587ec0cc38c08  flyff_farming_simulator/tests/test_steering_expansion_transplant.py
f8440bbd9c65fc5ea2bf8c7d596dcb26dadb820255bc3127f1e18ed40883343c  flyff_farming_simulator/tests/test_steering_oracle_escape_robust.py
2a3dedb8d31e99068e871589f6d3f11db5c991bd8a725f0ccf1dcc72804a392e  flyff_farming_simulator/tests/test_steering_oracle_v3_terminal_gate.py
174ee7399cf0e479adbf7e48053a7080e20293a6cc45ef9e080eb13acbda2bc2  flyff_farming_simulator/tests/test_synthetic_layout_validation.py
f9427786185b809ca6353f88b8863fc0abc4f38fc28c609e3d0b324a12174147  flyff_farming_simulator/tests/test_temporal_sidecar_parity.py
```

## 2. Test-suite baseline (all three roots)

### 2a. IMPORTANT — environment defect found while establishing this baseline

A first pass run with pytest's default temp directory produced **69 errors** across two
roots, every one of them identical:

```
failed on setup with "PermissionError: [WinError 5] Access is denied:
    'C:\\Users\\Ridd\\AppData\\Local\\Temp\\pytest-of-Ridd'"
```

This is an **environment/ACL defect on the machine**, not a code failure — the same
access-denied condition also caused 51 empty-directory failures during the external
snapshot copy. It masks 69 tests, so it is recorded here but is NOT the code baseline.
Re-running the affected roots with a writable `--basetemp` resolved all 69.

### 2b. Authoritative code baseline (writable basetemp)

| root | tests | passed | failed | errors | skipped |
|------|-------|--------|--------|--------|---------|
| `foreground_vision_bot` | 709 | 705 | 3 | 0 | 1 |
| `flyff_farming_simulator` | 357 | 355 | 0 | 0 | 2 |
| `flyff_farming_recorder` | 24 | 24 | 0 | 0 | 0 |

First-pass (default temp dir, environment-defective) figures, for the record:
`flyff_farming_simulator` 289 passed / 66 errors / 2 skipped; `flyff_farming_recorder` 21 passed / 3 errors.

### 2c. Pre-existing failures — the complete list

Only three genuine failures exist, all in `foreground_vision_bot`, a root with **zero**
modified files. They are therefore pre-existing and unrelated to the calibrated-arc WIP.

1. `tests/test_farming_environment_lifecycle.py::test_focus_loss_during_eva_discards_kill_and_transition`
   — `AttributeError: 'object' object has no attribute 'candidates'`.
   Best-effort cause: a bare `object()` test double lacks an attribute the code now reads
   (test-double drift behind production code).
2. `tests/test_farming_training_session.py::test_normal_training_status_is_concise_and_uses_total_model_steps`
   — status line reported `steps=0`, expected `steps=123,456`.
   Best-effort cause: status line reads a per-chunk counter rather than the model's
   lifetime `num_timesteps`.
3. `tests/test_farming_training_session.py::test_training_callback_publishes_structured_session_statistics`
   — `assert 0 == 5000`. Best-effort cause: same total-model-steps wiring as (2).

`flyff_farming_simulator` and `flyff_farming_recorder` have **no** failing tests once the
environment defect is bypassed.

## 3. Deleted-but-tracked entries

119 paths show as deleted; 115 are `.pytest-temp-v16/` and `.pytest-temp-v17/` scratch
phantoms. The substantive ones are handled separately (see the decision log for
`OVERNIGHT_20260809_PIPELINE.md`).
