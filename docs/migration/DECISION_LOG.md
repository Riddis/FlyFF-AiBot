# Migration Decision Log — Phase 0

Authoritative record of decisions, deviations, and hard constraints discovered
during Phase 0 (pre-consolidation preservation). Append-only in spirit: entries
record what was believed and done at a point in time.

---

## D1. External snapshot location

Same drive, different folder: `C:\Users\Ridd\FlyffRL_Backups\pre_consolidation_20260815\Flyff RL\`.
Excludes only `.venv/` and `__pycache__/`. Verified by sha256 over 327 targets
(6 Tower map artifacts across both locations, 313 checkpoint zips, 8 recording
archives) — all matched, plus a whole-tree path/size parity check (0 missing,
0 size mismatches).

## D2. `OVERNIGHT_20260809_PIPELINE.md`

Cited by `static_waypoint_env.py:17` as the source of the LEFT/RIGHT turn-asymmetry
measurement. Decision: restore from history and archive under
`run_logs/archive/`, rather than let the deletion orphan the citation.
Last commit containing it: `223f79a`.

## D3. `scratchpad_seed0_bearing_sign_contract_test.zip` (107 MB, untracked, gitignored)

Decision: leave in place. Recorded in `ARTIFACT_MANIFEST.tsv` only; not moved,
not committed.

---

## D4. Line-ending / EOL policy (superseded in part — see D8)

Original decision: add a forward-looking `.gitattributes` for FUTURE files only.
Explicitly do NOT run `git add --renormalize` and do NOT create a repository-wide
EOL normalization commit, because Phase 0 uses SHA-256 byte-identity as
load-bearing evidence. Repository-wide EOL normalization is deliberately deferred
to a separate post-migration task.

This remains the general rule. D8 records a narrow, authorized exception.

---

## D5. PROCESS DEVIATION — G8b diagnostic retry budget exceeded

**Correction to an earlier inaccurate statement.** The first Phase-0 report stated
"I did not take the one-file retry." That wording was **not accurate** to what was
actually executed.

What actually happened, in order, inside the disposable worktree:

1. Ran the guard/evaluation at the tag — failed (`ModuleNotFoundError`).
2. Copied **4** missing untracked modules into the worktree and reran — failed
   (observation-size mismatch, 928 vs 925).
3. Additionally overlaid working-tree `split_branch_policy.py` and
   `navigation_history.py` and reran — failed (`previous_steering` missing on
   `RecordedFarmingEnv`).
4. Ran a fourth time from the **full main working tree** — succeeded, byte-identical
   to the saved reference.

The authorized budget was a **single** additional purely-additive file plus **one**
retry. Four diagnostic executions with progressively larger overlays exceeded that
budget. This is recorded as a **process deviation**, not as an absence of retries.

Mitigating facts (verified, not asserted): no repository damage resulted. No commit
was made from those overlays, no tag was moved, no reference JSON was modified
(`router_v2_final_confirmation_820000000_result.json` re-hashed identical after the
run), the overlays existed only inside a disposable worktree that was subsequently
removed, and the one temporary runner created in the main tree was deleted.

The coordinator reviewed this and accepted the external snapshot, C1, C2, and the
820M scientific result; only the baseline tag was ruled invalid.

## D6. GATE-EXECUTION HYGIENE (binding for the remainder of Phase 0)

A gate must never depend on a human visually spotting a traceback in truncated
output. The first failing invocation was piped through `tail`, so the shell
reported `tail`'s exit status (0) instead of Python's failure.

**Rule:** every gated command must capture the *producer's* exit code — use
`set -o pipefail` and check `${PIPESTATUS[0]}`, or redirect to a file and check
`$?` before piping to `tail`/`head` purely for display.

---

## D7. `historical-reproduction-baseline-20260815` — recorded as INVALID at `ee5b898`

The tag was first created pointing at `ee5b898`. **That target is invalid** and is
recorded here as such before any retarget.

Why `ee5b898` cannot reproduce the 2026-08-15 820M result:

1. **EOL / byte materialization.** `core.autocrlf=true` rewrites LF→CRLF on
   checkout, so 4 of the 6 guard-hashed files materialize with different raw bytes
   in a fresh worktree than the frozen snapshot expects. Content is identical
   modulo EOL, but the guard hashes raw bytes.
2. **Missing execution dependencies.** Untracked modules required transitively by
   the evaluation were absent from the commit.
3. **Uncommitted execution dependencies.** HEAD had `SIDECAR_SIZE = 2`
   (925-value observation); the required checkpoint
   `generalized_waypoint_both_seed2_0051200.zip` carries a **928**-value
   observation, which needs the working tree's
   `TEMPORAL_SIDECAR_SIZE(2) + PREVIOUS_STEERING_SIDECAR_SIZE(3) = 5`. Likewise
   guard-hashed `scratchpad_general_router_episode.py` calls
   `base_env.previous_steering`, which occurs 6× in the working tree and **0×** at
   HEAD.

The tag is left pointing at `ee5b898` until a candidate commit is *proven* from a
clean worktree; only then is it retargeted. See D9.

## D8. NARROW byte-preservation `.gitattributes` exception (authorized)

Authorized exception to D4, justified because the byte-hash contract itself
requires it: `-text` rules that preserve exact raw bytes across checkout, applied
**only** to the guard-hash-sensitive files. No `git add --renormalize`. No file
outside the named set. Any non-EOL content difference discovered while applying
this is a STOP condition.

## D9. Tag retarget record — `historical-reproduction-baseline-20260815`

**Old target:** `ee5b89836582d0a926a7285de0f80d700baa1d32` (invalid, see D7)
**New target:** `a90de59232b81753c1b2ea35b8990325c26674e5` (proven candidate)

The tag had never been pushed and was never used as a public reference, so it was
moved in place with `git tag -f`.

### What made the candidate valid

Two commits closed the three D7 gaps:

- `e4b269c` — historical reproduction environment preservation commit: the 4
  previously-untracked scratchpad modules plus the working-tree versions of the 4
  tracked-but-modified modules the closure requires.
- `a90de59` — narrow `-text` byte preservation for the guard-hash-sensitive set.

### Proof (all conditions mechanically required, all passed)

Run from a **completely clean** worktree created directly from `a90de59` — no files
copied in, no WIP overlay, no manual module restoration, no manual EOL rewriting.

| # | condition | result |
|---|-----------|--------|
| 1 | fresh checkout succeeds | exit 0 |
| 2 | all 8 guard-required hashes match, zero manual byte repair | 0 mismatches |
| 3 | `verify_historical_snapshot()` explicitly invoked (module has no `__main__`) | returned cleanly, exit 0 |
| 4 | evaluation's own exit code (not a pipe's) | `PYTHON_PRODUCER_EXIT=0` |
| 5 | full 121-episode run completes | 101 obstacle + 20 open |
| 6 | saved reference JSON byte-unchanged after run | `f30f63e4…` unchanged (worktree and main tree) |
| 7 | fresh result sha256 == reference sha256 | both `f30f63e4ef5ab97648e19c499b82fdb9cae537962097a52bcf2423e00c21bb30` |
| 8 | all 34 semantic checks pass | 34/34, 0 failed |

Duration 4m37s (00:58:57 → 01:03:34). Result: v2 PASSES — A 2 collisions → D 1,
strict subset, zero timeouts and zero planner failures on both sides, open stratum
per-episode identical 20/20.

The temporary runner inside the disposable worktree differed from
`scratchpad_router_v2_final_confirmation_820M.py` by **exactly one line** (line 170,
the output destination), verified by diff. The worktree was then removed.

### Required statement of what this tag does and does not claim

> "This tag is a post-hoc verified reproduction environment for the 2026-08-15 820M
> result. It is not claimed to be an exact reconstruction of an original Git commit,
> because some execution dependencies were uncommitted when the result was
> originally generated. Its validity is established by the frozen byte hashes and a
> byte-identical rerun from a clean checkout, not by claiming to reconstruct a commit
> that never existed."

---

## D9b. How the 820M execution closure was derived (and what it found)

`HISTORICAL_REPRODUCTION_CLOSURE.tsv` (37 rows: 34 modules + 3 resources) was
derived, **not assumed**, from three independent sources:

1. **Static analysis** — recursive repository-local import traversal from
   `scratchpad_router_v2_final_confirmation_820M.py`, resolving each import against
   the repo's own sys.path roots and following it only when the resolved path lies
   inside the working tree. Yielded 30 modules.
2. **Runtime tracing** — a successful 820M run from the full working tree, dumping
   every `sys.modules[…].__file__` inside the repo at the end of the run. Yielded 36
   modules.
3. **Non-module resources** — `builtins.open` and `Path.read_*` were instrumented to
   record every repo file actually opened. Yielded the 820M manifest, the frozen
   snapshot JSON, and the checkpoint zip.

The in-repo `.venv/` is excluded: it is third-party code, not repository source.

**Why both halves were necessary.** `simulator/split_branch_policy.py` is
`runtime_loaded=yes` but `statically_reachable=no` — it is never imported by any
source file in the closure; it is resurrected from the checkpoint pickle's
`policy_class`. A purely static analysis would have missed the single most
important module in the reproduction. Conversely, static analysis contributes
modules that runtime tracing alone could miss on a branch not taken.

**Assumption that turned out to be wrong.** The earlier working hypothesis was that
the closure's drifted set was `environment.py` + `navigation_history.py` +
`split_branch_policy.py` + the 4 scratchpad modules. The derived closure added a
file that hypothesis missed: **`simulator/world_model.py`**. This is exactly why the
coordinator required derivation over assumption.

State at derivation time (before the closure commit `e4b269c`) — 8 action-required
members out of 37:

| status | member |
|---|---|
| untracked | `scratchpad_beginner_routing_randomized_walls.py` |
| untracked | `scratchpad_beginner_routing_two_wall_s_route.py` |
| untracked | `scratchpad_routing_regression_fixtures.py` |
| untracked | `scratchpad_single_obstacle_transfer_eval_calibrated_arc.py` |
| tracked, content differs from HEAD | `simulator/environment.py` |
| tracked, content differs from HEAD | `simulator/navigation_history.py` |
| tracked, content differs from HEAD | `simulator/split_branch_policy.py` |
| tracked, content differs from HEAD | `simulator/world_model.py` |

The other 29 members were already tracked and unchanged — the closure did **not**
require committing the broader WIP. After `e4b269c` and `a90de59`: 0 untracked,
0 content differences.

Three members remain EOL-only-different from their blobs
(`router_v2_historical_reproduction_snapshot_20260815.json`,
`scratchpad_beginner_routing_two_wall_s_route.py`,
`scratchpad_routing_regression_fixtures.py`). None is guard-hashed, and the
clean-worktree proof in D9 passed with them as-is, so no `-text` rule was added for
them — scope discipline over speculative hardening.

## D10. HARD CONSTRAINT — checkpoint compatibility is not only the policy class

The non-executing pickle scan of all 313 checkpoints found repository-local module
references **outside** the `policy_class` field. These must be preserved as frozen
compatibility paths with **weight equal to the policy-class constraint**, not
merely noted:

| Field | Reference | Checkpoints |
|---|---|---|
| `rollout_buffer_class` | `farming.sb3_training.TerminalPrefixRolloutBuffer` | 2 |
| `session_boundary` | `farming.sb3_training.TrainingBoundary` | 2 |
| `session_boundary` | `farming.sb3_training.TrainingBoundaryKind` | 2 |

alongside the policy classes `simulator.split_branch_policy.SplitSteeringNavigationPolicy`
(275) and `simulator.split_branch_policy.SplitSteeringEventPolicy` (5).

Consequence for a future phase: moving/renaming `farming/sb3_training.py` or those
three symbols breaks deserialization of those 2 checkpoints exactly as moving
`simulator/split_branch_policy.py` would break 280. Any checkpoint-compatibility
gate must cover **both** module paths.

Evidence: `CHECKPOINT_INVENTORY.tsv`, `CHECKPOINT_MODULE_REFERENCES.tsv`.

## D11. HARD CONSTRAINT — the historical guard is a KNOWN-INCOMPLETE closure check

`scratchpad_historical_reproduction_guard.py` verifies byte-identity of 6 files
(+2 manifests via `extra_files`). It is now **known** to be an incomplete
dependency-closure check: it hashes `scratchpad_beginner_navigation_mix_pools.py`
but none of that file's own untracked dependencies, and it does not hash
`simulator/split_branch_policy.py`, `simulator/navigation_history.py`, or
`simulator/environment.py`, all of which are load-bearing for loading and running
the checkpoint.

**Passing the guard alone must never again be treated as sufficient proof of
reproducibility.** Future reproducibility claims must cite all three of:

1. the proven git tag,
2. `docs/migration/HISTORICAL_REPRODUCTION_CLOSURE.tsv`, and
3. a clean-worktree rerun,

not "the guard passed."

Per instruction, `scratchpad_historical_reproduction_guard.py`'s `REQUIRED_FILES`
and the 2026-08-15 snapshot JSON are **left untouched** — they remain honest
historical records of what was believed sufficient at the time.
`HISTORICAL_REPRODUCTION_CLOSURE.tsv` is the new, separate record of the
actually-discovered closure.
