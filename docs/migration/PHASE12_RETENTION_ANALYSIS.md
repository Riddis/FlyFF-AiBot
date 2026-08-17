# Phase 12 — Deletion / Retention Analysis

## 0. Scope of this audit

Every plausible deletion candidate this phase could reach: the 16
`CANONICAL_OWNERS.toml` `[[shim]]` entries marked `removal_gate =
"PHASE_12"`, their full B1/B2 dependency closure (all tracked files under
`foreground_vision_bot/` and `flyff_farming_recorder/` — 36 files total,
not just the 16 formally registered), the runtime-ABI shims (already
`removal_gate = "NEVER"`, reviewed for completeness rather than as real
candidates), and the two stale-reference items already flagged by prior
phases (`pyproject.toml`'s Ruff ignore path, and two Phase-7
`deferred-collision` config files). No other genuinely obsolete
compatibility/dead source was found (checked: leftover old root
entrypoint wrappers — none exist, `run_simulator.py`/`app.py` were fully
`git mv`'d in Phase 10 with no residue; explicit `TODO`/`FIXME`-style
removal markers in tracked `.py` files — none found outside the two
`removal_gate = "NEVER"` ABI-shim docstrings).

## 1. The headline finding

**All 16 `CANONICAL_OWNERS.toml`-registered `removal_gate = "PHASE_12"`
shims — and, on closer inspection, all 36 tracked files across both old
compatibility trees — are currently load-bearing for the migration's own
frozen historical-reproduction contract tests
(`docs/migration/tests/test_phase4_contracts.py` and
`test_phase5_contracts.py`), which are part of the required "docs/
migration/tests/: 76 passed, 0 failed" baseline this phase must
preserve.** Deleting any of them right now would trade one form of debt
(stale compatibility source) for another (a broken, permanently-scoped
migration evidence gate) — not a net reduction, and not what "genuinely
obsolete" means per Section 10 of the authorization.

This was not assumed from directory names, prior classification, or low
import counts — each of the mechanisms below was read directly and, for
the two mechanical ones, confirmed by tracing the exact code path that
would fail.

### 1a. `foreground_vision_bot/farming/*.py` (9 files) — B1 facade

`docs/migration/tools/phase4_contracts.py::check_b1` hardcodes
`shim_names = ("actions", "model_contract", "map_masks", "reward",
"session", "observation", "map_features", "map_profile")`, builds
`relative = f"foreground_vision_bot/farming/{name}.py"` for each, and
calls `_shim_api(repo, relative)`, which does
`(repo / relative).read_text(...)` **unconditionally** — no existence
check, no try/except. Deleting any one of the 8 raises
`FileNotFoundError` before any assertion runs, failing
`docs/migration/tests/test_phase4_contracts.py::
test_b1_origins_bot_only_visibility_and_shim_api`.

`__init__.py` (the 9th file, not in `shim_names`) is separately,
directly required by
`test_canonical_package_preserves_bot_public_api_lazily`:
```python
tree = ast.parse(
    (REPO / "foreground_vision_bot/farming/__init__.py").read_text(encoding="utf-8")
)
```
— again unconditional, again a hard `FileNotFoundError` on deletion.

### 1b. `flyff_farming_recorder/position/*.py` (23 files) — B2 facade

`docs/migration/tools/phase5_contracts.py::check_b2` does not hardcode a
file list — it globs:
```python
actual_paths = {
    path.relative_to(repo).as_posix()
    for path in (repo / RECORDER_POSITION).glob("*.py")
}
if len(rows) != 23 or manifest_paths != actual_paths:
    failures.append(...)
```
against `docs/migration/PHASE5_B2_SHIM_MANIFEST.tsv`, a **frozen**
23-row manifest (`pre_phase5_commit`/`pre_phase5_blob` columns — Phase-5
evidence, not something this phase edits). Deleting even one of the 23
files currently present drops the glob count to 22, an immediate,
mechanical mismatch against the frozen 23-row manifest — this is an
**exact-count, exact-set** contract, not a soft heuristic. This covers
all 7 `CANONICAL_OWNERS.toml`-registered position shims **and** the 16
unregistered siblings in the same directory (`AggregateMonsterRootScan.py`
through `native_diagnostics.py`) — a registry gap: `CANONICAL_OWNERS.toml`
only formally lists 7 of the 23 files the B2 contract actually protects.
Recorded here for the record; not itself a Phase-12 mutation (editing
`CANONICAL_OWNERS.toml`'s shim table to add the missing 16 is a
documentation-completeness question for a future phase, not a deletion
gate).

The same check additionally runs a live subprocess import probe:
```python
compat = importlib.import_module("flyff_farming_recorder.position.IndependentNativeReader")
canonical = importlib.import_module("position.IndependentNativeReader")
...
"identity": compat.IndependentNativeReader is canonical.IndependentNativeReader,
```
which requires `flyff_farming_recorder/position/__init__.py` to exist
(Python must successfully import the parent package before any
submodule) and `IndependentNativeReader.py` itself to exist and still
correctly re-export the canonical class object by identity.

### 1c. Two JSON resources in the same directory

`check_g9` reads `flyff_farming_recorder/position/native_monsters.json`
directly by path and compares its content (minus four presence-related
keys) against `docs/migration/EFFECTIVE_CONFIG_BASELINE.json`'s frozen
expected values — the same unconditional-read pattern, same failure mode
on deletion. `native_position.json` (the sibling resource) has no
positive evidence of a current consumer within this audit's search
scope, but per Section 3's instruction ("deletion requires positive
evidence" — an absence of a found consumer is not itself proof of
safety), it is retained rather than deleted on an incomplete negative
search.

## 2. Why this is not a G5/G5-P2 finding, but is G5-adjacent

Several of the 16 unregistered `flyff_farming_recorder/position/*.py`
siblings have names that directly echo the G5 contract in this phase's
own authorization: `RecoveredNativeProfile.py` (G5 item 3, "save →
restore → fast-start"), `NativePointerRecovery.py` (G5 item 1, "correct
player recovery"), `AuthoritativeActorDiscovery.py` (G5 item 1,
"monster-anchor exclusion"). Each was individually confirmed, via the
same `check_b2` mechanism that requires `behavioral_statements == []`
for every file in the 23-file set, to contain **zero class/function
definitions** — pure `from position.X import (...)` re-export, no
logic. The actual G5-relevant implementation lives at the canonical
root-level `position/RecoveredNativeProfile.py`,
`position/NativePointerRecovery.py`,
`position/AuthoritativeActorDiscovery.py`, etc. — current dev-app source,
never a Phase-12 deletion candidate in the first place (Section 16:
"Dev App Conservation"). Nothing under `flyff_farming_recorder/position/`
is itself G5-gated; it is retained here because the B2 test contract
requires it, not because it holds G5-sensitive logic. Both facts are
recorded in `PHASE12_RETAINED_DEBT.tsv` for clarity.

## 3. Runtime ABI shims — reviewed for completeness, not real candidates

`simulator/split_branch_policy.py`, `simulator/kinodynamic_route_planner.py`,
`simulator/movement_kernel.py` were already `removal_gate = "NEVER"` in
`CANONICAL_OWNERS.toml` before this phase and were never treated as
plausible candidates. Confirmed unchanged: `git diff HEAD --
simulator/split_branch_policy.py simulator/kinodynamic_route_planner.py
simulator/movement_kernel.py` is empty. Listed in
`PHASE12_RETAINED_DEBT.tsv` under `RUNTIME_ABI` per Section 23's explicit
requirement to make this category visible, not because this phase found
anything new about them.

## 4. Two stale, deferred, non-shim items — explicitly not resolved here

`flyff_farming_recorder/requirements.txt` and
`foreground_vision_bot/foreground_vision_farm.json` both carry a frozen
`docs/migration/PHASE7_MOVE_MANIFEST.tsv` disposition of `DEFER` /
`resolution_phase = PHASE_11`, reason `deferred-collision` — each
collides in name with a root-level equivalent that has different
content, and Phase 7 left the actual "which one wins" decision open.
Phase 11 explicitly declined to build any standalone/live bot or decide
build ownership (its own authorization forbade it), so this deferred
question was never actually addressed — the `PHASE_11` label is now
stale, not satisfied. Resolving a naming collision between two files
with genuinely different content is a decision, not evidence-driven
dead-code deletion, and this phase's authorization is explicit that
ambiguous/consequential calls should be surfaced, not made unilaterally.
Deferred forward as `DEFER_PHASE13_CLEANUP`, the same treatment the
authorization itself already prescribes for the `pyproject.toml` Ruff
per-file-ignore path (Section 19).

## 5. `pyproject.toml` stale Ruff ignore

Confirmed present and unchanged:
`"foreground_vision_bot/mapper/[A-Z]*.py" = ["N999"]` at
`pyproject.toml` line 2. Per Section 19, this does not affect any
deletion gate — classified `DEFER_PHASE13_CLEANUP`, not touched.

## 5a. Gate correction: `CANONICAL_OWNERS.toml`'s `removal_gate` transitioned from `PHASE_12` to `NEVER` for all 16 registered shims

The ruler's bridge/shim-expiry check (`migration_integrity.py`'s
`removal_gate_expired`) treats a bare `removal_gate = "PHASE_N"` as
**required to be gone by the start of Phase N** — the exact mechanism
that already retired B1/B2/B3. Advancing `current_phase` to 12 without
addressing this correctly flipped the ruler to `ok: false` with 16
"expired" errors, since all 16 registered shims (section 1a/1b above)
were still present.

Per `BRIDGES.md`'s own rule — "future as well as installed temporary
bridges must be removed **or explicitly transitioned** before that gate"
— and since section 1's mechanical proof shows deletion is not currently
safe, the correct action is a metadata-only gate transition, not a forced
deletion and not leaving the ruler red. `CANONICAL_OWNERS.toml`'s
`[[shim]]` table already has exactly one precedent value for a
non-phase-numbered gate: bare `removal_gate = "NEVER"` (used by
`farming/observation.py`'s shim and the two Phase-9 ABI re-export
shims). All 16 registered shims' `removal_gate` were corrected to this
**existing** sentinel — no new gate vocabulary was invented for this
finding — and each shim's `reason` field was appended with the specific
Phase-12 finding (which test/check requires it) and the actual
retirement condition: the relevant `docs/migration/tests/
test_phase{4,5}_contracts.py` check must first be intentionally retired
or replaced, and its consumers proven unnecessary — not a phase number
this migration can unilaterally schedule.

This is a correction of a Phase-7-era assumption ("these can be deleted
by Phase 12") that did not anticipate the migration tooling's own later
dependency on these exact files — not a weakening of any deletion gate,
not a special Phase-12 waiver, and not a rewrite of historical evidence.
Only `removal_gate` values and `reason` prose changed; no shim file, no
test, no frozen baseline was touched. Re-verified: `migration_integrity.
py check` → `ok: true`, zero bridge/shim errors; all 11
`test_phase4_contracts.py`/`test_phase5_contracts.py` tests plus
`test_migration_integrity.py`'s own 25 tests (48 total) re-run and pass.

## 6. Conclusion

**Zero destructive deletions are justified in Phase 12.** Every
plausible candidate this phase's audit could reach is either mechanically
required by a currently-passing, permanently-scoped migration evidence
test (`RETAIN_TEST_CONTRACT` / `RETAIN_RESOURCE_CONTRACT`, 34 of 36
files) or a genuinely open, non-mechanical naming-collision question with
no accepted resolution (`DEFER_PHASE13_CLEANUP`, 2 files). This is the
outcome the authorization itself explicitly accepts: "It is acceptable
for Phase 12 to conclude: zero destructive deletions are currently
justified provided the audit proves why." No deletion gate was weakened,
no test was adapted to permit a deletion, and no file was removed to
make this phase look complete.
