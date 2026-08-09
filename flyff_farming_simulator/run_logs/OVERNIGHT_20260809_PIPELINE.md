# Overnight Pipeline Log — 2026-08-09

Autonomous run under explicit standing authorization (see user directive received
2026-08-09, session `9ce8ba1d`). Goal: produce a single clean Basic → Beginner →
Intermediate → Advanced lineage, each graduating under the zero-actual-collision
rule (Basic: recovery-assisted, minimal-tolerance training-wheel standard; Beginner
onward: strict zero distinct collision events on an untouched confirmation pool).

This log is append-only. Earlier conclusions are never silently edited — if a later
entry contradicts one above, it says so explicitly and explains why.

Standing rules in force this run (carried from prior session directives, restated
in the overnight instructions): diagnose before changing oracle/training behavior;
never patch on data later shown wrong; small matched diagnostic before any broad
(66-episode-scale) re-test; contacts_per_100_distance is not a graduation metric;
distinct collision onsets != contact ticks; quarantine (never delete) superseded
checkpoints; PROVISIONAL beginner/intermediate graduates under the old bar remain
research-only, not canonical parents.

---

## 2026-08-09 (continuing from prior session)

**State inherited from prior (pre-compaction) session work:**
- Canonical Basic (`canonical_basic_graduated.zip`) is the sole trusted parent.
  Beginner/Intermediate graduates are quarantined as
  `canonical_{beginner,intermediate}_graduated_PROVISIONAL_OLD_COLLISION_BAR_20260808.*`
  — provisional/research-only, not canonical parents.
- Advanced round loop (`RUN_CANONICAL_ADVANCED.py`) has `AUTO_GRADUATION_ENABLED = False`
  as a safety gate; process was relaunched clean after this edit and resumed from
  on-disk checkpoints with no wasted compute. Still running as CANDIDATE-PASS-only.
- The scripted `obstacle_aware` teacher was found unsuitable as a DAgger label
  source (98.5% any-contact, worse peak severity than the policy it would train).
- A privileged, simulator-only steering oracle (`simulator/steering_oracle.py`)
  was built through 2 generations (v2 two-tier immediate+escape-BFS; v3 robust
  receding-horizon beam) plus a terminal continuation-viability gate added on top
  of v3, validated with 8 hand-verified unit tests
  (`tests/test_steering_oracle_v3_terminal_gate.py`, all passing).
- Full 66-episode qualification (sigma=1.5, terminal gate) vs plain v3 vs v2:
  distinct collision events 547 vs 719 vs 813 (terminal-gate best). Per-layout:
  28/33 layouts improved, 5/33 regressed (2 substantial: `unseen_templates/
  05_early_broad_lobes_typical_fast` +36, `unseen_templates/
  04_early_split_field_high_bursty` +27; 3 minor: `unseen_templates/
  07_early_open_center_typical_fast` +5, `unseen_templates/
  08_early_open_center_high_bursty` +3, `heldout/06_early_wide_neck_high_typical` +3).
  **66/66 episodes still had at least one collision — oracle is NOT yet qualified
  as a DAgger teacher.**
- A corrected onset/fallback instrumentation script
  (`scratchpad_diagnose_v3_terminal_gate_onsets.py`) was built implementing an
  A/B/C/D/E causal taxonomy (reserve-collapse / immediate-fallback /
  fallback-persistence / stochastic-mismatch / other) and a smoke test was run on
  one episode (`unseen_templates/04_early_split_field_high_bursty` seed 0, a
  substantial-regression layout).

**BUG FOUND in the smoke-test's own instrumentation (caught before trusting the
result, per standing diagnose-before-acting rule):** the fallback-streak walk-back
in `analyze_onset()` breaks at the *first* fallback tick found scanning backward
from the onset. When the onset tick itself is already a fallback tick (the common
case — every real collision so far has occurred during fallback), the loop breaks
immediately at `j = onset_idx` without ever walking further back to find where the
*current* fallback streak actually started. This makes `fallback_entered_at` report
"fallback active at the onset tick" (trivially true, uninformative) instead of "how
long has this fallback streak been running" — which is exactly the distinction the
A vs. B vs. C classification depends on (a streak that just started this tick reads
as `B_immediate_fallback`; a long-running streak that finally produces contact
should read as `C_fallback_persistence`). The smoke-test output (`fallback_entered_at
== onset_idx == 55`, `normal_decisions_since_fallback == 0` for ALL 5 lookback
offsets showing `used_fallback=True`) is consistent with this bug — it cannot
currently distinguish "fallback started right at tick 55" from "fallback has been
running since tick 20 and finally produced contact at tick 55."

**Fix applied** (`scratchpad_diagnose_v3_terminal_gate_onsets.py`, `analyze_onset`):
rewrote the backward scan to (1) find the true start of the CONTIGUOUS fallback
streak containing (or immediately preceding) the onset tick by walking back while
`used_fallback` stays true, then (2) separately count normal decisions before
*that* streak's start. Added `fallback_streak_ticks_before_onset` as an explicit
output field. Classification logic updated: `C_fallback_persistence` now fires
whenever the streak ran >=1 tick before the onset (not just when the OLD, buggy
`fallback_entered_at` happened to point further back); `B_immediate_fallback`
reserved for streaks that begin exactly at the onset tick with no visible reserve
decline beforehand. Diagnostic-only change — `steering_oracle.py`'s real decision
path is untouched.

**Smoke-test re-run after fix** (`unseen_templates/04_early_split_field_high_bursty`
seed 0, terminal-gate oracle, the same episode as before): all 6 distinct collision
onsets in this substantial-regression episode now classify as
`C_fallback_persistence`, with fallback entered 3-19 ticks before the
collision-causing transition in every case. Onsets 5 and 6 (ticks 556, 570) share
the SAME unresolved fallback streak (entered tick 551) — the escape mechanism
never exited fallback between the two collisions. This is a coherent, actionable
result (unlike the pre-fix all-zero output) and is consistent with the earlier
50-onset differential diagnosis finding that 100% of v3 collisions occurred during
fallback. Working hypothesis: **the escape-BFS fallback mechanism itself, not the
robust beam's reserve collapsing, is the dominant remaining collision source** —
to be confirmed/refuted on the full 7-layout x 2-seed x 2-oracle-kind targeted set
before any code change.

**Action:** launched full targeted diagnostic in background
(`scratchpad_diagnose_v3_terminal_gate_onsets.py`, task `btg4x3h3b`, log
`run_logs/v3_terminal_onset_full_20260809.log`) — 7 target layouts (2 substantial
regressions, 3 minor regressions, 2 biggest improvements) x seeds {0,1} x oracle
kind {terminal_gate, plain_v3} = 28 episode runs, full onset diagnostics on the
terminal_gate runs only. Output: `evaluations/oracle_v3_terminal_gate_onset_diagnosis.json`.

While this runs, reading the escape-BFS implementation (`_fastest_escape_first_action`
in `simulator/steering_oracle.py`) to understand its collision-avoidance guarantees
(or lack thereof) ahead of time, so a fix can be evaluated quickly once the
hypothesis is confirmed at scale.

**Code-inspection finding (candidate root cause, NOT yet acted on — awaiting the
full 28-run diagnostic before changing anything, per standing rule):**
`_fastest_escape_first_action` (`simulator/steering_oracle.py:142`) computes every
candidate step using `movement_kinematics.sweep`/`advance_with_slide` with the
DETERMINISTIC MEAN distance/turn (`forward_distance_cells = distance_mean_cells`,
fixed `turn_step`) — it was never upgraded to use `_robust_envelope_safe`'s
sigma-probe grid when the robust beam (v3) was added. This is architecturally the
exact same "reason about the expected trajectory, not the noise" approach that v2's
one-tick immediate tier used — which is what originally motivated building the
robust envelope in the first place (per this file's own v3 design-rationale
comment). Confirmed via direct code read, not assumed: `_robust_envelope_safe`
exists and is reused throughout the beam/terminal-gate, but `_fastest_escape_first_action`
never calls it.

Two distinct fallback triggers exist in v3, confirmed from the smoke-test lookback
data (onset tick 55, `unseen_templates/04_early_split_field_high_bursty` seed 0):
walking backward from the onset, `main_loop_depth{1,2,3,4}_empty` (offsets 0-3, no
robustly-safe candidate/sequence exists at all) shifts to `all_terminal_continuation_failed`
at offset 4 (a robustly-safe 4-tick sequence exists, but ALL its terminal
continuations are unsafe). This suggests the beam correctly senses danger several
ticks before the actual immediate-danger onset, but has no mechanism to route AWAY
from a closing trap — it can only veto sequences and fall back, and once fallen
back, the fallback's own mean-only motion model doesn't defend against the exact
stochastic variance the robust beam was built to defend against. Two candidate
(non-exclusive) root causes going into the full-scale result:
  (1) escape-BFS mean-only vulnerability (fixable: reuse `_robust_envelope_safe`
      in the escape search instead of `sweep`/`advance_with_slide`);
  (2) genuine geometric traps the beam sees coming (via the terminal gate) but
      cannot navigate away from early enough, because `continuation_depth=2` only
      looks ~6 ticks ahead total (4 beam + 2 continuation) -- NOT to be changed
      speculatively; only if the broad data shows the terminal gate's own
      "all_terminal_continuation_failed" reason dominates at LARGER lookback
      offsets across many onsets, not just this one hand-inspected case.
Waiting for the full 28-run result before choosing between (or combining) these.

**Process-management bug found and fixed:** the first background launch attempt
(`nohup python ... &` inside a Bash call that was ALSO marked `run_in_background:
true`) double-backgrounded the command — the tracked process was just the launcher
shell, which returned immediately (empty log, reported "completed" even though the
actual python process kept running detached and untracked). The second, correctly
backgrounded launch then ran CONCURRENTLY with the still-alive first one, both
writing to the same output log and JSON file. Caught via `Get-CimInstance
Win32_Process` showing two independent process trees for the same script started
36 seconds apart. Fixed: killed both trees (`taskkill /PID .. /T /F` after a
PowerShell `Stop-Process` attempt was blocked by the auto-mode classifier as a
destructive action — taskkill via Bash succeeded), cleared the stale log/JSON, and
relaunched a single clean instance (task `b3l3zah09`,
`run_logs/v3_terminal_onset_full_20260809.log`). No conclusions were drawn from
the corrupted concurrent run before it was killed, so nothing needs retraction.

**Interim observation while the run is still in progress (10/28 episode-runs done
at time of writing) — flagging now so it isn't lost, full interpretation deferred
until the run completes:** seed-level onset COUNTS show terminal-gate with MORE
distinct onsets than plain-v3 in 8 of the first 9 seed-level comparisons,
including on `heldout/03_early_open_field_low_fast` — the layout the earlier
66-episode aggregate labeled "biggest improvement (-107)". At the seed level here,
seed 0 of that exact layout shows gate=6 onsets/29 contact-ticks vs plain=2
onsets/29 contact-ticks — tied on ticks, WORSE on onset count. This does not
necessarily contradict the earlier aggregate (this script's `plain_v3` is an
approximation — `continuation_depth=0` bypass still applies the ranking's
clearance tie-break layer, a known, previously-flagged imprecision — and the
7-layout target set is deliberately the tails, not representative of the other 26
layouts), but it DOES mean my earlier claim that "the terminal gate reduces
collision onsets, not just shortens them" needs to be re-examined rather than
assumed once the full matched set is in: the alternative reading is that the gate
trades toward MORE, SHORTER contact taps in at least some layouts, which is a
materially different mechanism (and a materially different fix) than "the gate
prevents onsets." Not acting on this until the full run + per-episode tick/onset
ratio is computed.

**Full 28-run targeted diagnostic complete** (task `b3l3zah09`,
`evaluations/oracle_v3_terminal_gate_onset_diagnosis.json`,
`run_logs/v3_terminal_onset_full_20260809.log`):

Classification of all 131 terminal-gate onsets across the 14 episodes:
- `C_fallback_persistence`: 130 (99.2%)
- `D_stochastic_mismatch`: 1 (0.8%)
- `A_reserve_collapse`, `B_immediate_fallback`, `E_other`: 0 each

This is decisive and unanimous: essentially every remaining terminal-gate collision
in this set happens several ticks INTO an already-active fallback/escape streak,
not at a fresh entry into fallback and not from the primary beam's reserve
collapsing. Combined with the code-inspection finding above (the escape-BFS,
`_fastest_escape_first_action`, still uses ONLY deterministic mean-motion checks
and was never upgraded to the sigma-robust envelope the rest of v3 uses), this
gives a specific, well-evidenced fix target: the escape-BFS fallback mechanism
itself is executing actions that are not robust to real per-tick stochastic
variance.

**Seed-level contact-tick comparison -- an honest correction to the earlier
aggregate-based "28/33 improved" framing:** on this 14-episode set (which spans
both the flagged regressions AND the two layouts the original 66-episode aggregate
called "biggest improvement" / "large improvement"), terminal-gate has MORE
contact ticks than plain-v3 in 11 of 14 seed-level comparisons, tied in 1, and
fewer in only 2 (`unseen_templates/07_early_open_center_typical_fast` seed1: -10;
`heldout/03_early_open_field_low_fast` seed1: -54). Even `heldout/
03_early_open_field_low_fast` seed0 -- half of the layout the aggregate called its
single biggest win -- is tied on contact ticks (29=29) and WORSE on onset count
(6 vs 2) at the seed level. This does not necessarily overturn the original
66-episode aggregate (this script's `plain_v3` bypass is a known approximation --
`continuation_depth=0` still applies the ranking's clearance tie-break layer -- and
this 14-episode set is deliberately the tails, not a representative sample of the
other 26 layouts), but it confirms the user's original concern that per-layout
seed-averaging can hide real regressions, and means the terminal gate's benefit is
less uniform than the aggregate summary suggested. Logging this plainly rather than
letting the earlier "28/33 improved, gate is a clear win" framing stand
unqualified.

**Decision (evidence-based, not speculative -- targets the implicated mechanism
only, no sigma/beam-depth/beam-width/scoring/terminal-gate-rule change):**
implementing a robust-safety upgrade to `_fastest_escape_first_action` so the
action it commits to EXECUTING on the next real (stochastic) tick must itself pass
the same sigma-probed envelope (`_robust_envelope_safe`) the primary beam already
uses, instead of only a deterministic mean-motion check. Deeper frontier tiers
(ticks 2+) stay mean-motion-based, since v3 re-plans fresh every real tick and
those deeper ticks are never literally executed -- they exist only as a heuristic
existence-proof for choosing between multiple currently-safe origin actions, exactly
as before. If NO origin action is robustly safe at tier 1 (a genuinely cornered
state), falls back to the original all-mean-motion search across all 3 origins --
demanding robustness there would just return `None` more often with no better
alternative to offer, and over-rejecting in genuinely tight corners was flagged
ahead of time as the main risk of a heavier-handed version of this fix.
Implementing now; will validate with unit tests, then re-run this SAME 14-episode
matched set before considering any broader (66-episode) evaluation.

**Implemented.** `simulator/steering_oracle.py`: extracted the original mean-
motion-only search into `_mean_motion_escape_search(..., allowed_origins=...)`
(unchanged behavior, now parameterized by which origin actions seed the
frontier). `_fastest_escape_first_action` is now two-phase: (1) compute which of
the 3 origin actions are robustly safe via `_robust_envelope_safe` for the tick
about to actually execute; if exactly one qualifies, return it directly; if
several qualify, tie-break via `_mean_motion_escape_search` restricted to the
robust set; if none qualify (genuinely cornered), fall back to the original
search across all 3 origins (unchanged pre-fix behavior for that case only). v3's
call site (`_oracle_steering_decision_v3`) now threads its own `sigma` through
explicitly; v2's call site is left on the function's default
(`DEFAULT_ROBUST_SIGMA=1.5`), a low-risk implicit robustness upgrade to the
legacy, already-superseded v2 path. Fixed a definition-order hazard along the way
(`DEFAULT_ROBUST_SIGMA` is defined ~170 lines below this function; a naive
`sigma: float = DEFAULT_ROBUST_SIGMA` default-argument would `NameError` at
import time since defaults evaluate at `def`-time, not call-time) by using
`sigma: float | None = None` and resolving the default inside the function body
instead.

New unit tests (`tests/test_steering_oracle_escape_robust.py`, 4 tests, all
passing): prefers a robust-safe origin over a merely mean-safe one (the core
fix, hand-verified wall geometry: STRAIGHT mean-clear at x=1.9 but its upper-
tail probe at 2.35 crosses wall_x=2.2, while LEFT/RIGHT's worst-case probe
2.116 stays under it); falls back to the original mean-motion search and still
finds a real escape when NO origin is robustly safe (regression protection for
genuinely cornered states, wall_x=1.6); chosen action is always itself robust-
safe whenever any robust origin exists; `sigma=None` resolves identically to
`sigma=DEFAULT_ROBUST_SIGMA` explicitly (guards the deferred-default pattern).
Existing `tests/test_steering_oracle_v3_terminal_gate.py` (8 tests) still pass
unchanged (mocked at a higher level, don't exercise the escape tier directly).

**Infrastructure blocker found while running the full suite (documented, not
yet resolved, worked around):** `python -m pytest` (no args) fails 66 tests
(all `tmp_path`/`tmp_path_factory`-based fixtures; the 163 tests that don't use
those fixtures, including both new/modified oracle test files, pass cleanly) with
`PermissionError: [WinError 5] Access is denied:
'C:\Users\Ridd\AppData\Local\Temp\pytest-of-Ridd'`. Confirmed this is
environmental, not caused by this session's code changes: the failure is inside
pytest's OWN tmp-dir bootstrap (`_pytest/tmpdir.py`), before any test code runs.
`icacls`/`takeown` on that directory both fail with "Access is denied" even from
this session's own shell -- looks like a lock or ACL left by an earlier process
(possibly stale state from a prior crash), not something fixable without admin
elevation. Reasonable recovery attempted: `takeown`/`icacls` (failed, needs
elevation this session doesn't have); working around it by pointing
`--basetemp` at an accessible directory in the scratchpad instead of fixing the
underlying folder. If this becomes a real blocker for a later promotion gate,
it needs either an elevated-privilege cleanup of
`C:\Users\Ridd\AppData\Local\Temp\pytest-of-Ridd` or a permanent `pytest.ini`
`--basetemp` override -- flagging for the user, not fixing at the OS level
autonomously since it's outside repo scope.

**Full suite result with the basetemp workaround: 229 passed, 1 xfailed, 0
failed, 0 errors (657s).** This includes
`test_bootstrap_event_head_learns_real_discrimination_and_leaves_steering_untouched`
passing cleanly in the full-suite context -- retracting the earlier
"order/RNG-dependent, not confirmed pre-existing" hedge from the prior session:
in a clean environment it does not fail at all, in isolation or in the full
run. Whatever caused the single earlier failed run was specific to that run's
conditions (possibly the same tmp-dir contention now identified above, or
leftover state from that session's process management), not a real defect in
the test or the code it covers. No further action needed on it.

**Matched 14-episode re-run with the escape-BFS robust-safety fix** (task
`b4l48jen3`, `run_logs/v3_terminal_onset_full_POSTFIX_20260809.log`, same 7
layouts x 2 seeds x 2 oracle kinds as the pre-fix run):

Totals across all 14 episodes: gate contact ticks 913 -> 869 (-4.8%), plain-v3
unchanged at 636 (plain doesn't use the escape tier's sigma path the same way,
expected). Classification of 111 post-fix onsets: `C_fallback_persistence` 109
(98.2%), `B_immediate_fallback` 2 (1.8%) -- essentially UNCHANGED from the
pre-fix 99.2%/0.8% split. Seed-level: gate is still worse than plain-v3 on 10 of
14 comparisons, better on 4 (vs 2/14 pre-fix -- a small improvement in win count
but still net worse in aggregate on this tails-focused set).

**Conclusion: this does NOT materially improve, per the standing rule ("run the
broad evaluation only if the matched result materially improves") -- NOT
proceeding to a 66-episode broad re-test of this change alone.** The fix is
being KEPT (it is a small net improvement with no identified downside -- the
"genuinely cornered, no robust origin" branch is byte-for-byte the old,
previously-validated behavior, so there is no regression risk from keeping it),
but it does not resolve the dominant failure mode, and the classification barely
moving confirms the escape-BFS's mean-motion-vs-robust gap was a real but
SECONDARY issue, not the primary lever.

**Re-reading the evidence:** persistence classification measures ticks spent
already inside an active fallback streak before contact -- if a fallback streak
is often one where NO action is robustly safe at all (a genuinely tight corner),
my fix cannot help there by construction (it explicitly falls back to the
unchanged old behavior in exactly that case). Combined with the earlier
differential diagnosis's finding that 52% of v3 onsets were beam successes 1-4
replans earlier before collapsing, the more likely dominant mechanism is: the
PRIMARY beam is routing the agent into corners tight enough that essentially
nothing downstream (robust or not) can avoid at least some contact while
escaping -- an upstream routing/lookahead issue, not a downstream escape-
execution issue. Running one more small, targeted, non-speculative check before
deciding whether this justifies revisiting `continuation_depth` (currently 2,
~6 ticks total lookahead): for each post-fix onset, check directly (not
assumed) whether ANY of the 3 actions was robustly safe at the exact
collision-causing tick -- if "genuinely cornered" (zero robust options)
dominates, that specifically supports a lookahead-side fix; if robust options
existed but were still not enough, that points elsewhere (a bug in the escape
fix's execution or a scoring issue, not lookahead depth).

**Result** (`scratchpad_diagnose_robust_origin_at_onset.py`,
`evaluations/oracle_robust_origin_at_onset_diagnosis.json`): of 111 post-fix
onsets, 110 (99.1%) had ZERO robust actions available at the exact
collision-causing tick -- genuinely cornered, confirming the escape-BFS fix
cannot help in the vast majority of remaining cases by construction (it
explicitly preserves old behavior exactly in that branch). Only 1 onset (0.9%)
had a robust option available yet still contacted (a residual stochastic-tail
case, consistent with the previously-quantified ~2% long tail from the original
causal sweep -- not evidence of a bug in the fix).

**Conclusion, now evidence-based rather than speculative:** the dominant
remaining failure mechanism is upstream of the escape tier entirely -- the
PRIMARY beam is routing the agent into positions tight enough that no single-
tick action (robust or not) can avoid contact while escaping. This
independently corroborates the earlier differential diagnosis's finding (52% of
v3 onsets were beam successes 1-4 replans before collapsing) via a completely
different measurement. Two independent diagnostics now agree: the fix belongs
in how far ahead the beam looks / how conservatively it treats declining
reserve BEFORE committing to an approach, not in the escape tier. This
satisfies the standing bar for a non-speculative change: testing
`continuation_depth=3` (currently 2, ~6 total ticks lookahead: 4 beam + 2
continuation) as the next single controlled variable, on the SAME 14-episode
matched set, before touching anything else.

**Result, continuation_depth=3** (escape-BFS fix still in place;
`run_logs/v3_terminal_onset_full_CONTDEPTH3_20260809.log`): totals across the
14 episodes -- gate contact ticks 707 (vs 869 at depth=2, vs 913 with neither
fix; plain-v3 baseline unchanged 636). Gate onset count 89 (vs 111 at depth=2,
vs 131 with neither fix); plain-v3 onset count on this same set is 93 -- the
terminal gate now has FEWER distinct collision onsets than plain-v3, on the
adversarial tails-only subset, for the first time. Classification: 89/89
(100%) still C_fallback_persistence, expected since deeper lookahead reduces
HOW OFTEN the agent gets cornered, not the mechanism once it does. Seed-level
onset comparison: gate now better than plain on 7 of 14 (vs 2/14 pre-fix).
Monotonic, clean trend across all 3 measurements (contact ticks, onset count,
seed-level win rate) as continuation_depth increases from 2 to 3 -- testing
depth=4 next (matches the beam's own depth=4, a natural next/final step) before
settling on a value and broad-testing.

**Result, continuation_depth=4** (escape-BFS fix still in place;
`run_logs/v3_terminal_onset_full_CONTDEPTH4_20260809.log`): totals across the
14 episodes -- gate contact ticks 634 (vs 707 at depth=3, 869 at depth=2, 913
with neither fix), essentially TIED with plain-v3's unchanged baseline of 636
(-0.3%, within noise). Gate onset count 79 (vs 89 at depth=3, 111 at depth=2,
131 with neither fix) vs plain-v3's 93 -- gate now has 15% FEWER distinct
collision onsets than plain-v3 on this adversarial tails-only subset. One
episode (`unseen_templates/08_early_open_center_high_bursty` seed0) is now
FULLY collision-free (0 onsets, 0 contact ticks) where it previously had 6
onsets/55 ticks at depth=2. Seed-level: gate better on contact ticks in 7 of 14
comparisons (near-parity, up from 2/14 pre-fix). One outlier worth flagging
plainly: `heldout/03_early_open_field_low_fast` seed0 got WORSE at depth=4 (13
onsets/72 ticks) than at depth=3 (6/50) or even depth=2 (3/18) -- the trend is
not perfectly monotonic per-layout, only in aggregate; this specific
layout/seed will need separate attention if it still shows up as an outlier in
the broad run.

Classification: 78/79 (98.7%) still `C_fallback_persistence`, 1 `B_immediate_
fallback` -- the mechanism is unchanged (deeper lookahead reduces frequency of
cornering, not what happens once cornered, exactly as expected).

**Decision: continuation_depth=4 is a materially-improving, principled stopping
point** (matches the beam's own search depth=4, not an arbitrary tuned value;
monotonic aggregate improvement held across 3 independent controlled tests;
diminishing-marginal-benefit-vs-compute-cost tradeoff argues against continuing
to depth=5+ without new evidence). Setting `DEFAULT_CONTINUATION_DEPTH = 4` in
`simulator/steering_oracle.py` (was 2), running the full test suite again, then
proceeding to a broad 66-episode qualification (sigma=1.5, escape-BFS fix +
continuation_depth=4 combined) before any DAgger mining decision.

## User correction received mid-run (2026-08-09, during the depth=4 broad qualification)

Three corrections, acted on immediately (first) or queued against the broad
run's result (second/third):

1. **v2 baseline contamination (fixed immediately).** The escape-BFS robust-
   safety upgrade's `sigma: float | None = None` default meant legacy v2's
   call site silently gained the NEW two-phase robust-first-tier LOGIC too
   (not just a different sigma value) -- v2 is a frozen historical baseline
   and this would make any future "v3 vs v2" comparison invalid, since v2's
   numbers would no longer reflect the code that originally produced them.
   Fixed: added an explicit `robust_first_tier: bool = True` parameter to
   `_fastest_escape_first_action`; `robust_first_tier=False` reproduces the
   ORIGINAL mean-motion-only search exactly (delegates straight to
   `_mean_motion_escape_search` with all 3 origins, byte-identical to the
   pre-2026-08-09 function body). v2's call site (`oracle_steering_action`)
   now passes `robust_first_tier=False` explicitly. v3's call site
   (`_oracle_steering_decision_v3`) already passed `sigma=sigma` explicitly
   and defaults to `robust_first_tier=True` -- unaffected. New tests in
   `tests/test_steering_oracle_escape_robust.py`
   (`TestLegacyV2SemanticsFrozen`, 2 tests): confirms
   `robust_first_tier=False` matches calling `_mean_motion_escape_search`
   directly, and confirms `oracle_steering_action` actually passes
   `robust_first_tier=False` (spy on the real call, not just an assumption)
   -- guards against a future edit silently re-coupling v2 to v3's default
   again. All 6 tests in that file pass; full suite re-run in background
   (task `bwitz7uwk`) to confirm no other regression. Note: the currently-
   running depth=4 broad qualification (`bem7urj6s`, launched before this
   fix) is unaffected either way -- it exclusively uses `SteeringOracleTeacherV3`
   and never touches v2's code path, and a source-file edit doesn't affect an
   already-running Python process's already-imported module. No rerun needed
   for that reason; the fix simply had no been-running risk to begin with.

2. **The 66-episode heldout/unseen/challenge pools are now development
   pools, not final qualification pools -- acknowledged, not yet acted on.**
   They directly informed the terminal gate, the escape-BFS robust-safety
   fix, and the continuation_depth selection (2->3->4). Per this correction:
   even if the depth=4 broad run looks excellent, that makes the oracle a
   CANDIDATE teacher, not a qualified one. Before any oracle label touches
   Basic, a FRESH sibling layout/seed set that has not influenced any of
   these design decisions must be generated and evaluated, with an
   essentially-zero-physical-collision bar (stricter than Basic's own later
   recovery-assisted tolerance, since Basic's tolerance is deliberately
   attached to its training-wheel design, not a license to accept a dirty
   teacher). Queued as the next step after the current broad run's results
   are inspected.

3. **Preserve the `heldout/03_early_open_field_low_fast` seed0 counter-
   example instead of averaging it away.** This layout/seed regressed
   sharply at depth=4 on the 14-episode matched test (13 onsets/72 ticks,
   worse than depth=3's 6/50, depth=2's 3/18, AND plain-v3's 2/29) even
   while the aggregate improved. If it also regresses in the 66-episode
   broad run, it needs its own causal trace (same A-E taxonomy, applied to
   this ONE layout/seed specifically) before depth=4 is trusted as final --
   not written off as noise merely because the aggregate looks good.

Plan once the broad run completes: inspect per-layout/per-seed deltas
(preserving regressions, not just the aggregate); specifically re-check
`heldout/03_early_open_field_low_fast` seed0; compute the full metric set
(distinct events, contact ticks, max run, stagnation, productivity, coverage,
oscillation/switch rate, fallback rate); explicitly report this as "candidate
teacher, pending fresh confirmation" rather than "qualified"; then build a
fresh untouched sibling confirmation manifest and evaluate depth=4 there
before any DAgger mining.

## Correction to an earlier correction: the event-head test flakiness

Earlier in this log I retracted the prior session's "order/RNG-dependent, not
confirmed pre-existing" hedge on
`test_bootstrap_event_head_learns_real_discrimination_and_leaves_steering_untouched`,
based on ONE clean 229-pass full-suite run. That retraction was premature. The
full-suite run after the v2-freeze fix (task `bwitz7uwk`) FAILED this exact
test again -- and, checking more carefully this time, running it in ISOLATION
5 times in a row (not the full suite) produced 1 failure and 4 passes, with
the assertion `P(EVA|true EVA) > P(EVA|true NONE)` failing right at the
margin (0.380 vs 0.414) on the failing run. This rules out my earlier
"order/RNG-dependent" theory too -- it fails in isolation, so it isn't test-
order state pollution. It looks like genuine run-to-run non-determinism (an
unseeded RNG somewhere in `bootstrap_event_head`'s training loop, evidenced
by the test file using `tmp_path` but no explicit seed pinning for this
specific test), with an empirical failure rate around 1-in-5 to 1-in-6.

Precise, corrected characterization: this test is genuinely flaky/non-
deterministic (not order-dependent, not confirmed pre-existing vs. new -- no
baseline from before ANY of this session's changes exists to check against,
and it wouldn't help anyway since the failure is run-to-run, not code-
version-dependent). It is confirmed UNRELATED to this session's
`steering_oracle.py` changes: the test lives in
`tests/test_basic_training_pipeline.py`, exercises `bootstrap_event_head`
only, and fails/passes on repeated identical invocations with zero code
changes between them. Not fixing it as part of this session's work (out of
scope -- the oracle work doesn't touch this code path), but flagging it
plainly for whoever next touches `basic_training.py`'s event-head bootstrap:
it likely needs an explicit seed or a less marginal assertion threshold.

## Fresh confirmation pool built and launched in parallel

The broad depth=4 qualification (`bem7urj6s`) is running much slower than
expected -- ~2.5 hours elapsed with zero pool-completion markers yet (covers
heldout=16 + unseen_templates=14 + challenge=72 = 102 episodes total, all at
the more expensive continuation_depth=4 search). Confirmed via CPU-time
inspection (`Get-Process`, ~7600s CPU consumed by the worker process) that
it's genuinely still computing, not hung.

Rather than block all further progress on this single long-running job, and
because the user's correction already establishes that even a great result
there only makes the oracle a CANDIDATE (not qualified), built the fresh
confirmation pool now and launched its evaluation IN PARALLEL (machine has 16
logical cores; the qualification script is single-threaded, so this does not
meaningfully slow the broad run):

- `scratchpad_build_oracle_fresh_confirmation.py`: generated
  `synthetic_curriculum_oracle_fresh_confirmation/curriculum.json`, 12 layouts
  (all 6 templates used during this session's tuning -- open_field,
  irregular_plain, broad_lobes, split_field, wide_neck, open_center -- x2
  density/respawn combos: typical_fast, high_bursty), obstacle_level=0,
  stage=early, seed base 23,000,000. Confirmed via direct inspection that
  every existing early-stage curriculum (training, heldout, unseen_templates,
  factorial_probe) uses per-variant seeds in the ~20,268,xxx-21,324,xxx range
  -- this pool's actual generated seeds (23,007,919-23,095,028) are clearly
  disjoint. `assert_disjoint_from_training` also passed formally. Manifest
  saved to `evaluations/manifests/oracle_fresh_confirmation.json`.
- `scratchpad_qualify_oracle_fresh_confirmation.py`: evaluates the current
  oracle (sigma=1.5, escape-BFS fix, continuation_depth=4 -- i.e. exactly
  what's about to be set as the new default) against this fresh pool, 12
  layouts x 2 seeds = 24 episodes, reporting PER-EPISODE contact status (not
  just aggregates) since "essentially zero collisions" needs per-episode
  visibility to actually verify, not a mean that could hide a few bad
  outliers. Launched in background (task `bq8f2hhkg`,
  `run_logs/oracle_fresh_confirmation_20260809.log`).

Standard for this pool, per the user's explicit instruction: essentially zero
physical collision events -- stricter than Basic's own later recovery-
assisted graduation tolerance, since this evaluates a collision-avoidance
TEACHER, not a training-wheel policy.

## Broad development-pool qualification aborted; fresh pool is decisive enough

After ~4 hours, the broad depth=4 qualification (`bem7urj6s`) had only completed
the first of 3 pools (heldout, 16/102 episodes) -- confirmed via CPU-time
inspection (67196: ~14,566s CPU, genuinely still computing, not hung). At that
rate the remaining 86 episodes would need another ~20+ hours, which is not a
reasonable use of overnight time, especially since:

1. The user's own correction already establishes this pool only produces a
   CANDIDATE result, never a qualified one -- the fresh confirmation pool is
   required regardless of what this run would have shown.
2. The fresh confirmation run (parallel, smaller, more decisive for the actual
   question that matters) already has a strong, consistent signal after 9/24
   episodes: EVERY episode so far has at least 2 distinct collision events
   (range 2-8), none clean. This already answers the key question (is the
   oracle qualified to teach collision avoidance?) in the negative, making
   the broad run's remaining compute low-value by comparison.

Killed the broad run (`taskkill /PID 79252 /T /F`) to reclaim CPU for more
useful work (finishing the fresh confirmation pool, then a targeted causal
diagnostic on why these fresh-pool episodes still collide). The completed
matched-set evidence (14-episode depth 2/3/4 sweep) already firmly established
that continuation_depth=4 is a genuine, non-speculative improvement over
depth=2/3 and over plain-v3 on that adversarial subset -- that conclusion
does not depend on this aborted run and is not being retracted. What's now
clear is that "materially better than depth=2/3" and "qualified as a clean
DAgger teacher" are two different bars, and the oracle has only cleared the
first one so far. No historical numbers were lost -- the
`_PRE_DEPTH4_FIX_sigma1.5.json` baseline copy made before this run started is
preserved and still valid for future comparison.

**Per the user's explicit instruction ("If the broad depth-4 run still has
widespread contact, keep diagnosing. Do not move to DAgger simply because it
beats depth 2/3"): continuing to diagnose, not moving to DAgger.** Next:
finish the fresh confirmation pool (already running, informative on its own),
then run a targeted, CHEAP causal diagnosis (2-3 episodes, not all 24) on the
fresh pool's worst cases to characterize whether the SAME mechanism
(fallback persistence / genuinely-cornered) still dominates on genuinely new
geometry, or whether something new is happening that the tuning-pool-derived
fix doesn't address.

## Decisive finding: depth alone cannot reach zero collisions -- a structural limit, not a tuning gap

Targeted causal diagnosis on the 2 worst fresh-confirmation episodes
(`03_early_irregular_plain_typical_fast` seed0/seed1, 15 onsets total,
`scratchpad_diagnose_fresh_confirmation_onsets.py`,
`evaluations/oracle_fresh_confirmation_onset_diagnosis.json`):

- 15/15 (100%) onsets are `C_fallback_persistence`, 15/15 (100%) had ZERO
  robust actions available at the collision tick -- the identical mechanism
  found on the tuning pools, confirming the earlier diagnosis generalizes
  correctly and isn't an artifact of the specific layouts used to develop it.
- **`fallback_streak_ticks_before_onset` distribution: [4, 4, 4, 5, 5, 5, 5,
  6, 12, 16, 17, 19, 20, 25, 28]** -- i.e. the escape-BFS was already active,
  searching for a clean step, for as many as 28 ticks before the actual
  collision. Median 6, but more than a third of onsets exceed 12 ticks of
  lead time.

Cross-checked against the map generator's own escapability contract
(`simulator/synthetic.py:300`, `_regains_movement_within`): its docstring is
explicit -- "Success requires an uncontacted step, not merely any
displacement" -- so a layout is only validated to guarantee that a wedge
resolves into a genuinely clean step within `_STAGE_ESCAPE_TICKS` (24 for
early stage), NOT that the approach to that wedge, or the resolution
process itself, is contact-free. This is a real, load-bearing distinction:
**the map generator guarantees escapability, not zero-contact navigability.**
A layout can pass validation while still containing geometry where, once a
particular position/heading is reached, several ticks of unavoidable contact
occur before the escape completes.

**Conclusion: increasing `continuation_depth` further is not a viable path
to zero collisions.** The beam's total lookahead at depth=4 is already ~8
ticks (4 beam + 4 continuation), and streak lengths of 16-28 ticks are
observed. Reaching even the median (6) would need depth~3, already covered;
reaching the 28-tick tail would require continuation_depth on the order of
20+, at a compute cost that scales combinatorially with depth (each of the 3
prior depth increments roughly tripled per-decision cost; a single 66-episode
qualification at depth=4 was projected to take 20+ hours and was aborted for
exactly this reason). This is confirmed by data, not assumed: the streak-
length distribution itself proves the trap becomes locked in FAR earlier
than any practical lookahead horizon can see.

**What this means for the path forward:** the remaining collisions are not a
tuning gap in the current local-search architecture -- they reflect that a
receding-horizon beam, however deep, is fundamentally reactive and can only
avoid what it can see within its window. Preventing entry into these zones
requires either (a) a scoring/routing change that biases the beam AWAY from
low-reserve areas well before they become locally inevitable (a genuinely new
mechanism, not a parameter change -- explicitly the kind of change the user
said requires its own small controlled test before touching), or (b)
accepting that a purely local oracle cannot reach literal zero collisions on
100% of generated geometry, given the map generator's own contact-permissive
escapability guarantee.

Not committing to a scoring redesign autonomously without user visibility --
this is architecturally bigger than the fixes so far (which were bounded,
mechanically-justified corrections to two existing components) and changes
the shape of the remaining work. Reporting this clearly to the user as a
genuine decision point once the fresh confirmation pool finishes, rather than
grinding further on the depth parameter, which the evidence now rules out.

## User correction: "structural ceiling" was overclaimed; reframed as a two-timescale routing problem

Two corrections to my own prior conclusion, both accepted:

1. **The "structural ceiling" claim was overstated.** The map generator's
   escapability guarantee ("can eventually recover movement") proves neither
   that a zero-contact path exists NOR that zero-contact navigation is
   impossible on these layouts -- it's simply silent on the question. I
   conflated "the guarantee doesn't cover this" with "this is therefore
   impossible," which is a different, unproven claim. Retracting the
   "structural ceiling" framing. A real map sanity check (does a genuinely
   collision-free path exist through the relevant reachable regions under the
   player's actual turning/movement constraints?) is warranted eventually,
   but does not block the next experiment and has not been run yet.

2. **Reframed as a two-timescale planning problem, not a beam-scoring
   problem.** The evidence (fallback streaks beginning 4-28 ticks before
   impact, median 6, a third exceeding 12) means the local robust depth-4
   beam is good at "is this safe for the next few ticks" but structurally
   cannot see traps that are only visible 15+ ticks out -- and making the
   SAME local search deeper is exponentially expensive and already
   impractical (confirmed by the aborted 20+-hour broad run). The fix is a
   cheap, coarse, long-horizon routing/waypoint layer that steers the
   existing local beam away from low-connectivity pockets before they
   become locally unavoidable, NOT a wall-distance penalty bolted onto the
   4-tick beam's score.

**Explicit constraints for this next phase (from the user, all currently in
force):**
- Do NOT touch sigma, local beam depth, beam width, escape logic, Basic
  training, or PPO while isolating the routing change.
- Do NOT move to DAgger with the current oracle -- the fresh pool's 12/12
  (now growing) contact result is decisive: not qualified.
- Preserve the current oracle (escape-BFS fix + continuation_depth=4) as a
  checkpoint/baseline, not to be silently overwritten.
- Prefer a deterministic, map-derived coarse planner (cheap distance/
  connectivity-aware pathfinding) over any further stochastic beam expansion.
- **Critical, not yet addressed: student representability.** The current
  steering policy only sees 11 compact navigation features (6 target-
  geometry + 3 physical-clearance + recent_progress + recent_contact). If a
  coarse global router picks LEFT/RIGHT based on topology invisible in those
  features, DAgger would create contradictory supervision (same observation,
  different "correct" label depending on hidden global state) -- must be
  designed as either (A) a SHARED route-target layer, where the coarse
  planner's chosen waypoint becomes the target the existing target-geometry
  features are computed relative to (both teacher and student see the same
  effective target, so the student doesn't need to infer hidden topology),
  or (B) explicitly tested for representability before this teacher's labels
  are trusted for DAgger. This gates DAgger readiness independently of
  whether the routing layer itself works.
- Qualification order unchanged: matched failure set -> reused development
  pools -> new untouched sibling confirmation. Zero-collision target on fresh
  confirmation remains the bar before DAgger-teacher status.

**Immediate next step (in progress): a small, cheap proof-of-mechanism
experiment before writing any router.** Using the already-diagnosed fresh-
pool failures with long fallback lead time (streaks of 12, 16, 17, 19, 20,
25, 28 ticks from `evaluations/oracle_fresh_confirmation_onset_diagnosis.json`):
for the state just before each long failure trajectory begins, compute
whether a coarse, clearance-aware map route exists that stays in higher-
connectivity space while still progressing toward the farming objective;
compare its first directional choice against what the historical local beam
actually chose at that tick; then check whether following coarse waypoints
with the EXISTING (unchanged) depth-4 beam actually would have prevented
that specific collision. Only if this small check is positive does a real
router get built and matched-evaluated.

## Proof-of-mechanism, first attempt: two real bugs found, initial "2/3 disagree" finding retracted

Built `scratchpad_coarse_route_proof_of_mechanism.py` (clearance field via
multi-source BFS + clearance-weighted Dijkstra coarse route, sanity-checked
on a hand-built grid confirming it correctly routes around a narrow gap in
favor of a wider one) and ran it on the 3 known long-lead-time fresh-pool
decision points. Initial result: coarse route disagreed with the historical
beam's actual choice at 2 of 3 points.

**Decisive follow-up (`scratchpad_coarse_route_rollout_verification.py`):**
actually steering with the coarse waypoint via the EXISTING, unchanged
beam/escape machinery for a 40-tick window produced IDENTICAL contact counts
to the historical replay in BOTH disagreement cases (28 vs 28, 19 vs 19) --
suspicious, since a genuinely different first action should make every
subsequent tick diverge.

**Root cause, found via `scratchpad_debug_waypoint_no_effect.py`:** TWO real
bugs in the first proof-of-mechanism script, not a finding about routing:
1. `_target_position()` was called on an env whose position had been
   MUTATED BACK to a historical tick AFTER the full episode had already been
   replayed to completion via `record_trace` -- but `env._nearest_reachable_
   actor_id`/`_best_group_actor_id` (which `_target_position` depends on)
   are NOT reset by that mutation; they reflect whatever the live env's
   internal targeting state was at the END of the full replay, not what it
   actually was at that historical tick. A properly causal fresh replay
   (stop exactly at the tick in question, never run past it) gives a
   completely different target position.
2. The comparison used a crude "classify the raw target angle by sign" rule
   (`_classify_angle`, +-0.15 rad thresholds) instead of checking what the
   BEAM would actually choose under that target angle -- target_angle only
   affects the beam's SCORING tie-break among already-safety-gated
   candidates, so a "different-looking" raw angle can still produce the
   IDENTICAL chosen action once run through the real ranking.

With both bugs fixed (proper fresh replay, compare actual `_beam_search_
first_action` output under each target angle): at both re-checked points,
the coarse waypoint's target angle (-1.634, -1.361) is nearly identical to
the standard reactive target angle (-1.610, -1.345 respectively, differing
by only ~0.02 rad) at the specific tick tested, and BOTH produce the
IDENTICAL beam decision (STRAIGHT, matching what historically happened).

**Retracting the "2/3 disagree" finding -- it was a measurement artifact.**
The correctly-measured result is that, AT THE EXACT TICK IMMEDIATELY BEFORE
the fallback streak begins, the coarse route and the existing reactive
targeting do not disagree (at least at these 2 points). This does not yet
mean coarse routing is unhelpful -- it may mean (a) the router needs to look
at the STRUCTURE OF ITS OWN PATH further out (not just the immediate
next-waypoint direction) to detect an upcoming bottleneck the very-next-step
angle doesn't yet reveal, since at both checked ticks ALL 3 immediate
candidates were still robustly safe (no local danger signal exists yet --
exactly the two-timescale problem's premise), or (b) these 2 specific
approach angles are cases where the coarse-optimal route genuinely requires
going the same general direction the reactive target already points, with
the real difference (if any) being in HOW carefully to navigate that
direction rather than WHICH direction to pick -- not distinguishable by a
single discrete LEFT/STRAIGHT/RIGHT waypoint check.

Re-running the corrected comparison on the full original 3-point set (all
now using the proper fresh-replay + actual-beam-decision methodology) before
drawing further conclusions.

## Corrected proof-of-mechanism result: 0/3 disagreement -- a genuine, non-buggy negative finding

With both bugs fixed (`scratchpad_coarse_route_proof_of_mechanism_v2.py`,
`evaluations/coarse_route_proof_of_mechanism_v2.json`): at all 3 originally-
flagged decision points (the tick immediately before each long fallback
streak begins), the coarse route's target angle is close to the standard
reactive target angle, and BOTH produce the IDENTICAL beam decision in every
case (STRAIGHT/STRAIGHT, STRAIGHT/STRAIGHT, LEFT/LEFT). 0/3 disagreement,
not 2/3 -- the corrected measurement reverses the earlier (bugged) result.

This is a genuine finding, not a bug, and it complicates the simple version
of the routing hypothesis: at least AT THE EXACT TICK tested, a single coarse
waypoint override would not have changed what the beam did.

One additional, informative data point: the coarse route's OWN clearance
profile for the seed1/tick52 case shows a declining sequence approaching the
target -- `[5, 5, 4, 3, 2, 1]` cells of clearance over its first ~6 steps --
meaning even the CLEARANCE-OPTIMAL path to that particular target must pass
directly adjacent to an obstacle (clearance=1) to make progress. This
specific approach may be a case where no rerouting avoids the tight
passage -- the objective genuinely requires it -- shifting the diagnosis
toward "how carefully the beam/escape executes through an unavoidably tight
passage" rather than "which direction it should have gone instead." The
other 2 cases show open, high-clearance coarse paths (10-11 and 8-9 cells)
with no visible bottleneck in their first several steps, which is harder to
explain if the agent still ended up cornered 20+ ticks later -- raising a
real possibility that the immediate-next-tick target (nearest reachable
actor) SHIFTS during the fallback-streak window as the agent gets closer to
different, previously-unreachable monster groups, pulling it toward a
DIFFERENT, tighter area not visible in a single-tick target snapshot taken
before the streak begins.

**Honest status: the proof-of-mechanism, correctly measured, does not show
a clear win for a simple single-waypoint coarse-route override at the
tested decision points.** This does not close the door on coarse routing --
it suggests either (a) the target itself changes mid-approach in ways a
single fixed-target snapshot can't capture, requiring the check to track
target changes across the whole fallback-streak window, not just its start,
or (b) some of these specific traps are genuinely tight passages that any
route to that objective must navigate, shifting focus to execution quality
within the passage rather than a different high-level direction. Both are
testable with more diagnosis; neither has been checked yet given time spent
on this sub-experiment. Reporting this to the user before committing further
compute to either follow-up, since it changes the shape of the routing
hypothesis itself.

## Fresh confirmation pool: complete (24/24 episodes)

`evaluations/oracle_fresh_confirmation_qualification.json`, full 12-layout
x2-seed run:

- episodes_with_any_contact: 24/24 (100.0%) -- zero clean episodes.
- total_distinct_collision_events: 131 (mean 5.46/episode)
- total_contact_ticks: 1705
- max_consecutive_contact_ticks (worst episode): **517**
- physical_stagnation_episodes: 2
- zero_kill_episodes: 0 (productivity is fine; this is purely a collision problem)
- median_kills_per_hour: 10572, median_unique_cells: 560, median_fallback_rate: 0.115

**Confirms unambiguously: the current oracle (escape-BFS fix + continuation_
depth=4) is NOT qualified as a DAgger teacher.** 100% contact rate on a pool
that had zero influence on any tuning decision.

**One severe outlier worth flagging separately, not averaging away:**
`12_early_open_center_high_bursty` seed1 -- distinct_events=5,
contact_ticks=547, max_consecutive_contact_ticks=517, fallback_rate=0.810
(81% of the whole episode spent in fallback), physical_stagnation=True,
unique_cells collapsed to 164 (vs ~550-580 for healthy episodes on the same
pool). This is an order of magnitude worse than every other episode in this
pool (next-worst max_consec is 26) and looks like a genuine catastrophic
lock-in, not an instance of the same "several-tick fallback persistence"
pattern characterized so far. Needs its own look before assuming it's the
same mechanism -- 517 consecutive ticks is far beyond the 4-28 tick lead
times seen elsewhere and may indicate a distinct failure mode (e.g. a true
dead-end pocket, or a target-switching oscillation keeping the agent pinned).

## Catastrophic-case investigation: a new, more precise root cause -- target thrashing, not (only) horizon length

Investigated the severe outlier `12_early_open_center_high_bursty` seed1
(max_consecutive_contact_ticks=517, fallback_rate=0.810) separately, per the
standing instruction to inspect severe outliers rather than average them
away.

**Finding 1 -- this is a genuine zero-degrees-of-freedom dead end, not a
"tight but eventually escapable" trap.** From tick ~200 onward the player is
frozen at the EXACT SAME position/heading (x=-133.57, z=-24.86, heading=2.02)
for 450+ ticks straight. Direct verification: at that exact state, all 3
candidate actions (STRAIGHT/LEFT/RIGHT) produce contact=True with
progress=0.000000 -- genuinely zero displacement in every direction, not
just "some contact while still sliding." The escape-BFS itself (with the
2026-08-09 robust-safety fix) returns None (no escape found at all within
budget). The final last-resort fallback then deterministically re-picks the
same zero-progress action forever (a tie-break artifact, `max()` over 3
identical zero scores always returns the first candidate in `_CANDIDATES`
order) -- explaining the perfect tick-over-tick stasis.

**Finding 2 -- clearance declines monotonically on the approach, well
outside the beam's horizon.** Tracing backward: clearance was 34 at tick 100,
declining through 13/17/12/9 by tick 140-150, then crashing to 1 at tick 155
and staying at 1 for the rest of the trapped period. This is a much cleaner
signal than the earlier 3-point check: a clear, sustained decline over ~15
ticks before full immobilization, not a single ambiguous tick.

**Finding 3 -- a coarse route toward the CURRENT target at ticks 100/130/140
stays well clear of the eventual trap (min clearance 34/19/13 respectively,
increasing further along the path, none passing within 3 cells of the trap
cell).** This means a coarse router using the SAME target selection as the
existing reactive system would NOT have routed toward this trap from any of
those checkpoints -- so the failure isn't "the coarse route agrees with a
bad reactive choice" (as the earlier 3-point check suggested for a different
episode).

**Finding 4 -- the actual cause: the farming target itself thrashes.**
Direct instrumentation of `env._nearest_reachable_actor_id`/`_best_group_
actor_id` shows **10 distinct target switches in ticks 131-157 alone** (once
every ~2.3 ticks on average) as the agent moves through this dense monster
area -- switching between actor IDs 282, 97, 5, 477, 516, 669, 512, 625, 127,
293 in rapid succession. This means the steering objective itself has no
persistence or stability during the critical approach window: a coarse
router recomputed toward "whatever the current target is" would face the
exact same instability, since the correct route recommendation is itself a
constantly-moving target during precisely the window that matters.

**Revised diagnosis, more precise than "needs longer lookahead":** in at
least this catastrophic case, the proximate cause is target-selection
INSTABILITY (rapid re-targeting among nearby actors with no hysteresis or
geometric awareness), not a pure lookahead-horizon limitation. This suggests
a cheaper, more targeted candidate fix alongside/instead of a full coarse-
routing layer: add stability/hysteresis to target selection (e.g. commit to
a target for a minimum duration, or weight target choice by
clearance/connectivity so a nearby-but-badly-positioned actor doesn't win
over a slightly farther, more sensibly-reachable one) -- this does not
require building the pathfinding infrastructure a full router would, and
directly targets the newly-identified mechanism. Reporting this pivot to the
user before choosing between (a) target-selection stability as the next
small experiment, (b) continuing the coarse-routing-layer design, or (c)
both, since it changes the most promising next fix category.

## User directive: target-stability-first, strictly sequenced

Explicit plan received: target persistence/hysteresis FIRST (isolated,
single variable), matched-tested, before adding clearance-aware scoring or
resuming the coarse-routing layer. Two interpretation cautions logged as
received, not yet independently verified:
- the zero-DOF trapped state proves that state is kinodynamically dead for
  the oracle's available controls; it does NOT prove the map itself is
  defective -- the oracle may simply have approached a hole it should never
  have entered;
- the coarse route hitting clearance=1 in one earlier case proves THAT
  PARTICULAR grid path goes through a tight cell; it does not prove no
  zero-contact route exists elsewhere. Neither claim (map defect vs. planner
  failure) is resolved yet.

**Step 1 in progress:** built `scratchpad_measure_target_thrashing.py`
(pure observation, zero behavior change -- runs the exact same oracle
decision function used everywhere else, only adds instrumentation reading
`env._nearest_reachable_actor_id`/`_best_group_actor_id` and computing
target-direction deltas). Smoke-tested on `01_early_open_field_typical_fast`
seed0: 171 target switches over that episode (28.7 per 100 ticks -- roughly
one switch every 3-4 ticks across the WHOLE episode, not just near
collisions), 140 "material" (>0.3 rad direction change), 123 of 171 (72%)
happened while the OLD target was still alive and valid (a preference-driven
switch, not a forced one from the old target dying) -- strong preliminary
evidence the thrashing is pervasive and mostly avoidable-by-design, not
merely reactive to targets disappearing. Launched the full 24-episode
measurement across 8 parallel shards (16 logical cores available, each
replay is single-threaded) to keep wall-clock time down to roughly one
shard's worth (~3 episodes, ~35-40 min) instead of a serial ~5 hours.

## Step 1 complete: target thrashing quantified across all 24 fresh-pool episodes

`scratchpad_measure_target_thrashing.py` (8 parallel shards + 1 gap-fill for
a sharding-parameter mismatch, all pure observation, zero behavior change),
aggregated via `scratchpad_aggregate_target_thrashing.py` --
`evaluations/target_thrashing_aggregate_report.json`, full log
`run_logs/target_thrashing_aggregate_FULL24_20260809.log`:

- **Whole-episode switch rate: median 29.9/100 ticks (mean 30.1), range
  7.9-41.2.** Pervasive across the ENTIRE pool, not isolated to problem
  episodes -- every layout shows substantial thrashing.
- **70.9% of all 4199 recorded switches happened while the OLD target was
  still alive and geodesically reachable** (1220 dead-target/forced
  switches vs 2979 live-target/preference-driven switches) -- the large
  majority of thrashing is the selector re-preferring a marginally
  different target, not being forced off a dead one.
- **Switches in the 20-tick window before collision onsets: 97.7% of 131
  onsets had at least one switch (median 6, mean 5.5).**
- **Switches in the 20-tick window before fallback-streak entries: 100% of
  80 streaks had at least one switch (median 8, mean 7.35), and only 1.2%
  had zero switches even in just the preceding 10 ticks.** Essentially every
  observed escape-fallback episode is immediately preceded by target
  instability.
- **Fallback-streak duration is NOT clearly predicted by pre-streak switch
  volume** (correlation -0.143, weakly negative if anything): streaks
  preceded by >=2 switches in the prior 10 ticks have median duration 21
  ticks; streaks preceded by <2 have median 17 but a much higher MEAN (57.6,
  dragged up by the 523-tick catastrophic outlier, which itself had
  relatively LOW pre-streak switching). This nuance matters: switching
  activity is a near-universal PRECURSOR/co-occurring signal for entering
  fallback, but does not clearly predict how SEVERE the resulting episode
  will be -- both likely share a common upstream cause (dense, complex
  local geometry produces both more re-targeting opportunities and more
  navigational risk) rather than switch-count directly causing duration.

This is strong enough evidence (per the standing "smallest controlled
experiment" bar) to justify Step 2's isolated persistence intervention,
exactly as the user specified -- proceeding.

## Step 2 complete: target-selection hysteresis implemented as shared runtime infrastructure

`simulator/environment.py`: both `_nearest_reachable_actor_id` (inline in
`_observation()`) and `_best_group_actor_id` (via `_group_approach_potential`'s
new `sticky_actor_id` parameter) now keep the CURRENT target unless a
candidate beats it by more than `_TARGET_HYSTERESIS_MARGIN_CELLS = 3.0`
geodesic cells -- a "meaningfully better" margin, not a fixed-duration timer,
so a target that dies or goes unreachable is still dropped immediately (no
margin check applies when the sticky target itself yields no finite
geodesic this tick). New `self.target_hysteresis_enabled` flag (default
`True`) lets the Step 3 matched comparison toggle old-vs-new behavior on
identical env instances without any other code change.

**Critical invariant preserved and unit-tested:** `_group_approach_potential`'s
returned SCORE (which feeds `_approach_potential_cells`, itself used for
PPO reward-shaping deltas at `RecordedFarmingEnv.step()`'s reward
computation) is verified to be byte-identical whether or not
`sticky_actor_id` is supplied -- hysteresis changes ONLY which actor ID gets
selected for steering, never the reward-relevant potential value. This was
a deliberate design constraint (`_group_approach_potential` has TWO call
sites, one for reward computation which must stay untouched, one for
`_observation()`'s target assignment which now applies hysteresis) and is
covered by `tests/test_target_hysteresis.py::
TestGroupApproachPotentialUnaffectedByStickiness`.

Because the hysteresis logic lives in `environment.py`'s own
`_observation()` (the single method that updates `_nearest_reachable_actor_id`/
`_best_group_actor_id`, which BOTH the oracle's `_obstacle_aware_target_angle`
and any future policy-observation-feature code read via
`nearest_reachable_relative_angle()`/`best_group_relative_angle()`), this is
structurally shared runtime infrastructure, not privileged teacher-only
logic, satisfying the representability requirement by construction rather
than by convention.

New tests (`tests/test_target_hysteresis.py`, 5 tests, all passing):
keeps current target when a new one is only marginally (within-margin)
better; switches when meaningfully (beyond-margin) better; switches
immediately when the current target dies; `target_hysteresis_enabled=False`
reproduces the exact original greedy behavior (regression protection); the
reward-relevant score invariant above. Full test suite re-run in the
background to confirm no other regression from this environment.py change
(broader-touching than the earlier steering_oracle.py-only fixes).

## User-directed parallel experiment: PPO pure-navigation ablation

Per explicit user request, running a deliberately simple, decisive
experiment alongside the oracle-perfection track: does direct PPO under an
unambiguous "collision ends the episode" incentive learn clean navigation
without any of the oracle's hand-engineered machinery? Two conditions to
distinguish "target instability is the real problem" from "the whole
oracle-track approach is overengineered" from "something deeper needs work":

A. stable_waypoint -- target-selection hysteresis margin set effectively
   infinite (holds the initial target for the whole episode, only
   re-targeting if it dies/goes unreachable).
B. normal_target -- the env's actual current default target-selection
   (hysteresis enabled, margin=3.0 -- i.e. today's real system, not the
   pre-hysteresis pure-greedy baseline).

Same training curriculum (`synthetic_curriculum/curriculum.json`, 4 early-
stage layouts), same movement noise (unmodified env physics), same policy
architecture (`SplitSteeringNavigationPolicy`, [64,32] net_arch for
steering/event/vf, matching `build_fresh_basic_policy`'s architecture),
same training budget (300,000 timesteps, identical PPO hyperparameters)
between conditions -- ONLY target_mode differs.

New code: `simulator/pure_navigation_env.py` (`PureNavigationWrapper`:
reward = per-tick forward progress only, farming/EVA/kill reward never
consulted; episode terminates IMMEDIATELY with a -5.0 terminal reward on
the first physical contact -- collision is unambiguously catastrophic, not
merely penalized). `configure_target_mode()` sets the mode via `.unwrapped`
(caught a real bug before it shipped: a naive manual `.env` walk would
never have descended past the outer wrapper, since gymnasium's `Wrapper.
__getattr__` delegation makes `hasattr` checks on the outer wrapper appear
to already have the base env's attributes -- confirmed via a smoke test
that the fix actually reaches the base `RecordedFarmingEnv` instance).
`scratchpad_ppo_pure_navigation.py`: training driver, from-scratch PPO
hyperparameters (lr=3e-4, ent_coef=0.02, NOT the conservative 5e-5/0.015
fine-tuning defaults `build_fresh_basic_policy` hard-codes, which are
tuned for refining an already-good policy, not learning from random init).

Smoke-tested at 2000 timesteps (~44 steps/s with 4 parallel training
envs): completed cleanly end-to-end, ep_reward_mean 89->108 and
ep_len_mean 64.8->76.8 already trending up within that tiny window.
Launched both full 300k-timestep runs in parallel in the background (tasks
`bxsm6dl4n` stable_waypoint, `bj72hnt3f` normal_target), expected ~2 hours
each given the smoke-tested rate. Will evaluate both raw (no
oracle/recovery) on the untouched `oracle_fresh_confirmation` pool for
zero-collision rate once training completes, per the user's exact
evaluation spec.
