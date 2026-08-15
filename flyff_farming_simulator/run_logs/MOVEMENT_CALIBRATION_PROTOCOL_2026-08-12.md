# Controlled movement-calibration recording protocol (revised 2026-08-12, v3)

**v3 update**: the benchmark came back decisively positive -- 20,219 Hz
achieved, 100% read success (303,288/303,288 reads over 15s), over 500x
the ~40Hz threshold that would justify tap-based capture. Position-only
reads really are essentially free, exactly as the code analysis
predicted. Consolidating the two-track plan below into ONE session using
`calibration_capture.py --capture` covering the full duration spectrum
(taps through long holds) -- no reason to split across two tools when
one now covers everything at higher resolution. **Skip the "Step 2"
long-hold-via-old-recorder section below; use the single session at the
very bottom of this doc instead.**

**Revision note**: the original version of this document proposed ~360
short (~0.2s) taps as the primary trial. That was wrong and was caught
before being sent for execution: `fit_world_model` only accepts a sample
from a PAIR of consecutive world-frames, and the historical recordings'
world-frames arrived at a median ~0.355s interval -- a ~0.2s tap can
easily fall entirely between two world-frame samples and yield zero or
one usable frame, not the two needed. Two things were done before
reissuing this: (A) the two existing historical recordings were reparsed
at raw frame-pair level (below), and (B) the recorder's native-read
architecture was inspected for whether a much higher sample rate is
feasible. Both are done now; this revision reflects both.

## What the existing data already shows (zero manual effort, done)

Reparsed `recordings/training/SEND_TO_RIDDIMS_Riddims_20260803...zip` and
`..._20260805...zip` at the raw frame-pair level (reproduces the fitted
model's exact 1432/121/98 sample counts, confirming the reimplementation
is faithful). Findings:

- **Nearly all historical LEFT/RIGHT samples come from very short runs.**
  Of 121 LEFT and 98 RIGHT accepted pairs, none belong to a run longer
  than 3 consecutive pairs -- the historical data contains essentially no
  sustained (>~1s) LEFT/RIGHT holds at all. STRAIGHT, by contrast, has
  runs up to 8+ pairs (511 of 1432 samples). This means the current
  LEFT/RIGHT statistics are built almost entirely from brief taps/course-
  corrections, not steady turning -- we don't actually know what sustained
  turning looks like from this data.
- **Large fractions of LEFT/RIGHT samples show negligible turn.** Only
  53/121 (44%) of LEFT pairs and 31/98 (32%) of RIGHT pairs clear the
  0.05 rad/s filter used for the turn statistic at all -- more than half
  of the "LEFT" samples aren't meaningfully turning.
- **Distance and turn magnitude are positively correlated, not
  independent.** r ≈ +0.47 (LEFT, turn-filtered subset) and r ≈ +0.51
  (RIGHT, same, sign-corrected) between distance and turn magnitude. The
  current simulator samples these independently, and the robust
  kinodynamic router's motion envelope explicitly combines weak-turn with
  long-distance and strong-turn with short-distance as separate corners --
  exactly the combinations this correlation says are *less* representative
  of what actually co-occurs. This is a real, data-grounded candidate
  explanation for why the more "principled" variance-aware router
  underperformed the crude fixed-margin one.
- **The dt-normalization assumption (rescale speed to a fixed 0.20s
  interval) shows a real, if moderate, systematic bias for STRAIGHT**:
  median distance/0.2s decreases roughly monotonically from ~2.68 cells
  (dt 0.10-0.20s) down to ~1.70 cells (dt 1.0-1.5s) -- longer sampling
  windows measure systematically LOWER average speed, not just noisier
  speed. So variable dt does contribute a real bias here, on top of extra
  variance, even though the rescaling logic itself is sound in principle
  (per the correction above).
- **LEFT/RIGHT magnitude symmetry roughly holds**: filtered turn means
  0.249 (LEFT) vs 0.219 (RIGHT), distance means 1.102 vs 1.101 -- LEFT
  turning moderately more than RIGHT, consistent with earlier measurements.
- Full numeric output: `run_logs/movement_provenance_analysis_output.txt`.
  Raw reparsed samples (every accepted frame-pair, with run/session/dt
  metadata): `run_logs/movement_provenance_raw_samples.csv`. Reproduction
  scripts: `scratchpad_movement_provenance_reparse.py`,
  `scratchpad_movement_provenance_analysis.py`.

## Native-read feasibility (investigated, tool built)

Traced the recorder's actual architecture: `IndependentNativeReader.
read_player()` is a handful of small direct memory reads (not a scan) and
is ALREADY invoked every ~40Hz lifecycle-loop iteration today -- position
reading was never the bottleneck. The historical ~0.355s median frame
interval comes from the SAME loop iteration also running the full
actor/monster-slot scan (`reader.snapshot()`'s `_scan_monsters*`) every
tick, which is the expensive part; position and that scan are currently
coupled every iteration. A capture loop that skips the actor scan
entirely and only calls `read_player()` + the existing cheap keyboard/
focus checks should be able to sustain a much higher rate -- but the
actual achieved Hz can't be confirmed without running it live, so this
needs a quick real test before anything else.

**Built**: `flyff_farming_recorder/calibration_capture.py` -- a small,
read-only, standalone tool that reuses the exact same attach/read
primitives as the normal recorder (nothing new discovered/attached, no
game writes) but skips the actor-scan step entirely. Two modes:
- `--benchmark`: unpaced loop, reports achieved position-read rate and
  inter-sample timing. Run this first.
- `--capture`: paced loop at a target rate, writes a plain CSV
  (`elapsed_s,player_x,player_z,forward,left,right,jump,focused`).
  Heading is deliberately NOT computed live (no memory field for it
  exists anywhere in this codebase -- the normal recorder also only
  ever derives it from position deltas); it'll be derived afterward from
  the dense position trajectory during analysis, which is better-
  conditioned than a live threshold-gated estimate at unfamiliar rates.

## Step 1 -- run the benchmark now (about 2 minutes)

```
cd flyff_farming_recorder
python calibration_capture.py --benchmark --seconds 15
```

Same window-select/HP prompts as the normal recorder. Just stand in the
Tower AoE map (move around a little if you like, doesn't matter) for the
15 seconds. It attaches read-only, samples as fast as possible, and
prints achieved Hz plus a failure rate -- writes no file. **Send me the
printed result.** If it's comfortably above ~20Hz, single-tap (~0.2s)
trials become directly measurable and I'll follow up with a short
`--capture` session for exactly that. If it's much lower, we'll skip
tap-based capture and rely on Step 2 alone.

## Step 2 -- controlled long-hold recording (do this regardless of Step 1;
uses the normal existing recorder, not the new tool)

This directly targets what the historical data is missing: sustained
holds long enough to separate onset/transition behavior from steady-
state movement, and to check whether distance/turn scale linearly with
hold duration. Longer holds are also far less tedious than hundreds of
taps, since each hold yields several frame-pairs even at the recorder's
normal (slower) frame rate.

**Tool**: `flyff_farming_recorder/app.py` (or the prebuilt `.exe`) -- the
normal recorder, same as before. Attach Client -> Start Logging -> run
trials -> End Logging -> send the `SEND_TO_RIDDIMS_*.zip`.

**Keys** (forward held together with the turn key, not alone -- confirmed
from the recorder's own key-mapping logic):

| primitive | qwerty | azerty |
|---|---|---|
| FORWARD | `W` alone | `Z` alone |
| LEFT | `W`+`A` together | `Z`+`Q` together |
| RIGHT | `W`+`D` together | `Z`+`D` |

No jump during trials. EVA is fine to cast normally -- set the recorder's
own EVA-hotkey dropdown to an F-key/number-key you won't otherwise touch
during trials; it only watches that specific key, independent of your
real in-game EVA bind, so your real casts simply won't be seen (unless
your real EVA key happens to also be W/A/D/Space).

**Trial**: press and hold for a target duration, release, stand
completely still for at least 2 seconds before the next trial (that gap
alone gives clean separation in the recorded key-state stream -- no
marker needed). Face into open space (~20+ cells clear) before each
trial; no need to reset to a specific heading, only heading *change* per
trial gets measured.

**Duration targets** (aim by feel -- actual achieved duration is measured
from timestamps, not trusted from intent):

- **~0.7s** -- roughly the shortest a human can reliably hold and still
  land 2+ world-frames inside it at the historical delivery rate.
- **~1.5s**
- **~3.0s**

**Repetitions** -- much lighter than the original ask, since each hold
now contributes multiple samples:

| primitive | ~0.7s | ~1.5s | ~3.0s |
|---|---|---|---|
| FORWARD | 15 | 15 | 10 |
| LEFT | 15 | 15 | 10 |
| RIGHT | 15 | 15 | 10 |

120 trials total. Interleave rather than block (rotate F, L, R within
each duration phase). Discard/redo freely if a trial is interrupted --
extra trials don't hurt.

## What happens after you send anything back

For the benchmark: I'll tell you whether tap-based capture is worth
doing and, if so, give you the exact `--capture` command and duration.

For the long-hold ZIP(s): I'll parse `frames.msgpack.gz` directly and
separate each hold into onset (first pair of the run) vs. interior
(later pairs) samples, report distance/turn distributions for each
duration bucket, joint distance-turn correlation from genuinely
sustained holds (not just brief taps), and whether duration scaling is
linear. Compared directly against both the current fitted model and the
Part-A historical reparse above.

No changes to the simulator, router, or curriculum until this is done
and reviewed.

## v3: single consolidated capture session (use this, not Step 2 above)

Same tool you just benchmarked, same window/HP prompts. From
`flyff_farming_recorder`:

```
python calibration_capture.py --capture --seconds 1800 --rate-hz 50 --out movement_calibration.csv --eva-hotkey F9
```

- `--seconds 1800` is just a generous ceiling (30 min) -- **press Ctrl+C
  as soon as you've finished all trials below**; it writes out whatever
  was captured up to that point, you don't need to fill the whole window.
- `--rate-hz 50` gives ~10 samples across even the shortest (~0.2s) tap,
  with huge headroom below what you just measured as achievable.
- `--eva-hotkey F9` (or any F-key/number you won't otherwise press) --
  same trick as before: this tool only watches whichever key you name
  here, independent of your real in-game EVA bind, so cast normally if
  you need it.
- Output goes to `flyff_farming_recorder\movement_calibration.csv` (or
  wherever you run it from) -- send me that one file when done.

**Keys** (forward held together with the turn key, not alone):

| primitive | qwerty | azerty |
|---|---|---|
| FORWARD | `W` alone | `Z` alone |
| LEFT | `W`+`A` together | `Z`+`Q` together |
| RIGHT | `W`+`D` together | `Z`+`D` |

No jump during trials. Face into open space (~20+ cells clear) before
each trial -- no need to reset to a specific heading.

**Trial**: press and hold for a target duration, release, stand
completely still for at least 2 seconds before the next trial (the CSV
already has dense enough timestamps that this gap alone gives clean
separation -- no marker needed).

**Duration targets and repetitions** (aim by feel; actual achieved
duration is measured from timestamps, not trusted from intent):

| primitive | ~0.2s | ~0.5s | ~1.5s | ~3.0s |
|---|---|---|---|---|
| FORWARD | 50 | 25 | 20 | 15 |
| LEFT | 50 | 25 | 20 | 15 |
| RIGHT | 50 | 25 | 20 | 15 |

330 trials total, roughly 15-20 minutes including the stationary gaps.
Interleave rather than block -- rotate F/L/R within each duration phase
rather than doing all of one primitive first. Discard/redo freely if a
trial is interrupted (monster, popup, wall clip, extra key) -- extra
trials don't hurt.

When done: Ctrl+C, then send `movement_calibration.csv`. I'll derive
heading from the dense position trajectory, segment trials from the
stationary gaps, and report the same full battery of statistics as the
Part-A historical reparse -- directly comparable, but now with genuine
single-tick resolution and real sustained holds, which the historical
data didn't have.

## v4: deployment-matching follow-up (small, do this next)

**Why**: everything recorded so far starts each trial from a dead stop
(press keys from stationary). But the actual navigator keeps forward
LATCHED continuously and only toggles the steering key -- turning while
already moving, never accelerating from rest while also turning. That's
a physically different regime, and it's the one that actually matters
for calibrating the simulator the navigator trains in. This recording is
much smaller than the previous ones -- same tool, no new setup.

**Command** (identical to before, new output file):
```
python calibration_capture.py --capture --seconds 600 --rate-hz 50 --out movement_calibration_steering.csv --eva-hotkey F9 --layout azerty
```

**What to do**: hold `Z` down and keep it held for the ENTIRE session --
don't release it between pulses. Get moving in open space, then:

1. Pulse `Q` (LEFT) for a target duration, release `Q` (Z stays held).
2. Keep moving straight (Z only) for about 1.5s.
3. Pulse `D` (RIGHT) for a target duration, release `D`.
4. Keep moving straight for about 1.5s.
5. Repeat, alternating LEFT/RIGHT and cycling through the duration
   targets below.

Same duration targets as before (~0.2s / ~0.5s / ~1.0s, aim by feel --
actual duration gets measured from timestamps): **25-30 reps of each
duration, for each of LEFT and RIGHT** (so ~150-180 pulses total). At
roughly a pulse + 1.5s settle per rep, this should take about 5-7
minutes -- much shorter than the earlier sessions. Reposition into open
space whenever you're getting close to a wall; Z can stay held while you
do this if it's just a straight line, otherwise release-and-reacquire is
fine, extra straight-line segments don't hurt.

Ctrl+C when done, send `movement_calibration_steering.csv`. This one
doesn't need the >=X-second-stationary-gap discipline the earlier
sessions needed -- since forward never releases, the turn-key presses
themselves are the trial boundaries, which is unambiguous from the key
state directly.

This will let me measure: LEFT/RIGHT response in the first ~0.2s of a
pulse while already moving, how that response changes as the pulse gets
longer, how quickly heading/speed settle back to straight-line after
release, LEFT/RIGHT symmetry under this regime, and whether forward
speed dips at all while steering -- directly comparable to (but distinct
from) the stationary-start numbers already measured. No model changes
until this is in and reviewed alongside everything else.
