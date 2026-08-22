# Audit Pass 2 — Runtime and Lifecycle

Status: complete (`AUD2-001` through `AUD2-004`).

Date: 2026-07-31

## Conclusion

Pass 2 reproduced the freeze/unresponsive-shutdown class with fake runtime and
fake process-memory backends. It also found behavior and performance defects
that static reachability alone could not establish:

- A normal null pose read performs about one second and 24.9 MB of synchronous
  recovery work.
- Position- and monster-like callers scan concurrently; failures have no
  cooldown.
- Shutdown can return with a live non-daemon worker, then close its
  dependencies; provider close can subsequently block past the worker timeout.
- A stale capture completion can detach the UI after a newer generation is
  active.
- Permanent capture failure leaves capture/preview/readiness falsely active.
- Active unified movement correctly preserves keys through EVA, but only
  because an import-time patch wraps a separate executor.
- One ordinary unified step reads pose six times and actors twice, rebuilds a
  316–356 ms legacy distance field, and adds about 23 ms of teleport-grid work.
- Stop can trigger a later Gym/SB3 reset and more native work; save/report is
  unconditional and uncancellable.
- External teleport can be policy-penalized when it starts near the forbidden
  warning radius without crossing the trigger.
- Unified farming has no ongoing focus-loss gate.

No real FlyFF process was attached. Live-client confirmation remains required,
but the fake reproductions are sufficient to authorize stabilization.

## Evidence index

- `runtime_lifecycle_challenge.md`: GUI, capture/preview, worker, focus,
  Stop/close, and unexpected-exit traces.
- `runtime_native_profile.md`: native scenarios, fake-memory measurements, and
  recovery budgets.
- `runtime_farming_challenge.md`: final patch behavior, PPO compatibility,
  EVA, kills/OCR, contact, teleport, and per-step performance.
- `profiles/runtime_lifecycle_*`: lifecycle reproduction and focused tests.
- `profiles/runtime_native_pointer_*`: executable fake-memory harness/results.
- `profiles/runtime_farming_pass2_measurements.md`: read-only model/map and
  final-patch behavior measurements.

## Required 17-scenario matrix

Every row records current call flow/thread, blocking/locks, resource/key
cleanup, status, and failure behavior. The appendices contain symbol-level
detail.

| # | Scenario | Current flow and thread | Blocking/locks | Cleanup/key state | Current status/failure behavior |
|---:|---|---|---|---|---|
| 1 | Launch GUI and attach | GUI constructs module-global Gui/Bot; main thread enumerates windows and synchronously attaches; CAPTURE then PREVIEW start | Sequential preview/capture joins; provider close can wait on monster lock; old resources close before preview stops | Reattach releases old input first, racing preview; new providers/keyboard then owned by Bot | Success before first valid frame; stale old completion can disable generation 2; failed source may still look attached |
| 2 | Capture and Bot Vision | CAPTURE loops GDI; PREVIEW synchronously builds CV/heading/native overlays; GUI encodes/renders | Capture unthrottled; preview native read can scan; snapshot copies under capture lock; lifecycle queues unbounded | Capture has source-close hook; preview has no interrupt hook and owns no keys | Latest values are bounded, but stale frame persists and optional error rate limit hides repeated work |
| 3 | Camera discovery focused/unfocused | CONTROL reset reads actors, then focus wait and four D turns, then forced actor discovery | Focus/turn waits cancel; pre-read and forced discovery do not and hold provider lock | D has local finally release; cancelled sweep is nevertheless marked warmed and reset continues | Useful focus messages; null read can block before them; no distinct cancelled state |
| 4 | Unified dry run | CONTROL imports/install patches, builds env, reset, heuristic four-action loop | Repeated native/map/OCR work; legacy distance field dominates | Persistent navigator keys; outer finally stops; cancellation can auto-reset | Periodic stats, but GUI calls it Training and Stopped can precede worker exit |
| 5 | New PPO training | CONTROL builds env/PPO, learn, final save/report | Callback only after step; checkpoint/save/report synchronous | Stop releases keys while worker continues; multiple release owners | Late save/report can overwrite Stopped; no atomic cancellation policy |
| 6 | Resume PPO | CONTROL builds env then `PPO.load(..., env=env)` and learns | Load/space check non-cancellable; later same as scenario 5 | No movement should occur before reset, but explicit native/model preflight is incomplete | 482/4 loads; shape/action mismatch is clear; same-shape semantic drift is unchecked |
| 7 | EVA while held | Final V0673-patched navigator casts over `NavigatorActionExecutor` | EVA actor/result reads can overrun nominal deadline | Active trace taps F1 only and keeps Z/Q held; separate ActionExecutor releases/restores | Behavior currently correct but depends on patch order and duplicate executors |
| 8 | Actor kill/OCR | V0672 helpers capture candidates, poll two absences, validate OCR | Actor lock/recovery can block; OCR lock separate | Movement preserved through cast; terminal release outside helper | Native delta rewards; OCR decrease/outlier rejected and diagnostic only |
| 9 | Wall/contact | V0700 compares before/after/base snapshot and increments contact | Six pose reads, two actor reads, distance-field/map work | Movement remains held unless Stop/session terminal | Contact penalty/info emitted; no termination for ordinary wall |
| 10 | Forbidden teleport zone | V0707 adds proximity/buffer/crossing reward and marks session | Local grid repeatedly recomputes forbidden points | Navigator stopped; env supplies idle no-input rollout steps | True crossing strongly policy-penalized; external driver is session flag, not Gym terminated/truncated |
| 11 | Server time teleport | V0707 compares jump/proximity/segment | Same native/map work and ineffective pointer-grace bound | Movement stopped when session marked | Far jump correctly external; near-warning non-crossing jump falsely forbidden/policy-caused |
| 12 | Pointer temporarily null | Ordinary preview/reset/player/actor reads call recovery | ~979 ms fake ordinary read; 386 reads/24.9 MB; concurrent duplicate scans; no cooldown | No per-attempt resource; cancellation cannot enter | No recovery progress; each caller can retry; nominal grace/deadlines are not wall-clock bounds |
| 13 | Pointer genuinely stale | Zero slot scans and may persist; nonzero stale target is merely dereferenced | Progressive overlap, repeated region enumeration, linear containment, stability sleeps, sync JSON writes | Success cache only; failure leaves no state; persistence from caller thread | Zero stale slot may recover; invalid/plausible nonzero stale slot never invokes recovery |
| 14 | Stop during work | GUI cancels CONTROL and immediately reports Idle; PREVIEW recovery is not targeted | Scan/discovery/step/reset/checkpoint/save/report can outlive cancel | Stop hook and Bot paths release repeatedly; no join in Stop | Premature Stopped; late status writes; post-cancel reset/native work reproduced |
| 15 | Close idle/active | GUI calls shutdown(8), then entry-point calls it again | Shared worker deadline; false joins ignored; provider close can then block unbounded | Resources/bus close under live workers; non-daemon survivor keeps process alive | Only Stopping shown; per-worker timeout invisible; double stop/release reproduced |
| 16 | Focus loss | Mapping pauses; unified farming calls executor directly | Keyboard repeat continues; focus API may fail open | Unified keys not centrally released; stale action can resume | No farming focus-lost/paused/regained state |
| 17 | FlyFF exits | Capture retries forever; preview uses stale frame/native; farming may enter position-loss path | Recovery may run; monster errors are not classified like position errors | Outer CONTROL finally releases, but attachment/handles persist until later cleanup | No `client_exited`; false readiness, teleport misclassification, or raw worker failure |

## Reproduced measurements

| Measurement | Result |
|---|---:|
| Provider construction | 0.021 ms; no pointer read |
| Default recovery miss | 973.887 ms; 385 reads; 24,924,876 bytes |
| Ordinary null pose read | 979.233 ms; 386 reads; 24,924,880 bytes |
| Concurrent miss callers | two region enumerations/scans, not one |
| Repeated scaled failures | identical work three times; zero failed-cache entries |
| Successful shifted recovery | 66.251 ms; verified cache hit 0.020 ms |
| Cancelled recovery worker | still alive after 15 ms join; finished 120.425 ms after cancellation |
| Shutdown with noncooperative CONTROL | returned in 36.9 ms with worker alive and bus closed |
| Lifecycle focused tests | 11 passed; reproduced gaps are untested |
| Unified movement step | six pose, two monster, one OCR read |
| Tower legacy distance field | 315.7–355.7 ms per step |
| Tower V0707 local grid | 22.6–22.9 ms per step |
| Active model | `Box(482,)`, `Discrete(4)`, 771 timesteps |

## Lock, queue, polling, and ownership findings

- No definitive two-lock inversion was reproduced. The concrete hang is
  blocking/starvation plus an unbounded provider-lock wait after a timed-out
  join.
- WorkerManager prevents duplicate live workers per kind, but events lack
  generation/session identity.
- Capture latest values, preview latest values, and logs are bounded.
  Completion/failure/confirmation/recovery-request queues are not.
- Capture success is unthrottled; capture failures retry at 4 Hz; stale preview
  repeats at 10 Hz; keyboard repeats at 40 Hz; null recovery can retrigger from
  ordinary reads.
- Movement/key cleanup, native resources, and lifecycle status each have
  multiple owners.

## Corrections to Static Pass 1 classifications

1. `NavigatorActionExecutor`, not `ActionExecutor`, is the active unified
   executor. Merge useful types/behavior from both into one canonical direct
   executor; delete neither first.
2. Active unified EVA currently preserves movement. The bug is architectural
   fragility and the duplicate low-level path, not a demonstrated unified
   key-up in scenario 7.
3. `V0672NativeFarmingFixes.py` contains live kill/OCR behavior that must be
   extracted before deletion.
4. `NativeFarmingEnv.py` retains live reset/snapshot/wait/close behavior even
   though V0700 replaces step.
5. Capture, preview, RuntimeController, Gui, NativePointerRecovery, both native
   providers, and CameraDiscoverySweep are stabilization-critical, not merely
   later cleanup/refactor targets.
6. End-to-end shutdown is unbounded. Only WorkerManager joins have a deadline.
7. Automatic recovery covers a zero old slot only; a stale nonzero target is a
   different typed failure.
8. `RuntimeBus` lifecycle/status records require generation/session identity.
9. No Pass 2 evidence made adaptive/legacy mapping production-live. Their
   archive classifications remain, subject to extracting generic `mapper/rl`
   helpers first.

## Stabilization acceptance budgets

- Null/stale ordinary native read: ≤20 ms p99 in fake tests and zero scan/region
  enumeration.
- Healthy fake native read: ≤10 ms p99.
- Explicit recovery only: one flight/process, one indexed region inventory,
  hard attempt deadline ≤500 ms, cancellation latency ≤50 ms, failed cooldown
  ≥5 s, no caller-thread config write.
- Preview: 10 FPS budget ≤100 ms/frame; native overlay ≤25 ms or skipped.
- Stop-to-key-release ≤50 ms. A cancelled reset/step must perform no later
  native/camera operation.
- Shutdown reports terminal success only with no project non-daemon worker
  alive; dependencies of a timed-out worker remain open and the timeout is
  visible.
- Canonical fake step compute ≤50 ms excluding configured hold/backend/OCR.
- Active model remains schema 482/actions 4 until explicit migration.

## Pass 2 decision

The evidence supports proceeding to the written target plan and then a narrow
first stabilization commit. Broad moves/deletions remain prohibited until
stabilization is committed and behavior-equivalent canonical replacements
exist.
