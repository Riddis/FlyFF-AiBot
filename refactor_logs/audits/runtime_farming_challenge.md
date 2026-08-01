# Runtime Audit Pass 2 — Farming Behavior Challenge

Status: complete for scenarios 4 through 11 and farming portions of 14, 16,
and 17.

Date: 2026-07-31

## Scope and safety

This track imported the final runtime patch chain, exercised real environment
objects with fake input/native providers, loaded the active PPO metadata
read-only, and measured the active Tower map. It did not attach to FlyFF, send
real input, save a model, or modify production config/model/map/asset files.
Exact captured results are in
`refactor_logs/profiles/runtime_farming_pass2_measurements.md`.

## Executive corrections to Pass 1

1. Active unified movement does not use `Bot.action_executor`. It uses
   `LiveNavigatorController.executor -> NavigatorActionExecutor ->
   HumanKeyboard`.
2. Scenario 7 currently preserves held movement: V0673 suppresses the
   stop/reassert behavior while patched `cast_eva()` runs. The behavior is
   correct but fragile and import-order-dependent. The separate
   `ActionExecutor.CAST_EVA` still emits movement key-up/key-down transitions.
3. The shipped PPO is exactly compatible with the assembled
   `Box(482,) -> Discrete(4)` spaces. SB3 rejects 221 observations or five
   actions, but cannot detect semantic reordering within an unchanged shape.
4. V0672 is partly live: its cast-scoped native kill and OCR validation helpers
   are called by V0700. It cannot be deleted until those helpers are extracted.
5. An ordinary final movement step performs six player-pose reads, two monster
   reads, and one OCR read. The old target-distance field is also rebuilt.
6. Tower distance-field construction costs 316–356 ms per call; V0707 local
   teleport-grid construction adds about 22.6–22.9 ms. With the configured
   200 ms hold, the loop cannot achieve five decisions/second even with healthy
   native reads.
7. Cancellation/truncation can cause dry-run, agent, or SB3 auto-reset to call
   `reset()` after Stop, re-entering actor/pointer/camera/snapshot work.
8. A server-sized teleport starting within the warning radius is misclassified
   as policy-caused even when its segment does not cross the trigger.

## Final runtime assembly

```text
CONTROL worker
  -> import native_farming
       -> install V0672 -> V0673 -> V0674 -> V0700 -> V0707
  -> build_live_native_env
       -> NativeMapContext
       -> legacy NativeFarmingObservationBuilder
       -> LiveNavigatorController(load_policy=False)
            -> NavigatorActionExecutor
       -> CameraDiscoverySweep
       -> NativeFarmingEnv
  -> final patched reset/step
```

The final observation is the legacy base vector plus 221 unified values. With
the shipped `max_targets=32`, that is `261 + 221 = 482`.

## Scenario matrix

| # | Scenario | Thread and current flow | Blocking/locks | Keys and cleanup | Current status/failure |
|---:|---|---|---|---|---|
| 4 | Unified no-learning dry run | CONTROL builds env, starts Bot, calls patched reset, then chooses one of four heuristic actions and steps until cancel/session end | Reset can recover/discover; every step performs repeated native reads, distance-field/map work, OCR, and cancellable hold | Persistent navigator keys; outer `finally` closes env/stops Bot, but post-cancel reset can re-enter work | Useful periodic stats; GUI relabels `rl_status` as Training; Stop is displayed before exit |
| 5 | New PPO training | CONTROL builds env, constructs PPO, calls `learn`, then unconditional final save/report | Callback sees cancel only after a full env step; checkpoints, final save, report are synchronous | GUI Stop releases keys promptly; worker can remain active through step/save/report | Late save/report messages can overwrite Stopped; no atomic/cancellable save policy |
| 6 | Resume PPO | Same as 5, but `PPO.load(path, env=env)` first | Load/space check is worker-side and non-cancellable; subsequent behavior matches 5 | Same cleanup as 5 | Shape mismatch gives a clear SB3 error wrapped with move-model guidance; same-shape semantic mismatch is undetectable |
| 7 | EVA while moving | V0700 calls patched navigator `cast_eva`; V0673 suppresses movement release and taps F1 | Animation/result polling includes native reads; nominal deadline can be exceeded by recovery | Fake trace: only `press F1`; forward-left remained held; final stop released Q then Z | Correct active behavior, but depends on monkeypatch and duplicate executor architecture |
| 8 | Native kill and OCR | CAST_EVA captures nearby `(base,species)` candidates, polls actors, confirms two consecutive absences; OCR samples captured frame | Monster reads hold provider lock and may recover; OCR uses kill-counter lock; no lock cycle found | Movement remains held through cast; outer stop owns terminal release only indirectly | Native delta drives reward; OCR is diagnostic and rejects decrease/outlier while retaining baseline |
| 9 | Ordinary wall/contact | V0700 before/after pose and base snapshot infer stationary/contact; contact count affects reward/info | Same six pose/two actor reads plus 316–356 ms legacy distance field | Requested persistent movement remains held unless terminal/Stop | Fake stationary step set `contact=True`, `contact_count=1`, contact component `-0.035`; no termination |
| 10 | Approach/cross forbidden zone | V0707 wraps step, computes proximity/buffer/segment crossing, adjusts reward, marks session ended on trigger/crossing | 11×11 local grid repeatedly recomputes forbidden coordinates; native reads precede/follow base step | Session marker stops navigator; environment then supplies no-input idle steps for rollout completion | Crossing yields strong policy-caused penalty; returns `terminated=False`, `truncated=False` while session flag drives external stop |
| 11 | Server-time external teleport | V0707 compares before/after cell, jump size, proximity, and segment crossing | Same native/map work; pointer-loss grace is not a real deadline because reads can scan | Movement stopped when session is marked | Far jump: external/non-policy and no strong penalty. Jump from six cells away: falsely `forbidden_teleport_zone`, policy-caused, about `-50` |
| 14 | Stop during rollout/save | GUI cancels token and releases keys; callback reacts after step; save/report always run | Step, recovery, OCR, reset, checkpoint, save, report are not uniformly cancellable | Multiple stop/release owners; keys likely up but not exactly once | UI says Stopped early; dry-run/agent/SB3 can reset after cancellation and perform more native work |
| 16 | Focus loss | Unified step calls executor directly; no `_require_foreground` gate | HumanKeyboard repeat thread continues its state; no focus-aware pause in farming | Held movement is not centrally released on loss | No paused/focus-lost message; stale held action may resume on focus regain |
| 17 | Client exits | Capture/native failures eventually reach patched position-loss/session paths or escape as raw monster errors | Recovery may run first; save/report may follow clean session path | Outer CONTROL `finally` attempts stop; repeated cleanup owners remain | No distinct `client_exited` reason; failure can be misclassified as external teleport or become worker traceback |

## Detailed behavioral evidence

### Model resume

Read-only `PPO.load()` reported:

- active space: `Box(-1, 1, (482,), float32)`, `Discrete(4)`;
- `n_steps=256`, `num_timesteps=771`;
- 221 observations: `ValueError: Observation spaces do not match`;
- five actions: `ValueError: Action spaces do not match`.

Canonicalization must preserve schema 482 until a versioned new-model
migration. A schema hash must cover field meaning/order, not dimensions alone.

### EVA and duplicate executors

Final patched unified trace while forward-left was held:

```text
before: down Z, down Q
EVA delta: press F1
held after: Z, Q
stop: up Q, up Z
```

Direct `Bot.action_executor` trace:

```text
before: down Z, down Q
EVA delta: up Q, up Z, press F1, down Z, down Q
```

The target needs one normal direct executor with the first behavior, not a
wrapper that temporarily replaces methods on a second executor.

### Per-step reads and CPU

One fully patched fake movement step measured:

```text
pose reads: 6
monster reads: 2
OCR reads: 1
```

The pose reads are V0707 before, V0700 before, base snapshot, V0700 after,
unified observation, and V0707 after. Actor reads are base snapshot and unified
slots. Tower measurements:

| Work | Observed |
|---|---:|
| legacy distance field | 315.7–355.7 ms/call |
| V0707 11×11 local grid | 22.6–22.9 ms/call |
| forbidden-distance helper | 17.6 ms/100 calls |

The canonical step must take one coherent native snapshot, cache static map
distance/mask data, and compute the local teleport view without repeated
`argwhere`.

### Kill/OCR

A near candidate absent for two successful reads produced one native kill; a
far actor was excluded. OCR advanced 100→105, then rejected 999 as an outlier
and 99 as a decrease without moving its accepted baseline. The final reward
uses native delta. Absence is not definitive HP-zero proof if an actor leaves
the discovery/radius set; reporting should preserve that diagnostic nuance.

### Teleport semantics

- Far 35-cell jump: `farm_time_expired_or_external_teleport`,
  `policy_caused=False`, no strong penalty.
- Non-crossing 35-cell jump beginning exactly six cells from the trigger:
  `forbidden_teleport_zone`, `policy_caused=True`, `-50` penalty.
- True crossing: same forbidden/policy result and penalty.

Classification must prioritize proven segment crossing/trigger entry over
before-distance when distinguishing policy behavior from an external teleport.

## Required behavior/performance gates

- Preserve movement across EVA with no movement key-up/down delta.
- Resume schema 482/4 exactly; reject schema/action mismatch before input.
- Ordinary step consumes one coherent player/actor snapshot.
- Cache static Tower distance/mask data; target fake step compute budget
  ≤50 ms excluding configured action hold and OCR/native backend latency.
- User cancellation must not auto-reset or perform further native/camera work.
- Stop-to-key-release ≤50 ms; final save policy is explicit, atomic, and
  status-visible.
- External non-crossing teleport is never policy-penalized solely because its
  starting cell is near the warning radius.
- Focus loss pauses farming and releases held movement once.
- Native kill/OCR behavior retains the validated semantics above.

## Command and evidence ledger

Source tracing used `rg` and `Get-Content` on `native_farming.py`,
`NativeFarmingEnv.py`, `NativeFarmingObservation.py`,
`LiveNavigatorController.py`, `NavigatorActionExecutor.py`,
`ActionExecutor.py`, V0672/V0673/V0674/V0700/V0707, `NativeMapContext.py`,
and the farming/version tests. Bounded fake harnesses ran with:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
@'
# fake input/native/model-space/map exercises captured in the measurement file
'@ | .\.venv\Scripts\python.exe -
```

Full outputs and the read-only model/Tower timings are preserved in
`refactor_logs/profiles/runtime_farming_pass2_measurements.md`.
