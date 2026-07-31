# Foreground Vision Bot Architecture

## Production path

`foreground_vision_farm.py` constructs `Gui` and `Bot`. The GUI translates
main-thread events into `RuntimeController` requests. The controller owns one
`WorkerManager`, which supervises capture, preview, control, and native
diagnostic workers. Farming commands import only `farming.trainer`; there is no
runtime patch installer or legacy movement-policy path.

```text
GUI event adapter
  -> RuntimeController
     -> CaptureService -> WindowCapture
     -> PreviewService -> Bot.build_preview -> RuntimeBus latest frame
     -> farming.trainer
        -> FarmingMapContext
        -> NativeProcessService -> player + actor readers
        -> WindowFocusService + DirectFarmingControl
        -> UnifiedFarmingEnv -> UnifiedFarmingGymEnv -> SessionAwarePPO
        -> atomic model + report + recovery manifest
     -> native diagnostic worker -> scan-free health or explicit recovery
```

## Ownership

| Resource | Owner | Contract |
| --- | --- | --- |
| Tk/PySimpleGUI widgets | `Gui` main thread | Workers never call Tk. The loop drains bounded logs and renders latest-only frames. |
| Long-running threads | `WorkerManager` | Non-daemon, one worker per kind, cooperative cancellation, bounded joins. |
| Capture source and frame generation | `CaptureService` | One generation at a time; copied snapshots; stale generations cannot publish. |
| Bot Vision rendering | `PreviewService` | Fixed-rate consumer; publishes only the newest frame. |
| Native memory handle and pointers | `NativeProcessService` | One coherent player/world pointer snapshot shared by both readers. Ordinary reads never recover. |
| Pointer diagnostics/recovery | Diagnostic or control worker | Health uses one fixed pointer/pose sample plus cached actor/OCR/focus facts and map conversion. Recovery is single-flight, cancellable, deadline-bounded, negatively cached, and never runs on Tk. Explicit GUI recovery persists a strongly validated pair transactionally; automatic startup recovery does not persist. |
| Farming key state and focus | `DirectFarmingControl` / `WindowFocusService` | One held-key ledger; autofocus plus cancellable manual grace; terminal paths release it. |
| Farming behavior | `UnifiedFarmingEnv` | Visible `reset()`/`step()`, four actions, coherent native frame per step, typed terminal outcome. |
| Model/report publication | `farming.reporting` | Temporary file plus atomic replace; report and recovery manifest identify the published artifact. |

## Canonical farming contract

The action space is `Discrete(4)` in this stable order:

1. `RUN_FORWARD`
2. `RUN_FORWARD_LEFT`
3. `RUN_FORWARD_RIGHT`
4. `CAST_EVA`

Movement is persistent. Steering changes only the lateral key while retaining
forward. EVA taps F1 without releasing movement. The observation is a stable
482-element `float32` vector containing normalized absolute pose, heading,
local Tower safety/teleport features, movement/contact/EVA state, and bounded
native actor features.

Native actor lifecycle/HP transitions scoped to a cast are the reward signal.
OCR can validate a kill count but cannot create reward. The mapped teleport
zone is explicit in observations and reward components. A policy-caused entry
is penalized; an external map/session teleport is a non-policy terminal that
saves the valid model and report without training that terminal prefix.

## Native and map boundaries

`NativeProcessService` owns the process memory object and pointer state.
Position and monster providers consume the same pointer snapshot. A null or
stale pointer returns a cheap typed unavailable state. Recovery scans the
actual module image when Win32 module metadata is available, correlates player
and world slots independently, rejects ambiguous/unstable candidates, and
falls back to bounded configured-slot bands only for backends without an image
extent. It may be initiated by **Recover Pointers** in the diagnostic worker or
once by farming startup in its control worker. The latter either verifies a
coherent snapshot or returns a clean no-input outcome before focus/environment
activation.

`FarmingMapContext` loads the selected `MapCatalog` entry, coordinate frame,
occupancy grid, safety mask, direct-path visibility, and buffered forbidden
teleport mask. The farming package does not import mapper RL or adaptive
navigation policy code. Map creation/editor workflows remain separate GUI
features.

## Shutdown and failure rules

Stop cancels control and diagnostics before requesting input release. Close
cancels all workers and joins control, diagnostic, preview, then capture within
one deadline. If a join times out, the GUI reports the live worker and retains
its dependencies; a later close can finish safely. Model-space or preflight
errors occur before `Bot.start()` enables movement. Fatal training failures
write a report but do not overwrite the last-known-good model.
