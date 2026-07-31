# Phase 09 - Current-client pointer recovery and startup

## Trigger

The real client attaches quickly and keeps the GUI responsive, but Native Health
reports `pointer_unavailable` at absolute slot `0x6352B8`. Explicit recovery
searched configured-slot radii through `0x800000` and returned `not_found`.
Dry-run then raised `NativePointerSnapshotError: Local-player pointer is null`
inside the managed control worker, before focus or environment activation.

## Preserved architecture

- `NativeProcessService` remains the sole process handle and pointer-state owner.
- Ordinary position/actor reads remain constant-time and never initiate scanning.
- Recovery remains single-flight, deadline-bounded, cancellable, and worker-only.
- Persistence remains multi-sample, cross-config, transactional, and explicit.
- The native-heading farming route does not regain the retired camera sweep.

## Leading hypotheses

1. The current player's global pointer slot moved outside the old-slot-centered search strategy or the useful portion of its bands.
2. The player and world globals moved independently, while recovery still treats their old relative delta as a strong acceptance condition.
3. The actor layout may have changed; detailed rejection evidence is needed to distinguish this from a discovery-range failure without weakening validation.
4. Startup treats an expected unavailable native state as an exceptional worker failure instead of attempting one bounded recovery and returning a clean stop.

## Intended correction

- Expose module base/size/path and pointer-width evidence from the Win32 backend.
- Search the actual module image for global pointer slots, with the old bounded neighborhood retained only as a fallback when module metadata is unavailable.
- Correlate the validated player's world base to independently discovered module slots instead of requiring the historical player/world slot delta.
- Rank candidates using self/world/layout/plausibility/reference evidence, reject ambiguity, and repeat validation before publishing or persisting state.
- Surface per-strategy progress and rejection summaries without candidate spam.
- Run shared recovery from the managed control worker during startup; if it fails, report one concise expected outcome before any focus/input/env work.
- Persist only a strongly validated result from the explicit GUI recovery action.

## Automated acceptance

- Current configured offsets can both be wrong and a valid independently paired module candidate is recovered.
- Actor-like false positives, instability, ambiguity, timeout, and cancellation do not publish or persist pointers.
- Ordinary reads do not scan and concurrent recovery remains single-flight.
- Startup success proceeds only after a coherent snapshot; startup failure and cancellation create no input/focus/environment side effect or traceback.
- Explicit recovery requests transactional persistence; automatic startup does not.
