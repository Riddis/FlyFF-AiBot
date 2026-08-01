# Phase 06 — Mapping, Native, and Vision Boundaries

Status: not started.
# Completion evidence

- `CaptureService` alone owns the frame source and publishes coherent copied
  samples; generation checks reject stale reattachments.
- `PreviewService` consumes snapshots at a fixed rate and publishes only the
  newest render. `RuntimeBus` stores one value per high-rate key and bounds GUI
  logs.
- OCR remains a read-only validation input to canonical farming kill tracking;
  reward comes from cast-scoped native actor transitions.
- `FarmingMapContext` owns the selected-map transform, local safety crop,
  direct-path feature, and buffered teleport mask without importing mapper RL.
- Behavior coverage includes map conversion/masks/direct paths, OCR outlier
  rejection, preview unavailability, capture generations, native snapshots,
  and both fake end-to-end session flows.

