# Performance

## Runtime Audit Pass 2 baseline

Environment: `.venv` Python 3.14.3 on Windows 11. All native measurements use
fake process memory; no live process was opened. Active model/map reads were
read-only.

| Scenario | Baseline |
|---|---:|
| Position + monster provider construction | 0.021 ms, zero pointer reads |
| Default four-radius recovery miss | 973.887 ms, 385 reads, 24,924,876 bytes, 4 region enumerations |
| Ordinary null pose read | 979.233 ms, 386 reads, 24,924,880 bytes |
| Stale nonzero invalid pose | 0.035 ms, no recovery |
| Three scaled failed recoveries | 10.421 / 10.234 / 10.336 ms, identical work, no failed cache |
| Successful shifted recovery | 66.251 ms; verified success-cache hit 0.020 ms |
| Two concurrent failed callers | 46.286 ms wall, two scans/region enumerations |
| Recovery cancellation overrun | short 15 ms join failed; 120.425 ms to worker finish |
| Noncooperative CONTROL shutdown | returned 36.9 ms with worker alive and bus closed |
| Unified step native/OCR calls | 6 pose, 2 monster, 1 OCR |
| Tower legacy distance field | 315.7–355.7 ms/call |
| Tower V0707 local grid | 22.6–22.9 ms/call |

Containment microprofile (3,000 lookups at the last region):

- 8 regions: 2.593 µs/lookup.
- 128 regions: 38.677 µs/lookup.
- 512 regions: 152.943 µs/lookup.

## Stabilization budgets

- Healthy fake native read ≤10 ms p99.
- Null/stale ordinary read ≤20 ms p99 and zero scan enumeration.
- Explicit recovery hard deadline ≤500 ms; cancellation ≤50 ms; one flight;
  failed cooldown ≥5 s.
- Preview ≤100 ms/frame at 10 FPS; optional native overlay ≤25 ms or skipped.
- Stop-to-key-release ≤50 ms.
- Canonical fake step compute ≤50 ms excluding action hold/native/OCR latency.
- Shutdown terminal success implies zero live project non-daemon workers.

Raw evidence:
`profiles/runtime_native_pointer_results.json`,
`profiles/runtime_lifecycle_mock_results.txt`, and
`profiles/runtime_farming_pass2_measurements.md`.

## Post-stabilization measurements — 2026-07-31

Fake-memory measurements after `STAB-001` through `STAB-004`:

| Scenario | Post-stabilization | Budget |
|---|---:|---:|
| 1,000 ordinary null position reads | 0.003494 ms mean; 0.008097 ms p99; 0.049800 ms max; zero region enumerations | ≤20 ms p99 |
| 1,000 ordinary null monster reads | 0.000870 ms mean; 0.002000 ms p99; 0.021700 ms max; zero region enumerations/find calls | ≤20 ms p99 |
| Explicit zero-deadline recovery | 0.067400 ms; zero region enumerations | ≤500 ms |
| Combined stabilization suite | 70 tests in 1.44 s | No regression |

The ordinary null position path improved from 979.233 ms and 24.9 MB scanned to
0.008097 ms p99 with no scan. Explicit recovery is one-flight, indexed,
cooldown-backed, and checks cancellation/deadline between bounded operations.
Its `deadline_is_cooperative` metric is intentionally true: an already-blocking
backend call cannot be preempted synchronously and remains `RISK-011` for
`PTR-001`.
