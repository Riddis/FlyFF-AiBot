# Risks and Open Questions

| ID | Severity | Evidence | Owner / phase | Mitigation | Status |
|---|---|---|---|---|---|
| `RISK-001` | Critical | Fake ordinary null read took 979 ms/24.9 MB; concurrent and repeated failed scans reproduced. | `STAB-001` / Phase 01 | Remove scan from reads; explicit bounded single-flight resolver with cooldown. | Mitigated in automation; live validation pending |
| `RISK-002` | Critical | Shutdown returned with live non-daemon worker, then closed bus/resources; provider close can exceed deadline. | `STAB-005` / Phase 01 | Idempotent shutdown; honor false joins; retain dependencies; visible timed-out state. | Mitigated in automation; live validation pending |
| `RISK-003` | High | Active unified behavior is layered through order-dependent monkeypatches. | `AUD1-001` / Phase 03 | Inventory install order; migrate only behind behavior parity tests. | Open |
| `RISK-004` | Medium | Initial worktree is dirty despite the stated recent commit. | `BASE-002` / Phase 00 | Attribute every path and use scoped commits. | Open |
| `RISK-005` | High | Live FlyFF/process-memory behavior cannot be safely assumed available in automation. | `DOC-003` / Phase 08 | Use fakes and mark exact live-client validation steps. | Open |
| `RISK-006` | High | Active model requires 482 values; same-shape semantic drift is not detected by SB3. | `FARM-003` / Phase 03 | Preserve 482, add semantic schema hash and explicit migration. | Open |
| `RISK-007` | High | External non-crossing teleport near warning radius is policy-penalized. | `FARM-002` / Phase 03 | Typed session reasons; prioritize proven crossing/entry over proximity. | Open |
| `RISK-008` | High | Unified farming bypasses focus checks and can retain movement while unfocused. | `INPUT-002` / Phase 04 | One session focus gate and exactly-once release/pause. | Open |
| `RISK-009` | High | One step performs six pose/two actor reads and ≥316 ms legacy map computation. | `FARM-002` / Phase 03 | One coherent snapshot; cache static distance/forbidden data. | Open |
| `RISK-010` | High | Cancellation can auto-reset into more native/camera work; save/report is uncancellable. | `STAB-004` / Phase 01 | Abort reset after cancel; explicit atomic save/report policy. | Open |
| `RISK-011` | High | A synchronous memory-backend call already in progress cannot be preempted by the resolver's cooperative deadline. | `PTR-001` / Phase 02 | Put explicit recovery behind one bounded lifecycle owner; add backend deadline checks and preserve dependencies after false joins. | Open; ordinary reads are unaffected |
| `RISK-012` | Medium | Two config files cannot be replaced in one filesystem primitive; a hard crash can expose a mixed pair until pre-load journal recovery. Directory fsync is best effort on Windows and writer locking is process-local. | `PTR-003` / Phase 02; `BOUND-001` / Phase 06 | Durable exact-byte journal, conservative all-old rollback, pre-load recovery in every factory; consolidate the duplicated pointer value into one versioned config later. | Mitigated in-process; live crash validation pending |
| `RISK-013` | Low | The low-level opt-in recovery API logs and suppresses persistence failure, while the supported diagnostics command deliberately never persists. | `PTR-003` / Phase 02; `GUI-004` / Phase 05 | Keep persistence behind explicit confirmation; surface a typed persistence result when the UI flow is added. | Open |
| `RISK-014` | Medium | Phase 02 diagnostics report map selection/overlay state but not a coordinate-conversion result because no cached pose/conversion contract exists yet. | `GUI-004` / Phase 05; `BOUND-001` / Phase 06 | Add conversion readiness/result to the supported diagnostics presenter without initiating extra GUI-thread native reads. | Open |
| `RISK-015` | High | The validated Phase 02 checkpoint and isolated Phase 03 core encountered scoped Git escalation quota rejection; the Phase 03 retry was again rejected after all gates passed. | User approval / current handoff | Preserve exact dirty ownership and use only documented path-scoped checkpoints; user may perform the exact recorded Git commands, or retry when elevation is available. | Phase 02 committed as `a63e222`; Phase 03 index remains empty and checkpoint-blocked |
| `RISK-016` | Low | RuntimeController recovery/control mutual exclusion is check-then-start rather than one atomic transition across arbitrary concurrent callers. | `PTR-004` / Phase 02; `GUI-002` / Phase 05 | Current GUI command ownership serializes callers; atomicize the transition if RuntimeController gains multiple command threads. | Tracked; not a current checkpoint blocker |
| `RISK-017` | Critical | The first isolated core recomputed its schema/contract hashes from current code, so the known metadata-less model could remain approved after semantic drift. | `FARM-003` / Phase 03 | Freeze reviewed literal schema/contract hashes, bind the active artifact SHA to that contract, and reject recomputed drift. | Mitigated in isolated core; production preflight integration pending |
| `RISK-018` | High | Legacy layout Y and direct native Z have opposite signs, and direct density excludes distant blocked actors while the legacy prefix does not. | `FARM-002` / Phase 03 | Carry both coordinate frames, maintain two density populations, and assert a full 482-value nonzero-Z golden vector. | Mitigated in isolated core; live-frame construction pending |
| `RISK-019` | High | External exits and EVA could inherit non-policy shaping/penalty terms. | `FARM-002` / Phase 03 | External truncation is kill-only, cancellation/fatal is neutral, and EVA gates density/contact. | Mitigated in core; SB3 boundary integration pending |

## Final reconciliation â€” 2026-07-31

- Closed in canonical production and automated coverage: `RISK-003`,
  `RISK-004`, `RISK-006`, `RISK-007`, `RISK-008`, `RISK-009`, `RISK-014`,
  `RISK-015`, `RISK-017`, `RISK-018`, and `RISK-019`.
- `RISK-010` is mitigated: cancellation cannot auto-reset and publication is a
  short atomic critical section. Model serialization already in progress is
  cooperatively non-preemptible; false joins preserve dependencies.
- Still open pending live/external evidence: `RISK-001`, `RISK-002`, and
  `RISK-005` (real FlyFF/Tk/Win32 behavior), `RISK-011` (an already-blocking
  backend call), `RISK-012` (hard-crash paired-config recovery), and
  `RISK-013` (confirmed persistence UI intentionally omitted).
- `RISK-016` remains a low-severity future concern only if runtime commands
  gain multiple concurrent callers; the current GUI serializes commands.
