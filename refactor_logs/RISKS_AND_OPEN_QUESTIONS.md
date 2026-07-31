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
