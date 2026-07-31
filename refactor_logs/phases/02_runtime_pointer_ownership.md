# Phase 02 — Runtime and Pointer Ownership

Status: in progress (`PTR-001`).

## Entry state

Phase 01 commit `63651e97d6d013ac41364d912e98b70ac5c76b88`
established cheap ordinary reads and an explicit one-flight recovery API.
This phase will replace module-global ownership with one injected shared pointer
state/resolver service, put explicit recovery behind a managed lifecycle owner,
share coherent native snapshots, and finish two-file transactional persistence
and supported diagnostics. The real backend is cooperative at individual Win32
query/read boundaries; arbitrary blocking fakes must not be hidden behind daemon
threads.
