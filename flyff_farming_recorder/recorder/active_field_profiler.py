"""Compatibility import for the canonical recording-only profiler."""

# BRIDGE B2 — removed in Phase 7
from position.profiling.active_field_profiler import ActiveFieldProfiler

__all__ = ["ActiveFieldProfiler"]
