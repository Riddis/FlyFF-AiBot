from __future__ import annotations


def classify_forward_heading_drift(
    change_degrees: float,
    *,
    nominal_limit: float,
    recoverable_limit: float,
) -> str:
    """Classify forward-time heading drift without depending on runtime input APIs."""
    magnitude = abs(float(change_degrees))
    if magnitude <= nominal_limit:
        return "nominal"
    if magnitude <= recoverable_limit:
        return "recoverable"
    return "unsafe"
