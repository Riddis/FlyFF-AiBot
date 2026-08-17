"""Development-only tooling: telemetry, native diagnostics, archive
utilities, calibration tools, session/artifact context, and the specialist
subprocess orchestrator.

Canonical runtime/shared packages (farming, position, navigation, recorder,
simulator.schema) must never import anything from this package -- the
dependency direction is strictly one-way, devtools -> canonical, enforced
by tests/test_devtools_dependency_direction.py."""
