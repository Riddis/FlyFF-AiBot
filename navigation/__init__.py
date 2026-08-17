"""Canonical shared production navigation.

Owns the qualified router (`kinodynamic_route_planner`), the calibrated
movement kernel (`movement_kernel`, `movement_kinematics`), and the pure
5-value navigation-evidence core (`navigation_evidence`). This package must
never import gymnasium, stable_baselines3, simulator training/env modules,
recorder, position profiling, GUI packages, telemetry, or Win32 process
attachment -- see `tests/test_navigation_dependency_boundary.py`.

`simulator.split_branch_policy` remains the sole checkpoint-ABI owner and is
not part of this package. `simulator.navigation_history.NavigationHistoryWrapper`
remains the training-only gymnasium wrapper and is not part of this package.
"""

from __future__ import annotations
