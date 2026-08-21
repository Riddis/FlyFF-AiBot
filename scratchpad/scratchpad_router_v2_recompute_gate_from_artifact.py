"""2026-08-15: recompute the v2 development gate -- INCLUDING the
planner-failure checks the first pass's `overall_pass` omitted (a real
procedural gap, user-caught: the module docstring predeclared "planner
failures: no increase anywhere" but the mechanical `overall_pass` never
actually combined a planner-failure comparison) -- from the ALREADY-SAVED
artifacts. Runs ZERO episodes: 830M/812M's planner-failure counts are
already in the saved `router_v2_guarded_development_validation.json`
(`combined_summary["planner_failure_rate"]`); 640M/663M/26-fixtures need
their cached "A" reference files re-read (same files scratchpad_router_v2_
guarded_development_validation.py already loaded once) plus the "D"
per-episode data already embedded in the saved artifact's `D_full`
sections -- a pure artifact audit, not a simulation rerun.
"""
from __future__ import annotations

import json
from pathlib import Path

from scratchpad_router_v2_guarded_development_validation import compute_gate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    saved = json.loads((ROOT / "evaluations" / "router_v2_guarded_development_validation.json").read_text(encoding="utf-8"))

    a_830_obstacle = saved["830M_obstacle"]["A"]
    d_830_obstacle = saved["830M_obstacle"]["D"]
    a_812 = saved["812M"]["A"]
    d_812 = saved["812M"]["D"]
    d_640 = saved["640M"]["D_full"]
    d_663 = saved["663M"]["D_full"]
    d_fixtures = saved["26_fixtures"]["D_full"]

    a_640 = json.loads((ROOT / "evaluations" / "general_router_bridge_check_corrected_previous_steering.json").read_text(encoding="utf-8"))
    a_663 = json.loads((ROOT / "evaluations" / "beginner_routing_two_wall_s_route_eval_corrected_previous_steering.json").read_text(encoding="utf-8"))
    a_fixtures = json.loads((ROOT / "evaluations" / "routing_regression_fixtures_result_pre_selector_patch.json").read_text(encoding="utf-8"))
    a_fixtures_by_name = {r["name"]: r for r in a_fixtures["results"]}

    gate = compute_gate(
        a_830_obstacle=a_830_obstacle, d_830_obstacle=d_830_obstacle, a_812=a_812, d_812=d_812,
        a_640=a_640, d_640=d_640, a_663=a_663, d_663=d_663,
        a_fixtures_by_name=a_fixtures_by_name, d_fixtures=d_fixtures,
    )

    saved["gate"] = gate
    out_path = ROOT / "evaluations" / "router_v2_guarded_development_validation.json"
    out_path.write_text(json.dumps(saved, indent=2, default=str), encoding="utf-8")
    print(f"\nCorrected gate (with planner-failure checks) saved to {out_path}")


if __name__ == "__main__":
    main()
