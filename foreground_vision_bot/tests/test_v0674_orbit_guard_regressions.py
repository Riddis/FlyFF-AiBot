from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v0674_installs_after_eva_continuity() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    assert source.index("install_v0673_fixes()") < source.index("install_v0674_fixes()")


def test_v0674_reverses_bad_arcs_and_abandons_orbits() -> None:
    source = (ROOT / "libs" / "V0674OrbitGuard.py").read_text(encoding="utf-8")
    assert "reverse_bad_arc" in source
    assert "abandon_orbit" in source
    assert "TARGET_BLACKLIST_SECONDS" in source
    assert "progress <= NEGATIVE_PROGRESS_CELLS" in source


def test_v0674_dry_run_reports_orbit_diagnostics() -> None:
    source = (ROOT / "native_farming.py").read_text(encoding="utf-8")
    assert "orbit={navigation.get('orbit_event', '--')}" in source
    assert "correction={navigation.get('orbit_correction', '--')}" in source
    assert "blacklist={info.get('orbit_blacklist_size', 0)}" in source
