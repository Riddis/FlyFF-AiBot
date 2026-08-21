"""Zero-compatibility product gate (2026-08-21 post-migration compatibility
purge). Structurally verifies the specific compatibility dirt this purge
removed does not silently come back -- not a general-purpose "no legacy
code" policy, and not brittle against historical prose: it checks current
source/config structure, never docs/migration/** content.

legacy/manifest_compat.py is intentionally NOT covered here: it was
investigated during the purge and kept, proven load-bearing for real
recorder-1.7.0/1.9.0-era archives (recordings/INDEX.json,
recordings/recording_provenance.json) that still need it to remain
readable -- a real current external dependency, not migration debt. A
gate asserting its absence would be asserting something false."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_canonical_owners_has_zero_shim_entries() -> None:
    registry = tomllib.loads((REPO / "CANONICAL_OWNERS.toml").read_text(encoding="utf-8"))
    assert registry.get("shim", []) == [], (
        "a [[shim]] entry reappeared in CANONICAL_OWNERS.toml -- re-justify it against real "
        "current evidence (per the 2026-08-21 compatibility purge's methodology) before adding it back"
    )


def test_pickle_identity_shim_modules_do_not_exist() -> None:
    for relative in ("simulator/kinodynamic_route_planner.py", "simulator/movement_kernel.py"):
        assert not (REPO / relative).is_file(), (
            f"{relative} reappeared -- KinoState/RouteEdgeInfo/AdvanceResult belong only in "
            "navigation/*, with their natural module identity (see ADR 0002's Retirement section)"
        )


def test_navigation_classes_carry_natural_module_identity() -> None:
    for relative, names in (
        ("navigation/kinodynamic_route_planner.py", ("KinoState", "RouteEdgeInfo")),
        ("navigation/movement_kernel.py", ("AdvanceResult",)),
    ):
        source = (REPO / relative).read_text(encoding="utf-8")
        for name in names:
            assert f"{name}.__module__" not in source, (
                f"{relative}: a __module__ override for {name} reappeared -- this class must "
                "carry its natural navigation.* identity, not a pinned simulator.* path"
            )


def test_farming_observation_module_does_not_reexport_schema_constants() -> None:
    import farming.observation as observation_module

    for name in ("OBSERVATION_SCHEMA_HASH", "OBSERVATION_SCHEMA_ID"):
        assert not hasattr(observation_module, name), (
            f"farming.observation re-exports {name} again -- its canonical owner is "
            "farming/observation_contract.py; import it from there, not from farming.observation"
        )


def test_utils_package_does_not_exist_at_root() -> None:
    assert not (REPO / "utils").exists(), (
        "utils/ reappeared at repository root -- its one file (helpers.py) belongs in libs/ "
        "(the 2026-08-21 root-accountability correction)"
    )
