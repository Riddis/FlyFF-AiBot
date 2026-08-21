"""Zero-compatibility product gate (2026-08-21 post-migration compatibility
purge, extended in the same day's follow-up correction, extended again the
same day by a broader stale-reference sweep that found two more instances
of the same pattern). Structurally
verifies the specific compatibility dirt these two passes removed does
not silently come back -- not a general-purpose "no legacy code" policy,
and not brittle against historical prose: it checks current source/config
structure, never docs/migration/** content.

legacy/manifest_compat.py is intentionally NOT covered here: it was
investigated during the purge (twice, independently) and kept both
times, proven load-bearing for real recorder-1.7.0/1.9.0-era archives
(recordings/INDEX.json, recordings/recording_provenance.json) that still
need it to remain readable -- a real current external dependency, not
migration debt. A gate asserting its absence would be asserting
something false."""

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


def test_farming_package_does_not_export_active_metadataless_constants() -> None:
    import farming

    for name in ("ACTIVE_METADATALESS_MODEL_SHA256", "ACTIVE_METADATALESS_MODEL_CONTRACT_HASH"):
        assert name not in farming.__all__, (
            f"{name} reappeared in farming.__all__ -- a repo-wide search found zero consumers "
            "for it anywhere; re-justify against real current evidence before re-adding it"
        )
        assert not hasattr(farming, name), f"farming.{name} reappeared as a live attribute"


def test_native_reader_does_not_accept_deprecated_hp_offset_argument() -> None:
    import inspect

    from position.IndependentNativeReader import IndependentNativeReader

    parameters = inspect.signature(IndependentNativeReader.__init__).parameters
    assert "current_hp_offset" not in parameters, (
        "IndependentNativeReader.__init__ regained the deprecated current_hp_offset argument -- "
        "its only caller was a self-labeled-deprecated devtools diagnostic flag, never a "
        "canonical production caller"
    )


def test_farming_config_has_no_old_key_alias_table() -> None:
    source = (REPO / "farming" / "config.py").read_text(encoding="utf-8")
    assert "unified_control_interval_seconds" not in source, (
        "farming/config.py reintroduced an old-key alias mapping -- the shipped "
        "farming/native_farming.json uses canonical key names directly; migrate the config "
        "file, don't reintroduce translation logic"
    )
    assert "teleport_pointer_grace_seconds" not in source
    assert "teleport_pointer_poll_seconds" not in source


def test_shipped_farming_config_uses_only_canonical_key_names() -> None:
    import json

    payload = json.loads((REPO / "farming" / "native_farming.json").read_text(encoding="utf-8"))
    for old_key in (
        "unified_control_interval_seconds",
        "teleport_pointer_grace_seconds",
        "teleport_pointer_poll_seconds",
    ):
        assert old_key not in payload, f"farming/native_farming.json still uses the old key {old_key!r}"


def test_router_selector_historical_scratchpad_family_stays_removed() -> None:
    for relative in (
        "simulator/scratchpad/scratchpad_historical_reproduction_guard.py",
        "simulator/scratchpad/scratchpad_general_router_episode.py",
        "simulator/scratchpad/scratchpad_beginner_navigation_mix_pools.py",
        "simulator/scratchpad/scratchpad_legacy_qualified_selector.py",
    ):
        assert not (REPO / relative).is_file(), (
            f"{relative} reappeared -- this dated, one-time 2026-08-14/15 router-selector "
            "investigation's outcome already lives in navigation/kinodynamic_route_planner.py; "
            "reproduce it via git tag router-selector-historical-scratchpad-pre-removal-20260821 "
            "instead of restoring frozen source to current HEAD"
        )


def test_native_kill_tracker_has_no_dead_backward_compat_constructor_args() -> None:
    import inspect

    from farming.kills import NativeKillTracker

    parameters = inspect.signature(NativeKillTracker.__init__).parameters
    for name in ("minimum_absence_seconds", "dedupe_seconds"):
        assert name not in parameters, (
            f"NativeKillTracker.__init__ regained {name!r} -- a repo-wide search found zero "
            "callers passing it anywhere (production or test); it was never stored or read "
            "even when accepted, pure dead backward-compat surface for the old absence tracker"
        )


def test_recorder_format_reexport_module_does_not_exist() -> None:
    assert not (REPO / "devtools" / "recorder" / "format.py").is_file(), (
        "devtools/recorder/format.py reappeared -- its only two consumers "
        "(devtools/recorder/session.py, tests/test_recorder_core.py) were migrated to import "
        "runtime.recording_format directly; re-justify a re-export module against real current "
        "evidence before adding it back"
    )


def test_active_field_profiler_facade_does_not_exist() -> None:
    assert not (REPO / "devtools" / "recorder" / "active_field_profiler.py").is_file(), (
        "devtools/recorder/active_field_profiler.py reappeared -- it was a behavior-free "
        "compatibility re-export of position.profiling.active_field_profiler.ActiveFieldProfiler; "
        "its only importers (devtools/recorder/session.py, tests/test_recorder_core.py) were "
        "migrated to import the canonical module directly"
    )


def test_independent_native_reader_has_no_hp_offset_backward_compat_alias() -> None:
    from position.IndependentNativeReader import IndependentNativeReader

    assert not hasattr(IndependentNativeReader, "hp_offset"), (
        "IndependentNativeReader.hp_offset (a backward-compatible alias for "
        "player_hp_offset) reappeared -- a repo-wide search found zero current call sites"
    )


def test_bot_set_config_rejects_unknown_keys() -> None:
    import pytest

    from bot.Bot import Bot

    bot = Bot.__new__(Bot)
    bot.config = {"show_frames": False}
    with pytest.raises(ValueError, match="Unknown bot config key"):
        bot.set_config(some_typo_or_obsolete_key=True)


def test_canonical_owners_has_no_migration_phase_bookkeeping_fields() -> None:
    registry = tomllib.loads((REPO / "CANONICAL_OWNERS.toml").read_text(encoding="utf-8"))
    assert "strategy" not in registry, "CANONICAL_OWNERS.toml regained an unused 'strategy' field"
    for concept in registry.get("concept", []):
        assert "resolution_phase" not in concept, (
            f"{concept.get('id')}: resolution_phase reappeared -- confirmed unused by any "
            "current tool/test; historical phase records belong in a commit message, not live config"
        )
        assert "permitted_compatibility" not in concept, (
            f"{concept.get('id')}: permitted_compatibility reappeared -- confirmed unused by any "
            "current tool/test"
        )
