from __future__ import annotations

import pytest

from simulator.curriculum_stages import (
    CANONICAL_STAGES,
    internal_stage,
    uses_recovery_assistance,
)


def test_basic_and_beginner_share_the_early_internal_stage() -> None:
    assert internal_stage("basic") == "early"
    assert internal_stage("beginner") == "early"


def test_intermediate_and_advanced_map_onto_themselves() -> None:
    assert internal_stage("intermediate") == "intermediate"
    assert internal_stage("advanced") == "advanced"


def test_only_basic_uses_recovery_assistance() -> None:
    assert uses_recovery_assistance("basic") is True
    for canonical_name in ("beginner", "intermediate", "advanced"):
        assert uses_recovery_assistance(canonical_name) is False


def test_lookup_is_case_and_whitespace_insensitive() -> None:
    assert internal_stage(" Basic ") == "early"
    assert uses_recovery_assistance("BEGINNER") is False


def test_unknown_stage_name_raises_with_valid_options_listed() -> None:
    with pytest.raises(KeyError, match="basic"):
        internal_stage("expert")
    with pytest.raises(KeyError):
        uses_recovery_assistance("nonexistent")


def test_canonical_stages_cover_exactly_the_four_project_names() -> None:
    assert set(CANONICAL_STAGES) == {"basic", "beginner", "intermediate", "advanced"}
