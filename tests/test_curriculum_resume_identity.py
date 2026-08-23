"""Proves the resume-identity gate (`simulator.curriculum_resume_identity`)
rejects incompatible historical AND same-generation-but-wrong state, not
just a different architecture generation.

Blocker A (final pre-merge remediation, 2026-08-23): an earlier version of
this gate only checked architecture-generation-level facts (action schema,
observation size, frozen-navigation SHA) -- a cache from the SAME
architecture but a DIFFERENT checkpoint, parent, curriculum stage, manifest,
or evaluation configuration was still silently reusable. This module now
also validates checkpoint identity by CONTENT (SHA-256, never path alone),
parent identity, curriculum stage, manifest content, and a canonical
evaluation-config fingerprint -- and writes/reads current-generation state
under a namespaced path (`current_generation_path`) so historical evidence
under the old filename is never read, written, or archived-then-overwritten
by current code at all.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from simulator.curriculum_resume_identity import (
    checkpoint_identity,
    current_generation_path,
    evaluation_cache_identity,
    identity_mismatch_reason,
    load_cached_evaluation_if_current,
    load_resumable_round_reports,
    manifest_identity,
    next_resumable_round,
    round_identity,
    round_record_validity_reason,
    sha256_of_file,
    stable_config_fingerprint,
    with_evaluation_cache_identity,
    with_round_identity,
)
from simulator.navigation_subpolicy import FROZEN_NAVIGATION_CHECKPOINT_SHA256

BEGINNER = importlib.import_module("simulator.tools.RUN_CANONICAL_BEGINNER")
INTERMEDIATE = importlib.import_module("simulator.tools.RUN_CANONICAL_INTERMEDIATE")
ADVANCED = importlib.import_module("simulator.tools.RUN_CANONICAL_ADVANCED")


def _write(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


# --- basic content-identity primitives ------------------------------------


def test_sha256_of_file_is_content_based(tmp_path: Path):
    a = _write(tmp_path / "a.zip", b"checkpoint bytes v1")
    b = _write(tmp_path / "b.zip", b"checkpoint bytes v1")  # same content, different path
    c = _write(tmp_path / "c.zip", b"checkpoint bytes v2")  # same-ish name pattern, different content
    assert sha256_of_file(a) == sha256_of_file(b)
    assert sha256_of_file(a) != sha256_of_file(c)


def test_checkpoint_identity_is_content_based_not_path_alone(tmp_path: Path):
    path = tmp_path / "canonical_beginner_ppo_010k.zip"
    _write(path, b"original bytes")
    identity_v1 = checkpoint_identity(path)
    _write(path, b"replaced bytes -- same path, different content")
    identity_v2 = checkpoint_identity(path)
    assert identity_v1["path"] == identity_v2["path"]
    assert identity_v1["sha256"] != identity_v2["sha256"], "a same-named file with different bytes must have a different identity"


def test_checkpoint_identity_is_none_for_missing_or_not_yet_materialized_path(tmp_path: Path):
    assert checkpoint_identity(None) is None
    assert checkpoint_identity(tmp_path / "does_not_exist.zip") is None


def test_manifest_identity_is_content_based(tmp_path: Path):
    m1 = _write(tmp_path / "heldout.json", b'{"layouts": {}}')
    id1 = manifest_identity(m1)
    _write(m1, b'{"layouts": {"new_layout": {}}}')
    id2 = manifest_identity(m1)
    assert id1["content_sha256"] != id2["content_sha256"]


def test_stable_config_fingerprint_is_order_independent():
    a = stable_config_fingerprint({"episode_seconds": 150.0, "seeds": [0, 1]})
    b = stable_config_fingerprint({"seeds": [0, 1], "episode_seconds": 150.0})
    assert a == b


def test_stable_config_fingerprint_changes_with_content():
    a = stable_config_fingerprint({"episode_seconds": 150.0})
    b = stable_config_fingerprint({"episode_seconds": 999.0})
    assert a != b


def test_current_generation_path_inserts_namespace_before_suffix():
    p = current_generation_path("simulator/evaluations/canonical_beginner_run_summary.json")
    assert p.name == "canonical_beginner_run_summary.target_event_v1.json"


def test_frozen_navigation_sha_matches_the_real_frozen_checkpoint():
    identity = round_identity(stage="early", declared_parent_checkpoint="x", current_checkpoint="y")
    assert identity["navigation_checkpoint_sha256"] == FROZEN_NAVIGATION_CHECKPOINT_SHA256


# --- round identity: identity_mismatch_reason + round_record_validity_reason ---


@pytest.fixture
def parent_checkpoint(tmp_path: Path) -> Path:
    return _write(tmp_path / "canonical_basic_graduated.zip", b"parent checkpoint bytes")


@pytest.fixture
def current_checkpoint(tmp_path: Path) -> Path:
    return _write(tmp_path / "canonical_beginner_ppo_010k.zip", b"round 1 checkpoint bytes")


def _valid_round_record(
    parent_checkpoint: Path, current_checkpoint: Path, *, stage: str = "early", round_number: int = 1,
    round_passed_absolute_bar: bool = True, consecutive_passes: int = 1,
) -> dict:
    return with_round_identity(
        {
            "round": round_number,
            "round_passed_absolute_bar": round_passed_absolute_bar,
            "consecutive_passes": consecutive_passes,
            "carried_forward_checkpoint": str(current_checkpoint.resolve()),
        },
        stage=stage, declared_parent_checkpoint=parent_checkpoint, current_checkpoint=current_checkpoint,
    )


def test_matching_round_record_is_valid(parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is None


def test_round_record_with_wrong_stage_is_rejected(parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint, stage="early")
    reason = round_record_validity_reason(record, stage="intermediate", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "curriculum_stage" in reason


def test_round_record_with_wrong_parent_path_is_rejected(parent_checkpoint, current_checkpoint, tmp_path):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    other_parent = _write(tmp_path / "some_other_parent.zip", b"a different parent entirely")
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=other_parent)
    assert reason is not None
    assert "declared_parent_checkpoint" in reason


def test_round_record_with_wrong_parent_sha_is_rejected(parent_checkpoint, current_checkpoint):
    """SAME parent path, but the parent checkpoint's bytes changed since
    this round was recorded (e.g. the graduated Basic checkpoint was
    regenerated/quarantined) -- must reject even though the path matches."""
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    _write(parent_checkpoint, b"parent checkpoint bytes -- REPLACED")
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "declared_parent_checkpoint_sha256" in reason


def test_round_record_with_stale_checkpoint_path_is_rejected(parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    current_checkpoint.unlink()
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "no longer exists" in reason


def test_round_record_with_changed_checkpoint_bytes_is_rejected(parent_checkpoint, current_checkpoint):
    """SAME path, DIFFERENT bytes -- the exact scenario a path-only resume
    check cannot catch."""
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    _write(current_checkpoint, b"round 1 checkpoint bytes -- REPLACED WITH DIFFERENT MODEL")
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "bytes changed" in reason


def test_round_record_with_legacy_action_schema_is_rejected(parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    record["identity"]["policy_action_nvec"] = [3, 3]  # retired steering+event contract
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "policy_action_nvec" in reason


def test_round_record_with_no_identity_is_rejected(parent_checkpoint, current_checkpoint):
    record = {"round": 4, "consecutive_passes": 2, "carried_forward_checkpoint": str(current_checkpoint)}
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "no identity recorded" in reason


def test_round_record_missing_carried_forward_checkpoint_is_rejected(parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    del record["carried_forward_checkpoint"]
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None


# --- round_record_validity_reason: checkpoint PATH consistency (defect 1) --
#
# `identity.current_checkpoint` (the round's own vouched-for checkpoint) and
# `carried_forward_checkpoint` (what the runner will actually resume from)
# must refer to the SAME canonical path, not merely matching bytes at a
# DIFFERENT path -- see task section 2/3/4.


def test_round_record_rejects_carried_forward_checkpoint_at_a_different_path_with_identical_bytes(parent_checkpoint, tmp_path):
    """Task section 4.A: checkpoint_A and checkpoint_B hold IDENTICAL bytes
    at DIFFERENT paths. The record's own identity vouches for checkpoint_A
    (that is what was actually produced/evaluated this round), but
    carried_forward_checkpoint names checkpoint_B. Matching bytes elsewhere
    must not launder a different path into "the same checkpoint"."""
    checkpoint_a = _write(tmp_path / "checkpoint_a.zip", b"identical checkpoint bytes")
    checkpoint_b = _write(tmp_path / "checkpoint_b.zip", b"identical checkpoint bytes")
    record = _valid_round_record(parent_checkpoint, checkpoint_a)
    record["carried_forward_checkpoint"] = str(checkpoint_b.resolve())
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None
    assert "does not refer to the same checkpoint" in reason


def test_round_record_accepts_same_path_same_bytes(parent_checkpoint, current_checkpoint):
    """Task section 4.B: identity.current_checkpoint and
    carried_forward_checkpoint both name the same checkpoint, bytes
    unchanged -- must accept."""
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is None


def test_round_record_accepts_carried_forward_checkpoint_differently_spelled_but_canonically_identical_path(parent_checkpoint, current_checkpoint):
    """Task section 4.D: same physical file, different spelling (redundant
    `..` traversal) -- must accept once canonicalized, must not compare raw
    strings."""
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    differently_spelled = current_checkpoint.parent / ".." / current_checkpoint.parent.name / current_checkpoint.name
    assert str(differently_spelled) != str(current_checkpoint.resolve()), "test setup must actually exercise differently-spelled paths"
    record["carried_forward_checkpoint"] = str(differently_spelled)
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is None


def test_round_record_with_no_identity_current_checkpoint_is_rejected(parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    del record["identity"]["current_checkpoint"]
    reason = round_record_validity_reason(record, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert reason is not None


# --- load_resumable_round_reports: real temporary files -------------------


def test_load_resumable_round_reports_accepts_a_valid_matching_chain(tmp_path, parent_checkpoint, current_checkpoint):
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps([_valid_round_record(parent_checkpoint, current_checkpoint)]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert len(result) == 1


def test_load_resumable_round_reports_rejects_orphaned_current_checkpoint(tmp_path, parent_checkpoint, current_checkpoint):
    """Task section 13: a validated round record whose checkpoint file has
    since vanished must not silently resume."""
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    current_checkpoint.unlink()
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps([record]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def test_load_resumable_round_reports_never_mutates_the_file_on_rejection(tmp_path, parent_checkpoint, current_checkpoint):
    record = {"round": 4, "consecutive_passes": 2, "carried_forward_checkpoint": str(current_checkpoint)}  # no identity: legacy shape
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    original_bytes = json.dumps([record]).encode("utf-8")
    summary_path.write_bytes(original_bytes)
    load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert summary_path.read_bytes() == original_bytes, "rejection must never mutate the file it read"


def test_load_resumable_round_reports_missing_file_returns_empty(tmp_path, parent_checkpoint):
    result = load_resumable_round_reports(tmp_path / "does_not_exist.json", log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def _build_round_chain(tmp_path: Path, parent_checkpoint: Path, round_numbers: list[int], *, stage: str = "early") -> list[dict]:
    """Builds a list of round records, one per entry in `round_numbers` (in
    the given order), each carrying its own distinct real temp checkpoint
    file so content-SHA validation is exercised for real, not stubbed.
    Every built record passes the absolute bar, with `consecutive_passes`
    incrementing by POSITION in the list (1, 2, 3, ...) so the chain also
    satisfies the pass-sequence consistency check by construction; tests
    that need a non-contiguous or otherwise-invalid chain are expected to
    be rejected by an earlier check (round numbering / identity) before
    pass-sequence consistency would even matter."""
    records = []
    for position, n in enumerate(round_numbers, start=1):
        checkpoint = _write(tmp_path / f"round_{n}_checkpoint.zip", f"round {n} checkpoint bytes".encode())
        records.append(_valid_round_record(
            parent_checkpoint, checkpoint, stage=stage, round_number=n,
            round_passed_absolute_bar=True, consecutive_passes=position,
        ))
    return records


# --- load_resumable_round_reports: whole-chain validation + round sequence
# (defect 2, task sections 5-11) -------------------------------------------


@pytest.mark.parametrize("round_numbers", [[1], [1, 2], [1, 2, 3]])
def test_load_resumable_round_reports_accepts_contiguous_valid_chains(tmp_path, parent_checkpoint, round_numbers):
    records = _build_round_chain(tmp_path, parent_checkpoint, round_numbers)
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps(records), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert [r["round"] for r in result] == round_numbers


@pytest.mark.parametrize("round_numbers", [[2], [1, 3], [1, 2, 4], [1, 1]])
def test_load_resumable_round_reports_rejects_non_contiguous_round_sequences(tmp_path, parent_checkpoint, round_numbers):
    records = _build_round_chain(tmp_path, parent_checkpoint, round_numbers)
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps(records), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def test_load_resumable_round_reports_rejects_non_integer_round_value(tmp_path, parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    record["round"] = "1"  # a string, not the int the real runners always write
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps([record]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def test_load_resumable_round_reports_rejects_missing_round_value(tmp_path, parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    del record["round"]
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps([record]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def test_load_resumable_round_reports_rejects_invalid_first_round_even_with_valid_final_round(tmp_path, parent_checkpoint):
    """The core defect-2 regression (task section 10): Codex found that only
    the LAST round was validated, so an invalid round 1 followed by a valid
    round 2 was silently accepted (round 2 alone validated). The whole
    summary must now be rejected."""
    records = _build_round_chain(tmp_path, parent_checkpoint, [1, 2])
    records[0]["identity"]["curriculum_stage"] = "corrupted-stage"  # round 1 now invalid
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps(records), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == [], "an invalid round 1 must reject the WHOLE chain, even though round 2 alone would validate"


def test_load_resumable_round_reports_rejects_valid_prefix_with_invalid_final_round(tmp_path, parent_checkpoint):
    records = _build_round_chain(tmp_path, parent_checkpoint, [1, 2])
    records[1]["identity"]["curriculum_stage"] = "corrupted-stage"  # round 2 now invalid
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps(records), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def test_load_resumable_round_reports_never_mutates_the_file_on_invalid_prefix_rejection(tmp_path, parent_checkpoint):
    records = _build_round_chain(tmp_path, parent_checkpoint, [1, 2])
    records[0]["identity"]["curriculum_stage"] = "corrupted-stage"
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    original_bytes = json.dumps(records).encode("utf-8")
    summary_path.write_bytes(original_bytes)
    load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert summary_path.read_bytes() == original_bytes


# --- load_resumable_round_reports: strict schema validation (round-summary
# schema hardening, 2026-08-23) -- persisted JSON is untrusted input and
# must never crash startup, manufacture graduation progress, or partially
# resume. Every check below uses EXACT type comparisons, never truthy/loose
# equality (Python's `True == 1` and `1.0 == 1` would otherwise let
# malformed persisted state masquerade as valid). ------------------------


@pytest.mark.parametrize("payload", [{}, "abc", 1, True, None])
def test_load_resumable_round_reports_rejects_non_list_top_level_payload(tmp_path, parent_checkpoint, payload):
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def test_load_resumable_round_reports_accepts_empty_list_as_valid_empty_state(tmp_path, parent_checkpoint):
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text("[]", encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def test_load_resumable_round_reports_rejects_malformed_json_without_raising(tmp_path, parent_checkpoint):
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text("{not valid json", encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


@pytest.mark.parametrize("entry", [None, "round", 1, []])
def test_load_resumable_round_reports_rejects_non_dict_round_entry(tmp_path, parent_checkpoint, entry):
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps([entry]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def test_load_resumable_round_reports_rejects_whole_summary_on_a_malformed_entry_amid_otherwise_valid_records(tmp_path, parent_checkpoint):
    records = _build_round_chain(tmp_path, parent_checkpoint, [1, 2, 3])
    payload = [records[0], "garbage", records[2]]
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


@pytest.mark.parametrize("bad_round_value", [1.0, True, False, "1", None])
def test_load_resumable_round_reports_rejects_non_exact_int_round_value(tmp_path, parent_checkpoint, current_checkpoint, bad_round_value):
    """`isinstance(True, int)` is `True` in Python -- an exact `type(x) is
    int` check is required so `round: true` (bool) is never treated as
    round `1`, and `round: 1.0` (float) is never treated as the exact
    integer `1` a real runner always writes."""
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    record["round"] = bad_round_value
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps([record]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


@pytest.mark.parametrize("bad_value", [-1, 1.0, True, False, "1", None])
def test_load_resumable_round_reports_rejects_non_exact_nonnegative_int_consecutive_passes(tmp_path, parent_checkpoint, current_checkpoint, bad_value):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    record["consecutive_passes"] = bad_value
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps([record]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def test_load_resumable_round_reports_rejects_missing_consecutive_passes(tmp_path, parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    del record["consecutive_passes"]
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps([record]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


@pytest.mark.parametrize("bad_value", [0, 1, "true", "false", None])
def test_load_resumable_round_reports_rejects_non_exact_bool_round_passed_absolute_bar(tmp_path, parent_checkpoint, current_checkpoint, bad_value):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    record["round_passed_absolute_bar"] = bad_value
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps([record]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


def test_load_resumable_round_reports_rejects_missing_round_passed_absolute_bar(tmp_path, parent_checkpoint, current_checkpoint):
    record = _valid_round_record(parent_checkpoint, current_checkpoint)
    del record["round_passed_absolute_bar"]
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps([record]), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


# --- load_resumable_round_reports: consecutive_passes must be mathematically
# consistent with round_passed_absolute_bar across the whole chain, never
# just individually well-typed (prevents persisted state from falsely
# manufacturing graduation progress). ---------------------------------------


def _build_pass_sequence_chain(tmp_path: Path, parent_checkpoint: Path, sequence: list[tuple[bool, int]]) -> list[dict]:
    records = []
    for round_number, (round_passed, consecutive_passes) in enumerate(sequence, start=1):
        checkpoint = _write(tmp_path / f"seq_round_{round_number}_checkpoint.zip", f"seq round {round_number} bytes".encode())
        records.append(_valid_round_record(
            parent_checkpoint, checkpoint, round_number=round_number,
            round_passed_absolute_bar=round_passed, consecutive_passes=consecutive_passes,
        ))
    return records


@pytest.mark.parametrize("sequence", [
    [(False, 0)],
    [(True, 1)],
    [(True, 1), (True, 2)],
    [(True, 1), (False, 0)],
    [(True, 1), (False, 0), (True, 1)],
])
def test_load_resumable_round_reports_accepts_consistent_pass_sequences(tmp_path, parent_checkpoint, sequence):
    records = _build_pass_sequence_chain(tmp_path, parent_checkpoint, sequence)
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps(records), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert len(result) == len(sequence)


@pytest.mark.parametrize("sequence", [
    [(False, 1)],
    [(True, 0)],
    [(True, 2)],
    [(True, 1), (True, 1)],
    [(True, 1), (True, 3)],
    [(True, 1), (False, 1)],
    [(False, 0), (True, 2)],
])
def test_load_resumable_round_reports_rejects_inconsistent_pass_sequences(tmp_path, parent_checkpoint, sequence):
    """Each of these has individually well-typed fields but a
    `consecutive_passes` value that does not follow from
    `round_passed_absolute_bar` history -- persisted state manufacturing
    graduation progress that was never actually earned."""
    records = _build_pass_sequence_chain(tmp_path, parent_checkpoint, sequence)
    summary_path = current_generation_path(tmp_path / "canonical_beginner_run_summary.json")
    summary_path.write_text(json.dumps(records), encoding="utf-8")
    result = load_resumable_round_reports(summary_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []


# --- next_resumable_round ---------------------------------------------------


def test_next_resumable_round_is_1_for_empty_chain():
    assert next_resumable_round([]) == 1


def test_next_resumable_round_is_last_validated_round_plus_1():
    assert next_resumable_round([{"round": 1}, {"round": 2}, {"round": 3}]) == 4


def test_legacy_filename_at_old_path_is_never_read_by_the_current_loader(tmp_path, parent_checkpoint, current_checkpoint):
    """The un-namespaced historical filename is a completely different
    path from what load_resumable_round_reports is ever called with in
    the real runners (current_generation_path(...)) -- proves the
    separation directly: writing legacy-shaped content to the OLD path
    and calling the loader with the NEW (namespaced, nonexistent) path
    returns empty, never touching or even opening the old file."""
    legacy_path = tmp_path / "canonical_beginner_run_summary.json"
    legacy_path.write_text(json.dumps([{"round": 4, "consecutive_passes": 2}]), encoding="utf-8")
    current_path = current_generation_path(legacy_path)
    assert not current_path.exists()
    result = load_resumable_round_reports(current_path, log=lambda _m: None, stage="early", declared_parent_checkpoint=parent_checkpoint)
    assert result == []
    assert legacy_path.read_text(encoding="utf-8") == json.dumps([{"round": 4, "consecutive_passes": 2}])


# --- evaluation-cache identity: real temporary files -----------------------


@pytest.fixture
def heldout_manifest(tmp_path: Path) -> Path:
    return _write(tmp_path / "early_heldout.json", b'{"layouts": {"L1": {}}}')


def _cache_identity_kwargs(parent_checkpoint, evaluated_checkpoint, heldout_manifest, *, stage="early", role="heldout", episode_seconds=150.0):
    return dict(
        stage=stage, declared_parent_checkpoint=parent_checkpoint, evaluated_checkpoint=evaluated_checkpoint,
        evaluation_role=role, manifests={"heldout": str(heldout_manifest)},
        config={"episode_seconds": episode_seconds, "seeds": [0, 1]},
    )


def test_matching_cache_identity_is_accepted(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    report = with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)
    cache_path.write_text(json.dumps(report), encoding="utf-8")
    expected = evaluation_cache_identity(**kwargs)
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is not None
    assert result["n_episodes"] == 4


def test_cache_with_wrong_evaluated_checkpoint_sha_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    report = with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)
    cache_path.write_text(json.dumps(report), encoding="utf-8")
    _write(current_checkpoint, b"round 1 checkpoint bytes -- REPLACED")  # same path, different bytes
    expected = evaluation_cache_identity(**kwargs)  # recomputed fresh -- picks up the new SHA
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_wrong_parent_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest, tmp_path_factory):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps(with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)), encoding="utf-8")
    other_parent = _write(tmp_path / "different_parent.zip", b"a genuinely different parent")
    expected = evaluation_cache_identity(**{**kwargs, "declared_parent_checkpoint": other_parent})
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_wrong_parent_sha_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps(with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)), encoding="utf-8")
    _write(parent_checkpoint, b"parent checkpoint bytes -- REPLACED")
    expected = evaluation_cache_identity(**kwargs)  # fresh -- picks up new parent SHA
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_wrong_stage_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest, stage="early")
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps(with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)), encoding="utf-8")
    expected = evaluation_cache_identity(**{**kwargs, "stage": "intermediate"})
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_wrong_manifest_hash_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps(with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)), encoding="utf-8")
    _write(heldout_manifest, b'{"layouts": {"L1": {}, "L2": {}}}')  # manifest content changed
    expected = evaluation_cache_identity(**kwargs)  # fresh -- picks up new manifest hash
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_wrong_evaluation_config_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest, episode_seconds=150.0)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps(with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)), encoding="utf-8")
    expected = evaluation_cache_identity(**{**kwargs, "config": {"episode_seconds": 999.0, "seeds": [0, 1]}})
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_wrong_evaluation_role_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    """A cache produced for 'heldout' must never satisfy a 'challenge' (or
    any other role) lookup, even against the identical checkpoint."""
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest, role="heldout")
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps(with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)), encoding="utf-8")
    expected = evaluation_cache_identity(**{**kwargs, "evaluation_role": "challenge"})
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_no_identity_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    cache_path.write_text(json.dumps({"n_episodes": 4}), encoding="utf-8")  # no "identity" key at all
    expected = evaluation_cache_identity(**_cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest))
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_cache_with_legacy_event_only_action_schema_is_rejected(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    kwargs = _cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest)
    cache_path = current_generation_path(tmp_path / "canonical_x_pre_rehearsal_heldout.json")
    report = with_evaluation_cache_identity({"n_episodes": 4}, **kwargs)
    report["identity"]["policy_action_nvec"] = [3, 3]
    cache_path.write_text(json.dumps(report), encoding="utf-8")
    expected = evaluation_cache_identity(**kwargs)
    result = load_cached_evaluation_if_current(cache_path, log=lambda _m: None, expected_identity=expected)
    assert result is None


def test_missing_cache_file_returns_none(tmp_path, parent_checkpoint, current_checkpoint, heldout_manifest):
    expected = evaluation_cache_identity(**_cache_identity_kwargs(parent_checkpoint, current_checkpoint, heldout_manifest))
    assert load_cached_evaluation_if_current(tmp_path / "does_not_exist.json", log=lambda _m: None, expected_identity=expected) is None


# --- each canonical runner actually wires the new API in -----------------


@pytest.mark.parametrize("module,expected_declared_parent_attr", [
    (BEGINNER, "GRADUATED_BASIC_CHECKPOINT"),
    (INTERMEDIATE, "GRADUATED_BEGINNER_CHECKPOINT"),
    (ADVANCED, "GRADUATED_INTERMEDIATE_CHECKPOINT"),
])
def test_runner_source_calls_the_new_resume_identity_gate(module, expected_declared_parent_attr):
    """Source-level proof (not just that the shared gate works in
    isolation) that each canonical runner resume-gates through the CURRENT
    (content-based, generation-namespaced) API via its own
    `_resume_round_state` helper (which `main()` calls), and no longer
    contains the removed orphan-checkpoint glob-resume mechanism anywhere
    in the module."""
    import inspect

    whole_module_source = inspect.getsource(module)
    assert "load_resumable_round_reports" in whole_module_source
    assert "load_cached_evaluation_if_current" in whole_module_source
    assert "current_generation_path" in whole_module_source
    assert "_resume_round_state" in inspect.getsource(module.main)
    resume_state_source = inspect.getsource(module._resume_round_state)
    assert f"declared_parent_checkpoint={expected_declared_parent_attr}" in inspect.getsource(module.main)
    assert "load_resumable_round_reports" in resume_state_source
    assert "existing_rounds" not in whole_module_source, "orphan-checkpoint glob resume must be fully removed"
    assert "MODELS_DIR.glob" not in resume_state_source, "_resume_round_state must never inspect the filesystem for checkpoints"


# --- each canonical runner's actual _resume_round_state(): behavioral,
# not source-string, proof (task section 12) --------------------------------


@pytest.mark.parametrize("module,stage", [
    (BEGINNER, "early"),
    (INTERMEDIATE, "intermediate"),
    (ADVANCED, "advanced"),
])
class TestRunnerResumeRoundState:
    def test_no_summary_starts_from_parent_with_zero_passes_at_initial_round(self, module, stage, tmp_path, parent_checkpoint):
        summary_path = current_generation_path(tmp_path / "canonical_x_run_summary.json")
        round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
            summary_path, declared_parent_checkpoint=parent_checkpoint,
        )
        assert round_reports == []
        assert consecutive_passes == 0
        assert current_checkpoint == parent_checkpoint
        assert next_resumable_round(round_reports) == 1

    def test_valid_contiguous_summary_resumes_the_final_validated_round(self, module, stage, tmp_path, parent_checkpoint):
        records = _build_round_chain(tmp_path, parent_checkpoint, [1, 2], stage=stage)
        records[0]["consecutive_passes"] = 1
        records[1]["consecutive_passes"] = 2
        summary_path = current_generation_path(tmp_path / "canonical_x_run_summary.json")
        summary_path.write_text(json.dumps(records), encoding="utf-8")
        round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
            summary_path, declared_parent_checkpoint=parent_checkpoint,
        )
        assert [r["round"] for r in round_reports] == [1, 2]
        assert consecutive_passes == 2
        assert current_checkpoint.resolve() == Path(records[1]["carried_forward_checkpoint"]).resolve()
        assert next_resumable_round(round_reports) == 3

    def test_invalid_first_round_with_valid_final_round_rejects_whole_summary(self, module, stage, tmp_path, parent_checkpoint):
        records = _build_round_chain(tmp_path, parent_checkpoint, [1, 2], stage=stage)
        records[0]["identity"]["curriculum_stage"] = "corrupted-stage"
        summary_path = current_generation_path(tmp_path / "canonical_x_run_summary.json")
        summary_path.write_text(json.dumps(records), encoding="utf-8")
        round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
            summary_path, declared_parent_checkpoint=parent_checkpoint,
        )
        assert round_reports == []
        assert consecutive_passes == 0
        assert current_checkpoint == parent_checkpoint
        assert next_resumable_round(round_reports) == 1

    def test_same_byte_alternate_checkpoint_path_rejects_whole_summary(self, module, stage, tmp_path, parent_checkpoint):
        checkpoint_a = _write(tmp_path / "checkpoint_a.zip", b"identical bytes")
        checkpoint_b = _write(tmp_path / "checkpoint_b.zip", b"identical bytes")
        record = _valid_round_record(parent_checkpoint, checkpoint_a, stage=stage, round_number=1)
        record["carried_forward_checkpoint"] = str(checkpoint_b.resolve())
        summary_path = current_generation_path(tmp_path / "canonical_x_run_summary.json")
        summary_path.write_text(json.dumps([record]), encoding="utf-8")
        round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
            summary_path, declared_parent_checkpoint=parent_checkpoint,
        )
        assert round_reports == []
        assert consecutive_passes == 0
        assert current_checkpoint == parent_checkpoint

    def test_changed_checkpoint_bytes_rejects_whole_summary(self, module, stage, tmp_path, parent_checkpoint):
        checkpoint = _write(tmp_path / "round_1_checkpoint.zip", b"round 1 bytes")
        record = _valid_round_record(parent_checkpoint, checkpoint, stage=stage, round_number=1)
        _write(checkpoint, b"round 1 bytes -- REPLACED")
        summary_path = current_generation_path(tmp_path / "canonical_x_run_summary.json")
        summary_path.write_text(json.dumps([record]), encoding="utf-8")
        round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
            summary_path, declared_parent_checkpoint=parent_checkpoint,
        )
        assert round_reports == []
        assert consecutive_passes == 0
        assert current_checkpoint == parent_checkpoint

    # --- schema-hardening fallback matrix (task section 14) ----------------

    def test_non_list_top_level_payload_falls_back_to_declared_parent(self, module, stage, tmp_path, parent_checkpoint):
        summary_path = current_generation_path(tmp_path / "canonical_x_run_summary.json")
        summary_path.write_text(json.dumps({"round": 1}), encoding="utf-8")
        round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
            summary_path, declared_parent_checkpoint=parent_checkpoint,
        )
        assert round_reports == []
        assert consecutive_passes == 0
        assert current_checkpoint == parent_checkpoint
        assert next_resumable_round(round_reports) == 1

    def test_float_round_value_falls_back_to_declared_parent(self, module, stage, tmp_path, parent_checkpoint):
        checkpoint = _write(tmp_path / "round_1_checkpoint.zip", b"round 1 bytes")
        record = _valid_round_record(parent_checkpoint, checkpoint, stage=stage, round_number=1)
        record["round"] = 1.0
        summary_path = current_generation_path(tmp_path / "canonical_x_run_summary.json")
        summary_path.write_text(json.dumps([record]), encoding="utf-8")
        round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
            summary_path, declared_parent_checkpoint=parent_checkpoint,
        )
        assert round_reports == []
        assert consecutive_passes == 0
        assert current_checkpoint == parent_checkpoint
        assert next_resumable_round(round_reports) == 1

    def test_bool_round_value_falls_back_to_declared_parent(self, module, stage, tmp_path, parent_checkpoint):
        checkpoint = _write(tmp_path / "round_1_checkpoint.zip", b"round 1 bytes")
        record = _valid_round_record(parent_checkpoint, checkpoint, stage=stage, round_number=1)
        record["round"] = True
        summary_path = current_generation_path(tmp_path / "canonical_x_run_summary.json")
        summary_path.write_text(json.dumps([record]), encoding="utf-8")
        round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
            summary_path, declared_parent_checkpoint=parent_checkpoint,
        )
        assert round_reports == []
        assert consecutive_passes == 0
        assert current_checkpoint == parent_checkpoint
        assert next_resumable_round(round_reports) == 1

    def test_impossible_consecutive_passes_falls_back_to_declared_parent(self, module, stage, tmp_path, parent_checkpoint):
        checkpoint = _write(tmp_path / "round_1_checkpoint.zip", b"round 1 bytes")
        record = _valid_round_record(
            parent_checkpoint, checkpoint, stage=stage, round_number=1,
            round_passed_absolute_bar=True, consecutive_passes=99,
        )
        summary_path = current_generation_path(tmp_path / "canonical_x_run_summary.json")
        summary_path.write_text(json.dumps([record]), encoding="utf-8")
        round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
            summary_path, declared_parent_checkpoint=parent_checkpoint,
        )
        assert round_reports == []
        assert consecutive_passes == 0
        assert current_checkpoint == parent_checkpoint
        assert next_resumable_round(round_reports) == 1

    def test_malformed_record_amid_otherwise_valid_chain_falls_back_to_declared_parent(self, module, stage, tmp_path, parent_checkpoint):
        records = _build_round_chain(tmp_path, parent_checkpoint, [1, 2, 3], stage=stage)
        payload = [records[0], "garbage", records[2]]
        summary_path = current_generation_path(tmp_path / "canonical_x_run_summary.json")
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        round_reports, consecutive_passes, current_checkpoint = module._resume_round_state(
            summary_path, declared_parent_checkpoint=parent_checkpoint,
        )
        assert round_reports == []
        assert consecutive_passes == 0
        assert current_checkpoint == parent_checkpoint
        assert next_resumable_round(round_reports) == 1
