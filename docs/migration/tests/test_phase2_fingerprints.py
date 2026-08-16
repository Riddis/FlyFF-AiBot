"""Phase-2 frozen-fingerprint gates as repeatable tests.

These are the CHEAP always-on gates (G4, G11, G10a). They must never import
Torch. The expensive Torch-backed G10b lives in
``docs/migration/tools/phase2_representative_load.py`` and is run explicitly,
not from this module.

All pinned literals live in ``docs/migration/PHASE2_FINGERPRINTS.toml``; this
file deliberately hardcodes none of them, so there is exactly one place to look
for a frozen value.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "docs/migration/tools/phase2_fingerprints.py"
SPEC = importlib.util.spec_from_file_location("phase2_fingerprints", TOOL)
assert SPEC is not None and SPEC.loader is not None
fingerprints = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fingerprints
SPEC.loader.exec_module(fingerprints)

FP = fingerprints.load_fingerprints(REPO)
CORPUS = Path(FP["g10"]["preserved_corpus"])
needs_corpus = pytest.mark.skipif(
    not CORPUS.is_dir(),
    reason=f"preserved checkpoint corpus unavailable at {CORPUS}",
)


def _torch_modules() -> set[str]:
    return {name for name in sys.modules if name == "torch" or name.startswith("torch.")}


# ---------------------------------------------------------------------------
# G4
# ---------------------------------------------------------------------------


def test_g4_contract_fingerprints_match_current_source() -> None:
    failures, evidence = fingerprints.check_g4(REPO, FP)
    assert failures == [], json.dumps(failures, indent=2)
    assert evidence


def test_g4_requires_live_schema_hash_recomputation() -> None:
    """5.2: catching descriptor drift needs recomputation, not constant equality."""
    pinned = FP["g4"]["observation_schema_hash"]["value"]
    for root in ("flyff_farming_simulator", "foreground_vision_bot"):
        probe = fingerprints.probe_root(REPO, root)
        assert probe["OBSERVATION_SCHEMA_HASH_RECOMPUTED"] == pinned, root
        # The declared constant and the recomputed value must also agree.
        assert probe["OBSERVATION_SCHEMA_HASH"] == probe["OBSERVATION_SCHEMA_HASH_RECOMPUTED"], root


def test_g4_covers_every_current_owner_not_one_convenient_copy() -> None:
    g4 = FP["g4"]
    for concept in ("observation_schema_id", "observation_schema_hash"):
        owners = g4[concept]["owners"]
        assert len(owners) == 3, concept
        for owner in owners:
            assert (REPO / owner).is_file(), owner
    for concept in ("observation_size", "policy_action_nvecs", "model_contract_metadata_version"):
        owners = g4[concept]["owners"]
        assert len(owners) == 2, concept
        for owner in owners:
            assert (REPO / owner).is_file(), owner


def test_g4_physics_version_is_pinned_from_source_not_from_a_prompt() -> None:
    physics = FP["g4"]["physics_version"]
    source = (REPO / physics["owners"][0]).read_text(encoding="utf-8")
    assert f'{physics["symbol"]} = "{physics["value"]}"' in source
    # The plan's name for this concept does not exist as a symbol; the registry
    # must say so rather than silently inventing one.
    assert physics["plan_name"] == "PHYSICS_VERSION"
    assert "PHYSICS_VERSION" not in {line.split("=")[0].strip() for line in source.splitlines() if "=" in line}


# ---------------------------------------------------------------------------
# G11
# ---------------------------------------------------------------------------


def test_g11_map_artifacts_match_pinned_bytes_in_both_locations() -> None:
    failures, evidence = fingerprints.check_g11(REPO, FP)
    assert failures == [], json.dumps(failures, indent=2)
    assert len(evidence["hashes"]) == 6
    assert all(evidence["pairs"].values())
    assert evidence["marker_present"]


def test_g11_pairs_remain_duplicated_and_are_not_deduplicated() -> None:
    """The two raw Tower copies stay duplicated on purpose."""
    for location in FP["g11"]["locations"]:
        for name in FP["g11"]["artifacts"]:
            assert (REPO / location / name).is_file(), f"{location}/{name} must remain a real file"


# ---------------------------------------------------------------------------
# G10a
# ---------------------------------------------------------------------------


@needs_corpus
def test_g10a_regenerates_every_phase0_field_exactly() -> None:
    failures, evidence = fingerprints.check_g10a(REPO, FP, CORPUS, write_supplement=False)
    assert failures == [], json.dumps(failures[:20], indent=2)
    assert evidence["checkpoints_compared"] == FP["g10"]["checkpoint_count"]
    assert evidence["field_mismatches"] == 0


@needs_corpus
def test_g10a_reproduces_full_serialized_reference_corpus() -> None:
    _failures, evidence = fingerprints.check_g10a(REPO, FP, CORPUS, write_supplement=False)
    assert evidence["reference_rows_equal"] is True
    assert evidence["regenerated_reference_rows"] == FP["g10"]["module_reference_rows"]


def test_g10a_supplement_documents_its_own_provenance_truthfully() -> None:
    """Phase-0 lacked policy_kwargs/net_arch; the supplement must say so, not backdate."""
    supplement = REPO / FP["g10"]["phase2_supplement"]
    assert supplement.is_file()
    with supplement.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == FP["g10"]["checkpoint_count"]
    assert all(row["provenance"] == fingerprints.SUPPLEMENT_PROVENANCE for row in rows)
    # Every row is keyed by repo-relative path + checkpoint SHA-256.
    assert len({(row["path"], row["checkpoint_sha256"]) for row in rows}) == len(rows)

    with (REPO / fingerprints.PHASE0_INVENTORY).open(encoding="utf-8", newline="") as handle:
        phase0_fields = set(next(csv.reader(handle, delimiter="\t")))
    assert "policy_kwargs" not in phase0_fields
    assert "net_arch" not in phase0_fields


def test_g10a_preserves_hard_repo_local_abi_paths() -> None:
    required = set(FP["g10"]["required_repo_local_references"]["values"])
    with (REPO / fingerprints.PHASE0_REFERENCES).open(encoding="utf-8", newline="") as handle:
        present = {
            f"{row['module_reference']}.{row['qualname_if_found']}"
            for row in csv.DictReader(handle, delimiter="\t")
            if row["module_reference"] != "NONE"
        }
    assert required.issubset(present), sorted(required - present)


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------


def test_cheap_phase2_gates_never_import_torch() -> None:
    before = _torch_modules()
    fingerprints.check_g11(REPO, FP)
    fingerprints.check_g4(REPO, FP)
    assert _torch_modules() == before


@needs_corpus
def test_g10a_is_deterministic_across_runs() -> None:
    first = fingerprints.check_g10a(REPO, FP, CORPUS, write_supplement=False)[1]
    second = fingerprints.check_g10a(REPO, FP, CORPUS, write_supplement=False)[1]
    assert first == second


def test_representative_selection_was_frozen_before_execution() -> None:
    """G10b's file list must be reproducible from the plan + Phase-0 inventory."""
    selection = REPO / "docs/migration/PHASE2_REPRESENTATIVE_SELECTION.tsv"
    baseline = REPO / "docs/migration/PHASE2_REPRESENTATIVE_LOAD_BASELINE.tsv"
    assert selection.is_file() and baseline.is_file()
    with selection.open(encoding="utf-8", newline="") as handle:
        declared = list(csv.DictReader(handle, delimiter="\t"))
    with baseline.open(encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle, delimiter="\t"))
    assert {r["path"] for r in declared} == {r["path"] for r in loaded}
    assert {r["sha256"] for r in declared} == {r["sha256"] for r in loaded}
    # All seven declared categories are represented.
    assert len({r["category"] for r in declared}) == 7
    assert len(declared) == 17


def test_representative_load_outcomes_are_internally_consistent() -> None:
    baseline = REPO / "docs/migration/PHASE2_REPRESENTATIVE_LOAD_BASELINE.tsv"
    with baseline.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        if row["outcome"] == "loaded":
            # A successful load must resolve to the recorded checkpoint class.
            assert row["matches_inventory"] == "True", row["path"]
            assert row["policy_class_module"] == row["inventory_policy_class_module"], row["path"]
            assert row["policy_class_qualname"] == row["inventory_policy_class_qualname"], row["path"]
        else:
            # An expected failure is valid evidence, but must be recorded exactly.
            assert row["exception_type"], row["path"]
            assert row["exception_message"], row["path"]
    # 928 navigation models keep their (928,) observation and [3 3] action contract.
    nav = [r for r in rows if r["outcome"] == "loaded" and "(928,)" in r["observation_space"]]
    assert nav, "expected at least one loaded 928-era navigation model"
    for row in nav:
        assert "MultiDiscrete([3 3])" in row["action_space"], row["path"]
