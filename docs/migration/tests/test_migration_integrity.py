from __future__ import annotations

import importlib.util
import json
import sys
import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "docs/migration/tools/migration_integrity.py"
SPEC = importlib.util.spec_from_file_location("phase1_migration_integrity", TOOL)
assert SPEC is not None and SPEC.loader is not None
integrity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = integrity
SPEC.loader.exec_module(integrity)


def test_r6_definition_scanner_ignores_imports_and_catches_new_definition() -> None:
    files = {
        "owner.py": "OBSERVATION_SIZE: int = 923\n",
        "use.py": "from owner import OBSERVATION_SIZE\nprint(OBSERVATION_SIZE)\n",
        "new_owner.py": "OBSERVATION_SIZE = 928\n",
    }
    owners = integrity.definition_owners(files, ["OBSERVATION_SIZE"])
    assert owners == {"OBSERVATION_SIZE": ["new_owner.py", "owner.py"]}


def test_ratchet_allows_shrink_but_rejects_growth() -> None:
    old = integrity.Finding("R7a", "contract", "old.py", "duplicate_definition=X")
    new = integrity.Finding("R7a", "contract", "new.py", "duplicate_definition=X")
    baseline = {"violations": {"R7a": [old.as_dict()]}}
    errors, resolved = integrity.ratchet_errors([], baseline)
    assert errors == []
    assert resolved == [old.key]
    errors, resolved = integrity.ratchet_errors([old, new], baseline)
    assert errors == [f"new_baseline_violation:{new.key}"]
    assert resolved == []


def test_r7a_rejects_unregistered_extra_owner() -> None:
    files = {"owner.py": "class Contract: pass\n", "rogue.py": "class Contract: pass\n"}
    registry = {
        "concept": [
            {
                "id": "contract",
                "rule": "R7a",
                "symbols": ["Contract"],
                "current_owners": ["owner.py"],
                "minimum_owners": 1,
                "accepted_baseline_violation": False,
            }
        ]
    }
    _findings, errors = integrity._concept_findings(files, registry)
    assert errors == ["R7a contract unregistered owners: ['rogue.py']"]


def test_r7c_catches_public_from_import_without_all() -> None:
    findings = integrity.registered_reexports(
        {"api.py": "from .owner import OBSERVATION_SIZE\n"},
        ["OBSERVATION_SIZE"],
    )
    assert [finding.key for finding in findings] == [
        "R7c|OBSERVATION_SIZE|api.py|reexport_from=.owner:OBSERVATION_SIZE"
    ]


def test_r7c_catches_public_alias_without_all() -> None:
    findings = integrity.registered_reexports(
        {"api.py": "from .owner import OBSERVATION_SIZE as PublicObservationSize\n"},
        ["OBSERVATION_SIZE"],
    )
    assert [finding.key for finding in findings] == [
        "R7c|OBSERVATION_SIZE|api.py|reexport_from=.owner:OBSERVATION_SIZE;binding=PublicObservationSize"
    ]


def test_r7c_catches_reexport_with_literal_all() -> None:
    findings = integrity.registered_reexports(
        {"api.py": 'from .owner import OBSERVATION_SIZE\n__all__ = ["OBSERVATION_SIZE"]\n'},
        ["OBSERVATION_SIZE"],
    )
    assert [finding.key for finding in findings] == [
        "R7c|OBSERVATION_SIZE|api.py|reexport_from=.owner:OBSERVATION_SIZE"
    ]


def test_r7c_registered_shim_passes() -> None:
    files = {
        "owner.py": "class Contract: pass\n",
        "shim.py": "from owner import Contract\n",
    }
    registry = {
        "concept": [
            {
                "id": "contract",
                "rule": "R7a",
                "symbols": ["Contract"],
                "current_owners": ["owner.py"],
                "minimum_owners": 1,
                "accepted_baseline_violation": False,
            }
        ],
        "shim": [{"location": "shim.py", "symbols": ["Contract"]}],
    }
    findings, errors = integrity._concept_findings(files, registry)
    assert findings == []
    assert errors == []


def test_r7c_private_alias_is_not_public_laundering() -> None:
    findings = integrity.registered_reexports(
        {"internal.py": "from .owner import OBSERVATION_SIZE as _ObservationSize\n"},
        ["OBSERVATION_SIZE"],
    )
    assert findings == []


def test_r9_flags_local_untracked_module_but_ignores_external(tmp_path: Path) -> None:
    root = tmp_path / "app"
    root.mkdir()
    importer = root / "main.py"
    importer.write_text("import json\nimport missing_local\n", encoding="utf-8")
    (root / "missing_local.py").write_text("VALUE = 1\n", encoding="utf-8")
    files = {"app/main.py": importer.read_text(encoding="utf-8")}
    edges = integrity.collect_import_edges(tmp_path, files, [root, tmp_path])
    findings = integrity.r9_findings(edges, {"app/main.py"})
    assert len(findings) == 1
    assert findings[0].concept == "missing_local"
    assert "app/missing_local.py" in findings[0].detail


def test_r10_unresolvable_module_fails_without_importing_torch(tmp_path: Path) -> None:
    before = {name for name in integrity.sys.modules if name == "torch" or name.startswith("torch.")}
    assert integrity.find_spec_without_import("definitely_missing_package.module", [tmp_path]) is None
    after = {name for name in integrity.sys.modules if name == "torch" or name.startswith("torch.")}
    assert after == before


def test_r10_repo_local_module_rejects_external_fallback(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    source = repo / "localpkg" / "abi.py"
    source.parent.mkdir(parents=True)
    source.write_text("class ExpectedSymbol: pass\n", encoding="utf-8")
    evidence = repo / "docs" / "migration"
    evidence.mkdir(parents=True)
    (evidence / "CHECKPOINT_INVENTORY.tsv").write_text(
        "policy_class_module\tpolicy_class_qualname\tloadable_under_current_source\n"
        "localpkg.abi\tExpectedSymbol\tyes_module_and_class_present\n",
        encoding="utf-8",
    )
    (evidence / "CHECKPOINT_MODULE_REFERENCES.tsv").write_text(
        "checkpoint_path\tmodule_reference\tqualname_if_found\n"
        "model.zip\tlocalpkg.abi\tExpectedSymbol\n",
        encoding="utf-8",
    )
    external = tmp_path / "site-packages" / "localpkg" / "abi.py"
    external.parent.mkdir(parents=True)
    external.write_text("class ExpectedSymbol: pass\n", encoding="utf-8")

    def external_spec(module: str, _roots):
        return integrity.importlib.machinery.ModuleSpec(module, loader=None, origin=str(external))

    monkeypatch.setattr(integrity, "find_spec_without_import", external_spec)
    result = integrity.r10_result(repo, [repo])
    assert result["module_classification"] == {"localpkg.abi": "repository-local"}
    assert any(error.startswith("repository_local_module_outside_repo:localpkg.abi:") for error in result["failures"])
    assert result["torch_modules_added"] == []


def test_d1_reports_duplicate_and_never_raises(tmp_path: Path) -> None:
    text = "VALUE = " + repr("x" * 220) + "\n"
    (tmp_path / "a.py").write_text(text, encoding="utf-8")
    (tmp_path / "b.py").write_text(text, encoding="utf-8")
    rows = integrity.duplicate_diagnostic(tmp_path, ["b.py", "a.py"])
    assert any(row["kind"] == "exact_sha256" for row in rows)
    assert any(row["kind"] == "ast_similarity" for row in rows)
    assert rows == sorted(rows, key=lambda row: (row["kind"], row["path_a"], row["path_b"], row["evidence"]))


def test_bridge_expiry_is_enforced(tmp_path: Path) -> None:
    (tmp_path / "bridge.py").write_text("# bridge\n", encoding="utf-8")
    payload = '''
<!-- bridge-registry:begin -->
```toml
[[bridge]]
id = "B1"
status = "existing"
reason = "fixture"
locations = ["bridge.py"]
users = ["fixture"]
protecting_rule = "fixture"
removal_gate = "PHASE_1"
live_closure_allowed = false
owner = "fixture"
```
<!-- bridge-registry:end -->
'''
    bridges_path = tmp_path / "docs" / "migration" / "BRIDGES.md"
    bridges_path.parent.mkdir(parents=True, exist_ok=True)
    bridges_path.write_text(payload, encoding="utf-8")
    errors = integrity.bridge_errors(tmp_path, {"shim": []}, current_phase=1)
    assert "Bridge B1 expired at PHASE_1" in errors


def test_phase_7_bridge_allowed_before_and_expired_at_boundary(tmp_path: Path) -> None:
    (tmp_path / "bridge.py").write_text("# bridge\n", encoding="utf-8")
    payload = '''
<!-- bridge-registry:begin -->
```toml
schema_version = 1

[[bridge]]
id = "B1"
status = "existing"
reason = "fixture"
locations = ["bridge.py"]
users = ["fixture"]
protecting_rule = "fixture"
removal_gate = "PHASE_7"
live_closure_allowed = false
owner = "fixture"
```
<!-- bridge-registry:end -->
'''
    bridges_path = tmp_path / "docs" / "migration" / "BRIDGES.md"
    bridges_path.parent.mkdir(parents=True, exist_ok=True)
    bridges_path.write_text(payload, encoding="utf-8")
    before = integrity.bridge_errors(tmp_path, {"current_phase": 6, "shim": []})
    at_boundary = integrity.bridge_errors(tmp_path, {"current_phase": 7, "shim": []})
    assert "Bridge B1 expired at PHASE_7" not in before
    assert "Bridge B1 expired at PHASE_7" in at_boundary


def test_actual_bridge_removal_schedule_is_exact() -> None:
    bridges = integrity._extract_bridge_toml(REPO / "docs" / "migration" / "BRIDGES.md")["bridge"]
    gates = {bridge["id"]: bridge["removal_gate"] for bridge in bridges}
    assert gates == {
        "B1": "PHASE_7",
        "B2": "PHASE_7",
        "B3": "PHASE_8",
        "B4": "NEVER",
    }
    assert integrity.removal_gate_expired(gates["B1"], 7)
    assert integrity.removal_gate_expired(gates["B2"], 7)
    assert not integrity.removal_gate_expired(gates["B3"], 7)
    assert integrity.removal_gate_expired(gates["B3"], 8)
    assert not integrity.removal_gate_expired(gates["B4"], 999)


def test_removed_bridge_cannot_claim_an_installed_location(tmp_path: Path) -> None:
    payload = '''
<!-- bridge-registry:begin -->
```toml
[[bridge]]
id = "B1"
status = "removed"
reason = "fixture"
locations = ["bridge.py"]
users = ["fixture"]
protecting_rule = "fixture"
removal_gate = "PHASE_7"
live_closure_allowed = false
owner = "fixture"
```
<!-- bridge-registry:end -->
'''
    bridges_path = tmp_path / "docs" / "migration" / "BRIDGES.md"
    bridges_path.parent.mkdir(parents=True, exist_ok=True)
    bridges_path.write_text(payload, encoding="utf-8")
    errors = integrity.bridge_errors(tmp_path, {"compatibility_surface": []}, current_phase=7)
    assert "Removed bridge B1 claims installed locations" in errors


def test_non_bridge_retained_shim_registers_reexports() -> None:
    files = {
        "canonical.py": "VALUE = 1\n",
        "compat.py": "from canonical import VALUE\n__all__ = ['VALUE']\n",
    }
    registry = {
        "concept": [
            {
                "id": "value",
                "rule": "R7a",
                "symbols": ["VALUE"],
                "current_owners": ["canonical.py"],
                "minimum_owners": 1,
                "accepted_baseline_violation": False,
            }
        ],
        "shim": [
            {
                "location": "compat.py",
                "symbols": ["VALUE"],
                "canonical_owner": "canonical.py",
                "reason": "fixture",
                "bridge_id": "NONE",
                "removal_gate": "PHASE_12",
            }
        ],
    }
    findings, errors = integrity._concept_findings(files, registry)
    assert findings == []
    assert errors == []


def test_actual_non_bridge_retained_shims_are_accepted_by_bridge_validator() -> None:
    registry = integrity.load_registry(REPO)
    retained = [shim for shim in registry.get("shim", []) if shim["bridge_id"] == "NONE"]
    # 0 shims remain. The 16 TEST_CONTRACT_RETIREMENT-conditioned shims
    # (foreground_vision_bot/farming/*, flyff_farming_recorder/position/*)
    # were retired -- see test_phase12_transitioned_shims_were_retired_
    # not_merely_retagged below -- in the 2026-08-21 repository cleanup,
    # per ADR 0005's own stated retirement condition. The 3 shims that
    # were genuinely permanent at that time (farming/observation.py,
    # simulator/kinodynamic_route_planner.py, simulator/movement_kernel.py)
    # were separately retired in the 2026-08-21 post-migration
    # compatibility purge -- see ADR 0002's Retirement section.
    assert len(retained) == 0
    errors = integrity.bridge_errors(REPO, registry, current_phase=7)
    assert not [error for error in errors if error.startswith("Retained shim")]


def test_phase7_ratchet_relocates_only_exact_one_to_one_move_rows() -> None:
    baseline = integrity.load_baseline(REPO / integrity.DEFAULT_BASELINE)
    moves = integrity.phase7_move_paths(REPO)
    assert len(moves) == 1_486
    assert moves["flyff_farming_simulator/simulator/environment.py"] == "simulator/environment.py"
    assert "foreground_vision_bot/farming/observation.py" not in moves

    relocated = [
        integrity.Finding(
            item["rule"],
            item["concept"],
            moves.get(item["path"], item["path"]),
            item["detail"],
        )
        for items in baseline["violations"].values()
        for item in items
    ]
    frozen_count = sum(len(items) for items in baseline["violations"].values())
    assert len(relocated) == frozen_count
    assert len({finding.key for finding in relocated}) == frozen_count
    errors, resolved = integrity.ratchet_errors(relocated, baseline, path_moves=moves)
    assert errors == []
    assert resolved == []


def test_phase7_test_migration_manifest_conserves_all_160_tests() -> None:
    with (REPO / "docs/migration/PHASE7_TEST_MIGRATION.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 160
    assert {action: sum(row["action"] == action for row in rows) for action in {row["action"] for row in rows}} == {
        "MOVE": 151,
        "MERGE": 2,
        "RETAIN-COMPAT": 7,
    }
    for row in rows:
        assert (REPO / row["destination"]).is_file()
        if row["action"] == "MOVE":
            assert not (REPO / row["old_path"]).exists()
        elif row["action"] == "MERGE":
            assert row["destination"] == "tests/conftest.py"
            assert not (REPO / row["old_path"]).exists()


def test_b3_removed_from_bridge_registry() -> None:
    registry = integrity.load_registry(REPO)
    bridges = integrity._extract_bridge_toml(REPO / integrity.DEFAULT_BRIDGES)["bridge"]
    b3 = next(bridge for bridge in bridges if bridge["id"] == "B3")
    assert b3["status"] == "removed"
    assert b3["locations"] == []
    errors = integrity.bridge_errors(REPO, registry, current_phase=8)
    assert not [error for error in errors if error.startswith("B3")]


def test_b3_bootstrap_pattern_no_longer_present_in_inventory_recordings() -> None:
    """B3's old sys.path bootstrap (_RECORDER_ROOT/_SIMULATOR_ROOT insertion)
    must not have been replaced by an equivalent hidden mechanism."""
    source = (REPO / "devtools/archives/inventory_recordings.py").read_text(encoding="utf-8")
    assert "sys.path.insert" not in source
    assert "_RECORDER_ROOT" not in source
    assert "_SIMULATOR_ROOT" not in source


def test_recorder_movement_classifier_resolves_as_a_normal_repository_import() -> None:
    """Origin test: after B3's removal, devtools.recorder.movement_classification and
    simulator.schema must resolve from this worktree through ordinary,
    unmodified sys.path -- no bootstrap, no sibling/reference-tree fallback."""
    spec = integrity.find_spec_without_import("devtools.recorder.movement_classification", [REPO])
    assert spec is not None and spec.origin is not None
    origin = Path(spec.origin).resolve()
    assert origin.is_relative_to(REPO.resolve())
    assert origin == (REPO / "devtools" / "recorder" / "movement_classification.py").resolve()

    schema_spec = integrity.find_spec_without_import("simulator.schema", [REPO])
    assert schema_spec is not None and schema_spec.origin is not None
    schema_origin = Path(schema_spec.origin).resolve()
    assert schema_origin.is_relative_to(REPO.resolve())
    assert schema_origin == (REPO / "simulator" / "schema.py").resolve()

    legacy_spec = integrity.find_spec_without_import("simulator.legacy_manifest_compat", [REPO])
    assert legacy_spec is not None and legacy_spec.origin is not None
    legacy_origin = Path(legacy_spec.origin).resolve()
    assert legacy_origin.is_relative_to(REPO.resolve())
    assert legacy_origin == (REPO / "simulator" / "legacy_manifest_compat.py").resolve()


def test_b4_permanent_historical_tag_is_mechanically_protected() -> None:
    assert integrity.git_tag_target(REPO, "historical-reproduction-baseline-20260815") == (
        "a90de59232b81753c1b2ea35b8990325c26674e5"
    )


def test_actual_repository_integrity_gate_is_green() -> None:
    payload, errors = integrity.check(REPO)
    assert errors == [], json.dumps(payload, indent=2, sort_keys=True)
    assert payload["r9_violations"] == 0
    assert payload["r10_failures"] == []
    assert payload["torch_modules_added"] == []
    assert payload["r10_module_classification"] == {
        "farming.sb3_training": "repository-local",
        "simulator.split_branch_policy": "repository-local",
        "stable_baselines3.common.policies": "external",
    }


def test_ratchet_accepts_explicit_forward_supplement_but_rejects_unrelated_growth() -> None:
    old = integrity.Finding("R7a", "contract", "old.py", "duplicate_definition=X")
    baseline = {"violations": {"R7a": [old.as_dict()]}}
    supplemental = integrity.Finding(
        "R7c", "SplitSteeringNavigationPolicy", "new_helper.py",
        "reexport_from=simulator.split_branch_policy:SplitSteeringNavigationPolicy",
    )
    unrelated = integrity.Finding(
        "R7c", "SomeOtherSymbol", "unrelated.py", "reexport_from=somewhere:SomeOtherSymbol",
    )

    # A finding explicitly listed in the supplement is accepted, exactly
    # like a frozen-baseline entry -- no rewrite of `baseline` occurred.
    errors, resolved = integrity.ratchet_errors(
        [old, supplemental], baseline, supplement={supplemental.key},
    )
    assert errors == []
    assert resolved == []

    # A DIFFERENT new R7c finding -- same rule, not itself in the
    # supplement -- still ratchets as growth. The supplement grants
    # forward acceptance for specifically enumerated keys only, never a
    # rule-wide or path-wide exemption.
    errors, resolved = integrity.ratchet_errors(
        [old, supplemental, unrelated], baseline, supplement={supplemental.key},
    )
    assert errors == [f"new_baseline_violation:{unrelated.key}"]

    # Omitting the supplement entirely reproduces the pre-supplement
    # behavior exactly: the same finding that was accepted above now
    # ratchets, proving the supplement is additive, not a silent bypass.
    errors, resolved = integrity.ratchet_errors([old, supplemental], baseline)
    assert errors == [f"new_baseline_violation:{supplemental.key}"]


def test_supplement_loading_validates_key_and_matches_finding() -> None:
    rows = [
        {
            "rule": "R7c",
            "concept": "SplitSteeringNavigationPolicy",
            "path": "scratchpad_single_obstacle_train.py",
            "detail": "reexport_from=simulator.split_branch_policy:SplitSteeringNavigationPolicy",
            "key": (
                "R7c|SplitSteeringNavigationPolicy|scratchpad_single_obstacle_train.py"
                "|reexport_from=simulator.split_branch_policy:SplitSteeringNavigationPolicy"
            ),
        }
    ]
    keys = integrity.supplement_keys(rows)
    assert keys == {rows[0]["key"]}

    bad_rows = [dict(rows[0], key="wrong-key")]
    try:
        integrity.supplement_keys(bad_rows)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for a malformed supplement key")


def test_supplement_loading_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert integrity.load_supplement(tmp_path / "does_not_exist.tsv") == []


def test_frozen_phase1_baseline_bytes_are_restored_exact() -> None:
    """The Post-Phase-7 repository-completeness repair (commit 860a990)
    incorrectly hand-edited the frozen baseline directly; that was
    corrected forward (commit follows) by restoring these two files to
    their exact pre-860a990 bytes and moving the 3 newly-visible R7c
    edges into POST_PHASE7_R7C_SUPPLEMENT.tsv instead. This proves the
    frozen baseline is -- and stays -- byte-identical to its state
    immediately before that incorrect edit."""
    frozen_json = integrity._run_git(
        REPO, "show", "966f5fb5c4c06091d55c1161abf80a34ed09b602:docs/migration/BASELINE_VIOLATIONS.json",
    )
    current_json = (REPO / integrity.DEFAULT_BASELINE).read_bytes()
    assert current_json == frozen_json

    frozen_md = integrity._run_git(
        REPO, "show", "966f5fb5c4c06091d55c1161abf80a34ed09b602:docs/migration/BASELINE_VIOLATIONS.md",
    )
    current_md = (REPO / integrity.DEFAULT_BASELINE_MD).read_bytes()
    assert current_md == frozen_md


def test_actual_supplement_covers_exactly_the_three_post_phase7_r7c_edges() -> None:
    rows = integrity.load_supplement(REPO / integrity.DEFAULT_SUPPLEMENT)
    assert len(rows) == 3
    assert integrity.supplement_keys(rows) == {
        "R7c|SplitSteeringNavigationPolicy|simulator/scratchpad/scratchpad_generalized_waypoint_train_reward_ablation.py"
        "|reexport_from=simulator.split_branch_policy:SplitSteeringNavigationPolicy",
        "R7c|SplitSteeringNavigationPolicy|simulator/scratchpad/scratchpad_single_obstacle_train.py"
        "|reexport_from=simulator.split_branch_policy:SplitSteeringNavigationPolicy",
        "R7c|SteeringAction|simulator/scratchpad/scratchpad_single_obstacle_train.py"
        "|reexport_from=farming.actions:SteeringAction",
    }


def _all_supplement_keys() -> set[str]:
    rows = [row for relative in integrity.DEFAULT_SUPPLEMENTS for row in integrity.load_supplement(REPO / relative)]
    return integrity.supplement_keys(rows)


def test_actual_supplement_covers_exactly_the_35_post_phase9_r7c_edges() -> None:
    rows = integrity.load_supplement(REPO / "docs/migration/POST_PHASE9_R7C_SUPPLEMENT.tsv")
    assert len(rows) == 35
    keys = integrity.supplement_keys(rows)
    assert all(key.startswith("R7c|") for key in keys)
    assert all("navigation" in key for key in keys)


def test_actual_repository_integrity_gate_is_green_via_frozen_baseline_plus_supplement() -> None:
    payload, errors = integrity.check(REPO)
    assert errors == [], json.dumps(payload, indent=2, sort_keys=True)
    # 200 (was 205): the 2026-08-21 post-migration compatibility purge's
    # scratchpad-historical-removal batch deleted 20 files
    # (simulator/scratchpad/scratchpad_router_v2_*, scratchpad_router_
    # patch_*, scratchpad_audit_*, scratchpad_diagnose_*,
    # scratchpad_promotion_equivalence_check.py, scratchpad_general_
    # router_episode.py, scratchpad_beginner_navigation_mix_pools.py,
    # scratchpad_legacy_qualified_selector.py, scratchpad_historical_
    # reproduction_guard.py) that carried several of the R7c import
    # edges accepted by docs/migration/POST_PHASE9_R7C_SUPPLEMENT.tsv/
    # POST_PHASE14_R7C_SUPPLEMENT.tsv -- those edges no longer exist to
    # be detected at all now that their source files are gone, so the
    # live-detected total dropped with them (baseline_counts reflects
    # everything currently found in source, frozen-baseline-matched and
    # supplement-matched combined, not merely the frozen baseline file's
    # own row count).
    assert payload["baseline_counts"] == {"R6": 0, "R7a": 0, "R7b": 0, "R7c": 200}
    assert payload["r9_violations"] == 0
    assert payload["r10_failures"] == []
    assert set(payload["supplement_entries_applied"]) == _all_supplement_keys()
    # An unrelated fourth new R7c entry must still fail: verified by
    # constructing one extra finding not present in either the frozen
    # baseline or the supplement, and confirming it alone ratchets.
    bogus = integrity.Finding(
        "R7c", "NotARealSymbol", "not_a_real_file.py", "reexport_from=nowhere:NotARealSymbol",
    )
    result = integrity.collect(REPO, integrity.load_registry(REPO))
    supplement = _all_supplement_keys()
    ratchet, _resolved = integrity.ratchet_errors(
        [*result["findings"], bogus],
        integrity.load_baseline(REPO / integrity.DEFAULT_BASELINE),
        path_moves=integrity.phase7_move_paths(REPO),
        supplement=supplement,
    )
    assert ratchet == [f"new_baseline_violation:{bogus.key}"]


# Phase-12 P12-CORRECTION: bare removal_gate = "NEVER" means "no automatic
# phase-number expiry" (the pre-existing sentinel), not "permanently
# immortal" -- these 16 shims were transitioned off removal_gate =
# "PHASE_12" in P12-A2 onto retirement_condition = "TEST_CONTRACT_
# RETIREMENT" (eligible once the named migration test contract is
# deliberately retired/replaced, not tied to any phase number).
#
# 2026-08-21 repository cleanup: that retirement condition was exercised.
# check_b1/check_b2 (docs/migration/tools/phase4_contracts.py,
# phase5_contracts.py) were rewritten to prove these shims' historical
# purity via the frozen legacy-roots-pre-removal-20260821 git tag instead
# of requiring them to exist live on disk (per ADR 0005's own stated
# retirement condition), so foreground_vision_bot/ and
# flyff_farming_recorder/ -- and their CANONICAL_OWNERS.toml [[shim]]
# entries -- were removed entirely. This test now asserts that retirement
# stuck: none of these 16 locations may reappear in the live registry.
_PHASE12_TRANSITIONED_SHIMS = {
    "foreground_vision_bot/farming/__init__.py",
    "foreground_vision_bot/farming/actions.py",
    "foreground_vision_bot/farming/model_contract.py",
    "foreground_vision_bot/farming/map_masks.py",
    "foreground_vision_bot/farming/reward.py",
    "foreground_vision_bot/farming/session.py",
    "foreground_vision_bot/farming/map_profile.py",
    "foreground_vision_bot/farming/observation.py",
    "foreground_vision_bot/farming/map_features.py",
    "flyff_farming_recorder/position/__init__.py",
    "flyff_farming_recorder/position/attachment_factory.py",
    "flyff_farming_recorder/position/factory.py",
    "flyff_farming_recorder/position/monster_factory.py",
    "flyff_farming_recorder/position/NativeFlyffMonsterProvider.py",
    "flyff_farming_recorder/position/NativeFlyffPositionProvider.py",
    "flyff_farming_recorder/position/native_process_service.py",
}


def test_phase12_transitioned_shims_were_retired_not_merely_retagged() -> None:
    registry = integrity.load_registry(REPO)
    shims = {shim["location"]: shim for shim in registry.get("shim", [])}

    # The 16 TEST_CONTRACT_RETIREMENT-conditioned shims must be gone from
    # the registry, not merely still present-and-tagged -- their backing
    # files (foreground_vision_bot/farming/*.py,
    # flyff_farming_recorder/position/*.py) no longer exist on disk, and a
    # registry entry pointing at a deleted location would itself fail
    # migration_integrity.py's "Retained shim location missing" check.
    assert _PHASE12_TRANSITIONED_SHIMS.isdisjoint(shims.keys()), (
        f"still registered (should have been retired): {_PHASE12_TRANSITIONED_SHIMS & shims.keys()}"
    )
    assert not (REPO / "foreground_vision_bot").exists()
    assert not (REPO / "flyff_farming_recorder").exists()

    # No shim anywhere still claims the expired PHASE_12 gate, and no shim
    # anywhere still claims the now-exercised TEST_CONTRACT_RETIREMENT
    # condition (its only prior claimants were the 16 retired shims above).
    assert [s["location"] for s in registry.get("shim", []) if s.get("removal_gate") == "PHASE_12"] == []
    assert [
        s["location"] for s in registry.get("shim", []) if s.get("retirement_condition") == "TEST_CONTRACT_RETIREMENT"
    ] == []

    # The 3 shims that were genuinely permanent as of this Phase-12
    # finding (unaffected by ITS retirement) were later also retired,
    # separately, in the 2026-08-21 post-migration compatibility purge --
    # not because a phase number advanced (this test's own point: that is
    # never sufficient), but because a static pickle disassembly of the
    # actual frozen checkpoint proved two of the three were never
    # checkpoint-load-bearing in the first place, and the third's
    # accidental re-export was separable from its real canonical
    # implementation. See
    # docs/decisions/0002-preserve-abi-compatibility-shims.md's
    # Retirement section for the full evidence trail. The registry's
    # [[shim]] table is now empty.
    assert registry.get("shim", []) == []

    # The registry shrinking does not upset the ruler's own bridge/shim
    # validation.
    assert integrity.bridge_errors(REPO, registry) == []
