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
    (tmp_path / "BRIDGES.md").write_text(payload, encoding="utf-8")
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
    (tmp_path / "BRIDGES.md").write_text(payload, encoding="utf-8")
    before = integrity.bridge_errors(tmp_path, {"current_phase": 6, "shim": []})
    at_boundary = integrity.bridge_errors(tmp_path, {"current_phase": 7, "shim": []})
    assert "Bridge B1 expired at PHASE_7" not in before
    assert "Bridge B1 expired at PHASE_7" in at_boundary


def test_actual_bridge_removal_schedule_is_exact() -> None:
    bridges = integrity._extract_bridge_toml(REPO / "BRIDGES.md")["bridge"]
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
    (tmp_path / "BRIDGES.md").write_text(payload, encoding="utf-8")
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
    retained = [shim for shim in registry["shim"] if shim["bridge_id"] == "NONE"]
    assert len(retained) == 17
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


def test_b3_source_evidence_includes_registered_module_and_symbol() -> None:
    source = (REPO / "tools/inventory_recordings.py").read_text(encoding="utf-8")
    assert integrity._b3_source_matches(
        source,
        "recorder.movement_classification",
        "MovementControlClassifier",
    )
    assert not integrity._b3_source_matches(
        source,
        "recorder.movement_classification",
        "DifferentClassifier",
    )


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
