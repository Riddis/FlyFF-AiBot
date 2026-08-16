from __future__ import annotations

import importlib.util
import json
import sys
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


def test_r7c_catches_unregistered_reexport() -> None:
    findings = integrity.registered_reexports(
        {"api.py": 'from .owner import OBSERVATION_SIZE\n__all__ = ["OBSERVATION_SIZE"]\n'},
        ["OBSERVATION_SIZE"],
    )
    assert [finding.key for finding in findings] == [
        "R7c|OBSERVATION_SIZE|api.py|reexport_from=.owner:OBSERVATION_SIZE"
    ]


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


def test_actual_repository_integrity_gate_is_green() -> None:
    payload, errors = integrity.check(REPO)
    assert errors == [], json.dumps(payload, indent=2, sort_keys=True)
    assert payload["r9_violations"] == 0
    assert payload["r10_failures"] == []
    assert payload["torch_modules_added"] == []
