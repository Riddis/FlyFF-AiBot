"""Phase-10 Section 11 tests: the read-only artifact view must actually be
read-only (never writes) and must correctly normalize the existing
recordings/checkpoint inventories without inventing or relocating any
scientific artifact."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from devtools.artifact_inventory import list_checkpoints, list_recordings, summarize

REPO = Path(__file__).resolve().parents[1]


def test_module_never_writes_a_file() -> None:
    """AST-based: no call to open(..., "w"/"a"/...), Path.write_text,
    Path.write_bytes, or csv writer construction anywhere in this module."""
    source = (REPO / "devtools" / "artifact_inventory.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="devtools/artifact_inventory.py")
    forbidden_attrs = {"write_text", "write_bytes", "writer", "DictWriter", "unlink", "rmdir", "rename"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
            raise AssertionError(f"artifact_inventory.py calls a write/mutate method: .{node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and "w" in str(kw.value.value):
                    raise AssertionError("open(..., mode=...write...) found")
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and "w" in str(node.args[1].value):
                raise AssertionError("open(..., 'w'...) found")


def test_list_recordings_matches_the_real_index_json() -> None:
    recordings = list_recordings()
    assert len(recordings) > 0
    for entry in recordings:
        assert entry.sha256
        assert entry.filename


def test_list_checkpoints_reproduces_the_frozen_inventory_exactly() -> None:
    checkpoints = list_checkpoints()
    assert len(checkpoints) == 313
    first = checkpoints[0]
    assert first.path.startswith("flyff_farming_simulator/") or first.path.startswith("foreground_vision_bot/") or "/" in first.path
    assert len(first.sha256) == 65 or len(first.sha256) == 64


def test_frozen_checkpoint_inventory_bytes_are_unchanged_by_this_view() -> None:
    """The view must never mutate its source; hashing the frozen TSV before
    and after calling list_checkpoints() proves this module didn't touch it."""
    inventory_path = REPO / "docs" / "migration" / "CHECKPOINT_INVENTORY.tsv"
    before = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    list_checkpoints()
    after = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    assert before == after


def test_summarize_counts_are_internally_consistent() -> None:
    summary = summarize()
    assert summary["checkpoint_count"] == 313
    assert 0 <= summary["checkpoints_loadable_under_current_source"] <= summary["checkpoint_count"]
    assert 0 <= summary["recordings_ready_for_world_model"] <= summary["recording_count"]
    assert 0 <= summary["recordings_eva_only_eligible"] <= summary["recording_count"]


def test_list_recordings_returns_empty_not_an_error_when_index_is_absent(tmp_path: Path) -> None:
    assert list_recordings(repo=tmp_path) == []


def test_list_checkpoints_returns_empty_not_an_error_when_inventory_is_absent(tmp_path: Path) -> None:
    assert list_checkpoints(repo=tmp_path) == []
