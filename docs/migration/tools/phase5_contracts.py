"""Deterministic Phase-5 position consolidation gates."""

from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_DEFAULT = Path(__file__).resolve().parents[3]
PHASE4_ENTRY = "210e4e91a1cce8f6f7db56b8f4b77f4522f56d73"
LEGACY_ROOTS_TAG = "legacy-roots-pre-removal-20260821"  # last commit with foreground_vision_bot/flyff_farming_recorder facades, see ADR 0005
BOT_POSITION = "foreground_vision_bot/position"
RECORDER_POSITION = "flyff_farming_recorder/position"
CANONICAL_POSITION = "position"
PRESENCE_KEYS = (
    "presence_clear_confirmation_samples",
    "presence_cold_poll_batch_size",
    "presence_cold_verification_batch_size",
    "presence_dead_read_grace_seconds",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _top_level_bindings(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _historical_bindings(repo: Path) -> dict[str, set[str]]:
    old_paths = _git(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        PHASE4_ENTRY,
        BOT_POSITION,
        RECORDER_POSITION,
    ).splitlines()
    old_modules: dict[str, set[str]] = {}
    for relative in old_paths:
        if not relative.endswith(".py"):
            continue
        module = Path(relative).name
        source = _git(repo, "show", f"{PHASE4_ENTRY}:{relative}")
        old_modules.setdefault(module, set()).update(_top_level_bindings(source))
    return old_modules


def check_g1(repo: Path) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    old_modules = _historical_bindings(repo)
    missing: dict[str, list[str]] = {}
    for module, expected in sorted(old_modules.items()):
        current = repo / CANONICAL_POSITION / module
        if not current.is_file():
            missing[module] = sorted(expected)
            continue
        absent = expected - _top_level_bindings(current.read_text(encoding="utf-8-sig"))
        if absent:
            missing[module] = sorted(absent)
    if missing:
        failures.append(f"G1 canonical public API misses historical union: {missing}")
    return failures, {
        "historical_module_count": len(old_modules),
        "missing": missing,
        "canonical_owner": CANONICAL_POSITION,
    }


def check_np(repo: Path) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    layer1 = repo / CANONICAL_POSITION
    profiling_importers: list[str] = []
    for path in sorted(layer1.glob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        if "position.profiling" in source or "from .profiling" in source:
            profiling_importers.append(path.name)
    if profiling_importers:
        failures.append(f"NP2/NP4 Layer 1 imports profiling: {profiling_importers}")

    trace_source = (layer1 / "NativeTraceTargets.py").read_text(encoding="utf-8")
    service_source = (layer1 / "native_process_service.py").read_text(encoding="utf-8")
    reader_source = (layer1 / "IndependentNativeReader.py").read_text(encoding="utf-8")
    if "attach_policy.player_discrimination" not in trace_source:
        failures.append("NP3 trace discrimination is not driven by passed policy")
    if "self.attach_policy.activate_presence_sampling_on_attach" not in service_source:
        failures.append("NP3 attach-time sampling is not driven by passed policy")
    if "promote_validated_presence_offset" in reader_source:
        failures.append("NP5 longitudinal promotion remains in Layer 1 reader")
    if "def install_validated_presence_offset" not in reader_source:
        failures.append("NP5 narrow reader install API is missing")
    promotion = layer1 / "profiling" / "presence_promotion.py"
    if not promotion.is_file():
        failures.append("NP5 profiling/presence_promotion.py is missing")

    probe = r'''
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
import position
import position.attachment_factory
print(json.dumps({
    "origin": str(pathlib.Path(position.__file__).resolve()),
    "profiling": sorted(name for name in sys.modules if name.startswith("position.profiling")),
}))
'''
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(repo)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    live = json.loads(result.stdout)
    if live["profiling"]:
        failures.append(f"NP4 live closure imported profiling: {live['profiling']}")
    return failures, {
        "layer1_profiling_importers": profiling_importers,
        "live_import": live,
    }


def check_g9(repo: Path) -> tuple[list[str], dict[str, Any]]:
    """G9 effective-configuration parity. The bot-side comparisons
    (bot_monster/position) are current product invariants, read live
    from canonical position/. The recorder-side comparisons
    (rec_monster/recorder) compare against flyff_farming_recorder/'s
    now-deleted historical config -- proven by extracting its frozen
    content at LEGACY_ROOTS_TAG (recorder_config.json specifically at PHASE4_ENTRY, predating it) to temp files and running it through the
    real loader, rather than requiring the file to exist in current
    HEAD (see ADR 0005's TEST_CONTRACT_RETIREMENT condition)."""
    failures: list[str] = []
    baseline = json.loads(
        (repo / "docs/migration/EFFECTIVE_CONFIG_BASELINE.json").read_text(
            encoding="utf-8"
        )
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rec_monster_path = tmp_path / "rec_native_monsters.json"
        rec_monster_path.write_text(
            _git(repo, "show", f"{LEGACY_ROOTS_TAG}:{RECORDER_POSITION}/native_monsters.json"),
            encoding="utf-8",
        )
        rec_config_path = tmp_path / "rec_recorder_config.json"
        rec_config_path.write_text(
            # recorder_config.json never existed under flyff_farming_recorder/
            # at LEGACY_ROOTS_TAG (only position/ did) -- this specific
            # historical file predates that, at PHASE4_ENTRY.
            _git(repo, "show", f"{PHASE4_ENTRY}:flyff_farming_recorder/recorder_config.json"),
            encoding="utf-8",
        )
        probe = r'''
import dataclasses, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rec_monster_path = pathlib.Path(sys.argv[2])
rec_config_path = pathlib.Path(sys.argv[3])
sys.path.insert(0, str(root))
from position.MonsterConfig import load_native_monster_config
from position.PositionConfig import load_native_position_config
from devtools.recorder.config import RecorderConfig
bot_monster = dataclasses.asdict(load_native_monster_config(root / "position/native_monsters.json"))
rec_monster = dataclasses.asdict(load_native_monster_config(rec_monster_path))
recorder = dataclasses.asdict(RecorderConfig.load(rec_config_path))
position = dataclasses.asdict(load_native_position_config(root / "position/native_position.json"))
print(json.dumps({"bot_monster": bot_monster, "rec_monster": rec_monster, "recorder": recorder, "position": position}, sort_keys=True))
'''
        result = subprocess.run(
            [sys.executable, "-I", "-c", probe, str(repo), str(rec_monster_path), str(rec_config_path)],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
    actual = json.loads(result.stdout)
    components = baseline["components"]
    expected_bot = components["foreground_vision_bot"]["monster_config"]["effective"]
    expected_rec = components["flyff_farming_recorder"]["monster_config"]["effective"]
    expected_recorder = components["flyff_farming_recorder"]["recorder_config"]["effective"]
    expected_position = components["foreground_vision_bot"]["position_config"]["effective"]
    rec_without_live = {
        key: value for key, value in actual["rec_monster"].items() if key not in PRESENCE_KEYS
    }
    comparisons = {
        "live_monster": actual["bot_monster"] == expected_bot,
        "recording_monster": rec_without_live == expected_rec,
        "recording_config": actual["recorder"] == expected_recorder,
        "position": actual["position"] == expected_position,
        "presence_values": {
            key: actual["bot_monster"][key] == actual["recorder"][key]
            for key in PRESENCE_KEYS
        },
    }
    if not all(value for key, value in comparisons.items() if key != "presence_values"):
        failures.append(f"G9 effective configuration mismatch: {comparisons}")
    if not all(comparisons["presence_values"].values()):
        failures.append(f"G9 presence value mismatch: {comparisons['presence_values']}")
    recorder_json = json.loads(
        _git(repo, "show", f"{LEGACY_ROOTS_TAG}:{RECORDER_POSITION}/native_monsters.json")
    )
    if "presence_sampling" in recorder_json:
        failures.append("G9 recorder position JSON incorrectly owns presence settings")
    return failures, comparisons


def check_b2(repo: Path) -> tuple[list[str], dict[str, Any]]:
    """Prove B2: canonical position/ owns the live implementation (a
    current product invariant, checked live), and the retired
    flyff_farming_recorder/position/ facade was a pure historical
    re-export at LEGACY_ROOTS_TAG (a historical fact,
    checked via git show -- the 2026-08-21 repository cleanup deleted
    these facade files from current HEAD; see ADR 0005's
    TEST_CONTRACT_RETIREMENT condition). The facade's identity-re-export
    property is proven by construction: zero behavioral statements +
    exact import parity with its historical bindings means `from
    position.X import Y` mechanically binds the same object Python
    itself would resolve `position.X.Y` to -- no live import of the
    now-deleted package is needed to prove that."""
    failures: list[str] = []
    manifest_path = repo / "docs/migration/PHASE5_B2_SHIM_MANIFEST.tsv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(
                (line for line in handle if not line.startswith("#")),
                delimiter="\t",
            )
            if row.get("path")
        ]
    manifest_paths = {row["path"] for row in rows}
    historical_paths = set(
        _git(
            repo, "ls-tree", "-r", "--name-only", LEGACY_ROOTS_TAG, RECORDER_POSITION
        ).splitlines()
    )
    historical_paths = {path for path in historical_paths if path.endswith(".py")}
    if len(rows) != 23 or manifest_paths != historical_paths:
        failures.append(
            f"B2 manifest/historical-files differ: rows={len(rows)}, "
            f"missing={sorted(historical_paths - manifest_paths)}, "
            f"extra={sorted(manifest_paths - historical_paths)}"
        )
    historical = _historical_bindings(repo)
    shim_evidence: dict[str, Any] = {}
    for relative in sorted(historical_paths):
        source = _git(repo, "show", f"{LEGACY_ROOTS_TAG}:{relative}")
        tree = ast.parse(source, filename=relative)
        behavior = [
            type(node).__name__
            for node in tree.body
            if not isinstance(node, (ast.Expr, ast.ImportFrom))
        ]
        imported = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        module_name = Path(relative).name
        missing_bindings = historical[module_name] - imported
        if "BRIDGE B2" in source:
            failures.append(f"B2 bootstrap marker remains: {relative}")
        if behavior:
            failures.append(f"B2 behavioral statements in {relative}: {behavior}")
        if missing_bindings:
            failures.append(
                f"B2 historical bindings missing in {relative}: {sorted(missing_bindings)}"
            )
        shim_evidence[relative] = {
            "behavioral_statements": behavior,
            "historical_bindings": len(historical[module_name]),
            "missing_bindings": sorted(missing_bindings),
        }
    probe = r'''
import importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
names = ["position", "position.IndependentNativeReader", "position.NativeTraceTargets", "position.native_process_service", "position.RecoveredNativeProfile"]
origins = {name: str(pathlib.Path(importlib.import_module(name).__file__).resolve()) for name in names}
print(json.dumps({"origins": origins}, sort_keys=True))
'''
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(repo)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    origins = payload["origins"]
    owner = (repo / CANONICAL_POSITION).resolve()
    wrong = {
        name: origin
        for name, origin in origins.items()
        if not Path(origin).resolve().is_relative_to(owner)
    }
    if wrong:
        failures.append(f"B2 noncanonical origins: {wrong}")
    return failures, {
        "origins": origins,
        "manifest_rows": len(rows),
        "shims": shim_evidence,
    }


def check_all(repo: Path) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    evidence: dict[str, Any] = {}
    for name, gate in (("G1", check_g1), ("NP", check_np), ("G9", check_g9), ("B2", check_b2)):
        gate_failures, gate_evidence = gate(repo)
        failures.extend(gate_failures)
        evidence[name] = gate_evidence
    return failures, evidence


if __name__ == "__main__":
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO_DEFAULT
    problems, details = check_all(repo)
    print(json.dumps({"ok": not problems, "failures": problems, "evidence": details}, indent=2, sort_keys=True))
    raise SystemExit(bool(problems))
