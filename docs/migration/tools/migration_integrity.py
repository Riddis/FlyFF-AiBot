"""Phase-1 repository ownership, import, bridge, and checkpoint integrity rules.

This tool is deliberately standalone. It inspects source and migration evidence
without importing project modules, Torch, SB3, or product runtimes.
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import importlib.machinery
import importlib.util
import itertools
import json
import re
import subprocess
import sys
import tokenize
import types
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import tomllib


SCHEMA_VERSION = 1
BASE_SHA = "dc734bb82a4d6c99deb7dd1251c4f7c3f0c99e34"
DEFAULT_OWNERS = "CANONICAL_OWNERS.toml"
DEFAULT_BRIDGES = "BRIDGES.md"
DEFAULT_BASELINE = "docs/migration/BASELINE_VIOLATIONS.json"
DEFAULT_BASELINE_MD = "docs/migration/BASELINE_VIOLATIONS.md"
DEFAULT_D1 = "docs/migration/DUPLICATE_CONTENT_REPORT.tsv"
DEFAULT_SUPPLEMENT = "docs/migration/POST_PHASE7_R7C_SUPPLEMENT.tsv"
# Each later phase that legitimately grows R7c (through ownership/path
# translation, never new coupling) gets its own separate, phase-labeled
# supplement file rather than appending to an earlier phase's -- keeps each
# phase's forward evidence independently attributable and reviewable.
DEFAULT_SUPPLEMENTS = (
    DEFAULT_SUPPLEMENT,
    "docs/migration/POST_PHASE9_R7C_SUPPLEMENT.tsv",
    "docs/migration/POST_PHASE10_R7C_SUPPLEMENT.tsv",
    "docs/migration/POST_PHASE14_R7C_SUPPLEMENT.tsv",
)
PHASE7_MOVE_MANIFEST = "docs/migration/PHASE7_MOVE_MANIFEST.tsv"
BRIDGE_BEGIN = "<!-- bridge-registry:begin -->"
BRIDGE_END = "<!-- bridge-registry:end -->"
_MISSING = object()


@dataclass(frozen=True, order=True)
class Finding:
    rule: str
    concept: str
    path: str
    detail: str

    @property
    def key(self) -> str:
        return "|".join((self.rule, self.concept, self.path, self.detail))

    def as_dict(self) -> dict[str, str]:
        return {
            "rule": self.rule,
            "concept": self.concept,
            "path": self.path,
            "detail": self.detail,
            "key": self.key,
        }


@dataclass(frozen=True, order=True)
class ImportEdge:
    importer: str
    imported_name: str
    level: int
    resolved_path: str | None
    dynamic: bool = False


def _run_git(repo: Path, *args: str) -> bytes:
    repo = repo.resolve()
    safe = ["-c", f"safe.directory={repo.as_posix()}"]
    dot_git = repo / ".git"
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text.split(":", 1)[1].strip()).resolve()
            marker = f"{Path('.git').as_posix()}/worktrees/"
            normalized = gitdir.as_posix()
            if marker in normalized:
                main = Path(normalized.split(marker, 1)[0].rstrip("/"))
                safe.extend(("-c", f"safe.directory={main.as_posix()}"))
    return subprocess.check_output(["git", *safe, "-C", str(repo), *args])


def tracked_paths(repo: Path) -> list[str]:
    raw = _run_git(repo, "ls-files", "-z")
    return sorted(part.decode("utf-8", "surrogateescape") for part in raw.split(b"\0") if part)


def tracked_python_paths(repo: Path) -> list[str]:
    return [path for path in tracked_paths(repo) if path.endswith(".py")]


def read_python(path: Path) -> str:
    with tokenize.open(path) as handle:
        return handle.read()


def parse_python(path: Path) -> ast.Module:
    return ast.parse(read_python(path), filename=str(path))


def top_level_definitions(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def definition_owners(files: dict[str, str], symbols: Iterable[str]) -> dict[str, list[str]]:
    wanted = set(symbols)
    result: dict[str, list[str]] = {symbol: [] for symbol in sorted(wanted)}
    for path, source in sorted(files.items()):
        defined = top_level_definitions(ast.parse(source, filename=path))
        for symbol in sorted(wanted & defined):
            result[symbol].append(path)
    return result


def _literal_all(tree: ast.Module) -> set[str]:
    exported: set[str] = set()
    for node in tree.body:
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            value = node.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            exported.update(
                element.value
                for element in value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
    return exported


def registered_reexports(files: dict[str, str], symbols: Iterable[str]) -> list[Finding]:
    wanted = set(symbols)
    findings: list[Finding] = []
    for path, source in sorted(files.items()):
        tree = ast.parse(source, filename=path)
        exported = _literal_all(tree)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                origin = "." * node.level + (node.module or "")
                for alias in node.names:
                    binding = alias.asname or alias.name
                    if alias.name not in wanted:
                        continue
                    if binding.startswith("_") and binding not in exported:
                        continue
                    alias_evidence = "" if binding == alias.name else f";binding={binding}"
                    findings.append(
                        Finding("R7c", alias.name, path, f"reexport_from={origin}:{alias.name}{alias_evidence}")
                    )
    return findings


def load_registry(repo: Path, relative: str = DEFAULT_OWNERS) -> dict[str, Any]:
    with (repo / relative).open("rb") as handle:
        registry = tomllib.load(handle)
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported owner registry schema: {registry.get('schema_version')!r}")
    return registry


def active_phase(registry: dict[str, Any]) -> int:
    phase = registry.get("current_phase")
    if not isinstance(phase, int) or phase < 1:
        raise ValueError(f"Invalid current_phase in {DEFAULT_OWNERS}: {phase!r}")
    return phase


def python_roots(repo: Path, registry: dict[str, Any]) -> list[Path]:
    return [(repo / path).resolve() for path in registry["repository"]["python_roots"]]


def _tracked_sources(repo: Path) -> tuple[dict[str, str], list[Finding]]:
    files: dict[str, str] = {}
    errors: list[Finding] = []
    for relative in tracked_python_paths(repo):
        path = repo / relative
        try:
            source = read_python(path)
            ast.parse(source, filename=relative)
        except (OSError, SyntaxError, UnicodeError) as error:
            errors.append(Finding("R9", "parse_error", relative, f"{type(error).__name__}:{error}"))
            continue
        files[relative] = source
    return files, errors


def _root_for_importer(importer: str, roots: Sequence[Path], repo: Path) -> Path:
    absolute = (repo / importer).resolve()
    matches = [root for root in roots if absolute.is_relative_to(root)]
    return max(matches, key=lambda path: len(path.parts), default=repo.resolve())


def _module_candidates(module: str, roots: Sequence[Path]) -> Iterator[Path]:
    parts = module.split(".") if module else []
    for root in roots:
        stem = root.joinpath(*parts)
        yield stem.with_suffix(".py")
        yield stem / "__init__.py"


def resolve_local_import(
    repo: Path,
    importer: str,
    imported_name: str,
    level: int,
    roots: Sequence[Path],
) -> str | None:
    importer_root = _root_for_importer(importer, roots, repo)
    search_roots = [importer_root, *[root for root in roots if root != importer_root]]
    module = imported_name
    if level:
        importer_path = (repo / importer).resolve()
        try:
            package_parts = list(importer_path.parent.relative_to(importer_root).parts)
        except ValueError:
            package_parts = []
        remove = max(level - 1, 0)
        if remove > len(package_parts):
            return None
        package_parts = package_parts[: len(package_parts) - remove]
        if module:
            package_parts.extend(module.split("."))
        module = ".".join(package_parts)
        search_roots = [importer_root]
    for candidate in _module_candidates(module, search_roots):
        if candidate.is_file() and candidate.resolve().is_relative_to(repo.resolve()):
            return candidate.resolve().relative_to(repo.resolve()).as_posix()
    return None


def import_requests(tree: ast.Module) -> list[tuple[str, int, bool]]:
    requests: list[tuple[str, int, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            requests.extend((alias.name, 0, False) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            requests.append((base, node.level, False))
            requests.extend(
                (f"{base}.{alias.name}" if base else alias.name, node.level, False)
                for alias in node.names
                if alias.name != "*"
            )
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            name: str | None = None
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                name = node.args[0].value
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                name = node.args[0].value
            if name:
                level = len(name) - len(name.lstrip("."))
                requests.append((name.lstrip("."), level, True))
    return sorted(set(requests))


def collect_import_edges(repo: Path, files: dict[str, str], roots: Sequence[Path]) -> list[ImportEdge]:
    edges: list[ImportEdge] = []
    for importer, source in sorted(files.items()):
        for imported, level, dynamic in import_requests(ast.parse(source, filename=importer)):
            resolved = resolve_local_import(repo, importer, imported, level, roots)
            edges.append(ImportEdge(importer, "." * level + imported, level, resolved, dynamic))
    return sorted(set(edges))


def r9_findings(edges: Iterable[ImportEdge], tracked: set[str]) -> list[Finding]:
    return sorted(
        Finding(
            "R9",
            edge.imported_name,
            edge.importer,
            f"resolved_untracked={edge.resolved_path};dynamic={str(edge.dynamic).lower()}",
        )
        for edge in edges
        if edge.resolved_path is not None and edge.resolved_path not in tracked
    )


def r7b_findings(edges: Iterable[ImportEdge], registry: dict[str, Any]) -> list[Finding]:
    rule = registry["rules"]["R7b"]
    legacy_segments = set(rule["legacy_path_segments"])
    allowed = tuple(rule["allowed_importer_prefixes"])
    findings: list[Finding] = []
    for edge in edges:
        if edge.resolved_path is None:
            continue
        if legacy_segments & set(Path(edge.resolved_path).parts) and not edge.importer.startswith(allowed):
            findings.append(Finding("R7b", edge.imported_name, edge.importer, f"legacy_target={edge.resolved_path}"))
    return sorted(findings)


def _concept_findings(files: dict[str, str], registry: dict[str, Any]) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []
    errors: list[str] = []
    shims = registry.get("shim", [])
    allowed_reexports = {(shim["location"], symbol) for shim in shims for symbol in shim["symbols"]}
    all_symbols: set[str] = set()
    for concept in registry["concept"]:
        rule = concept["rule"]
        symbols = concept.get("symbols", [])
        all_symbols.update(symbols)
        owners = definition_owners(files, symbols)
        detected = sorted({path for paths in owners.values() for path in paths})
        registered = set(concept["current_owners"])
        new = sorted(set(detected) - registered)
        if new:
            errors.append(f"{rule} {concept['id']} unregistered owners: {new}")
        minimum = int(concept.get("minimum_owners", 1))
        if len(detected) < minimum:
            errors.append(f"{rule} {concept['id']} has {len(detected)} owners; minimum is {minimum}")
        if concept.get("accepted_baseline_violation", False) and len(detected) > 1:
            for symbol, paths in sorted(owners.items()):
                if len(paths) > 1:
                    findings.extend(Finding(rule, concept["id"], path, f"duplicate_definition={symbol}") for path in paths)
    for finding in registered_reexports(files, all_symbols):
        if (finding.path, finding.concept) not in allowed_reexports:
            findings.append(finding)
    return sorted(set(findings)), errors


def _extract_bridge_toml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if BRIDGE_BEGIN not in text or BRIDGE_END not in text:
        raise ValueError("BRIDGES.md lacks machine-readable bridge registry markers")
    payload = text.split(BRIDGE_BEGIN, 1)[1].split(BRIDGE_END, 1)[0]
    payload = payload.replace("```toml", "", 1).rsplit("```", 1)[0].strip()
    return tomllib.loads(payload)


def removal_gate_expired(removal_gate: str, current_phase: int) -> bool:
    match = re.fullmatch(r"PHASE_(\d+)", removal_gate)
    return bool(match and current_phase >= int(match.group(1)))


def git_tag_target(repo: Path, tag: str) -> str | None:
    try:
        return _run_git(repo, "rev-parse", f"refs/tags/{tag}^{{}}").decode().strip()
    except subprocess.CalledProcessError:
        return None


def _b3_source_matches(source: str, target_module: str, target_symbol: str) -> bool:
    tree = ast.parse(source)
    imports_target = any(
        isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == target_module
        and any(alias.name == target_symbol for alias in node.names)
        for node in tree.body
    )
    defines_root = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_RECORDER_ROOT" for target in node.targets)
        for node in tree.body
    )
    inserts_root = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "insert"
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "sys"
        and node.func.value.attr == "path"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 0
        and isinstance(node.args[1], ast.Call)
        and isinstance(node.args[1].func, ast.Name)
        and node.args[1].func.id == "str"
        and len(node.args[1].args) == 1
        and isinstance(node.args[1].args[0], ast.Name)
        and node.args[1].args[0].id == "_RECORDER_ROOT"
        for node in ast.walk(tree)
    )
    return imports_target and defines_root and inserts_root


def bridge_errors(repo: Path, registry: dict[str, Any], current_phase: int | None = None) -> list[str]:
    errors: list[str] = []
    current_phase = active_phase(registry) if current_phase is None else current_phase
    bridges = _extract_bridge_toml(repo / DEFAULT_BRIDGES).get("bridge", [])
    ids = {bridge["id"] for bridge in bridges}
    mandatory = {"B1", "B2", "B3", "B4"}
    if not mandatory.issubset(ids):
        errors.append(f"Missing mandatory bridge IDs: {sorted(mandatory - ids)}")
    required = {"id", "status", "reason", "locations", "users", "protecting_rule", "removal_gate", "live_closure_allowed", "owner"}
    for bridge in bridges:
        missing = required - bridge.keys()
        if missing:
            errors.append(f"Bridge {bridge.get('id', '?')} missing fields: {sorted(missing)}")
            continue
        if bridge["status"] in {"future", "existing"} and removal_gate_expired(
            str(bridge["removal_gate"]), current_phase
        ):
            errors.append(f"Bridge {bridge['id']} expired at {bridge['removal_gate']}")
        if bridge["status"] == "future" and bridge["locations"]:
            errors.append(f"Future bridge {bridge['id']} claims installed locations")
        if bridge["status"] == "removed" and bridge["locations"]:
            errors.append(f"Removed bridge {bridge['id']} claims installed locations")
        for location in bridge["locations"]:
            if bridge["status"] == "existing" and not (repo / location).exists():
                errors.append(f"Bridge {bridge['id']} location missing: {location}")
    for shim in registry.get("shim", []):
        required_shim = {"location", "symbols", "canonical_owner", "reason", "bridge_id", "removal_gate"}
        missing = required_shim - shim.keys()
        if missing:
            errors.append(f"Shim {shim.get('location', '?')} missing fields: {sorted(missing)}")
        if shim.get("bridge_id") not in ids | {"NONE"}:
            errors.append(f"Shim {shim.get('location', '?')} references unknown bridge {shim.get('bridge_id')}")
        if shim.get("bridge_id") == "NONE":
            location = str(shim.get("location", "?"))
            if removal_gate_expired(str(shim.get("removal_gate", "")), current_phase):
                errors.append(f"Retained shim {location} expired at {shim.get('removal_gate')}")
            if not (repo / location).is_file():
                errors.append(f"Retained shim location missing: {location}")
            owner = str(shim.get("canonical_owner", ""))
            if not owner or not (repo / owner).is_file():
                errors.append(f"Retained shim canonical owner missing: {location} -> {owner}")
    b1 = next((bridge for bridge in bridges if bridge["id"] == "B1"), None)
    if 4 <= current_phase < 7 and (b1 is None or b1.get("status") != "existing"):
        errors.append("B1 must be installed while Phase 4 is active")
    if b1 and b1.get("status") == "existing":
        marker = "# BRIDGE B1 — removed in Phase 7"
        locations = set(b1["locations"])
        b1_shims = [shim for shim in registry.get("shim", []) if shim.get("bridge_id") == "B1"]
        for shim in b1_shims:
            location = shim.get("location", "")
            if location not in locations:
                errors.append(f"B1 shim is absent from bridge locations: {location}")
            if shim.get("removal_gate") != b1.get("removal_gate"):
                errors.append(f"B1 shim removal gate differs from bridge: {location}")
            owner = shim.get("canonical_owner", "")
            if not owner or not (repo / owner).is_file():
                errors.append(f"B1 shim canonical owner missing: {location} -> {owner}")
        for location in sorted(locations):
            path = repo / location
            if path.suffix not in {".py", ".spec"}:
                continue
            source = path.read_text(encoding="utf-8") if path.is_file() else ""
            if marker not in source:
                errors.append(f"B1 source marker missing: {location}")
    b3 = next((bridge for bridge in bridges if bridge["id"] == "B3"), None)
    if b3 and b3["status"] == "existing":
        required_b3 = {"target_module", "target_symbol"}
        missing = required_b3 - b3.keys()
        if missing:
            errors.append(f"Bridge B3 missing source-evidence fields: {sorted(missing)}")
        source_path = repo / "tools/inventory_recordings.py"
        source = source_path.read_text(encoding="utf-8") if source_path.is_file() else ""
        if missing or not _b3_source_matches(source, b3.get("target_module", ""), b3.get("target_symbol", "")):
            errors.append("B3 source evidence no longer matches inventory_recordings.py")
    b4 = next((bridge for bridge in bridges if bridge["id"] == "B4"), None)
    if b4 and b4["status"] == "permanent-historical":
        expected = b4.get("expected_target")
        tag_locations = [location for location in b4["locations"] if location.startswith("git-tag:")]
        if not isinstance(expected, str) or len(tag_locations) != 1:
            errors.append("B4 permanent tag evidence is incomplete")
        else:
            tag = tag_locations[0].split(":", 1)[1]
            actual = git_tag_target(repo, tag)
            if actual != expected:
                errors.append(f"B4 tag target mismatch: {tag} expected {expected} got {actual}")
    return sorted(errors)


@contextmanager
def _stubbed_parents(module: str, search_paths: Sequence[str]) -> Iterator[None]:
    parts = module.split(".")
    current_path = list(search_paths)
    saved: dict[str, object] = {}
    try:
        for index in range(1, len(parts)):
            fullname = ".".join(parts[:index])
            spec = importlib.machinery.PathFinder.find_spec(fullname, current_path)
            if spec is None or spec.submodule_search_locations is None:
                raise ModuleNotFoundError(fullname)
            saved[fullname] = sys.modules.get(fullname, _MISSING)
            stub = types.ModuleType(fullname)
            stub.__spec__ = spec
            stub.__path__ = list(spec.submodule_search_locations)
            stub.__package__ = fullname
            sys.modules[fullname] = stub
            current_path = list(spec.submodule_search_locations)
        yield
    finally:
        for fullname, previous in reversed(saved.items()):
            if previous is _MISSING:
                sys.modules.pop(fullname, None)
            else:
                sys.modules[fullname] = previous  # type: ignore[assignment]


def find_spec_without_import(module: str, search_paths: Sequence[Path]) -> importlib.machinery.ModuleSpec | None:
    paths = [str(path.resolve()) for path in search_paths]
    paths.extend(path for path in sys.path if path and path not in paths)
    old_path = list(sys.path)
    try:
        sys.path[:] = paths
        with _stubbed_parents(module, paths):
            return importlib.util.find_spec(module)
    except (ImportError, AttributeError, ValueError):
        return None
    finally:
        sys.path[:] = old_path


def _source_symbols(path: Path) -> set[str]:
    return top_level_definitions(parse_python(path))


def _module_candidate_relatives(repo: Path, module: str, roots: Sequence[Path]) -> set[str]:
    relatives: set[str] = set()
    for candidate in _module_candidates(module, roots):
        try:
            relatives.add(candidate.resolve().relative_to(repo.resolve()).as_posix())
        except ValueError:
            continue
    return relatives


def classify_r10_modules(
    repo: Path,
    roots: Sequence[Path],
    inventory: Sequence[dict[str, str]],
    references: Sequence[dict[str, str]],
) -> dict[str, str]:
    modules = {
        *(row["policy_class_module"] for row in inventory),
        *(row["module_reference"] for row in references if row["module_reference"] != "NONE"),
    }
    inventory_markers: dict[str, set[str]] = defaultdict(set)
    for row in inventory:
        inventory_markers[row["policy_class_module"]].add(row.get("loadable_under_current_source", ""))
    phase0_paths: set[str] = set()
    if (repo / ".git").exists():
        try:
            phase0_paths = set(_run_git(repo, "ls-tree", "-r", "--name-only", BASE_SHA).decode().splitlines())
        except subprocess.CalledProcessError:
            phase0_paths = set()
    classification: dict[str, str] = {}
    for module in sorted(modules):
        if any("third_party" in marker for marker in inventory_markers[module]):
            classification[module] = "external"
            continue
        candidates = _module_candidate_relatives(repo, module, roots)
        present_now = any((repo / candidate).is_file() for candidate in candidates)
        present_at_phase0 = bool(candidates & phase0_paths)
        classification[module] = "repository-local" if present_now or present_at_phase0 else "external"
    return classification


def r10_result(repo: Path, roots: Sequence[Path]) -> dict[str, Any]:
    torch_before = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    with (repo / "docs/migration/CHECKPOINT_INVENTORY.tsv").open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle, delimiter="\t"))
    with (repo / "docs/migration/CHECKPOINT_MODULE_REFERENCES.tsv").open(encoding="utf-8", newline="") as handle:
        references = list(csv.DictReader(handle, delimiter="\t"))
    policy_modules = sorted({row["policy_class_module"] for row in inventory})
    serialized = sorted({(row["module_reference"], row["qualname_if_found"]) for row in references if row["module_reference"] != "NONE"})
    expected_symbols = {
        (row["policy_class_module"], row["policy_class_qualname"].split(".", 1)[0])
        for row in inventory
        if row.get("policy_class_qualname")
    }
    expected_symbols.update((module, symbol.split(".", 1)[0]) for module, symbol in serialized if symbol)
    module_classification = classify_r10_modules(repo, roots, inventory, references)
    failures: list[str] = []
    resolutions: dict[str, str] = {}
    for module in sorted(set(policy_modules) | {module for module, _ in serialized}):
        spec = find_spec_without_import(module, roots)
        if spec is None or not spec.origin:
            failures.append(f"unresolvable_module:{module}")
        else:
            origin = Path(spec.origin).resolve()
            resolutions[module] = str(origin)
            if module_classification[module] == "repository-local" and not origin.is_relative_to(repo.resolve()):
                failures.append(f"repository_local_module_outside_repo:{module}:{origin}")
    for module, symbol in sorted(expected_symbols):
        if module_classification[module] != "repository-local":
            continue
        origin = resolutions.get(module)
        if not origin:
            continue
        path = Path(origin)
        if path.is_file() and path.resolve().is_relative_to(repo.resolve()) and symbol not in _source_symbols(path):
            failures.append(f"missing_symbol:{module}.{symbol}")
    torch_after = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    if torch_after != torch_before:
        failures.append("torch_imported_during_r10")
    return {
        "checkpoint_count": len(inventory),
        "module_reference_rows": len(references),
        "policy_modules": policy_modules,
        "serialized_references": [f"{module}.{symbol}" for module, symbol in serialized],
        "module_classification": dict(sorted(module_classification.items())),
        "resolutions": dict(sorted(resolutions.items())),
        "torch_modules_added": sorted(torch_after - torch_before),
        "failures": sorted(failures),
    }


def _normalized_ast_tokens(path: Path) -> tuple[str, ...]:
    dump = ast.dump(parse_python(path), annotate_fields=False, include_attributes=False)
    return tuple(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[-+]?\d+(?:\.\d+)?|[^\s]", dump))


def duplicate_diagnostic(repo: Path, tracked: Sequence[str] | None = None) -> list[dict[str, str]]:
    tracked = list(tracked if tracked is not None else tracked_paths(repo))
    exact_groups: dict[str, list[str]] = defaultdict(list)
    for relative in tracked:
        path = repo / relative
        if path.is_file() and path.stat().st_size >= 200:
            exact_groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(relative)
    rows: list[dict[str, str]] = []
    for digest, paths in sorted(exact_groups.items()):
        if len(paths) >= 2:
            for left, right in itertools.combinations(sorted(paths), 2):
                rows.append({"kind": "exact_sha256", "path_a": left, "path_b": right, "evidence": digest})
    normalized: dict[str, tuple[str, ...]] = {}
    for relative in sorted(path for path in tracked if path.endswith(".py")):
        try:
            normalized[relative] = _normalized_ast_tokens(repo / relative)
        except (OSError, SyntaxError, UnicodeError):
            continue
    for left, right in itertools.combinations(sorted(normalized), 2):
        a, b = normalized[left], normalized[right]
        if not a and not b:
            ratio = 1.0
        else:
            length_ratio = min(len(a), len(b)) / max(len(a), len(b), 1)
            if length_ratio < 0.95:
                continue
            matcher = difflib.SequenceMatcher(None, a, b, autojunk=True)
            if matcher.quick_ratio() < 0.95:
                continue
            ratio = matcher.ratio()
        if ratio >= 0.95:
            rows.append({"kind": "ast_similarity", "path_a": left, "path_b": right, "evidence": f"ratio={ratio:.6f};tokens={len(a)},{len(b)}"})
    return sorted(rows, key=lambda row: (row["kind"], row["path_a"], row["path_b"], row["evidence"]))


def write_d1(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("kind", "path_a", "path_b", "evidence"), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _baseline_payload(findings: Sequence[Finding], phase: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for finding in sorted(findings):
        grouped[finding.rule].append(finding.as_dict())
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "base_sha": BASE_SHA,
        "policy": "exact known debt may shrink but may not grow",
        "violations": {rule: grouped.get(rule, []) for rule in ("R6", "R7a", "R7b", "R7c")},
    }


def write_baseline_json(path: Path, findings: Sequence[Finding], phase: int) -> None:
    path.write_text(json.dumps(_baseline_payload(findings, phase), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_baseline(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_supplement(path: Path) -> list[dict[str, str]]:
    """Explicit, hand-maintained forward evidence: findings that are known,
    accepted additions on top of the frozen baseline at `path`, without ever
    rewriting that frozen baseline itself. Each row must carry its own
    `key` matching `rule|concept|path|detail` exactly (validated by
    `supplement_keys`); this is deliberately the same one-row-per-finding,
    exact-tuple-match shape as the frozen baseline, so a supplement can only
    ever accept the specific findings it lists -- never a rule or path
    wildcard, and never a substitute for updating the frozen baseline via
    its own `snapshot` mechanism when a future phase is ready to do that."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def supplement_keys(rows: Sequence[dict[str, str]]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        finding = Finding(row["rule"], row["concept"], row["path"], row["detail"])
        if row["key"] != finding.key:
            raise RuntimeError(f"Malformed supplement key: {row['key']}")
        keys.add(finding.key)
    return keys


def phase7_move_paths(repo: Path) -> dict[str, str]:
    with (repo / PHASE7_MOVE_MANIFEST).open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["phase7_action"] == "MOVE"
        ]
    moves = {row["old_path"]: row["new_path"] for row in rows}
    if len(moves) != len(rows) or len(set(moves.values())) != len(rows):
        raise RuntimeError("Phase-7 MOVE manifest is not one-to-one")
    return moves


def ratchet_errors(
    current: Sequence[Finding],
    baseline: dict[str, Any],
    *,
    path_moves: dict[str, str] | None = None,
    supplement: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Compare current findings against the frozen baseline plus an
    explicit forward supplement (each an exact rule|concept|path|detail
    key). The frozen baseline itself is never mutated to absorb new,
    post-freeze-phase findings; a supplement grants forward acceptance for
    specifically enumerated keys only -- any other new finding, including
    other findings for the same rule or symbol, still ratchets as growth."""
    baseline_keys: set[str] = set()
    for items in baseline.get("violations", {}).values():
        for item in items:
            frozen = Finding(item["rule"], item["concept"], item["path"], item["detail"])
            if item["key"] != frozen.key:
                raise RuntimeError(f"Malformed frozen baseline key: {item['key']}")
            path = (path_moves or {}).get(frozen.path, frozen.path)
            baseline_keys.add(Finding(frozen.rule, frozen.concept, path, frozen.detail).key)
    baseline_keys |= set(supplement or ())
    current_keys = {finding.key for finding in current}
    added = sorted(current_keys - baseline_keys)
    resolved = sorted(baseline_keys - current_keys)
    return [f"new_baseline_violation:{key}" for key in added], resolved


def write_baseline_markdown(
    path: Path,
    findings: Sequence[Finding],
    d1_rows: Sequence[dict[str, str]],
    r9: Sequence[Finding],
    r10: dict[str, Any],
    phase: int,
) -> None:
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        grouped[finding.rule].append(finding)
    exact = sum(row["kind"] == "exact_sha256" for row in d1_rows)
    similar = sum(row["kind"] == "ast_similarity" for row in d1_rows)
    lines = [
        "# Phase-1 Baseline Violations",
        "",
        "> Generated deterministically by `docs/migration/tools/migration_integrity.py snapshot`.",
        "> Do not normalize this debt away by hand. Later phases may deliberately shrink it; growth fails.",
        "",
        f"Base: `{BASE_SHA}`. Phase: {phase} (from `{DEFAULT_OWNERS}`).",
        "",
        "## 1. ACCEPTED PRE-EXISTING MIGRATION DEBT",
        "",
    ]
    for rule in ("R6", "R7a", "R7b", "R7c"):
        items = sorted(grouped.get(rule, []))
        lines.extend((f"### {rule} ({len(items)})", "", "| concept/symbol | exact file | evidence |", "|---|---|---|"))
        lines.extend(f"| `{item.concept}` | `{item.path}` | `{item.detail}` |" for item in items)
        if not items:
            lines.append("| _none_ | _none_ | rule currently green |")
        lines.append("")
    unresolved = (
        [
            "- Shared farming and position ownership are canonical at repository root; R6 and R7a are green.",
            "- B1 and B2 are removed; retained old paths are registered behavior-free compatibility surfaces through Phase 12.",
            "- B3 remains active through Phase 7 and expires at the Phase-8 archive extraction boundary.",
            "- R7b remains active with current-layout semantics and tightens as legacy/archive boundaries appear.",
        ]
        if phase >= 7
        else
        [
            "- Shared farming ownership is canonical under `flyff_farming_simulator/farming`; R6 and farming R7a are green.",
            "- Native position ownership debt resolves during Phase 5 after its real-client gate.",
            "- R7b remains active with current-layout semantics and tightens as legacy/archive boundaries appear.",
            "- R7c contains current repository re-exports; every Phase-4 B1 re-export is explicitly registered through Phase 6.",
        ]
        if phase >= 4
        else [
            "- R6/R7a farming ownership debt resolves during Phase 4 canonical farming.",
            "- Native position ownership debt resolves during Phase 5 after its real-client gate.",
            "- R7b remains active with current-layout semantics and tightens as legacy/archive boundaries appear.",
            "- R7c baseline re-exports must shrink or become explicit registered shims; none was installed in Phase 1.",
        ]
    )
    lines.extend(
        (
            "## 2. RULES ALREADY GREEN",
            "",
            f"- R9: **GREEN** ({len(r9)} violations).",
            f"- R10 policy-class modules: **{'GREEN' if not r10['failures'] else 'RED'}** across {r10['checkpoint_count']} inventory rows.",
            f"- R10 full serialized references: **{'GREEN' if not r10['failures'] else 'RED'}** across {r10['module_reference_rows']} reference rows.",
            f"- R10 module classification: `{json.dumps(r10['module_classification'], sort_keys=True)}`.",
            "- Phase-1 R10 protects the frozen Phase-0 checkpoint corpus described by `CHECKPOINT_INVENTORY.tsv` and `CHECKPOINT_MODULE_REFERENCES.tsv`.",
            "- Phase-2 G10a independently regenerates that corpus inventory against preserved artifacts.",
            "",
            "## 3. DIAGNOSTIC-ONLY FINDINGS",
            "",
            f"- D1 exact SHA-256 pairs (tracked files >=200 bytes): **{exact}**.",
            f"- D1 AST-normalized Python pairs >=95% similar: **{similar}**.",
            f"- Exact deterministic evidence: `{DEFAULT_D1}`.",
            "- D1 never gates and did not trigger source deletion or merging.",
            "",
            "## 4. UNRESOLVED/REQUIRES LATER PHASE",
            "",
        )
    )
    lines.extend(unresolved)
    lines.extend(("", "## 5. NEW BLOCKING VIOLATIONS", "", "None at snapshot time.", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def collect(repo: Path, registry: dict[str, Any]) -> dict[str, Any]:
    files, parse_errors = _tracked_sources(repo)
    roots = python_roots(repo, registry)
    findings, ownership_errors = _concept_findings(files, registry)
    edges = collect_import_edges(repo, files, roots)
    r9 = sorted([*parse_errors, *r9_findings(edges, set(tracked_paths(repo)))])
    findings = sorted([*findings, *r7b_findings(edges, registry)])
    return {
        "findings": findings,
        "ownership_errors": ownership_errors,
        "r9": r9,
        "r10": r10_result(repo, roots),
        "bridge_errors": bridge_errors(repo, registry),
    }


def snapshot(repo: Path) -> dict[str, Any]:
    registry = load_registry(repo)
    phase = active_phase(registry)
    result = collect(repo, registry)
    d1_rows = duplicate_diagnostic(repo)
    write_d1(repo / DEFAULT_D1, d1_rows)
    write_baseline_json(repo / DEFAULT_BASELINE, result["findings"], phase)
    write_baseline_markdown(repo / DEFAULT_BASELINE_MD, result["findings"], d1_rows, result["r9"], result["r10"], phase)
    return summary(result, d1_rows=d1_rows)


def summary(result: dict[str, Any], *, d1_rows: Sequence[dict[str, str]] | None = None) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for finding in result["findings"]:
        counts[finding.rule] += 1
    payload: dict[str, Any] = {
        "baseline_counts": {rule: counts.get(rule, 0) for rule in ("R6", "R7a", "R7b", "R7c")},
        "r9_violations": len(result["r9"]),
        "r10_checkpoint_count": result["r10"]["checkpoint_count"],
        "r10_module_reference_rows": result["r10"]["module_reference_rows"],
        "r10_module_classification": result["r10"]["module_classification"],
        "r10_failures": result["r10"]["failures"],
        "torch_modules_added": result["r10"]["torch_modules_added"],
        "ownership_errors": result["ownership_errors"],
        "bridge_errors": result["bridge_errors"],
    }
    if d1_rows is not None:
        payload["d1_exact_pairs"] = sum(row["kind"] == "exact_sha256" for row in d1_rows)
        payload["d1_ast_similar_pairs"] = sum(row["kind"] == "ast_similarity" for row in d1_rows)
    return payload


def check(repo: Path) -> tuple[dict[str, Any], list[str]]:
    result = collect(repo, load_registry(repo))
    supplement_rows = [row for relative in DEFAULT_SUPPLEMENTS for row in load_supplement(repo / relative)]
    supplement = supplement_keys(supplement_rows)
    ratchet, resolved = ratchet_errors(
        result["findings"],
        load_baseline(repo / DEFAULT_BASELINE),
        path_moves=phase7_move_paths(repo),
        supplement=supplement,
    )
    errors = [
        *result["ownership_errors"],
        *result["bridge_errors"],
        *ratchet,
        *[f"R9:{finding.key}" for finding in result["r9"]],
        *[f"R10:{failure}" for failure in result["r10"]["failures"]],
    ]
    payload = summary(result)
    payload.update({
        "resolved_baseline_entries": resolved,
        "supplement_entries_applied": sorted(supplement),
        "errors": sorted(errors),
        "ok": not errors,
    })
    return payload, sorted(errors)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("raw", "snapshot", "check", "diagnose"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo = args.repo.resolve()
    if args.command == "snapshot":
        payload = snapshot(repo)
        print(json.dumps(payload, indent=2, sort_keys=True))
        failures = payload["r10_failures"] or payload["ownership_errors"] or payload["bridge_errors"] or payload["r9_violations"]
        return 1 if failures else 0
    if args.command == "diagnose":
        rows = duplicate_diagnostic(repo)
        output = args.output or repo / DEFAULT_D1
        if not output.is_absolute():
            output = repo / output
        write_d1(output, rows)
        print(json.dumps({"rows": len(rows), "exact": sum(row["kind"] == "exact_sha256" for row in rows), "similar": sum(row["kind"] == "ast_similarity" for row in rows)}, sort_keys=True))
        return 0
    if args.command == "raw":
        result = collect(repo, load_registry(repo))
        print(json.dumps({**summary(result), "findings": [item.as_dict() for item in result["findings"]]}, indent=2, sort_keys=True))
        return 0
    payload, errors = check(repo)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
