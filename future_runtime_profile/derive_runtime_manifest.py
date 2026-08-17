"""Phase-11 future-derivation dry-run resolver.

Reads future_runtime_profile/dependency_profiles.toml's
[profiles.future_runtime_candidate] and statically resolves what a future
deployment/live derivative's import closure would actually contain if
built from THIS SAME canonical source tree (never a copied fork). This
tool builds NOTHING: no PyInstaller, no Nuitka, no dist/ output, no file
copy. It is a read-only, dry-run report.

Deliberately NOT named packaging/ (the authorization's own suggested
location): a real, well-known PyPI library is also named "packaging"
(a transitive dependency of pip/setuptools/pytest, confirmed installed in
this venv) and Python's import system resolves a regular package (that
library, which has __init__.py) ahead of a same-named directory here that
lacks one -- confirmed empirically: `import packaging` from this
repository's root still resolved to site-packages/packaging/__init__.py,
never this directory, regardless of sys.path order, because a namespace
portion never outranks a regular package found anywhere later in the
scan. `python -m packaging.derive_runtime_manifest` would therefore have
silently tried to run against the wrong "packaging" entirely. Named
future_runtime_profile/ instead -- confirmed collision-free.

Usage:
    python -m future_runtime_profile.derive_runtime_manifest
    python -m future_runtime_profile.derive_runtime_manifest --json

Exit code is 0 only if every check passes (candidate closure clean of
forbidden dev/training implementation outside the registered exact
exceptions, every ABI-compatibility/resource path tracked and present, no
duplicate algorithm ownership between a compatibility shim and its
canonical navigation/* owner). Never commit this script's stdout as a
"final standalone file list" -- the development source keeps changing;
re-run this fresh whenever the question actually needs answering.

The static import-closure walker below is deliberately self-contained
(not imported from tests/test_dev_app_import_closure.py, and not imported
by it) so this tool has no dependency on the test suite and can be run
standalone by a developer. The two walkers cover different entry-point
sets (this one walks every module in [shared_runtime_packages], the test
walks from apps/dev_app.py specifically) and are kept independently
verifiable rather than sharing one implementation.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python <3.11 fallback, unused here
    import tomli as tomllib  # type: ignore[no-redef]

REPO = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO / "future_runtime_profile" / "dependency_profiles.toml"


def load_profile(path: Path = PROFILE_PATH) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


# ---------------------------------------------------------------------------
# Static AST closure walker (self-contained -- see module docstring)
# ---------------------------------------------------------------------------


def _module_file(module_name: str, repo: Path) -> Path | None:
    rel = Path(*module_name.split("."))
    candidate = repo / rel.with_suffix(".py")
    if candidate.is_file():
        return candidate
    package_candidate = repo / rel / "__init__.py"
    if package_candidate.is_file():
        return package_candidate
    return None


def _package_parts(path: Path, repo: Path) -> tuple[str, ...]:
    relative = path.resolve().relative_to(repo)
    parts = relative.with_suffix("").parts
    return parts[:-1] if parts[-1] == "__init__" else parts[:-1]


def _string_constants(node: ast.AST) -> set[str]:
    return {c.value for c in ast.walk(node) if isinstance(c, ast.Constant) and isinstance(c.value, str)}


def _lazy_getattr_targets(module_ast: ast.Module, requested_names: set[str]) -> set[str]:
    """PEP 562 module-level __getattr__ lazy-import dispatch (used by
    mapper/__init__.py) -- only the branch(es) matching a requested name
    execute at runtime."""
    targets: set[str] = set()
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            for branch in ast.walk(node):
                if isinstance(branch, ast.If) and _string_constants(branch.test) & requested_names:
                    for stmt in branch.body:
                        if isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module:
                            targets.add(stmt.module)
                        elif isinstance(stmt, ast.Import):
                            targets.update(alias.name for alias in stmt.names)
    return targets


def _has_lazy_getattr(module_ast: ast.Module) -> bool:
    return any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in module_ast.body)


def _import_edges(path: Path, repo: Path) -> list[tuple[str, frozenset[str] | None]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    edges: list[tuple[str, frozenset[str] | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            edges.extend((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    edges.append((node.module, frozenset(a.name for a in node.names)))
            else:
                package = _package_parts(path, repo)
                base = package[: len(package) - (node.level - 1)] if node.level - 1 <= len(package) else ()
                if node.module:
                    dotted = ".".join((*base, node.module)) if base else node.module
                    edges.append((dotted, frozenset(a.name for a in node.names)))
                elif base:
                    edges.append((".".join(base), frozenset(a.name for a in node.names)))
    return edges


def _resolve_lazy(path: Path, module_name: str, symbols: frozenset[str] | None, repo: Path) -> list[tuple[str, frozenset[str] | None]]:
    target = _module_file(module_name, repo)
    if target is None or target.name != "__init__.py" or symbols is None:
        return [(module_name, symbols)]
    target_ast = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    if not _has_lazy_getattr(target_ast):
        return [(module_name, symbols)]
    return [(name, None) for name in _lazy_getattr_targets(target_ast, set(symbols))]


@dataclass
class ClosureResult:
    modules: set[str] = field(default_factory=set)
    files_walked: set[Path] = field(default_factory=set)


def walk_closure(
    entry_files: list[Path],
    repo: Path,
    exceptions: dict[tuple[str, str], frozenset[str]] | None = None,
) -> ClosureResult:
    """BFS over every locally-resolvable module name reachable from
    `entry_files`. An edge matching `exceptions` exactly (importer path +
    dependency + exact symbol subset) is recorded but not expanded past."""
    exceptions = exceptions or {}
    result = ClosureResult()
    result.files_walked.update(f.resolve() for f in entry_files)
    frontier = list(entry_files)
    while frontier:
        current = frontier.pop()
        importer_rel = str(current.resolve().relative_to(repo)).replace("\\", "/")
        for module_name, symbols in _import_edges(current, repo):
            for resolved_name, resolved_symbols in _resolve_lazy(current, module_name, symbols, repo):
                result.modules.add(resolved_name)
                key = (importer_rel, resolved_name)
                if key in exceptions and resolved_symbols is not None and resolved_symbols <= exceptions[key]:
                    continue
                target = _module_file(resolved_name, repo)
                if target is not None and target.resolve() not in result.files_walked:
                    result.files_walked.add(target.resolve())
                    frontier.append(target)
    return result


# ---------------------------------------------------------------------------
# Dry-run report
# ---------------------------------------------------------------------------


@dataclass
class DerivationReport:
    candidate_first_party_modules: list[str] = field(default_factory=list)
    abi_compatibility_modules: list[str] = field(default_factory=list)
    candidate_resources: list[str] = field(default_factory=list)
    forbidden_dependency_edges: list[str] = field(default_factory=list)
    unresolved_future_choices: list[str] = field(default_factory=list)
    duplicate_ownership_issues: list[str] = field(default_factory=list)
    missing_tracked_files: list[str] = field(default_factory=list)
    exceptions_applied: list[str] = field(default_factory=list)
    ok: bool = True

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "candidate_first_party_modules": sorted(self.candidate_first_party_modules),
            "abi_compatibility_modules": self.abi_compatibility_modules,
            "candidate_resources": self.candidate_resources,
            "forbidden_dependency_edges": self.forbidden_dependency_edges,
            "unresolved_future_choices": self.unresolved_future_choices,
            "duplicate_ownership_issues": self.duplicate_ownership_issues,
            "missing_tracked_files": self.missing_tracked_files,
            "exceptions_applied": self.exceptions_applied,
        }


def _tracked_files(repo: Path) -> set[str]:
    import subprocess

    out = subprocess.run(["git", "ls-files"], cwd=str(repo), capture_output=True, text=True, check=True)
    return set(out.stdout.splitlines())


def derive(profile: dict | None = None, repo: Path = REPO) -> DerivationReport:
    profile = profile or load_profile()
    frc = profile["profiles"]["future_runtime_candidate"]
    report = DerivationReport()
    report.unresolved_future_choices = list(frc.get("unresolved_future_choices", []))
    report.abi_compatibility_modules = list(frc.get("runtime_abi_compatibility_modules", []))
    report.candidate_resources = list(frc.get("candidate_runtime_resources", []))

    exceptions: dict[tuple[str, str], frozenset[str]] = {
        (exc["importer"], exc["dependency"]): frozenset(exc["symbols"])
        for exc in frc.get("known_exact_exceptions", [])
    }
    report.exceptions_applied = [f"{i} -> {d}" for (i, d) in exceptions]

    excluded = {
        (repo / rel).resolve() for rel in frc.get("excluded_from_shared_entry_walk", [])
    }
    entry_files: list[Path] = []
    for package in frc.get("shared_runtime_packages", []):
        pkg_path = repo / package
        if pkg_path.is_dir():
            entry_files.extend(
                p for p in sorted(pkg_path.rglob("*.py")) if p.resolve() not in excluded
            )
        else:
            single = repo / f"{package}.py"
            if single.is_file() and single.resolve() not in excluded:
                entry_files.append(single)
    for rel in frc.get("additional_shared_entry_files", []):
        extra = repo / rel
        if extra.is_file() and extra.resolve() not in excluded:
            entry_files.append(extra)

    closure = walk_closure(entry_files, repo, exceptions=exceptions)
    report.candidate_first_party_modules = sorted(
        name for name in closure.modules if _module_file(name, repo) is not None
    )

    forbidden = tuple(frc.get("forbidden_first_party_prefixes", []))
    for name in closure.modules:
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
            report.forbidden_dependency_edges.append(name)
    report.forbidden_dependency_edges.sort()

    tracked = _tracked_files(repo)
    for module_name in report.abi_compatibility_modules:
        target = _module_file(module_name, repo)
        if target is None:
            report.missing_tracked_files.append(module_name)
            continue
        rel = str(target.relative_to(repo)).replace("\\", "/")
        if rel not in tracked:
            report.missing_tracked_files.append(rel)
    for resource in report.candidate_resources:
        resource_path = repo / resource
        if not resource_path.is_file() or resource.replace("\\", "/") not in tracked:
            report.missing_tracked_files.append(resource)

    # Duplicate-ownership check: an ABI-compatibility module must not
    # define (class/def) the same top-level name its canonical
    # navigation/* counterpart defines -- it may only re-export.
    canonical_defs: dict[str, set[str]] = {}
    for canonical in ("navigation/kinodynamic_route_planner.py", "navigation/movement_kernel.py"):
        path = repo / canonical
        if path.is_file():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=canonical)
            canonical_defs[canonical] = {
                n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            }
    for compat in ("simulator/kinodynamic_route_planner.py", "simulator/movement_kernel.py"):
        path = repo / compat
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=compat)
        compat_defs = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
        if compat_defs:
            report.duplicate_ownership_issues.append(
                f"{compat} defines {sorted(compat_defs)} -- ABI-compatibility modules must be re-export only"
            )

    report.ok = not (report.forbidden_dependency_edges or report.missing_tracked_files or report.duplicate_ownership_issues)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a human report")
    args = parser.parse_args(argv)

    report = derive()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"FUTURE DEPLOYMENT DERIVATION PROFILE: {'PASS' if report.ok else 'FAIL'}")
        print(f"  candidate first-party modules: {len(report.candidate_first_party_modules)}")
        print(f"  ABI compatibility modules: {report.abi_compatibility_modules}")
        print(f"  candidate resources: {report.candidate_resources}")
        print(f"  exceptions applied: {report.exceptions_applied}")
        print(f"  forbidden dependency edges: {report.forbidden_dependency_edges}")
        print(f"  missing tracked files: {report.missing_tracked_files}")
        print(f"  duplicate ownership issues: {report.duplicate_ownership_issues}")
        print(f"  unresolved future choices ({len(report.unresolved_future_choices)}):")
        for item in report.unresolved_future_choices:
            print(f"    - {item}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
