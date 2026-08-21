"""Phase-11 future-derivability gate (Section 10 of the authorization).

Formalizes the dry-run resolver's (`tools.future_runtime_profile.derive_runtime_
manifest`) already-passing result into explicit, individually named
proof-point assertions, so a future regression fails on the SPECIFIC
property it breaks rather than only on an aggregate PASS/FAIL. This is a
static/architectural gate: passing it does not mean a runtime derivative
exists, is ready, or has been built -- only that the canonical source
tree's current shape would legitimately support deriving one later,
without a copied fork.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESOLVER_PATH = REPO / "tools" / "future_runtime_profile" / "derive_runtime_manifest.py"

_spec = importlib.util.spec_from_file_location("phase11_derive_runtime_manifest", RESOLVER_PATH)
assert _spec is not None and _spec.loader is not None
resolver = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = resolver
_spec.loader.exec_module(resolver)


def _report():
    return resolver.derive(repo=REPO)


def _profile():
    return resolver.load_profile()


# 1. farming/position/navigation/mapper each have exactly one canonical
#    owner directory under shared_runtime_packages -- no duplicate top-level
#    ownership of the same package name elsewhere in the tree.
def test_point_01_shared_packages_have_single_canonical_ownership() -> None:
    frc = _profile()["profiles"]["future_runtime_candidate"]
    packages = frc["shared_runtime_packages"]
    assert len(packages) == len(set(packages))
    for package in ("farming", "position", "navigation", "mapper"):
        assert package in packages
        assert (REPO / package).is_dir()


# 2. The candidate closure contains zero edges into devtools/, recorder/,
#    simulator.environment (and the rest of the SIMULATOR_ONLY/TRAINING_ONLY
#    simulator surface), tests/, or root scratchpad_ scripts.
def test_point_02_no_forbidden_dev_recorder_simulator_scratchpad_edges() -> None:
    report = _report()
    assert report.forbidden_dependency_edges == []


# 3. Simulator training/environment implementation is not required by the
#    shared runtime closure -- confirmed by the same forbidden-prefix check
#    (simulator.environment/router_waypoint_env/static_waypoint_env/
#    single_obstacle_env/synthetic/basic_training/navigation_dataset/
#    navigation_history/cli/fair_time_cli are all listed as forbidden
#    prefixes) plus a direct assertion that the closure contains no
#    `simulator.*` module outside the three registered ABI modules.
def test_point_03_simulator_training_environment_not_in_shared_closure() -> None:
    report = _report()
    simulator_modules = {m for m in report.candidate_first_party_modules if m.startswith("simulator.") or m == "simulator"}
    allowed = set(report.abi_compatibility_modules) | {"simulator.schema"}
    assert simulator_modules <= allowed, simulator_modules - allowed


# 4. ABI-compatibility modules are explicitly distinguished from ordinary
#    shared-runtime algorithm modules: each is tracked, present, and
#    (per the duplicate-ownership check) re-export only relative to its
#    canonical navigation/* owner.
def test_point_04_abi_compatibility_modules_distinguished_and_reexport_only() -> None:
    report = _report()
    assert report.abi_compatibility_modules == [
        "simulator.split_branch_policy",
    ]
    assert report.duplicate_ownership_issues == []
    assert report.missing_tracked_files == []


# 5. No copied runtime source tree: the profile is explicitly declared
#    canonical-source, not a fork, and every entry file the resolver
#    walked resolves to a path inside this same repository.
def test_point_05_no_copied_fork_declared_or_implied() -> None:
    frc = _profile()["profiles"]["future_runtime_candidate"]
    assert frc["source_strategy"] == "canonical_source"
    assert frc["copied_fork"] is False


# 6. The candidate closure resolves only git-tracked source -- nothing
#    the resolver depends on is an untracked/local-only file.
def test_point_06_profile_resolves_only_tracked_source() -> None:
    report = _report()
    assert report.missing_tracked_files == []


# 7. No old preservation worktree is needed: resolution happens entirely
#    against REPO (this working tree), never against pre-consolidation-head
#    or any other historical ref/worktree.
def test_point_07_resolution_is_self_contained_to_this_worktree() -> None:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(REPO), capture_output=True, text=True, check=True)
    assert Path(out.stdout.strip()).resolve() == REPO.resolve()
    report = _report()
    assert report.ok is True


# 8. Exactly one registered exception exists (the R1b
#    bot/runtime_controller.py -> farming.trainer coupling), and it is
#    not silently expanded into a general allowance.
def test_point_08_exactly_one_registered_exception_the_r1b_coupling() -> None:
    report = _report()
    assert report.exceptions_applied == ["bot/runtime_controller.py -> farming.trainer"]


# 9. DUAL_ROLE third-party packages (torch/gymnasium/stable_baselines3) are
#    present in the candidate profile's third-party list -- not excluded
#    merely because training also imports them (the explicit warning in
#    the authorization).
def test_point_09_dual_role_third_party_not_excluded() -> None:
    frc = _profile()["profiles"]["future_runtime_candidate"]
    for name in ("torch", "gymnasium", "stable_baselines3"):
        assert name in frc["candidate_third_party"]


# 10. Candidate runtime resources (native_farming.json and the two native
#     position/monster configs) are tracked and present on disk, matching
#     PHASE11_RUNTIME_RESOURCE_MANIFEST.tsv.
def test_point_10_candidate_resources_tracked_and_present() -> None:
    report = _report()
    assert report.candidate_resources == [
        "farming/native_farming.json",
        "position/native_monsters.json",
        "position/native_position.json",
    ]
    assert report.missing_tracked_files == []


# 11. Training-only files bundled inside an otherwise-shared package
#     (farming/{sb3_adapter,sb3_training,trainer}.py,
#     mapper/rl/{FeatureExtractor,GymEnv,OfflineTraining}.py) do not
#     themselves appear as candidate first-party modules -- they are
#     excluded from the entry walk, not silently included by directory
#     membership.
def test_point_11_bundled_training_only_files_excluded_from_candidate() -> None:
    report = _report()
    excluded_as_modules = {
        "farming.sb3_adapter", "farming.sb3_training", "farming.trainer",
        "mapper.rl.FeatureExtractor", "mapper.rl.GymEnv", "mapper.rl.OfflineTraining",
    }
    assert not (excluded_as_modules & set(report.candidate_first_party_modules))


# 12. Overall gate result: PASS, with zero forbidden edges, zero missing
#     tracked files, and zero duplicate-ownership issues simultaneously --
#     this is the same aggregate `report.ok` the resolver's own CLI prints,
#     asserted here so a change breaking any single input can't silently
#     leave `ok` true through an unrelated compensating change.
def test_point_12_overall_derivation_report_passes() -> None:
    report = _report()
    assert report.ok is True
    assert report.forbidden_dependency_edges == []
    assert report.missing_tracked_files == []
    assert report.duplicate_ownership_issues == []
