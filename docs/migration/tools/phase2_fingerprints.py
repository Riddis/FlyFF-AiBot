"""Phase-2 frozen-fingerprint gates: G4 (contracts), G11 (map bytes), G10a (checkpoints).

This tool is deliberately CHEAP and Torch-free. It never imports torch,
stable_baselines3, or any product runtime into its own process:

* G4 evaluates each farming owner in an isolated subprocess, because
  ``farming.observation`` exists in two roots and cannot be imported twice in
  one interpreter. The recorder's metadata-only copy is read via AST.
* G10a reads checkpoint ZIPs as data. ``policy_class`` and serialized module
  references are recovered by walking pickle OPCODES (never unpickling, so no
  repo/torch import happens). Observation/action spaces are unpickled through a
  restricted unpickler whose allowlist admits only gymnasium/numpy/builtins.

The expensive real ``PPO.load`` gate (G10b) lives in
``phase2_representative_load.py`` and is intentionally NOT invoked from here.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import pickle
import pickletools
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_DEFAULT = Path(__file__).resolve().parents[3]
FINGERPRINTS = "docs/migration/PHASE2_FINGERPRINTS.toml"
PHASE0_INVENTORY = "docs/migration/CHECKPOINT_INVENTORY.tsv"
PHASE0_REFERENCES = "docs/migration/CHECKPOINT_MODULE_REFERENCES.tsv"
SUPPLEMENT = "docs/migration/PHASE2_CHECKPOINT_SUPPLEMENT.tsv"

SUPPLEMENT_PROVENANCE = "first frozen during Phase 2 because Phase-0 inventory did not contain this field"

# Modules a checkpoint space pickle may legitimately reference. Anything else
# (notably torch and repo-local policy modules) is refused rather than imported.
_SPACE_ALLOWED_PREFIXES = ("gymnasium.", "numpy.", "numpy", "collections", "builtins", "_codecs")


def load_fingerprints(repo: Path) -> dict[str, Any]:
    with (repo / FINGERPRINTS).open("rb") as handle:
        data = tomllib.load(handle)
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported fingerprint schema: {data.get('schema_version')!r}")
    return data


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# G4
# ---------------------------------------------------------------------------

_G4_PROBE = r"""
import json, sys
out = {}
from farming.observation import (
    OBSERVATION_SCHEMA_ID, OBSERVATION_SIZE, OBSERVATION_SCHEMA_HASH,
    OBSERVATION_FIELDS, observation_schema_hash,
)
from farming.actions import POLICY_ACTION_NVECS, SteeringAction, FarmingEvent
from farming.model_contract import MODEL_CONTRACT_METADATA_VERSION
out["OBSERVATION_SCHEMA_ID"] = OBSERVATION_SCHEMA_ID
out["OBSERVATION_SIZE"] = int(OBSERVATION_SIZE)
out["OBSERVATION_FIELDS_LEN"] = len(OBSERVATION_FIELDS)
out["OBSERVATION_SCHEMA_HASH"] = OBSERVATION_SCHEMA_HASH
out["OBSERVATION_SCHEMA_HASH_RECOMPUTED"] = observation_schema_hash()
out["POLICY_ACTION_NVECS"] = list(POLICY_ACTION_NVECS)
out["ACTION_ENUM_LENGTHS"] = [len(SteeringAction), len(FarmingEvent)]
out["MODEL_CONTRACT_METADATA_VERSION"] = int(MODEL_CONTRACT_METADATA_VERSION)
try:
    from simulator.navigation_history import (
        RAW_OBSERVATION_SIZE, SIDECAR_SIZE, POLICY_INPUT_SIZE,
        TEMPORAL_SIDECAR_SIZE, PREVIOUS_STEERING_SIDECAR_SIZE,
    )
    out["RAW_OBSERVATION_SIZE"] = int(RAW_OBSERVATION_SIZE)
    out["SIDECAR_SIZE"] = int(SIDECAR_SIZE)
    out["POLICY_INPUT_SIZE"] = int(POLICY_INPUT_SIZE)
    out["TEMPORAL_SIDECAR_SIZE"] = int(TEMPORAL_SIDECAR_SIZE)
    out["PREVIOUS_STEERING_SIDECAR_SIZE"] = int(PREVIOUS_STEERING_SIDECAR_SIZE)
except Exception:
    pass
try:
    from simulator.movement_kernel import (
        MOVEMENT_PHYSICS_MODEL_ID, LEGACY_MOVEMENT_PHYSICS_MODEL_ID,
        PATH_LENGTH_CELLS_PER_TICK, ONSET_TURN_RADIANS, STEADY_TURN_RADIANS,
        DEFAULT_SUBSTEPS,
    )
    out["MOVEMENT_PHYSICS_MODEL_ID"] = MOVEMENT_PHYSICS_MODEL_ID
    out["LEGACY_MOVEMENT_PHYSICS_MODEL_ID"] = LEGACY_MOVEMENT_PHYSICS_MODEL_ID
    out["PATH_LENGTH_CELLS_PER_TICK"] = float(PATH_LENGTH_CELLS_PER_TICK)
    out["ONSET_TURN_RADIANS"] = float(ONSET_TURN_RADIANS)
    out["STEADY_TURN_RADIANS"] = float(STEADY_TURN_RADIANS)
    out["DEFAULT_SUBSTEPS"] = int(DEFAULT_SUBSTEPS)
except Exception:
    pass
sys.stdout.write(json.dumps(out))
"""


def probe_root(repo: Path, root: str) -> dict[str, Any]:
    """Import one farming owner in a clean subprocess and report its contract.

    ``-s`` (not ``-I``) because isolated mode would also discard PYTHONPATH,
    which is exactly how each root is put on the import path in turn.
    """
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-s", "-c", _G4_PROBE],
        cwd=str(repo / root),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"G4 probe failed for {root}: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def _ast_constants(path: Path, names: Iterable[str]) -> dict[str, Any]:
    wanted, found = set(names), {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        targets: list[str] = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        for name in targets:
            if name in wanted:
                try:
                    found[name] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return found


def _ast_imports(path: Path, module: str, names: Iterable[str]) -> dict[str, str]:
    wanted, found = set(names), {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 0 or node.module != module:
            continue
        for alias in node.names:
            if alias.name in wanted:
                found[alias.name] = alias.asname or alias.name
    return found


def check_g4(repo: Path, fp: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    g4 = fp["g4"]
    failures: list[str] = []
    evidence: dict[str, Any] = {}

    def expect(label: str, actual: Any, pinned: Any) -> None:
        evidence[label] = actual
        if actual != pinned:
            failures.append(f"G4 {label}: source={actual!r} pinned={pinned!r}")

    for root in ("flyff_farming_simulator", "foreground_vision_bot"):
        probe = probe_root(repo, root)
        evidence[f"{root}.probe"] = probe
        expect(f"{root}.OBSERVATION_SCHEMA_ID", probe["OBSERVATION_SCHEMA_ID"], g4["observation_schema_id"]["value"])
        expect(f"{root}.OBSERVATION_SIZE", probe["OBSERVATION_SIZE"], g4["observation_size"]["value"])
        expect(f"{root}.OBSERVATION_FIELDS_LEN", probe["OBSERVATION_FIELDS_LEN"], g4["observation_size"]["value"])
        expect(f"{root}.OBSERVATION_SCHEMA_HASH", probe["OBSERVATION_SCHEMA_HASH"], g4["observation_schema_hash"]["value"])
        # 5.2: require live recomputation, not constant-to-constant comparison.
        expect(
            f"{root}.observation_schema_hash()recomputed",
            probe["OBSERVATION_SCHEMA_HASH_RECOMPUTED"],
            g4["observation_schema_hash"]["value"],
        )
        expect(f"{root}.POLICY_ACTION_NVECS", probe["POLICY_ACTION_NVECS"], list(g4["policy_action_nvecs"]["value"]))
        expect(f"{root}.ACTION_ENUM_LENGTHS", probe["ACTION_ENUM_LENGTHS"], list(g4["policy_action_nvecs"]["value"]))
        expect(
            f"{root}.MODEL_CONTRACT_METADATA_VERSION",
            probe["MODEL_CONTRACT_METADATA_VERSION"],
            g4["model_contract_metadata_version"]["value"],
        )
        if root == "flyff_farming_simulator":
            expect(f"{root}.RAW_OBSERVATION_SIZE", probe["RAW_OBSERVATION_SIZE"], g4["raw_observation_size"]["value"])
            expect(f"{root}.SIDECAR_SIZE", probe["SIDECAR_SIZE"], g4["sidecar_size"]["value"])
            expect(f"{root}.POLICY_INPUT_SIZE", probe["POLICY_INPUT_SIZE"], g4["policy_input_size"]["value"])
            expect(
                f"{root}.SIDECAR_DECOMPOSITION",
                probe["TEMPORAL_SIDECAR_SIZE"] + probe["PREVIOUS_STEERING_SIDECAR_SIZE"],
                g4["sidecar_size"]["value"],
            )
            expect(
                f"{root}.POLICY_INPUT_DECOMPOSITION",
                probe["RAW_OBSERVATION_SIZE"] + probe["SIDECAR_SIZE"],
                g4["policy_input_size"]["value"],
            )
            expect(f"{root}.PHYSICS_VERSION", probe["MOVEMENT_PHYSICS_MODEL_ID"], g4["physics_version"]["value"])
            expect(
                f"{root}.LEGACY_PHYSICS_VERSION",
                probe["LEGACY_MOVEMENT_PHYSICS_MODEL_ID"],
                g4["legacy_physics_version"]["value"],
            )
            for const in ("PATH_LENGTH_CELLS_PER_TICK", "ONSET_TURN_RADIANS", "STEADY_TURN_RADIANS", "DEFAULT_SUBSTEPS"):
                expect(f"{root}.{const}", probe[const], g4["physics_constants"][const])

    # Recorder consumes dependency-free canonical metadata and defines no copy.
    recorder_imports = _ast_imports(
        repo / "flyff_farming_recorder/recorder/session.py",
        "farming.observation_contract",
        ("OBSERVATION_SCHEMA_ID", "OBSERVATION_SCHEMA_HASH"),
    )
    expect(
        "recorder.canonical_metadata_imports",
        recorder_imports,
        {
            "OBSERVATION_SCHEMA_ID": "_OBSERVATION_SCHEMA_ID",
            "OBSERVATION_SCHEMA_HASH": "_OBSERVATION_SCHEMA_HASH",
        },
    )
    recorder_literals = _ast_constants(
        repo / "flyff_farming_recorder/recorder/session.py",
        ("OBSERVATION_SCHEMA_ID", "OBSERVATION_SCHEMA_HASH"),
    )
    expect("recorder.schema_literal_copies", recorder_literals, {})
    return failures, evidence


# ---------------------------------------------------------------------------
# G11
# ---------------------------------------------------------------------------


def check_g11(repo: Path, fp: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    g11 = fp["g11"]
    failures: list[str] = []
    evidence: dict[str, Any] = {"hashes": {}, "pairs": {}, "blob_ids": {}}
    locations = list(g11["locations"])
    for name, pinned in g11["artifacts"].items():
        digests = []
        for loc in locations:
            path = repo / loc / name
            if not path.is_file():
                failures.append(f"G11 missing artifact: {loc}/{name}")
                digests.append(None)
                continue
            digest = sha256_file(path)
            size = path.stat().st_size
            evidence["hashes"][f"{loc}/{name}"] = digest
            digests.append(digest)
            if digest != pinned:
                failures.append(f"G11 {loc}/{name}: sha256={digest} pinned={pinned}")
            if size != g11["sizes"][name]:
                failures.append(f"G11 {loc}/{name}: size={size} pinned={g11['sizes'][name]}")
        identical = len(set(digests)) == 1 and digests[0] is not None
        evidence["pairs"][name] = identical
        if g11.get("pairs_must_be_byte_identical", True) and not identical:
            failures.append(f"G11 paired copies of {name} are not byte-identical")

    marker = repo / g11["marker"]
    evidence["marker_present"] = marker.is_file()
    if not marker.is_file():
        failures.append(f"G11 persistent-map marker missing: {g11['marker']}")

    # Supplementary only -- never replaces the raw-byte contract above.
    try:
        for loc in locations:
            for name in g11["artifacts"]:
                out = subprocess.run(
                    ["git", "-C", str(repo), "rev-parse", f"HEAD:{loc}/{name}"],
                    capture_output=True, text=True,
                )
                if out.returncode == 0:
                    evidence["blob_ids"][f"{loc}/{name}"] = out.stdout.strip()
    except OSError:
        pass
    return failures, evidence


# ---------------------------------------------------------------------------
# G10a
# ---------------------------------------------------------------------------


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickle gymnasium spaces without importing repo or torch modules."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith(_SPACE_ALLOWED_PREFIXES):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused import of {module}.{name}")


_STRING_OPS = {
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "SHORT_BINSTRING", "BINSTRING", "STRING",
}


def _pickle_globals(blob: bytes) -> list[tuple[str, str]]:
    """Recover (module, qualname) pairs by reading opcodes -- never executing.

    STACK_GLOBAL takes its two operands from the stack, and either operand may
    arrive via the MEMO rather than as a literal. A scanner that only watches
    literal strings silently mis-pairs those cases (observed for
    ``farming.sb3_training.TrainingBoundaryKind``, whose module string is
    reused through ``BINGET``), so the memo is tracked here.
    """
    pairs: list[tuple[str, str]] = []
    recent: list[str] = []          # operand-eligible string values, in push order
    memo: dict[int, str | None] = {}
    memo_index = 0
    last: str | None = None         # most recently produced value

    for opcode, arg, _pos in pickletools.genops(blob):
        name = opcode.name
        if name in _STRING_OPS:
            value = arg.decode("utf-8", "replace") if isinstance(arg, (bytes, bytearray)) else str(arg)
            last = value
            recent.append(value)
        elif name == "MEMOIZE":
            memo[memo_index] = last
            memo_index += 1
        elif name in {"BINPUT", "LONG_BINPUT", "PUT"}:
            try:
                memo[int(arg)] = last
            except (TypeError, ValueError):
                pass
        elif name in {"BINGET", "LONG_BINGET", "GET"}:
            try:
                last = memo.get(int(arg))
            except (TypeError, ValueError):
                last = None
            if isinstance(last, str):
                recent.append(last)
        elif name == "STACK_GLOBAL":
            if len(recent) >= 2:
                pairs.append((recent[-2], recent[-1]))
                del recent[-2:]
            last = None
        elif name in {"GLOBAL", "INST"}:
            if isinstance(arg, str) and " " in arg:
                module, qualname = arg.split(" ", 1)
                pairs.append((module, qualname))
            last = None
        else:
            last = None
        if len(recent) > 16:
            del recent[:-16]
    return pairs


def _decode_field(field: Any) -> bytes | None:
    import base64

    if isinstance(field, dict) and ":serialized:" in field:
        return base64.b64decode(field[":serialized:"])
    return None


def _space_descriptor(field: Any) -> dict[str, Any]:
    blob = _decode_field(field)
    if blob is None:
        return {}
    globals_ = _pickle_globals(blob)
    cls = next(((m, n) for m, n in globals_ if m.startswith("gymnasium.")), None)
    out: dict[str, Any] = {"type": f"{cls[0]}.{cls[1]}" if cls else "UNKNOWN"}
    try:
        space = _RestrictedUnpickler(io.BytesIO(blob)).load()
    except Exception as error:  # noqa: BLE001 - recorded, never raised
        out["decode_error"] = f"{type(error).__name__}:{error}"
        return out
    shape = getattr(space, "shape", None)
    if shape is not None:
        out["shape"] = list(shape)
    dtype = getattr(space, "dtype", None)
    if dtype is not None:
        out["dtype"] = str(getattr(dtype, "str", dtype)).lstrip("<>=|")
    if hasattr(space, "nvec"):
        out["nvec"] = [int(v) for v in space.nvec.tolist()]
        out["spec"] = f"nvec={out['nvec']}"
    elif hasattr(space, "n"):
        out["n"] = int(space.n)
        out["spec"] = f"n={out['n']}"
    if hasattr(space, "start"):
        try:
            out["start"] = int(space.start)
        except (TypeError, ValueError):
            pass
    low, high = getattr(space, "low", None), getattr(space, "high", None)
    if low is not None and high is not None:
        try:
            out["low_min"] = float(low.min())
            out["high_max"] = float(high.max())
        except (AttributeError, ValueError):
            pass
    return out


def _repo_local_modules(repo: Path, roots: Sequence[str]) -> Any:
    def is_local(module: str) -> bool:
        parts = module.split(".")
        for root in roots:
            stem = repo / root / Path(*parts)
            if stem.with_suffix(".py").is_file() or (stem / "__init__.py").is_file():
                return True
        return False

    return is_local


def read_checkpoint(path: Path, is_local: Any) -> dict[str, Any]:
    """Extract all G10a fields from one checkpoint ZIP without importing it."""
    raw = path.read_bytes()
    record: dict[str, Any] = {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        record["sb3_version"] = (
            archive.read("_stable_baselines3_version").decode().strip()
            if "_stable_baselines3_version" in names else "ABSENT"
        )
        data = json.loads(archive.read("data"))

    policy_blob = _decode_field(data.get("policy_class"))
    policy_globals = _pickle_globals(policy_blob) if policy_blob else []
    if policy_globals:
        record["policy_class_module"], record["policy_class_qualname"] = policy_globals[0]
    else:
        record["policy_class_module"] = record["policy_class_qualname"] = "UNKNOWN"

    obs = _space_descriptor(data.get("observation_space"))
    act = _space_descriptor(data.get("action_space"))
    record["obs_space_type"] = obs.get("type", "UNKNOWN")
    record["obs_space_shape"] = str(obs.get("shape", []))
    record["obs_space_dtype"] = obs.get("dtype", "UNKNOWN")
    record["action_space_type"] = act.get("type", "UNKNOWN")
    record["action_space_spec"] = act.get("spec", "UNKNOWN")
    record["_obs_full"] = obs
    record["_act_full"] = act

    meta = data.get("farming_contract_metadata")
    if isinstance(meta, dict) and "metadata_version" in meta:
        record["farming_contract_metadata_version"] = str(meta["metadata_version"])
    else:
        record["farming_contract_metadata_version"] = "ABSENT"
    record["_contract_metadata"] = meta if isinstance(meta, dict) else None

    module = record["policy_class_module"]
    record["loadable_under_current_source"] = (
        "yes_third_party_policy_class" if module.startswith("stable_baselines3.")
        else "yes_module_and_class_present" if is_local(module)
        else "unknown_policy_class_origin"
    )

    # Every serialized field may carry repo-local ABI references, not just
    # policy_class -- this is the Phase-0 discovery G10 must preserve.
    references: list[tuple[str, str]] = []
    for value in data.values():
        blob = _decode_field(value)
        if blob is None:
            continue
        for mod, name in _pickle_globals(blob):
            if is_local(mod) and (mod, name) not in references:
                references.append((mod, name))
    record["_references"] = sorted(references)

    kwargs = data.get("policy_kwargs")
    if isinstance(kwargs, dict):
        rendered = {k: v for k, v in kwargs.items() if not k.startswith(":")}
        record["_policy_kwargs"] = json.dumps(rendered, sort_keys=True)
        record["_net_arch"] = json.dumps(rendered.get("net_arch"), sort_keys=True) if "net_arch" in rendered else "ABSENT"
    else:
        record["_policy_kwargs"] = json.dumps(kwargs, sort_keys=True) if kwargs is not None else "ABSENT"
        record["_net_arch"] = "ABSENT"
    return record


PHASE0_FIELDS = (
    "size_bytes", "sha256", "policy_class_module", "policy_class_qualname", "sb3_version",
    "farming_contract_metadata_version", "obs_space_type", "obs_space_shape", "obs_space_dtype",
    "action_space_type", "action_space_spec", "loadable_under_current_source",
)

SUPPLEMENT_FIELDS = (
    "path", "checkpoint_sha256", "policy_kwargs_json", "net_arch_json", "obs_space_low_min",
    "obs_space_high_max", "action_space_start", "contract_hash", "contract_observation_schema_id",
    "contract_observation_schema_hash", "contract_observation_size", "serialized_repo_local_references",
    "provenance",
)


def check_g10a(repo: Path, fp: dict[str, Any], corpus: Path, write_supplement: bool) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    roots = ["foreground_vision_bot", "flyff_farming_simulator", "flyff_farming_recorder", "."]
    is_local = _repo_local_modules(repo, roots)

    with (repo / PHASE0_INVENTORY).open(encoding="utf-8", newline="") as handle:
        baseline = list(csv.DictReader(handle, delimiter="\t"))
    with (repo / PHASE0_REFERENCES).open(encoding="utf-8", newline="") as handle:
        baseline_refs = list(csv.DictReader(handle, delimiter="\t"))

    expected_count = fp["g10"]["checkpoint_count"]
    if len(baseline) != expected_count:
        failures.append(f"G10a Phase-0 inventory has {len(baseline)} rows; pinned {expected_count}")

    field_mismatches = 0
    regenerated_refs: list[tuple[str, str, str]] = []
    supplement_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for row in baseline:
        rel = row["path"]
        source = corpus / rel
        if not source.is_file():
            missing.append(rel)
            continue
        actual = read_checkpoint(source, is_local)
        for field in PHASE0_FIELDS:
            if str(actual[field]) != row[field]:
                field_mismatches += 1
                failures.append(f"G10a {rel} {field}: regenerated={actual[field]!r} phase0={row[field]!r}")
        refs = actual["_references"]
        if refs:
            regenerated_refs.extend((rel, mod, name) for mod, name in refs)
        else:
            regenerated_refs.append((rel, "NONE", "NONE"))
        meta = actual["_contract_metadata"] or {}
        supplement_rows.append({
            "path": rel,
            "checkpoint_sha256": actual["sha256"],
            "policy_kwargs_json": actual["_policy_kwargs"],
            "net_arch_json": actual["_net_arch"],
            "obs_space_low_min": actual["_obs_full"].get("low_min", "ABSENT"),
            "obs_space_high_max": actual["_obs_full"].get("high_max", "ABSENT"),
            "action_space_start": actual["_act_full"].get("start", "ABSENT"),
            "contract_hash": meta.get("contract_hash", "ABSENT"),
            "contract_observation_schema_id": meta.get("observation_schema_id", "ABSENT"),
            "contract_observation_schema_hash": meta.get("observation_schema_hash", "ABSENT"),
            "contract_observation_size": meta.get("observation_size", "ABSENT"),
            "serialized_repo_local_references": ";".join(f"{m}.{n}" for m, n in refs) or "NONE",
            "provenance": SUPPLEMENT_PROVENANCE,
        })

    if missing:
        failures.append(f"G10a {len(missing)} checkpoints absent from corpus, e.g. {missing[:3]}")

    baseline_ref_set = sorted((r["checkpoint_path"], r["module_reference"], r["qualname_if_found"]) for r in baseline_refs)
    regenerated_ref_set = sorted(regenerated_refs)
    if regenerated_ref_set != baseline_ref_set:
        only_new = sorted(set(regenerated_ref_set) - set(baseline_ref_set))[:5]
        only_old = sorted(set(baseline_ref_set) - set(regenerated_ref_set))[:5]
        failures.append(f"G10a module references differ: +{only_new} -{only_old}")
    if len(baseline_refs) != fp["g10"]["module_reference_rows"]:
        failures.append(f"G10a reference rows={len(baseline_refs)}; pinned {fp['g10']['module_reference_rows']}")

    required = set(fp["g10"]["required_repo_local_references"]["values"])
    present = {f"{m}.{n}" for _p, m, n in regenerated_ref_set if m != "NONE"}
    if not required.issubset(present):
        failures.append(f"G10a missing required repo-local ABI references: {sorted(required - present)}")

    if write_supplement:
        supplement_rows.sort(key=lambda r: (r["path"], r["checkpoint_sha256"]))
        with (repo / SUPPLEMENT).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUPPLEMENT_FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(supplement_rows)

    evidence = {
        "checkpoints_compared": len(baseline) - len(missing),
        "phase0_fields_per_row": len(PHASE0_FIELDS),
        "field_mismatches": field_mismatches,
        "regenerated_reference_rows": len(regenerated_ref_set),
        "phase0_reference_rows": len(baseline_refs),
        "reference_rows_equal": regenerated_ref_set == baseline_ref_set,
        "supplement_rows": len(supplement_rows),
        "supplement_written": write_supplement,
        "phase0_contained_policy_kwargs": "policy_kwargs" in (baseline[0] if baseline else {}),
        "phase0_contained_net_arch": "net_arch" in (baseline[0] if baseline else {}),
    }
    return failures, evidence


# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("g4", "g11", "g10a", "all"))
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--write-supplement", action="store_true")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    fp = load_fingerprints(repo)
    corpus = (args.corpus or Path(fp["g10"]["preserved_corpus"])).resolve()

    failures: list[str] = []
    payload: dict[str, Any] = {}
    if args.command in ("g4", "all"):
        f, e = check_g4(repo, fp)
        failures += f
        payload["g4"] = {"failures": f, "evidence": e}
    if args.command in ("g11", "all"):
        f, e = check_g11(repo, fp)
        failures += f
        payload["g11"] = {"failures": f, "evidence": e}
    if args.command in ("g10a", "all"):
        f, e = check_g10a(repo, fp, corpus, args.write_supplement)
        failures += f
        payload["g10a"] = {"failures": f, "evidence": e}

    payload["ok"] = not failures
    payload["failure_count"] = len(failures)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
