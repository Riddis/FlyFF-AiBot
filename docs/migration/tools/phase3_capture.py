"""Generate/check Phase-3 pre-migration behavior fixtures without product writes."""

from __future__ import annotations

import argparse
import base64
import csv
import dataclasses
import gzip
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

FORMAT_VERSION = "flyffrl-phase3-v1"
BASE_SHA = "82e908d6028d5869a6ff6d6bb27d5a2aeaaebc46"
RANDOM_OBSERVATIONS = 10_000
EDGE_OBSERVATIONS = 16
OBSERVATION_SIZE = 923
FIXTURE_REL = Path("tests/fixtures/migration")
MANIFEST_REL = Path("docs/migration/PHASE3_FIXTURE_MANIFEST.tsv")
SPEC_REL = Path("docs/migration/PHASE3_CAPTURE_SPEC.toml")
CONFIG_REL = Path("docs/migration/EFFECTIVE_CONFIG_BASELINE.json")
ARCHIVES = (
    ("flyff_farming_simulator/recordings/eva_only/SEND_TO_RIDDIMS_WetFartChan_20260804T013220.661349Z.zip", 5165181, "f791348deae2bd3b0086b4256a0c902403dbfc8a9ec9d3f66124f54bb0ea3d0c"),
    ("flyff_farming_simulator/recordings/eva_only/SEND_TO_RIDDIMS_WetFartChan_20260804T015033.737143Z.zip", 2838153, "bb68a68f27343221d5268a85bc14accbe4fb4e9ea00568977dbad36214c2a40a"),
    ("flyff_farming_simulator/recordings/eva_only/SEND_TO_RIDDIMS_WetFartChan_20260804T020125.838396Z.zip", 6110924, "c46c320a6b479e1fe4f4aae93b162672f72eddd25c5855f6b4bd589f4a17c496"),
    ("flyff_farming_simulator/recordings/eva_only/SEND_TO_RIDDIMS_poot_20260804T163533.542137Z.zip", 462321, "63b44bafcb687e5035d1af4256cb8deaa0465fda4b17a9bdd7a2ede8b5d0e287"),
    ("flyff_farming_simulator/recordings/eva_only/SEND_TO_RIDDIMS_poot_20260804T163750.567290Z.zip", 4342092, "902f2a014ac0524087157f1ec416c524b3bc304f345d7adb7fa91b97ba83f699"),
    ("flyff_farming_simulator/recordings/eva_only/SEND_TO_RIDDIMS_poot_20260804T164942.048891Z.zip", 6554870, "b377f2e8da92442bd8e0dca0a04f0c58f293d1c6c838d8bdafc01f3b1e061fcb"),
    ("flyff_farming_simulator/recordings/training/SEND_TO_RIDDIMS_Riddims_20260803T212218.573109Z.zip", 5425884, "27934e5167c8f4a03e7b376f2106c714b7e7d187ed96a080e49ce7e1ff7bfccb"),
    ("flyff_farming_simulator/recordings/training/SEND_TO_RIDDIMS_Riddims_20260805T172406.597517Z.zip", 6425583, "352e59177c2a9850c87116deeb8e2301fd6e22938468ffa313dc5db8614fe92c"),
)
MAP_LUT = np.asarray((-1.0, -0.75, -0.5, 0.0, 1.0), dtype=np.float32)
SENTINELS = (0, 1, 2, 15, 16, 9999, 10015)


def seed_for(suite_name: str) -> int:
    label = f"FlyffRL-Phase3-v1|{suite_name}".encode("utf-8")
    return int(hashlib.sha256(label).hexdigest()[:16], 16)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def gzip_bytes(value: bytes) -> bytes:
    return gzip.compress(value, compresslevel=9, mtime=0)


def typed_encode(value: Any) -> bytes:
    """Canonical, type-sensitive semantic encoding used for G7."""
    if value is None:
        return b"N"
    if isinstance(value, bool):
        return b"B1" if value else b"B0"
    if isinstance(value, int):
        raw = str(value).encode("ascii")
        return b"I" + struct.pack(">I", len(raw)) + raw
    if isinstance(value, float):
        return b"F" + struct.pack(">d", value)
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"S" + struct.pack(">I", len(raw)) + raw
    if isinstance(value, bytes):
        return b"Y" + struct.pack(">I", len(value)) + value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = f"{type(value).__module__}.{type(value).__qualname__}"
        fields = tuple((field.name, getattr(value, field.name)) for field in dataclasses.fields(value))
        return b"O" + typed_encode(name) + typed_encode(fields)
    if isinstance(value, list):
        return b"L" + struct.pack(">I", len(value)) + b"".join(typed_encode(item) for item in value)
    if isinstance(value, tuple):
        return b"T" + struct.pack(">I", len(value)) + b"".join(typed_encode(item) for item in value)
    if isinstance(value, Mapping):
        pairs = sorted(((typed_encode(key), key, item) for key, item in value.items()), key=lambda row: row[0])
        return b"M" + struct.pack(">I", len(pairs)) + b"".join(key_bytes + typed_encode(item) for key_bytes, _key, item in pairs)
    raise TypeError(f"Unsupported canonical type: {type(value)!r}")


def _float_bits(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def _typed_json(value: Any) -> Any:
    if isinstance(value, float):
        return {"float64_be": _float_bits(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "fields": {field.name: _typed_json(getattr(value, field.name)) for field in dataclasses.fields(value)}}
    if isinstance(value, tuple):
        return {"tuple": [_typed_json(item) for item in value]}
    if isinstance(value, list):
        return [_typed_json(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _typed_json(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    return value


def _source_hash(repo: Path, relative: str) -> str:
    return sha256_file(repo / relative)


def _base_case(index: int, rng: np.random.Generator) -> dict[str, Any]:
    count = int(rng.integers(0, 21))
    actors = []
    for actor_id in range(count):
        legacy = rng.uniform(-70.0, 70.0, size=2)
        direct = rng.uniform(-70.0, 70.0, size=2)
        geodesic = float("inf") if rng.random() < 0.08 else float(rng.uniform(0.0, 75.0))
        actors.append([actor_id + 1, float(legacy[0]), float(legacy[1]), float(direct[0]), float(direct[1]), geodesic, int(rng.integers(-1, 2)), bool(rng.random() > 0.08)])
    player = [
        float(rng.uniform(-1.5, 1.5)), float(rng.uniform(-1.5, 1.5)),
        float(rng.uniform(-4.0 * math.pi, 4.0 * math.pi)), float(rng.uniform(-0.2, 1.2)),
        float(rng.uniform(0.0, 8.0)), bool(rng.integers(0, 2)),
        None if rng.random() < 0.25 else int(rng.integers(0, 3)), int(rng.integers(0, 5)),
        float(rng.uniform(-0.2, 1.2)), bool(rng.integers(0, 2)),
    ]
    local = rng.integers(0, len(MAP_LUT), size=121, dtype=np.uint8).tobytes()
    context = rng.integers(0, len(MAP_LUT), size=441, dtype=np.uint8).tobytes()
    return {"i": index, "name": f"random_{index:05d}", "p": player, "a": actors, "l": local, "c": context}


def _edge_cases(start: int) -> list[dict[str, Any]]:
    radius = 8.0
    below = float(np.nextafter(radius, -np.inf))
    above = float(np.nextafter(radius, np.inf))
    diagonal = radius / math.sqrt(2.0)
    diagonal_below = float(np.nextafter(diagonal, -np.inf))
    diagonal_above = float(np.nextafter(diagonal, np.inf))
    names_and_positions = [
        ("zero_actors", []), ("one_self_actor", [(0.0, 0.0)]),
        ("many_actors", [(float(i % 9), float(i // 9)) for i in range(45)]),
        ("dead_actors_filtered", [(1.0, 1.0), (2.0, 2.0)]),
        ("eva_axis_exact", [(0.0, 0.0), (radius, 0.0)]),
        ("eva_axis_nextbelow", [(0.0, 0.0), (below, 0.0)]),
        ("eva_axis_nextabove", [(0.0, 0.0), (above, 0.0)]),
        ("eva_diagonal_exact", [(0.0, 0.0), (diagonal, diagonal)]),
        ("eva_diagonal_nextbelow", [(0.0, 0.0), (diagonal_below, diagonal_below)]),
        ("eva_diagonal_nextabove", [(0.0, 0.0), (diagonal_above, diagonal_above)]),
        ("negative_quadrant_boundary", [(0.0, 0.0), (-diagonal, -diagonal)]),
        ("translated_bucket_boundary", [(7.999999999999, -8.0), (15.999999999999, -8.0)]),
        ("actor_order_forward", [(0.0, 0.0), (2.0, 1.0), (7.0, 0.0)]),
        ("actor_order_reverse", [(7.0, 0.0), (2.0, 1.0), (0.0, 0.0)]),
        ("blocked_far_actor", [(0.0, 0.0), (25.0, 0.0)]),
        ("normalization_extremes", [(-100.0, 100.0), (100.0, -100.0)]),
    ]
    result = []
    for offset, (name, positions) in enumerate(names_and_positions):
        actors = []
        for actor_index, (x, y) in enumerate(positions):
            actors.append([actor_index + 1, x, y, x, y, abs(x) + abs(y), -1 if name == "blocked_far_actor" else 1, not (name == "dead_actors_filtered" and actor_index == 1)])
        if name == "actor_order_reverse":
            actors = list(reversed(actors))
        player = [2.0 if name == "normalization_extremes" else 0.0, -2.0 if name == "normalization_extremes" else 0.0, math.pi, 1.5, 12.0, True, 2, 3, -0.5, False]
        result.append({"i": start + offset, "name": name, "p": player, "a": actors, "l": bytes([4]) * 121, "c": bytes([4]) * 441})
    return result


def build_observation_corpus() -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed_for("observation-corpus"))
    rows = [_base_case(index, rng) for index in range(RANDOM_OBSERVATIONS)]
    rows.extend(_edge_cases(RANDOM_OBSERVATIONS))
    return rows


def _run(repo: Path, *args: str, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-I", str(Path(__file__).resolve()), *args]
    result = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"worker failed ({result.returncode}): {' '.join(args)}\n{result.stdout}\n{result.stderr}")
    return result


def _obs_worker(root: Path, input_path: Path, output_path: Path, meta_path: Path) -> None:
    sys.path.insert(0, str(root))
    from farming.actions import FarmingAction
    from farming.map_features import DirectPathState
    from farming.observation import ActorObservation, ObservationBuilder, ObservationFrame, PlayerObservation

    rows = msgpack.unpackb(gzip.decompress(input_path.read_bytes()), raw=False)
    builder = ObservationBuilder()
    finite = True
    with output_path.open("wb") as handle:
        for row in rows:
            player_values = row["p"]
            player = PlayerObservation(
                normalized_x=player_values[0], normalized_z=player_values[1], heading_radians=player_values[2],
                eva_cooldown_fraction=player_values[3], displacement_cells=player_values[4], contact=player_values[5],
                held_movement=None if player_values[6] is None else FarmingAction(player_values[6]),
                last_policy_action=FarmingAction(player_values[7]), jump_cooldown_fraction=player_values[8], map_available=player_values[9],
            )
            actors = tuple(ActorObservation(actor_id=item[0], legacy_dx_cells=item[1], legacy_dy_cells=item[2], direct_dx_cells=item[3], direct_dz_cells=item[4], geodesic_cells=item[5], direct_path=DirectPathState(item[6]), alive=item[7]) for item in row["a"])
            local = MAP_LUT[np.frombuffer(row["l"], dtype=np.uint8)]
            context = MAP_LUT[np.frombuffer(row["c"], dtype=np.uint8)]
            vector = builder.build_vector(ObservationFrame(player=player, actors=actors, local_map=local, context_map=context))
            if vector.shape != (OBSERVATION_SIZE,) or vector.dtype != np.float32:
                raise RuntimeError(f"invalid observation contract at row {row['i']}: {vector.shape} {vector.dtype}")
            finite = finite and bool(np.all(np.isfinite(vector)))
            handle.write(np.ascontiguousarray(vector, dtype="<f4").tobytes())
    meta_path.write_bytes(json_bytes({"count": len(rows), "dtype": "float32", "finite": finite, "shape": [OBSERVATION_SIZE]}))


def capture_observations(repo: Path, temporary: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    rows = build_observation_corpus()
    packed = msgpack.packb(rows, use_bin_type=True)
    input_bytes = gzip_bytes(packed)
    input_path = temporary / "observation_inputs.msgpack.gz"
    input_path.write_bytes(input_bytes)
    outputs = []
    metas = []
    for kind, root_rel in (("bot", "foreground_vision_bot"), ("simulator", "flyff_farming_simulator")):
        output = temporary / f"observation_{kind}.f32"
        meta = temporary / f"observation_{kind}.json"
        _run(repo, "_obs_worker", str(repo / root_rel), str(input_path), str(output), str(meta))
        outputs.append(output)
        metas.append(json.loads(meta.read_text(encoding="utf-8")))
    bot = np.memmap(outputs[0], mode="r", dtype="<f4", shape=(len(rows), OBSERVATION_SIZE))
    simulator = np.memmap(outputs[1], mode="r", dtype="<f4", shape=(len(rows), OBSERVATION_SIZE))
    equal_rows = [np.array_equal(bot[index], simulator[index]) for index in range(len(rows))]
    differing = [index for index, equal in enumerate(equal_rows) if not equal]
    first_differences = []
    for row_index in differing[:32]:
        columns = np.flatnonzero(bot[row_index] != simulator[row_index])
        column = int(columns[0])
        first_differences.append({"row": int(row_index), "name": rows[row_index]["name"], "column": column, "bot_bits": bytes(bot[row_index, column : column + 1]).hex(), "simulator_bits": bytes(simulator[row_index, column : column + 1]).hex()})
    row_hashes = []
    aggregate = hashlib.sha256()
    for index in range(len(rows)):
        raw = np.ascontiguousarray(bot[index], dtype="<f4").tobytes()
        row_hashes.append(hashlib.sha256(raw).hexdigest())
        aggregate.update(raw)
    sentinels = {str(index): base64.b64encode(np.ascontiguousarray(bot[index], dtype="<f4").tobytes()).decode("ascii") for index in SENTINELS}
    expected = {
        "fixture_format_version": FORMAT_VERSION, "count": len(rows), "random_count": RANDOM_OBSERVATIONS,
        "edge_count": EDGE_OBSERVATIONS, "shape": [OBSERVATION_SIZE], "dtype": "<f4", "finite": all(meta["finite"] for meta in metas),
        "input_corpus_sha256": sha256_bytes(input_bytes), "expected_row_sha256": row_hashes,
        "aggregate_output_sha256": aggregate.hexdigest(), "sentinel_rows_base64_le_f4": sentinels,
        "cross_implementation_equal_count": len(rows) - len(differing), "cross_implementation_differing_count": len(differing),
        "differing_rows": differing, "first_differences": first_differences,
    }
    return {"observation_inputs.msgpack.gz": input_bytes, "observation_expected.json": json_bytes(expected)}, expected


def build_boundary_cases() -> list[dict[str, Any]]:
    radius = 8.0
    below = float(np.nextafter(radius, -np.inf)); above = float(np.nextafter(radius, np.inf))
    cases: list[dict[str, Any]] = []
    def add(name: str, points: list[tuple[float, float]]) -> None:
        cases.append({"name": name, "positions": [[index + 1, x, y] for index, (x, y) in enumerate(points)]})
    add("zero", []); add("one_self", [(0.0, 0.0)])
    for axis, vector in (("x_pos", (1.0, 0.0)), ("x_neg", (-1.0, 0.0)), ("y_pos", (0.0, 1.0)), ("y_neg", (0.0, -1.0))):
        for label, distance in (("below", below), ("exact", radius), ("above", above)):
            add(f"{axis}_{label}", [(0.0, 0.0), (vector[0] * distance, vector[1] * distance)])
    diagonal = radius / math.sqrt(2.0)
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        for label, coordinate in (("below", float(np.nextafter(diagonal, -np.inf))), ("exact", diagonal), ("above", float(np.nextafter(diagonal, np.inf)))):
            add(f"diag_{sx}_{sy}_{label}", [(0.0, 0.0), (sx * coordinate, sy * coordinate)])
    add("bucket_translation_a", [(7.999999999999, -8.0), (15.999999999999, -8.0)])
    add("bucket_translation_b", [(-16.000000000001, 24.0), (-8.000000000001, 24.0)])
    points = [(0.0, 0.0), (2.0, 1.0), (7.0, 0.0), (9.0, -1.0)]
    add("order_forward", points); add("order_reverse", list(reversed(points)))
    rng = np.random.default_rng(seed_for("neighbour-boundary"))
    for index in range(4096):
        count = int(rng.integers(0, 33))
        points = [(float(rng.uniform(-40, 40)), float(rng.uniform(-40, 40))) for _ in range(count)]
        add(f"random_{index:04d}", points)
    return cases


def _nearby_worker(root: Path, cases_path: Path, output_path: Path) -> None:
    sys.path.insert(0, str(root))
    from farming.observation import ObservationBuilder
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    builder = ObservationBuilder()
    result = []
    for case in cases:
        positions = {int(item[0]): (float(item[1]), float(item[2])) for item in case["positions"]}
        result.append({"name": case["name"], "counts": builder._nearby_counts(positions)})
    output_path.write_bytes(json_bytes(result))


def capture_boundary(repo: Path, temporary: Path, observation: dict[str, Any]) -> bytes:
    cases = build_boundary_cases()
    cases_path = temporary / "boundary_cases.json"; cases_path.write_bytes(json_bytes(cases))
    results = []
    for kind, root_rel in (("bot", "foreground_vision_bot"), ("simulator", "flyff_farming_simulator")):
        output = temporary / f"boundary_{kind}.json"
        _run(repo, "_nearby_worker", str(repo / root_rel), str(cases_path), str(output))
        results.append(json.loads(output.read_text(encoding="utf-8")))
    mismatches = []
    for case, bot, simulator in zip(cases, results[0], results[1], strict=True):
        if bot["counts"] != simulator["counts"]:
            mismatches.append({"name": case["name"], "positions": case["positions"], "bot": bot["counts"], "simulator": simulator["counts"]})
    broader = [index for index in observation["differing_rows"] if index < RANDOM_OBSERVATIONS]
    payload = {
        "fixture_format_version": FORMAT_VERSION, "case_count": len(cases), "random_fuzz_count": 4096,
        "direct_mismatch_count": len(mismatches), "direct_mismatches": mismatches,
        "full_vector_differing_rows": observation["differing_rows"], "full_vector_first_differences": observation["first_differences"],
        "broader_random_observation_divergence": broader,
        "classification": "KNOWN_HYPOT_VS_SQUARED_ONLY" if mismatches and not broader else ("BIT_EQUIVALENT" if not mismatches and not observation["differing_rows"] else "UNEXPLAINED_BROADER_DIVERGENCE"),
        "phase4_constraint": "simulator optimized _nearby_counts must not silently become canonical; preserve the bit-level live behavior" if mismatches and not broader else None,
    }
    return json_bytes(payload)


def _geodesic_worker(root: Path, output_path: Path) -> None:
    sys.path.insert(0, str(root))
    from farming.map_features import FarmingMapFeatures
    def feature(kind: str, size: int) -> Any:
        safe = np.ones((size, size), dtype=bool)
        if kind == "corner_cut": safe[1, 2] = False; safe[2, 1] = False
        elif kind == "narrow": safe[:, 12] = False; safe[12, 12] = True
        elif kind == "disconnected": safe[:, 8] = False
        return FarmingMapFeatures(traversable=safe, forbidden=np.zeros_like(safe), safe_traversable=safe)
    declared = []
    fixed = [
        ("ordinary", "open", 17, (1, 1), (10, 10), 30.0, 256),
        ("blocked_start", "corner_cut", 17, (2, 1), (5, 5), 30.0, 256),
        ("blocked_goal", "corner_cut", 17, (1, 1), (2, 1), 30.0, 256),
        ("out_of_map", "open", 17, (-1, 0), (2, 2), 30.0, 256),
        ("corner_cut", "corner_cut", 17, (1, 1), (2, 2), 30.0, 256),
        ("narrow", "narrow", 25, (4, 12), (20, 12), 40.0, 256),
        ("unreachable", "disconnected", 17, (2, 8), (14, 8), 40.0, 256),
        ("distance_exact", "open", 17, (1, 1), (9, 1), 8.0, 256),
        ("distance_outside", "open", 17, (1, 1), (10, 1), float(np.nextafter(8.0, np.inf)), 256),
    ]
    for expansions in (1, 2, 8, 32, 256):
        fixed.append((f"expansion_{expansions}", "open", 17, (1, 1), (12, 12), 30.0, expansions))
    rng = np.random.default_rng(seed_for("bounded-geodesic"))
    for index in range(512):
        size = 25; start = (int(rng.integers(0, size)), int(rng.integers(0, size))); goal = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        fixed.append((f"random_{index:03d}", "narrow", size, start, goal, 40.0, 2048))
    mismatches = []
    for name, kind, size, start, goal, limit, expansions in fixed:
        features = feature(kind, size)
        field = features.bounded_geodesic_field(start, maximum_distance_cells=limit, maximum_expansions=expansions)
        field_repeat = features.bounded_geodesic_field(start, maximum_distance_cells=limit, maximum_expansions=expansions)
        point = features.geodesic_distance(start, goal, maximum_distance_cells=limit, maximum_expansions=expansions)
        field_value = field.get(goal, math.inf)
        equal = field_value == point
        row = {"name": name, "map": kind, "size": size, "start": start, "goal": goal, "maximum_distance_cells_bits": _float_bits(limit), "maximum_expansions": expansions, "field_value_bits": _float_bits(field_value), "point_value_bits": _float_bits(point), "equal": equal, "cache_same_object": field_repeat is field}
        declared.append(row)
        if not equal: mismatches.append(row)
    output_path.write_bytes(json_bytes({"fixture_format_version": FORMAT_VERSION, "comparison_count": len(declared), "exact_match_count": len(declared) - len(mismatches), "mismatch_count": len(mismatches), "mismatches": mismatches, "cases": declared, "status": "PASS" if not mismatches else "BLOCKING_NON_EQUIVALENCE"}))


def capture_geodesic(repo: Path, temporary: Path) -> bytes:
    output = temporary / "geodesic.json"
    _run(repo, "_geodesic_worker", str(repo / "flyff_farming_simulator"), str(output))
    return output.read_bytes()


def _encode_bool_array(array: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(array, dtype=np.bool_)
    raw = value.tobytes(order="C")
    packed = np.packbits(value.reshape(-1), bitorder="big").tobytes()
    return {"shape": list(value.shape), "dtype": "bool", "c_order_sha256": sha256_bytes(raw), "packbits_big_base64": base64.b64encode(packed).decode("ascii")}


def _derived_content_hash(arrays: Mapping[str, np.ndarray], source_bounds: Any, grid_origin: int) -> str:
    digest = hashlib.sha256(json_bytes({"source_bounds": list(source_bounds), "grid_origin": int(grid_origin)}))
    for name in ("traversable", "forbidden", "safe_traversable"):
        digest.update(name.encode("ascii")); digest.update(np.ascontiguousarray(arrays[name], dtype=np.bool_).tobytes())
    return digest.hexdigest()


def _map_worker(kind: str, root: Path, output_path: Path) -> None:
    sys.path.insert(0, str(root))
    if kind == "live":
        from farming.map_context import FarmingMapContext
        loaded = FarmingMapContext.load("Tower AoE")
        arrays = {"traversable": loaded.features.traversable, "forbidden": loaded.features.forbidden, "safe_traversable": loaded.features.safe_traversable}
        source_bounds = loaded.source_bounds; grid_origin = loaded.grid_origin
        coordinate = dataclasses.asdict(loaded.coordinate_frame)
        runtime_hash = loaded.content_hash; radius = 2
    else:
        from simulator.map_model import MapModel
        loaded = MapModel.load()
        arrays = {"traversable": loaded.traversable, "forbidden": loaded.forbidden, "safe_traversable": loaded.features.safe_traversable}
        source_bounds = loaded.source_bounds; grid_origin = loaded.grid_origin
        coordinate = {"origin_native_x": loaded.origin_native_x, "origin_native_z": loaded.origin_native_z, "native_units_per_cell": loaded.native_units_per_cell}
        runtime_hash = None; radius = 0
    payload = {"fixture_format_version": FORMAT_VERSION, "loader": kind, "obstacle_radius_cells": radius, "teleport_radius_cells": 2, "source_bounds": list(source_bounds), "grid_origin": int(grid_origin), "coordinate_frame": _typed_json(coordinate), "runtime_content_hash": runtime_hash, "content_hash": _derived_content_hash(arrays, source_bounds, grid_origin), "arrays": {name: _encode_bool_array(value) for name, value in arrays.items()}, "counts": {name: int(np.count_nonzero(value)) for name, value in arrays.items()}}
    output_path.write_bytes(json_bytes(payload))


def _decode_bool_array(value: Mapping[str, Any]) -> np.ndarray:
    size = math.prod(value["shape"])
    packed = np.frombuffer(base64.b64decode(value["packbits_big_base64"]), dtype=np.uint8)
    return np.unpackbits(packed, bitorder="big")[:size].astype(bool).reshape(value["shape"])


def capture_maps(repo: Path, temporary: Path) -> dict[str, bytes]:
    g11 = subprocess.run([sys.executable, str(repo / "docs/migration/tools/phase2_fingerprints.py"), "g11", "--repo", str(repo)], cwd=repo, text=True, capture_output=True, check=False)
    if g11.returncode:
        raise RuntimeError(f"G11 precondition failed\n{g11.stdout}\n{g11.stderr}")
    payloads = {}
    for kind, root_rel in (("live", "foreground_vision_bot"), ("simulator", "flyff_farming_simulator")):
        output = temporary / f"map_{kind}.json"; _run(repo, "_map_worker", kind, str(repo / root_rel), str(output)); payloads[kind] = json.loads(output.read_text(encoding="utf-8"))
    live_safe = _decode_bool_array(payloads["live"]["arrays"]["safe_traversable"])
    sim_safe = _decode_bool_array(payloads["simulator"]["arrays"]["safe_traversable"])
    aligned = live_safe.shape == sim_safe.shape and payloads["live"]["source_bounds"] == payloads["simulator"]["source_bounds"]
    diagnostic = {"fixture_format_version": FORMAT_VERSION, "label": "DIAGNOSTIC ONLY - NOT A MIGRATION PASS/FAIL GATE - NOT AUTHORIZATION TO UNIFY RADII", "live_obstacle_radius_cells": 2, "simulator_obstacle_radius_cells": 0, "live_counts": payloads["live"]["counts"], "simulator_counts": payloads["simulator"]["counts"], "same_shape_and_source_bounds": aligned, "safe_mask_xor_count": int(np.count_nonzero(live_safe ^ sim_safe)) if aligned else None, "live_source_bounds": payloads["live"]["source_bounds"], "simulator_source_bounds": payloads["simulator"]["source_bounds"], "live_coordinate_frame": payloads["live"]["coordinate_frame"], "simulator_coordinate_frame": payloads["simulator"]["coordinate_frame"]}
    return {"map_live.json": json_bytes(payloads["live"]), "map_simulator.json": json_bytes(payloads["simulator"]), "map6_diagnostic.json": json_bytes(diagnostic)}


def _router_worker(root: Path, output_path: Path) -> None:
    sys.path.insert(0, str(root))
    from simulator.kinodynamic_route_planner import TargetPersistenceController, annotate_route_edges, plan_route, select_persistent_waypoint
    from simulator.map_model import MapModel
    from simulator.movement_kernel import SteeringDirection, advance_player_tick, arc_endpoint_local, arc_endpoint_world, resolve_signed_turn_radians
    size = 61; center = size // 2
    def make(kind: str) -> Any:
        array = np.ones((size, size), dtype=bool)
        if kind == "single_wall": array[center - 8:center + 9, center + 8:center + 11] = False
        elif kind == "two_wall": array[center:center + 10, center + 6:center + 8] = False; array[center - 10:center, center + 16:center + 18] = False
        elif kind == "narrow": array[center - 15:center - 1, center + 8:center + 10] = False; array[center + 2:center + 15, center + 8:center + 10] = False
        return MapModel.from_arrays(array)
    route_specs = (("open_straight", "open", 20, 0), ("open_left", "open", 15, 8), ("open_right", "open", 15, -8), ("single_wall_detour", "single_wall", 20, 0), ("two_wall_s", "two_wall", 26, 0), ("narrow_passage", "narrow", 18, 0))
    routes = []
    route_objects = {}
    map_objects = {}
    for name, kind, dx, dy in route_specs:
        model = make(kind); start = model.layout_to_native(center, center); destination = model.layout_to_native(center + dx, center + dy); stats = {}
        route = plan_route(model, start_x=start[0], start_z=start[1], start_heading=0.0, destination_x=destination[0], destination_z=destination[1], max_distance_cells=80.0, stats=stats)
        waypoint = select_persistent_waypoint(model, route, player_x=start[0], player_z=start[1], heading=0.0)
        routes.append({"name": name, "kind": kind, "start": start, "destination": destination, "stats": stats, "route": route, "edges": annotate_route_edges(model, route), "waypoint": waypoint})
        route_objects[name] = route; map_objects[name] = (model, start, destination)
    movement = []
    open_model = make("open")
    for heading in (0.0, 0.7, -2.2):
        for scale in (0.0, 0.5, 1.0):
            for previous in SteeringDirection:
                for current in SteeringDirection:
                    turn = resolve_signed_turn_radians(current, previous)
                    movement.append({"heading": heading, "scale": scale, "previous": previous.name, "current": current.name, "signed_turn": turn, "local_endpoint": arc_endpoint_local(2.738491 * scale, turn * scale), "world_endpoint": arc_endpoint_world(0.0, 0.0, heading, 2.738491 * scale, turn * scale, 1.6), "advance": advance_player_tick(open_model, 0.0, 0.0, heading, previous, current, distance_scale=scale)})
    collision_model = make("single_wall")
    for name, steering in (("straight_into_wall", SteeringDirection.NONE), ("left_near_wall", SteeringDirection.LEFT), ("right_near_wall", SteeringDirection.RIGHT)):
        position = collision_model.layout_to_native(center + 6, center)
        movement.append({"name": name, "previous": SteeringDirection.NONE.name, "current": steering.name, "advance": advance_player_tick(collision_model, position[0], position[1], 0.0, SteeringDirection.NONE, steering)})
    model, start, destination = map_objects["single_wall_detour"]; route = route_objects["single_wall_detour"]
    midpoint = route[len(route) // 2]
    replan_stats = {}
    replan = plan_route(model, start_x=midpoint.x, start_z=midpoint.z, start_heading=midpoint.heading, destination_x=destination[0], destination_z=destination[1], max_distance_cells=80.0, stats=replan_stats)
    routes.append({"name": "replan_from_single_wall_route_midpoint", "kind": "single_wall", "start": (midpoint.x, midpoint.z), "destination": destination, "stats": replan_stats, "route": replan, "edges": annotate_route_edges(model, replan), "waypoint": select_persistent_waypoint(model, replan, player_x=midpoint.x, player_z=midpoint.z, heading=midpoint.heading)})
    def controller_case(name: str, previous: tuple[float, float] | None, candidate: tuple[float, float], player: tuple[float, float]) -> dict[str, Any]:
        controller = TargetPersistenceController(model, destination[0], destination[1]); controller.previous_target = previous
        target = controller.update(candidate, player_x=player[0], player_z=player[1], route=route)
        return {"name": name, "previous_target": previous, "candidate": candidate, "player": player, "target": target, "reason": None if controller.last_switch_reason is None else controller.last_switch_reason.value, "switches": controller.target_switches, "locked": controller.locked_onto_final}
    first = (route[1].x, route[1].z); later = (route[min(3, len(route)-1)].x, route[min(3, len(route)-1)].z)
    controller_rows = [
        controller_case("INITIAL", None, first, start),
        controller_case("KEEP_CURRENT", first, first, start),
        controller_case("CURRENT_UNSAFE", destination, first, start),
        controller_case("CURRENT_REACHED_OR_PASSED", start, later, start),
        controller_case("BETTER_FORWARD_TARGET", first, later, start),
        controller_case("FINAL_TARGET_LOCK", first, later, destination),
    ]
    payload = {"fixture_format_version": FORMAT_VERSION, "route_case_count": len(routes), "movement_case_count": len(movement), "controller_case_count": len(controller_rows), "routes": routes, "movement": movement, "controller": controller_rows}
    output_path.write_bytes(json_bytes(_typed_json(payload)))


def capture_router(repo: Path, temporary: Path) -> bytes:
    output = temporary / "router.json"; _run(repo, "_router_worker", str(repo / "flyff_farming_simulator"), str(output), timeout=900); return output.read_bytes()


def _config_worker(kind: str, root: Path, repo: Path, output_path: Path) -> None:
    sys.path.insert(0, str(root))
    from dataclasses import asdict
    from position.MonsterConfig import load_native_monster_config
    from position.PositionConfig import load_native_position_config
    prefix = "foreground_vision_bot" if kind == "bot" else "flyff_farming_recorder"
    def record(class_name: str, value: Any, source: str, loader: str) -> dict[str, Any]:
        return {"class": class_name, "effective": asdict(value), "loader_module": {"exists": True, "path": loader, "sha256": sha256_file(repo / loader)}, "source": {"exists": True, "path": source, "sha256": sha256_file(repo / source)}}
    monster_source = f"{prefix}/position/native_monsters.json"; position_source = f"{prefix}/position/native_position.json"
    result = {"import_root": prefix, "monster_config": record("position.MonsterConfig.NativeMonsterConfig", load_native_monster_config(repo / monster_source), monster_source, f"{prefix}/position/MonsterConfig.py"), "position_config": record("position.PositionConfig.NativePositionConfig", load_native_position_config(repo / position_source), position_source, f"{prefix}/position/PositionConfig.py")}
    if kind == "recorder":
        from recorder.config import RecorderConfig
        source = "flyff_farming_recorder/recorder_config.json"; loader = "flyff_farming_recorder/recorder/config.py"
        result["recorder_config"] = record("recorder.config.RecorderConfig", RecorderConfig.load(repo / source), source, loader)
    output_path.write_bytes(json_bytes(result))


def capture_config(repo: Path, temporary: Path) -> bytes:
    components = {}
    for kind, root_rel, key in (("bot", "foreground_vision_bot", "foreground_vision_bot"), ("recorder", "flyff_farming_recorder", "flyff_farming_recorder")):
        output = temporary / f"config_{kind}.json"; _run(repo, "_config_worker", kind, str(repo / root_rel), str(repo), str(output)); components[key] = json.loads(output.read_text(encoding="utf-8"))
    values = {"presence_clear_confirmation_samples": 3, "presence_cold_poll_batch_size": 1024, "presence_cold_verification_batch_size": 256, "presence_dead_read_grace_seconds": 2.0}
    current = {"components": components, "loading_isolation": {"cross_root_import_contamination": False, "flyff_farming_recorder_import_root": "flyff_farming_recorder", "foreground_vision_bot_import_root": "foreground_vision_bot", "method": "separate python -I subprocesses"}, "phase": "Phase 0", "presence_field_ownership": {"equal_effective_values": values, "flyff_farming_recorder": {"config_layer": "recorder.config.RecorderConfig", "owner": "flyff_farming_recorder/recorder_config.json", "recorder_position_monster_config_contains_presence_fields": False, "values": values}, "foreground_vision_bot": {"config_layer": "position.MonsterConfig.NativeMonsterConfig", "owner": "foreground_vision_bot/position/native_monsters.json", "values": values}, "ownership_conclusion": "Values are equal, but ownership is intentionally different; Phase 0 does not unify them."}, "purpose": "Effective resolved config baseline for the future position/ merge; preservation evidence only, not a unification."}
    baseline_bytes = (repo / CONFIG_REL).read_bytes(); baseline = json.loads(baseline_bytes)
    if current != baseline:
        raise RuntimeError("G9 current isolated effective configuration does not equal the authoritative Phase-0 baseline")
    return json_bytes({"fixture_format_version": FORMAT_VERSION, "authoritative_path": CONFIG_REL.as_posix(), "authoritative_sha256": sha256_bytes(baseline_bytes), "current_recomputation_equal": True, "ownership_difference_preserved": True})


def _stream_semantics(values: Iterable[Any], block_size: int = 1024) -> dict[str, Any]:
    overall = hashlib.sha256(b"FlyffRL-typed-stream-v1")
    blocks = []; block = hashlib.sha256(b"FlyffRL-typed-block-v1"); block_count = 0; count = 0
    for value in values:
        encoded = typed_encode(value); framed = struct.pack(">Q", len(encoded)) + encoded
        overall.update(framed); block.update(framed); count += 1; block_count += 1
        if block_count == block_size:
            blocks.append(block.hexdigest()); block = hashlib.sha256(b"FlyffRL-typed-block-v1"); block_count = 0
    if block_count or count == 0: blocks.append(block.hexdigest())
    return {"count": count, "sha256": overall.hexdigest(), "block_items": block_size, "block_sha256": blocks}


def _recording_worker(root: Path, archive_path: Path, output_path: Path) -> None:
    sys.path.insert(0, str(root))
    from simulator.schema import RecordingArchive
    archive = RecordingArchive(archive_path)
    manifest_hash = sha256_bytes(typed_encode(archive.manifest))
    frames = _stream_semantics(archive.frames()); events = _stream_semantics(archive.events()); inputs = _stream_semantics(archive.inputs())
    overall = sha256_bytes(typed_encode((manifest_hash, archive.quantum, frames, events, inputs)))
    output_path.write_bytes(json_bytes({"manifest_semantic_sha256": manifest_hash, "quantum_float64_be": _float_bits(archive.quantum), "sampling": _typed_json(archive.manifest.get("sampling")), "recorder_version": archive.manifest.get("recorder_version"), "frames": frames, "events": events, "inputs": inputs, "overall_decoded_semantic_sha256": overall}))


def capture_recordings(repo: Path, corpus: Path, temporary: Path) -> bytes:
    with (repo / "docs/migration/ARTIFACT_MANIFEST.tsv").open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = {row["path"]: (int(row["size_bytes"]), row["sha256"].lower()) for row in csv.DictReader(handle, delimiter="\t") if row["category"] == "recording_archive"}
    if set(manifest_rows) != {row[0] for row in ARCHIVES}:
        raise RuntimeError("Phase-0 artifact manifest recording-archive path set is not the preregistered all-eight set")
    output_rows = []
    for index, (relative, expected_size, expected_sha) in enumerate(ARCHIVES):
        if manifest_rows[relative] != (expected_size, expected_sha): raise RuntimeError(f"Phase-0 artifact manifest mismatch: {relative}")
        source = corpus / Path(relative)
        if not source.is_file() or source.stat().st_size != expected_size or sha256_file(source).lower() != expected_sha:
            raise RuntimeError(f"Phase-0 archive source mismatch: {relative}")
        output = temporary / f"recording_{index}.json"; _run(repo, "_recording_worker", str(repo / "flyff_farming_simulator"), str(source), str(output), timeout=900)
        row = json.loads(output.read_text(encoding="utf-8")); row.update({"path": relative, "size_bytes": expected_size, "source_sha256": expected_sha}); output_rows.append(row)
    return json_bytes({"fixture_format_version": FORMAT_VERSION, "archive_count": len(output_rows), "typed_encoding": "FlyffRL-typed-v1", "archives": output_rows})


def _fixture_metadata(repo: Path) -> dict[str, dict[str, str]]:
    obs_sources = ";".join(_source_hash(repo, path) for path in ("foreground_vision_bot/farming/observation.py", "flyff_farming_simulator/farming/observation.py"))
    return {
        "observation_inputs.msgpack.gz": {"gate": "G3-observation", "seed": "observation-corpus", "provenance": "two isolated observation implementations", "source": obs_sources, "role": "complete replayable 10016-case input corpus"},
        "observation_expected.json": {"gate": "G3-observation", "seed": "observation-corpus", "provenance": "two isolated observation implementations", "source": obs_sources, "role": "exact per-row and aggregate 923-vector evidence"},
        "neighbour_boundary.json": {"gate": "G3", "seed": "neighbour-boundary", "provenance": "ObservationBuilder._nearby_counts in bot and simulator", "source": obs_sources, "role": "boundary/fuzz classification and Phase-4 constraint"},
        "bounded_geodesic.json": {"gate": "geodesic", "seed": "bounded-geodesic", "provenance": "simulator FarmingMapFeatures point and field APIs", "source": _source_hash(repo, "flyff_farming_simulator/farming/map_features.py"), "role": "bounded field versus point-query equivalence"},
        "map_live.json": {"gate": "G12-live", "seed": "derived-maps fixed invocation", "provenance": "FarmingMapContext.load(Tower AoE), current defaults", "source": _source_hash(repo, "foreground_vision_bot/farming/map_context.py"), "role": "separate live radius-2 derived-map golden"},
        "map_simulator.json": {"gate": "G12-simulator", "seed": "derived-maps fixed invocation", "provenance": "MapModel.load(), current defaults", "source": _source_hash(repo, "flyff_farming_simulator/simulator/map_model.py"), "role": "separate simulator radius-0 derived-map golden"},
        "map6_diagnostic.json": {"gate": "MAP6-diagnostic-only", "seed": "derived-maps fixed invocation", "provenance": "mechanical comparison of separate G12 captures", "source": "", "role": "non-gating radius-difference diagnostic"},
        "router_kernel.json": {"gate": "G8c", "seed": "router-kernel", "provenance": "fixed synthetic routing/controller/kernel sweep", "source": _source_hash(repo, "flyff_farming_simulator/simulator/kinodynamic_route_planner.py") + ";" + _source_hash(repo, "flyff_farming_simulator/simulator/movement_kernel.py"), "role": "migration continuity, not renewed scientific qualification"},
        "effective_config.json": {"gate": "G9.1", "seed": "effective-config exact Phase-0 replay", "provenance": CONFIG_REL.as_posix(), "source": _source_hash(repo, CONFIG_REL.as_posix()), "role": "authoritative baseline provenance and current equality"},
        "recordings.json": {"gate": "G7", "seed": "recording-archives fixed all-eight list", "provenance": "read-only external Phase-0 snapshot", "source": ";".join(row[2] for row in ARCHIVES), "role": "typed ordered semantic archive baselines"},
    }


def manifest_bytes(repo: Path, fixture_bytes: Mapping[str, bytes]) -> bytes:
    metadata = _fixture_metadata(repo)
    header = "path\tsize_bytes\tsha256\tgate\tgenerator_tool\tfixture_format_version\tseed_or_case_spec\tsource_provenance\tsource_sha256\tsemantic_role\n"
    rows = []
    for name in sorted(fixture_bytes):
        item = metadata[name]; path = (FIXTURE_REL / name).as_posix()
        rows.append("\t".join((path, str(len(fixture_bytes[name])), sha256_bytes(fixture_bytes[name]), item["gate"], "docs/migration/tools/phase3_capture.py", FORMAT_VERSION, item["seed"], item["provenance"], item["source"], item["role"])))
    return (header + "\n".join(rows) + "\n").encode("utf-8")


def generate_all(repo: Path, corpus: Path, output_root: Path, manifest_path: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True); manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flyffrl-phase3-work-") as raw:
        temporary = Path(raw)
        fixtures, observation = capture_observations(repo, temporary)
        fixtures["neighbour_boundary.json"] = capture_boundary(repo, temporary, observation)
        fixtures["bounded_geodesic.json"] = capture_geodesic(repo, temporary)
        fixtures.update(capture_maps(repo, temporary))
        fixtures["router_kernel.json"] = capture_router(repo, temporary)
        fixtures["effective_config.json"] = capture_config(repo, temporary)
        fixtures["recordings.json"] = capture_recordings(repo, corpus, temporary)
    for name, data in fixtures.items(): (output_root / name).write_bytes(data)
    manifest = manifest_bytes(repo, fixtures); manifest_path.write_bytes(manifest)
    return {"fixture_count": len(fixtures), "fixture_bytes": sum(map(len, fixtures.values())), "manifest_sha256": sha256_bytes(manifest), "fixtures": {name: sha256_bytes(data) for name, data in sorted(fixtures.items())}}


def check_all(repo: Path, corpus: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="flyffrl-phase3-check-") as raw:
        root = Path(raw); candidate_fixtures = root / FIXTURE_REL; candidate_manifest = root / MANIFEST_REL
        summary = generate_all(repo, corpus, candidate_fixtures, candidate_manifest)
        expected_names = sorted(path.name for path in (repo / FIXTURE_REL).iterdir() if path.is_file())
        candidate_names = sorted(path.name for path in candidate_fixtures.iterdir() if path.is_file())
        if expected_names != candidate_names: raise RuntimeError(f"fixture path-set mismatch: expected={expected_names}, candidate={candidate_names}")
        mismatches = [name for name in expected_names if (repo / FIXTURE_REL / name).read_bytes() != (candidate_fixtures / name).read_bytes()]
        if mismatches: raise RuntimeError(f"fixture byte mismatch: {mismatches}")
        if (repo / MANIFEST_REL).read_bytes() != candidate_manifest.read_bytes(): raise RuntimeError("fixture manifest byte mismatch")
        return {**summary, "check": "PASS", "byte_identical": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "check"):
        command = sub.add_parser(name); command.add_argument("--repo", type=Path, default=Path.cwd()); command.add_argument("--corpus", type=Path, required=True)
        if name == "generate": command.add_argument("--output-root", type=Path); command.add_argument("--manifest", type=Path)
    for worker, count in (("_obs_worker", 4), ("_nearby_worker", 3), ("_geodesic_worker", 2), ("_map_worker", 3), ("_router_worker", 2), ("_config_worker", 4), ("_recording_worker", 3)):
        command = sub.add_parser(worker); command.add_argument("args", nargs=count)
    args = parser.parse_args()
    if args.command.startswith("_"):
        values = [Path(value) for value in args.args]
        if args.command == "_obs_worker": _obs_worker(*values)
        elif args.command == "_nearby_worker": _nearby_worker(*values)
        elif args.command == "_geodesic_worker": _geodesic_worker(*values)
        elif args.command == "_map_worker": _map_worker(args.args[0], Path(args.args[1]), Path(args.args[2]))
        elif args.command == "_router_worker": _router_worker(*values)
        elif args.command == "_config_worker": _config_worker(args.args[0], Path(args.args[1]), Path(args.args[2]), Path(args.args[3]))
        elif args.command == "_recording_worker": _recording_worker(*values)
        return 0
    repo = args.repo.resolve(); corpus = args.corpus.resolve()
    if args.command == "generate":
        output_root = (args.output_root or repo / FIXTURE_REL).resolve(); manifest_path = (args.manifest or repo / MANIFEST_REL).resolve(); result = generate_all(repo, corpus, output_root, manifest_path)
    else: result = check_all(repo, corpus)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
