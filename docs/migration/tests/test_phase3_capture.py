from __future__ import annotations

import importlib.util
import struct
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools" / "phase3_capture.py"
SPEC = importlib.util.spec_from_file_location("phase3_capture", TOOL)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_seed_rule_is_exact_and_suite_specific() -> None:
    assert module.seed_for("observation-corpus") == int(
        module.hashlib.sha256(b"FlyffRL-Phase3-v1|observation-corpus").hexdigest()[:16], 16
    )
    assert module.seed_for("observation-corpus") != module.seed_for("neighbour-boundary")


def test_typed_encoder_distinguishes_required_types_and_float_bits() -> None:
    values = [None, False, 0, 0.0, "0", [1], (1,), {"a": 1}]
    encoded = [module.typed_encode(value) for value in values]
    assert len(set(encoded)) == len(encoded)
    assert module.typed_encode(-0.0) != module.typed_encode(0.0)
    assert module.typed_encode(1.5).endswith(struct.pack(">d", 1.5))


def test_mapping_encoding_is_key_order_independent() -> None:
    assert module.typed_encode({"b": 2, "a": 1}) == module.typed_encode({"a": 1, "b": 2})


def test_deterministic_gzip_and_observation_corpus() -> None:
    payload = b"phase3" * 100
    assert module.gzip_bytes(payload) == module.gzip_bytes(payload)
    first = module.build_observation_corpus()
    second = module.build_observation_corpus()
    assert len(first) == len(second) == 10016
    assert module.gzip_bytes(module.msgpack.packb(first, use_bin_type=True)) == module.gzip_bytes(module.msgpack.packb(second, use_bin_type=True))


def test_manifest_is_sorted_and_complete_for_supplied_fixture_set(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    names = tuple(module._fixture_metadata(repo))
    fixture_bytes = {name: name.encode("utf-8") for name in reversed(names)}
    rendered = module.manifest_bytes(repo, fixture_bytes).decode("utf-8").splitlines()
    paths = [line.split("\t", 1)[0] for line in rendered[1:]]
    assert paths == sorted(paths)
    assert len(paths) == len(names)
