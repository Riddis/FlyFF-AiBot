from __future__ import annotations

from pathlib import Path

import mapper


def test_mapper_import_resolves_inside_application_root() -> None:
    app_root = Path(__file__).resolve().parents[1]
    mapper_path = Path(mapper.__file__).resolve()

    assert mapper_path.is_relative_to(app_root)
    assert mapper_path.parent == app_root / "mapper"
