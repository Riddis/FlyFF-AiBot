from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent

MODELS_DIR = APP_ROOT / "models"
FARMING_MODELS_DIR = MODELS_DIR / "farming"
MAPPING_MODELS_DIR = MODELS_DIR / "mapping"

TRAINING_LOGS_DIR = APP_ROOT / "training_logs"
FARMING_TRAINING_LOGS_DIR = TRAINING_LOGS_DIR / "farming"
MAPPING_TRAINING_LOGS_DIR = TRAINING_LOGS_DIR / "mapping"

TESTS_DIR = APP_ROOT / "tests"

FARMING_MODEL_RELATIVE = Path("models") / "farming" / "flyff_ppo"
FARMING_CHECKPOINTS_RELATIVE = Path("models") / "farming" / "checkpoints"
FARMING_TRAINING_LOGS_RELATIVE = Path("training_logs") / "farming"

MAPPING_MODEL_RELATIVE = Path("models") / "mapping" / "mapper_explorer_ppo"
MAPPING_CHECKPOINTS_RELATIVE = Path("models") / "mapping" / "checkpoints"
MAPPING_BEST_RELATIVE = Path("models") / "mapping" / "best"
MAPPING_EVALUATIONS_RELATIVE = Path("models") / "mapping" / "evaluations"
MAPPING_ARCHIVE_RELATIVE = Path("models") / "mapping" / "archive"
MAPPING_TRAINING_LOGS_RELATIVE = Path("training_logs") / "mapping"


def resolve_app_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the application directory.

    This keeps model and training output stable even when scripts are launched
    from the repository root, an IDE, or another working directory. Absolute
    paths remain supported for explicit overrides.
    """

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return APP_ROOT / path


def display_app_path(value: str | Path) -> str:
    """Return a portable application-relative path when possible."""

    path = resolve_app_path(value)
    try:
        return str(path.relative_to(APP_ROOT))
    except ValueError:
        return str(path)


def ensure_project_layout() -> None:
    for path in (
        FARMING_MODELS_DIR,
        MAPPING_MODELS_DIR,
        FARMING_TRAINING_LOGS_DIR,
        MAPPING_TRAINING_LOGS_DIR,
        TESTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
