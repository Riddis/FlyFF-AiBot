from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, TextIO

SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
for candidate in (APP_ROOT, SCRIPT_DIR):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

TESTER_VERSION = "1.1"
WINDOW_TITLE_PREFIX = "Spirit Of Madrigal - "
SPAWN_X = 253
SPAWN_Z = 86
MONSTER_SPECIES = 944
MONSTER_FULL_HP = 400236
SAMPLE_SECONDS = 180
SAMPLE_INTERVAL = 0.5
POLL_INTERVAL = 0.05
SLOTS_EACH_DIRECTION = 31
REDISCOVERY_INTERVAL = 3
REDISCOVERY_TIMEOUT = 45
REDISCOVERY_CHUNK_MIB = 4
KILL_EVENT_RADIUS = 80
VISION_RADIUS = 250


@dataclass(frozen=True)
class TestRunResult:
    exit_code: int
    output_dir: Path
    console_log: Path
    report_files: tuple[Path, ...]
    return_zip: Path | None
    package_error: str | None = None


class _TeeText:
    def __init__(self, *targets: TextIO) -> None:
        self._targets = targets

    def write(self, text: str) -> int:
        for target in self._targets:
            target.write(text)
            target.flush()
        return len(text)

    def flush(self) -> None:
        for target in self._targets:
            target.flush()

    def isatty(self) -> bool:
        return any(bool(getattr(target, "isatty", lambda: False)()) for target in self._targets)


@contextmanager
def _tee_console(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = _TeeText(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _TeeText(original_stderr, log_file)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def normalize_character_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise ValueError("Character name cannot be empty.")
    if any(character in name for character in "\r\n\t"):
        raise ValueError("Character name must be entered on one line.")
    if len(name) > 64:
        raise ValueError("Character name is unexpectedly long.")
    return name


def parse_max_hp(value: str | int) -> int:
    try:
        hp = int(str(value).strip(), 10)
    except ValueError as error:
        raise ValueError("Max HP must be a whole number.") from error
    if hp <= 0:
        raise ValueError("Max HP must be greater than zero.")
    if hp > 2_147_483_647:
        raise ValueError("Max HP is outside the supported range.")
    return hp


def build_window_title(character_name: str) -> str:
    return f"{WINDOW_TITLE_PREFIX}{normalize_character_name(character_name)}"


def build_reader_arguments(
    character_name: str,
    max_hp: int,
    output_dir: Path,
) -> list[str]:
    return [
        "--window-title",
        build_window_title(character_name),
        "--spawn-x",
        str(SPAWN_X),
        "--spawn-z",
        str(SPAWN_Z),
        "--player-hp",
        str(parse_max_hp(max_hp)),
        "--monster-hp",
        f"{MONSTER_SPECIES}={MONSTER_FULL_HP}",
        "--timeout",
        "1200",
        "--sample-seconds",
        str(SAMPLE_SECONDS),
        "--sample-interval",
        str(SAMPLE_INTERVAL),
        "--poll-interval",
        str(POLL_INTERVAL),
        "--slots-each-direction",
        str(SLOTS_EACH_DIRECTION),
        "--rediscovery-interval",
        str(REDISCOVERY_INTERVAL),
        "--rediscovery-timeout",
        str(REDISCOVERY_TIMEOUT),
        "--rediscovery-chunk-mib",
        str(REDISCOVERY_CHUNK_MIB),
        "--kill-event-radius",
        str(KILL_EVENT_RADIUS),
        "--vision-radius",
        str(VISION_RADIUS),
        "--output-dir",
        str(output_dir),
    ]


def result_directory() -> Path:
    return Path.home() / "Documents" / "FlyffPointerRecoveryTest"


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _safe_file_component(value: str) -> str:
    normalized = normalize_character_name(value)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
    return safe or "character"


def _snapshot_files(directory: Path) -> set[Path]:
    if not directory.exists():
        return set()
    return {path.resolve() for path in directory.iterdir() if path.is_file()}


def _new_report_files(directory: Path, before: set[Path]) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in directory.glob("independent_native_reader_*.json")
                if path.resolve() not in before
            ),
            key=lambda path: path.name,
        )
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_console_line(console_log: Path, line: str) -> None:
    with console_log.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def create_return_package(
    *,
    output_dir: Path,
    run_id: str,
    character_name: str,
    max_hp: int,
    exit_code: int,
    started_at: datetime,
    completed_at: datetime,
    console_log: Path,
    report_files: tuple[Path, ...],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_file_component(character_name)
    status = "success" if exit_code == 0 else "failed"
    summary_path = output_dir / f"friend_pointer_test_summary_{run_id}.txt"
    metadata_path = output_dir / f"friend_pointer_test_metadata_{run_id}.json"
    return_zip = output_dir / f"SEND_TO_RIDDIMS_{safe_name}_{run_id}.zip"

    report_names = [path.name for path in report_files]
    summary_lines = [
        "FLYFF POINTER RECOVERY TEST RESULT",
        "==================================",
        "",
        f"Status: {status}",
        f"Exit code: {exit_code}",
        f"Character: {normalize_character_name(character_name)}",
        f"Expected game window: {build_window_title(character_name)}",
        f"Entered max HP: {parse_max_hp(max_hp)}",
        f"Started (UTC): {started_at.isoformat()}",
        f"Finished (UTC): {completed_at.isoformat()}",
        f"Tester version: {TESTER_VERSION}",
        "",
        "Files included in this ZIP:",
        f"- {console_log.name}",
        f"- {metadata_path.name}",
        f"- {summary_path.name}",
    ]
    if report_names:
        summary_lines.extend(f"- {name}" for name in report_names)
    else:
        summary_lines.append("- No independent-reader JSON report was produced.")
    summary_lines.extend(
        [
            "",
            "Send this ZIP file back to Riddims unchanged.",
            "You do not need to select or send any individual log files.",
            "",
        ]
    )
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    metadata: dict[str, object] = {
        "schema_version": 1,
        "tester_version": TESTER_VERSION,
        "status": status,
        "exit_code": exit_code,
        "character_name": normalize_character_name(character_name),
        "window_title": build_window_title(character_name),
        "entered_max_hp": parse_max_hp(max_hp),
        "started_utc": started_at.isoformat(),
        "completed_utc": completed_at.isoformat(),
        "runtime": {
            "frozen_executable": bool(getattr(sys, "frozen", False)),
            "python_version": platform.python_version(),
            "operating_system": platform.platform(),
            "machine": platform.machine(),
        },
        "recovery_parameters": {
            "spawn_x": SPAWN_X,
            "spawn_z": SPAWN_Z,
            "monster_species": MONSTER_SPECIES,
            "monster_full_hp": MONSTER_FULL_HP,
            "sample_seconds": SAMPLE_SECONDS,
            "sample_interval": SAMPLE_INTERVAL,
            "poll_interval": POLL_INTERVAL,
            "slots_each_direction": SLOTS_EACH_DIRECTION,
            "rediscovery_interval": REDISCOVERY_INTERVAL,
            "rediscovery_timeout": REDISCOVERY_TIMEOUT,
            "rediscovery_chunk_mib": REDISCOVERY_CHUNK_MIB,
            "kill_event_radius": KILL_EVENT_RADIUS,
            "vision_radius": VISION_RADIUS,
        },
        "included_files": [
            console_log.name,
            metadata_path.name,
            summary_path.name,
            *report_names,
        ],
    }
    _write_json(metadata_path, metadata)

    _append_console_line(console_log, "")
    _append_console_line(console_log, f"Return package prepared: {return_zip}")
    _append_console_line(console_log, "Send that one ZIP file back to Riddims.")

    files_to_package = [console_log, metadata_path, summary_path, *report_files]
    with zipfile.ZipFile(return_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files_to_package:
            if path.is_file():
                archive.write(path, arcname=path.name)
    return return_zip


def _show_error(title: str, message: str) -> None:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    try:
        messagebox.showerror(title, message, parent=root)
    finally:
        root.destroy()


def _show_info(title: str, message: str) -> None:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    try:
        messagebox.showinfo(title, message, parent=root)
    finally:
        root.destroy()


def prompt_for_inputs() -> tuple[str, int] | None:
    import tkinter as tk
    from tkinter import messagebox, simpledialog

    root = tk.Tk()
    root.withdraw()
    try:
        messagebox.showinfo(
            "FlyFF Pointer Recovery Test",
            "Before continuing:\n\n"
            "1. Log into the farm map.\n"
            "2. Stand at the tower spawn.\n"
            "3. Heal to full HP.\n"
            "4. Keep the game open during the test.\n\n"
            "You will only be asked for your character name and max HP.",
            parent=root,
        )
        while True:
            raw_name = simpledialog.askstring(
                "Character name",
                "Enter only your character name:\n"
                "Example: Riddims",
                parent=root,
            )
            if raw_name is None:
                return None
            try:
                character_name = normalize_character_name(raw_name)
                break
            except ValueError as error:
                messagebox.showerror("Invalid character name", str(error), parent=root)

        while True:
            raw_hp = simpledialog.askstring(
                "Max HP",
                "Enter the max HP shown in the game.\n"
                "Make sure your character is fully healed first.",
                parent=root,
            )
            if raw_hp is None:
                return None
            try:
                max_hp = parse_max_hp(raw_hp)
                break
            except ValueError as error:
                messagebox.showerror("Invalid max HP", str(error), parent=root)
        return character_name, max_hp
    finally:
        root.destroy()


def _show_result_in_explorer(path: Path) -> None:
    try:
        if os.name == "nt":
            subprocess.Popen(["explorer.exe", "/select,", str(path)])
    except OSError:
        try:
            if os.name == "nt":
                os.startfile(path.parent)  # type: ignore[attr-defined]
        except OSError:
            pass


def run_test(character_name: str, max_hp: int) -> TestRunResult:
    output_dir = result_directory()
    output_dir.mkdir(parents=True, exist_ok=True)
    before = _snapshot_files(output_dir)
    run_id = _timestamp()
    console_log = output_dir / f"friend_pointer_test_console_{run_id}.log"
    arguments = build_reader_arguments(character_name, max_hp, output_dir)
    window_title = build_window_title(character_name)
    started_at = datetime.now(UTC)
    code = 1

    with _tee_console(console_log):
        print(f"FlyFF friend pointer-recovery test v{TESTER_VERSION}")
        print(f"Character: {character_name}")
        print(f"Expected window: {window_title}")
        print(f"Entered max HP: {max_hp}")
        print(f"Results folder: {output_dir}")
        print()
        try:
            from test_native_independent_reader import main as run_independent_reader

            code = int(run_independent_reader(arguments))
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else 1
            print(f"Test stopped before scanning: {error}")
        except BaseException as error:
            code = 1
            print(f"Unexpected launcher failure: {type(error).__name__}: {error}")

    completed_at = datetime.now(UTC)
    report_files = _new_report_files(output_dir, before)
    try:
        return_zip = create_return_package(
            output_dir=output_dir,
            run_id=run_id,
            character_name=character_name,
            max_hp=max_hp,
            exit_code=code,
            started_at=started_at,
            completed_at=completed_at,
            console_log=console_log,
            report_files=report_files,
        )
        package_error = None
    except BaseException as error:
        return_zip = None
        package_error = f"{type(error).__name__}: {error}"
        _append_console_line(console_log, f"Failed to create return ZIP: {package_error}")

    return TestRunResult(
        exit_code=int(code),
        output_dir=output_dir,
        console_log=console_log,
        report_files=report_files,
        return_zip=return_zip,
        package_error=package_error,
    )


def main() -> int:
    inputs = prompt_for_inputs()
    if inputs is None:
        return 2
    character_name, max_hp = inputs
    result = run_test(character_name, max_hp)

    if result.return_zip is not None:
        if result.exit_code == 0:
            title = "Test complete"
            message = (
                "The pointer-recovery test completed.\n\n"
                "One return ZIP was created:\n"
                f"{result.return_zip.name}\n\n"
                "Send that ZIP file back to Riddims. "
                "It will be selected in File Explorer after you press OK."
            )
            _show_info(title, message)
        else:
            _show_error(
                "Test finished with an error",
                "The test did not complete successfully, but a return ZIP was still created.\n\n"
                f"Send this file to Riddims:\n{result.return_zip.name}\n\n"
                "It will be selected in File Explorer after you press OK.",
            )
        _show_result_in_explorer(result.return_zip)
    else:
        _show_error(
            "Could not create return ZIP",
            "The test ran, but the automatic return ZIP could not be created.\n\n"
            f"Open this folder and send it to Riddims:\n{result.output_dir}\n\n"
            f"Packaging error: {result.package_error or 'unknown error'}",
        )
        try:
            if os.name == "nt":
                os.startfile(result.output_dir)  # type: ignore[attr-defined]
        except OSError:
            pass
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
