from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tools.friend_pointer_recovery_test import (
    MONSTER_FULL_HP,
    MONSTER_SPECIES,
    build_reader_arguments,
    build_window_title,
    create_return_package,
    normalize_character_name,
    parse_max_hp,
)


def test_window_title_uses_character_name() -> None:
    assert build_window_title(" Riddims ") == "Spirit Of Madrigal - Riddims"


def test_friend_arguments_hardcode_validated_recovery_parameters(tmp_path: Path) -> None:
    arguments = build_reader_arguments("FriendName", 32348, tmp_path)

    assert arguments[arguments.index("--window-title") + 1] == (
        "Spirit Of Madrigal - FriendName"
    )
    assert arguments[arguments.index("--player-hp") + 1] == "32348"
    assert arguments[arguments.index("--spawn-x") + 1] == "253"
    assert arguments[arguments.index("--spawn-z") + 1] == "86"
    assert arguments[arguments.index("--monster-hp") + 1] == (
        f"{MONSTER_SPECIES}={MONSTER_FULL_HP}"
    )
    assert arguments[arguments.index("--slots-each-direction") + 1] == "31"
    assert arguments[arguments.index("--rediscovery-interval") + 1] == "3"
    assert arguments[arguments.index("--poll-interval") + 1] == "0.05"
    assert arguments[arguments.index("--output-dir") + 1] == str(tmp_path)


def test_input_validation_rejects_empty_name_and_invalid_hp() -> None:
    with pytest.raises(ValueError):
        normalize_character_name("   ")
    with pytest.raises(ValueError):
        normalize_character_name("Bad\nName")
    with pytest.raises(ValueError):
        parse_max_hp("not a number")
    with pytest.raises(ValueError):
        parse_max_hp(0)


def test_return_package_contains_all_relevant_run_files(tmp_path: Path) -> None:
    console_log = tmp_path / "friend_pointer_test_console_run.log"
    report = tmp_path / "independent_native_reader_run.json"
    console_log.write_text("console output\n", encoding="utf-8")
    report.write_text('{"status": "success"}\n', encoding="utf-8")
    started = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    completed = datetime(2026, 8, 1, 12, 3, tzinfo=UTC)

    return_zip = create_return_package(
        output_dir=tmp_path,
        run_id="20260801T120000.000000Z",
        character_name="Friend Name",
        max_hp=32348,
        exit_code=0,
        started_at=started,
        completed_at=completed,
        console_log=console_log,
        report_files=(report,),
    )

    assert return_zip.name.startswith("SEND_TO_RIDDIMS_Friend_Name_")
    with zipfile.ZipFile(return_zip) as archive:
        names = set(archive.namelist())
        assert console_log.name in names
        assert report.name in names
        summary_name = next(name for name in names if "summary" in name)
        metadata_name = next(name for name in names if "metadata" in name)
        summary = archive.read(summary_name).decode("utf-8")
        metadata = json.loads(archive.read(metadata_name).decode("utf-8"))

    assert "Send this ZIP file back to Riddims unchanged." in summary
    assert metadata["character_name"] == "Friend Name"
    assert metadata["window_title"] == "Spirit Of Madrigal - Friend Name"
    assert metadata["entered_max_hp"] == 32348
    assert metadata["status"] == "success"
    assert report.name in metadata["included_files"]


def test_failed_run_still_creates_a_return_zip_without_report(tmp_path: Path) -> None:
    console_log = tmp_path / "friend_pointer_test_console_failed.log"
    console_log.write_text("failure output\n", encoding="utf-8")
    moment = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

    return_zip = create_return_package(
        output_dir=tmp_path,
        run_id="20260801T120000.000000Z",
        character_name="Tester",
        max_hp=10000,
        exit_code=1,
        started_at=moment,
        completed_at=moment,
        console_log=console_log,
        report_files=(),
    )

    with zipfile.ZipFile(return_zip) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if "metadata" in name)
        summary_name = next(name for name in names if "summary" in name)
        metadata = json.loads(archive.read(metadata_name).decode("utf-8"))
        summary = archive.read(summary_name).decode("utf-8")

    assert metadata["status"] == "failed"
    assert "No independent-reader JSON report was produced." in summary
    assert console_log.name in names


def test_windows_installer_provides_normal_executable_uninstaller() -> None:
    app_root = Path(__file__).resolve().parents[1]
    exe_builder = (
        app_root / "tools" / "build_friend_pointer_recovery_exe.ps1"
    ).read_text(encoding="utf-8")
    installer_builder = (
        app_root / "tools" / "build_friend_pointer_recovery_installer.ps1"
    ).read_text(encoding="utf-8")
    installer_definition = (
        app_root / "tools" / "FlyffPointerRecoveryTestInstaller.iss"
    ).read_text(encoding="utf-8")

    assert "PyInstaller" in exe_builder
    assert "-SkipPortablePackage" in installer_builder
    assert "FlyffPointerRecoveryTestInstallerPackage.zip" in installer_builder
    assert "FlyffPointerRecoveryTestSetup.exe" in installer_builder
    assert "SkipExeBuild" in installer_builder
    assert "IsccPath" in installer_builder
    assert "LOCALAPPDATA" in installer_builder
    assert "CurrentVersion\\Uninstall" in installer_builder
    assert "Get-ChildItem" in installer_builder
    assert "Uninstallable=yes" in installer_definition
    assert "{uninstallexe}" in installer_definition
    assert "[UninstallDelete]" in installer_definition
    assert "{userdocs}\\FlyffPointerRecoveryTest" in installer_definition


def test_installer_payload_does_not_require_friend_system_python() -> None:
    app_root = Path(__file__).resolve().parents[1]
    installer_definition = (
        app_root / "tools" / "FlyffPointerRecoveryTestInstaller.iss"
    ).read_text(encoding="utf-8")

    assert "FlyffPointerRecoveryTest.exe" in installer_definition
    assert "python.exe" not in installer_definition.lower()
    assert ".venv" not in installer_definition.lower()
