# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

spec_path = Path(SPEC).resolve()
project_root = spec_path.parent.parent
entry_script = project_root / "tools" / "friend_pointer_recovery_test.py"
config_file = project_root / "position" / "native_monsters.json"

analysis = Analysis(
    [str(entry_script)],
    pathex=[str(project_root), str(project_root / "tools")],
    binaries=[],
    datas=[(str(config_file), "position")],
    hiddenimports=[
        "test_native_independent_reader",
        "win32api",
        "win32gui",
        "win32process",
        "pywintypes",
        "pythoncom",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "numpy",
        "pandas",
        "PySimpleGUI",
        "stable_baselines3",
        "torch",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="FlyffPointerRecoveryTest",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
