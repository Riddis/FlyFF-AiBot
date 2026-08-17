# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

spec_path = Path(SPEC).resolve()
app_root = spec_path.parent
entry_script = app_root / "app.py"

hidden = collect_submodules("position") + [
    "msgpack",
    "win32api",
    "win32gui",
    "win32process",
    "pywintypes",
    "pythoncom",
    "farming.observation_contract",
]

a = Analysis(
    [str(entry_script)],
    pathex=[str(app_root)],
    binaries=[],
    datas=[
        (str(app_root / "position" / "native_monsters.json"), "position"),
        (str(app_root / "position" / "native_position.json"), "position"),
        (str(app_root / "position" / "native_monsters.json"), "recorder_position"),
    ],
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pandas",
        "stable_baselines3",
        "torch",
        "gymnasium",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FlyffFarmingRecorder",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
