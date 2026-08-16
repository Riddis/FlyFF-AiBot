# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

spec_path = Path(SPEC).resolve()
app_root = spec_path.parent
entry_script = app_root / "app.py"

# BRIDGE B1 — removed in Phase 7
canonical_farming_parent = app_root.parent / "flyff_farming_simulator"
canonical_position_parent = app_root.parent / "foreground_vision_bot"
if not (canonical_farming_parent / "farming" / "observation_contract.py").is_file():
    raise RuntimeError(f"Canonical farming package is missing at {canonical_farming_parent}")
if not (canonical_position_parent / "position" / "policy.py").is_file():
    raise RuntimeError(f"Canonical position package is missing at {canonical_position_parent}")

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
    # BRIDGE B2 — removed in Phase 7
    pathex=[str(canonical_farming_parent), str(canonical_position_parent), str(app_root)],
    binaries=[],
    datas=[
        (str(canonical_position_parent / "position" / "native_monsters.json"), "position"),
        (str(canonical_position_parent / "position" / "native_position.json"), "position"),
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
