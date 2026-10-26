# PyInstaller spec: builds PhotoDesk.exe (Windows x64) or PhotoDesk.app (macOS arm64).
# Run from the repository root:  pyinstaller packaging/photodesk.spec
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

a = Analysis(
    ["../photodesk/__main__.py"],
    pathex=[".."],
    binaries=collect_dynamic_libs("rawpy"),
    hiddenimports=["photodesk.cli", "photodesk.gui.app"],
    excludes=["tkinter", "matplotlib", "scipy", "PySide6.QtWebEngineCore", "PySide6.Qt3DCore"],
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="PhotoDesk",
    console=False,
    target_arch="arm64" if sys.platform == "darwin" else None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PhotoDesk")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PhotoDesk.app",
        bundle_identifier="org.photodesk.PhotoDesk",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
