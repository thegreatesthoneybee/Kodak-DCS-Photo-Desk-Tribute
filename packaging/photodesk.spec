# PyInstaller spec: builds PhotoDesk.exe (Windows x64) or PhotoDesk.app (macOS arm64).
# Run from the repository root:  pyinstaller packaging/photodesk.spec
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image
from PyInstaller.utils.hooks import collect_dynamic_libs

repo_root = Path(SPECPATH).resolve().parent
sys.path.insert(0, str(repo_root))  # the package need not be pip-installed to build
from photodesk import APP_NAME, AUTHOR, BASED_ON, __version__
from photodesk.peres import PEResources

version_tuple = tuple(int(part) for part in __version__.split(".")) + (0,) * (4 - len(__version__.split(".")))

block_cipher = None

datas = []
app_icon = None
if sys.platform == "darwin":
    installed_profiles = Path.home() / "Library/Application Support/PhotoDesk/profiles"
    required = [
        installed_profiles / "ERIMM_PCS3.icm",
        installed_profiles / "ui/PhotoDsk.exe",
        installed_profiles / "ui/PhotoDeskUIStrings.u16",
    ]
    if not all(resource.is_file() for resource in required):
        raise SystemExit(
            "Kodak profiles are required for the macOS app bundle. Install them from your "
            "Photo_Desk_V_4_3_0.exe first."
        )
    datas.extend(
        (str(profile), "profiles")
        for profile in installed_profiles.glob("*.icm")
    )
    datas.extend(
        (str(resource), "profiles/ui")
        for resource in (installed_profiles / "ui").glob("*")
    )
    app_icon = Path("build/PhotoDesk.icns").resolve()
    app_icon.parent.mkdir(parents=True, exist_ok=True)
    original_icon = PEResources(required[1].read_bytes()).icon(128)
    with Image.open(BytesIO(original_icon)) as image:
        image.convert("RGBA").save(app_icon, format="ICNS")
elif sys.platform == "win32":
    from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                     VarFileInfo, VarStruct, VSVersionInfo)

    app_icon = repo_root / "packaging" / "PhotoDesk.ico"
    # Explorer > Properties > Details
    win_version = VSVersionInfo(
        ffi=FixedFileInfo(filevers=version_tuple, prodvers=version_tuple),
        kids=[
            StringFileInfo([StringTable("040904B0", [
                StringStruct("CompanyName", AUTHOR),
                StringStruct("FileDescription", APP_NAME),
                StringStruct("FileVersion", __version__),
                StringStruct("InternalName", "PhotoDesk"),
                StringStruct("LegalCopyright", f"{AUTHOR}. Based on {BASED_ON}."),
                StringStruct("OriginalFilename", "PhotoDesk.exe"),
                StringStruct("ProductName", APP_NAME),
                StringStruct("ProductVersion", __version__),
            ])]),
            VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
        ],
    )

a = Analysis(
    ["../photodesk/__main__.py"],
    pathex=[".."],
    binaries=collect_dynamic_libs("rawpy"),
    datas=datas,
    hiddenimports=["photodesk.cli", "photodesk.gui.app", "PySide6.QtMultimedia", "PySide6.QtPrintSupport"],
    excludes=["tkinter", "matplotlib", "scipy", "PySide6.QtWebEngineCore", "PySide6.Qt3DCore"],
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="PhotoDesk",
    console=False,
    icon=str(app_icon) if app_icon and sys.platform == "win32" else None,
    version=win_version if sys.platform == "win32" else None,
    target_arch="arm64" if sys.platform == "darwin" else None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PhotoDesk")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PhotoDesk.app",
        icon=str(app_icon),
        bundle_identifier="org.photodesk.PhotoDesk",
        info_plist={
            "CFBundleName": "PhotoDesk",
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": __version__,
            "NSHumanReadableCopyright": f"Created by {AUTHOR}. Based on {BASED_ON}.",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
