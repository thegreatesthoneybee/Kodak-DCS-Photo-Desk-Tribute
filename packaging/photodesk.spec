# PyInstaller spec: builds PhotoDesk.exe (Windows x64) or PhotoDesk.app (macOS arm64).
# Run from the repository root:  pyinstaller packaging/photodesk.spec
import os
import shutil
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image
from PyInstaller.utils.hooks import collect_dynamic_libs

repo_root = Path(SPECPATH).resolve().parent
sys.path.insert(0, str(repo_root))  # the package need not be pip-installed to build
from photodesk import APP_NAME, AUTHOR, BASED_ON, __version__
from photodesk import peres, profiles

version_tuple = tuple(int(part) for part in __version__.split(".")) + (0,) * (4 - len(__version__.split(".")))

block_cipher = None

datas = []
app_icon = None

# Bundle the user's installed Kodak profiles, the official translation table and the interface
# resources of PhotoDsk.exe (dialogs, menus, bitmaps, strings) as a pack without its program code.
# The original executable is never shipped.
installed_profiles = Path(os.environ.get("PHOTODESK_PROFILES") or profiles.data_dir() / "profiles")
required = [installed_profiles / "ERIMM_PCS3.icm", installed_profiles / "ui/PhotoDsk.exe",
            installed_profiles / "ui/PhotoDeskUIStrings.u16"]
original = None
if all(resource.is_file() for resource in required):
    staging = Path(workpath) / "bundled_profiles"
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "ui").mkdir(parents=True)
    for profile in installed_profiles.glob("*.icm"):
        shutil.copy2(profile, staging / profile.name)
    shutil.copy2(required[2], staging / "ui" / required[2].name)
    original = peres.PEResources(required[1].read_bytes())
    (staging / "ui" / profiles.UI_PACK).write_bytes(peres.write_pack(original))
    datas.append((str(staging), "profiles"))
    print(f"Bundling Kodak profiles and interface resources from {installed_profiles}")
else:
    # e.g. CI: the app then asks for Photo_Desk_V_4_3_0.exe on first start.
    print(f"WARNING: no Kodak profiles in {installed_profiles}; building without bundled profiles")

if sys.platform == "darwin" and original is not None:
    app_icon = Path(workpath) / "PhotoDesk.icns"
    with Image.open(BytesIO(original.icon(128))) as image:
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
# Per-platform folder: the repository is often shared (e.g. Synology Drive) between a Windows PC and a
# Mac, and a common dist/PhotoDesk would let one platform's build wipe the other's.
coll = COLLECT(exe, a.binaries, a.datas, name="PhotoDesk-macOS" if sys.platform == "darwin" else "PhotoDesk-Windows")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PhotoDesk.app",
        icon=str(app_icon) if app_icon else None,
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
