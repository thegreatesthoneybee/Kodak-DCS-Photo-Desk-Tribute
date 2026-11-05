"""Kodak ICC profile management.

``install_from_installer`` unpacks the Kodak profiles, the interface resources and
the help files from the user's own copy of the original ``Photo_Desk_V_4_3_0.exe``
without executing it.  Release builds bundle the profiles, the translation table
and an interface resource pack (no program code, no help files).
"""
import os
import shutil
import sys
from pathlib import Path

from . import installshield, mscab
from .working_space_profiles import builtin_profile_paths

PROFILE_GROUPS = ("Standard ICC Profiles", "Custom Looks ICC Profiles")
# Original UI resources: PhotoDsk.exe is parsed (never executed) for its dialog
# layouts, menus, toolbar, bitmaps and cursors; the .u16 file holds the
# official translations of every UI string.
UI_FILES = ("PhotoDsk.exe", "PhotoDeskUIStrings.u16")
# Release builds bundle the profiles plus only the interface resources of PhotoDsk.exe
# (peres.write_pack), never the executable itself. The original help (CHM) is not used.
UI_PACK = "PhotoDeskUI.pack"


def data_dir():
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "PhotoDesk"


def profile_dir():
    configured = os.environ.get("PHOTODESK_PROFILES")
    if configured:
        return Path(configured)
    bundled = bundled_profile_dir()
    if bundled and (bundled / "ERIMM_PCS3.icm").is_file():
        return bundled
    return data_dir() / "profiles"


def bundled_profile_dir():
    """Profiles shipped inside a frozen build: PhotoDesk.app/Contents/Resources/profiles on macOS,
    _internal/profiles next to PhotoDesk.exe on Windows."""
    if not getattr(sys, "frozen", False):
        return None
    if sys.platform == "darwin":
        return Path(sys.executable).resolve().parent.parent / "Resources" / "profiles"
    return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "profiles"


def _cabinet(installer_bytes):
    for _, cab in mscab.find_cabinets(installer_bytes):
        files = mscab.extract(cab)
        names = {k.split("\\")[-1].lower(): v for k, v in files.items()}
        if "data1.hdr" in names:
            # The outer cabinet holds Disk1\data1.hdr, Disk1\data1.cab, ...
            return installshield.Cabinet(names)
    raise ValueError("no InstallShield payload found; is this the Photo Desk 4.3 installer?")


def extract_profiles(installer_bytes):
    """Return ({profile name: bytes}, {UI file name: bytes}) from the installer."""
    inner = _cabinet(installer_bytes)
    profiles = {}
    for group in PROFILE_GROUPS:
        profiles.update(inner.extract_group(group))
    program = inner.extract_group("Program Files")
    ui = {k: v for k, v in program.items() if k in UI_FILES}
    return {k: v for k, v in profiles.items() if k.lower().endswith(".icm")}, ui


def install_from_installer(installer_path, dest=None):
    dest = Path(dest) if dest else data_dir() / "profiles"
    profiles, ui = extract_profiles(Path(installer_path).read_bytes())
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in profiles.items():
        (dest / name).write_bytes(data)
    ui_dest = dest / "ui"
    ui_dest.mkdir(exist_ok=True)
    # Help files extracted by earlier versions are no longer used.
    shutil.rmtree(ui_dest / "help", ignore_errors=True)
    for name, data in ui.items():
        (ui_dest / name).parent.mkdir(parents=True, exist_ok=True)
        (ui_dest / name).write_bytes(data)
    return dest, sorted(profiles)


def system_color_dirs_profiles():
    """ICC profiles installed in the operating system's colour folders."""
    if sys.platform == "win32":
        dirs = [Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/spool/drivers/color"]
    elif sys.platform == "darwin":
        dirs = [Path("/System/Library/ColorSync/Profiles"), Path("/Library/ColorSync/Profiles"),
                Path.home() / "Library/ColorSync/Profiles"]
    else:
        dirs = [Path("/usr/share/color/icc"), Path.home() / ".local/share/icc"]
    for d in dirs:
        if d.is_dir():
            yield from (p for p in d.iterdir() if p.suffix.lower() in (".icc", ".icm"))


def ui_dir(dest=None):
    return (Path(dest) if dest else profile_dir()) / "ui"


def is_installed(dest=None):
    d = Path(dest) if dest else profile_dir()
    return (d / "ERIMM_PCS3.icm").is_file()
