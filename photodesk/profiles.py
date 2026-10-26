"""Kodak ICC profile management.

The Kodak profiles are copyrighted and are not redistributed with this
program.  ``install_from_installer`` unpacks them from the user's own copy of
the original ``Photo_Desk_V_4_3_0.exe`` without executing it.
"""
import os
import sys
from pathlib import Path

from . import installshield, mscab

PROFILE_GROUPS = ("Standard ICC Profiles", "Custom Looks ICC Profiles")
# Original UI resources: PhotoDsk.exe is parsed (never executed) for its dialog
# layouts, menus, toolbar, bitmaps and cursors; the .u16 file holds the
# official translations of every UI string.
UI_FILES = ("PhotoDsk.exe", "PhotoDeskUIStrings.u16")
# Help Topics: the original HTML help of each language (installer file groups "Help Files - <name>").
HELP_GROUPS = ("English", "French", "German", "Italian", "Japanese", "Korean", "Portuguese",
               "Simplified Chinese", "Spanish", "Traditional Chinese")


def data_dir():
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "PhotoDesk"


def profile_dir():
    return Path(os.environ.get("PHOTODESK_PROFILES", data_dir() / "profiles"))


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
    for group in HELP_GROUPS:
        for name, data in inner.extract_group("Help Files - " + group).items():
            if name.lower().endswith(".chm"):
                ui[f"help/{group}/{name}"] = data
    return {k: v for k, v in profiles.items() if k.lower().endswith(".icm")}, ui


def install_from_installer(installer_path, dest=None):
    dest = Path(dest) if dest else profile_dir()
    profiles, ui = extract_profiles(Path(installer_path).read_bytes())
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in profiles.items():
        (dest / name).write_bytes(data)
    ui_dest = dest / "ui"
    ui_dest.mkdir(exist_ok=True)
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


def help_file(language, dest=None):
    """PHOTODSK.chm for a UI language name (falls back to English)."""
    group = {"Chinese": "Simplified Chinese", "Portugese": "Portuguese"}.get(language, language)
    for g in (group, "English"):
        p = ui_dir(dest) / "help" / g / "PHOTODSK.chm"
        if p.is_file():
            return p
    return None


def is_installed(dest=None):
    d = Path(dest) if dest else profile_dir()
    return (d / "ERIMM_PCS3.icm").is_file()
