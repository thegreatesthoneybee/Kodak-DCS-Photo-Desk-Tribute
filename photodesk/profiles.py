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


def extract_profiles(installer_bytes):
    """Return {file name: bytes} of every Kodak .icm in the installer."""
    for _, cab in mscab.find_cabinets(installer_bytes):
        files = mscab.extract(cab)
        names = {k.split("\\")[-1].lower(): v for k, v in files.items()}
        if "data1.hdr" not in names:
            continue
        # The outer cabinet holds Disk1\data1.hdr, Disk1\data1.cab, ...
        inner = installshield.Cabinet(names)
        profiles = {}
        for group in PROFILE_GROUPS:
            profiles.update(inner.extract_group(group))
        return {k: v for k, v in profiles.items() if k.lower().endswith(".icm")}
    raise ValueError("no InstallShield payload found; is this the Photo Desk 4.3 installer?")


def install_from_installer(installer_path, dest=None):
    dest = Path(dest) if dest else profile_dir()
    profiles = extract_profiles(Path(installer_path).read_bytes())
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in profiles.items():
        (dest / name).write_bytes(data)
    return dest, sorted(profiles)


def is_installed(dest=None):
    d = Path(dest) if dest else profile_dir()
    return (d / "ERIMM_PCS3.icm").is_file()
