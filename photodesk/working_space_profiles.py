"""RGB working-space ICC files shipped as read-only application resources.

AdobeRGB1998.icc and image-P3.icc are independently generated matrix/TRC profiles,
not vendor binaries. The five Kodak-era profiles retain their original bytes.
"""
from pathlib import Path


PROFILE_NAMES = (
    "DCSRGB.icm",
    "ERIMM_PCS3.icm",
    "ProPhotoRGB.icm",
    "KODAK_sRGB.icm",
    "sRGB Color Space Profile.icm",
    "AdobeRGB1998.icc",
    "image-P3.icc",
)


def builtin_profile_paths():
    """Return installed package resources; never write to the app or user directory."""
    directory = Path(__file__).resolve().parent / "data" / "working-spaces"
    return [directory / name for name in PROFILE_NAMES]


def builtin_profiles():
    """Read the actual bundled ICC bytes, including those embedded in exported images."""
    return {path.name: path.read_bytes() for path in builtin_profile_paths()}
