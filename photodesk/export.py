"""Writing processed images to JPEG / TIFF with embedded ICC profile and EXIF."""
import datetime
import struct
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from . import pipeline

FORMATS = ("JPEG", "TIFF 8-bit", "TIFF 16-bit")
_EXT = {"JPEG": ".jpg", "TIFF 8-bit": ".tif", "TIFF 16-bit": ".tif"}


# ------------------------------------------------------------- ICC writing
def _s15(v):
    return struct.pack(">i", int(round(v * 65536)))


def _tag_xyz(xyz):
    return b"XYZ \x00\x00\x00\x00" + b"".join(_s15(v) for v in xyz)


def _tag_text(s):
    return b"text\x00\x00\x00\x00" + s.encode("ascii") + b"\x00"


def _tag_desc(s):
    a = s.encode("ascii") + b"\x00"
    return (b"desc\x00\x00\x00\x00" + struct.pack(">I", len(a)) + a
            + b"\x00" * 8 + b"\x00" * 3 + b"\x00" * 67)


def _tag_curve(kind):
    if kind[0] == "gamma":
        return b"curv\x00\x00\x00\x00" + struct.pack(">IH", 1, int(round(kind[1] * 256)))
    x = np.linspace(0, 1, 1024)
    y = kind[1](x)
    return b"curv\x00\x00\x00\x00" + struct.pack(">I", 1024) + (np.round(y * 65535).astype(">u2")).tobytes()


def _srgb_eotf(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _prophoto_eotf(x):
    return np.where(x < 16 / 512, x / 16, x ** 1.8)


_TRC = {
    "sRGB": ("table", _srgb_eotf),
    "Adobe RGB (1998)": ("gamma", 563 / 256),
    "ProPhoto RGB": ("table", _prophoto_eotf),
}


def icc_profile(space):
    """Build an ICC v2 matrix/TRC display profile for an output space (D50 PCS)."""
    matrix, _ = pipeline.OUTPUT_SPACES[space]
    primaries = np.linalg.inv(matrix)  # columns = XYZ(D50) of R, G, B
    curve = _tag_curve(_TRC[space])
    tags = [
        (b"desc", _tag_desc(space)),
        (b"cprt", _tag_text("No copyright, use freely")),
        (b"wtpt", _tag_xyz((0.9642, 1.0, 0.8249))),
        (b"rXYZ", _tag_xyz(primaries[:, 0])),
        (b"gXYZ", _tag_xyz(primaries[:, 1])),
        (b"bXYZ", _tag_xyz(primaries[:, 2])),
        (b"rTRC", curve), (b"gTRC", curve), (b"bTRC", curve),
    ]
    offset = 128 + 4 + 12 * len(tags)
    table, body, placed = b"", b"", {}
    for sig, data in tags:
        if data not in placed:
            placed[data] = offset + len(body)
            body += data + b"\x00" * (-len(data) % 4)
        table += sig + struct.pack(">II", placed[data], len(data))
    size = offset + len(body)
    header = struct.pack(">I4sI4s4s4s", size, b"none", 0x02100000, b"mntr", b"RGB ", b"XYZ ")
    header += b"\x00" * 12 + b"acsp" + b"\x00" * 24 + struct.pack(">I", 0)
    header += _s15(0.9642) + _s15(1.0) + _s15(0.8249) + b"\x00" * 48
    assert len(header) == 128
    return header + struct.pack(">I", len(tags)) + table + body


# ------------------------------------------------------------------ export
def output_path(raw_path, out_dir, fmt):
    return Path(out_dir) / (Path(raw_path).stem + _EXT[fmt])


def _exif(meta):
    exif = Image.Exif()
    if meta.get("make"):
        exif[0x010F] = meta["make"]
    if meta.get("model"):
        exif[0x0110] = meta["model"]
    if meta.get("artist"):
        exif[0x013B] = meta["artist"]
    exif[0x0131] = "PhotoDesk"
    sub = {}
    if meta.get("timestamp"):
        ts = datetime.datetime.fromisoformat(meta["timestamp"]).strftime("%Y:%m:%d %H:%M:%S")
        exif[0x0132] = ts
        sub[0x9003] = ts
    if meta.get("iso"):
        sub[0x8827] = int(meta["iso"])
    if meta.get("shutter"):
        sub[0x829A] = float(meta["shutter"])
    if meta.get("aperture"):
        sub[0x829D] = float(meta["aperture"])
    if meta.get("focal_length"):
        sub[0x920A] = float(meta["focal_length"])
    if sub:
        exif.get_ifd(0x8769).update(sub)
    return exif


def save(img, path, fmt, space, metadata, quality=92, long_edge=None):
    """img: float32 HxWx3 in [0, 1] encoded in `space`."""
    img = pipeline.resize_long_edge(img, long_edge)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    icc = icc_profile(space)
    if fmt == "JPEG":
        im = Image.fromarray(np.round(img * 255).astype(np.uint8), "RGB")
        im.save(path, "JPEG", quality=quality, subsampling=0, icc_profile=icc, exif=_exif(metadata))
    elif fmt in ("TIFF 8-bit", "TIFF 16-bit"):
        if fmt == "TIFF 8-bit":
            data = np.round(img * 255).astype(np.uint8)
        else:
            data = np.round(img * 65535).astype(np.uint16)
        tifffile.imwrite(
            path, data, photometric="rgb", compression="zlib", software="PhotoDesk",
            extratags=[(34675, "B", len(icc), icc, True)],
            metadata=None,
        )
    else:
        raise ValueError(f"unknown format {fmt}")
    return path


def process_file(raw_path, out_dir, fmt="JPEG", space="sRGB", quality=92, long_edge=None,
                 settings=None, profile_directory=None):
    raw = pipeline.decode(raw_path)
    settings = settings or pipeline.load_settings(raw_path, raw.make)
    img = pipeline.render(raw, settings, space, profile_directory)
    return save(img, output_path(raw_path, out_dir, fmt), fmt, space, raw.metadata, quality, long_edge)
