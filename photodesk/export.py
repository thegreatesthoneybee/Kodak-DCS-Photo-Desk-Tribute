"""Saving images as Photo Desk does: Standard TIFF, Standard JPEG and As Is.

Output options follow the original Preferences dialog: RGB working space,
export size (percentage / fill frame / fit to frame), resolution, TIFF image
data (Linear Mode or Render To RGB Working Space), 8/16 bit, embedding the
working-space profile, and JPEG quality (Best / Better / Good).
"""
import dataclasses
import datetime
import functools
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

from . import camera, iptc, pipeline, profiles

JPEG_QUALITY = {"best": 95, "better": 85, "good": 70}
FILE_TYPES = ("tiff", "jpeg", "asis")
_EXT = {"tiff": ".tif", "jpeg": ".jpg"}


@dataclasses.dataclass
class OutputOptions:
    working_space: str = pipeline.DEFAULT_WORKING_SPACE   # profile file name or path
    resize: bool = False
    scale_style: str = "percent"          # percent | fill | fit_trim | fit_fill
    percent: int = 100                    # 10..400
    width: int = 3000                     # pixels, 50..18000
    height: int = 2000
    resolution: float = 300.0             # 1..4000
    resolution_unit: str = "in"           # in | cm
    tiff_linear: bool = False
    tiff_16bit: bool = False
    render_gps: bool = False
    embed_tiff_profile: bool = True
    jpeg_quality: str = "best"
    embed_jpeg_profile: bool = True

    def to_json(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d):
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ------------------------------------------------------------------ sizing
def export_size(w, h, opts):
    """Return (scaled w, h, canvas w, h) for the export size options."""
    if not opts.resize:
        return w, h, w, h
    if opts.scale_style == "percent":
        p = min(max(opts.percent, 10), 400) / 100
        nw, nh = max(1, round(w * p)), max(1, round(h * p))
        return nw, nh, nw, nh
    fw, fh = opts.width, opts.height
    if opts.scale_style == "fill":
        s = max(fw / w, fh / h)
        return max(1, round(w * s)), max(1, round(h * s)), fw, fh
    s = min(fw / w, fh / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return (nw, nh, nw, nh) if opts.scale_style == "fit_trim" else (nw, nh, fw, fh)


def apply_size(img, opts):
    h, w = img.shape[:2]
    nw, nh, cw, ch = export_size(w, h, opts)
    if (nw, nh) != (w, h):
        interp = cv2.INTER_AREA if nw < w else cv2.INTER_CUBIC
        img = cv2.resize(img, (nw, nh), interpolation=interp)
    if (cw, ch) == (nw, nh):
        return img
    canvas = np.ones((ch, cw, 3), img.dtype)
    if cw <= nw and ch <= nh:          # fill frame: centre and crop
        x, y = (nw - cw) // 2, (nh - ch) // 2
        return np.ascontiguousarray(img[y:y + ch, x:x + cw])
    x, y = (cw - nw) // 2, (ch - nh) // 2  # fit frame, fill extra space with white
    canvas[y:y + nh, x:x + nw] = img
    return canvas


def _dpi(opts):
    return opts.resolution * (2.54 if opts.resolution_unit == "cm" else 1.0)


# ------------------------------------------------------------------ export
def output_path(raw_path, out_dir, file_type, name=None):
    ext = _EXT.get(file_type, Path(raw_path).suffix)
    return Path(out_dir) / ((name or Path(raw_path).stem) + ext)


def sequence_names(base, digits, start, count):
    """Rename Setup: base name with a zero-padded sequence number at '*' or the end."""
    names = []
    for i in range(count):
        num = str(start + i).zfill(digits)
        names.append(base.replace("*", num) if "*" in base else base + num)
    return names


def _exif(meta, dpi):
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
    gps = meta.get("gps")
    if gps:
        def dms(v):
            a = abs(v)
            d, m = int(a), int((a - int(a)) * 60)
            return (d, m, round((a - d - m / 60) * 3600, 3))
        g = {0x0000: b"\x02\x02\x00\x00", 0x0001: "N" if gps["lat"] >= 0 else "S", 0x0002: dms(gps["lat"]),
             0x0003: "E" if gps["lon"] >= 0 else "W", 0x0004: dms(gps["lon"])}
        if "alt" in gps:
            g[0x0005] = b"\x01" if gps["alt"] < 0 else b"\x00"
            g[0x0006] = abs(float(gps["alt"]))
        exif.get_ifd(0x8825).update(g)
    return exif


def save(img, path, file_type, opts, metadata, settings=None, space_bytes=None, *,
         _quantize_12bit_lattice=False):
    """img: float32 HxWx3 in [0, 1]; space_bytes: ICC profile to embed (None for linear)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = apply_size(img, opts)
    dpi = _dpi(opts)
    iptc_blob = iptc.encode(settings.iptc) if settings and settings.iptc else None
    if file_type == "jpeg":
        im = Image.fromarray(np.round(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8), "RGB")
        kw = dict(quality=JPEG_QUALITY[opts.jpeg_quality], subsampling=0 if opts.jpeg_quality == "best" else 2,
                  dpi=(dpi, dpi), exif=_exif(metadata, dpi))
        if space_bytes and opts.embed_jpeg_profile:
            kw["icc_profile"] = space_bytes
        im.save(path, "JPEG", **kw)
        if iptc_blob:
            iptc.insert_into_jpeg(path, iptc_blob)
    elif file_type == "tiff":
        quantized = np.clip(img, 0.0, 1.0)
        if opts.tiff_16bit and _quantize_12bit_lattice:
            # Pro SLR/n linear TIFF samples show a 12-bit code lattice in supplied references.
            # This nearest-code mapping is provisional and does not claim native rounding parity.
            codes = np.floor(quantized * 4095 + 0.5).astype(np.uint32)
            data = (codes * 16).astype(np.uint16)
        elif opts.tiff_16bit:
            data = np.round(quantized * 65535).astype(np.uint16)
        else:
            data = np.round(quantized * 255).astype(np.uint8)
        extratags = []
        if space_bytes and opts.embed_tiff_profile:
            extratags.append((34675, "B", len(space_bytes), space_bytes, True))
        if iptc_blob:
            # IPTC-NAA (33723) is stored as LONG for Photoshop compatibility.
            padded = iptc_blob + b"\0" * (-len(iptc_blob) % 4)
            extratags.append((33723, "B", len(padded), padded, True))
        if settings and settings.annotation:
            extratags.append((270, "s", 0, settings.annotation, True))
        tifffile.imwrite(path, data, photometric="rgb", software="PhotoDesk",
                         resolution=(dpi / (2.54 if opts.resolution_unit == "cm" else 1),) * 2,
                         resolutionunit="CENTIMETER" if opts.resolution_unit == "cm" else "INCH",
                         extratags=extratags, metadata=None)
    else:
        raise ValueError(f"unknown file type {file_type}")
    return path


def save_as_is(raw_path, dest, settings):
    """As Is: copy the DCS file with its settings (Photo Desk keeps them inside the file)."""
    dest = Path(dest)
    src = Path(raw_path)
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    pipeline.save_settings(dest, settings)
    return dest


def process_file(raw_path, dest, file_type="jpeg", opts=None, settings=None, profile_directory=None):
    """Render one raw file and save it to dest (a file path)."""
    opts = opts or OutputOptions()
    pdir = Path(profile_directory or profiles.profile_dir())
    settings = settings or pipeline.load_settings(raw_path)
    if file_type == "asis":
        return save_as_is(raw_path, dest, settings)
    if not is_raw(raw_path):
        return process_finished(raw_path, dest, file_type, opts, settings)
    t0 = time.perf_counter()
    raw = pipeline.decode(raw_path)
    t1 = time.perf_counter()
    linear = file_type == "tiff" and opts.tiff_linear and raw.caps.dcs
    space = pipeline.resolve_space(opts.working_space, pdir)
    img = pipeline.render(raw, settings, space, pdir, linear=linear)
    if os.environ.get("PHOTODESK_TIMING"):
        print(f"{Path(raw_path).name}: decode {t1 - t0:.2f} s, render {time.perf_counter() - t1:.2f} s",
              file=sys.stderr)
    if file_type == "tiff" and opts.render_gps and raw.metadata.get("gps"):
        img = render_gps(img, camera.gps_text(raw.metadata["gps"]))
    space_bytes = None if linear else Path(space).read_bytes()
    model = str(raw.metadata.get("model") or "").strip().casefold()
    quantize_12bit_lattice = (
        linear and opts.tiff_16bit and not opts.resize and settings.crop is None and
        not opts.render_gps and model == "dcs pro slr/n"
    )
    return save(img, dest, file_type, opts, raw.metadata, settings, space_bytes,
                _quantize_12bit_lattice=quantize_12bit_lattice)


def is_raw(path):
    try:
        import rawpy
        with rawpy.imread(str(path)):
            return True
    except Exception:  # finished TIFF/JPEG files are not raw
        return False


def load_finished(path, long_edge=None):
    """See _load_finished; full-size decodes are cached (close-up panning, previews, export)."""
    p = Path(path)
    st = p.stat()
    img, icc_bytes = _load_finished_cached(str(p), st.st_mtime_ns, st.st_size, long_edge)
    return img, icc_bytes


@functools.lru_cache(maxsize=4)
def _load_finished_cached(path, mtime_ns, size, long_edge):
    img, icc_bytes = _load_finished(path, long_edge)
    img.flags.writeable = False         # shared between callers
    return img, icc_bytes


def _load_finished(path, long_edge=None):
    """Finished (camera-processed) TIFF/JPEG as float32 RGB plus its embedded ICC profile.

    long_edge: decode at a reduced size of at least that many pixels (JPEG DCT scaling), for
    thumbnails and previews."""
    im = Image.open(path)
    icc_bytes = im.info.get("icc_profile")
    if long_edge:
        im.draft("RGB", (long_edge, long_edge))
    arr = np.asarray(im.convert("RGB"), np.float32) / 255
    return arr, icc_bytes


def process_finished(path, dest, file_type, opts, settings):
    """Finished files only support rotation and crop (as in Photo Desk)."""
    img, icc_bytes = load_finished(path)
    img = pipeline.apply_crop(pipeline.orient(img, 0, settings.rotation), settings.crop)
    # 8-bit sources stay 8-bit: the 16 Bits/Channel option does not apply to finished files.
    opts = dataclasses.replace(opts, tiff_16bit=False)
    meta = {"software": "PhotoDesk"}
    return save(img, dest, file_type, opts, meta, settings, icc_bytes)


def render_gps(img, text):
    """Burn GPS text into the upper-left corner (Render GPS Information In TIFF Files)."""
    scale = max(0.5, img.shape[1] / 2000)
    org = (int(10 * scale), int(30 * scale))
    thick = max(1, int(2 * scale))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.putText(mask, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)
    halo = cv2.dilate(mask, np.ones((2 * thick + 1,) * 2, np.uint8))
    out = img * (1 - halo[..., None] / 255.0)                    # dark outline keeps it legible
    out = out + (mask[..., None] / 255.0) * (1 - out)            # white text
    return out.astype(img.dtype)
