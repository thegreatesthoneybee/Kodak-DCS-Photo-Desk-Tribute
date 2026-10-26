"""Raw processing pipeline: LibRaw decode -> Kodak ICC colour chain -> finishing.

Colour chain (as in DCS Photo Desk):
    white-balanced linear camera RGB
      -> camera/illuminant "Source Correction" profile (A2B0)  -> scene Lab
      -> ERIMM_PCS3 (B2A0)                                     -> ERIMM RGB
      -> "Look" profile (A2B0)                                 -> output Lab
      -> output RGB working space
The whole chain is baked into one 3D LUT per (source, look, output space).
"""
import dataclasses
import functools
import json
from pathlib import Path

import cv2
import numpy as np
import rawpy

from . import camera, icc, profiles

LUT_SIZE = 65
LUT_INPUT_GAMMA = 2.2  # LUT is indexed by lin ** (1/2.2) for shadow precision
# Scene-linear gain of the Kodak source profiles (0.093 input -> 0.18 scene grey),
# used to place LibRaw-matrix renders of unsupported cameras at the same exposure.
FALLBACK_SCENE_GAIN = 1.94
SIDE_CAR_SUFFIX = ".photodesk.json"

DEFAULT_LOOK = "DCSProductLook.icm"

# XYZ(D50) -> linear RGB matrices, and transfer functions.
_BRADFORD_D50_TO_D65 = np.array([
    [0.9555766, -0.0230393, 0.0631636],
    [-0.0282895, 1.0099416, 0.0210077],
    [0.0122982, -0.0204830, 1.3299098]])
_XYZ65_TO_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]])
_XYZ65_TO_ADOBE = np.array([
    [2.0413690, -0.5649464, -0.3446944],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0134474, -0.1183897, 1.0154096]])
XYZ50_TO_PROPHOTO = np.array([
    [1.3459433, -0.2556075, -0.0511118],
    [-0.5445989, 1.5081673, 0.0205351],
    [0.0000000, 0.0000000, 1.2118128]])


def _srgb_oetf(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def _prophoto_oetf(x):
    return np.where(x < 1 / 512, 16 * x, np.power(np.maximum(x, 0), 1 / 1.8))


OUTPUT_SPACES = {
    "sRGB": (_XYZ65_TO_SRGB @ _BRADFORD_D50_TO_D65, _srgb_oetf),
    "Adobe RGB (1998)": (_XYZ65_TO_ADOBE @ _BRADFORD_D50_TO_D65, lambda x: np.power(np.maximum(x, 0), 1 / 2.19921875)),
    "ProPhoto RGB": (XYZ50_TO_PROPHOTO, _prophoto_oetf),
}


@dataclasses.dataclass
class Settings:
    illuminant: str = "Daylight"
    look: str = DEFAULT_LOOK
    wb_mode: str = "as_shot"            # as_shot | daylight | auto | custom
    wb_custom: tuple = (1.0, 1.0, 1.0)  # camera RGB multipliers, used when wb_mode == custom
    exposure: float = 0.0               # EV
    sharpen: float = 0.5                # unsharp-mask amount, 0..2
    noise_luma: float = 0.0             # 0..1
    noise_chroma: float = 0.3           # 0..1
    rotation: int = 0                   # extra clockwise rotation in 90-degree steps

    def to_json(self):
        d = dataclasses.asdict(self)
        d["wb_custom"] = list(self.wb_custom)
        return d

    @classmethod
    def from_json(cls, d):
        known = {f.name for f in dataclasses.fields(cls)}
        s = cls(**{k: v for k, v in d.items() if k in known})
        s.wb_custom = tuple(float(v) for v in s.wb_custom)
        s.rotation %= 4
        return s


def sidecar_path(raw_path):
    p = Path(raw_path)
    return p.with_name(p.name + SIDE_CAR_SUFFIX)


def default_settings(make):
    # LibRaw misreads the as-shot multipliers of Kodak DCS files (a neutral
    # target shot with the Daylight preset renders cyan), so start from the
    # LibRaw daylight multipliers there.
    return Settings(wb_mode="daylight" if make.lower().startswith("kodak") else "as_shot")


def load_settings(raw_path, make=""):
    sc = sidecar_path(raw_path)
    if sc.is_file():
        return Settings.from_json(json.loads(sc.read_text("utf-8")))
    return default_settings(make)


def save_settings(raw_path, settings):
    sidecar_path(raw_path).write_text(json.dumps(settings.to_json(), indent=2), "utf-8")


# --------------------------------------------------------------------- decode
@dataclasses.dataclass
class RawImage:
    path: Path
    make: str
    model: str
    camera_rgb: np.ndarray      # float32 HxWx3 linear, not white balanced, 1.0 = sensor clip
    as_shot_wb: np.ndarray      # camera RGB multipliers (G = 1)
    daylight_wb: np.ndarray
    auto_wb: np.ndarray
    xyz_from_cam: np.ndarray    # LibRaw camera->XYZ(D65) matrix for the fallback path
    flip: int                   # LibRaw orientation code
    metadata: dict


def decode(path, half_size=False, algorithm="AHD"):
    path = Path(path)
    make, model = camera.read_make_model(path)
    with rawpy.imread(str(path)) as r:
        params = dict(
            output_color=rawpy.ColorSpace.raw, gamma=(1, 1), no_auto_bright=True,
            output_bps=16, user_wb=[1, 1, 1, 1], user_flip=0, half_size=half_size,
            adjust_maximum_thr=0.0, highlight_mode=rawpy.HighlightMode.Clip,
            demosaic_algorithm=rawpy.DemosaicAlgorithm[algorithm],
        )
        cam = r.postprocess(**params).astype(np.float32) * (1 / 65535)
        daylight = np.array(r.daylight_whitebalance[:3], np.float64)
        if not np.all(daylight > 0):
            daylight = np.ones(3)
        wb = np.array(r.camera_whitebalance[:3], np.float64)
        if not np.all(wb > 0):
            wb = daylight
        rgb_xyz = np.array(r.rgb_xyz_matrix[:3], np.float64)
        flip = r.sizes.flip
        o = r.other
        metadata = {
            "make": make, "model": model, "iso": o.iso_speed, "shutter": o.shutter_speed,
            "aperture": o.aperture, "focal_length": o.focal_length,
            "timestamp": o.timestamp.isoformat() if o.timestamp else None, "artist": o.artist,
        }
    xyz_from_cam = camera_to_xyz(rgb_xyz)
    return RawImage(path, make, model, cam, wb / wb[1], daylight / daylight[1], gray_world_wb(cam),
                    xyz_from_cam, flip, metadata)


def camera_to_xyz(rgb_xyz):
    """Invert LibRaw's XYZ->camera matrix, scaled so camera (1,1,1) maps to D65 white."""
    if not np.any(rgb_xyz):
        return _BRADFORD_D50_TO_D65 @ np.linalg.inv(XYZ50_TO_PROPHOTO)
    m = np.linalg.inv(rgb_xyz)
    d65 = np.array([0.95047, 1.0, 1.08883])
    return m * np.linalg.solve(m, d65)[None, :]


def gray_world_wb(cam):
    """Auto white balance: average of unclipped mid-tone pixels."""
    small = cam[::4, ::4].reshape(-1, 3)
    lum = small.mean(1)
    m = (small.max(1) < 0.95) & (lum > np.percentile(lum, 10))
    mean = small[m].mean(0) if m.sum() > 100 else small.mean(0)
    return (mean[1] / np.maximum(mean, 1e-6)).astype(np.float64)


def wb_from_patch(cam, x0, y0, x1, y1):
    patch = cam[y0:y1, x0:x1].reshape(-1, 3)
    patch = patch[patch.max(1) < 0.98]
    if len(patch) == 0:
        raise ValueError("selected area is clipped")
    mean = patch.mean(0)
    return tuple(float(v) for v in mean[1] / np.maximum(mean, 1e-6))


def thumbnail(path):
    """Return the embedded camera preview as an HxWx3 uint8 array."""
    with rawpy.imread(str(path)) as r:
        try:
            th = r.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            return r.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
        if th.format == rawpy.ThumbFormat.JPEG:
            img = cv2.imdecode(np.frombuffer(th.data, np.uint8), cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(th.data[..., :3])


# ------------------------------------------------------------------ colour LUT
@functools.lru_cache(maxsize=8)
def _profile(path):
    return icc.Profile(path)


def available_looks(directory=None):
    d = Path(directory) if directory else profiles.profile_dir()
    looks = []
    for p in sorted(d.glob("*.icm")):
        prof = _profile(str(p))
        if prof.device_class == "scnr" and "Source Correction" not in prof.description:
            looks.append((p.name, prof.description))
    return looks


@functools.lru_cache(maxsize=16)
def color_lut(source_name, look_name, output_space, profile_directory):
    """Bake the colour chain into a LUT_SIZE^3 table indexed by lin^(1/2.2).

    source_name None selects the LibRaw-matrix fallback; the LUT is then
    indexed by linear XYZ(D65) instead of camera RGB.
    """
    d = Path(profile_directory)
    g = np.linspace(0, 1, LUT_SIZE, dtype=np.float32)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    lin = grid ** LUT_INPUT_GAMMA
    erimm_pcs = _profile(str(d / "ERIMM_PCS3.icm"))
    if source_name:
        scene_lab = icc.lab_decode(_profile(str(d / source_name)).lut("A2B0")(lin))
        erimm = erimm_pcs.lut("B2A0")(icc.xyz_encode(icc.lab_to_xyz(scene_lab)))
    else:
        xyz50 = lin @ np.linalg.inv(_BRADFORD_D50_TO_D65).T.astype(np.float32)
        romm = (xyz50 @ XYZ50_TO_PROPHOTO.T.astype(np.float32)) * FALLBACK_SCENE_GAIN
        erimm = erimm_encode(romm)
    out_lab = icc.lab_decode(_profile(str(d / look_name)).lut("A2B0")(erimm))
    matrix, oetf = OUTPUT_SPACES[output_space]
    rgb = icc.lab_to_xyz(out_lab) @ matrix.T.astype(np.float32)
    return np.clip(oetf(np.clip(rgb, 0, 1)), 0, 1).astype(np.float32).reshape(LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)


def erimm_encode(scene):
    """ERIMM RGB (ISO 22028-3) non-linear encoding of scene-linear ROMM values."""
    e_min, e_clip = 0.001, 10 ** 2.5
    e_t = np.e * e_min
    den = np.log10(e_clip) - np.log10(e_min)
    s = np.asarray(scene, np.float32)
    low = (np.log10(e_t) - np.log10(e_min)) / den * np.maximum(s, 0) / e_t
    high = (np.log10(np.clip(s, e_t, e_clip)) - np.log10(e_min)) / den
    return np.where(s <= e_t, low, high).astype(np.float32)


# ---------------------------------------------------------------------- render
def white_balance(raw, settings):
    if settings.wb_mode == "auto":
        wb = raw.auto_wb
    elif settings.wb_mode == "daylight":
        wb = raw.daylight_wb
    elif settings.wb_mode == "custom":
        wb = np.array(settings.wb_custom, np.float64)
    else:
        wb = raw.as_shot_wb
    return np.asarray(wb, np.float64) / np.min(wb)  # min = 1 keeps clipped highlights neutral


def render(raw, settings, output_space="sRGB", profile_directory=None):
    """Return a float32 HxWx3 image in [0, 1], encoded in output_space."""
    pdir = str(profile_directory or profiles.profile_dir())
    source = camera.source_profile(raw.model, settings.illuminant)
    if source and not (Path(pdir) / source).is_file():
        source = None
    lut = color_lut(source, settings.look, output_space, pdir)

    wb = white_balance(raw, settings).astype(np.float32)
    lin = np.clip(raw.camera_rgb * wb, 0, 1)
    if source is None:
        lin = np.clip(lin @ raw.xyz_from_cam.T.astype(np.float32), 0, 1)
    lin *= np.float32(2.0 ** settings.exposure)
    np.clip(lin, 0, 1, out=lin)
    out = icc.trilinear(lut, np.power(lin, np.float32(1 / LUT_INPUT_GAMMA))).astype(np.float32, copy=False)
    out = finish(out, settings)
    return orient(out, raw.flip, settings.rotation)


def finish(img, settings):
    if settings.noise_luma <= 0 and settings.noise_chroma <= 0 and settings.sharpen <= 0:
        return img
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycc)
    if settings.noise_chroma > 0:
        k = 2 * int(1 + 6 * settings.noise_chroma) + 1
        cr = cv2.GaussianBlur(cr, (k, k), 0)
        cb = cv2.GaussianBlur(cb, (k, k), 0)
    if settings.noise_luma > 0:
        y = cv2.bilateralFilter(y, 5, 0.02 + 0.08 * settings.noise_luma, 3)
    if settings.sharpen > 0:
        blur = cv2.GaussianBlur(y, (0, 0), 1.0)
        y = y + settings.sharpen * (y - blur)
    ycc = cv2.merge([y, cr, cb])
    return np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0, 1).astype(np.float32, copy=False)


def orient(img, flip, rotation):
    # LibRaw flip codes: 3 = 180, 5 = 90 CCW, 6 = 90 CW.
    quarter_turns = {0: 0, 3: 2, 5: 3, 6: 1}.get(flip, 0) + rotation
    return np.ascontiguousarray(np.rot90(img, -(quarter_turns % 4)))


def resize_long_edge(img, long_edge):
    h, w = img.shape[:2]
    if not long_edge or max(h, w) <= long_edge:
        return img
    s = long_edge / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
