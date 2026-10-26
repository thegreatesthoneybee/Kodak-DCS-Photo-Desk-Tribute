"""Raw processing pipeline modelled on DCS Photo Desk 4.3.

Colour chain (as in Photo Desk):
    white-balanced linear camera RGB  (exposure, tone and colour balance applied)
      -> camera/lighting "Source Correction" profile (A2B0)  -> scene Lab
      -> ERIMM_PCS3 (B2A0)                                   -> ERIMM RGB
      -> "Look" profile (A2B0)                               -> output Lab
      -> RGB working space profile (matrix/TRC)
The profile part of the chain is baked into one 3D LUT per (source, look, space).

Image settings mirror the original's processing options: crop, rotation,
exposure compensation, tone (shadow/midtone/highlight EV), balance method
(Preset/Click/Custom/Auto), white point, lighting, colour balance, lens
optimization, look, noise reduction and sharpening, plus tag, annotation and
IPTC data.
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
# Camera value of an 18% grey in the Kodak source profiles (L* 57.3 scene grey,
# Photo Desk's "midpoint gray").  Exposure/tone EV are relative to this level.
MID_GREY = 0.093
# Scene-linear gain of the Kodak source profiles (0.093 input -> 0.18 scene grey),
# used to place LibRaw-matrix renders of unsupported cameras at the same exposure.
FALLBACK_SCENE_GAIN = 1.94
IDEAL_WHITE_EV = 2.47
SIDE_CAR_SUFFIX = ".photodesk.json"

DEFAULT_LOOK = "DCSProductLook.icm"
DEFAULT_WORKING_SPACE = "sRGB Color Space Profile.icm"
DISPLAY_SPACE = "sRGB Color Space Profile.icm"

LIGHTINGS = camera.ILLUMINANTS
BALANCES = ("preset", "click", "custom", "auto")
NOISE_TYPES = ("none", "basic", "advanced", "moire", "expert")
# Correlated colour temperatures of the preset balance for each lighting.
LIGHTING_CCT = {"Daylight": 5500, "Flash": 5900, "Tungsten": 3200, "Fluorescent": 4100}
WHITE_POINT_DEFAULT = 5000
WHITE_POINT_RANGE = (2000, 7500)
TONE_DEFAULT = (-9.5, 0.0, 3.4)
TONE_MIN_GAP = 2.0
SHARPEN_PRESETS = {"none": (0.0, 0.0, 0), "low": (3.0, 0.8, 8), "medium": (6.0, 1.0, 6),
                   "high": (10.0, 1.2, 4)}
NOISE_PRESETS = {"low": (2, 30), "medium": (3, 50), "high": (5, 75)}
# Crop aspect ratios of the Image Controls panel / Image > Crop Aspect Ratio menu (width, height).
CROP_RATIOS = {"freeform": None, "square": (1, 1), "2x3l": (3, 2), "3x2p": (2, 3), "4x5l": (5, 4),
               "5x4p": (4, 5), "5x7l": (7, 5), "7x5p": (5, 7)}

_BRADFORD_D50_TO_D65 = np.array([
    [0.9555766, -0.0230393, 0.0631636],
    [-0.0282895, 1.0099416, 0.0210077],
    [0.0122982, -0.0204830, 1.3299098]])
XYZ50_TO_PROPHOTO = np.array([
    [1.3459433, -0.2556075, -0.0511118],
    [-0.5445989, 1.5081673, 0.0205351],
    [0.0000000, 0.0000000, 1.2118128]])


@dataclasses.dataclass
class Settings:
    lighting: str = "Daylight"
    look: str = DEFAULT_LOOK
    balance: str = "preset"
    click_wb: tuple = (1.0, 1.0, 1.0)   # camera RGB multipliers from the Click Balance tool
    white_point: int = WHITE_POINT_DEFAULT
    color_balance: tuple = (0, 0, 0)    # R, G, B in -100..100
    exposure: float = 0.0               # EV
    tone: tuple = TONE_DEFAULT          # shadow, midtone, highlight EV
    lens_optimization: int = 0          # 0..99
    noise_type: str = "none"
    noise_radius: int = 3               # 1..10
    noise_strength: int = 50            # 0..100 % (Expert: Strength)
    noise_luma: int = 50                # 0..100 % (Expert: Luma Strength)
    moire_threshold: int = 50           # 0..100
    sharpen_amount: float = 6.0         # 0..20
    sharpen_radius: float = 1.0         # 0.0..10.0
    sharpen_threshold: int = 6          # 0..255
    rotation: int = 0                   # extra clockwise rotation in 90-degree steps
    crop: tuple = None                  # (x0, y0, x1, y1) normalised to the oriented image
    tagged: bool = False
    annotation: str = ""
    iptc: dict = dataclasses.field(default_factory=dict)

    def to_json(self):
        d = dataclasses.asdict(self)
        for k in ("click_wb", "color_balance", "tone", "crop"):
            if d[k] is not None:
                d[k] = list(d[k])
        return d

    @classmethod
    def from_json(cls, d):
        known = {f.name for f in dataclasses.fields(cls)}
        s = cls(**{k: v for k, v in d.items() if k in known})
        s.click_wb = tuple(float(v) for v in s.click_wb)
        s.color_balance = tuple(int(v) for v in s.color_balance)
        s.tone = tuple(float(v) for v in s.tone)
        s.crop = tuple(float(v) for v in s.crop) if s.crop else None
        s.rotation %= 4
        s.iptc = dict(s.iptc)
        return s

    def copy(self, **changes):
        return dataclasses.replace(Settings.from_json(self.to_json()), **changes)

    def sharpening_on(self):
        return self.sharpen_amount > 0


def sidecar_path(raw_path):
    p = Path(raw_path)
    return p.with_name(p.name + SIDE_CAR_SUFFIX)


def at_capture(info):
    """Settings the image was given by the camera (Use Original Camera Settings)."""
    s = Settings()
    if info is None:
        return s
    if info.get("lighting") in LIGHTINGS:
        s.lighting = info["lighting"]
    if info.get("look"):
        s.look = info["look"]
    if info.get("sharpening") is False:
        s.sharpen_amount, s.sharpen_radius, s.sharpen_threshold = SHARPEN_PRESETS["none"]
    return s


def load_settings(raw_path, info=None):
    sc = sidecar_path(raw_path)
    if sc.is_file():
        return Settings.from_json(json.loads(sc.read_text("utf-8")))
    return at_capture(info)


def save_settings(raw_path, settings):
    sidecar_path(raw_path).write_text(json.dumps(settings.to_json(), indent=2, ensure_ascii=False), "utf-8")


def has_saved_settings(raw_path):
    return sidecar_path(raw_path).is_file()


# --------------------------------------------------------------------- decode
@dataclasses.dataclass
class RawImage:
    path: Path
    make: str
    model: str
    camera_rgb: np.ndarray      # float32 HxWx3 linear, not white balanced, 1.0 = sensor clip
    as_shot_wb: np.ndarray      # camera RGB multipliers (G = 1)
    daylight_wb: np.ndarray
    rgb_xyz: np.ndarray         # LibRaw XYZ(D65)->camera matrix (zero when unknown)
    xyz_from_cam: np.ndarray    # camera->XYZ(D65) for the fallback path
    flip: int                   # LibRaw orientation code
    metadata: dict
    scale: float = 1.0          # camera_rgb resolution relative to the full sensor image

    @property
    def caps(self):
        return camera.capabilities(self.model)


def decode(path, half_size=False, algorithm="AHD"):
    path = Path(path)
    make, model = camera.read_make_model(path)
    with rawpy.imread(str(path)) as r:
        full_w, full_h = r.sizes.width, r.sizes.height   # before postprocess, which halves them
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
    metadata["width"], metadata["height"] = full_w, full_h
    return RawImage(path, make, model, cam, wb / wb[1], daylight / daylight[1], rgb_xyz,
                    camera_to_xyz(rgb_xyz), flip, metadata, max(cam.shape[:2]) / max(full_w, full_h))


def reduced(raw, long_edge):
    """Copy of raw with camera_rgb downsampled (for thumbnails and fast previews)."""
    h, w = raw.camera_rgb.shape[:2]
    s = long_edge / max(h, w)
    if s >= 1:
        return raw
    cam = cv2.resize(raw.camera_rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return dataclasses.replace(raw, camera_rgb=cam, scale=raw.scale * cam.shape[1] / w)


def camera_to_xyz(rgb_xyz):
    """Invert LibRaw's XYZ->camera matrix, scaled so camera (1,1,1) maps to D65 white."""
    if not np.any(rgb_xyz):
        return _BRADFORD_D50_TO_D65 @ np.linalg.inv(XYZ50_TO_PROPHOTO)
    m = np.linalg.inv(rgb_xyz)
    d65 = np.array([0.95047, 1.0, 1.08883])
    return m * np.linalg.solve(m, d65)[None, :]


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


# ----------------------------------------------------------------- white balance
def _cct_xy(t):
    """CIE xy chromaticity of a Planckian (< 4000 K) or daylight (>= 4000 K) illuminant."""
    if t < 4000:
        x = -0.2661239e9 / t ** 3 - 0.2343589e6 / t ** 2 + 0.8776956e3 / t + 0.179910
        if t <= 2222:
            y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
        else:
            y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else:
        if t <= 7000:
            x = -4.6070e9 / t ** 3 + 2.9678e6 / t ** 2 + 0.09911e3 / t + 0.244063
        else:
            x = -2.0064e9 / t ** 3 + 1.9018e6 / t ** 2 + 0.24748e3 / t + 0.237040
        y = -3 * x * x + 2.87 * x - 0.275
    return x, y


def _camera_response(raw, cct):
    """Camera RGB of a white lit by an illuminant of the given temperature (G = 1)."""
    if not np.any(raw.rgb_xyz):
        return np.ones(3)
    x, y = _cct_xy(cct)
    c = raw.rgb_xyz @ np.array([x / y, 1.0, (1 - x - y) / y])
    return c / c[1]


def cct_shift(raw, from_cct, to_cct):
    """Multipliers that re-balance an image balanced for from_cct to a to_cct light."""
    return _camera_response(raw, from_cct) / _camera_response(raw, to_cct)


def preset_wb(raw, lighting):
    """Fixed balance point corresponding to the lighting (Preset balance)."""
    return raw.daylight_wb * cct_shift(raw, LIGHTING_CCT["Daylight"], LIGHTING_CCT[lighting])


def auto_wb(cam):
    """Scene balance: grey-world over unclipped mid-tone pixels."""
    small = cam[::4, ::4].reshape(-1, 3)
    lum = small.mean(1)
    m = (small.max(1) < 0.95) & (lum > np.percentile(lum, 10))
    mean = small[m].mean(0) if m.sum() > 100 else small.mean(0)
    return (mean[1] / np.maximum(mean, 1e-6)).astype(np.float64)


class ClickBalanceError(ValueError):
    pass


SATURATED_MSG = "Image is saturated at that point.  Try click balancing at a different point."
TOO_DARK_MSG = "Image is too dark at that point.  Try click balancing at a different point."


def click_wb(cam, x, y, radius=2):
    """Multipliers that neutralise a (2r+1)^2 patch of camera RGB at pixel (x, y)."""
    h, w = cam.shape[:2]
    patch = cam[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
    patch = patch.reshape(-1, 3)
    if len(patch) == 0 or patch.max() >= 0.98:
        raise ClickBalanceError(SATURATED_MSG)
    mean = patch.mean(0)
    if mean.min() < 0.002:
        raise ClickBalanceError(TOO_DARK_MSG)
    return tuple(float(v) for v in mean[1] / mean)


def white_balance(raw, settings):
    if settings.balance == "click":
        wb = np.array(settings.click_wb, np.float64)
    elif settings.balance == "custom":
        wb = raw.as_shot_wb
    elif settings.balance == "auto":
        wb = auto_wb(raw.camera_rgb)
    else:
        wb = preset_wb(raw, settings.lighting)
    wb = np.asarray(wb, np.float64) * cct_shift(raw, WHITE_POINT_DEFAULT, settings.white_point)
    gains = 2.0 ** (np.asarray(settings.color_balance, np.float64) / 100)
    # Colour balance preserves overall luminance.
    gains /= gains @ np.array([0.3, 0.59, 0.11])
    return wb * gains


# ----------------------------------------------------------------- exposure/tone
def exposure_range(raw_or_model):
    model = raw_or_model.model if isinstance(raw_or_model, RawImage) else raw_or_model
    return 1.0 if camera.capabilities(model).pro_back else 2.0


def scene_luminance(lin):
    return lin @ np.array([0.3, 0.59, 0.11], np.float32)


def tone_map_ev(ev, tone):
    """Piecewise-linear EV remap putting shadow/midtone/highlight on their defaults."""
    return np.interp(ev, tone, TONE_DEFAULT, left=None, right=None) + np.where(
        ev < tone[0], ev - tone[0], np.where(ev > tone[2], ev - tone[2], 0))


def validate_tone(tone):
    s, m, h = tone
    if not (TONE_DEFAULT[0] - 4 <= s <= m - TONE_MIN_GAP):
        return "Shadow value is too high." if s > m - TONE_MIN_GAP else "Shadow value is too low."
    if not (s + TONE_MIN_GAP <= m <= h - TONE_MIN_GAP):
        return "Midtone value is too high." if m > h - TONE_MIN_GAP else "Midtone value is too low."
    if not (m + TONE_MIN_GAP <= h <= TONE_DEFAULT[2] + 4):
        return "Highlight value is too high." if h > TONE_DEFAULT[2] + 4 else "Highlight value is too low."
    return None


def auto_exposure(raw, settings):
    """Exposure that places the log-average scene luminance on mid grey (limited to +-1 EV)."""
    lin = np.clip(raw.camera_rgb[::4, ::4] * white_balance(raw, settings).astype(np.float32), 0, 1)
    y = scene_luminance(lin).ravel()
    y = y[(y > 1e-4) & (y < 0.95)]
    if len(y) == 0:
        return 0.0
    ev = -float(np.log2(np.exp(np.mean(np.log(y))) / MID_GREY))
    lim = min(1.0, exposure_range(raw))
    return round(float(np.clip(ev, -lim, lim)), 1)


def ev_histogram(raw, settings, bins=128, lo=-10.0, hi=6.0):
    """Histogram of scene EV (relative to mid grey) with exposure compensation applied."""
    lin = np.clip(raw.camera_rgb[::2, ::2] * white_balance(raw, settings).astype(np.float32), 0, 1)
    y = scene_luminance(lin).ravel() * np.float32(2.0 ** settings.exposure)
    ev = np.log2(np.maximum(y, 1e-6) / MID_GREY)
    hist, edges = np.histogram(ev, bins, (lo, hi))
    return hist, edges


def tone_curve(settings, look, profile_directory=None, n=256, lo=-10.0, hi=6.0):
    """Output L* (0..100) as a function of scene EV for the look and tone settings."""
    d = Path(profile_directory or profiles.profile_dir())
    ev = np.linspace(lo, hi, n)
    mapped = tone_map_ev(ev, settings.tone)
    scene = (0.18 * 2.0 ** mapped).astype(np.float32)
    lab = icc.lab_decode(_profile(str(d / look)).lut("A2B0")(erimm_encode(np.repeat(scene[:, None], 3, 1))))
    return ev, lab[:, 0]


# ------------------------------------------------------------------ colour LUT
@functools.lru_cache(maxsize=64)
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


def look_label(description):
    """'Kodak DCS Product Look' -> 'Product' (as listed in Photo Desk's Look menu)."""
    s = description
    for prefix in ("Kodak DCS ", "Kodak "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s[:-5] if s.endswith(" Look") else s


def working_spaces(directory=None):
    """[(file path, description)] of RGB working spaces: the Kodak ones plus the system's."""
    d = Path(directory) if directory else profiles.profile_dir()
    found, seen = [], set()
    for p in sorted(d.glob("*.ic[cm]")) + sorted(profiles.system_color_dirs_profiles()):
        try:
            prof = _profile(str(p))
        except (ValueError, OSError):
            continue
        if prof.is_working_space and prof.description not in seen:
            seen.add(prof.description)
            found.append((str(p), prof.description))
    return found


def resolve_space(space, directory=None):
    """Accept a profile path or a file name in the profile directory."""
    p = Path(space)
    if p.is_file():
        return str(p)
    return str(Path(directory or profiles.profile_dir()) / space)


@functools.lru_cache(maxsize=16)
def color_lut(source_name, look_name, space_path, profile_directory):
    """Bake the colour chain into a LUT_SIZE^3 table indexed by lin^(1/2.2).

    source_name None selects the LibRaw-matrix fallback; the LUT is then
    indexed by linear XYZ(D65) instead of camera RGB.  space_path None
    produces linear ROMM (Photo Desk's Linear Mode: no Look, no profile).
    """
    d = Path(profile_directory)
    g = np.linspace(0, 1, LUT_SIZE, dtype=np.float32)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    lin = grid ** LUT_INPUT_GAMMA
    if source_name:
        scene_xyz = icc.lab_to_xyz(icc.lab_decode(_profile(str(d / source_name)).lut("A2B0")(lin)))
    else:
        scene_xyz = (lin @ np.linalg.inv(_BRADFORD_D50_TO_D65).T.astype(np.float32)) * FALLBACK_SCENE_GAIN
    if space_path is None:
        out = np.clip(scene_xyz @ XYZ50_TO_PROPHOTO.T.astype(np.float32) / (0.18 * 2 ** IDEAL_WHITE_EV), 0, 1)
        return out.astype(np.float32).reshape(LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)
    if source_name:
        erimm = _profile(str(d / "ERIMM_PCS3.icm")).lut("B2A0")(icc.xyz_encode(scene_xyz))
    else:
        erimm = erimm_encode(scene_xyz @ XYZ50_TO_PROPHOTO.T.astype(np.float32))
    out_xyz = icc.lab_to_xyz(icc.lab_decode(_profile(str(d / look_name)).lut("A2B0")(erimm)))
    rgb = _profile(space_path).encoder()(out_xyz)
    return np.clip(rgb, 0, 1).astype(np.float32).reshape(LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)


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
def source_profile_for(raw, settings, profile_directory):
    source = camera.source_profile(raw.model, settings.lighting)
    if source and not (Path(profile_directory) / source).is_file():
        source = None
    return source


def scene_linear(raw, settings):
    """White-balanced camera RGB with exposure and tone applied (float32, clipped to 1)."""
    lin = raw.camera_rgb * white_balance(raw, settings).astype(np.float32)
    lin *= np.float32(2.0 ** settings.exposure)
    if tuple(settings.tone) != TONE_DEFAULT:
        y = np.maximum(scene_luminance(lin), 1e-7)
        ev = np.log2(y / MID_GREY)
        lin *= (2.0 ** (tone_map_ev(ev, settings.tone) - ev)).astype(np.float32)[..., None]
    return np.clip(lin, 0, 1, out=lin)


def render(raw, settings, space=DISPLAY_SPACE, profile_directory=None, quality="better", linear=False):
    """Return a float32 HxWx3 image in [0, 1] in the working space (oriented and cropped).

    quality "faster" skips lens optimization, noise reduction and sharpening, as
    Photo Desk's Faster display mode does.
    """
    pdir = str(profile_directory or profiles.profile_dir())
    source = source_profile_for(raw, settings, pdir)
    lut = color_lut(source, settings.look, None if linear else resolve_space(space, pdir), pdir)
    lin = scene_linear(raw, settings)
    if source is None:
        lin = np.clip(lin @ raw.xyz_from_cam.T.astype(np.float32), 0, 1)
    out = icc.trilinear(lut, np.power(lin, np.float32(1 / LUT_INPUT_GAMMA))).astype(np.float32, copy=False)
    if quality == "better":
        out = finish(out, settings, raw.caps, raw.scale)
    out = orient(out, raw.flip, settings.rotation)
    return apply_crop(out, settings.crop)


def render_region(raw, settings, rect, space=DISPLAY_SPACE, profile_directory=None, margin=16):
    """Render the normalised rect (x0, y0, x1, y1) of the oriented, uncropped image at full quality."""
    turns = ({0: 0, 3: 2, 5: 3, 6: 1}.get(raw.flip, 0) + settings.rotation) % 4
    x0, y0, x1, y1 = rotate_crop(rect, (4 - turns) % 4)
    h, w = raw.camera_rgb.shape[:2]
    px0, py0 = max(0, int(x0 * w) - margin), max(0, int(y0 * h) - margin)
    px1, py1 = min(w, int(np.ceil(x1 * w)) + margin), min(h, int(np.ceil(y1 * h)) + margin)
    sub = dataclasses.replace(raw, camera_rgb=np.ascontiguousarray(raw.camera_rgb[py0:py1, px0:px1]))
    img = render(sub, settings.copy(crop=None), space, profile_directory)
    sh, sw = py1 - py0, px1 - px0
    inner = ((x0 * w - px0) / sw, (y0 * h - py0) / sh, (x1 * w - px0) / sw, (y1 * h - py0) / sh)
    return apply_crop(img, rotate_crop(inner, turns))


def finish(img, settings, caps=None, scale=1.0):
    """Lens optimization, noise reduction and sharpening on the rendered image.

    scale is the render resolution relative to the full image so that radii
    stay proportional in reduced-size previews.
    """
    caps = caps or camera.capabilities("")
    if settings.lens_optimization and caps.lens_optimization:
        img = lens_optimize(img, settings.lens_optimization)
    if settings.noise_type != "none":
        img = reduce_noise(img, settings, scale)
    if settings.sharpen_amount > 0:
        img = sharpen(img, settings, scale)
    return img


def lens_optimize(img, strength):
    """Remove the radial hue shift some lenses introduce (corner colour cast)."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    small = cv2.resize(lab, (64, 64), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    r2 = ((xx - 31.5) ** 2 + (yy - 31.5) ** 2) / (2 * 31.5 ** 2)
    k = np.float32(strength / 99)
    for c in (1, 2):
        ch = small[..., c]
        center = np.median(ch[r2 < 0.1])
        # Fit the chroma drift as a quadratic function of the radius and remove it.
        a = np.polyfit(r2.ravel(), (ch - center).ravel(), 2)
        drift = np.polyval(a, r2) - np.polyval(a, 0)
        lab[..., c] -= k * cv2.resize(drift.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def reduce_noise(img, settings, scale=1.0):
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycc)
    s = settings.noise_strength / 100
    radius = max(1, round(settings.noise_radius * scale))
    t = settings.noise_type
    if t in ("basic", "advanced", "moire"):
        k = 2 * radius + 1
        cr = cv2.GaussianBlur(cr, (k, k), 0) * s + cr * (1 - s)
        cb = cv2.GaussianBlur(cb, (k, k), 0) * s + cb * (1 - s)
        if t != "basic":
            y = cv2.bilateralFilter(y, k, 0.01 + 0.06 * s, radius)
    if t == "moire" and settings.moire_threshold > 0:
        k = 2 * max(1, round((1 + settings.moire_threshold / 12) * scale)) + 1
        m = settings.moire_threshold / 100
        cr = cv2.medianBlur(cr, min(k, 5)) * m + cr * (1 - m)
        cb = cv2.medianBlur(cb, min(k, 5)) * m + cb * (1 - m)
    if t == "expert":
        d = max(3, round(5 * scale) | 1)
        y = cv2.bilateralFilter(y, d, 0.02 + 0.1 * settings.noise_luma / 100, 3)
        cr = cv2.bilateralFilter(cr, d * 2 + 1, 0.05 + 0.1 * s, 6)
        cb = cv2.bilateralFilter(cb, d * 2 + 1, 0.05 + 0.1 * s, 6)
    out = cv2.cvtColor(cv2.merge([y, cr.astype(np.float32), cb.astype(np.float32)]), cv2.COLOR_YCrCb2RGB)
    return np.clip(out, 0, 1).astype(np.float32, copy=False)


def sharpen(img, settings, scale=1.0):
    """Unsharp mask on luminance: Amount 0..20, Radius 0..10 px, Threshold 0..255."""
    radius = max(0.3, settings.sharpen_radius * scale)
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0]
    detail = y - cv2.GaussianBlur(y, (0, 0), radius)
    thr = settings.sharpen_threshold / 255
    detail = np.where(np.abs(detail) >= thr, detail, 0)
    ycc[..., 0] = y + (settings.sharpen_amount / 10) * detail
    return np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0, 1).astype(np.float32, copy=False)


def orient(img, flip, rotation):
    # LibRaw flip codes: 3 = 180, 5 = 90 CCW, 6 = 90 CW.
    quarter_turns = {0: 0, 3: 2, 5: 3, 6: 1}.get(flip, 0) + rotation
    return np.ascontiguousarray(np.rot90(img, -(quarter_turns % 4)))


def apply_crop(img, crop):
    if not crop:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = crop
    c = img[int(round(y0 * h)):max(int(round(y1 * h)), int(round(y0 * h)) + 1),
            int(round(x0 * w)):max(int(round(x1 * w)), int(round(x0 * w)) + 1)]
    return np.ascontiguousarray(c)


def rotate_crop(crop, turns):
    """Rotate a normalised crop rectangle clockwise by quarter turns."""
    if not crop:
        return crop
    x0, y0, x1, y1 = crop
    for _ in range(turns % 4):
        x0, y0, x1, y1 = 1 - y1, x0, 1 - y0, x1
    return (x0, y0, x1, y1)


def resize_long_edge(img, long_edge):
    h, w = img.shape[:2]
    if not long_edge or max(h, w) <= long_edge:
        return img
    s = long_edge / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
