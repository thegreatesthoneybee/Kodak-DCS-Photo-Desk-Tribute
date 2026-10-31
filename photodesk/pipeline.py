"""Raw processing pipeline modelled on DCS Photo Desk 4.3.

Colour chain (as in Photo Desk):
    white-balanced linear camera RGB  (exposure, tone and colour balance applied)
      -> camera/lighting "Source Correction" profile (A2B0)  -> scene Lab
      -> ERIMM_PCS3 (B2A0)                                   -> ERIMM RGB
      -> "Look" profile (A2B0)                               -> output Lab
      -> RGB working space profile (matrix/TRC)
The profile part of the chain is baked into one 3D LUT per (source, look, space).
Linear mode bypasses that chain and uses white-balanced, exposed and toned camera RGB.

Image settings mirror the original's processing options: crop, rotation,
exposure compensation, tone (shadow/midtone/highlight EV), balance method
(Preset/Click/Custom/Auto), white point, lighting, colour balance, lens
optimization, look, noise reduction and sharpening, plus tag, annotation and
IPTC data.
"""
import dataclasses
import functools
import json
import math
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
PRO_SLR_TONE_REFERENCE = (-9.535300254821777, 0.0, 3.464400053024292)
PRO_SLR_TONE_THRESHOLD = float(np.float32(0.1))
# Sharpening dialog presets (amount, radius, threshold) as built into Photo Desk 4.3 (DCS SDK
# defaults and DCSProBack.dll preset tables: amount 0..20, radius in 0.1 steps, threshold 0..255).
SHARPEN_PRESETS = {"none": (0.0, 0.0, 0), "low": (5.0, 1.0, 2), "medium": (10.0, 1.0, 2),
                   "high": (17.0, 1.4, 2)}
# Noise Reduction presets of Pro Back / Pro SLR images (Photo Desk's ProImage), Low/Medium/High:
# Basic/Advanced/Moire take (radius level 0..100, strength %) and Expert a strength %. Pro SLR files
# carry their own tables (Kodak tags 3695 and 3683: type, level, strength per preset); these are
# the camera DLLs' built-in defaults.
NOISE_PRESETS = {"radius": ((20, 23), (50, 50), (80, 80)), "expert": (50, 75, 100)}
NOISE_PRESET_NAMES = ("low", "medium", "high")
NOISE_RADIUS_TAG, NOISE_EXPERT_TAG = 3695, 3683
# Radius slider range per type; Basic moves in steps of 2.
NOISE_RADIUS_RANGE = {"basic": (2, 6), "advanced": (1, 4), "moire": (1, 10)}
BASIC_PRESET_RADII = (2, 4, 6)
# Radius level (0..100) -> radius: upper bounds of each radius, from ProImage.
_RADIUS_LEVELS = {"advanced": (17, 50, 83, 100), "moire": (9, 19, 29, 39, 49, 61, 71, 80, 90, 100)}
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
    auto_balance_wb: tuple = None       # scene-wide balance calculated when Auto is selected
    white_point: int = WHITE_POINT_DEFAULT
    # Colour temperature the balance method's own balance corresponds to; White Point re-balances
    # from it, so white_point == white_point_ref leaves the balance as the method set it.
    white_point_ref: int = WHITE_POINT_DEFAULT
    kelvin_mode: int = None             # saved Kelvin mode, when restored from the RAW
    kelvin_illumination: str = None     # separate from the saved image-lighting selector
    saved_balance_point: tuple = None  # RGB neutral point restored from the camera processing record
    color_balance: tuple = (0, 0, 0)    # R, G, B in -100..100
    exposure: float = 0.0               # EV
    tone: tuple = TONE_DEFAULT          # shadow, midtone, highlight EV
    tone_enabled: bool = True
    lens_optimization: int = 0          # 0..99
    noise_type: str = "none"
    noise_radius: int = 3               # 1..10
    noise_strength: int = 50            # 0..100 % (Expert: Strength)
    noise_luma: int = 50                # 0..100 % (Expert: Luma Strength)
    moire_threshold: int = 50           # 0..100
    sharpen_amount: float = 10.0        # 0..20 (default: Medium preset)
    sharpen_radius: float = 1.0         # 0.0..10.0
    sharpen_threshold: int = 2          # 0..255
    rotation: int = 0                   # extra clockwise rotation in 90-degree steps
    crop: tuple = None                  # (x0, y0, x1, y1) normalised to the oriented image
    tagged: bool = False
    annotation: str = ""
    iptc: dict = dataclasses.field(default_factory=dict)

    def to_json(self):
        d = dataclasses.asdict(self)
        for k in ("click_wb", "auto_balance_wb", "color_balance", "tone", "crop", "saved_balance_point"):
            if d[k] is not None:
                d[k] = list(d[k])
        return d

    @classmethod
    def from_json(cls, d):
        known = {f.name for f in dataclasses.fields(cls)}
        s = cls(**{k: v for k, v in d.items() if k in known})
        s.click_wb = tuple(float(v) for v in s.click_wb)
        s.auto_balance_wb = tuple(float(v) for v in s.auto_balance_wb) if s.auto_balance_wb is not None else None
        s.color_balance = tuple(int(v) for v in s.color_balance)
        s.tone = tuple(float(v) for v in s.tone)
        s.saved_balance_point = tuple(s.saved_balance_point) if s.saved_balance_point is not None else None
        s.crop = tuple(float(v) for v in s.crop) if s.crop else None
        s.rotation %= 4
        s.iptc = dict(s.iptc)
        return s

    def copy(self, **changes):
        if "tone" in changes and tuple(changes["tone"]) != tuple(self.tone) and "tone_enabled" not in changes:
            changes["tone_enabled"] = True
        balance_fields = ("balance", "lighting", "white_point", "click_wb")
        if "saved_balance_point" not in changes and any(
            field in changes and changes[field] != getattr(self, field) for field in balance_fields
        ):
            changes["saved_balance_point"] = None
        balance_changed = any(field in changes and changes[field] != getattr(self, field)
                              for field in ("balance", "click_wb"))
        lighting_changed = "lighting" in changes and changes["lighting"] != self.lighting
        if "kelvin_mode" not in changes and (balance_changed or
                                               (lighting_changed and self.kelvin_mode not in (4, 5, 6))):
            changes["kelvin_mode"] = None
            if "kelvin_illumination" not in changes:
                changes["kelvin_illumination"] = None
        elif (lighting_changed and not balance_changed and
              changes.get("kelvin_mode", self.kelvin_mode) in (4, 5, 6)):
            changes.setdefault("kelvin_illumination", changes["lighting"])
        if "balance" in changes and changes["balance"] != self.balance and "auto_balance_wb" not in changes:
            changes["auto_balance_wb"] = None
        return dataclasses.replace(Settings.from_json(self.to_json()), **changes)

    def sharpening_on(self):
        return self.sharpen_amount > 0


def sidecar_path(raw_path):
    p = Path(raw_path)
    return p.with_name(p.name + SIDE_CAR_SUFFIX)


def at_capture(info):
    """Settings the image was given by the camera (At Capture / Use Original Camera Settings);
    ``info`` is camera.capture_info() of the image."""
    s = Settings()
    if info is None:
        return s
    caps = camera.capabilities(info.get("model"))
    if caps.pro_back or caps.pro_slr:
        # ProImage (Pro Back / Pro SLR files) falls back to the Low preset, older DCS files to Medium.
        s.sharpen_amount, s.sharpen_radius, s.sharpen_threshold = SHARPEN_PRESETS["low"]
    if info.get("lighting") in LIGHTINGS:
        s.lighting = info["lighting"]
    # The camera's white balance setting (Kodak tag 1020): a lighting selects the Preset balance
    # for it, the camera's Custom setting the Custom balance (camera auto stays on Preset).
    wb = info.get("white_balance")
    if wb in LIGHTINGS:
        s.balance, s.lighting = "preset", wb
    elif wb == "custom" and "custom" in caps.balances:
        s.balance = "custom"
    if info.get("look"):
        s.look = info["look"]
    if info.get("sharpening") is False:
        s.sharpen_amount, s.sharpen_radius, s.sharpen_threshold = SHARPEN_PRESETS["none"]
    if "lens_optimization" in info:
        s.lens_optimization = int(info["lens_optimization"])
    return s


def noise_radius_for_level(noise_type, level):
    """Radius Photo Desk derives from a preset's radius level for a noise type."""
    if noise_type == "basic":
        return 2 if level <= 24 else 4 if level <= 74 else 6
    bounds = _RADIUS_LEVELS.get(noise_type, _RADIUS_LEVELS["advanced"])
    return next((i + 1 for i, b in enumerate(bounds) if level <= b), len(bounds))


def noise_presets(raw_path=None):
    """{"radius": ((level, strength) x3), "expert": (strength x3)} Low/Medium/High presets of a file."""
    out = dict(NOISE_PRESETS)
    if raw_path is None:
        return out
    try:
        tags = camera.read_kodak_tags(raw_path, (NOISE_RADIUS_TAG, NOISE_EXPERT_TAG))
    except OSError:
        return out
    v = tags.get(NOISE_RADIUS_TAG)
    if v and len(v) == 9:
        out["radius"] = tuple((v[i + 1], v[i + 2]) for i in (0, 3, 6))
    v = tags.get(NOISE_EXPERT_TAG)
    if v and len(v) == 9:
        out["expert"] = tuple(v[i + 2] for i in (0, 3, 6))
    return out


def apply_noise_preset(values, name, presets):
    """Set the Low/Medium/High preset ``name`` into the noise fields dict ``values``."""
    i = NOISE_PRESET_NAMES.index(name)
    t = values["noise_type"]
    if t == "expert":
        values["noise_strength"] = values["noise_luma"] = presets["expert"][i]
    elif t != "none":
        level, strength = presets["radius"][i]
        values["noise_radius"] = BASIC_PRESET_RADII[i] if t == "basic" else noise_radius_for_level(t, level)
        values["noise_strength"] = strength
    return values


def capture_settings(raw_path):
    """at_capture() of a file, reading what the camera recorded in it."""
    try:
        _, model = camera.read_make_model(raw_path)
        return at_capture(camera.capture_info(raw_path, model))
    except OSError:
        return Settings()


def load_settings(raw_path):
    sc = sidecar_path(raw_path)
    if sc.is_file():
        return Settings.from_json(json.loads(sc.read_text("utf-8")))
    settings = capture_settings(raw_path)
    try:
        params = camera.stored_process_parameters(raw_path)
        render_params = camera.stored_render_parameters(raw_path)
        _, model = camera.read_make_model(raw_path)
    except OSError:
        params = {}
        render_params = {}
        model = None
    levels = params.get("tone")
    if levels is not None and validate_tone(levels) is None:
        settings.tone = levels
    if "color_balance" in params:
        settings.color_balance = params["color_balance"]
    if "tone_enabled" in params:
        settings.tone_enabled = params["tone_enabled"]
    if "white_point" in params and camera.balance_curves(raw_path).get(params["kelvin_illumination"]):
        settings.kelvin_mode = params["kelvin_mode"]
        settings.kelvin_illumination = params["kelvin_illumination"]
        settings.white_point = settings.white_point_ref = params["white_point"]
    caps = camera.capabilities(model)
    if caps.pro_slr:
        exposure = render_params.get("exposure")
        if exposure is not None and abs(exposure) <= caps.exposure_range:
            settings.exposure = exposure
        if "lighting" in render_params:
            settings.lighting = render_params["lighting"]
        settings.saved_balance_point = render_params.get("balance_point")
    return settings


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
    camera_rgb: np.ndarray      # float32 HxWx3 linear, unbalanced, normalized by LibRaw
    as_shot_wb: np.ndarray      # camera RGB multipliers (G = 1)
    daylight_wb: np.ndarray
    rgb_xyz: np.ndarray         # LibRaw XYZ(D65)->camera matrix (zero when unknown)
    xyz_from_cam: np.ndarray    # camera->XYZ(D65) for the fallback path
    flip: int                   # LibRaw orientation code
    metadata: dict
    scale: float = 1.0          # camera_rgb resolution relative to the full-size image
    # Neutral camera RGB the camera recorded for each of its balance settings
    # (camera.balance_points: lighting names, "custom", "camera_auto").
    balance_points: dict = dataclasses.field(default_factory=dict)
    balance_curves: dict = dataclasses.field(default_factory=dict)
    balance_factors: dict = dataclasses.field(default_factory=dict)
    auto_kelvin_illumination: str = None
    custom_kelvin_illumination: str = None
    auto_scene_exposure: float = None
    lens_optimization_baseline: int = None

    @property
    def caps(self):
        return camera.capabilities(self.model)


def _pro_slr_default_crop(model, sizes):
    """Return the Pro SLR/n or /c default crop in visible-image coordinates."""
    model_name = (model or "").lower()
    if not camera.capabilities(model).pro_slr or not any(name in model_name for name in ("slr/n", "slr/c")):
        return None
    try:
        width, height = int(sizes.width), int(sizes.height)
        raw_width, raw_height = int(sizes.raw_width), int(sizes.raw_height)
        left, top = int(sizes.left_margin), int(sizes.top_margin)
        crop_left, crop_top = int(sizes.crop_left_margin), int(sizes.crop_top_margin)
        crop_width, crop_height = int(sizes.crop_width), int(sizes.crop_height)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if min(width, height, raw_width, raw_height, crop_width, crop_height) <= 0:
        return None
    if min(left, top, crop_left, crop_top) < 0:
        return None
    if left + width > raw_width or top + height > raw_height:
        return None

    # LibRaw's crop offsets are relative to the raw frame; postprocess starts at the
    # visible area, so remove its margins before using the crop on the RGB result.
    x0, y0 = crop_left - left, crop_top - top
    x1, y1 = x0 + crop_width, y0 + crop_height
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None
    return x0, y0, x1, y1


def _crop_rendered_image(image, crop, visible_width, visible_height):
    """Map visible-image crop boundaries to the actual full/half-size RGB array."""
    x0, y0, x1, y1 = crop
    out_height, out_width = image.shape[:2]
    px0 = x0 * out_width // visible_width
    py0 = y0 * out_height // visible_height
    px1 = (x1 * out_width + visible_width - 1) // visible_width
    py1 = (y1 * out_height + visible_height - 1) // visible_height
    px0, py0 = min(px0, out_width - 1), min(py0, out_height - 1)
    px1 = min(out_width, max(px0 + 1, px1))
    py1 = min(out_height, max(py0 + 1, py1))
    return image[py0:py1, px0:px1]


def decode(path, half_size=False, algorithm="AHD"):
    path = Path(path)
    make, model = camera.read_make_model(path)
    with rawpy.imread(str(path)) as r:
        sizes = r.sizes
        visible_w, visible_h = sizes.width, sizes.height
        crop = _pro_slr_default_crop(model, sizes)
        full_w, full_h = (crop[2] - crop[0], crop[3] - crop[1]) if crop else (visible_w, visible_h)
        params = dict(
            output_color=rawpy.ColorSpace.raw, gamma=(1, 1), no_auto_bright=True,
            output_bps=16, user_wb=[1, 1, 1, 1], user_flip=0, half_size=half_size,
            adjust_maximum_thr=0.0, highlight_mode=rawpy.HighlightMode.Clip,
            demosaic_algorithm=rawpy.DemosaicAlgorithm[algorithm],
        )
        cam = r.postprocess(**params)
        if crop:
            cam = _crop_rendered_image(cam, crop, visible_w, visible_h)
        cam = cam.astype(np.float32)
        cam *= np.float32(1 / 65535)
        daylight = np.array(r.daylight_whitebalance[:3], np.float64)
        if not np.all(daylight > 0):
            daylight = np.ones(3)
        wb = np.array(r.camera_whitebalance[:3], np.float64)
        if not np.all(wb > 0):
            wb = daylight
        rgb_xyz = np.array(r.rgb_xyz_matrix[:3], np.float64)
        flip = sizes.flip
        o = r.other
        metadata = {
            "make": make, "model": model, "iso": o.iso_speed, "shutter": o.shutter_speed,
            "aperture": o.aperture, "focal_length": o.focal_length,
            "timestamp": o.timestamp.isoformat() if o.timestamp else None, "artist": o.artist,
        }
    metadata["width"], metadata["height"] = full_w, full_h
    gps = camera.read_gps(path)
    if gps:
        metadata["gps"] = gps
    auto_illumination, auto_exposure = camera.auto_capture_settings(path)
    return RawImage(path, make, model, cam, wb / wb[1], daylight / daylight[1], rgb_xyz,
                    camera_to_xyz(rgb_xyz), flip, metadata, max(cam.shape[:2]) / max(full_w, full_h),
                    camera.balance_points(path), camera.balance_curves(path), camera.balance_factors(path),
                    auto_illumination, camera.custom_kelvin_illumination(path), auto_exposure,
                    camera.lens_optimization_baseline(path, model))


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


def point_wb(point):
    """Multipliers (G = 1) that balance the neutral camera RGB ``point`` to grey."""
    p = np.asarray(point, np.float64)
    return p[1] / p


def preset_wb(raw, lighting):
    """Preset balance. Older DCS files (Photo Desk's DCSImage) use the camera's own balance point
    for the lighting (Kodak tags 2120-2123); Pro Back / Pro SLR files, which Photo Desk balances
    through the camera DLL's colour model, get LibRaw's daylight balance shifted to the lighting's
    colour temperature."""
    caps = raw.caps
    if lighting in raw.balance_points and not (caps.pro_back or caps.pro_slr):
        return point_wb(raw.balance_points[lighting])
    return raw.daylight_wb * cct_shift(raw, LIGHTING_CCT["Daylight"], LIGHTING_CCT[lighting])


def custom_wb(raw):
    """Custom balance: the camera's custom (click) balance point, Kodak tag 2124 (Photo Desk's
    DCSImage reads it with GetBalancePointFromTag(0x84C)). Pro Back / Pro SLR points are in the
    camera DLL's colour space, so they are applied relative to the Daylight point."""
    bp = raw.balance_points
    if "custom" not in bp:
        return raw.as_shot_wb
    caps = raw.caps
    if not (caps.pro_back or caps.pro_slr):
        return point_wb(bp["custom"])
    if "Daylight" in bp:
        return raw.daylight_wb * point_wb(bp["custom"]) / point_wb(bp["Daylight"])
    return raw.as_shot_wb


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


def _click_balance_roi(cam, x, y, size=16):
    """Bound an even-aligned square around a sensor pixel, shifting it inward at image edges."""
    h, w = cam.shape[:2]
    if h == 0 or w == 0:
        return cam[:0, :0]
    roi_h, roi_w = min(size, h), min(size, w)
    x, y = min(w - 1, max(0, int(x))), min(h - 1, max(0, int(y)))
    x0 = max(0, min(w - roi_w, 2 * ((x + 1) // 2) - roi_w // 2))
    y0 = max(0, min(h - roi_h, 2 * ((y + 1) // 2) - roi_h // 2))
    x0 -= x0 % 2
    y0 -= y0 % 2
    return cam[y0:y0 + roi_h, x0:x0 + roi_w]


def click_wb_pro_slr(cam, x, y):
    """Pro SLR click-balance approximation over a 16x16 sensor RGB ROI.

    LibRaw supplies demosaiced, normalized RGB rather than the native CFA samples used by the
    camera DLL.  Per-channel middle-half trimming limits isolated outliers, but does not reproduce
    the native sampler or its integer thresholds.
    """
    patch = _click_balance_roi(cam, x, y).reshape(-1, 3)
    if len(patch) == 0:
        raise ClickBalanceError(SATURATED_MSG)
    samples = np.sort(patch, axis=0)
    trim = len(samples) // 4
    central = samples[trim:len(samples) - trim] if trim else samples
    mean = central.mean(0, dtype=np.float64)
    if mean.max() >= 0.98:
        raise ClickBalanceError(SATURATED_MSG)
    if mean.min() < 0.002:
        raise ClickBalanceError(TOO_DARK_MSG)
    return tuple(float(v) for v in mean[1] / mean)


def balance_temperature(raw, settings):
    """Colour temperature (K, rounded to 100 within WHITE_POINT_RANGE) whose Preset-style balance
    (Daylight balance shifted along the illuminant locus) is closest to the balance method's own."""
    if settings.balance == "auto" and settings.auto_balance_wb is None and raw.caps.pro_slr:
        calibration = auto_kelvin_calibration(raw)
        if calibration is not None:
            return ((calibration[0] + 50) // 100) * 100
    curve = raw.balance_curves.get(settings.lighting) if raw.caps.pro_slr else None
    if settings.balance == "preset" and curve:
        nominal = curve[25]
        if WHITE_POINT_RANGE[0] <= nominal <= WHITE_POINT_RANGE[1]:
            return int(round(nominal / 100) * 100)
    wb = np.asarray(method_wb(raw, settings), np.float64)
    day = np.asarray(raw.daylight_wb, np.float64)
    if not np.any(raw.rgb_xyz) or not np.all(wb > 0):
        return WHITE_POINT_DEFAULT
    lo, hi = WHITE_POINT_RANGE
    temps = np.arange(lo, hi + 1, 100)
    target = np.log(wb / wb[1])[[0, 2]]
    best, best_d = WHITE_POINT_DEFAULT, np.inf
    for t in temps:
        c = day * cct_shift(raw, LIGHTING_CCT["Daylight"], int(t))
        d = np.sum((np.log(c / c[1])[[0, 2]] - target) ** 2)
        if d < best_d:
            best, best_d = int(t), d
    return best


def camera_auto_point(raw):
    """Return a valid camera Auto point in the applicable driver's coordinate scale."""
    point = raw.balance_points.get("camera_auto")
    if raw.caps.pro_back or point is None:
        return None
    point = np.floor(np.asarray(point, np.float64))
    if not np.all(point >= 1) or np.all(point == 4095) or (raw.caps.pro_slr and np.any(point > 10000)):
        return None
    if raw.caps.pro_slr and auto_kelvin_calibration(raw) is not None:
        point = np.floor(np.asarray(2048.0 * point / point[0], dtype=np.float32) + 0.5)
    return point


def camera_auto_wb(raw):
    """Return valid camera-computed Auto multipliers for supported DCS images."""
    point = camera_auto_point(raw)
    if point is None:
        return None
    return np.max(point) / point if raw.caps.pro_slr else point_wb(point)


def method_wb(raw, settings):
    """Balance of the balance method alone (before White Point and Colour Balance)."""
    if settings.balance == "click":
        wb = np.array(settings.click_wb, np.float64)
    elif settings.balance == "custom":
        wb = custom_wb(raw)
    elif settings.balance == "auto":
        if settings.auto_balance_wb is not None:
            wb = np.asarray(settings.auto_balance_wb)
        else:
            wb = camera_auto_wb(raw)
            if wb is None:
                wb = auto_wb(raw.camera_rgb)
    else:
        wb = preset_wb(raw, settings.lighting)
    return wb


def kelvin_point(curve, temperature):
    try:
        if isinstance(temperature, (int, np.integer)):
            temperature_word = int(temperature) & 0xffff
        else:
            temperature = float(temperature)
            if not math.isfinite(temperature):
                return None
            temperature_word = math.trunc(temperature) & 0xffff
    except (TypeError, ValueError, OverflowError):
        return None
    x = temperature_word * 0.01
    point = []
    for channel, i in enumerate((0, 4, 8)):
        with np.errstate(over="ignore", invalid="ignore"):
            c0, c1, c2, c3 = np.asarray([curve[i + j] for j in range(4)], dtype=np.float32)
        if not np.all(np.isfinite((c0, c1, c2, c3))):
            return None
        c0, c1, c2, c3 = (float(c) for c in (c0, c1, c2, c3))

        # Preserve the native polynomial order. Python uses binary64 intermediates rather than
        # x87 extended precision, but the native binary32 coefficient loads and red/green spills
        # are explicit here.
        value = c3 * x
        value *= x
        value *= x
        value += c2 * x * x
        value += c1 * x
        value += c0
        if channel < 2:
            with np.errstate(over="ignore", invalid="ignore"):
                value = float(np.float32(value))
        if not math.isfinite(value):
            return None
        rounded = math.trunc(value + 0.5)
        point.append(min(10000, max(100, rounded)))
    return np.asarray(point, dtype=np.float64)


def auto_kelvin_calibration(raw):
    """Mode-5 temperature and factors derived from the camera Auto point."""
    return _kelvin_calibration(raw, getattr(raw, "auto_kelvin_illumination", None),
                               raw.balance_points.get("camera_auto"), 5)


def _kelvin_calibration(raw, illumination, point, mode):
    curve = raw.balance_curves.get(illumination)
    factors = getattr(raw, "balance_factors", {}).get(illumination)
    if illumination not in camera.ILLUMINANTS or curve is None or len(curve) != 27 or point is None or mode not in (4, 5, 6):
        return None
    point = np.floor(np.asarray(point, dtype=np.float64))
    if len(point) != 3 or not np.all(np.isfinite(point)) or not np.all((10 <= point) & (point <= 10000)):
        return None
    point = point.astype(np.float32)
    norm = np.asarray(2048.0 * point / point[0], dtype=np.float32)
    words = np.floor(norm + 0.5) % 65536
    if mode != 5:
        if factors is None or len(factors) != 3 or not np.all(np.isfinite(factors)) or not np.all(
            np.asarray(factors) >= 0.1
        ) or not np.all((10 <= words) & (words <= 10000)):
            return None
        words = np.floor(words / np.asarray(factors, dtype=np.float32) + 0.5) % 65536
    if words[0] == 0:
        return None
    inverse_input = (np.floor(2048.0 * words / words[0]) % 65536).astype(np.float64) * 0.01
    coefficients = np.asarray(curve, dtype=np.float32)
    estimates = [sum(float(coefficients[i + j]) * inverse_input[k] ** j for j in range(4))
                 for k, i in enumerate((12, 16, 20))]
    estimates = np.asarray(estimates, dtype=np.float32)
    if not np.all(np.isfinite(estimates)) or np.count_nonzero((estimates < 100) | (estimates > 11000)) > 1:
        return None
    first, second = ((1, 2) if curve[12] == 0 else (0, 2) if curve[16] == 0 else (0, 1))
    if mode != 5:
        bounds = ((0.6, 1.3), (0.56, 1.11), (0.71, 1.43), (0.9, 1.8))
        lo, hi = (float(np.float32(value)) for value in bounds[camera.ILLUMINANTS.index(illumination)])
        if estimates[second] == 0 or not lo <= float(estimates[first]) / float(estimates[second]) <= hi:
            return None
    temperature = int(np.clip((float(estimates[first]) + float(estimates[second])) / 2, 100, 11000))
    reference = kelvin_point(curve, temperature)
    if reference is None:
        return None
    return temperature, np.asarray(norm / reference, dtype=np.float32)


def custom_kelvin_calibration(raw):
    """Mode-4 temperature and factors derived from the captured Custom point."""
    illumination = getattr(raw, "custom_kelvin_illumination", None)
    day_curve = raw.balance_curves.get("Daylight")
    if day_curve is None or not 2000 <= day_curve[25] <= 11000:
        return None
    return _kelvin_calibration(raw, illumination, raw.balance_points.get("custom"), 4)


def white_balance(raw, settings):
    wb = np.asarray(method_wb(raw, settings), np.float64)
    illumination = settings.kelvin_illumination if settings.kelvin_mode is not None else settings.lighting
    curve = raw.balance_curves.get(illumination) if raw.caps.pro_slr else None
    if raw.caps.pro_slr and settings.saved_balance_point is not None:
        point = np.asarray(settings.saved_balance_point, np.float64)
        wb = np.max(point) / point
    elif curve and (settings.balance == "preset" or settings.kelvin_mode in (0, 1, 2, 3, 4, 5, 6)):
        point = kelvin_point(curve, settings.white_point)
        if settings.kelvin_mode == 4:
            calibration = custom_kelvin_calibration(raw)
            factors = calibration[1] if calibration is not None else None
        elif settings.kelvin_mode == 5:
            calibration = auto_kelvin_calibration(raw)
            factors = calibration[1] if calibration is not None else None
        elif settings.kelvin_mode == 6:
            calibration = _kelvin_calibration(raw, illumination, point, 6) if point is not None else None
            factors = calibration[1] if calibration is not None else None
        elif settings.kelvin_mode is None or settings.kelvin_mode < 4:
            factors = getattr(raw, "balance_factors", {}).get(illumination)
        else:
            factors = None
        if point is not None and (factors is not None or settings.kelvin_mode is None):
            with np.errstate(over="ignore", invalid="ignore"):
                factors = np.asarray(factors if factors is not None else (1, 1, 1), dtype=np.float32)
            if factors.shape != (3,) or not np.all(np.isfinite(factors)):
                raise ValueError("Kelvin balance factors must be three finite values")
            scaled_words = []
            for channel, factor in zip(point, factors):
                scaled = (int(channel) & 0xffff) * float(factor)
                if not math.isfinite(scaled):
                    raise ValueError("Kelvin scaling produced a non-finite channel")
                # The native conversion adds 0.5, truncates to an integer, then stores a WORD.
                scaled_words.append(math.trunc(scaled + 0.5) & 0xffff)
            point = np.asarray(scaled_words, dtype=np.float64)
            if np.any(point == 0):
                raise ValueError("Kelvin scaling produced a zero channel")
            wb = np.max(point) / point
        elif settings.white_point != settings.white_point_ref:
            wb = wb * cct_shift(raw, settings.white_point_ref, settings.white_point)
    elif settings.white_point != settings.white_point_ref:
        wb = wb * cct_shift(raw, settings.white_point_ref, settings.white_point)
    controls = np.asarray(settings.color_balance, np.float64)
    if raw.caps.pro_slr:
        gains = 10.0 ** (0.0030103 * controls) if np.all(np.abs(controls) <= 100) else np.ones(3)
        weights = [0.299, 0.587, 0.114]
    else:
        gains = 2.0 ** (controls / 100)
        weights = [0.3, 0.59, 0.11]
    # Colour balance preserves overall luminance.
    gains /= gains @ np.array(weights)
    return wb * gains


# ----------------------------------------------------------------- exposure/tone
def exposure_range(raw_or_model):
    model = raw_or_model.model if isinstance(raw_or_model, RawImage) else raw_or_model
    return camera.capabilities(model).exposure_range


def scene_luminance(lin):
    return lin @ np.array([0.3, 0.59, 0.11], np.float32)


def tone_map_ev(ev, tone):
    """Piecewise-linear EV remap putting shadow/midtone/highlight on their defaults."""
    return np.interp(ev, tone, TONE_DEFAULT, left=None, right=None) + np.where(
        ev < tone[0], ev - tone[0], np.where(ev > tone[2], ev - tone[2], 0))


def tone_marker_value(value):
    """Round a binary32 tone marker to a tenth, with midpoint ties toward positive infinity."""
    value = float(np.float32(value))
    if value == 0:
        return value
    return float(np.float32(math.floor(value * 10 + 0.5) / 10))


@functools.lru_cache(maxsize=32)
def pro_slr_tone_lut(tone):
    """Build the Pro SLR tone table in its 12-bit, log2 code domain.

    LibRaw's decoded input and the original sensor pipeline do not necessarily have identical
    scaling, and Python floating-point calculations do not reproduce x87 precision exactly.
    """
    source = np.asarray(tone, dtype=np.float32).astype(np.float64)
    reference = np.asarray(PRO_SLR_TONE_REFERENCE)
    offset = 8.533673272
    slopes = np.diff(reference) / np.diff(source)
    x = np.array([source[0], (source[0] + source[1]) / 2,
                  (source[1] + source[2]) / 2, source[2]]) + offset
    y = np.array([reference[0], (reference[0] + reference[1]) / 2,
                  (reference[1] + reference[2]) / 2, reference[2]]) + offset
    m = np.array([slopes[0], slopes[0], slopes[1], slopes[1]])
    return tone_integer_table(tuple(zip(x, y, m))).astype(np.float32) / 4095


def tone_integer_table(points, end=4095, output_max=4095, scale=1.0):
    """Apply the camera DLL's rounded knot boundaries and integer output rules."""
    out = np.empty(end + 1, dtype=np.uint16)
    scale = float(np.float32(scale))
    upper = output_max * scale
    boundary = max(0, int(2 ** points[0][0] + 0.5))
    last = max(0, int(scale * 2 ** points[0][1] + 0.5))
    i = min(boundary, end + 1)
    out[:i] = last
    for (x0, y0, m0), (x1, y1, m1) in zip(points, points[1:]):
        h = x1 - x0
        secant = (y1 - y0) / h
        c2 = (3 * secant - 2 * m0 - m1) / h
        c3 = (m0 + m1 - 2 * secant) / (h * h)
        stop = min(int(2 ** x1 + 0.5), end + 1)
        if stop <= i:
            continue
        codes = np.arange(i, stop, dtype=np.float64)
        t = np.log2(np.maximum(codes, 1)) - x0
        t[codes == 0] = 0
        y = ((c3 * t + c2) * t + m0) * t + y0
        values = scale * np.exp2(y)
        values = np.where(values > upper, np.trunc(upper), values)
        quantized = np.floor(np.maximum(values, 0) + 0.5).astype(np.uint16)
        out[i:stop] = quantized
        last = int(quantized[-1])
        i = stop
    out[i:] = last
    return out


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
    """Use the Pro camera's Auto result or estimate exposure for other camera families."""
    if raw.caps.pro_slr:
        return raw.auto_scene_exposure
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
    lin = raw.camera_rgb[::2, ::2] * white_balance(raw, settings).astype(np.float32)
    lin = np.clip(lin * np.float32(2.0 ** settings.exposure), 0, 1)
    y = scene_luminance(lin).ravel()
    ev = np.log2(np.maximum(y, 1e-6) / MID_GREY)
    hist, edges = np.histogram(ev, bins, (lo, hi))
    return hist, edges


def tone_curve(raw, settings, space=DISPLAY_SPACE, profile_directory=None, n=256, lo=-12.0, hi=6.0):
    """Approximate weighted output RGB over the scene-EV domain using the active render chain.

    This is an approximation of the current renderer, not Kodak pixel parity. Exposure is removed
    so the curve shares the histogram's scene-EV axis without applying exposure twice.
    """
    if raw is None:
        return None
    pdir = str(profile_directory or profiles.profile_dir())
    ev = np.linspace(lo, hi, n, dtype=np.float32)
    scene = (np.float32(MID_GREY) * np.exp2(ev)).astype(np.float32)
    render_settings = settings.copy(exposure=0)
    wb = white_balance(raw, render_settings).astype(np.float32)
    camera_rgb = (scene[:, None] / wb[None, :])[None, :, :]
    lin = scene_linear(raw, render_settings, camera_rgb=camera_rgb)
    source = source_profile_for(raw, render_settings, pdir)
    if source is None:
        lin = np.clip(lin @ raw.xyz_from_cam.T.astype(np.float32), 0, 1)
    lut = color_lut(source, render_settings.look, resolve_space(space, pdir), pdir)
    rgb = icc.trilinear(lut, lin, power=1 / LUT_INPUT_GAMMA)[0]
    return ev, rgb @ np.array([0.299, 0.587, 0.114], np.float32)


# ------------------------------------------------------------------ colour LUT
@functools.lru_cache(maxsize=64)
def _profile(path):
    return icc.Profile(path)


def _is_look(prof):
    """Look profiles are input-class profiles that map ERIMM to PCS (Kodak and Custom Looks)."""
    return prof.device_class == "scnr" and "Source Correction" not in prof.description


def available_looks(directory=None):
    """[(file name, description)] of the Kodak Looks plus Custom Looks (e.g. the Kodak Professional
    DCS Custom Looks product) installed in the system's profile folders."""
    d = Path(directory) if directory else profiles.profile_dir()
    looks, seen = [], set()
    kodak = sorted(d.glob("*.icm"))
    for p in kodak + sorted(profiles.system_color_dirs_profiles()):
        if p.name in seen:
            continue
        try:
            prof = _profile(str(p))
        except (ValueError, OSError):
            continue
        # System folders hold scanner profiles too: only Kodak DCS (Custom) Looks are listed from there.
        if p not in kodak and not prof.description.startswith("Kodak DCS"):
            continue
        if _is_look(prof) and "A2B0" in prof.tags:
            seen.add(p.name)
            looks.append((p.name, prof.description))
    return looks


def look_path(name, profile_directory):
    """Look profile file: the Kodak profile folder first, then the system profile folders."""
    p = Path(profile_directory) / name
    if p.is_file():
        return p
    return next((q for q in profiles.system_color_dirs_profiles() if q.name == name), p)


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


def space_setting(path, directory=None):
    """How a chosen working space is stored: the file name for a profile of the profile directory
    (which moves with the app: macOS runs downloaded apps from a new random path each launch),
    the full path for a system profile."""
    p = Path(path)
    d = Path(directory) if directory else profiles.profile_dir()
    return p.name if (d / p.name).is_file() and p.parent.resolve() == d.resolve() else str(p)


def resolve_space(space, directory=None):
    """Profile file of a stored working space: a path, or a file name in the profile directory or
    the system colour folders. A path that no longer exists (an app that moved since it was saved)
    is looked up by its file name, and the default working space used if it is gone."""
    p = Path(space)
    if p.is_file():
        return str(p)
    d = Path(directory) if directory else profiles.profile_dir()
    for candidate in [d / p.name] + [q for q in profiles.system_color_dirs_profiles() if q.name == p.name]:
        if candidate.is_file():
            return str(candidate)
    return str(d / DEFAULT_WORKING_SPACE)


@functools.lru_cache(maxsize=16)
def color_lut(source_name, look_name, space_path, profile_directory):
    """Bake the colour chain into a LUT_SIZE^3 table indexed by lin^(1/2.2).

    source_name None selects the LibRaw-matrix fallback; the LUT is then
    indexed by linear XYZ(D65) instead of camera RGB.
    """
    d = Path(profile_directory)
    g = np.linspace(0, 1, LUT_SIZE, dtype=np.float32)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    lin = grid ** LUT_INPUT_GAMMA
    if source_name:
        scene_xyz = icc.lab_to_xyz(icc.lab_decode(_profile(str(d / source_name)).lut("A2B0")(lin)))
    else:
        scene_xyz = (lin @ np.linalg.inv(_BRADFORD_D50_TO_D65).T.astype(np.float32)) * FALLBACK_SCENE_GAIN
    if source_name:
        erimm = _profile(str(d / "ERIMM_PCS3.icm")).lut("B2A0")(icc.xyz_encode(scene_xyz))
    else:
        erimm = erimm_encode(scene_xyz @ XYZ50_TO_PROPHOTO.T.astype(np.float32))
    out_lab = _profile(str(look_path(look_name, d))).lut("A2B0")(erimm)
    tone_profiles = (("DCSProSLRn", "DCSProSLRnToneCurve.icm"),
                     ("DCSProSLRc", "DCSProSLRcToneCurve.icm"),
                     ("DCSPro14", "DCSPro14ToneCurve.icm"))
    tone_profile = next((name for prefix, name in tone_profiles if source_name and source_name.startswith(prefix)), None)
    if tone_profile:
        out_lab = _profile(str(d / tone_profile)).lut("A2B0")(out_lab)
    out_xyz = icc.lab_to_xyz(icc.lab_decode(out_lab))
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


def scene_linear(raw, settings, camera_rgb=None):
    """White-balanced camera RGB with exposure and tone applied (float32, clipped to 1).

    ``camera_rgb`` optionally supplies samples while retaining ``raw`` as the balance/driver context.
    """
    cam = raw.camera_rgb if camera_rgb is None else camera_rgb
    lin = cam * white_balance(raw, settings).astype(np.float32)
    lin *= np.float32(2.0 ** settings.exposure)
    if raw.caps.pro_slr:
        if settings.tone_enabled and any(abs(float(np.float32(value)) - reference) > PRO_SLR_TONE_THRESHOLD
               for value, reference in zip(settings.tone, PRO_SLR_TONE_REFERENCE)):
            indices = np.floor(np.clip(lin, 0, 1) * 4095 + 0.5).astype(np.uint16)
            return pro_slr_tone_lut(tuple(settings.tone))[indices]
        return np.clip(lin, 0, 1, out=lin)
    if settings.tone_enabled and tuple(settings.tone) != TONE_DEFAULT:
        y = np.maximum(scene_luminance(lin), 1e-7)
        ev = np.log2(y / MID_GREY)
        lin *= (2.0 ** (tone_map_ev(ev, settings.tone) - ev)).astype(np.float32)[..., None]
    return np.clip(lin, 0, 1, out=lin)


def render(raw, settings, space=DISPLAY_SPACE, profile_directory=None, quality="better", linear=False):
    """Return a float32 HxWx3 image in [0, 1], oriented and cropped.

    Linear mode uses white-balanced, exposed and toned camera RGB directly, without applying
    source, Look or working-space profiles. Its decoder scaling remains an approximation.
    quality "faster" skips lens optimization, noise
    reduction and sharpening, as Photo Desk's Faster display mode does.
    """
    if linear:
        out = scene_linear(raw, settings)
    else:
        pdir = str(profile_directory or profiles.profile_dir())
        source = source_profile_for(raw, settings, pdir)
        lut = color_lut(source, settings.look, resolve_space(space, pdir), pdir)
        lin = scene_linear(raw, settings)
        if source is None:
            lin = np.clip(lin @ raw.xyz_from_cam.T.astype(np.float32), 0, 1)
        out = icc.trilinear(lut, lin, power=1 / LUT_INPUT_GAMMA)
    if quality == "better":
        out = finish(out, settings, raw.caps, raw.scale,
                     lens_optimization_baseline=getattr(raw, "lens_optimization_baseline", None))
    out = orient(out, raw.flip, settings.rotation)
    return apply_crop(out, settings.crop)


def render_region(raw, settings, rect, space=DISPLAY_SPACE, profile_directory=None, margin=16):
    """Render the normalised rect (x0, y0, x1, y1) of the oriented, uncropped image at full quality."""
    if settings.balance == "auto" and settings.auto_balance_wb is None and camera_auto_wb(raw) is None:
        settings = settings.copy(auto_balance_wb=tuple(float(v) for v in auto_wb(raw.camera_rgb)))
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


def finish(img, settings, caps=None, scale=1.0, lens_optimization_baseline=None):
    """Lens optimization, noise reduction and sharpening on the rendered image.

    scale is the render resolution relative to the full image so that radii
    stay proportional in reduced-size previews.
    """
    caps = caps or camera.capabilities("")
    if (settings.lens_optimization and caps.lens_optimization
            and settings.lens_optimization != lens_optimization_baseline):
        img = lens_optimize(img, settings.lens_optimization)
    if settings.noise_type != "none":
        img = reduce_noise(img, settings, scale, caps)
    if settings.sharpen_amount > 0:
        img = sharpen(img, settings, scale)
    return img


LENS_GRID = 64                       # lens drift is measured on a LENS_GRID x LENS_GRID Lab thumbnail
_LENS_RINGS, _LENS_SECTORS = 6, 8
_ly, _lx = np.mgrid[0:LENS_GRID, 0:LENS_GRID].astype(np.float32)
_LENS_C = (LENS_GRID - 1) / 2
_LENS_R2 = ((_lx - _LENS_C) ** 2 + (_ly - _LENS_C) ** 2) / (2 * _LENS_C ** 2)   # 0 centre .. 1 corner
_LENS_RING = np.minimum((_LENS_R2 * _LENS_RINGS).astype(int), _LENS_RINGS - 1)
_LENS_SECTOR = ((np.arctan2(_ly - _LENS_C, _lx - _LENS_C) + np.pi) / (2 * np.pi) * _LENS_SECTORS).astype(int) \
    % _LENS_SECTORS
del _ly, _lx


def _lens_lab(img):
    return cv2.cvtColor(np.ascontiguousarray(img, np.float32), cv2.COLOR_RGB2Lab)


def _lens_drift(small):
    """a*/b* drift (LENS_GRID x LENS_GRID x 2) of a Lab thumbnail: the chroma change fitted as a
    quadratic function of the radius, zero at the centre."""
    out = np.empty(small.shape[:2] + (2,), np.float32)
    for i, c in enumerate((1, 2)):
        ch = small[..., c]
        center = np.median(ch[_LENS_R2 < 0.1])
        a = np.polyfit(_LENS_R2.ravel(), (ch - center).ravel(), 2)
        out[..., i] = np.polyval(a, _LENS_R2) - np.polyval(a, 0)
    return out


def lens_optimize(img, strength):
    """Remove the radial hue shift some lenses introduce (corner colour cast)."""
    h, w = img.shape[:2]
    lab = _lens_lab(img)
    drift = _lens_drift(cv2.resize(lab, (LENS_GRID, LENS_GRID), interpolation=cv2.INTER_AREA))
    lab[..., 1:] -= np.float32(strength / 99) * cv2.resize(drift, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def lens_auto(img):
    """Lens Optimization strength (0..99) that best removes the corner colour cast of img, the image
    rendered without lens optimization; None when it has too few near-neutral areas to measure.

    lens_optimize removes the drift fitted over every pixel, which coloured subjects also bend. The
    cast is therefore measured on near-neutral areas only, as the median a*/b* of each ring (robust
    median over angular sectors) relative to the centre, and the strength leaving the least residual
    is chosen.
    """
    small = cv2.resize(_lens_lab(img), (LENS_GRID, LENS_GRID), interpolation=cv2.INTER_AREA)
    drift = _lens_drift(small)
    neutral = (np.hypot(small[..., 1], small[..., 2]) < 20) & (small[..., 0] > 10) & (small[..., 0] < 97)
    rings = []                       # per ring: the sector masks holding enough neutral pixels
    for i in range(_LENS_RINGS):
        masks = (neutral & (_LENS_RING == i) & (_LENS_SECTOR == j) for j in range(_LENS_SECTORS))
        rings.append([m for m in masks if m.sum() >= 4])
    if len(rings[0]) < 3 or sum(len(r) >= 3 for r in rings[1:]) < 2:
        return None
    ab = small[..., 1:]
    best, best_cost = 0, None
    for k in range(100):
        v = ab - np.float32(k / 99) * drift          # what lens_optimize(img, k) leaves in a*/b*
        level = [np.median([np.median(v[m], axis=0) for m in r], axis=0) if len(r) >= 3 else None
                 for r in rings]
        cost = sum(float(np.sum((x - level[0]) ** 2)) for x in level[1:] if x is not None)
        if best_cost is None or cost < best_cost - 1e-6:
            best, best_cost = k, cost
    return best


def _mix(a, b, w):
    """a * w + b * (1 - w) (OpenCV: multithreaded, no numpy temporaries)."""
    return cv2.addWeighted(a, float(w), b, float(1 - w), 0)


def _clip01(img):
    return np.clip(img, 0, 1, out=img)


# Expert noise reduction: bilateral tone range (YCrCb, 0..1) at strength 100.
EXPERT_LUMA_SIGMA, EXPERT_CHROMA_SIGMA = 0.12, 0.15


def reduce_noise(img, settings, scale=1.0, caps=None):
    """Noise reduction. Moire reduction strength is the Threshold for older DCS images and the
    Strength for Pro Back / Pro SLR images, whose dialog has no threshold."""
    caps = caps or camera.capabilities("")
    ycc = cv2.cvtColor(np.asarray(img, np.float32), cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycc)
    s = settings.noise_strength / 100
    radius = max(1, round(settings.noise_radius * scale))
    t = settings.noise_type
    if t in ("basic", "advanced", "moire"):
        k = 2 * radius + 1
        cr = _mix(cv2.GaussianBlur(cr, (k, k), 0), cr, s)
        cb = _mix(cv2.GaussianBlur(cb, (k, k), 0), cb, s)
        if t != "basic":
            y = cv2.bilateralFilter(y, k, 0.01 + 0.06 * s, radius)
    moire = settings.noise_strength if caps.pro_image else settings.moire_threshold
    if t == "moire" and moire > 0:
        k = 2 * max(1, round((1 + moire / 12) * scale)) + 1
        m = moire / 100
        cr = _mix(cv2.medianBlur(cr, min(k, 5)), cr, m)
        cb = _mix(cv2.medianBlur(cb, min(k, 5)), cb, m)
    if t == "expert":
        # Edge-preserving (bilateral) filtering whose reach in tone and the share of the filtered
        # result both grow with the strength: 0 leaves the image unchanged, 100 is the strongest.
        d = max(3, round(5 * scale) | 1)
        ls = settings.noise_luma / 100
        if ls > 0:
            y = _mix(cv2.bilateralFilter(y, d, EXPERT_LUMA_SIGMA * ls, 3 * scale), y, ls)
        if s > 0:
            k = d * 2 + 1
            cr = _mix(cv2.bilateralFilter(cr, k, EXPERT_CHROMA_SIGMA * s, 6 * scale), cr, s)
            cb = _mix(cv2.bilateralFilter(cb, k, EXPERT_CHROMA_SIGMA * s, 6 * scale), cb, s)
    return _clip01(cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB))


def sharpen(img, settings, scale=1.0):
    """Unsharp mask on luminance: Amount 0..20, Radius 0..10 px, Threshold 0..255."""
    radius = max(0.3, settings.sharpen_radius * scale)
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0]
    detail = y - cv2.GaussianBlur(y, (0, 0), radius)
    thr = settings.sharpen_threshold / 255
    detail = np.where(np.abs(detail) >= thr, detail, 0)
    ycc[..., 0] = y + (settings.sharpen_amount / 10) * detail
    return _clip01(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB))


# ------------------------------------------------------------------ Auto noise reduction / sharpening
# Auto samples the image at many places: ANALYSIS_GRID x ANALYSIS_GRID tiles of ANALYSIS_TILE pixels
# at full size, rendered without noise reduction and sharpening. Noise is measured on the flattest
# blocks of each brightness band (Immerkaer's estimator, in 8-bit levels), edge blur on the strongest
# edges of each tile (gradient ratio of the image and a re-blurred copy, as a Gaussian sigma in
# pixels); medians over the samples keep one area from deciding the result.
ANALYSIS_GRID, ANALYSIS_TILE, _BLOCK = 7, 160, 16
# Noise left after Auto noise reduction (8-bit levels of the working space): chroma noise is
# removed down to this level, luminance noise only where it is visible grain.
NOISE_TARGET_CHROMA, NOISE_TARGET_LUMA = 0.3, 0.6
# Auto sharpening narrows the measured edge blur (Gaussian sigma, full-size pixels) to this fraction,
# not below SHARPEN_MIN_SIGMA (edges that sharp get no sharpening), with at most the High amount.
SHARPEN_BLUR_FRACTION, SHARPEN_MIN_SIGMA = 0.85, 0.65
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def analysis_tiles(raw, settings, offset=(0.0, 0.0)):
    """Tiles of the image rendered with settings but without noise reduction and sharpening, one at
    the centre of each cell of an ANALYSIS_GRID x ANALYSIS_GRID grid (shifted by offset cells)."""
    s = settings.copy(noise_type="none", sharpen_amount=0.0, crop=None)
    h, w = raw.camera_rgb.shape[:2]
    tw, th = min(1.0, ANALYSIS_TILE * raw.scale / w), min(1.0, ANALYSIS_TILE * raw.scale / h)
    tiles = []
    for i in range(ANALYSIS_GRID):
        for j in range(ANALYSIS_GRID):
            cx = min(max((j + 0.5 + offset[0]) / ANALYSIS_GRID, tw / 2), 1 - tw / 2)
            cy = min(max((i + 0.5 + offset[1]) / ANALYSIS_GRID, th / 2), 1 - th / 2)
            tiles.append(render_region(raw, s, (cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2), margin=0))
    return tiles


def _block_noise(ch):
    """Noise sigma (8-bit levels) of each _BLOCK x _BLOCK block of a channel."""
    lap = np.abs(cv2.filter2D(ch, -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT))
    hb, wb = ch.shape[0] // _BLOCK, ch.shape[1] // _BLOCK
    lap = lap[:hb * _BLOCK, :wb * _BLOCK].reshape(hb, _BLOCK, wb, _BLOCK).mean(axis=(1, 3))
    return np.sqrt(np.pi / 2) / 6 * 255 * lap


def _ycc(img):
    return cv2.cvtColor(np.ascontiguousarray(img, np.float32), cv2.COLOR_RGB2YCrCb)


# Brightness bands (luma) noise is measured in separately: noise depends on the level, and the
# quietest blocks overall would all be highlights.
_NOISE_BANDS = ((0.06, 0.3), (0.3, 0.65), (0.65, 0.94))
_BAND_MIN_BLOCKS = 80                # a band counts only with this many blocks over all tiles


def _block_luma(y, shape):
    return y[:shape[0] * _BLOCK, :shape[1] * _BLOCK].reshape(shape[0], _BLOCK, shape[1], _BLOCK).mean(axis=(1, 3))


def _flat_blocks(tiles):
    """Per tile, an int map of its flat blocks: the band index + 1 for blocks whose luminance noise
    is in the lowest 30 % of their brightness band over all tiles (flat areas, where texture does
    not count as noise), 0 elsewhere. None if no band has enough blocks."""
    ys = [_ycc(t)[..., 0] for t in tiles]
    sig = [_block_noise(y) for y in ys]
    mean = [_block_luma(y, s.shape) for y, s in zip(ys, sig)]
    out = [np.zeros(s.shape, np.int8) for s in sig]
    found = False
    for b, (lo, hi) in enumerate(_NOISE_BANDS):
        band = [(m >= lo) & (m < hi) for m in mean]
        pool = np.concatenate([s[k] for s, k in zip(sig, band)])
        if pool.size < _BAND_MIN_BLOCKS:
            continue
        limit = np.percentile(pool, 30)
        for o, s, k in zip(out, sig, band):
            o[k & (s <= limit)] = b + 1
        found = True
    return out if found else None


def image_noise(tiles, flat=None):
    """(luma, chroma) noise (8-bit levels) of the flat areas of tiles: the median per brightness
    band, averaged over the bands present. None if there are no flat areas."""
    flat = flat if flat is not None else _flat_blocks(tiles)
    if flat is None:
        return None
    acc = {}
    for t, m in zip(tiles, flat):
        ycc = _ycc(t)
        sig = [_block_noise(ycc[..., c]) for c in range(3)]
        for b in np.unique(m[m > 0]):
            acc.setdefault(b, [[], [], []])
            for c in range(3):
                acc[b][c].append(sig[c][m == b])
    if not acc:
        return None
    bands = [[float(np.median(np.concatenate(a))) for a in v] for v in acc.values()]
    y = float(np.mean([b[0] for b in bands]))
    c = float(np.mean([np.hypot(b[1], b[2]) / np.sqrt(2) for b in bands]))
    return y, c


def _noise_values(noise_type, kind, radius, level):
    if kind == "strength":
        return {"noise_type": noise_type, "noise_strength": level, "noise_luma": level}
    if kind == "threshold":
        return {"noise_type": noise_type, "moire_threshold": level}
    return {"noise_type": noise_type, "noise_radius": radius, "noise_strength": level}


def noise_auto(tiles, settings, caps, noise_type):
    """Noise Reduction values for noise_type measured on analysis_tiles(): the smallest radius and
    then the lowest strength that bring the flat areas' noise down to the targets (the gentlest
    setting, least loss of detail); the strongest setting if the targets cannot be reached.
    None if the image has no flat areas."""
    kind = caps.noise_controls(noise_type)
    if kind == "none":
        return {"noise_type": noise_type}
    flat = _flat_blocks(tiles)
    if flat is None:
        return None
    flat_i = [i for i, m in enumerate(flat) if m.any()]
    luma_matters = noise_type in ("advanced", "moire", "expert") and kind != "threshold"

    def residual(v):
        s = settings.copy(**v)
        out = [reduce_noise(tiles[i], s, 1.0, caps) for i in flat_i]
        y, c = image_noise(out, [flat[i] for i in flat_i])
        return max(c / NOISE_TARGET_CHROMA, y / NOISE_TARGET_LUMA if luma_matters else 0)

    if kind == "radius":
        lo, hi = NOISE_RADIUS_RANGE.get(noise_type, (1, 10))
        radii = range(lo, hi + 1, 2 if noise_type == "basic" else 1)
    else:
        radii = (None,)
    levels = list(range(0, 101, 5))
    strongest = None
    for r in radii:
        top = _noise_values(noise_type, kind, r, levels[-1])
        res = residual(top)
        if res > 1:
            if strongest is None or res < strongest[0]:
                strongest = (res, top)
            continue
        # The residual falls as the strength rises: the lowest level reaching the targets.
        a, b = 0, len(levels) - 1
        while a < b:
            m = (a + b) // 2
            if residual(_noise_values(noise_type, kind, r, levels[m])) <= 1:
                b = m
            else:
                a = m + 1
        return _noise_values(noise_type, kind, r, levels[a])
    return strongest[1]


_EDGE_MIN_PIXELS, _EDGE_MIN_TILES = 30, 3


def edge_blur(tiles):
    """Gaussian sigma (pixels) of the strongest edges: the median over the tiles with enough edges
    of each tile's median, so one area (a single sharp or soft subject) does not decide it.
    None if fewer than _EDGE_MIN_TILES tiles have edges."""
    per_tile = []
    for t in tiles:
        y = _ycc(t)[..., 0]
        g1 = np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3))
        yb = cv2.GaussianBlur(y, (0, 0), 1.0)
        g2 = np.hypot(cv2.Sobel(yb, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(yb, cv2.CV_32F, 0, 1, ksize=3))
        edges = cv2.Canny((np.clip(y, 0, 1) * 255).astype(np.uint8), 40, 100) > 0
        m = edges & (g1 > np.percentile(g1, 90)) & (g2 > 1e-4)
        r = g1[m] / g2[m]
        r = r[r > 1.01]
        if r.size >= _EDGE_MIN_PIXELS:
            per_tile.append(np.median(1.0 / np.sqrt(r ** 2 - 1)))
    if len(per_tile) < _EDGE_MIN_TILES:
        return None
    return float(np.median(per_tile))


def sharpen_auto(tiles, settings):
    """Sharpening (amount, radius, threshold) measured on analysis_tiles(): radius from the edge
    blur, threshold above the noise so grain is not sharpened, and the lowest amount that narrows
    the edges to SHARPEN_BLUR_FRACTION of their blur. None if the image has too few edges."""
    blur = edge_blur(tiles)
    if blur is None:
        return None
    noise = image_noise(tiles)
    threshold = int(min(25, max(1, round(2.5 * noise[0])))) if noise else SHARPEN_PRESETS["medium"][2]
    radius = float(min(3.0, max(0.5, round(blur, 1))))
    target = max(SHARPEN_MIN_SIGMA, SHARPEN_BLUR_FRACTION * blur)
    top = SHARPEN_PRESETS["high"][0]
    if blur <= target:
        return 0.0, radius, threshold
    lo, hi = 0.5, top                       # edge blur falls as the amount rises
    s = settings.copy(sharpen_radius=radius, sharpen_threshold=threshold)
    if (edge_blur([sharpen(t, s.copy(sharpen_amount=hi)) for t in tiles]) or 0) > target:
        return top, radius, threshold
    while hi - lo > 0.5:
        m = round((lo + hi) / 2 * 2) / 2
        after = edge_blur([sharpen(t, s.copy(sharpen_amount=m)) for t in tiles])
        if after is None or after <= target:
            hi = m
        else:
            lo = m
    return float(hi), radius, threshold


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
