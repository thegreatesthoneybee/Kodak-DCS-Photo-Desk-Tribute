"""DCS camera models: TIFF Make/Model reading, source-correction profiles and capabilities."""
import dataclasses
import math
import struct

import numpy as np

# Order of the Image > Lighting menu.
ILLUMINANTS = ("Daylight", "Tungsten", "Fluorescent", "Flash")

# (substring of TIFF Model tag, profile file pattern); first match wins.
_FAMILIES = (
    ("ProBack645", "DCSPB645{}Source.icm"),
    ("Pro Back 645", "DCSPB645{}Source.icm"),
    ("ProBack", "DCSPB{}Source.icm"),
    ("Pro Back", "DCSPB{}Source.icm"),
    ("SLR/c", "DCSProSLRc{}.icm"),
    ("SLR/n", "DCSProSLRn{}.icm"),
    ("Pro 14", "DCSPro14{}.icm"),
    ("DCS Pro", "DCSPro14{}.icm"),
    ("720x", "DCS720x{}Source.icm"),
    ("760", "DCS760C{}Source.icm"),
)


# Preferences > "Unlock all camera features": every processing option for every DCS model,
# ignoring the per-model limits of the original Photo Desk.
_unlock_all = False
ALL_NOISE_TYPES = ("none", "basic", "advanced", "moire", "expert")


def set_unlock_all(on):
    global _unlock_all
    _unlock_all = bool(on)


def unlock_all():
    return _unlock_all


@dataclasses.dataclass(frozen=True)
class Capabilities:
    """Processing options Photo Desk offers for a camera model."""
    dcs: bool              # DCS raw file (all processing options); False for other cameras
    pro_back: bool         # DCS Pro Back: +-1 EV, no Auto Scene Balance / Auto Exposure, no .wb files
    pro_slr: bool          # DCS Pro 14n/14nx/SLR: Basic/Advanced/Moire/Expert noise reduction
    lens_optimization: bool
    look_set: tuple = None  # restricted Looks of older cameras (None: every installed Look)
    # Other DCS cameras (help: Reducing Noise): Basic for the 300/500/600/700 series and Pro Backs,
    # Advanced for all but the DCS 315, Advanced with Moire Reduction for the 700 series and Pro Backs.
    other_noise_types: tuple = ("none", "basic", "advanced", "moire")
    original_pro_back: bool = False   # the first DCS Pro Back: no Custom or Auto balance
    unlocked: bool = False            # per-model limits lifted (pro_back/pro_slr still name the family)

    @property
    def exposure_range(self):
        return 1.0 if self.pro_back and not self.unlocked else 2.0

    @property
    def auto_scene(self):
        """Auto Scene Balance / Auto Exposure (not for Pro Back images)."""
        return self.unlocked or not self.pro_back

    @property
    def balance_file(self):
        """Save White Balance File (Pro Backs do not accept .wb files)."""
        return self.unlocked or not self.pro_back

    @property
    def noise_types(self):
        if self.unlocked:
            return ALL_NOISE_TYPES
        if self.pro_slr:
            return ("none", "basic", "advanced", "moire", "expert")
        return self.other_noise_types

    @property
    def pro_image(self):
        """Pro Back / Pro SLR file, which Photo Desk processes through the camera DLL (ProImage):
        Radius/Strength noise reduction with presets, instead of the older DCS Moire threshold."""
        return self.pro_back or self.pro_slr

    def noise_controls(self, noise_type):
        """Settings the Noise Reduction dialog offers for a type: "none", "radius" (Radius and
        Strength), "strength" (Expert) or "threshold" (older DCS Advanced with Moire Reduction)."""
        if noise_type == "none":
            return "none"
        if noise_type == "expert":
            return "strength"
        if not self.pro_image:
            if noise_type == "moire":
                return "threshold"
            if not self.unlocked:
                return "none"
        return "radius"

    def looks(self, available):
        """Looks this camera supports, from [(file name, description)] (help: Setting the Image Look)."""
        allowed = self.look_set
        if allowed is None:
            return list(available)
        return [(n, d) for n, d in available if d.replace("Kodak DCS ", "").replace(" Look", "") in allowed]

    @property
    def white_point(self):
        """White Point (Kelvin) control: DCS Pro 14n, Pro SLR/n and Pro SLR/c images only."""
        return self.pro_slr or self.unlocked

    @property
    def balances(self):
        # "The Custom and Auto balance methods are available on all ProBack models except the original
        # ProBack" (help: Specifying the White Balance Method).
        return ("preset", "click") if self.original_pro_back else ("preset", "click", "custom", "auto")


def capabilities(model):
    m = (model or "").lower()
    pro_back = "proback" in m or "pro back" in m
    pro_slr = any(k in m for k in ("slr/", "pro 14", "pro14"))
    dcs = pro_back or pro_slr or "dcs" in m
    lens = any(k in m for k in ("14nx", "slr/n", "slr/c"))
    look_set = None
    basic = ("Portrait", "Portrait Grayscale", "Product", "Product Grayscale")
    if "620x" in m:
        look_set = basic + ("Portrait Reduced", "Product Reduced")
    elif any(k in m for k in ("315", "330", "520", "560", "620", "660")):
        look_set = basic
    noise = ["none", "basic", "advanced", "moire"]
    if "315" in m:
        noise.remove("advanced")
    if not (pro_back or any(k in m for k in ("720", "760"))):
        noise.remove("moire")
    original = pro_back and not any(k in m for k in ("645", "plus", " 16", "cf", "+"))
    if _unlock_all and dcs:
        return Capabilities(dcs, pro_back, pro_slr, True, None, ALL_NOISE_TYPES[:4], False, True)
    return Capabilities(dcs, pro_back, pro_slr, lens, look_set, tuple(noise), original)


def read_make_model(path):
    """Return (make, model) from the first TIFF IFD of a DCR file."""
    vals = _ifd0_strings(path, (0x10F, 0x110))
    return vals.get(0x10F, ""), vals.get(0x110, "")


def read_serial(path):
    """Camera serial number (TIFF tag 0x827D, Copyright, which DCS cameras use for it)."""
    return _ifd0_strings(path, (0x827D,)).get(0x827D, "")


def _tiff_header(f):
    """Return byte order and first-IFD offset for a classic TIFF, or None otherwise."""
    f.seek(0)
    head = f.read(8)
    if len(head) != 8 or head[:4] not in (b"MM\x00*", b"II*\x00"):
        return None
    byteorder = ">" if head[:2] == b"MM" else "<"
    return byteorder, struct.unpack(byteorder + "I", head[4:])[0]


def _tiff_entries(f, byteorder, offset):
    """Read complete 12-byte entries from one bounded classic-TIFF IFD."""
    if offset < 8:
        return []
    f.seek(offset)
    raw_count = f.read(2)
    if len(raw_count) != 2:
        return []
    count, = struct.unpack(byteorder + "H", raw_count)
    raw_entries = f.read(12 * count)
    return [struct.unpack_from(byteorder + "HHI4s", raw_entries, 12 * i)
            for i in range(len(raw_entries) // 12)]


def _ifd0_tags(path, tags):
    """Read requested ASCII or scalar integer tags from the first TIFF IFD only."""
    ascii_limit = 1 << 16
    integer_types = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i"}
    try:
        with open(path, "rb") as f:
            header = _tiff_header(f)
            if header is None:
                return {}
            byteorder, ifd0 = header
            values = {}
            for tag, typ, count, field in _tiff_entries(f, byteorder, ifd0):
                if tag not in tags:
                    continue
                if typ == 2 and 0 < count <= ascii_limit:
                    if count <= 4:
                        raw = field[:count]
                    else:
                        f.seek(struct.unpack(byteorder + "I", field)[0])
                        raw = f.read(count)
                    if len(raw) == count:
                        values[tag] = (typ, raw)
                    continue
                code = integer_types.get(typ)
                if code is None or count != 1:
                    continue
                size = struct.calcsize(code)
                if size <= 4:
                    raw = field[:size]
                else:
                    f.seek(struct.unpack(byteorder + "I", field)[0])
                    raw = f.read(size)
                if len(raw) == size:
                    values[tag] = (typ, struct.unpack(byteorder + code, raw)[0])
            return values
    except FileNotFoundError:
        return {}


def _ascii_value(field):
    if field is None or field[0] != 2:
        return None
    value = field[1]
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None
    return value.split("\x00", 1)[0]


def _ifd0_strings(path, tags):
    vals = {}
    for tag, field in _ifd0_tags(path, tags).items():
        if field[0] == 2:
            vals[tag] = field[1].split(b"\x00", 1)[0].decode("latin1").strip()
    return vals


_TIFF_INTS = {1: "B", 3: "H", 4: "I", 7: "B", 8: "h", 9: "i", 5: "II", 10: "ii"}   # 5/10: rationals
KODAK_IFD = 0x8290
# Kodak IFD tags 2654-2656: [56, 0, strength, 0] per channel. DCSPro3SLR/DCSPro4SLR.dll
# reads these records and accepts strengths of 0..99. For DCSPro4SLR the third LONG in tag 2654
# is the baseline compared with the requested Lens Optimization strength.
LENS_BASELINE_TAG = 2654
# Kodak IFD tags 2120-2125: camera RGB of a neutral under each white balance setting of the camera
# (Daylight, Tungsten, Fluorescent, Flash, Custom, Auto); tag 1020 is the setting used for the shot.
# Photo Desk takes the Preset balance from the lighting's point and Custom from tag 2124 (0x84C).
BALANCE_POINT_TAGS = {"Daylight": 2120, "Tungsten": 2121, "Fluorescent": 2122, "Flash": 2123,
                      "custom": 2124, "camera_auto": 2125}
WB_SETTING_TAG = 1020
WB_SETTINGS = ("Daylight", "Tungsten", "Fluorescent", "Flash", "custom", "camera_auto")


def read_kodak_tags(path, tags, expected_types=None, max_count=64):
    """Integer values of the given Kodak IFD (tag 0x8290) tags as {tag: tuple}; reads only the
    IFD entries and those values, not the whole file. ``expected_types`` optionally maps tag IDs
    to required TIFF type codes; tags with more than ``max_count`` values are skipped. ASCII tags
    are returned as tuples of byte values only when their type 2 is explicitly requested in
    ``expected_types``."""
    out = {}
    with open(path, "rb") as f:
        header = _tiff_header(f)
        if header is None:
            return out
        bo, ifd0 = header

        kodak = next((v for t, typ, _, v in _tiff_entries(f, bo, ifd0)
                      if t == KODAK_IFD and typ in (4, 13)), None)
        if kodak is None:
            return out
        for tag, typ, cnt, val in _tiff_entries(f, bo, struct.unpack(bo + "I", kodak)[0]):
            if (tag in tags and typ == 2 and expected_types is not None
                    and expected_types.get(tag) == 2):
                if cnt == 0 or cnt > max_count:
                    continue
                if cnt <= 4:
                    raw = val[:cnt]
                else:
                    f.seek(struct.unpack(bo + "I", val)[0])
                    raw = f.read(cnt)
                if len(raw) == cnt:
                    out[tag] = tuple(raw)
                continue
            code = _TIFF_INTS.get(typ)
            if (tag not in tags or code is None
                    or (expected_types is not None and typ != expected_types.get(tag, typ))
                    or (tag in (1021, 5011) and typ != 7)):
                continue
            if tag in (1021, 5011):
                count = min(cnt, 72 if tag == 1021 else 112)
                if cnt <= 4:
                    raw = val[:count]
                else:
                    f.seek(struct.unpack(bo + "I", val)[0])
                    raw = f.read(count)
                if len(raw) == count:
                    out[tag] = tuple(raw)
                continue
            if cnt > max_count:
                continue
            size = struct.calcsize(code) * cnt
            if size <= 4:
                raw = val[:size]
            else:
                f.seek(struct.unpack(bo + "I", val)[0])
                raw = f.read(size)
            if len(raw) == size:
                v = struct.unpack(bo + code * cnt, raw)
                if len(code) == 2:
                    if any(d == 0 for d in v[1::2]):
                        continue
                    v = tuple(n / d for n, d in zip(v[::2], v[1::2]))
                out[tag] = v
    return out


def read_capture_metadata(path):
    """Return camera-recorded capture metadata fields present in a DCS TIFF/RAW file."""
    standard = _ifd0_tags(path, (0x827D, 0x131, 0x9211, 0x9003, 0x132))
    metadata = {}
    for tag, key in ((0x827D, "serial"), (0x131, "firmware")):
        value = _ascii_value(standard.get(tag))
        if value is not None:
            metadata[key] = value

    image_number = standard.get(0x9211)
    if image_number is not None and image_number[0] in (1, 3, 4, 6, 8, 9, 16, 17):
        value = image_number[1]
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            metadata["image_number"] = int(value)
        elif (isinstance(value, (tuple, list)) and len(value) == 1
              and isinstance(value[0], (int, np.integer)) and not isinstance(value[0], (bool, np.bool_))):
            metadata["image_number"] = int(value[0])

    for tag in (0x9003, 0x132):
        value = _ascii_value(standard.get(tag))
        if value is not None:
            metadata["capture_timestamp"] = value
            break

    # Kodak IFD tag 1009 is the native CameraSettingString. Camera screenshots verify these are
    # the exact displayed setting rows, so preserve the row text rather than mapping or normalizing it.
    try:
        settings = read_kodak_tags(path, (1009,), expected_types={1009: 2}, max_count=65536).get(1009)
    except FileNotFoundError:
        return metadata
    if settings is not None:
        try:
            text = bytes(settings).decode("ascii").split("\x00", 1)[0]
        except UnicodeDecodeError:
            pass
        else:
            metadata["camera_settings"] = tuple(text.splitlines())
    return metadata


def stored_render_parameters(path):
    """Read the Kodak processing record's exposure and selected RGB point."""
    data = read_kodak_tags(path, (1021,)).get(1021, ())
    if not data:
        return {}
    with open(path, "rb") as f:
        order = ">" if f.read(2) == b"MM" else "<"
    # The original reads at most 72 bytes into a zero-filled buffer, then checks the
    # record's declared size. It does not reject an unknown version or checksum.
    data = bytes(data[:72]).ljust(72, b"\0")
    size = struct.unpack_from(order + "I", data, 8)[0]
    if size > 72:
        return {}
    exposure = struct.unpack_from(order + "i", data, 32)[0] / 1_000_000
    illumination = struct.unpack_from(order + "I", data, 36)[0]
    point = struct.unpack_from(order + "3H", data, 40)
    result = {"exposure": exposure}
    if illumination < len(ILLUMINANTS):
        result["lighting"] = ILLUMINANTS[illumination]
    if all(1 <= value <= 10000 for value in point):
        result["balance_point"] = point
    return result


def stored_process_parameters(path):
    """Read validated fields from Kodak's version 2 or later processing record."""
    data = read_kodak_tags(path, (5011,)).get(5011, ())
    if not data:
        return {}
    with open(path, "rb") as f:
        order = ">" if f.read(2) == b"MM" else "<"
    data = bytes(data[:112]).ljust(112, b"\0")
    version = struct.unpack_from(order + "I", data, 4)[0]
    if version < 2:
        return {}
    result = {}
    temperature, mode, illuminant = struct.unpack_from(order + "III", data, 16)
    if mode < 7 and illuminant < len(ILLUMINANTS) and 100 <= temperature <= 11000:
        result["white_point"] = temperature
        result["kelvin_mode"] = mode
        result["kelvin_illumination"] = ILLUMINANTS[mode if mode < 4 else illuminant]
    color_balance = struct.unpack_from(order + "3h", data, 60)
    result["color_balance"] = color_balance if all(-100 <= value <= 100 for value in color_balance) else (0, 0, 0)
    result["tone_enabled"] = bool(struct.unpack_from(order + "I", data, 92)[0])
    levels = []
    for offset in (68, 76, 84):
        numerator, denominator = struct.unpack_from(order + "ii", data, offset)
        if denominator == 0:
            return result
        levels.append(numerator / denominator)
    result["tone"] = tuple(levels)
    return result


def stored_process_levels(path):
    return stored_process_parameters(path).get("tone")


def balance_curves(path):
    """Camera-recorded Kelvin polynomials for each illuminant (Kodak tags 2140–2143)."""
    tags = read_kodak_tags(path, (2140, 2141, 2142, 2143))
    return {name: tags[2140 + i] for i, name in enumerate(ILLUMINANTS)
            if len(tags.get(2140 + i, ())) == 27}


def balance_factors(path):
    """Camera-recorded RGB correction factors for the four preset illuminants."""
    tags = read_kodak_tags(path, (2130, 2131, 2132, 2133))
    factors = {}
    for i, name in enumerate(ILLUMINANTS):
        values = tags.get(2130 + i, ())
        factors[name] = values if len(values) == 3 and all(math.isfinite(v) and v >= 0.1 for v in values) else (1.0, 1.0, 1.0)
    return factors


def color_matrices(path):
    """Camera-recorded 3x3 colour matrices per illuminant (Kodak tags 2010-2013), which the Pro SLR
    renderer applies to balanced camera RGB (ipDeviantMat, tags loaded into scene +0x560..+0x56c)."""
    tags = read_kodak_tags(path, (2010, 2011, 2012, 2013))
    matrices = {}
    for i, name in enumerate(ILLUMINANTS):
        values = tags.get(2010 + i, ())
        if len(values) == 9 and all(math.isfinite(v) for v in values):
            matrices[name] = values
    return matrices


def unique_matrices(path):
    """Per-illuminant 3x3 matrices (Kodak tags 2020-2023) that the Pro SLR RGB renderer applies to
    toned camera RGB before its ERIMM table (ipUniqueMat, scene +0x57c)."""
    tags = read_kodak_tags(path, (2020, 2021, 2022, 2023))
    matrices = {}
    for i, name in enumerate(ILLUMINANTS):
        values = tags.get(2020 + i, ())
        if len(values) == 9 and all(math.isfinite(v) for v in values):
            matrices[name] = values
    return matrices


def auto_capture_settings(path):
    """Return a usable camera Auto illuminant and its recorded scene exposure."""
    tags = read_kodak_tags(path, (1019, 2011, 2012, 2121, 2122, 2125, 3110, 3122, 3125))
    illumination = tags.get(1019, ())
    if len(illumination) != 1 or illumination[0] >= len(ILLUMINANTS) or illumination[0] < 0:
        return None, None
    if not tags.get(3125) or not tags[3125][0]:
        return None, None
    for tag in (2011, 2012):
        matrix = tags.get(tag, ())
        if len(matrix) != 9 or not all(math.isfinite(value) for value in matrix) or not any(matrix):
            return None, None
    for tag in (2121, 2122):
        point = tags.get(tag, ())
        if len(point) != 3 or not all(value >= 2 for value in point):
            return None, None
    point = tags.get(2125, ())
    if len(point) != 3 or not all(10 <= value <= 10000 for value in point) or all(value == 4095 for value in point):
        return None, None
    source = (tags.get(3110) or (0,))[0] or 2571
    reference = (tags.get(3122) or (0,))[0]
    difference = (source - reference) * 0.0033222591362126247
    steps = math.trunc((difference + (0.05 if difference > 0 else -0.05)) * 10)
    exposure = struct.unpack("<f", struct.pack("<f", steps * 0.10000000149011612))[0]
    return ILLUMINANTS[illumination[0]], max(-2.0, min(2.0, exposure))


def auto_kelvin_illumination(path):
    return auto_capture_settings(path)[0]


def custom_kelvin_illumination(path):
    """Return the recorded custom-balance illuminant when its point is usable."""
    tags = read_kodak_tags(path, (1029, 2124, 3125))
    illumination, point = tags.get(1029, ()), tags.get(2124, ())
    if len(illumination) != 1 or not 0 <= illumination[0] < len(ILLUMINANTS):
        return None
    if not tags.get(3125) or not tags[3125][0] or len(point) != 3:
        return None
    if not all(10 <= value <= 10000 for value in point) or all(value == 4095 for value in point):
        return None
    return ILLUMINANTS[illumination[0]]


def balance_points(path):
    """{"Daylight"|...|"custom"|"camera_auto": (R, G, B)} neutral points the camera recorded."""
    tags = read_kodak_tags(path, tuple(BALANCE_POINT_TAGS.values()))
    return {k: tuple(float(x) for x in tags[t]) for k, t in BALANCE_POINT_TAGS.items()
            if len(tags.get(t, ())) == 3 and all(x > 0 for x in tags[t])}


def capture_info(path, model):
    """Processing settings the camera recorded with the image, for At Capture / Auto buttons."""
    info = {"model": model}
    v = read_kodak_tags(path, (WB_SETTING_TAG,)).get(WB_SETTING_TAG)
    if v and 0 <= v[0] < len(WB_SETTINGS):
        info["white_balance"] = WB_SETTINGS[v[0]]
    if capabilities(model).lens_optimization:
        v = read_kodak_tags(path, (LENS_BASELINE_TAG,)).get(LENS_BASELINE_TAG)
        if v and len(v) >= 3 and 0 <= v[2] <= 99:
            info["lens_optimization"] = v[2]
    return info


# DCSPro4SLR.dll lens optimization (ipApplyOmenCFA) record, loaded by 0x65ff7c60: tag 0xa64 selects
# the gain map tag 0xa46 + index (tag 0xa5b x 0xa5c words, unity tag 0xa5a); tag 0xa67 holds the
# rational strength scale (lo, hi); tags 0xa60 and 0xa5e carry the default and the in-camera
# strengths as the third LONG of [56, 0, strength, 0].
OMEN_TAGS = (0xa5a, 0xa5b, 0xa5c, 0xa5e, 0xa5f, 0xa60, 0xa61, 0xa64, 0xa67, 0xa73)
# Kodak IFD tag 0x90d: SHORT table from the stored CFA codes to linear sensor codes. 0x65ff9b90
# builds ipApplyBalanceGainsCFA's tables from it, so stages before the balance (ipApplyOmenCFA)
# work on the stored codes.
LINEARIZATION_TAG = 0x90d


def lens_omen(path, model):
    """The validated Pro SLR/n or /c lens optimization record as a dict, or None.

    Keys: ``grid`` (rows x cols int array), ``unit`` (tag 0xa5a), ``scale`` ((lo, hi) of tag 0xa67),
    ``baseline`` (tag 0xa5e strength) and ``linearization`` (tag 0x90d as an int array, or None when
    the stored codes are already linear). The checks are those 0x65ff7c60 makes before it enables
    the correction (scene +0xae8).
    """
    model_name = (model or "").lower()
    if not capabilities(model).pro_slr or not any(name in model_name for name in ("slr/n", "slr/c")):
        return None
    tags = read_kodak_tags(path, OMEN_TAGS)
    index = tags.get(0xa64, (0,))[0]
    grid_tag = 0xa46 + index
    grid = read_kodak_tags(path, (grid_tag,), expected_types={grid_tag: 3}, max_count=1 << 16).get(grid_tag)
    rows, cols = tags.get(0xa5b, (0,))[0], tags.get(0xa5c, (0,))[0]
    unit = tags.get(0xa5a, (0,))[0]
    scale = tags.get(0xa67, ())
    records = [tags.get(tag, ()) for tag in (0xa60, 0xa5f, LENS_BASELINE_TAG)]
    step = tags.get(0xa73, ())
    if (grid is None or (rows, cols) != (48, 71) or len(grid) != rows * cols or unit not in (0x1000, 0x400)
            or len(scale) != 2 or scale[1] == 0 or 0xa61 not in tags
            or any(len(r) != 4 or r[2] > 99 for r in records)
            or not 1 <= len(step) <= 14 or not 0 < step[0] < 1 << 15):
        return None
    curve = read_kodak_tags(path, (LINEARIZATION_TAG,), expected_types={LINEARIZATION_TAG: 3},
                            max_count=4096).get(LINEARIZATION_TAG)
    curve = np.asarray(curve, np.int64) if curve is not None else None
    if curve is not None and (len(curve) < 2 or np.any(np.diff(curve) < 0)):
        curve = None
    return {"grid": np.asarray(grid, np.int64).reshape(rows, cols), "unit": unit,
            "scale": (float(scale[0]), float(scale[1])), "baseline": records[2][2], "linearization": curve}


def read_gps(path):
    """GPS position recorded by the camera (TIFF GPS IFD, tag 0x8825) as a dict with ``lat``,
    ``lon`` (signed degrees) and optional ``alt`` (metres), or None when no fix was recorded."""
    with open(path, "rb") as f:
        data = f.read(1 << 20)
    if data[:4] not in (b"MM\x00*", b"II*\x00"):
        return None
    bo = ">" if data[:2] == b"MM" else "<"

    def ifd(off):
        if off + 2 > len(data):
            return {}
        n, = struct.unpack_from(bo + "H", data, off)
        out = {}
        for i in range(n):
            p = off + 2 + 12 * i
            if p + 12 > len(data):
                break
            tag, typ, cnt = struct.unpack_from(bo + "HHI", data, p)
            size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}.get(typ, 0) * cnt
            if size <= 4:
                raw = data[p + 8:p + 8 + size]
            else:
                ptr, = struct.unpack_from(bo + "I", data, p + 8)
                raw = data[ptr:ptr + size]
            out[tag] = (typ, cnt, raw)
        return out

    ifd0 = ifd(struct.unpack_from(bo + "I", data, 4)[0])
    if 0x8825 not in ifd0:
        return None
    g = ifd(struct.unpack(bo + "I", ifd0[0x8825][2][:4])[0])

    def rationals(tag):
        if tag not in g or g[tag][0] != 5:
            return None
        v = struct.unpack(bo + "%dI" % (2 * g[tag][1]), g[tag][2])
        return [a / b if b else 0.0 for a, b in zip(v[::2], v[1::2])]

    def ref(tag):
        return g[tag][2][:1].decode("latin1") if tag in g else ""

    lat, lon = rationals(0x2), rationals(0x4)
    if not lat or not lon or not ref(0x1).strip("\0") or not ref(0x3).strip("\0"):
        return None
    deg = lambda dms: dms[0] + (dms[1] if len(dms) > 1 else 0) / 60 + (dms[2] if len(dms) > 2 else 0) / 3600
    out = {"lat": deg(lat) * (-1 if ref(0x1) == "S" else 1), "lon": deg(lon) * (-1 if ref(0x3) == "W" else 1)}
    alt = rationals(0x6)
    if alt:
        below = 0x5 in g and g[0x5][2][:1] == b"\x01"
        out["alt"] = -alt[0] if below else alt[0]
    return out


def gps_text(gps):
    """'Lat 37d 33' 59.4" N  Lon 126d 58' 40.7" E  Alt 38 m' (ASCII, burned into TIFF output)"""
    def dms(v, pos, neg):
        a = abs(v)
        d = int(a)
        m = int((a - d) * 60)
        s = (a - d - m / 60) * 3600
        return f"{d}d {m}' {s:.1f}\" {pos if v >= 0 else neg}"
    text = f"Lat {dms(gps['lat'], 'N', 'S')}  Lon {dms(gps['lon'], 'E', 'W')}"
    if "alt" in gps:
        text += f"  Alt {gps['alt']:.0f} m"
    return text


def source_profile(model, illuminant):
    if illuminant not in ILLUMINANTS:
        raise ValueError(f"unknown illuminant {illuminant}")
    for key, pattern in _FAMILIES:
        if key.lower() in model.lower():
            return pattern.format(illuminant)
    return None
